"""
Sentiment Data Collector
Collects sentiment from Twitter and News (without Reddit)
"""

import tweepy
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import os


class SentimentCollector:
    """Collect sentiment data from Twitter and News"""

    def __init__(
        self,
        twitter_bearer_token: Optional[str] = None,
        newsapi_key: Optional[str] = None
    ):
        """
        Initialize sentiment collectors

        Args:
            twitter_bearer_token: Twitter API v2 bearer token
            newsapi_key: NewsAPI.org API key
        """
        # Sentiment analyzer
        self.analyzer = SentimentIntensityAnalyzer()

        # Twitter client (v2 API)
        self.twitter_client = None
        if twitter_bearer_token:
            try:
                self.twitter_client = tweepy.Client(bearer_token=twitter_bearer_token)
                print("✅ Twitter client initialized")
            except Exception as e:
                print(f"⚠️  Twitter client error: {e}")
        else:
            print("⚠️  Twitter bearer token not provided")

        # NewsAPI key
        self.newsapi_key = newsapi_key
        if newsapi_key:
            print("✅ NewsAPI key configured")
        else:
            print("⚠️  NewsAPI key not provided")

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text using VADER

        Args:
            text: Text to analyze

        Returns:
            Dictionary with sentiment scores
        """
        scores = self.analyzer.polarity_scores(text)
        return {
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu'],
            'compound': scores['compound']  # Overall sentiment (-1 to 1)
        }

    def collect_twitter_sentiment(
        self,
        query: str,
        max_results: int = 100
    ) -> pd.DataFrame:
        """
        Collect tweets and analyze sentiment

        Args:
            query: Search query (e.g., "Bitcoin OR BTC")
            max_results: Maximum number of tweets (10-100)

        Returns:
            DataFrame with tweets and sentiment
        """
        if not self.twitter_client:
            print("⚠️  Twitter client not configured")
            return pd.DataFrame()

        try:
            print(f"🐦 Collecting tweets for: {query}")

            # Search recent tweets
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=min(max_results, 100),  # API limit
                tweet_fields=['created_at', 'public_metrics', 'lang']
            )

            if not tweets.data:
                print("⚠️  No tweets found")
                return pd.DataFrame()

            data = []
            for tweet in tweets.data:
                if tweet.lang == 'en':  # Only English tweets
                    sentiment = self.analyze_sentiment(tweet.text)
                    data.append({
                        'timestamp': tweet.created_at,
                        'text': tweet.text,
                        'likes': tweet.public_metrics['like_count'],
                        'retweets': tweet.public_metrics['retweet_count'],
                        'sentiment_compound': sentiment['compound'],
                        'sentiment_positive': sentiment['positive'],
                        'sentiment_negative': sentiment['negative']
                    })

            df = pd.DataFrame(data)
            print(f"✅ Collected {len(df)} tweets")

            return df

        except Exception as e:
            print(f"❌ Twitter error: {e}")
            return pd.DataFrame()

    def collect_news_sentiment(
        self,
        query: str,
        days_back: int = 7
    ) -> pd.DataFrame:
        """
        Collect news articles

        Args:
            query: Search query (e.g., "Bitcoin")
            days_back: Number of days to look back

        Returns:
            DataFrame with news and sentiment
        """
        if not self.newsapi_key:
            print("⚠️  NewsAPI key not configured")
            return pd.DataFrame()

        try:
            print(f"📰 Collecting news for: {query}")

            # Calculate date range
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            # NewsAPI endpoint
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.newsapi_key,
                'pageSize': 100
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            articles = response.json().get('articles', [])

            if not articles:
                print("⚠️  No articles found")
                return pd.DataFrame()

            data = []
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                sentiment = self.analyze_sentiment(text)

                data.append({
                    'timestamp': datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', ''),
                    'sentiment_compound': sentiment['compound'],
                    'sentiment_positive': sentiment['positive'],
                    'sentiment_negative': sentiment['negative']
                })

            df = pd.DataFrame(data)
            print(f"✅ Collected {len(df)} articles")

            return df

        except Exception as e:
            print(f"❌ News API error: {e}")
            return pd.DataFrame()

    def aggregate_sentiment(
        self,
        twitter_df: pd.DataFrame,
        news_df: pd.DataFrame,
        timeframe: str = '1H'
    ) -> pd.DataFrame:
        """
        Aggregate sentiment from Twitter and News by timeframe

        Args:
            twitter_df: Twitter sentiment data
            news_df: News sentiment data
            timeframe: Pandas timeframe (1H, 4H, 1D)

        Returns:
            Aggregated sentiment DataFrame
        """
        print(f"📊 Aggregating sentiment by {timeframe}")

        dfs = []

        # Process Twitter
        if not twitter_df.empty:
            twitter_df = twitter_df.set_index('timestamp')
            twitter_agg = twitter_df['sentiment_compound'].resample(timeframe).agg([
                ('mean', 'mean'),
                ('std', 'std'),
                ('count', 'count')
            ])
            twitter_agg.columns = [f'twitter_{col}' for col in twitter_agg.columns]
            dfs.append(twitter_agg)

        # Process News
        if not news_df.empty:
            news_df = news_df.set_index('timestamp')
            news_agg = news_df['sentiment_compound'].resample(timeframe).agg([
                ('mean', 'mean'),
                ('std', 'std'),
                ('count', 'count')
            ])
            news_agg.columns = [f'news_{col}' for col in news_agg.columns]
            dfs.append(news_agg)

        # Combine all sources
        if dfs:
            result = pd.concat(dfs, axis=1)
            result.fillna(0, inplace=True)

            # Calculate overall sentiment (weighted by count)
            twitter_weight = result.get('twitter_count', 0)
            news_weight = result.get('news_count', 0)
            total_weight = twitter_weight + news_weight

            if 'twitter_mean' in result.columns and 'news_mean' in result.columns:
                result['sentiment_overall'] = (
                    result['twitter_mean'] * twitter_weight +
                    result['news_mean'] * news_weight
                ) / (total_weight + 1e-10)  # Avoid division by zero
            elif 'twitter_mean' in result.columns:
                result['sentiment_overall'] = result['twitter_mean']
            elif 'news_mean' in result.columns:
                result['sentiment_overall'] = result['news_mean']
            else:
                result['sentiment_overall'] = 0

            print(f"✅ Aggregated to {len(result)} rows")
            return result

        print("⚠️  No sentiment data to aggregate")
        return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # Load API keys from environment
    collector = SentimentCollector(
        twitter_bearer_token=os.getenv("TWITTER_BEARER_TOKEN"),
        newsapi_key=os.getenv("NEWSAPI_KEY")
    )

    # Example: Analyze Bitcoin sentiment
    query = "Bitcoin"

    print("\n" + "="*60)
    print("Collecting Sentiment Data")
    print("="*60)

    # Collect from each source
    twitter_df = collector.collect_twitter_sentiment(f"{query} OR BTC -is:retweet", max_results=50)
    news_df = collector.collect_news_sentiment(query, days_back=7)

    # Show samples
    if not twitter_df.empty:
        print("\n📊 Twitter Sample:")
        print(twitter_df[['text', 'sentiment_compound']].head(3))

    if not news_df.empty:
        print("\n📊 News Sample:")
        print(news_df[['title', 'sentiment_compound']].head(3))

    # Aggregate
    sentiment_agg = collector.aggregate_sentiment(twitter_df, news_df, timeframe='1H')

    if not sentiment_agg.empty:
        print("\n📊 Aggregated sentiment:")
        print(sentiment_agg.tail())

        # Save
        os.makedirs("data", exist_ok=True)
        sentiment_agg.to_csv("data/sentiment_aggregated.csv")
        print("\n💾 Saved to data/sentiment_aggregated.csv")
    else:
        print("\n⚠️  No sentiment data collected")
        print("\nTo enable sentiment collection:")
        print("1. Get Twitter API Bearer Token: https://developer.twitter.com/")
        print("2. Get NewsAPI Key: https://newsapi.org/")
        print("3. Set environment variables:")
        print("   export TWITTER_BEARER_TOKEN='your_token'")
        print("   export NEWSAPI_KEY='your_key'")
