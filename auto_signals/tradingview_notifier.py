"""
TradingView Signal Notifier
Polls backend for TradingView signals and sends notifications
"""

import asyncio
import httpx
from typing import List, Dict
from aiogram import Bot
from aiogram.exceptions import TelegramForbiddenError, TelegramBadRequest
import os


class TradingViewNotifier:
    """Notifier for TradingView signals"""

    def __init__(self, bot_token: str, backend_url: str = "http://localhost:8000"):
        """
        Initialize notifier

        Args:
            bot_token: Telegram bot token
            backend_url: Backend API URL
        """
        self.bot = Bot(token=bot_token)
        self.backend_url = backend_url
        self.last_signal_id = 0

        print("✅ TradingView Notifier initialized")

    def format_tradingview_message(
        self,
        signal: Dict,
        tradingview_data: Dict,
        lang: str = "en"
    ) -> str:
        """
        Format TradingView signal message

        Args:
            signal: Signal data
            tradingview_data: Original TradingView alert data
            lang: User language

        Returns:
            Formatted message
        """
        emoji = "🚀" if signal['action'] == "BUY" else "📉"

        if lang == "uk":
            text = f"{emoji} <b>СИГНАЛ TRADINGVIEW + ML</b> {emoji}\n\n"
            text += f"📊 <b>Джерело:</b> TradingView Alert\n"

            if 'indicator' in tradingview_data:
                text += f"📈 <b>Індикатор:</b> {tradingview_data['indicator']}\n"

            if 'timeframe' in tradingview_data:
                text += f"⏰ <b>Таймфрейм:</b> {tradingview_data['timeframe']}\n"

            text += f"\n💎 <b>Символ:</b> {signal['symbol']}\n"
            text += f"{'🟢' if signal['action'] == 'BUY' else '🔴'} <b>Дія:</b> {signal['action']}\n"
            text += f"💰 <b>Вхід:</b> ${signal['entry_price']:.4f}\n"
            text += f"🛑 <b>Стоп:</b> ${signal['stop_loss']:.4f}\n"
            text += f"🎯 <b>Ціль:</b> ${signal['take_profit']:.4f}\n"
            text += f"🤖 <b>ML Впевненість:</b> {signal['confidence']:.0%}\n\n"

            if 'message' in tradingview_data:
                text += f"💬 <b>Повідомлення:</b>\n{tradingview_data['message']}\n\n"

            text += f"📊 <b>ML Аналіз:</b>\n{signal['analysis']}"

        else:  # English
            text = f"{emoji} <b>TRADINGVIEW + ML SIGNAL</b> {emoji}\n\n"
            text += f"📊 <b>Source:</b> TradingView Alert\n"

            if 'indicator' in tradingview_data:
                text += f"📈 <b>Indicator:</b> {tradingview_data['indicator']}\n"

            if 'timeframe' in tradingview_data:
                text += f"⏰ <b>Timeframe:</b> {tradingview_data['timeframe']}\n"

            text += f"\n💎 <b>Symbol:</b> {signal['symbol']}\n"
            text += f"{'🟢' if signal['action'] == 'BUY' else '🔴'} <b>Action:</b> {signal['action']}\n"
            text += f"💰 <b>Entry:</b> ${signal['entry_price']:.4f}\n"
            text += f"🛑 <b>Stop Loss:</b> ${signal['stop_loss']:.4f}\n"
            text += f"🎯 <b>Take Profit:</b> ${signal['take_profit']:.4f}\n"
            text += f"🤖 <b>ML Confidence:</b> {signal['confidence']:.0%}\n\n"

            if 'message' in tradingview_data:
                text += f"💬 <b>Alert Message:</b>\n{tradingview_data['message']}\n\n"

            text += f"📊 <b>ML Analysis:</b>\n{signal['analysis']}"

        return text

    async def send_notification(self, chat_id: int, message: str) -> bool:
        """Send notification to user"""
        try:
            await self.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode="HTML"
            )
            return True
        except (TelegramForbiddenError, TelegramBadRequest) as e:
            print(f"⚠️  Cannot send to {chat_id}: {e}")
            return False
        except Exception as e:
            print(f"❌ Error sending to {chat_id}: {e}")
            return False

    async def check_and_notify(self):
        """Check for new TradingView signals and notify users"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Get pending TradingView signals
                response = await client.get(
                    f"{self.backend_url}/tradingview/pending",
                    params={"since_id": self.last_signal_id}
                )

                if response.status_code != 200:
                    return

                data = response.json()
                pending_signals = data.get('signals', [])

                for signal_info in pending_signals:
                    signal = signal_info['signal']
                    users = signal_info['users']
                    tradingview_data = signal.get('tradingview_data', {})

                    print(f"\n📤 Sending TradingView signal {signal['symbol']} to {len(users)} users...")

                    sent = 0
                    failed = 0

                    for user in users:
                        chat_id = user['chat_id']
                        lang = user.get('language', 'en')

                        message = self.format_tradingview_message(
                            signal,
                            tradingview_data,
                            lang
                        )

                        if await self.send_notification(chat_id, message):
                            sent += 1
                        else:
                            failed += 1

                        await asyncio.sleep(0.05)  # Rate limit

                    print(f"✅ Sent: {sent}, Failed: {failed}")

                    # Update last processed
                    self.last_signal_id = signal_info.get('id', self.last_signal_id)

        except Exception as e:
            print(f"❌ Check and notify error: {e}")

    async def run(self):
        """Run notification loop"""
        print("🔄 Starting TradingView notification service...")

        while True:
            try:
                await self.check_and_notify()
            except Exception as e:
                print(f"❌ Error in notification loop: {e}")

            # Check every 10 seconds
            await asyncio.sleep(10)

    async def close(self):
        """Close bot session"""
        await self.bot.session.close()


# CLI
if __name__ == "__main__":
    import sys

    bot_token = os.getenv("BOT_TOKEN")
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

    if not bot_token:
        print("❌ BOT_TOKEN environment variable required")
        sys.exit(1)

    notifier = TradingViewNotifier(bot_token, backend_url)

    try:
        asyncio.run(notifier.run())
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping TradingView notifier...")
        asyncio.run(notifier.close())
