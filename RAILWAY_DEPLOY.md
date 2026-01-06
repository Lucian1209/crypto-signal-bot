# Railway Deployment Guide

## Підготовка

1. **Створіть акаунт на Railway:**
   - Зайдіть на [railway.app](https://railway.app)
   - Зареєструйтеся через GitHub

2. **Встановіть Railway CLI (опціонально):**
   ```bash
   npm i -g @railway/cli
   railway login
   ```

## Варіант 1: Через Railway Dashboard (найпростіше)

### Крок 1: Створити Signal Service (API)

1. Натисніть "New Project" → "Deploy from GitHub repo"
2. Оберіть ваш репозиторій
3. Налаштування:
   - **Name:** `crypto-signal-api`
   - **Start Command:** `python start.py`

4. **Environment Variables:**
   ```
   SERVICE_TYPE=api
   PORT=8001
   DISABLE_ML=false
   ```

5. Deploy! Railway автоматично дасть вам URL типу: `https://crypto-signal-api.railway.app`

### Крок 2: Створити Telegram Bot Service

1. В тому ж проекті: "New Service" → "GitHub Repo"
2. Оберіть той самий репозиторій
3. Налаштування:
   - **Name:** `crypto-telegram-bot`
   - **Start Command:** `python start.py`

4. **Environment Variables:**
   ```
   SERVICE_TYPE=bot
   TELEGRAM_BOT_TOKEN=your_token_from_botfather
   SIGNAL_SERVICE_URL=https://crypto-signal-api.railway.app
   ```

5. Deploy!

## Варіант 2: Через Railway CLI

```bash
# 1. Ініціалізувати проект
railway init

# 2. Додати env змінні
railway variables set SERVICE_TYPE=api
railway variables set DISABLE_ML=false

# 3. Deploy
railway up

# 4. Отримати URL
railway status
```

## Структура для Railway

Railway буде шукати:
1. `requirements.txt` - залежності Python
2. `Procfile` або `railway.toml` - команди запуску
3. `start.py` - наш універсальний стартовий скрипт

## Environment Variables (важливо!)

### Для API Service:
- `SERVICE_TYPE=api`
- `PORT=8001` (Railway встановить автоматично)
- `DISABLE_ML=false` (або `true` якщо хочете тільки technical analysis)

### Для Bot Service:
- `SERVICE_TYPE=bot`
- `TELEGRAM_BOT_TOKEN=ваш_токен`
- `SIGNAL_SERVICE_URL=https://ваш-api.railway.app`

## Моніторинг

Railway надає:
- Логи в реальному часі
- Метрики (CPU, RAM, Network)
- Automatic deployments при push в GitHub

## Вартість

- **Free tier:** $5 в місяць credit
- **Hobby plan:** $5/місяць (500 годин виконання)
- **Pro plan:** $20/місяць (необмежено)

Для нашого боту:
- API: ~$3-5/місяць
- Bot: ~$2-3/місяць

## Troubleshooting

### Логи показують "Module not found"
- Перевірте `requirements.txt`
- Переконайтесь що всі файли в git

### "Connection refused" від бота до API
- Перевірте `SIGNAL_SERVICE_URL` в bot service
- Переконайтесь що API service запущений

### TensorFlow не завантажується
- Встановіть `DISABLE_ML=true`
- Або використовуйте `tensorflow-cpu`

### Бот не відповідає
- Перевірте `TELEGRAM_BOT_TOKEN`
- Подивіться логи: `railway logs` або в Dashboard

## Додаткові команди

```bash
# Переглянути логи
railway logs

# Відкрити проект
railway open

# Список змінних
railway variables

# Підключитись до shell
railway shell
```

## Альтернативи Railway

Якщо Railway не підходить:
- **Render.com** - схожий на Railway
- **Heroku** - класика (платний після 2022)
- **DigitalOcean App Platform** - $5/місяць
- **Fly.io** - для containerized apps
- **AWS Lambda** - serverless (складніше налаштувати)
