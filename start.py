"""
Универсальний старт для Railway
Запускає API, Bot, або обидва залежно від SERVICE_TYPE env
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

SERVICE_TYPE = os.getenv("SERVICE_TYPE", "api")  # api, bot, or both
PORT = int(os.getenv("PORT", 8001))

logger.info("="*70)
logger.info(f"Starting service: {SERVICE_TYPE}")
logger.info("="*70)

if SERVICE_TYPE == "api":
    # Запустити тільки FastAPI
    logger.info("Starting FastAPI Signal Service...")
    import uvicorn
    uvicorn.run(
        "signal_service.app:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info"
    )

elif SERVICE_TYPE == "bot":
    # Запустити тільки Telegram Bot
    logger.info("Starting Telegram Bot...")
    sys.path.append(str(Path(__file__).parent))
    from telegram_bot.bot import main
    main()

elif SERVICE_TYPE == "both":
    # Запустити обидва (для локального тестування)
    logger.info("Starting both API and Bot...")
    import multiprocessing

    def run_api():
        import uvicorn
        uvicorn.run(
            "signal_service.app:app",
            host="0.0.0.0",
            port=PORT,
            log_level="info"
        )

    def run_bot():
        sys.path.append(str(Path(__file__).parent))
        from telegram_bot.bot import main
        main()

    # Запустити в окремих процесах
    api_process = multiprocessing.Process(target=run_api)
    bot_process = multiprocessing.Process(target=run_bot)

    api_process.start()
    bot_process.start()

    try:
        api_process.join()
        bot_process.join()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        api_process.terminate()
        bot_process.terminate()

else:
    logger.error(f"Unknown SERVICE_TYPE: {SERVICE_TYPE}")
    logger.error("Use: api, bot, or both")
    sys.exit(1)
