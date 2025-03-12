# Client Installation Scope

No, the client installation doesn't include all code in the project. When you run `pip install -e .` in the `clients/python` directory, only a small portion of the codebase is actually installed.

## What IS Included in Client Installation

When you install the client with `pip install -e .`, only these files are installed as a Python package:

```
clients/python/src/bot_playground/
├── __init__.py
└── client.py
```

This is the core client SDK that provides the `BotPlaygroundClient` class for connecting to the gateway.

## What's NOT Included

The installation does NOT include:

1. **Backend Services**: Nothing in the `services/` directory
   - Gateway service
   - Vector service 
   - Other microservices

2. **Example Bots**: The bot implementations in `clients/python/examples/`
   - enhanced_bot.py
   - vector_bot.py
   - advanced_bot.py
   - transformer_bot.py

3. **Infrastructure**: Docker configurations and scripts
   - docker-compose.yml
   - Dockerfiles
   - start.sh

## How It Works

1. You install the client SDK: `pip install -e .`
2. This makes the `bot_playground` package available to import
3. Example bots import this package: `from bot_playground.client import BotPlaygroundClient`
4. But the example bots themselves remain as standalone scripts

The example bots are meant to be run directly (with `python examples/transformer_bot.py`), not installed as packages. They use the installed client SDK to connect to the gateway service.

Would you like me to explain how to package any of these components differently?