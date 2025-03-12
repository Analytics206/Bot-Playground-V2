try:
    from bot_playground.client import BotPlaygroundClient
    print('Module imported successfully')
    print(BotPlaygroundClient)
except ImportError as e:
    print(f'Import failed: {e}')