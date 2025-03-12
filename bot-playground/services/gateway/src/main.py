import asyncio
import logging
import os
import websockets
from src.gateway_service import GatewayService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gateway-main")

# Get configuration from environment
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/bot_playground")
REDIS_URI = os.environ.get("REDIS_URI", "redis://localhost:6379")

async def main():
    # Initialize Gateway Service
    gateway = GatewayService(
        mongodb_uri=MONGODB_URI,
        redis_uri=REDIS_URI
    )
    
    # Start WebSocket server
    logger.info(f"Starting Gateway Service on {HOST}:{PORT}")
    async with websockets.serve(gateway.handle_connection, HOST, PORT):
        logger.info(f"Gateway Service started on {HOST}:{PORT}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())