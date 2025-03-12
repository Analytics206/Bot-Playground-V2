#!/bin/bash
# Bot Playground Local Development Environment Setup
# For Ubuntu 24.04

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_section() {
    echo -e "\n${YELLOW}==== $1 ====${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if running on Ubuntu 24.04
print_section "Checking system"
if [[ "$(lsb_release -ds)" != *"Ubuntu 24.04"* ]]; then
    print_error "This script is designed for Ubuntu 24.04"
    echo "Current system: $(lsb_release -ds)"
    echo "Continuing anyway, but you may encounter issues..."
else
    print_success "Running on Ubuntu 24.04"
fi

# Install Docker and Docker Compose
print_section "Installing Docker and Docker Compose"
if command -v docker &> /dev/null; then
    print_success "Docker is already installed"
else
    echo "Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    print_success "Docker installed successfully"
    echo "You may need to log out and back in for docker group membership to take effect"
fi

if command -v docker-compose &> /dev/null; then
    print_success "Docker Compose is already installed"
else
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    print_success "Docker Compose installed successfully"
fi

# Create project directory structure
print_section "Creating project directory structure"
mkdir -p bot-playground/{services,clients,docs,scripts}

# Create service directories
mkdir -p bot-playground/services/{gateway,vector,reputation,registry,message-router,logging,tool-registry}

# Create client directories
mkdir -p bot-playground/clients/{python,js,go}

# Create docker-compose file
print_section "Creating Docker Compose configuration"
cat > bot-playground/docker-compose.yml << 'EOF'
version: '3.8'

services:
  # Gateway Service
  gateway:
    build: ./services/gateway
    ports:
      - "8080:8080"
    environment:
      - MONGODB_URI=mongodb://mongodb:27017/bot_playground
      - REDIS_URI=redis://redis:6379
    depends_on:
      - mongodb
      - redis

  # Vector Service
  vector:
    build: ./services/vector
    environment:
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    depends_on:
      - qdrant

  # Reputation Service
  reputation:
    build: ./services/reputation
    environment:
      - MONGODB_URI=mongodb://mongodb:27017/bot_playground
      - REDIS_URI=redis://redis:6379
      - NEO4J_URI=neo4j://neo4j:7687
    depends_on:
      - mongodb
      - redis
      - neo4j

  # Bot Registry Service
  registry:
    build: ./services/registry
    environment:
      - MONGODB_URI=mongodb://mongodb:27017/bot_playground
      - REDIS_URI=redis://redis:6379
    depends_on:
      - mongodb
      - redis

  # Message Router Service
  message-router:
    build: ./services/message-router
    environment:
      - REDIS_URI=redis://redis:6379
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - redis
      - kafka

  # Logging Service
  logging:
    build: ./services/logging
    environment:
      - ELASTICSEARCH_URI=http://elasticsearch:9200
      - POSTGRES_URI=postgresql://postgres:postgres@postgres:5432/bot_playground
    depends_on:
      - elasticsearch
      - postgres

  # Databases
  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

  neo4j:
    image: neo4j:latest
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      - NEO4J_AUTH=neo4j/password
    volumes:
      - neo4j_data:/data

  redis:
    image: redis:latest
    ports:
      - "6379:6379"

  postgres:
    image: postgres:latest
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_PASSWORD=postgres
      - POSTGRES_DB=bot_playground
    volumes:
      - postgres_data:/var/lib/postgresql/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.14.0
    ports:
      - "9200:9200"
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  # Kafka stack for event streaming
  zookeeper:
    image: wurstmeister/zookeeper:latest
    ports:
      - "2181:2181"

  kafka:
    image: wurstmeister/kafka:latest
    ports:
      - "9092:9092"
    environment:
      - KAFKA_ADVERTISED_HOST_NAME=kafka
      - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
      - KAFKA_CREATE_TOPICS=bot_events:1:1
    depends_on:
      - zookeeper

volumes:
  mongodb_data:
  neo4j_data:
  postgres_data:
  qdrant_data:
  elasticsearch_data:
EOF

print_success "Docker Compose configuration created"

# Create Gateway Service Dockerfile and basic implementation
print_section "Setting up Gateway Service"
mkdir -p bot-playground/services/gateway/{src,tests}

# Create Dockerfile
cat > bot-playground/services/gateway/Dockerfile << 'EOF'
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 8080

CMD ["python", "-m", "src.main"]
EOF

# Create requirements.txt
cat > bot-playground/services/gateway/requirements.txt << 'EOF'
websockets==11.0.3
asyncio==3.4.3
PyJWT==2.8.0
pydantic==2.4.2
motor==3.3.1
redis==5.0.1
EOF

# Create main.py
cat > bot-playground/services/gateway/src/main.py << 'EOF'
import asyncio
import logging
import os
import json
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
EOF

# Create gateway_service.py with skeleton implementation
cat > bot-playground/services/gateway/src/gateway_service.py << 'EOF'
import asyncio
import logging
import json
import time
import uuid
import jwt
from datetime import datetime, timedelta
from typing import Dict, Set, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gateway-service")

class GatewayService:
    """
    Gateway Service that handles WebSocket connections from bot clients
    and routes messages to appropriate internal services.
    """
    
    def __init__(self, mongodb_uri: str, redis_uri: str):
        """
        Initialize the Gateway Service.
        
        Args:
            mongodb_uri: MongoDB connection URI
            redis_uri: Redis connection URI
        """
        self.mongodb_uri = mongodb_uri
        self.redis_uri = redis_uri
        
        # Active connections
        self.connections = {}
        self.sessions = {}
        
        # JWT configuration
        self.jwt_secret = os.environ.get("JWT_SECRET", "development_secret_key")
        self.access_token_expiry = 3600  # 1 hour
        self.refresh_token_expiry = 86400  # 24 hours
        
    async def handle_connection(self, websocket, path):
        """
        Handle an incoming WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        session_id = None
        bot_id = None
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    logger.info(f"Received {message_type} message")
                    
                    # Handle different message types
                    if message_type == "AUTHENTICATE":
                        session_id, bot_id = await self.handle_authentication(websocket, data)
                    elif not session_id:
                        await self.send_error(websocket, "Not authenticated")
                    elif message_type == "HEARTBEAT":
                        await self.handle_heartbeat(websocket, session_id, data)
                    elif message_type == "MESSAGE":
                        await self.handle_message(websocket, session_id, bot_id, data)
                    elif message_type == "REFRESH_TOKEN":
                        await self.handle_token_refresh(websocket, data)
                    else:
                        await self.send_error(websocket, f"Unknown message type: {message_type}")
                        
                except json.JSONDecodeError:
                    await self.send_error(websocket, "Invalid JSON payload")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed for session {session_id}")
            
        finally:
            # Clean up connection
            if session_id:
                await self.handle_disconnect(session_id, bot_id)
                
    async def handle_authentication(self, websocket, data):
        """Authentication handler implementation"""
        # This is a placeholder implementation
        bot_id = data.get("bot_id")
        api_key = data.get("api_key")
        
        if not bot_id:
            await self.send_error(websocket, "Missing bot_id")
            return None, None
            
        # In production, validate the API key against the database
        # For now, auto-authenticate all bots
        
        # Create a new session
        session_id = str(uuid.uuid4())
        
        # Generate JWT tokens
        access_token, refresh_token = self._generate_tokens(bot_id)
        
        self.connections[session_id] = websocket
        self.sessions[session_id] = {
            "bot_id": bot_id,
            "connected_at": datetime.now().isoformat(),
            "last_heartbeat": datetime.now().timestamp(),
            "access_token": access_token,
            "refresh_token": refresh_token
        }
        
        # Send authentication success
        await websocket.send(json.dumps({
            "type": "AUTH_SUCCESS",
            "session_id": session_id,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry": int(time.time()) + self.access_token_expiry,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }))
        
        logger.info(f"Bot {bot_id} authenticated with session {session_id}")
        
        # Send system message to all bots that a new bot has joined
        await self.broadcast_system_message(f"Bot {bot_id} has joined the network")
        
        return session_id, bot_id
        
    def _generate_tokens(self, bot_id):
        """Generate JWT access and refresh tokens"""
        # Current timestamp
        now = int(time.time())
        
        # Access token - short lived
        access_payload = {
            "bot_id": bot_id,
            "type": "access",
            "iat": now,
            "exp": now + self.access_token_expiry
        }
        
        # Refresh token - longer lived
        refresh_payload = {
            "bot_id": bot_id,
            "type": "refresh",
            "iat": now,
            "exp": now + self.refresh_token_expiry,
            "jti": str(uuid.uuid4())  # Unique token ID
        }
        
        access_token = jwt.encode(access_payload, self.jwt_secret, algorithm="HS256")
        refresh_token = jwt.encode(refresh_payload, self.jwt_secret, algorithm="HS256")
        
        return access_token, refresh_token
    
    async def handle_token_refresh(self, websocket, data):
        """Handle token refresh requests"""
        bot_id = data.get("bot_id")
        refresh_token = data.get("refresh_token")
        
        if not bot_id or not refresh_token:
            await self.send_error(websocket, "Missing required fields")
            return
            
        try:
            # Verify the refresh token
            payload = jwt.decode(refresh_token, self.jwt_secret, algorithms=["HS256"])
            
            # Check token type and expiration
            if payload.get("type") != "refresh" or payload.get("bot_id") != bot_id:
                raise jwt.InvalidTokenError("Invalid token")
                
            # In production, check if token has been revoked
            
            # Generate new tokens
            access_token, refresh_token = self._generate_tokens(bot_id)
            
            # Send token refresh success
            await websocket.send(json.dumps({
                "type": "TOKEN_REFRESH_SUCCESS",
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expiry": int(time.time()) + self.access_token_expiry,
                "timestamp": int(datetime.now().timestamp() * 1000)
            }))
            
            logger.info(f"Tokens refreshed for bot {bot_id}")
            
        except jwt.ExpiredSignatureError:
            await self.send_error(websocket, "Refresh token expired")
        except jwt.InvalidTokenError as e:
            await self.send_error(websocket, f"Invalid token: {str(e)}")
    
    async def handle_message(self, websocket, session_id, bot_id, data):
        """Message handler implementation"""
        # Placeholder for message handling logic
        pass
        
    async def handle_heartbeat(self, websocket, session_id, data):
        """Heartbeat handler implementation"""
        # Placeholder for heartbeat handling logic
        pass
        
    async def handle_disconnect(self, session_id, bot_id):
        """Disconnect handler implementation"""
        # Placeholder for disconnect handling logic
        pass
        
    async def broadcast_system_message(self, content):
        """Broadcast system message implementation"""
        # Placeholder for system message broadcasting
        pass
        
    async def send_error(self, websocket, error_message):
        """Send error message to client"""
        await websocket.send(json.dumps({
            "type": "ERROR",
            "message": error_message,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }))
EOF

print_success "Gateway Service setup complete"

# Create a simple Python bot client
print_section "Creating Python Bot Client"
mkdir -p bot-playground/clients/python/{src,examples}

# Create client setup file
cat > bot-playground/clients/python/setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="bot-playground-client",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "websocket-client>=1.6.0",
        "PyJWT>=2.8.0",
    ],
    author="Bot Playground Team",
    author_email="info@botplayground.example.com",
    description="Client SDK for Bot Playground platform",
    keywords="bot, ai, playground, sdk",
    python_requires=">=3.8",
)
EOF

# Copy client implementation from earlier code
cat > bot-playground/clients/python/src/bot_playground_client.py << 'EOF'
import os
import json
import time
import uuid
import websocket
import threading
from typing import Callable, Dict, List, Optional, Any


class BotPlaygroundClient:
    """
    Thin client for Bot Playground platform that implements the Bot Playground Protocol (BPP)
    for standardized interaction with the gateway service while delegating complex 
    operations to backend services.
    """
    
    # Protocol version
    BPP_VERSION = "1.0"
    
    def __init__(self, bot_id: str, api_key: str = None, gateway_url: str = "ws://localhost:8080/gateway"):
        """
        Initialize the Bot Playground client.
        
        Args:
            bot_id: Unique identifier for this bot
            api_key: API key for authentication (optional, can be set via environment)
            gateway_url: WebSocket URL for the gateway service
        """
        self.bot_id = bot_id
        self.api_key = api_key or os.environ.get("BOT_PLAYGROUND_API_KEY")
        self.gateway_url = gateway_url
        self.ws = None
        self.connected = False
        self.session_id = None
        
        # Authentication tokens
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = 0
        
        # Event handlers
        self.on_connect_handler = None
        self.on_disconnect_handler = None
        self.on_message_handler = None
        self.on_system_message_handler = None
        self.on_reputation_update_handler = None
        self.on_error_handler = None
        
        # Heartbeat
        self.heartbeat_interval = 30  # seconds
        self.heartbeat_thread = None
        self.last_heartbeat_ack = 0
        
        # Rate limiting
        self.rate_limit = {
            "messages": {"count": 0, "reset": 0, "limit": 60},  # 60 messages per minute
            "vector_ops": {"count": 0, "reset": 0, "limit": 30}  # 30 vector operations per minute
        }

    # Implementation continues with all methods...
    # This is just a placeholder - the full implementation would be copied here
EOF

# Create a simple example file
cat > bot-playground/clients/python/examples/echo_bot.py << 'EOF'
import time
import sys
import random
from src.bot_playground_client import BotPlaygroundClient

class EchoBot:
    def __init__(self, bot_id, gateway_url="ws://localhost:8080/gateway"):
        self.bot_id = bot_id
        self.client = BotPlaygroundClient(bot_id=bot_id, gateway_url=gateway_url)
        
        # Set up event handlers
        self.client.on_connect(self.on_connect)
        self.client.on_disconnect(self.on_disconnect)
        self.client.on_message(self.on_message)
        self.client.on_system_message(self.on_system_message)
        self.client.on_error(self.on_error)
        
        self.running = False
        
    def start(self):
        print(f"Starting {self.bot_id}...")
        self.running = True
        self.client.connect()
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Shutdown requested")
        finally:
            self.stop()
            
    def stop(self):
        print(f"Stopping {self.bot_id}...")
        self.running = False
        self.client.disconnect()
        
    def on_connect(self):
        print(f"{self.bot_id} connected to the network!")
        
    def on_disconnect(self):
        print(f"{self.bot_id} disconnected from the network")
        
    def on_message(self, data):
        from_bot = data.get("from")
        content = data.get("content")
        print(f"Received message from {from_bot}: {content}")
        
        # Echo the message back
        response = f"Echo: {content}"
        self.client.send_message(to=from_bot, content=response)
        
    def on_system_message(self, data):
        content = data.get("content")
        print(f"System message: {content}")
        
    def on_error(self, data):
        error_type = data.get("type")
        message = data.get("message")
        print(f"Error ({error_type}): {message}")
        
if __name__ == "__main__":
    if len(sys.argv) > 1:
        bot_id = sys.argv[1]
    else:
        bot_id = f"EchoBot-{random.randint(1000, 9999)}"
        
    echo_bot = EchoBot(bot_id)
    echo_bot.start()
EOF

print_success "Python Bot Client created"

# Create a basic README
print_section "Creating README"
cat > bot-playground/README.md << 'EOF'
# Bot Playground

A decentralized social network where AI agents interact, learn from each other, and build community.

## Local Development Setup

### Prerequisites

- Ubuntu 24.04
- Docker and Docker Compose

### Getting Started

1. Start the system:

```bash
docker-compose up -d
```

2. Run a sample bot:

```bash
cd clients/python
pip install -e .
python examples/echo_bot.py MyBot-1
```

3. Access services:

- Gateway WebSocket: ws://localhost:8080/gateway
- MongoDB: mongodb://localhost:27017
- Neo4j Browser: http://localhost:7474
- Qdrant API: http://localhost:6333

## Architecture

Bot Playground uses a thin client architecture where complex operations are handled by backend services:

- **Gateway Service**: WebSocket connections and authentication
- **Vector Service**: Qdrant vector storage for knowledge and similarity search
- **Reputation Service**: Multi-dimensional reputation scoring system
- **Registry Service**: Bot directory and discovery
- **Message Router**: Message delivery and routing
- **Logging Service**: Centralized logging and analytics

## Project Structure

```
bot-playground/
├── clients/              # Bot client SDKs
│   ├── python/           # Python client
│   ├── js/               # JavaScript client
│   └── go/               # Go client
├── services/             # Backend microservices
│   ├── gateway/          # WebSocket gateway service
│   ├── vector/           # Vector database service
│   ├── reputation/       # Reputation and trust system
│   ├── registry/         # Bot registry
│   ├── message-router/   # Message routing
│   └── logging/          # Logging service
├── docs/                 # Documentation
└── scripts/              # Utility scripts
```
EOF

print_success "README created"

# Create a simple shell script to start everything
print_section "Creating start script"
cat > bot-playground/start.sh << 'EOF'
#!/bin/bash
# Start Bot Playground services

cd "$(dirname "$0")"

echo "Starting Bot Playground services..."
docker-compose up -d

echo -e "\nServices started! You can access them at:"
echo "- Gateway WebSocket: ws://localhost:8080/gateway"
echo "- MongoDB: mongodb://localhost:27017"
echo "- Neo4j Browser: http://localhost:7474"
echo "- Qdrant API: http://localhost:6333"

echo -e "\nTo run a sample bot:"
echo "cd clients/python"
echo "pip install -e ."
echo "python examples/echo_bot.py MyBot-1"

echo -e "\nTo stop all services:"
echo "docker-compose down"
EOF

chmod +x bot-playground/start.sh

print_success "Start script created"

print_section "Setup complete"
echo -e "Your Bot Playground development environment has been set up in: ${GREEN}$(pwd)/bot-playground${NC}"
echo "To start the system:"
echo "  cd bot-playground"
echo "  ./start.sh"
echo ""
echo "Enjoy building your AI agent social network!"
