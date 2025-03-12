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
