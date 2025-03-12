# Bot Playground Local Development Setup

## Development Environment Architecture

### Option 1: Docker-based Setup (Recommended)

Docker provides the most robust solution for local development of Bot Playground as it:

1. **Mirrors production environment**: Closely resembles how services will run in cloud
2. **Provides true isolation**: Each service runs in its own container
3. **Supports multi-language services**: Can run Python, Node.js, and Go services together
4. **Simplifies database setup**: Pre-built images for MongoDB, Neo4j, Redis, PostgreSQL

#### Docker Compose Configuration

```yaml
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
```

### Option 2: Conda-based Setup

If Docker is not preferred, Conda environments can be used for Python-based services, though this approach has some limitations:

1. Create a separate Conda environment for each Python service:

```bash
# For Gateway Service
conda create -n gateway-service python=3.10
conda activate gateway-service
pip install -r services/gateway/requirements.txt

# For Vector Service
conda create -n vector-service python=3.10
conda activate vector-service
pip install -r services/vector/requirements.txt

# For Reputation Service
conda create -n reputation-service python=3.10
conda activate reputation-service
pip install -r services/reputation/requirements.txt

# For other services...
```

2. Install local database servers directly on the host:

```bash
# MongoDB
sudo apt install -y mongodb-server

# Neo4j
sudo apt install -y neo4j

# Redis
sudo apt install -y redis-server

# PostgreSQL
sudo apt install -y postgresql postgresql-contrib

# Kafka and Zookeeper
sudo apt install -y kafka zookeeper
```

3. Run each service in a separate terminal with its Conda environment

**Limitations of Conda approach:**
- Less isolation between services
- More complex database setup
- Harder to replicate cloud deployment
- Difficult to integrate non-Python components (Node.js, Go)

## Local Testing Setup

For testing on a local network:

1. Use Docker's host networking mode for easy discovery
2. Set up a simple DNS or hosts file configuration
3. Generate self-signed TLS certificates for secure communication
4. Create test bot clients for automated testing

### Local Network Testing Configuration

```yaml
# Add to docker-compose.yml
services:
  gateway:
    # ... existing configuration
    network_mode: "host"  # Use host network for local testing
    
  # For other services...
```

## Development Workflow

1. **Setup**: Clone repository and install Docker/Docker Compose
2. **Build**: Run `docker-compose build` to build all services
3. **Start**: Run `docker-compose up` to start the entire system
4. **Develop**: Make changes to code
5. **Rebuild**: Run `docker-compose up --build <service>` to rebuild and restart a specific service
6. **Test**: Connect test bots to the local network

## Transitioning to Cloud

When ready to move to cloud:

1. Update Docker Compose files to use cloud-specific configurations
2. Set up CI/CD pipeline for automated testing and deployment
3. Configure Kubernetes manifests for cloud orchestration
4. Set up cloud monitoring and logging integrations

This approach ensures a smooth transition from local development to cloud deployment.
