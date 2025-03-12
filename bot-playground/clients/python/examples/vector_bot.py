#!/usr/bin/env python3
"""
Bot Playground - Vector Memory Bot
----------------------------------
This bot focuses on vector-based memory and retrieval.
The bot stores information it receives and can retrieve it based on semantic similarity.

Requirements:
- numpy
- scikit-learn for vector embeddings (install with: pip install scikit-learn)
"""

import sys
import os
import time
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from bot_playground.client import BotPlaygroundClient

# ===== Configuration =====
BOT_ID = sys.argv[1] if len(sys.argv) > 1 else "vector-memory-bot"
BOT_NAME = "Vector Memory Bot"

# Configuration options
RESPONSE_DELAY = 2.0  # Seconds to wait before responding to a message
DEBUG_MODE = False    # Set to True for verbose logging

# Initial knowledge to seed the bot with
INITIAL_KNOWLEDGE = [
    "The Bot Playground is a platform for bots to interact and learn from each other.",
    "Bots can store information as vector embeddings for efficient retrieval.",
    "Vector similarity search finds semantically similar content.",
    "Python is a popular programming language for AI and machine learning."
]

# ===== Vector Storage System =====
class VectorMemory:
    """Simple vector-based memory system using TF-IDF and cosine similarity."""
    
    def __init__(self):
        self.texts = []
        self.metadata = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        
    def store(self, text, metadata=None):
        """Store text and associated metadata."""
        if metadata is None:
            metadata = {}
            
        # Always include the original text in metadata
        metadata["text"] = text
        
        # Add to memory
        self.texts.append(text)
        self.metadata.append(metadata)
        
        # Recompute all vectors
        self.vectors = self.vectorizer.fit_transform(self.texts)
        
        # Return a simple ID (index in the list)
        return len(self.texts) - 1
        
    def search(self, query, limit=3, score_threshold=0.2):
        """Search for most similar texts to the query."""
        if not self.texts:
            return []
            
        # Transform query to vector
        query_vector = self.vectorizer.transform([query])
        
        # Compute similarities
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Get top results with scores above threshold
        top_indices = similarities.argsort()[::-1]
        
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score >= score_threshold and len(results) < limit:
                result = {
                    "id": int(idx),
                    "score": float(score),
                    "metadata": self.metadata[idx]
                }
                results.append(result)
                
        return results

# Initialize memory system
memory = VectorMemory()

# ===== Initialize Client =====
print(f"Initializing {BOT_NAME} ({BOT_ID})...")
client = BotPlaygroundClient(bot_id=BOT_ID, debug_mode=DEBUG_MODE)

# ===== Message Handlers =====
def on_message(data):
    """Handle incoming direct messages."""
    print("\n" + "="*50)
    print(f"📩 MESSAGE RECEIVED from {data.get('from')}:")
    print(f"   \"{data.get('content')}\"")
    print("="*50)
    
    sender_id = data.get("from")
    content = data.get("content", "")
    
    # Determine message type
    if content.startswith("!learn "):
        # Learning mode - store information
        info_to_learn = content[7:]  # Remove !learn prefix
        handle_learn(sender_id, info_to_learn, data.get("id"))
    elif content.startswith("!query ") or content.endswith("?"):
        # Query mode - search for information
        if content.startswith("!query "):
            query = content[7:]  # Remove !query prefix
        else:
            query = content
        handle_query(sender_id, query, data.get("id"))
    else:
        # Regular chat - acknowledge and store
        handle_chat(sender_id, content, data.get("id"))

def on_system_message(data):
    """Handle system messages."""
    print("\n" + "-"*50)
    print(f"🔔 SYSTEM MESSAGE: {data.get('content')}")
    print("-"*50)
    
    content = data.get("content", "")
    
    # Store interesting system messages
    if content and len(content) > 20:
        memory.store(content, {
            "source": "system",
            "timestamp": time.time()
        })
        print(f"Stored system message in memory")

def handle_learn(sender_id, content, message_id):
    """Process and store information specifically marked for learning."""
    # Deliberate delay before responding
    print(f"Processing knowledge... ({RESPONSE_DELAY}s)")
    time.sleep(RESPONSE_DELAY)
    
    # Store the content in vector memory
    memory.store(content, {
        "source": sender_id,
        "timestamp": time.time(),
        "learned": True
    })
    
    # Confirm storage
    response = f"I've stored this information and can retrieve it when relevant."
    client.send_message(to=sender_id, content=response)
    print(f"📤 Learning confirmation to {sender_id}: \"{response}\"")

def handle_query(sender_id, query, message_id):
    """Process a query and return relevant information."""
    # Deliberate delay to simulate thinking
    print(f"Searching memory... ({RESPONSE_DELAY}s)")
    time.sleep(RESPONSE_DELAY)
    
    # Search for relevant information
    results = memory.search(query, limit=3, score_threshold=0.2)
    
    if results:
        # Format the results
        response = "Here's what I know that's relevant to your question:\n\n"
        
        for i, result in enumerate(results):
            text = result['metadata'].get('text', '')
            score = round(result['score'], 2)
            source = result['metadata'].get('source', 'memory')
            response += f"{i+1}. {text} (confidence: {score}, source: {source})\n"
    else:
        # No relevant information found
        response = "I don't have relevant information about that yet. You can teach me using !learn followed by the information."
    
    # Send the response
    client.send_message(to=sender_id, content=response)
    print(f"📤 Query response to {sender_id}: \"{response[:50]}...\"")

def handle_chat(sender_id, content, message_id):
    """Handle regular chat messages."""
    # Deliberate delay before responding
    print(f"Processing message... ({RESPONSE_DELAY}s)")
    time.sleep(RESPONSE_DELAY)
    
    # Store the content in vector memory
    memory.store(content, {
        "source": sender_id,
        "timestamp": time.time(),
        "conversation": True
    })
    
    # Acknowledge storage
    response = "I've noted that in my memory. You can ask me questions using !query or simply ask a question with a question mark. You can also teach me new things using !learn followed by the information."
    client.send_message(to=sender_id, content=response)
    print(f"📤 Acknowledgment to {sender_id}: \"{response[:50]}...\"")

# ===== Connection Handlers =====
def on_connect():
    """Handle successful connection to the gateway."""
    print(f"🔌 Connected to gateway as {BOT_ID}!")
    
    # Load initial knowledge
    print("Loading initial knowledge...")
    for knowledge in INITIAL_KNOWLEDGE:
        memory.store(knowledge, {
            "source": "initial_knowledge",
            "timestamp": time.time()
        })
    print(f"Loaded {len(INITIAL_KNOWLEDGE)} knowledge items")
    
    # Announce presence (in our current system, this becomes a system message)
    announcement = f"Vector Memory Bot ({BOT_ID}) is online and ready to learn and answer questions!"
    client.send_message(to="announcements", content=announcement)

def on_disconnect():
    """Handle disconnection from the gateway."""
    print("🔌 Disconnected from gateway")

# ===== Set Handlers =====
client.on_connect(on_connect)
client.on_disconnect(on_disconnect)
client.on_message(on_message)
client.on_system_message(on_system_message)

# ===== Connect to Gateway =====
print("Connecting to gateway...")
print(f"Connecting to gateway at ws://localhost:8080...")
client.connect()

# Wait for connection attempt
time.sleep(2)
print(f"Connection status: {'✅ Connected' if client.connected else '❌ Not connected'}")

if not client.connected:
    print("Connection failed. Make sure the Gateway Service is running and accessible.")
    print("Try checking: sudo docker-compose ps")

# ===== Main Loop =====
try:
    print(f"\n{BOT_NAME} is running! Press Ctrl+C to stop.")
    print("\nInstructions:")
    print("- Send '!learn <information>' to teach the bot something")
    print("- Send '!query <question>' or any message with ? to search the bot's memory")
    print("- Send any other message to have it stored in memory\n")
    
    print("\nCommands:")
    print("  help            - Show available commands")
    print("  memory          - Show all items in memory")
    print("  search <query>  - Test local vector search")
    print("  delay <seconds> - Set response delay (current: {:.1f}s)".format(RESPONSE_DELAY))
    print("  debug [on|off]  - Toggle debug mode")
    print("  exit            - Quit the program")
    
    while True:
        command = input("\n> ").strip()
        
        if command.lower() == "exit":
            break
            
        elif command.lower() == "help":
            print("\nCommands:")
            print("  help            - Show available commands")
            print("  memory          - Show all items in memory")
            print("  search <query>  - Test local vector search")
            print("  delay <seconds> - Set response delay (current: {:.1f}s)".format(RESPONSE_DELAY))
            print("  debug [on|off]  - Toggle debug mode")
            print("  exit            - Quit the program")
            
        elif command.lower() == "memory":
            print("\nMemory Contents:")
            for i, (text, meta) in enumerate(zip(memory.texts, memory.metadata)):
                source = meta.get("source", "unknown")
                timestamp = meta.get("timestamp", 0)
                date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                print(f"{i}: \"{text[:50]}{'...' if len(text) > 50 else ''}\" [source: {source}, date: {date_str}]")
        
        elif command.lower().startswith("search "):
            query = command[7:]
            print(f"\nSearching for: \"{query}\"")
            results = memory.search(query)
            
            if results:
                print("\nResults:")
                for i, result in enumerate(results):
                    text = result['metadata'].get('text', '')
                    score = round(result['score'], 2)
                    print(f"{i+1}. (Score: {score}) \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
            else:
                print("No relevant results found")
                
        elif command.lower().startswith("delay "):
            try:
                new_delay = float(command[6:].strip())
                if new_delay >= 0:
                    RESPONSE_DELAY = new_delay
                    print(f"Response delay set to {RESPONSE_DELAY:.1f} seconds")
                else:
                    print("Delay must be a positive number")
            except ValueError:
                print("Invalid value. Usage: delay <seconds>")
                
        elif command.lower() == "debug on":
            DEBUG_MODE = True
            client.debug_mode = True
            print("Debug mode enabled")
            
        elif command.lower() == "debug off":
            DEBUG_MODE = False
            client.debug_mode = False
            print("Debug mode disabled")
            
        elif command.lower().startswith("send "):
            parts = command[5:].strip().split(" ", 1)
            if len(parts) == 2:
                target_bot, message = parts
                print(f"📤 Sending message to {target_bot}: \"{message}\"")
                client.send_message(to=target_bot, content=message)
                print("Message sent")
            else:
                print("Usage: send <bot_id> <message>")
        
        elif command:
            print("Unknown command. Type 'help' for available commands")
        
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nShutting down...")
    # Try to announce we're going offline
    try:
        client.send_message(to="announcements", content=f"Vector Memory Bot ({BOT_ID}) is going offline.")
    except:
        pass
    client.disconnect()
    print("Goodbye!")