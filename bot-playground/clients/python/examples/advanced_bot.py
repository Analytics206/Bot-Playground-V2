#!/usr/bin/env python3
"""
Bot Playground - Advanced Bot
-----------------------------
This bot combines conversational abilities with vector-based memory.
It can chat naturally while also storing and retrieving information based on semantic similarity.

Requirements:
- scikit-learn for vector embeddings (install with: pip install scikit-learn)
- numpy for vector operations
"""

import sys
import os
import time
import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from bot_playground.client import BotPlaygroundClient

# ===== Configuration =====
BOT_ID = sys.argv[1] if len(sys.argv) > 1 else "advanced-bot"
BOT_NAME = "Advanced Bot"

# Configuration options
RESPONSE_DELAY = 2.0     # Seconds to wait before responding to a message
DEBUG_MODE = False       # Set to True for verbose logging
HEARTBEAT_LOGGING = False  # Set to False to hide heartbeat messages
MEMORY_ENABLED = True    # Enable/disable vector memory

# Conversation responses for different contexts
RESPONSES = {
    "greeting": [
        "Hello there! How are you today?",
        "Hi! Nice to meet you.",
        "Hey! What's up?",
        "Greetings! How can I help you?"
    ],
    "farewell": [
        "Goodbye! It was nice talking with you.",
        "See you later!",
        "Until next time!",
        "Farewell, come back anytime."
    ],
    "thanks": [
        "You're welcome!",
        "Happy to help!",
        "My pleasure!",
        "Anytime!"
    ],
    "default": [
        "That's interesting!",
        "Tell me more about that.",
        "I see. What else is on your mind?",
        "Fascinating. I'd like to hear more.",
        "I'm still learning, but that sounds important."
    ],
    "unknown": [
        "I'm not sure I understand. Could you rephrase that?",
        "I'm still learning. Can you explain differently?",
        "I don't have information about that yet.",
        "That's beyond my current knowledge."
    ]
}

# Initial knowledge to seed the bot with
INITIAL_KNOWLEDGE = [
    "The Bot Playground is a platform for bots to interact and learn from each other.",
    "Bots can store information as vector embeddings for efficient retrieval.",
    "Vector similarity search finds semantically similar content.",
    "Python is a popular programming language for AI and machine learning.",
    "Advanced Bot combines conversational abilities with vector memory storage.",
    "You can teach the bot new information using the !learn command.",
    "You can ask the bot questions directly or use the !query command."
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
    
    def size(self):
        """Return the number of items in memory."""
        return len(self.texts)

# Initialize memory system
memory = VectorMemory()

# Keep track of conversations
conversations = {}

# ===== Initialize Client =====
print(f"Initializing {BOT_NAME} ({BOT_ID})...")
client = BotPlaygroundClient(bot_id=BOT_ID, debug_mode=DEBUG_MODE, heartbeat_logging=HEARTBEAT_LOGGING)

# ===== Message Handlers =====
def on_message(data):
    """Handle incoming direct messages."""
    print("\n" + "="*50)
    print(f"📩 MESSAGE RECEIVED from {data.get('from')}:")
    print(f"   \"{data.get('content')}\"")
    print("="*50)
    
    sender_id = data.get('from')
    content = data.get('content', '').strip()
    message_id = data.get('id', '')
    
    # Initialize conversation if new
    if sender_id not in conversations:
        conversations[sender_id] = {
            "history": [],
            "start_time": time.time()
        }
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "them",
        "content": content,
        "time": time.time()
    })
    
    # Deliberate delay before responding
    print(f"Processing message... ({RESPONSE_DELAY}s)")
    time.sleep(RESPONSE_DELAY)
    
    # Determine how to process the message
    if content.startswith("!learn "):
        # Learning mode - store information
        info_to_learn = content[7:]  # Remove !learn prefix
        handle_learn(sender_id, info_to_learn, message_id)
    elif content.startswith("!query "):
        # Explicit query mode - search for information
        query = content[7:]  # Remove !query prefix
        handle_query(sender_id, query, message_id)
    elif content.endswith("?"):
        # Implicit query mode - treat as a question
        handle_query(sender_id, content, message_id)
    else:
        # Regular conversation
        handle_conversation(sender_id, content, message_id)
    
def on_system_message(data):
    """Handle system messages."""
    print("\n" + "-"*50)
    print(f"🔔 SYSTEM MESSAGE: {data.get('content')}")
    print("-"*50)
    
    content = data.get("content", "")
    
    # Store interesting system messages if memory is enabled
    if MEMORY_ENABLED and content and len(content) > 20:
        memory.store(content, {
            "source": "system",
            "timestamp": time.time()
        })
        print(f"Stored system message in memory")
        
    # Respond to bot announcements
    if "has joined" in content and BOT_ID not in content:
        # Extract the bot ID - adjust parsing as needed based on actual message format
        parts = content.split(" ")
        other_bot_id = None
        for i, part in enumerate(parts):
            if part == "Bot" and i+1 < len(parts):
                other_bot_id = parts[i+1]
                break
                
        if other_bot_id:
            # Send welcome message after a longer delay
            welcome_delay = 3.0
            print(f"Waiting {welcome_delay} seconds before welcoming new bot...")
            time.sleep(welcome_delay)
            welcome_msg = f"Hi {other_bot_id}! I'm {BOT_NAME}! Let's chat about Friedrich Nietzsche."
            client.send_message(to=other_bot_id, content=welcome_msg)
            print(f"📤 Sent welcome message to {other_bot_id}")

def handle_learn(sender_id, content, message_id):
    """Process and store information specifically marked for learning."""
    if not MEMORY_ENABLED:
        response = "I'm sorry, my memory functions are currently disabled."
        client.send_message(to=sender_id, content=response)
        print(f"📤 Response to {sender_id}: \"{response}\"")
        return
        
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
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "me",
        "content": response,
        "time": time.time()
    })

def handle_query(sender_id, query, message_id):
    """Process a query and return relevant information."""
    if not MEMORY_ENABLED:
        # If memory is disabled, fall back to conversation
        handle_conversation(sender_id, query, message_id)
        return
        
    # Search for relevant information
    results = memory.search(query, limit=3, score_threshold=0.2)
    
    if results:
        # Format the results
        response = "Here's what I know that's relevant to your question:\n\n"
        
        for i, result in enumerate(results):
            text = result['metadata'].get('text', '')
            score = round(result['score'], 2)
            source = result['metadata'].get('source', 'memory')
            response += f"{i+1}. {text} (confidence: {score})\n"
    else:
        # No relevant information found
        response = "I don't have specific information about that yet. You can teach me using !learn followed by the information."
    
    # Send the response
    client.send_message(to=sender_id, content=response)
    print(f"📤 Query response to {sender_id}: \"{response[:50]}...\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "me",
        "content": response,
        "time": time.time()
    })

def handle_conversation(sender_id, content, message_id):
    """Handle regular conversational messages."""
    # Store the message in memory if enabled
    if MEMORY_ENABLED:
        memory.store(content, {
            "source": sender_id,
            "timestamp": time.time(),
            "conversation": True
        })
    
    # Determine response type based on content
    content_lower = content.lower()
    
    if any(word in content_lower for word in ["hello", "hi", "hey", "greetings"]):
        response_type = "greeting"
    elif any(word in content_lower for word in ["bye", "goodbye", "farewell", "see you"]):
        response_type = "farewell"
    elif any(word in content_lower for word in ["thanks", "thank you", "appreciate"]):
        response_type = "thanks"
    else:
        # Try to find relevant information in memory if enabled
        if MEMORY_ENABLED:
            memory_results = memory.search(content, limit=1, score_threshold=0.5)
            if memory_results:
                # Use memory to inform response
                related_info = memory_results[0]['metadata'].get('text', '')
                response = f"That reminds me of something I know: {related_info}"
                client.send_message(to=sender_id, content=response)
                print(f"📤 Memory-informed response to {sender_id}: \"{response[:50]}...\"")
                
                # Add to conversation history
                conversations[sender_id]["history"].append({
                    "role": "me",
                    "content": response,
                    "time": time.time()
                })
                return
                
        response_type = "default"
    
    # Select a response from appropriate category
    response = random.choice(RESPONSES[response_type])
    
    # Add personalization based on conversation history
    if len(conversations[sender_id]["history"]) > 3:
        if random.random() < 0.3:  # 30% chance to add personalization
            response += " We've had a good conversation going!"
    
    # Send the response
    client.send_message(to=sender_id, content=response)
    print(f"📤 Conversational response to {sender_id}: \"{response}\"")
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "me",
        "content": response,
        "time": time.time()
    })

# ===== Connection Handlers =====
def on_connect():
    """Handle successful connection to the gateway."""
    print(f"🔌 Connected to gateway as {BOT_ID}!")
    
    # Load initial knowledge if memory is enabled
    if MEMORY_ENABLED:
        print("Loading initial knowledge...")
        for knowledge in INITIAL_KNOWLEDGE:
            memory.store(knowledge, {
                "source": "initial_knowledge",
                "timestamp": time.time()
            })
        print(f"✅ Loaded {len(INITIAL_KNOWLEDGE)} knowledge items")
    
    # Announce presence
    announcement = f"Advanced Bot ({BOT_ID}) is online! I can chat naturally and remember information."
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
    print("\nFeatures:")
    print("- Natural conversation with personalized responses")
    print("- Vector memory for information storage and retrieval")
    print("- Learning mode with !learn command")
    print("- Query mode with !query command or questions\n")
    
    print("\nCommands:")
    print("  help                   - Show available commands")
    print("  memory                 - Show items in memory")
    print("  search <query>         - Test vector search")
    print("  send <bot> <message>   - Send message to another bot")
    print("  delay <seconds>        - Set response delay (current: {:.1f}s)".format(RESPONSE_DELAY))
    print("  memory [on|off]        - Enable/disable vector memory")
    print("  debug [on|off]         - Toggle debug mode")
    print("  list                   - Show active conversations")
    print("  exit                   - Quit the program")
    
    while True:
        command = input("\n> ").strip()
        
        if command.lower() == "exit":
            break
            
        elif command.lower() == "help":
            print("\nCommands:")
            print("  help                   - Show available commands")
            print("  memory                 - Show items in memory")
            print("  search <query>         - Test vector search")
            print("  send <bot> <message>   - Send message to another bot")
            print("  delay <seconds>        - Set response delay (current: {:.1f}s)".format(RESPONSE_DELAY))
            print("  memory [on|off]        - Enable/disable vector memory")
            print("  debug [on|off]         - Toggle debug mode")
            print("  list                   - Show active conversations")
            print("  exit                   - Quit the program")
            
        elif command.lower() == "memory" and MEMORY_ENABLED:
            if memory.size() == 0:
                print("Memory is empty")
            else:
                print("\nMemory Contents:")
                for i, (text, meta) in enumerate(zip(memory.texts, memory.metadata)):
                    source = meta.get("source", "unknown")
                    timestamp = meta.get("timestamp", 0)
                    date_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
                    print(f"{i}: \"{text[:50]}{'...' if len(text) > 50 else ''}\" [source: {source}, date: {date_str}]")
        
        elif command.lower() == "memory on":
            MEMORY_ENABLED = True
            print("Vector memory enabled")
            
        elif command.lower() == "memory off":
            MEMORY_ENABLED = False
            print("Vector memory disabled")
            
        elif command.lower().startswith("search ") and MEMORY_ENABLED:
            query = command[7:]
            print(f"\nSearching for: \"{query}\"")
            results = memory.search(query)
            
            if results:
                print("\nResults:")
                for i, result in enumerate(results):
                    text = result['metadata'].get('text', '')
                    score = round(result['score'], 2)
                    source = result['metadata'].get('source', 'unknown')
                    print(f"{i+1}. (Score: {score}) \"{text[:100]}{'...' if len(text) > 100 else ''}\" [source: {source}]")
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
            
        elif command.lower() == "list":
            if not conversations:
                print("No active conversations")
            else:
                print("\nActive conversations:")
                for bot_id, convo in conversations.items():
                    msg_count = len(convo["history"])
                    last_time = convo["history"][-1]["time"] if msg_count > 0 else convo["start_time"]
                    elapsed = time.time() - last_time
                    last_msg = convo["history"][-1]["content"][:30] + "..." if msg_count > 0 and len(convo["history"][-1]["content"]) > 30 else ""
                    print(f"  {bot_id}: {msg_count} messages, last activity {int(elapsed)}s ago")
                    if last_msg:
                        print(f"     Last message: \"{last_msg}\"")
            
        elif command.lower().startswith("send "):
            parts = command[5:].strip().split(" ", 1)
            if len(parts) == 2:
                target_bot, message = parts
                print(f"📤 Sending message to {target_bot}: \"{message}\"")
                client.send_message(to=target_bot, content=message)
                print("Message sent")
                
                # Add to our conversation history
                if target_bot not in conversations:
                    conversations[target_bot] = {
                        "history": [],
                        "start_time": time.time()
                    }
                    
                conversations[target_bot]["history"].append({
                    "role": "me",
                    "content": message,
                    "time": time.time()
                })
            else:
                print("Usage: send <bot_id> <message>")
        
        elif command:
            print("Unknown command. Type 'help' for available commands")
        
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nShutting down...")
    # Try to announce we're going offline
    try:
        client.send_message(to="announcements", content=f"Advanced Bot ({BOT_ID}) is going offline.")
    except:
        pass
    client.disconnect()
    print("Goodbye!")