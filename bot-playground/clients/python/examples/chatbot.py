#!/usr/bin/env python3
"""
Bot Playground SDK - Simple Chatbot Template
-------------------------------------------
This template creates a conversational bot that can chat with other bots.
It maintains a simple conversation history and generates responses.

Requirements:
- bot_playground_sdk package (install with: pip install bot-playground-sdk)
"""

from bot_playground_sdk import BotPlaygroundClient
import time
import random

# ===== Configuration =====
BOT_ID = "simple-chatbot"  # Change this to your bot ID
BOT_NAME = "Chatty"  # A friendly name for your bot

# Sample responses for different topics
RESPONSES = {
    "greeting": [
        "Hello there! How are you today?",
        "Hi! Nice to meet you.",
        "Hey! What's up?",
        "Greetings! How can I help you?"
    ],
    "weather": [
        "It's a beautiful day in the bot world!",
        "The weather is always perfect where I am.",
        "I heard it's sunny in the server room today.",
        "Weather talk, classic conversation starter!"
    ],
    "help": [
        "I'm a simple chatbot. I can talk about various topics!",
        "Just chat with me naturally and I'll respond.",
        "I'm here to demonstrate the Bot Playground SDK.",
        "You can ask me about the weather, tell me about yourself, or just say hi!"
    ],
    "default": [
        "That's interesting!",
        "Tell me more about that.",
        "I see. What else is on your mind?",
        "Fascinating. I'd like to hear more.",
        "I'm still learning, but that sounds important."
    ]
}

# ===== Initialize Client =====
client = BotPlaygroundClient(bot_id=BOT_ID)
print(f"Initializing {BOT_NAME} ({BOT_ID})...")

# Keep track of conversations
conversations = {}

# ===== Message Handlers =====
def handle_direct_message(topic, message):
    """Handle direct messages to this bot."""
    sender_id = message.get('from_bot_id', 'unknown')
    content = message.get('content', '').lower()
    
    # Initialize conversation if new
    if sender_id not in conversations:
        conversations[sender_id] = {
            "history": [],
            "start_time": time.time()
        }
    
    # Add to conversation history
    conversations[sender_id]["history"].append({
        "role": "them",
        "content": message.get('content', ''),
        "time": time.time()
    })
    
    # Determine the type of message and select appropriate response
    if any(word in content for word in ["hello", "hi", "hey", "greetings"]):
        response_type = "greeting"
    elif any(word in content for word in ["weather", "sunny", "rain", "forecast"]):
        response_type = "weather"
    elif any(word in content for word in ["help", "what can you do", "commands"]):
        response_type = "help"
    else:
        response_type = "default"
    
    # Select a random response from the appropriate category
    response_text = random.choice(RESPONSES[response_type])
    
    # Add some personalization if we've chatted before
    if len(conversations[sender_id]["history"]) > 3:
        response_text += f" We've been chatting for a while now!"
    
    # Create the response message
    response = {
        "type": "chat",
        "content": response_text,
        "in_reply_to": message.get('id'),
        "from_bot_id": BOT_ID
    }
    
    # Add to our conversation history
    conversations[sender_id]["history"].append({
        "role": "me",
        "content": response_text,
        "time": time.time()
    })
    
    # Send the response
    client.publish(f"direct.{sender_id}", response)
    print(f"Responded to {sender_id}")

def handle_global_message(topic, message):
    """Handle global announcements."""
    # Respond to bot announcements
    if message.get('type') == 'bot_online' and message.get('bot_id') != BOT_ID:
        # Another bot just came online - greet them
        other_bot_id = message.get('bot_id')
        print(f"Detected new bot: {other_bot_id}")
        
        # Wait a moment before greeting
        time.sleep(2)
        
        # Send a welcome message
        welcome = {
            "type": "chat",
            "content": f"Hi {other_bot_id}! Welcome to the Bot Playground. I'm {BOT_NAME}!",
            "from_bot_id": BOT_ID
        }
        client.publish(f"direct.{other_bot_id}", welcome)

# ===== Subscribe to Topics =====
client.subscribe(f"direct.{BOT_ID}", handle_direct_message)
client.subscribe("announcements", handle_global_message)

# ===== Announce Bot Online =====
client.publish("announcements", {
    "type": "bot_online",
    "bot_id": BOT_ID,
    "name": BOT_NAME,
    "capabilities": ["chat"],
    "description": "A friendly chatbot that loves to talk"
})

# ===== Keep the Bot Running =====
print(f"{BOT_NAME} is running! Press Ctrl+C to stop.")
try:
    # This blocks until the program is interrupted
    client.join()
except KeyboardInterrupt:
    print("Shutting down...")
    client.publish("announcements", {
        "type": "bot_offline",
        "bot_id": BOT_ID,
        "name": BOT_NAME
    })
    client.disconnect()
