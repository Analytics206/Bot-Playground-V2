#!/usr/bin/env python3
import sys
import os
import time
import random
import json

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from bot_playground.client import BotPlaygroundClient

# Bot configuration
BOT_ID = sys.argv[1] if len(sys.argv) > 1 else "enhanced-bot"
BOT_NAME = "EnhancedBot"

# Configuration options
RESPONSE_DELAY = 3.0  # Seconds to wait before responding to a message
DEBUG_MODE = False    # Set to True for verbose logging
HEARTBEAT_LOGGING = False  # Set to False to hide heartbeat messages

# Sample responses
RESPONSES = {
    "greeting": [
        "Hello there! How are you today?",
        "Hi! Nice to meet you.",
        "Let's chat about Friedrich Nietzsche.",
        "Let's chat about Friedrich Nietzsche."
    ],
    "default": [
        "That's interesting!",
        "Tell me more about Friedrich Nietzsche.",
        "I see. What else is on your mind?",
        "Fascinating. I'd like to hear more.",
        "I'm still learning, but that sounds important."
    ]
}

# Initialize client
print(f"Initializing {BOT_NAME} ({BOT_ID})...")
client = BotPlaygroundClient(bot_id=BOT_ID, gateway_url="ws://localhost:8080", debug_mode=DEBUG_MODE, heartbeat_logging=HEARTBEAT_LOGGING)

# Keep track of conversations
conversations = {}

# Message handler
def on_message(data):
    print("\n" + "="*50)
    print(f"📩 MESSAGE RECEIVED from {data.get('from')}:")
    print(f"   \"{data.get('content')}\"")
    print("="*50)
    
    sender_id = data.get("from")
    content = data.get("content", "").lower()
    
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
    
    # Determine response type
    if any(word in content for word in ["hello", "hi", "hey", "greetings"]):
        response_type = "greeting"
    else:
        response_type = "default"
    
    # Select response
    response_text = random.choice(RESPONSES[response_type])
    
    # Add personalization
    if len(conversations[sender_id]["history"]) > 3:
        response_text += " We've been chatting for a while now!"
    
    # Deliberate delay before responding
    print(f"Thinking for {RESPONSE_DELAY} seconds before responding...")
    time.sleep(RESPONSE_DELAY)
    
    # Send response
    client.send_message(to=sender_id, content=response_text)
    print(f"📤 Responded to {sender_id}: \"{response_text}\"")
    
    # Save to history
    conversations[sender_id]["history"].append({
        "role": "me",
        "content": response_text,
        "time": time.time()
    })

# System message handler
def on_system_message(data):
    print("\n" + "-"*50)
    print(f"🔔 SYSTEM MESSAGE: {data.get('content')}")
    print("-"*50)
    
    # Respond to bot announcements
    content = data.get("content", "")
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
            welcome_delay = 5.0
            print(f"Waiting {welcome_delay} seconds before welcoming new bot...")
            time.sleep(welcome_delay)
            welcome_msg = f"Hi {other_bot_id}! Welcome to the playground. I'm {BOT_NAME}!"
            client.send_message(to=other_bot_id, content=welcome_msg)
            print(f"📤 Sent welcome message to {other_bot_id}: \"{welcome_msg}\"")

# Connection success handler
def on_connect():
    print(f"🔌 Connected to gateway as {BOT_ID}!")

# Connection closed handler
def on_disconnect():
    print("🔌 Disconnected from gateway")

# Set handlers
client.on_connect(on_connect)
client.on_disconnect(on_disconnect)
client.on_message(on_message)
client.on_system_message(on_system_message)

# Connect to gateway
print("Connecting to gateway...")
print(f"Connecting to gateway at ws://localhost:8080...")
client.connect()

# Wait for connection attempt
time.sleep(2)
print(f"Connection status: {'✅ Connected' if client.connected else '❌ Not connected'}")

if not client.connected:
    print("Connection failed. Make sure the Gateway Service is running and accessible.")
    print("Try checking: sudo docker-compose ps")

# Main loop
try:
    print(f"\n{BOT_NAME} is running! Press Ctrl+C to stop.")
    print("\nCommands:")
    print("  auth              - Send authentication message")
    print("  send <bot> <msg>  - Send a message to another bot")
    print("  list              - List active conversations")
    print("  delay <seconds>   - Set response delay (current: {:.1f}s)".format(RESPONSE_DELAY))
    print("  debug [on|off]    - Toggle debug mode")
    print("  exit              - Quit the program")
    
    while True:
        command = input("\n> ").strip()
        
        if command.lower() == "exit":
            break
            
        elif command.lower() == "auth":
            # Send explicit authentication message
            auth_message = {
                "type": "AUTHENTICATE",
                "bot_id": BOT_ID,
                "timestamp": int(time.time() * 1000)
            }
            client.ws.send(json.dumps(auth_message))
            print("Sent authentication message")
        
        elif command.lower() == "list":
            if not conversations:
                print("No active conversations")
            else:
                print("\nActive conversations:")
                for bot_id, convo in conversations.items():
                    msg_count = len(convo["history"])
                    last_time = convo["history"][-1]["time"] if msg_count > 0 else convo["start_time"]
                    elapsed = time.time() - last_time
                    print(f"  {bot_id}: {msg_count} messages, last activity {int(elapsed)}s ago")
        
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
            print("Unknown command. Available commands: auth, send, list, delay, debug, exit")
        
        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nShutting down...")
    client.disconnect()
    print("Goodbye!")