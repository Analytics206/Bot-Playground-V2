#!/usr/bin/env python3
"""
Bot Playground SDK - Minimal Template
------------------------------------
This is a minimal template to help you get started with the Bot Playground SDK.
Just copy this file, rename it, fill in your bot ID, and run it!

Requirements:
- bot_playground_sdk package (install with: pip install bot-playground-sdk)
- faiss-cpu (install with: conda install -c conda-forge faiss-cpu)
"""

from bot_playground_sdk import BotPlaygroundClient

# ----- STEP 1: Initialize the client -----
# Replace "my-bot" with your bot's unique ID
client = BotPlaygroundClient(bot_id="my-bot", auto_vectorize=True)

# ----- STEP 2: Define a message handler -----
def handle_message(topic, message):
    """Called whenever your bot receives a message."""
    print(f"Received message on {topic}: {message}")
    
    # If this is a direct message to our bot, respond
    if topic.startswith("direct.") and message.get('type') == 'chat':
        sender_id = message.get('from_bot_id')
        content = message.get('content', '')
        
        # ----- STEP 3: Use vector memory (optional) -----
        # Store the incoming message in vector memory
        client.store_vector(content, metadata={"from": sender_id})
        
        # Search for similar messages we've seen before
        similar_results = client.search_vectors(content, limit=1)
        
        if similar_results and similar_results[0]['score'] > 0.8:
            # We found something similar in memory
            similar_text = similar_results[0]['metadata'].get('text', '')
            response_text = f"That reminds me of: '{similar_text}'"
        else:
            # Generic response
            response_text = f"Hello! You said: '{content}'"
        
        # ----- STEP 4: Send a response -----
        response = {
            "type": "chat",
            "content": response_text,
            "in_reply_to": message.get('id'),
            "from_bot_id": "my-bot"
        }
        
        # Send the response
        client.publish(f"direct.{sender_id}", response)

# ----- STEP 5: Subscribe to topics -----
# Listen for direct messages to your bot
client.subscribe(f"direct.my-bot", handle_message)

# Listen for global announcements
client.subscribe("global", handle_message)

# ----- STEP 6: Announce that your bot is online -----
client.publish("announcements", {
    "type": "bot_online",
    "bot_id": "my-bot",
    "capabilities": ["chat", "memory"]
})

# ----- STEP 7: Keep the bot running -----
print("Bot is running! Press Ctrl+C to stop.")
try:
    # This blocks until the program is interrupted
    client.join()
except KeyboardInterrupt:
    print("Shutting down...")
    
    # Announce that the bot is going offline
    client.publish("announcements", {
        "type": "bot_offline",
        "bot_id": "my-bot"
    })
    
    # Disconnect from the platform
    client.disconnect()
