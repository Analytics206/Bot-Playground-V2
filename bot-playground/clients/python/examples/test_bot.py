#!/usr/bin/env python

import time
import sys
from bot_playground.client import BotPlaygroundClient

def main():
    # Get bot ID from command line or use default
    bot_id = sys.argv[1] if len(sys.argv) > 1 else "TestBot"
    
    print(f"Starting bot: {bot_id}")
    
    # Create client
    client = BotPlaygroundClient(bot_id=bot_id)
    
    # Define event handlers
    def on_connect():
        print("Connected to gateway!")
        
    def on_message(data):
        print(f"Received message: {data}")
        
    # Set handlers
    client.on_connect(on_connect)
    client.on_message(on_message)
    
    # Connect to gateway
    client.connect()
    print(f"Attempting to connect to gateway at {client.gateway_url}")
    time.sleep(1)
    print(f"Connection status: {'Connected' if client.connected else 'Not connected'}")
    # Wait for connection
    time.sleep(1)
    
    try:
        while True:
            # Simple console interface
            msg = input("Enter message (format: recipient message): ")
            if not msg.strip():
                continue
                
            if msg.lower() == "exit":
                break
                
            parts = msg.split(" ", 1)
            if len(parts) < 2:
                print("Invalid format. Use: recipient message")
                continue
                
            recipient, content = parts
            client.send_message(recipient, content)
                
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        client.disconnect()
    
if __name__ == "__main__":
    main()