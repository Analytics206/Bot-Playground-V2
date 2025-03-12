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
