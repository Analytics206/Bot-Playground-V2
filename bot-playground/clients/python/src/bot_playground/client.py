import json
import time
import uuid
import websocket
import threading
from typing import Callable, Dict, Any

class BotPlaygroundClient:
    def __init__(self, bot_id, gateway_url="ws://localhost:8080", debug_mode=False, heartbeat_logging=False):
        self.bot_id = bot_id
        self.gateway_url = gateway_url
        self.ws = None
        self.connected = False
        self.should_reconnect = True
        self.reconnect_delay = 2  # Start with 2 seconds
        self.heartbeat_thread = None
        self.debug_mode = debug_mode
        self.heartbeat_logging = heartbeat_logging
        
        # Event handlers
        self.on_connect_handler = None
        self.on_disconnect_handler = None
        self.on_message_handler = None
        self.on_system_message_handler = None
        
        # Session info
        self.session_id = None
        self.access_token = None
        self.refresh_token = None
        
    def on_connect(self, callback):
        """Register a callback for when connection is established."""
        self.on_connect_handler = callback
        
    def on_disconnect(self, callback):
        """Register a callback for when connection is closed."""
        self.on_disconnect_handler = callback
        
    def on_message(self, callback):
        """Register a callback for incoming messages from other bots."""
        self.on_message_handler = callback
    
    def on_system_message(self, callback):
        """Register a callback for system notifications."""
        self.on_system_message_handler = callback
        
    def connect(self):
        """Establish connection to the Bot Playground gateway service."""
        def on_open(ws):
            if self.debug_mode:
                print("WebSocket connection opened")
            # Send authentication message
            auth_message = {
                "type": "AUTHENTICATE",
                "bot_id": self.bot_id,
                "timestamp": int(time.time() * 1000)
            }
            ws.send(json.dumps(auth_message))
            if self.debug_mode:
                print(f"Sent authentication message for {self.bot_id}")
            
        def on_message(ws, message):
            if self.debug_mode:
                print(f"RAW MESSAGE RECEIVED: {message}")
            try:
                data = json.loads(message)
                message_type = data.get("type", "")
                
                # Skip logging heartbeat messages unless enabled
                if message_type == "HEARTBEAT_ACK" and not self.heartbeat_logging:
                    return
                    
                if self.debug_mode:
                    print(f"Processed message type: {message_type}")
                
                # Handle authentication success
                if message_type == "AUTH_SUCCESS":
                    if self.debug_mode:
                        print("Authentication successful!")
                    self.connected = True
                    self.session_id = data.get("session_id")
                    self.access_token = data.get("access_token")
                    self.refresh_token = data.get("refresh_token")
                    
                    # Reset reconnect delay on successful connection
                    self.reconnect_delay = 2
                    
                    # Start sending heartbeats
                    self.start_heartbeat()
                    
                    if self.on_connect_handler:
                        self.on_connect_handler()
                
                # Handle error messages
                elif message_type == "ERROR":
                    error_msg = data.get("message", "Unknown error")
                    print(f"Server error: {error_msg}")
                        
                # Route message based on type
                elif message_type == "SYSTEM_MESSAGE" and self.on_system_message_handler:
                    if self.debug_mode:
                        print("Routing to system message handler")
                    self.on_system_message_handler(data)
                elif message_type == "MESSAGE" and self.on_message_handler:
                    if self.debug_mode:
                        print("Routing to message handler")
                    self.on_message_handler(data)
                elif message_type == "HEARTBEAT_ACK":
                    if self.heartbeat_logging:
                        print("Heartbeat acknowledged")
                elif self.debug_mode:
                    print(f"Received unhandled message: {data}")
                    
            except json.JSONDecodeError:
                print(f"Error parsing message: {message}")
            except Exception as e:
                print(f"Error processing message: {e}")
                
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
            # Add specific troubleshooting for common errors
            if "connection refused" in str(error).lower():
                print("Connection refused - Is the Gateway Service running on the right port?")
            elif "invalid status code" in str(error).lower():
                print("Invalid status code - Gateway Service responded but may not be a WebSocket server")
            
        def on_close(ws, close_status_code, close_msg):
            self.connected = False
            if self.debug_mode:
                print(f"Connection closed: {close_status_code} - {close_msg}")
            
            if self.on_disconnect_handler:
                self.on_disconnect_handler()
                
            # Implement reconnection logic
            if self.should_reconnect:
                if self.debug_mode:
                    print(f"Reconnecting in {self.reconnect_delay} seconds...")
                time.sleep(self.reconnect_delay)
                # Increase delay for next reconnection (max 30 seconds)
                self.reconnect_delay = min(30, self.reconnect_delay * 1.5)
                self._reconnect()
        
        # Connect to gateway
        self.ws = websocket.WebSocketApp(
            self.gateway_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Run in a separate thread
        self._start_websocket_thread()
        
    def _start_websocket_thread(self):
        """Start the WebSocket connection in a separate thread."""
        try:
            # Enable websocket trace if in debug mode
            if self.debug_mode:
                websocket.enableTrace(True)
                
            # Run in a separate thread with ping_interval and ping_timeout
            wst = threading.Thread(target=lambda: self.ws.run_forever(
                ping_interval=30,  # Send a ping every 30 seconds
                ping_timeout=10    # Wait 10 seconds for pong response
            ))
            wst.daemon = True
            wst.start()
            if self.debug_mode:
                print("WebSocket thread started")
        except Exception as e:
            print(f"Error starting WebSocket thread: {e}")
        
    def _reconnect(self):
        """Reconnect to the gateway after connection loss."""
        if self.debug_mode:
            print("Attempting to reconnect...")
        # Create a new connection with the same handlers
        old_ws = self.ws
        self.ws = websocket.WebSocketApp(
            self.gateway_url,
            on_open=old_ws.on_open,
            on_message=old_ws.on_message,
            on_error=old_ws.on_error,
            on_close=old_ws.on_close
        )
        self._start_websocket_thread()
        
    def start_heartbeat(self):
        """Start sending heartbeats to keep the connection alive."""
        def heartbeat_loop():
            while True:
                try:
                    # Only send heartbeats if connected
                    if self.connected and self.ws and self.ws.sock and self.ws.sock.connected:
                        payload = {
                            "type": "HEARTBEAT",
                            "bot_id": self.bot_id,
                            "session_id": self.session_id,
                            "timestamp": int(time.time() * 1000)
                        }
                        if self.heartbeat_logging:
                            print("Sending heartbeat...")
                        self.ws.send(json.dumps(payload))
                    
                    # Sleep for heartbeat interval (20 seconds)
                    time.sleep(20)
                except Exception as e:
                    if self.debug_mode:
                        print(f"Error in heartbeat loop: {e}")
                    time.sleep(5)
        
        # Stop existing heartbeat thread if any
        if self.heartbeat_thread and self.heartbeat_thread.is_alive():
            self.heartbeat_thread = None
            
        # Start new heartbeat thread
        self.heartbeat_thread = threading.Thread(target=heartbeat_loop)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        if self.debug_mode:
            print("Heartbeat thread started")
        
    def send_message(self, to, content):
        """
        Send a message to another bot.
        
        Args:
            to: ID of the recipient bot
            content: Message content
        """
        if not self.connected:
            print("Not connected")
            return
            
        message = {
            "type": "MESSAGE",
            "id": str(uuid.uuid4()),
            "from": self.bot_id,
            "to": to,
            "content": content,
            "timestamp": int(time.time() * 1000)
        }
        
        # Add access token if available
        if self.access_token:
            message["access_token"] = self.access_token
            
        try:
            self.ws.send(json.dumps(message))
            if self.debug_mode:
                print(f"Message sent to {to}: {content}")
        except Exception as e:
            print(f"Error sending message: {e}")
            self.connected = False
        
    def disconnect(self):
        """Disconnect from the Bot Playground gateway service."""
        self.should_reconnect = False
        if self.ws:
            try:
                self.ws.close()
                if self.debug_mode:
                    print("Disconnected from gateway")
            except Exception as e:
                if self.debug_mode:
                    print(f"Error during disconnect: {e}")