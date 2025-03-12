import asyncio
import logging
import json
import time
import uuid
import jwt
import os
import websockets
from datetime import datetime, timedelta
from typing import Dict, Set, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gateway-service")

class GatewayService:
    """
    Gateway Service that handles WebSocket connections from bot clients
    and routes messages to appropriate internal services.
    """
    
    def __init__(self, mongodb_uri: str = None, redis_uri: str = None):
        """
        Initialize the Gateway Service.
        
        Args:
            mongodb_uri: MongoDB connection URI
            redis_uri: Redis connection URI
        """
        self.mongodb_uri = mongodb_uri
        self.redis_uri = redis_uri
        
        # Active connections
        self.connections = {}
        self.sessions = {}
        
        # JWT configuration
        self.jwt_secret = os.environ.get("JWT_SECRET", "development_secret_key")
        self.access_token_expiry = 3600  # 1 hour
        self.refresh_token_expiry = 86400  # 24 hours
        
    async def handle_connection(self, websocket, path):
        """
        Handle an incoming WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        # Connection is not authenticated yet
        session_id = None
        bot_id = None
        
        try:
            logger.info(f"New connection from {websocket.remote_address}")
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    # Log incoming message
                    logger.info(f"Received {message_type} message")
                    
                    # Handle authentication
                    if message_type == "AUTHENTICATE":
                        session_id, bot_id = await self.handle_authentication(websocket, data)
                        logger.info(f"Bot {bot_id} authenticated with session {session_id}")
                        
                    # All other messages require authentication
                    elif not session_id:
                        logger.warning(f"Unauthenticated message: {message_type}")
                        await self.send_error(websocket, "Not authenticated")
                        continue
                        
                    # Handle different message types
                    elif message_type == "HEARTBEAT":
                        await self.handle_heartbeat(websocket, session_id, data)
                        
                    elif message_type == "MESSAGE":
                        await self.handle_message(websocket, session_id, bot_id, data)
                        
                    elif message_type == "REFRESH_TOKEN":
                        await self.handle_token_refresh(websocket, data)
                        
                    else:
                        logger.warning(f"Unknown message type: {message_type}")
                        await self.send_error(websocket, f"Unknown message type: {message_type}")
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON payload: {message}")
                    await self.send_error(websocket, "Invalid JSON payload")
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    await self.send_error(websocket, f"Internal server error: {str(e)}")
                    
        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"Connection closed normally for session {session_id}")
        except websockets.exceptions.ConnectionClosedError as e:
            logger.warning(f"Connection closed with error for session {session_id}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error handling connection: {str(e)}")
        finally:
            # Clean up connection
            if session_id:
                logger.info(f"Cleaning up session {session_id}")
                await self.handle_disconnect(session_id, bot_id)
            else:
                logger.info("Unauthenticated connection closed")
                
    async def handle_authentication(self, websocket, data):
        """
        Handle bot authentication.
        
        Args:
            websocket: WebSocket connection
            data: Authentication message data
            
        Returns:
            tuple: (session_id, bot_id)
        """
        bot_id = data.get("bot_id")
        api_key = data.get("api_key")
        
        if not bot_id:
            await self.send_error(websocket, "Missing bot_id")
            return None, None
            
        # In production, validate the API key against the database
        # For now, auto-authenticate all bots
        
        # Create a new session
        session_id = str(uuid.uuid4())
        
        # Generate JWT tokens
        access_token, refresh_token = self._generate_tokens(bot_id)
        
        self.connections[session_id] = websocket
        self.sessions[session_id] = {
            "bot_id": bot_id,
            "connected_at": datetime.now().isoformat(),
            "last_heartbeat": datetime.now().timestamp(),
            "access_token": access_token,
            "refresh_token": refresh_token
        }
        
        # Send authentication success
        await websocket.send(json.dumps({
            "type": "AUTH_SUCCESS",
            "session_id": session_id,
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expiry": int(time.time()) + self.access_token_expiry,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }))
        
        logger.info(f"Bot {bot_id} authenticated with session {session_id}")
        
        # Send system message to all bots that a new bot has joined
        await self.broadcast_system_message(f"Bot {bot_id} has joined the network")
        
        return session_id, bot_id
        
    def _generate_tokens(self, bot_id):
        """Generate JWT access and refresh tokens"""
        # Current timestamp
        now = int(time.time())
        
        # Access token - short lived
        access_payload = {
            "bot_id": bot_id,
            "type": "access",
            "iat": now,
            "exp": now + self.access_token_expiry
        }
        
        # Refresh token - longer lived
        refresh_payload = {
            "bot_id": bot_id,
            "type": "refresh",
            "iat": now,
            "exp": now + self.refresh_token_expiry,
            "jti": str(uuid.uuid4())  # Unique token ID
        }
        
        access_token = jwt.encode(access_payload, self.jwt_secret, algorithm="HS256")
        refresh_token = jwt.encode(refresh_payload, self.jwt_secret, algorithm="HS256")
        
        return access_token, refresh_token
    
    async def handle_token_refresh(self, websocket, data):
        """Handle token refresh requests"""
        bot_id = data.get("bot_id")
        refresh_token = data.get("refresh_token")
        
        if not bot_id or not refresh_token:
            await self.send_error(websocket, "Missing required fields")
            return
            
        try:
            # Verify the refresh token
            payload = jwt.decode(refresh_token, self.jwt_secret, algorithms=["HS256"])
            
            # Check token type and expiration
            if payload.get("type") != "refresh" or payload.get("bot_id") != bot_id:
                raise jwt.InvalidTokenError("Invalid token")
                
            # In production, check if token has been revoked
            
            # Generate new tokens
            access_token, refresh_token = self._generate_tokens(bot_id)
            
            # Send token refresh success
            await websocket.send(json.dumps({
                "type": "TOKEN_REFRESH_SUCCESS",
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expiry": int(time.time()) + self.access_token_expiry,
                "timestamp": int(datetime.now().timestamp() * 1000)
            }))
            
            logger.info(f"Tokens refreshed for bot {bot_id}")
            
        except jwt.ExpiredSignatureError:
            await self.send_error(websocket, "Refresh token expired")
        except jwt.InvalidTokenError as e:
            await self.send_error(websocket, f"Invalid token: {str(e)}")
    
    async def handle_heartbeat(self, websocket, session_id, data):
        """
        Handle heartbeat message.
        
        Args:
            websocket: WebSocket connection
            session_id: Session ID
            data: Heartbeat message data
        """
        logger.info(f"Received heartbeat from session {session_id}")
        
        if session_id in self.sessions:
            # Update last heartbeat timestamp
            self.sessions[session_id]["last_heartbeat"] = datetime.now().timestamp()
            
            # Send heartbeat acknowledgement
            await websocket.send(json.dumps({
                "type": "HEARTBEAT_ACK",
                "timestamp": int(datetime.now().timestamp() * 1000)
            }))
            logger.info(f"Sent heartbeat acknowledgement to session {session_id}")
    
    async def handle_message(self, websocket, session_id, bot_id, data):
        """
        Handle message between bots.
        
        Args:
            websocket: WebSocket connection
            session_id: Session ID
            bot_id: Bot ID
            data: Message data
        """
        to_bot_id = data.get("to")
        content = data.get("content")
        
        if not to_bot_id or not content:
            await self.send_error(websocket, "Missing required fields")
            return
            
        logger.info(f"Routing message from {bot_id} to {to_bot_id}: {content}")
        
        # Find the recipient's session
        recipient_session_id = None
        for s_id, session in self.sessions.items():
            if session["bot_id"] == to_bot_id:
                recipient_session_id = s_id
                break
                
        # If recipient is online, deliver the message directly
        if recipient_session_id and recipient_session_id in self.connections:
            try:
                logger.info(f"Recipient {to_bot_id} found with session {recipient_session_id}")
                await self.connections[recipient_session_id].send(json.dumps(data))
                logger.info(f"Message delivered from {bot_id} to {to_bot_id}")
            except Exception as e:
                logger.error(f"Failed to deliver message to {to_bot_id}: {e}")
        else:
            logger.info(f"Bot {to_bot_id} is offline or not found, message not delivered")
            await self.send_error(websocket, f"Bot {to_bot_id} is offline or not found")
        
    async def handle_disconnect(self, session_id, bot_id):
        """
        Handle bot disconnection.
        
        Args:
            session_id: Session ID
            bot_id: Bot ID
        """
        # Remove from active connections and sessions
        self.connections.pop(session_id, None)
        self.sessions.pop(session_id, None)
        
        logger.info(f"Bot {bot_id} disconnected, session {session_id} closed")
        
        # Send system message to all bots that a bot has left
        await self.broadcast_system_message(f"Bot {bot_id} has left the network")
        
    async def broadcast_system_message(self, content):
        """
        Broadcast a system message to all connected bots.
        
        Args:
            content: Message content
        """
        system_message = {
            "type": "SYSTEM_MESSAGE",
            "id": str(uuid.uuid4()),
            "content": content,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        
        # Send to all connected bots
        for websocket in self.connections.values():
            try:
                await websocket.send(json.dumps(system_message))
            except:
                # Ignore errors, they'll be handled when the bot tries to send a message
                pass
                
        logger.info(f"System message broadcast: {content}")
        
    async def send_error(self, websocket, error_message):
        """
        Send an error message to a client.
        
        Args:
            websocket: WebSocket connection
            error_message: Error message
        """
        await websocket.send(json.dumps({
            "type": "ERROR",
            "message": error_message,
            "timestamp": int(datetime.now().timestamp() * 1000)
        }))
        
        logger.warning(f"Error sent to client: {error_message}")