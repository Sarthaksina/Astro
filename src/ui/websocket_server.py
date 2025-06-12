#!/usr/bin/env python
"""
WebSocket Server for Real-time Alerts and Notifications

Provides real-time communication between the backend services and dashboard clients.
"""

import asyncio
import websockets
import json
# import logging # Removed
from datetime import datetime, timedelta
from typing import Dict, Set, Any, Optional
import threading
from src.utils.logger import get_logger # Added
import queue
import signal
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.astro_engine.planetary_positions import PlanetaryCalculator
from src.trading.strategy_framework import VedicAstrologyStrategy
from .constants import DEFAULT_HOST, DEFAULT_PORT # MAX_CONNECTIONS is not used here

logger = get_logger(__name__) # Changed

class WebSocketServer:
    """WebSocket server for real-time dashboard communication."""
    
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        """
        Initialize WebSocket server.
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.authenticated_clients: Dict[websockets.WebSocketServerProtocol, str] = {}
        self.alert_queue = queue.Queue()
        self.is_running = False
        
        # Initialize components
        self.planetary_calculator = PlanetaryCalculator()
        
        # Background tasks
        self.background_tasks = []
        
        # Alert types and priorities
        self.alert_types = {
            "market": "Market Alert",
            "astrological": "Astrological Alert", 
            "strategy": "Strategy Alert",
            "system": "System Alert",
            "prediction": "Prediction Alert"
        }
        
        self.priority_levels = ["Low", "Medium", "High", "Critical"]
    
    async def register_client(self, websocket: websockets.WebSocketServerProtocol):
        """Register new client connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        
        # Send welcome message
        welcome_message = {
            "type": "system_status",
            "message": "Connected to Cosmic Market Oracle",
            "timestamp": datetime.now().isoformat(),
            "server_info": {
                "version": "1.0.0",
                "capabilities": ["alerts", "market_updates", "predictions"]
            }
        }
        
        await self.send_to_client(websocket, welcome_message)
    
    async def unregister_client(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister client connection."""
        self.clients.discard(websocket)
        self.authenticated_clients.pop(websocket, None)
        logger.info(f"Client disconnected: {websocket.remote_address}")
    
    async def authenticate_client(self, websocket: websockets.WebSocketServerProtocol, auth_data: Dict[str, Any]) -> bool:
        """
        Authenticate client connection.
        
        Args:
            websocket: Client websocket
            auth_data: Authentication data
            
        Returns:
            True if authentication successful
        """
        # Simple token-based authentication (enhance for production)
        token = auth_data.get("token", "")
        
        if token == "dashboard_client":  # In production, validate JWT properly
            client_id = f"client_{len(self.authenticated_clients) + 1}"
            self.authenticated_clients[websocket] = client_id
            
            # Send authentication success
            auth_response = {
                "type": "auth_response",
                "status": "success",
                "client_id": client_id,
                "timestamp": datetime.now().isoformat()
            }
            
            await self.send_to_client(websocket, auth_response)
            logger.info(f"Client authenticated: {client_id}")
            return True
        
        # Send authentication failure
        auth_response = {
            "type": "auth_response", 
            "status": "failed",
            "message": "Invalid authentication token",
            "timestamp": datetime.now().isoformat()
        }
        
        await self.send_to_client(websocket, auth_response)
        return False
    
    async def handle_client_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Handle incoming client message."""
        try:
            data = json.loads(message)
            message_type = data.get("type", "")
            
            if message_type == "auth":
                await self.authenticate_client(websocket, data)
            
            elif message_type == "subscribe":
                await self.handle_subscription(websocket, data)
            
            elif message_type == "unsubscribe":
                await self.handle_unsubscription(websocket, data)
            
            elif message_type == "request_data":
                await self.handle_data_request(websocket, data)
            
            elif message_type == "ping":
                await self.send_to_client(websocket, {"type": "pong", "timestamp": datetime.now().isoformat()})
            
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON from client: {message}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    async def handle_subscription(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle client subscription request."""
        subscription_type = data.get("subscription_type", "")
        
        # Store subscription preferences (implement as needed)
        response = {
            "type": "subscription_response",
            "subscription_type": subscription_type,
            "status": "subscribed",
            "timestamp": datetime.now().isoformat()
        }
        
        await self.send_to_client(websocket, response)
    
    async def handle_unsubscription(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle client unsubscription request."""
        subscription_type = data.get("subscription_type", "")
        
        response = {
            "type": "subscription_response",
            "subscription_type": subscription_type, 
            "status": "unsubscribed",
            "timestamp": datetime.now().isoformat()
        }
        
        await self.send_to_client(websocket, response)
    
    async def handle_data_request(self, websocket: websockets.WebSocketServerProtocol, data: Dict[str, Any]):
        """Handle client data request."""
        request_type = data.get("request_type", "")
        
        if request_type == "planetary_positions":
            await self.send_planetary_positions(websocket)
        elif request_type == "market_status":
            await self.send_market_status(websocket)
        elif request_type == "recent_alerts":
            await self.send_recent_alerts(websocket)
        else:
            error_response = {
                "type": "error",
                "message": f"Unknown request type: {request_type}",
                "timestamp": datetime.now().isoformat()
            }
            await self.send_to_client(websocket, error_response)
    
    async def send_planetary_positions(self, websocket: websockets.WebSocketServerProtocol):
        """Send current planetary positions to client."""
        try:
            current_date = datetime.now()
            planets_data = self.planetary_calculator.get_all_planets(current_date)
            
            response = {
                "type": "planetary_positions",
                "data": planets_data,
                "timestamp": current_date.isoformat()
            }
            
            await self.send_to_client(websocket, response)
            
        except Exception as e:
            logger.error(f"Error sending planetary positions: {e}")
    
    async def send_market_status(self, websocket: websockets.WebSocketServerProtocol):
        """Send market status to client."""
        # Mock market data (replace with real data source)
        market_data = {
            "SPY": {"price": 456.78, "change": 2.34, "change_percent": 0.51},
            "QQQ": {"price": 378.90, "change": -1.23, "change_percent": -0.32},
            "IWM": {"price": 198.45, "change": 0.67, "change_percent": 0.34}
        }
        
        response = {
            "type": "market_update",
            "data": market_data,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.send_to_client(websocket, response)
    
    async def send_recent_alerts(self, websocket: websockets.WebSocketServerProtocol):
        """Send recent alerts to client."""
        # Mock recent alerts (replace with real alert system)
        recent_alerts = [
            {
                "id": "alert_001",
                "type": "astrological",
                "message": "Jupiter entering Taurus - Bullish signal",
                "priority": "High",
                "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat()
            },
            {
                "id": "alert_002", 
                "type": "market",
                "message": "S&P 500 approaching resistance at 4580",
                "priority": "Medium",
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat()
            }
        ]
        
        response = {
            "type": "recent_alerts",
            "data": recent_alerts,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.send_to_client(websocket, response)
    
    async def send_to_client(self, websocket: websockets.WebSocketServerProtocol, message: Dict[str, Any]):
        """Send message to specific client."""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            await self.unregister_client(websocket)
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
    
    async def broadcast_message(self, message: Dict[str, Any], authenticated_only: bool = True):
        """Broadcast message to all connected clients."""
        if not self.clients:
            return
        
        target_clients = self.authenticated_clients.keys() if authenticated_only else self.clients
        
        # Send to all clients concurrently
        tasks = []
        for client in list(target_clients):
            if client in self.clients:  # Check if still connected
                tasks.append(self.send_to_client(client, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def send_alert(self, alert_type: str, message: str, priority: str = "Medium", data: Dict[str, Any] = None):
        """
        Send alert to all connected clients.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            priority: Alert priority
            data: Additional alert data
        """
        alert = {
            "type": "alert",
            "alert_type": alert_type,
            "message": message,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
            "data": data or {}
        }
        
        await self.broadcast_message(alert)
        logger.info(f"Alert sent: {alert_type} - {message}")
    
    async def client_handler(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle individual client connection."""
        await self.register_client(websocket)
        
        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception as e:
            logger.error(f"Error in client handler: {e}")
        finally:
            await self.unregister_client(websocket)
    
    async def periodic_updates(self):
        """Send periodic updates to clients."""
        while self.is_running:
            try:
                # Send market updates every 30 seconds
                if self.authenticated_clients:
                    market_update = {
                        "type": "market_update",
                        "timestamp": datetime.now().isoformat(),
                        "data": {
                            "SPY": {"price": 456.78, "change": 2.34},
                            "status": "market_open"
                        }
                    }
                    await self.broadcast_message(market_update)
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in periodic updates: {e}")
                await asyncio.sleep(30)
    
    async def astrological_monitor(self):
        """Monitor for significant astrological events."""
        while self.is_running:
            try:
                # Check for significant astrological events
                current_date = datetime.now()
                
                # Mock astrological event detection
                # In production, integrate with actual astrological calculation engine
                if current_date.minute % 15 == 0:  # Every 15 minutes for demo
                    await self.send_alert(
                        "astrological",
                        "Planetary aspect detected: Mars trine Jupiter",
                        "Medium",
                        {"aspect": "trine", "planets": ["Mars", "Jupiter"]}
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in astrological monitor: {e}")
                await asyncio.sleep(60)
    
    async def start_background_tasks(self):
        """Start background monitoring tasks."""
        self.background_tasks = [
            asyncio.create_task(self.periodic_updates()),
            asyncio.create_task(self.astrological_monitor())
        ]
    
    async def stop_background_tasks(self):
        """Stop background monitoring tasks."""
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
    
    def start_server(self):
        """Start the WebSocket server."""
        self.is_running = True
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal")
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start server
        start_server = websockets.serve(
            self.client_handler,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        
        logger.info(f"WebSocket server starting on {self.host}:{self.port}")
        
        # Run server with background tasks
        loop = asyncio.get_event_loop()
        
        try:
            loop.run_until_complete(start_server)
            loop.run_until_complete(self.start_background_tasks())
            loop.run_forever()
        except KeyboardInterrupt:
            logger.info("Server interrupted by user")
        finally:
            self.is_running = False
            loop.run_until_complete(self.stop_background_tasks())
            logger.info("WebSocket server stopped")
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "total_clients": len(self.clients),
            "authenticated_clients": len(self.authenticated_clients),
            "is_running": self.is_running,
            "host": self.host,
            "port": self.port,
            "background_tasks": len(self.background_tasks)
        }

# Server runner script
if __name__ == "__main__":
    import argparse
    # Import constants for main block defaults
    from .constants import DEFAULT_HOST, DEFAULT_PORT

    parser = argparse.ArgumentParser(description="Cosmic Market Oracle WebSocket Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Configure logging
    # logging.basicConfig(...) # Removed
    # The module-level logger (now using get_logger) will be used.
    # If specific configuration for __main__ is needed, setup_logger could be used here,
    # potentially passing args.log_level. For now, relying on get_logger's default setup.
    logger.info(f"Log level set to {args.log_level} (Note: This may not be effective if get_logger doesn't use it)")

    # Create and start server
    server = WebSocketServer(host=args.host, port=args.port)
    server.start_server()