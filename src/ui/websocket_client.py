#!/usr/bin/env python
"""
WebSocket Client for Real-time Alerts and Notifications

Handles real-time communication between the dashboard and the backend services.
"""

import asyncio
import websockets
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import threading
import queue
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class WebSocketClient:
    """WebSocket client for real-time alerts and data updates."""
    
    def __init__(self, server_url: str = "ws://localhost:8765"):
        """
        Initialize WebSocket client.
        
        Args:
            server_url: WebSocket server URL
        """
        self.server_url = server_url
        self.websocket = None
        self.is_connected = False
        self.alerts_queue = queue.Queue()
        self.callbacks = {}
        self.background_thread = None
        self.should_stop = False
        
        # Alert storage
        self.recent_alerts = []
        self.max_alerts = 100
        
        # Start background connection
        self.start_background_connection()
    
    def start_background_connection(self):
        """Start background thread for WebSocket connection."""
        if self.background_thread is None or not self.background_thread.is_alive():
            self.should_stop = False
            self.background_thread = threading.Thread(target=self._run_background_loop)
            self.background_thread.daemon = True
            self.background_thread.start()
    
    def _run_background_loop(self):
        """Run the background event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._maintain_connection())
        except Exception as e:
            logger.error(f"WebSocket background loop error: {e}")
        finally:
            loop.close()
    
    async def _maintain_connection(self):
        """Maintain WebSocket connection with auto-reconnect."""
        while not self.should_stop:
            try:
                await self._connect_and_listen()
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
                self.is_connected = False
                
                # Wait before reconnecting
                await asyncio.sleep(5)
    
    async def _connect_and_listen(self):
        """Connect to WebSocket server and listen for messages."""
        try:
            async with websockets.connect(self.server_url) as websocket:
                self.websocket = websocket
                self.is_connected = True
                logger.info(f"Connected to WebSocket server: {self.server_url}")
                
                # Send authentication if needed
                await self._authenticate()
                
                # Listen for messages
                async for message in websocket:
                    await self._handle_message(message)
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.is_connected = False
            raise
    
    async def _authenticate(self):
        """Send authentication message to server."""
        auth_message = {
            "type": "auth",
            "token": "dashboard_client",  # In production, use proper JWT
            "timestamp": datetime.now().isoformat()
        }
        
        if self.websocket:
            await self.websocket.send(json.dumps(auth_message))
    
    async def _handle_message(self, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "alert":
                self._handle_alert(data)
            elif message_type == "market_update":
                self._handle_market_update(data)
            elif message_type == "prediction_update":
                self._handle_prediction_update(data)
            elif message_type == "system_status":
                self._handle_system_status(data)
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
    
    def _handle_alert(self, data: Dict[str, Any]):
        """Handle alert message."""
        alert = {
            "id": data.get("id", ""),
            "type": data.get("alert_type", "General"),
            "message": data.get("message", ""),
            "priority": data.get("priority", "Medium"),
            "timestamp": data.get("timestamp", datetime.now().isoformat()),
            "data": data.get("data", {})
        }
        
        # Add to recent alerts
        self.recent_alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.recent_alerts) > self.max_alerts:
            self.recent_alerts = self.recent_alerts[-self.max_alerts:]
        
        # Add to queue for processing
        self.alerts_queue.put(alert)
        
        # Call registered callbacks
        if "alert" in self.callbacks:
            for callback in self.callbacks["alert"]:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    def _handle_market_update(self, data: Dict[str, Any]):
        """Handle market update message."""
        if "market_update" in self.callbacks:
            for callback in self.callbacks["market_update"]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in market update callback: {e}")
    
    def _handle_prediction_update(self, data: Dict[str, Any]):
        """Handle prediction update message."""
        if "prediction_update" in self.callbacks:
            for callback in self.callbacks["prediction_update"]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in prediction update callback: {e}")
    
    def _handle_system_status(self, data: Dict[str, Any]):
        """Handle system status message."""
        if "system_status" in self.callbacks:
            for callback in self.callbacks["system_status"]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in system status callback: {e}")
    
    def register_callback(self, message_type: str, callback: Callable):
        """
        Register callback for specific message type.
        
        Args:
            message_type: Type of message (alert, market_update, etc.)
            callback: Callback function
        """
        if message_type not in self.callbacks:
            self.callbacks[message_type] = []
        
        self.callbacks[message_type].append(callback)
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent alerts.
        
        Args:
            limit: Maximum number of alerts to return
            
        Returns:
            List of recent alerts
        """
        return self.recent_alerts[-limit:] if self.recent_alerts else []
    
    def get_alerts_by_type(self, alert_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get alerts by type.
        
        Args:
            alert_type: Type of alert to filter
            limit: Maximum number of alerts to return
            
        Returns:
            List of filtered alerts
        """
        filtered_alerts = [
            alert for alert in self.recent_alerts 
            if alert.get("type", "").lower() == alert_type.lower()
        ]
        
        return filtered_alerts[-limit:] if filtered_alerts else []
    
    def send_message(self, message: Dict[str, Any]):
        """
        Send message to WebSocket server.
        
        Args:
            message: Message to send
        """
        if self.is_connected and self.websocket:
            asyncio.create_task(self._send_message_async(message))
    
    async def _send_message_async(self, message: Dict[str, Any]):
        """Send message asynchronously."""
        try:
            if self.websocket:
                await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    def stop(self):
        """Stop WebSocket client."""
        self.should_stop = True
        self.is_connected = False
        
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=5)
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status information."""
        return {
            "is_connected": self.is_connected,
            "server_url": self.server_url,
            "alerts_count": len(self.recent_alerts),
            "callbacks_registered": {
                msg_type: len(callbacks) 
                for msg_type, callbacks in self.callbacks.items()
            }
        }

# Mock WebSocket client for development/testing
class MockWebSocketClient(WebSocketClient):
    """Mock WebSocket client for testing without actual"""