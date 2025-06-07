#!/usr/bin/env python
"""
Cosmic Market Oracle - User Interface Package

This package provides the web-based user interface for the Cosmic Market Oracle,
including dashboard, authentication, real-time alerts, and user preferences.
"""

from src.ui.auth import AuthManager
from src.ui.websocket_client import WebSocketClient, MockWebSocketClient
from src.ui.user_preferences import UserPreferences
from src.ui.feedback_system import FeedbackSystem
from src.ui.websocket_server import WebSocketServer

__all__ = [
    'AuthManager',
    'WebSocketClient', 
    'MockWebSocketClient',
    'UserPreferences',
    'FeedbackSystem',
    'WebSocketServer'
]

__version__ = "1.0.0"
__author__ = "Cosmic Market Oracle Team"
__description__ = "Web-based user interface for astrological market analysis"