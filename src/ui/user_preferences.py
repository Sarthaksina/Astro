#!/usr/bin/env python
"""
User Preferences Manager for Cosmic Market Oracle Dashboard

Handles user-specific settings, preferences, and customization options.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class UserPreferences:
    """Manages user preferences and settings."""
    
    def __init__(self, preferences_dir: str = "data/preferences"):
        """
        Initialize user preferences manager.
        
        Args:
            preferences_dir: Directory to store user preferences
        """
        self.preferences_dir = Path(preferences_dir)
        self.preferences_dir.mkdir(parents=True, exist_ok=True)
        
        # Default preferences
        self.default_preferences = {
            # General Settings
            "timezone_index": 0,  # UTC
            "language_index": 0,  # English
            "auto_refresh": True,
            "refresh_interval": 60,  # seconds
            
            # Trading Preferences
            "default_symbol_index": 0,  # SPY
            "risk_tolerance_index": 1,  # Moderate
            "default_timeframe_index": 0,  # 1D
            "auto_execute": False,
            "position_size_percent": 10.0,
            "stop_loss_percent": 5.0,
            "take_profit_percent": 15.0,
            
            # Notification Settings
            "email_notifications": True,
            "push_notifications": True,
            "notification_frequency_index": 0,  # Real-time
            "quiet_hours": False,
            "quiet_start_time": "22:00",
            "quiet_end_time": "08:00",
            "alert_sound": True,
            
            # Display Settings
            "theme_index": 0,  # Light
            "chart_style_index": 0,  # Modern
            "show_tooltips": True,
            "compact_view": False,
            "sidebar_collapsed": False,
            "grid_density": "comfortable",
            
            # Dashboard Layout
            "dashboard_layout": "default",
            "visible_widgets": [
                "market_overview",
                "planetary_positions",
                "recent_predictions",
                "alerts",
                "performance_metrics"
            ],
            "widget_positions": {},
            
            # Chart Preferences
            "default_chart_type": "candlestick",
            "show_volume": True,
            "show_indicators": True,
            "default_indicators": ["SMA_20", "SMA_50", "RSI"],
            "chart_timeframe": "1D",
            
            # Astrological Preferences
            "preferred_astrological_system": "vedic",
            "show_planetary_aspects": True,
            "show_transits": True,
            "aspect_orb_tolerance": 5.0,
            "include_minor_aspects": False,
            
            # Alert Preferences
            "alert_types": {
                "market_movement": True,
                "astrological_events": True,
                "strategy_signals": True,
                "system_notifications": False
            },
            "alert_thresholds": {
                "price_change_percent": 2.0,
                "volume_spike_multiplier": 2.0,
                "volatility_threshold": 1.5
            },
            
            # Privacy Settings
            "data_sharing": False,
            "analytics_tracking": True,
            "crash_reporting": True,
            
            # Advanced Settings
            "api_rate_limit": 100,
            "cache_duration": 300,  # seconds
            "debug_mode": False,
            "experimental_features": False
        }
    
    def get_preferences_file(self, username: str) -> Path:
        """Get preferences file path for user."""
        return self.preferences_dir / f"{username}_preferences.json"
    
    def get_preferences(self, username: str) -> Dict[str, Any]:
        """
        Get user preferences.
        
        Args:
            username: Username
            
        Returns:
            User preferences dictionary
        """
        preferences_file = self.get_preferences_file(username)
        
        if preferences_file.exists():
            try:
                with open(preferences_file, 'r') as f:
                    user_prefs = json.load(f)
                
                # Merge with defaults to ensure all keys exist
                merged_prefs = self.default_preferences.copy()
                merged_prefs.update(user_prefs)
                
                return merged_prefs
                
            except Exception as e:
                logger.error(f"Error loading preferences for {username}: {e}")
                return self.default_preferences.copy()
        
        return self.default_preferences.copy()
    
    def save_preferences(self, username: str, preferences: Dict[str, Any]) -> bool:
        """
        Save user preferences.
        
        Args:
            username: Username
            preferences: Preferences dictionary
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            preferences_file = self.get_preferences_file(username)
            
            # Add metadata
            preferences["last_updated"] = datetime.now().isoformat()
            preferences["version"] = "1.0"
            
            with open(preferences_file, 'w') as f:
                json.dump(preferences, f, indent=2)
            
            logger.info(f"Saved preferences for user: {username}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving preferences for {username}: {e}")
            return False
    
    def update_preference(self, username: str, key: str, value: Any) -> bool:
        """
        Update a single preference.
        
        Args:
            username: Username
            key: Preference key
            value: New value
            
        Returns:
            True if updated successfully, False otherwise
        """
        preferences = self.get_preferences(username)
        preferences[key] = value
        return self.save_preferences(username, preferences)
    
    def reset_preferences(self, username: str) -> bool:
        """
        Reset user preferences to defaults.
        
        Args:
            username: Username
            
        Returns:
            True if reset successfully, False otherwise
        """
        return self.save_preferences(username, self.default_preferences.copy())
    
    def export_preferences(self, username: str, export_path: str) -> bool:
        """
        Export user preferences to file.
        
        Args:
            username: Username
            export_path: Path to export file
            
        Returns:
            True if exported successfully, False otherwise
        """
        try:
            preferences = self.get_preferences(username)
            
            export_data = {
                "username": username,
                "exported_at": datetime.now().isoformat(),
                "preferences": preferences
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported preferences for {username} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting preferences for {username}: {e}")
            return False
    
    def import_preferences(self, username: str, import_path: str) -> bool:
        """
        Import user preferences from file.
        
        Args:
            username: Username
            import_path: Path to import file
            
        Returns:
            True if imported successfully, False otherwise
        """
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            if "preferences" in import_data:
                preferences = import_data["preferences"]
                
                # Validate preferences against defaults
                validated_prefs = self._validate_preferences(preferences)
                
                return self.save_preferences(username, validated_prefs)
            
            return False
            
        except Exception as e:
            logger.error(f"Error importing preferences for {username}: {e}")
            return False
    
    def _validate_preferences(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate preferences against defaults.
        
        Args:
            preferences: Preferences to validate
            
        Returns:
            Validated preferences
        """
        validated = self.default_preferences.copy()
        
        for key, value in preferences.items():
            if key in self.default_preferences:
                # Type validation
                expected_type = type(self.default_preferences[key])
                if isinstance(value, expected_type):
                    validated[key] = value
                else:
                    logger.warning(f"Invalid type for preference {key}: expected {expected_type}, got {type(value)}")
        
        return validated
    
    def get_theme_settings(self, username: str) -> Dict[str, Any]:
        """
        Get theme-specific settings.
        
        Args:
            username: Username
            
        Returns:
            Theme settings
        """
        preferences = self.get_preferences(username)
        
        themes = ["Light", "Dark", "Auto"]
        theme_index = preferences.get("theme_index", 0)
        theme_name = themes[theme_index] if theme_index < len(themes) else "Light"
        
        return {
            "theme_name": theme_name,
            "theme_index": theme_index,
            "chart_style": ["Modern", "Classic", "Minimal"][preferences.get("chart_style_index", 0)],
            "show_tooltips": preferences.get("show_tooltips", True),
            "compact_view": preferences.get("compact_view", False),
            "sidebar_collapsed": preferences.get("sidebar_collapsed", False)
        }
    
    def get_trading_settings(self, username: str) -> Dict[str, Any]:
        """
        Get trading-specific settings.
        
        Args:
            username: Username
            
        Returns:
            Trading settings
        """
        preferences = self.get_preferences(username)
        
        symbols = ["SPY", "QQQ", "IWM", "DIA"]
        timeframes = ["1D", "1W", "1M", "3M"]
        risk_levels = ["Conservative", "Moderate", "Aggressive"]
        
        return {
            "default_symbol": symbols[preferences.get("default_symbol_index", 0)],
            "default_timeframe": timeframes[preferences.get("default_timeframe_index", 0)],
            "risk_tolerance": risk_levels[preferences.get("risk_tolerance_index", 1)],
            "auto_execute": preferences.get("auto_execute", False),
            "position_size_percent": preferences.get("position_size_percent", 10.0),
            "stop_loss_percent": preferences.get("stop_loss_percent", 5.0),
            "take_profit_percent": preferences.get("take_profit_percent", 15.0)
        }
    
    def get_notification_settings(self, username: str) -> Dict[str, Any]:
        """
        Get notification settings.
        
        Args:
            username: Username
            
        Returns:
            Notification settings
        """
        preferences = self.get_preferences(username)
        
        frequencies = ["Real-time", "Hourly", "Daily"]
        
        return {
            "email_notifications": preferences.get("email_notifications", True),
            "push_notifications": preferences.get("push_notifications", True),
            "notification_frequency": frequencies[preferences.get("notification_frequency_index", 0)],
            "quiet_hours": preferences.get("quiet_hours", False),
            "quiet_start_time": preferences.get("quiet_start_time", "22:00"),
            "quiet_end_time": preferences.get("quiet_end_time", "08:00"),
            "alert_sound": preferences.get("alert_sound", True),
            "alert_types": preferences.get("alert_types", {}),
            "alert_thresholds": preferences.get("alert_thresholds", {})
        }
    
    def get_dashboard_layout(self, username: str) -> Dict[str, Any]:
        """
        Get dashboard layout settings.
        
        Args:
            username: Username
            
        Returns:
            Dashboard layout settings
        """
        preferences = self.get_preferences(username)
        
        return {
            "layout": preferences.get("dashboard_layout", "default"),
            "visible_widgets": preferences.get("visible_widgets", []),
            "widget_positions": preferences.get("widget_positions", {}),
            "grid_density": preferences.get("grid_density", "comfortable")
        }
    
    def update_dashboard_layout(self, username: str, layout_data: Dict[str, Any]) -> bool:
        """
        Update dashboard layout.
        
        Args:
            username: Username
            layout_data: Layout data
            
        Returns:
            True if updated successfully, False otherwise
        """
        preferences = self.get_preferences(username)
        
        if "visible_widgets" in layout_data:
            preferences["visible_widgets"] = layout_data["visible_widgets"]
        
        if "widget_positions" in layout_data:
            preferences["widget_positions"] = layout_data["widget_positions"]
        
        if "dashboard_layout" in layout_data:
            preferences["dashboard_layout"] = layout_data["dashboard_layout"]
        
        return self.save_preferences(username, preferences)
    
    def get_user_statistics(self) -> Dict[str, Any]:
        """Get statistics about user preferences."""
        preference_files = list(self.preferences_dir.glob("*_preferences.json"))
        
        total_users = len(preference_files)
        theme_usage = {"Light": 0, "Dark": 0, "Auto": 0}
        risk_tolerance = {"Conservative": 0, "Moderate": 0, "Aggressive": 0}
        
        for pref_file in preference_files:
            try:
                with open(pref_file, 'r') as f:
                    prefs = json.load(f)
                
                # Count theme usage
                theme_index = prefs.get("theme_index", 0)
                themes = ["Light", "Dark", "Auto"]
                if 0 <= theme_index < len(themes):
                    theme_usage[themes[theme_index]] += 1
                
                # Count risk tolerance
                risk_index = prefs.get("risk_tolerance_index", 1)
                risk_levels = ["Conservative", "Moderate", "Aggressive"]
                if 0 <= risk_index < len(risk_levels):
                    risk_tolerance[risk_levels[risk_index]] += 1
                    
            except Exception as e:
                logger.error(f"Error reading preferences file {pref_file}: {e}")
        
        return {
            "total_users": total_users,
            "theme_usage": theme_usage,
            "risk_tolerance_distribution": risk_tolerance,
            "preferences_directory": str(self.preferences_dir)
        }