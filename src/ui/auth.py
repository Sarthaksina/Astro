#!/usr/bin/env python
"""
Authentication Manager for Cosmic Market Oracle Dashboard

Handles user registration, login, session management, and security.
"""

import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import secrets
import redis
from pathlib import Path

class AuthManager:
    """Manages user authentication and sessions."""
    
    def __init__(self, users_file: str = "data/users.json"):
        """
        Initialize the authentication manager.
        
        Args:
            users_file: Path to the users database file
        """
        self.users_file = Path(users_file)
        self.users_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize Redis for session management (fallback to file-based)
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            self.use_redis = True
        except:
            self.use_redis = False
            self.sessions_file = self.users_file.parent / "sessions.json"
        
        # Load users
        self.users = self._load_users()
        
        # Create default admin user if no users exist
        if not self.users:
            self._create_default_admin()
    
    def _load_users(self) -> Dict[str, Any]:
        """Load users from file."""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_users(self):
        """Save users to file."""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def _create_default_admin(self):
        """Create default admin user."""
        admin_password = "cosmic123"  # Change this in production!
        self.register_user("admin", "admin@cosmicoracle.com", admin_password, role="admin")
        print(f"Created default admin user - Username: admin, Password: {admin_password}")
    
    def _hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt."""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # Use PBKDF2 for secure password hashing
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return password_hash.hex(), salt
    
    def register_user(self, username: str, email: str, password: str, role: str = "user") -> bool:
        """
        Register a new user.
        
        Args:
            username: Username
            email: Email address
            password: Password
            role: User role (user, admin)
            
        Returns:
            True if registration successful, False otherwise
        """
        # Check if username already exists
        if username in self.users:
            return False
        
        # Hash password
        password_hash, salt = self._hash_password(password)
        
        # Create user record
        self.users[username] = {
            "email": email,
            "password_hash": password_hash,
            "salt": salt,
            "role": role,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "is_active": True,
            "login_attempts": 0,
            "locked_until": None
        }
        
        # Save users
        self._save_users()
        return True
    
    def authenticate(self, username: str, password: str) -> bool:
        """
        Authenticate user credentials.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            True if authentication successful, False otherwise
        """
        # Check if user exists
        if username not in self.users:
            return False
        
        user = self.users[username]
        
        # Check if account is locked
        if user.get("locked_until"):
            locked_until = datetime.fromisoformat(user["locked_until"])
            if datetime.now() < locked_until:
                return False
            else:
                # Unlock account
                user["locked_until"] = None
                user["login_attempts"] = 0
        
        # Check if account is active
        if not user.get("is_active", True):
            return False
        
        # Verify password
        password_hash, _ = self._hash_password(password, user["salt"])
        
        if password_hash == user["password_hash"]:
            # Successful login
            user["last_login"] = datetime.now().isoformat()
            user["login_attempts"] = 0
            self._save_users()
            return True
        else:
            # Failed login
            user["login_attempts"] = user.get("login_attempts", 0) + 1
            
            # Lock account after 5 failed attempts
            if user["login_attempts"] >= 5:
                user["locked_until"] = (datetime.now() + timedelta(minutes=30)).isoformat()
            
            self._save_users()
            return False
    
    def create_session(self, username: str) -> str:
        """
        Create a new session for user.
        
        Args:
            username: Username
            
        Returns:
            Session token
        """
        session_token = secrets.token_urlsafe(32)
        session_data = {
            "username": username,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
            "is_active": True
        }
        
        if self.use_redis:
            # Store in Redis with expiration
            self.redis_client.setex(
                f"session:{session_token}",
                timedelta(hours=24),
                json.dumps(session_data)
            )
        else:
            # Store in file
            sessions = self._load_sessions()
            sessions[session_token] = session_data
            self._save_sessions(sessions)
        
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[str]:
        """
        Validate session token.
        
        Args:
            session_token: Session token
            
        Returns:
            Username if session is valid, None otherwise
        """
        if self.use_redis:
            session_data = self.redis_client.get(f"session:{session_token}")
            if session_data:
                session_data = json.loads(session_data)
                return session_data.get("username")
        else:
            sessions = self._load_sessions()
            session_data = sessions.get(session_token)
            
            if session_data:
                expires_at = datetime.fromisoformat(session_data["expires_at"])
                if datetime.now() < expires_at and session_data.get("is_active", True):
                    return session_data.get("username")
                else:
                    # Remove expired session
                    del sessions[session_token]
                    self._save_sessions(sessions)
        
        return None
    
    def logout(self, session_token: str):
        """
        Logout user by invalidating session.
        
        Args:
            session_token: Session token
        """
        if self.use_redis:
            self.redis_client.delete(f"session:{session_token}")
        else:
            sessions = self._load_sessions()
            if session_token in sessions:
                del sessions[session_token]
                self._save_sessions(sessions)
    
    def _load_sessions(self) -> Dict[str, Any]:
        """Load sessions from file."""
        if not self.use_redis and self.sessions_file.exists():
            try:
                with open(self.sessions_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_sessions(self, sessions: Dict[str, Any]):
        """Save sessions to file."""
        if not self.use_redis:
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions, f, indent=2)
    
    def get_user_info(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user information.
        
        Args:
            username: Username
            
        Returns:
            User information (without password)
        """
        if username in self.users:
            user_info = self.users[username].copy()
            # Remove sensitive information
            user_info.pop("password_hash", None)
            user_info.pop("salt", None)
            return user_info
        return None
    
    def update_user(self, username: str, updates: Dict[str, Any]) -> bool:
        """
        Update user information.
        
        Args:
            username: Username
            updates: Dictionary of updates
            
        Returns:
            True if update successful, False otherwise
        """
        if username not in self.users:
            return False
        
        # Don't allow updating sensitive fields directly
        forbidden_fields = {"password_hash", "salt", "login_attempts", "locked_until"}
        
        for key, value in updates.items():
            if key not in forbidden_fields:
                self.users[username][key] = value
        
        self._save_users()
        return True
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """
        Change user password.
        
        Args:
            username: Username
            old_password: Current password
            new_password: New password
            
        Returns:
            True if password changed successfully, False otherwise
        """
        # Verify old password
        if not self.authenticate(username, old_password):
            return False
        
        # Hash new password
        password_hash, salt = self._hash_password(new_password)
        
        # Update password
        self.users[username]["password_hash"] = password_hash
        self.users[username]["salt"] = salt
        
        self._save_users()
        return True
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics."""
        total_users = len(self.users)
        active_users = sum(1 for user in self.users.values() if user.get("is_active", True))
        admin_users = sum(1 for user in self.users.values() if user.get("role") == "admin")
        
        # Recent logins (last 24 hours)
        recent_logins = 0
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for user in self.users.values():
            if user.get("last_login"):
                last_login = datetime.fromisoformat(user["last_login"])
                if last_login > cutoff_time:
                    recent_logins += 1
        
        return {
            "total_users": total_users,
            "active_users": active_users,
            "admin_users": admin_users,
            "recent_logins": recent_logins
        }