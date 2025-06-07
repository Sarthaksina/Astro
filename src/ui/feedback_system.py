#!/usr/bin/env python
"""
Feedback System for Cosmic Market Oracle Dashboard

Handles user feedback collection, rating system, and improvement suggestions.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import uuid
import statistics

logger = logging.getLogger(__name__)

class FeedbackSystem:
    """Manages user feedback and rating system."""
    
    def __init__(self, feedback_dir: str = "data/feedback"):
        """
        Initialize feedback system.
        
        Args:
            feedback_dir: Directory to store feedback data
        """
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        
        # Feedback categories
        self.feedback_categories = [
            "General Feedback",
            "Bug Report", 
            "Feature Request",
            "Improvement Suggestion",
            "Performance Issue",
            "UI/UX Feedback",
            "Documentation",
            "Training Request"
        ]
        
        # Rating categories
        self.rating_categories = [
            "prediction_accuracy",
            "ui_usability", 
            "response_time",
            "astrological_insights",
            "alert_system",
            "documentation",
            "overall_satisfaction"
        ]
        
        # Priority levels
        self.priority_levels = ["Low", "Medium", "High", "Critical"]
        
        # Status options
        self.status_options = [
            "New",
            "Under Review", 
            "In Progress",
            "Resolved",
            "Closed",
            "Rejected"
        ]
    
    def submit_feedback(self, feedback_data: Dict[str, Any]) -> str:
        """
        Submit new feedback.
        
        Args:
            feedback_data: Feedback information
            
        Returns:
            Feedback ID
        """
        # Generate unique feedback ID
        feedback_id = str(uuid.uuid4())
        
        # Prepare feedback record
        feedback_record = {
            "id": feedback_id,
            "username": feedback_data.get("username", "anonymous"),
            "timestamp": feedback_data.get("timestamp", datetime.now().isoformat()),
            "feedback_type": feedback_data.get("feedback_type", "General Feedback"),
            "feedback_text": feedback_data.get("feedback_text", ""),
            "contact_email": feedback_data.get("contact_email", ""),
            "contact_preference": feedback_data.get("contact_preference", "No follow-up needed"),
            "overall_rating": feedback_data.get("overall_rating", 3),
            "feature_ratings": feedback_data.get("feature_ratings", {}),
            "priority": self._determine_priority(feedback_data),
            "status": "New",
            "tags": self._extract_tags(feedback_data),
            "attachments": feedback_data.get("attachments", []),
            "browser_info": feedback_data.get("browser_info", {}),
            "system_info": feedback_data.get("system_info", {}),
            "session_data": feedback_data.get("session_data", {}),
            "resolved_at": None,
            "resolution_notes": "",
            "admin_notes": []
        }
        
        # Save feedback
        feedback_file = self.feedback_dir / f"feedback_{feedback_id}.json"
        
        try:
            with open(feedback_file, 'w') as f:
                json.dump(feedback_record, f, indent=2)
            
            logger.info(f"Feedback submitted: {feedback_id}")
            
            # Update statistics
            self._update_feedback_stats(feedback_record)
            
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error saving feedback {feedback_id}: {e}")
            return ""
    
    def get_feedback(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        """
        Get feedback by ID.
        
        Args:
            feedback_id: Feedback ID
            
        Returns:
            Feedback data or None if not found
        """
        feedback_file = self.feedback_dir / f"feedback_{feedback_id}.json"
        
        if feedback_file.exists():
            try:
                with open(feedback_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading feedback {feedback_id}: {e}")
        
        return None
    
    def update_feedback_status(self, feedback_id: str, status: str, notes: str = "") -> bool:
        """
        Update feedback status.
        
        Args:
            feedback_id: Feedback ID
            status: New status
            notes: Resolution notes
            
        Returns:
            True if updated successfully, False otherwise
        """
        feedback = self.get_feedback(feedback_id)
        
        if feedback:
            feedback["status"] = status
            
            if notes:
                feedback["resolution_notes"] = notes
            
            if status in ["Resolved", "Closed"]:
                feedback["resolved_at"] = datetime.now().isoformat()
            
            # Save updated feedback
            feedback_file = self.feedback_dir / f"feedback_{feedback_id}.json"
            
            try:
                with open(feedback_file, 'w') as f:
                    json.dump(feedback, f, indent=2)
                
                logger.info(f"Updated feedback {feedback_id} status to {status}")
                return True
                
            except Exception as e:
                logger.error(f"Error updating feedback {feedback_id}: {e}")
        
        return False
    
    def add_admin_note(self, feedback_id: str, note: str, admin_user: str) -> bool:
        """
        Add admin note to feedback.
        
        Args:
            feedback_id: Feedback ID """