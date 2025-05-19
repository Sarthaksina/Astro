#!/usr/bin/env python
# Cosmic Market Oracle - Conversational Interface

"""
Conversational Interface for the Cosmic Market Oracle.

This module implements a conversational interface for interactive market analysis,
allowing users to ask questions and receive astrological insights.
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime
from pathlib import Path
import re
import uuid

from src.llm_integration.retrieval_augmentation import LLMProvider, RAGSystem
from src.llm_integration.knowledge_base import AstrologicalKnowledgeBase
from src.llm_integration.explanation_module import ExplanationGenerator
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("conversation")


class Message:
    """
    Message in a conversation.
    """
    
    def __init__(self, 
                 role: str,
                 content: str,
                 message_id: Optional[str] = None,
                 timestamp: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a message.
        
        Args:
            role: Message role (user, assistant, system)
            content: Message content
            message_id: Message ID
            timestamp: Message timestamp
            metadata: Additional metadata
        """
        self.role = role
        self.content = content
        self.message_id = message_id or str(uuid.uuid4())
        self.timestamp = timestamp or datetime.now().isoformat()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert message to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "role": self.role,
            "content": self.content,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """
        Create message from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Message
        """
        return cls(
            role=data["role"],
            content=data["content"],
            message_id=data.get("message_id"),
            timestamp=data.get("timestamp"),
            metadata=data.get("metadata", {})
        )


class Conversation:
    """
    Conversation with the Cosmic Market Oracle.
    """
    
    def __init__(self, 
                 conversation_id: Optional[str] = None,
                 title: str = "New Conversation",
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a conversation.
        
        Args:
            conversation_id: Conversation ID
            title: Conversation title
            metadata: Additional metadata
        """
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.title = title
        self.metadata = metadata or {}
        self.messages = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
    
    def add_message(self, message: Message) -> str:
        """
        Add a message to the conversation.
        
        Args:
            message: Message to add
            
        Returns:
            Message ID
        """
        self.messages.append(message)
        self.updated_at = datetime.now().isoformat()
        return message.message_id
    
    def get_messages(self, 
                    limit: Optional[int] = None, 
                    role_filter: Optional[str] = None) -> List[Message]:
        """
        Get messages from the conversation.
        
        Args:
            limit: Maximum number of messages to return
            role_filter: Filter messages by role
            
        Returns:
            List of messages
        """
        # Apply role filter
        if role_filter:
            filtered_messages = [m for m in self.messages if m.role == role_filter]
        else:
            filtered_messages = self.messages
        
        # Apply limit
        if limit:
            return filtered_messages[-limit:]
        
        return filtered_messages
    
    def get_message_history(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get message history in a format suitable for LLM context.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries
        """
        messages = self.get_messages(limit=limit)
        
        return [
            {"role": m.role, "content": m.content}
            for m in messages
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert conversation to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "conversation_id": self.conversation_id,
            "title": self.title,
            "metadata": self.metadata,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Conversation':
        """
        Create conversation from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Conversation
        """
        conversation = cls(
            conversation_id=data.get("conversation_id"),
            title=data.get("title", "Conversation"),
            metadata=data.get("metadata", {})
        )
        
        conversation.created_at = data.get("created_at", conversation.created_at)
        conversation.updated_at = data.get("updated_at", conversation.updated_at)
        
        # Add messages
        for message_data in data.get("messages", []):
            message = Message.from_dict(message_data)
            conversation.messages.append(message)
        
        return conversation


class ConversationManager:
    """
    Manager for conversations.
    """
    
    def __init__(self, base_path: str = "data/conversations"):
        """
        Initialize the conversation manager.
        
        Args:
            base_path: Path to store conversations
        """
        self.base_path = Path(base_path)
        
        # Create directory
        os.makedirs(self.base_path, exist_ok=True)
        
        # Conversation cache
        self.conversations = {}
        
        # Load conversations
        self._load_conversations()
    
    def _load_conversations(self):
        """Load conversations from disk."""
        for conversation_file in self.base_path.glob("*.json"):
            try:
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    conversation_data = json.load(f)
                
                conversation = Conversation.from_dict(conversation_data)
                self.conversations[conversation.conversation_id] = conversation
            except Exception as e:
                logger.error(f"Error loading conversation {conversation_file}: {e}")
        
        logger.info(f"Loaded {len(self.conversations)} conversations")
    
    def create_conversation(self, 
                           title: str = "New Conversation",
                           metadata: Optional[Dict[str, Any]] = None) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            title: Conversation title
            metadata: Additional metadata
            
        Returns:
            New conversation
        """
        conversation = Conversation(title=title, metadata=metadata)
        
        # Add to cache
        self.conversations[conversation.conversation_id] = conversation
        
        # Save to disk
        self._save_conversation(conversation)
        
        logger.info(f"Created conversation {conversation.conversation_id}")
        
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """
        Get conversation by ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation or None if not found
        """
        return self.conversations.get(conversation_id)
    
    def add_message(self, 
                   conversation_id: str,
                   role: str,
                   content: str,
                   metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            role: Message role
            content: Message content
            metadata: Additional metadata
            
        Returns:
            Message ID or None if conversation not found
        """
        conversation = self.get_conversation(conversation_id)
        
        if not conversation:
            return None
        
        # Create message
        message = Message(role=role, content=content, metadata=metadata)
        
        # Add to conversation
        message_id = conversation.add_message(message)
        
        # Save conversation
        self._save_conversation(conversation)
        
        logger.info(f"Added message {message_id} to conversation {conversation_id}")
        
        return message_id
    
    def _save_conversation(self, conversation: Conversation):
        """
        Save conversation to disk.
        
        Args:
            conversation: Conversation to save
        """
        conversation_file = self.base_path / f"{conversation.conversation_id}.json"
        
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump(conversation.to_dict(), f, indent=2)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete conversation by ID.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            True if conversation was deleted, False otherwise
        """
        if conversation_id not in self.conversations:
            return False
        
        # Remove from cache
        del self.conversations[conversation_id]
        
        # Remove file
        conversation_file = self.base_path / f"{conversation_id}.json"
        
        if conversation_file.exists():
            os.remove(conversation_file)
        
        logger.info(f"Deleted conversation {conversation_id}")
        
        return True
    
    def list_conversations(self, 
                          limit: Optional[int] = None,
                          filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List conversations.
        
        Args:
            limit: Maximum number of conversations to return
            filter_metadata: Filter conversations by metadata
            
        Returns:
            List of conversations
        """
        results = []
        
        for conversation_id, conversation in self.conversations.items():
            # Apply metadata filter
            if filter_metadata:
                skip = False
                
                for key, value in filter_metadata.items():
                    if key not in conversation.metadata or conversation.metadata[key] != value:
                        skip = True
                        break
                
                if skip:
                    continue
            
            # Add to results
            results.append({
                "conversation_id": conversation_id,
                "title": conversation.title,
                "metadata": conversation.metadata,
                "message_count": len(conversation.messages),
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at
            })
        
        # Sort by updated_at (newest first)
        results.sort(key=lambda x: x["updated_at"], reverse=True)
        
        # Apply limit
        if limit:
            return results[:limit]
        
        return results


class ConversationalInterface:
    """
    Conversational interface for the Cosmic Market Oracle.
    """
    
    def __init__(self, 
                 llm_provider: LLMProvider,
                 rag_system: Optional[RAGSystem] = None,
                 explanation_generator: Optional[ExplanationGenerator] = None,
                 conversation_manager: Optional[ConversationManager] = None,
                 system_message: Optional[str] = None):
        """
        Initialize the conversational interface.
        
        Args:
            llm_provider: LLM provider
            rag_system: RAG system
            explanation_generator: Explanation generator
            conversation_manager: Conversation manager
            system_message: System message
        """
        self.llm_provider = llm_provider
        self.rag_system = rag_system
        self.explanation_generator = explanation_generator
        self.conversation_manager = conversation_manager or ConversationManager()
        self.system_message = system_message or self._default_system_message()
    
    def _default_system_message(self) -> str:
        """
        Get default system message.
        
        Returns:
            Default system message
        """
        return """
You are the Cosmic Market Oracle, an AI assistant specializing in financial astrology.
You analyze market trends using astrological principles and provide insights to traders.
Be precise, informative, and helpful. When you don't know something, admit it rather than speculating.
"""
    
    def create_conversation(self, 
                           title: str = "New Conversation",
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new conversation.
        
        Args:
            title: Conversation title
            metadata: Additional metadata
            
        Returns:
            Conversation ID
        """
        conversation = self.conversation_manager.create_conversation(title=title, metadata=metadata)
        
        # Add system message
        self.conversation_manager.add_message(
            conversation_id=conversation.conversation_id,
            role="system",
            content=self.system_message
        )
        
        return conversation.conversation_id
    
    def process_message(self, 
                       conversation_id: str,
                       message: str,
                       planetary_data: Optional[Dict[str, Any]] = None,
                       market_data: Optional[Dict[str, Any]] = None,
                       use_rag: bool = True) -> Dict[str, Any]:
        """
        Process a user message.
        
        Args:
            conversation_id: Conversation ID
            message: User message
            planetary_data: Planetary data
            market_data: Market data
            use_rag: Whether to use RAG
            
        Returns:
            Response
        """
        # Get conversation
        conversation = self.conversation_manager.get_conversation(conversation_id)
        
        if not conversation:
            return {"error": f"Conversation {conversation_id} not found"}
        
        # Add user message
        self.conversation_manager.add_message(
            conversation_id=conversation_id,
            role="user",
            content=message
        )
        
        # Get message history
        message_history = conversation.get_message_history()
        
        # Generate response
        if use_rag and self.rag_system:
            # Use RAG
            rag_result = self.rag_system.generate_response(
                query=message,
                template_name="General Astrological Analysis",
                planetary_data=planetary_data or {},
                market_data=market_data or {},
                additional_context={"conversation_history": message_history}
            )
            
            response_text = rag_result["response"]
            metadata = {"rag_sources": rag_result["sources"]}
        else:
            # Use direct LLM
            prompt = self._create_conversation_prompt(message_history, planetary_data, market_data)
            response_text = self.llm_provider.generate(prompt)
            metadata = {}
        
        # Add assistant message
        message_id = self.conversation_manager.add_message(
            conversation_id=conversation_id,
            role="assistant",
            content=response_text,
            metadata=metadata
        )
        
        # Prepare response
        response = {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "response": response_text,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        }
        
        return response
    
    def _create_conversation_prompt(self, 
                                   message_history: List[Dict[str, str]],
                                   planetary_data: Optional[Dict[str, Any]] = None,
                                   market_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Create conversation prompt.
        
        Args:
            message_history: Message history
            planetary_data: Planetary data
            market_data: Market data
            
        Returns:
            Prompt
        """
        # Format planetary data
        planetary_data_str = ""
        
        if planetary_data:
            planetary_data_str = "## Planetary Data\n"
            
            for planet, data in planetary_data.items():
                if isinstance(data, dict):
                    planetary_data_str += f"- {planet}: {data.get('sign', '')} at {data.get('degrees', '')}° "
                    planetary_data_str += f"({data.get('retrograde', False) and 'Retrograde' or 'Direct'})\n"
                else:
                    planetary_data_str += f"- {planet}: {data}\n"
        
        # Format market data
        market_data_str = ""
        
        if market_data:
            market_data_str = "## Market Data\n"
            
            for key, value in market_data.items():
                market_data_str += f"- {key}: {value}\n"
        
        # Format message history
        history_str = ""
        
        for msg in message_history:
            if msg["role"] == "system":
                continue
            
            history_str += f"{msg['role'].upper()}: {msg['content']}\n\n"
        
        # Create prompt
        prompt = f"""
# Cosmic Market Oracle Conversation

{planetary_data_str}

{market_data_str}

## Conversation History
{history_str}

Please respond to the user's latest message as the Cosmic Market Oracle, providing astrological insights and market analysis.
"""
        
        return prompt
    
    def detect_intent(self, message: str) -> Dict[str, Any]:
        """
        Detect user intent from message.
        
        Args:
            message: User message
            
        Returns:
            Intent
        """
        # Create prompt
        prompt = f"""
# Intent Detection

## User Message
{message}

## Task
Identify the primary intent of the user's message. Choose from the following intents:
1. Market Analysis - User wants analysis of a specific market
2. Prediction - User wants a price prediction
3. Explanation - User wants explanation of an astrological concept
4. Timing - User wants to know when to enter/exit a market
5. General Question - User has a general question about astrology or markets
6. Other - None of the above

Also extract any relevant entities:
- Market symbol (e.g., BTC, AAPL, S&P 500)
- Time horizon (e.g., day, week, month, year)
- Astrological concept (e.g., retrograde, aspect, transit)

Format your response as a JSON object with the following structure:
```json
{
  "intent": "Market Analysis",
  "confidence": 0.85,
  "entities": {
    "market_symbol": "AAPL",
    "time_horizon": "month",
    "astrological_concept": null
  }
}
```
"""
        
        # Generate response
        response = self.llm_provider.generate(prompt)
        
        # Extract JSON
        try:
            # Find JSON in response
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find any JSON-like structure
                json_match = re.search(r'(\{.*\})', response, re.DOTALL)
                
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Use the entire response
                    json_str = response
            
            # Parse JSON
            intent = json.loads(json_str)
        except Exception as e:
            logger.error(f"Error parsing intent: {e}")
            intent = {
                "intent": "General Question",
                "confidence": 0.5,
                "entities": {},
                "error": str(e)
            }
        
        return intent
