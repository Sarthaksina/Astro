#!/usr/bin/env python
# Cosmic Market Oracle - Retrieval-Augmented Generation

"""
Retrieval-Augmented Generation (RAG) System for the Cosmic Market Oracle.

This module implements a RAG system that combines the knowledge base with
LLM capabilities for enhanced astrological market analysis.
"""

import os
import json
import re
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from pathlib import Path
import numpy as np
import requests
from datetime import datetime
import time

from src.llm_integration.knowledge_base import AstrologicalKnowledgeBase
from src.llm_integration.prompt_engineering import PromptTemplate, PromptLibrary
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("retrieval_augmentation")


class LLMProvider:
    """
    Provider for Large Language Model API access.
    """
    
    from .constants import (
        DEFAULT_LLM_PROVIDER,
        DEFAULT_LLM_MODEL,
        DEFAULT_MAX_TOKENS,
        DEFAULT_TEMPERATURE
    )

    def __init__(self, 
                 provider: str = DEFAULT_LLM_PROVIDER,
                 api_key: Optional[str] = None,
                 model: str = DEFAULT_LLM_MODEL,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = DEFAULT_TEMPERATURE):
        """
        Initialize the LLM provider.
        
        Args:
            provider: Provider name (openai, anthropic, etc.)
            api_key: API key (if None, read from environment)
            model: Model name
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
        """
        self.provider = provider.lower()
        self.api_key = api_key or os.environ.get(f"{provider.upper()}_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        if not self.api_key:
            logger.warning(f"No API key provided for {provider}. Set {provider.upper()}_API_KEY environment variable.")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using the LLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        if self.provider == "openai":
            return self._generate_openai(prompt, **kwargs)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _generate_openai(self, prompt: str, **kwargs) -> str:
        """
        Generate text using OpenAI API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        try:
            import openai
            
            openai.api_key = self.api_key
            
            # Prepare parameters
            params = {
                "model": kwargs.get("model", self.model),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", 1.0),
                "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                "presence_penalty": kwargs.get("presence_penalty", 0.0)
            }
            
            # Call API
            response = openai.ChatCompletion.create(**params)
            
            # Extract text
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"Error generating text with OpenAI: {e}")
            return f"Error: {str(e)}"
    
    def _generate_anthropic(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Anthropic API.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Prepare parameters
            params = {
                "model": kwargs.get("model", self.model),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Call API
            response = client.messages.create(**params)
            
            # Extract text
            return response.content[0].text
        
        except Exception as e:
            logger.error(f"Error generating text with Anthropic: {e}")
            return f"Error: {str(e)}"


class RAGSystem:
    """
    Retrieval-Augmented Generation System for astrological market analysis.
    """
    
    def __init__(self, 
                 knowledge_base: AstrologicalKnowledgeBase,
                 prompt_library: PromptLibrary,
                 llm_provider: LLMProvider,
                 num_documents: int = 5,
                 max_tokens_per_document: int = 500):
        """
        Initialize the RAG system.
        
        Args:
            knowledge_base: Astrological knowledge base
            prompt_library: Prompt library
            llm_provider: LLM provider
            num_documents: Number of documents to retrieve
            max_tokens_per_document: Maximum tokens per document
        """
        self.knowledge_base = knowledge_base
        self.prompt_library = prompt_library
        self.llm_provider = llm_provider
        self.num_documents = num_documents
        self.max_tokens_per_document = max_tokens_per_document
    
    def generate_response(self, 
                         query: str,
                         template_name: str,
                         planetary_data: Dict[str, Any],
                         market_data: Optional[Dict[str, Any]] = None,
                         additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a response using RAG.
        
        Args:
            query: User query
            template_name: Prompt template name
            planetary_data: Planetary data
            market_data: Market data
            additional_context: Additional context
            
        Returns:
            Dictionary with response and metadata
        """
        # Search knowledge base
        search_results = self.knowledge_base.search(
            query=query, 
            top_k=self.num_documents
        )
        
        # Extract relevant context
        context = self._extract_context(search_results)
        
        # Create prompt
        if additional_context is None:
            additional_context = {}
        
        additional_context["retrieved_context"] = context
        additional_context["query"] = query
        
        prompt = self.prompt_library.create_astrological_prompt(
            template_name=template_name,
            planetary_data=planetary_data,
            market_data=market_data,
            additional_context=additional_context
        )
        
        # Generate response
        response_text = self.llm_provider.generate(prompt)
        
        # Prepare result
        result = {
            "response": response_text,
            "sources": [
                {
                    "doc_id": result["doc_id"],
                    "chunk_id": result["chunk_id"],
                    "metadata": result["metadata"]
                }
                for result in search_results
            ],
            "prompt": prompt,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _extract_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Extract context from search results.
        
        Args:
            search_results: Search results
            
        Returns:
            Extracted context
        """
        context_parts = []
        
        for i, result in enumerate(search_results):
            # Add source information
            source_info = f"[Source {i+1}: {result['metadata'].get('title', 'Unknown')}]"
            
            # Add text
            text = result["text"]
            
            # Truncate if too long
            if len(text) > self.max_tokens_per_document * 4:  # Approximate token count
                text = text[:self.max_tokens_per_document * 4] + "..."
            
            # Add to context
            context_parts.append(f"{source_info}\n{text}\n")
        
        return "\n".join(context_parts)


class RAGQueryEngine:
    """
    Query engine for the RAG system.
    """
    
    def __init__(self, rag_system: RAGSystem):
        """
        Initialize the query engine.
        
        Args:
            rag_system: RAG system
        """
        self.rag_system = rag_system
        self.query_history = []
    
    def query(self, 
             query: str,
             template_name: str = "General Astrological Analysis",
             planetary_data: Optional[Dict[str, Any]] = None,
             market_data: Optional[Dict[str, Any]] = None,
             additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            query: User query
            template_name: Prompt template name
            planetary_data: Planetary data
            market_data: Market data
            additional_context: Additional context
            
        Returns:
            Dictionary with response and metadata
        """
        # Generate response
        result = self.rag_system.generate_response(
            query=query,
            template_name=template_name,
            planetary_data=planetary_data or {},
            market_data=market_data or {},
            additional_context=additional_context or {}
        )
        
        # Add to history
        self.query_history.append({
            "query": query,
            "template_name": template_name,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get query history.
        
        Args:
            limit: Maximum number of items to return
            
        Returns:
            List of query history items
        """
        return self.query_history[-limit:]
