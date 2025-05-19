#!/usr/bin/env python
# Cosmic Market Oracle - Continuous Learning Pipeline

"""
Continuous Learning Pipeline for the Cosmic Market Oracle.

This module implements a continuous learning pipeline for LLM knowledge updating,
ensuring the system stays current with the latest astrological and market insights.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
import schedule

from src.llm_integration.knowledge_base import AstrologicalKnowledgeBase
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("continuous_learning")


class ContinuousLearningPipeline:
    """
    Pipeline for continuously updating LLM knowledge.
    """
    
    def __init__(self, 
                 knowledge_base: AstrologicalKnowledgeBase,
                 data_sources: Dict[str, Dict[str, Any]],
                 update_interval: int = 24,  # hours
                 output_dir: str = "data/updates"):
        """
        Initialize the continuous learning pipeline.
        
        Args:
            knowledge_base: Knowledge base to update
            data_sources: Dictionary of data sources
            update_interval: Update interval in hours
            output_dir: Directory to save update logs
        """
        self.knowledge_base = knowledge_base
        self.data_sources = data_sources
        self.update_interval = update_interval
        self.output_dir = Path(output_dir)
        self.running = False
        self.last_update = datetime.now() - timedelta(hours=update_interval)
        self.update_thread = None
        
        # Create directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def start(self):
        """Start the continuous learning pipeline."""
        if self.running:
            logger.warning("Continuous learning pipeline already running")
            return
        
        self.running = True
        
        # Schedule updates
        schedule.every(self.update_interval).hours.do(self.update)
        
        # Start scheduler thread
        self.update_thread = threading.Thread(target=self._run_scheduler)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info(f"Started continuous learning pipeline (update interval: {self.update_interval} hours)")
    
    def _run_scheduler(self):
        """Run the scheduler."""
        while self.running:
            schedule.run_pending()
            time.sleep(60)
    
    def stop(self):
        """Stop the continuous learning pipeline."""
        if not self.running:
            logger.warning("Continuous learning pipeline not running")
            return
        
        self.running = False
        
        # Clear schedule
        schedule.clear()
        
        logger.info("Stopped continuous learning pipeline")
    
    def update(self) -> Dict[str, Any]:
        """
        Update the knowledge base with new data.
        
        Returns:
            Update results
        """
        logger.info("Starting knowledge base update")
        
        # Track results
        results = {
            "start_time": datetime.now().isoformat(),
            "sources": {},
            "documents_added": 0,
            "errors": []
        }
        
        # Process each data source
        for source_name, source_config in self.data_sources.items():
            try:
                # Collect data from source
                logger.info(f"Collecting data from {source_name}")
                
                source_data = self._collect_data(source_name, source_config)
                
                # Process data
                if source_data:
                    num_documents = self._process_data(source_name, source_data)
                    
                    # Update results
                    results["sources"][source_name] = {
                        "items_collected": len(source_data),
                        "documents_added": num_documents
                    }
                    
                    results["documents_added"] += num_documents
                else:
                    results["sources"][source_name] = {
                        "items_collected": 0,
                        "documents_added": 0
                    }
            
            except Exception as e:
                logger.error(f"Error updating from {source_name}: {e}")
                
                # Add error to results
                results["errors"].append({
                    "source": source_name,
                    "error": str(e)
                })
        
        # Update last update time
        self.last_update = datetime.now()
        results["end_time"] = self.last_update.isoformat()
        
        # Save update log
        self._save_update_log(results)
        
        logger.info(f"Completed knowledge base update: {results['documents_added']} documents added")
        
        return results
    
    def _collect_data(self, source_name: str, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect data from a source.
        
        Args:
            source_name: Source name
            source_config: Source configuration
            
        Returns:
            Collected data
        """
        source_type = source_config.get("type", "api")
        
        if source_type == "api":
            return self._collect_from_api(source_config)
        elif source_type == "file":
            return self._collect_from_file(source_config)
        elif source_type == "web":
            return self._collect_from_web(source_config)
        else:
            logger.error(f"Unknown source type: {source_type}")
            return []
    
    def _collect_from_api(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect data from API.
        
        Args:
            config: Source configuration
            
        Returns:
            Collected data
        """
        try:
            import requests
            
            # Prepare request
            url = config["url"]
            method = config.get("method", "GET")
            headers = config.get("headers", {})
            params = config.get("params", {})
            data = config.get("data", {})
            
            # Make request
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, params=params)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, params=params, json=data)
            else:
                return []
            
            # Check response
            response.raise_for_status()
            
            # Parse response
            if response.headers.get("content-type", "").startswith("application/json"):
                result = response.json()
            else:
                result = [{"text": response.text}]
            
            # Handle different response formats
            if isinstance(result, dict):
                # Extract items from response if path is specified
                if "items_path" in config:
                    path = config["items_path"].split(".")
                    
                    for key in path:
                        result = result[key]
                
                # Convert to list if not already
                if not isinstance(result, list):
                    result = [result]
            
            return result
        
        except Exception as e:
            logger.error(f"Error collecting data from API: {e}")
            return []
    
    def _collect_from_file(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect data from file.
        
        Args:
            config: Source configuration
            
        Returns:
            Collected data
        """
        try:
            import pandas as pd
            
            file_path = config["path"]
            file_format = config.get("format", "").lower()
            
            # Determine format from extension if not specified
            if not file_format and isinstance(file_path, str):
                ext = os.path.splitext(file_path)[1].lower()
                
                if ext == ".json":
                    file_format = "json"
                elif ext in [".csv", ".tsv"]:
                    file_format = "csv"
                elif ext in [".xls", ".xlsx"]:
                    file_format = "excel"
                else:
                    file_format = "text"
            
            # Read file
            if file_format == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert to list if not already
                if not isinstance(data, list):
                    data = [data]
            
            elif file_format == "csv":
                df = pd.read_csv(file_path)
                data = df.to_dict(orient="records")
            
            elif file_format == "excel":
                df = pd.read_excel(file_path)
                data = df.to_dict(orient="records")
            
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = [{"text": f.read()}]
            
            return data
        
        except Exception as e:
            logger.error(f"Error collecting data from file: {e}")
            return []
    
    def _collect_from_web(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Collect data from web.
        
        Args:
            config: Source configuration
            
        Returns:
            Collected data
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Make request
            url = config["url"]
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract content based on selector
            selector = config.get("selector")
            
            if selector:
                elements = soup.select(selector)
                
                # Extract data from each element
                data = []
                
                for element in elements:
                    item = {
                        "url": url,
                        "title": element.get_text().strip(),
                        "content": element.get_text()
                    }
                    
                    data.append(item)
            else:
                # Extract page content
                data = [{
                    "url": url,
                    "title": soup.title.string if soup.title else "",
                    "content": soup.get_text()
                }]
            
            return data
        
        except Exception as e:
            logger.error(f"Error collecting data from web: {e}")
            return []
    
    def _process_data(self, source_name: str, data: List[Dict[str, Any]]) -> int:
        """
        Process collected data and add to knowledge base.
        
        Args:
            source_name: Source name
            data: Collected data
            
        Returns:
            Number of documents added
        """
        documents_added = 0
        
        for item in data:
            try:
                # Extract content
                content = self._extract_content(item)
                
                if not content:
                    continue
                
                # Create metadata
                metadata = {
                    "source": source_name,
                    "collected_at": datetime.now().isoformat(),
                    "type": "collected"
                }
                
                # Add additional metadata from item
                for key, value in item.items():
                    if key != "content" and key != "text" and isinstance(value, (str, int, float, bool)):
                        metadata[key] = value
                
                # Add to knowledge base
                self.knowledge_base.add_document(
                    content=content,
                    metadata=metadata
                )
                
                documents_added += 1
            
            except Exception as e:
                logger.error(f"Error processing item: {e}")
        
        return documents_added
    
    def _extract_content(self, item: Dict[str, Any]) -> str:
        """
        Extract content from data item.
        
        Args:
            item: Data item
            
        Returns:
            Extracted content
        """
        # Check for content field
        if "content" in item:
            return item["content"]
        
        # Check for text field
        if "text" in item:
            return item["text"]
        
        # Check for body field
        if "body" in item:
            return item["body"]
        
        # Check for description field
        if "description" in item:
            return item["description"]
        
        # Convert item to string
        return json.dumps(item)
    
    def _save_update_log(self, results: Dict[str, Any]):
        """
        Save update log.
        
        Args:
            results: Update results
        """
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"update_log_{timestamp}.json"
        
        # Save log
        log_path = self.output_dir / filename
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved update log to {log_path}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get pipeline status.
        
        Returns:
            Pipeline status
        """
        return {
            "running": self.running,
            "last_update": self.last_update.isoformat(),
            "next_update": (self.last_update + timedelta(hours=self.update_interval)).isoformat(),
            "update_interval": self.update_interval,
            "data_sources": list(self.data_sources.keys())
        }
