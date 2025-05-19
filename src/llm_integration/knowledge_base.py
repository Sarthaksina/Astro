#!/usr/bin/env python
# Cosmic Market Oracle - Astrological Knowledge Base

"""
Astrological Knowledge Base for the Cosmic Market Oracle.

This module implements a knowledge base for storing and retrieving
astrological literature and concepts for financial analysis.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from pathlib import Path
import pickle
import hashlib
from datetime import datetime
import re
import faiss
from sentence_transformers import SentenceTransformer

from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("knowledge_base")


class Document:
    """
    Document in the knowledge base.
    """
    
    def __init__(self, 
                 content: str, 
                 metadata: Dict[str, Any],
                 doc_id: Optional[str] = None):
        """
        Initialize a document.
        
        Args:
            content: Document content
            metadata: Document metadata
            doc_id: Document ID (generated if not provided)
        """
        self.content = content
        self.metadata = metadata
        self.doc_id = doc_id or self._generate_id()
        self.chunks = []
        self.embedding = None
    
    def _generate_id(self) -> str:
        """
        Generate a document ID.
        
        Returns:
            Document ID
        """
        # Generate hash from content and timestamp
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"{content_hash}_{timestamp}"
    
    def chunk(self, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Split document into chunks.
        
        Args:
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of chunks
        """
        # Split content into chunks
        text = self.content
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            
            # Skip small chunks at the end
            if len(chunk_text) < 100 and chunks:
                chunks[-1]["text"] += " " + chunk_text
                continue
            
            # Create chunk
            chunk = {
                "text": chunk_text,
                "metadata": self.metadata.copy(),
                "doc_id": self.doc_id,
                "chunk_id": f"{self.doc_id}_{len(chunks)}",
                "position": len(chunks)
            }
            
            chunks.append(chunk)
        
        self.chunks = chunks
        return chunks
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert document to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata,
            "chunks": self.chunks
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """
        Create document from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Document
        """
        doc = cls(
            content=data["content"],
            metadata=data["metadata"],
            doc_id=data["doc_id"]
        )
        doc.chunks = data.get("chunks", [])
        return doc


class AstrologicalKnowledgeBase:
    """
    Knowledge base for astrological literature and concepts.
    """
    
    def __init__(self, 
                 base_path: str = "data/knowledge_base",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 index_name: str = "astro_index"):
        """
        Initialize the knowledge base.
        
        Args:
            base_path: Path to store knowledge base
            embedding_model: Name of the embedding model
            index_name: Name of the FAISS index
        """
        self.base_path = Path(base_path)
        self.documents_path = self.base_path / "documents"
        self.index_path = self.base_path / "index"
        self.embedding_model_name = embedding_model
        self.index_name = index_name
        
        # Create directories
        os.makedirs(self.documents_path, exist_ok=True)
        os.makedirs(self.index_path, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = None
        self.chunk_ids = []
        
        # Load or create index
        self._load_or_create_index()
        
        # Document cache
        self.documents = {}
        
        # Load documents
        self._load_documents()
    
    def _load_or_create_index(self):
        """Load or create FAISS index."""
        index_file = self.index_path / f"{self.index_name}.index"
        ids_file = self.index_path / f"{self.index_name}_ids.pkl"
        
        if index_file.exists() and ids_file.exists():
            # Load existing index
            self.index = faiss.read_index(str(index_file))
            
            with open(ids_file, 'rb') as f:
                self.chunk_ids = pickle.load(f)
                
            logger.info(f"Loaded index with {len(self.chunk_ids)} chunks")
        else:
            # Create new index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.chunk_ids = []
            
            logger.info("Created new index")
    
    def _save_index(self):
        """Save FAISS index."""
        index_file = self.index_path / f"{self.index_name}.index"
        ids_file = self.index_path / f"{self.index_name}_ids.pkl"
        
        faiss.write_index(self.index, str(index_file))
        
        with open(ids_file, 'wb') as f:
            pickle.dump(self.chunk_ids, f)
            
        logger.info(f"Saved index with {len(self.chunk_ids)} chunks")
    
    def _load_documents(self):
        """Load documents from disk."""
        for doc_file in self.documents_path.glob("*.json"):
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    doc_data = json.load(f)
                
                doc = Document.from_dict(doc_data)
                self.documents[doc.doc_id] = doc
            except Exception as e:
                logger.error(f"Error loading document {doc_file}: {e}")
        
        logger.info(f"Loaded {len(self.documents)} documents")
    
    def add_document(self, 
                     content: str, 
                     metadata: Dict[str, Any],
                     chunk_size: int = 1000,
                     overlap: int = 200) -> str:
        """
        Add a document to the knowledge base.
        
        Args:
            content: Document content
            metadata: Document metadata
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            Document ID
        """
        # Create document
        doc = Document(content=content, metadata=metadata)
        
        # Chunk document
        chunks = doc.chunk(chunk_size=chunk_size, overlap=overlap)
        
        # Embed chunks
        for chunk in chunks:
            embedding = self.embedding_model.encode([chunk["text"]])[0]
            
            # Add to index
            self.index.add(np.array([embedding], dtype=np.float32))
            self.chunk_ids.append(chunk["chunk_id"])
        
        # Save document
        self.documents[doc.doc_id] = doc
        
        with open(self.documents_path / f"{doc.doc_id}.json", 'w', encoding='utf-8') as f:
            json.dump(doc.to_dict(), f, indent=2)
        
        # Save index
        self._save_index()
        
        logger.info(f"Added document {doc.doc_id} with {len(chunks)} chunks")
        
        return doc.doc_id
    
    def search(self, 
               query: str, 
               top_k: int = 5,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for documents similar to query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Filter results by metadata
            
        Returns:
            List of search results
        """
        # Embed query
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search index
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32), 
            min(top_k * 2, len(self.chunk_ids))  # Get more results for filtering
        )
        
        # Get chunk IDs
        results = []
        
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
                
            chunk_id = self.chunk_ids[idx]
            doc_id, chunk_pos = chunk_id.rsplit('_', 1)
            
            if doc_id not in self.documents:
                continue
            
            doc = self.documents[doc_id]
            chunk_pos = int(chunk_pos)
            
            if chunk_pos >= len(doc.chunks):
                continue
            
            chunk = doc.chunks[chunk_pos]
            
            # Apply metadata filter
            if filter_metadata:
                skip = False
                
                for key, value in filter_metadata.items():
                    if key not in chunk["metadata"] or chunk["metadata"][key] != value:
                        skip = True
                        break
                
                if skip:
                    continue
            
            # Add to results
            results.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "distance": float(distances[0][i])
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document or None if not found
        """
        return self.documents.get(doc_id)
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk or None if not found
        """
        doc_id, chunk_pos = chunk_id.rsplit('_', 1)
        
        if doc_id not in self.documents:
            return None
        
        doc = self.documents[doc_id]
        chunk_pos = int(chunk_pos)
        
        if chunk_pos >= len(doc.chunks):
            return None
        
        return doc.chunks[chunk_pos]
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if document was deleted, False otherwise
        """
        if doc_id not in self.documents:
            return False
        
        # Get document
        doc = self.documents[doc_id]
        
        # Remove from index
        # Note: FAISS doesn't support direct deletion, so we need to rebuild the index
        chunk_ids_to_remove = [chunk["chunk_id"] for chunk in doc.chunks]
        
        # Create new index
        new_index = faiss.IndexFlatL2(self.embedding_dim)
        new_chunk_ids = []
        
        # Copy data
        for i, chunk_id in enumerate(self.chunk_ids):
            if chunk_id in chunk_ids_to_remove:
                continue
            
            # Get embedding
            embedding = np.array([self.index.reconstruct(i)], dtype=np.float32)
            
            # Add to new index
            new_index.add(embedding)
            new_chunk_ids.append(chunk_id)
        
        # Replace index
        self.index = new_index
        self.chunk_ids = new_chunk_ids
        
        # Remove document
        del self.documents[doc_id]
        
        # Remove file
        doc_file = self.documents_path / f"{doc_id}.json"
        
        if doc_file.exists():
            os.remove(doc_file)
        
        # Save index
        self._save_index()
        
        logger.info(f"Deleted document {doc_id}")
        
        return True
    
    def list_documents(self, 
                       filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List documents in the knowledge base.
        
        Args:
            filter_metadata: Filter documents by metadata
            
        Returns:
            List of documents
        """
        results = []
        
        for doc_id, doc in self.documents.items():
            # Apply metadata filter
            if filter_metadata:
                skip = False
                
                for key, value in filter_metadata.items():
                    if key not in doc.metadata or doc.metadata[key] != value:
                        skip = True
                        break
                
                if skip:
                    continue
            
            # Add to results
            results.append({
                "doc_id": doc_id,
                "metadata": doc.metadata,
                "num_chunks": len(doc.chunks)
            })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Dictionary with statistics
        """
        # Count documents by type
        doc_types = {}
        
        for doc in self.documents.values():
            doc_type = doc.metadata.get("type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        # Count total chunks
        total_chunks = sum(len(doc.chunks) for doc in self.documents.values())
        
        return {
            "num_documents": len(self.documents),
            "num_chunks": total_chunks,
            "document_types": doc_types,
            "index_size": self.index.ntotal
        }
