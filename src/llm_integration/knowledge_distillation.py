#!/usr/bin/env python
# Cosmic Market Oracle - Knowledge Distillation

"""
Knowledge Distillation Module for the Cosmic Market Oracle.

This module implements specialized knowledge distillation from LLM to smaller models,
allowing efficient deployment of astrological knowledge.
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

from src.llm_integration.retrieval_augmentation import LLMProvider
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("knowledge_distillation")


class AstrologicalQADataset(Dataset):
    """
    Dataset for astrological question-answering pairs.
    """
    
    def __init__(self, 
                 data: List[Dict[str, str]],
                 tokenizer,
                 max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            data: List of QA pairs
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        """
        Get dataset length.
        
        Returns:
            Dataset length
        """
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Encoded item
        """
        item = self.data[idx]
        question = item["question"]
        answer = item["answer"]
        
        # Encode input
        inputs = self.tokenizer(
            question,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode output
        outputs = self.tokenizer(
            answer,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Convert to tensors
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        labels = outputs["input_ids"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class KnowledgeDistiller:
    """
    Distiller for astrological knowledge from LLM to smaller models.
    """
    
    def __init__(self, 
                 llm_provider: LLMProvider,
                 model_name: str = "distilgpt2",
                 output_dir: str = "models/distilled",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the knowledge distiller.
        
        Args:
            llm_provider: LLM provider
            model_name: Student model name
            output_dir: Directory to save models
            device: Device for training
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.device = device
        
        # Create directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize tokenizer and model
        self.tokenizer = None
        self.model = None
        
        logger.info(f"Initialized knowledge distiller with model {model_name} on {device}")
    
    def load_model(self):
        """Load student model and tokenizer."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            
            logger.info(f"Loaded model {self.model_name}")
        
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def generate_training_data(self, 
                              num_samples: int = 1000,
                              topics: List[str] = None,
                              output_file: str = None) -> List[Dict[str, str]]:
        """
        Generate training data using the teacher LLM.
        
        Args:
            num_samples: Number of samples to generate
            topics: List of topics to cover
            output_file: File to save data
            
        Returns:
            Generated training data
        """
        # Default topics
        if topics is None:
            topics = [
                "planetary aspects",
                "retrograde planets",
                "market cycles",
                "astrological timing",
                "financial astrology basics",
                "planetary influences on markets",
                "astrological support and resistance",
                "lunar cycles in trading",
                "solar returns in financial markets",
                "planetary transits"
            ]
        
        # Generate questions and answers
        data = []
        
        logger.info(f"Generating {num_samples} training samples...")
        
        for _ in tqdm(range(num_samples)):
            # Select random topic
            topic = np.random.choice(topics)
            
            # Generate question
            question_prompt = f"""
Generate a specific question about {topic} in the context of financial astrology and market analysis.
The question should be detailed and require expert knowledge to answer.
Only return the question itself, nothing else.
"""
            
            question = self.llm_provider.generate(question_prompt)
            
            # Generate answer
            answer_prompt = f"""
Question: {question}

Please provide a detailed and accurate answer to this question about financial astrology.
Your answer should be comprehensive, educational, and reflect expert knowledge.
"""
            
            answer = self.llm_provider.generate(answer_prompt)
            
            # Add to data
            data.append({
                "question": question,
                "answer": answer,
                "topic": topic
            })
        
        # Save data if requested
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(data)} training samples to {output_file}")
        
        return data
    
    def train(self, 
             training_data: List[Dict[str, str]],
             epochs: int = 3,
             batch_size: int = 8,
             learning_rate: float = 5e-5,
             save_steps: int = 500) -> Dict[str, Any]:
        """
        Train the student model.
        
        Args:
            training_data: Training data
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            save_steps: Steps between checkpoints
            
        Returns:
            Training metrics
        """
        # Load model if not loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Create dataset
        dataset = AstrologicalQADataset(
            data=training_data,
            tokenizer=self.tokenizer,
            max_length=512
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Setup optimizer
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        total_loss = 0
        global_step = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update loss
                epoch_loss += loss.item()
                total_loss += loss.item()
                
                # Save checkpoint
                if global_step > 0 and global_step % save_steps == 0:
                    self._save_checkpoint(global_step)
                
                global_step += 1
            
            # Log epoch loss
            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}")
        
        # Save final model
        self._save_model()
        
        # Return metrics
        metrics = {
            "total_loss": total_loss,
            "average_loss": total_loss / (epochs * len(dataloader)),
            "epochs": epochs,
            "global_steps": global_step,
            "model_path": str(self.output_dir / "final")
        }
        
        return metrics
    
    def _save_checkpoint(self, step: int):
        """
        Save model checkpoint.
        
        Args:
            step: Current step
        """
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(checkpoint_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def _save_model(self):
        """Save final model."""
        final_dir = self.output_dir / "final"
        os.makedirs(final_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(final_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(final_dir)
        
        logger.info(f"Saved final model to {final_dir}")
    
    def evaluate(self, 
                test_data: List[Dict[str, str]],
                metrics: List[str] = ["bleu", "rouge"]) -> Dict[str, float]:
        """
        Evaluate the student model.
        
        Args:
            test_data: Test data
            metrics: Metrics to compute
            
        Returns:
            Evaluation metrics
        """
        # Load model if not loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Set model to eval mode
        self.model.eval()
        
        # Generate predictions
        predictions = []
        references = []
        
        logger.info(f"Evaluating model on {len(test_data)} test samples...")
        
        for item in tqdm(test_data):
            question = item["question"]
            reference = item["answer"]
            
            # Generate prediction
            inputs = self.tokenizer(
                question,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=512,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode prediction
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Add to lists
            predictions.append(prediction)
            references.append(reference)
        
        # Compute metrics
        results = {}
        
        if "bleu" in metrics:
            try:
                from nltk.translate.bleu_score import corpus_bleu
                import nltk
                
                # Download NLTK data if needed
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt')
                
                # Tokenize
                tokenized_predictions = [nltk.word_tokenize(p.lower()) for p in predictions]
                tokenized_references = [[nltk.word_tokenize(r.lower())] for r in references]
                
                # Compute BLEU
                bleu = corpus_bleu(tokenized_references, tokenized_predictions)
                results["bleu"] = bleu
            
            except Exception as e:
                logger.error(f"Error computing BLEU: {e}")
                results["bleu"] = None
        
        if "rouge" in metrics:
            try:
                from rouge import Rouge
                
                rouge = Rouge()
                scores = rouge.get_scores(predictions, references, avg=True)
                
                results["rouge-1"] = scores["rouge-1"]["f"]
                results["rouge-2"] = scores["rouge-2"]["f"]
                results["rouge-l"] = scores["rouge-l"]["f"]
            
            except Exception as e:
                logger.error(f"Error computing ROUGE: {e}")
                results["rouge-1"] = None
                results["rouge-2"] = None
                results["rouge-l"] = None
        
        logger.info(f"Evaluation results: {results}")
        
        return results
    
    def generate_response(self, question: str, max_length: int = 512) -> str:
        """
        Generate a response using the student model.
        
        Args:
            question: Input question
            max_length: Maximum response length
            
        Returns:
            Generated response
        """
        # Load model if not loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        # Set model to eval mode
        self.model.eval()
        
        # Encode input
        inputs = self.tokenizer(
            question,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response
