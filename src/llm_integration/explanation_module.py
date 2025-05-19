#!/usr/bin/env python
# Cosmic Market Oracle - Explanation Module

"""
Explanation Module for the Cosmic Market Oracle.

This module implements an explanation generator for astrological market predictions,
making the reasoning transparent and understandable.
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime
from pathlib import Path

from src.llm_integration.retrieval_augmentation import LLMProvider
from src.llm_integration.knowledge_base import AstrologicalKnowledgeBase
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("explanation_module")


class ExplanationGenerator:
    """
    Generator for explanations of astrological market predictions.
    """
    
    def __init__(self, 
                 llm_provider: LLMProvider,
                 knowledge_base: Optional[AstrologicalKnowledgeBase] = None,
                 explanation_levels: List[str] = ["beginner", "intermediate", "advanced"]):
        """
        Initialize the explanation generator.
        
        Args:
            llm_provider: LLM provider
            knowledge_base: Astrological knowledge base
            explanation_levels: Available explanation levels
        """
        self.llm_provider = llm_provider
        self.knowledge_base = knowledge_base
        self.explanation_levels = explanation_levels
        self.explanation_cache = {}
    
    def explain_prediction(self, 
                          prediction: Dict[str, Any],
                          level: str = "intermediate",
                          include_sources: bool = True) -> Dict[str, Any]:
        """
        Generate an explanation for a prediction.
        
        Args:
            prediction: Prediction to explain
            level: Explanation level
            include_sources: Whether to include sources
            
        Returns:
            Explanation
        """
        # Validate level
        if level not in self.explanation_levels:
            level = "intermediate"
        
        # Create cache key
        prediction_str = json.dumps(prediction, sort_keys=True)
        cache_key = f"{prediction_str}_{level}_{include_sources}"
        
        # Check cache
        if cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]
        
        # Extract key elements from prediction
        prediction_elements = self._extract_prediction_elements(prediction)
        
        # Get relevant sources if requested
        sources = []
        
        if include_sources and self.knowledge_base:
            sources = self._get_relevant_sources(prediction_elements)
        
        # Generate explanation
        explanation = self._generate_explanation(prediction, prediction_elements, level, sources)
        
        # Cache explanation
        self.explanation_cache[cache_key] = explanation
        
        return explanation
    
    def _extract_prediction_elements(self, prediction: Dict[str, Any]) -> List[str]:
        """
        Extract key elements from prediction.
        
        Args:
            prediction: Prediction
            
        Returns:
            List of key elements
        """
        elements = []
        
        # Extract astrological factors
        if "astrological_factors" in prediction:
            for factor in prediction["astrological_factors"]:
                if isinstance(factor, dict):
                    elements.append(factor.get("factor", ""))
                else:
                    elements.append(str(factor))
        
        # Extract aspects
        if "aspects" in prediction:
            for aspect in prediction["aspects"]:
                if isinstance(aspect, dict):
                    elements.append(f"{aspect.get('planet1', '')} {aspect.get('aspect_type', '')} {aspect.get('planet2', '')}")
                else:
                    elements.append(str(aspect))
        
        # Extract other key elements
        for key in ["direction", "forecast", "recommendation"]:
            if key in prediction:
                elements.append(str(prediction[key]))
        
        return [e for e in elements if e]
    
    def _get_relevant_sources(self, elements: List[str]) -> List[Dict[str, Any]]:
        """
        Get relevant sources for elements.
        
        Args:
            elements: List of elements
            
        Returns:
            List of sources
        """
        sources = []
        
        for element in elements:
            # Search knowledge base
            results = self.knowledge_base.search(
                query=element,
                top_k=2
            )
            
            # Add to sources
            for result in results:
                sources.append({
                    "element": element,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "doc_id": result["doc_id"]
                })
        
        return sources
    
    def _generate_explanation(self, 
                             prediction: Dict[str, Any],
                             elements: List[str],
                             level: str,
                             sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate explanation.
        
        Args:
            prediction: Prediction
            elements: Key elements
            level: Explanation level
            sources: Relevant sources
            
        Returns:
            Explanation
        """
        # Create prompt
        prompt = self._create_explanation_prompt(prediction, elements, level, sources)
        
        # Generate explanation
        explanation_text = self.llm_provider.generate(prompt)
        
        # Create explanation
        explanation = {
            "prediction": prediction,
            "explanation": explanation_text,
            "level": level,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }
        
        return explanation
    
    def _create_explanation_prompt(self, 
                                  prediction: Dict[str, Any],
                                  elements: List[str],
                                  level: str,
                                  sources: List[Dict[str, Any]]) -> str:
        """
        Create explanation prompt.
        
        Args:
            prediction: Prediction
            elements: Key elements
            level: Explanation level
            sources: Relevant sources
            
        Returns:
            Prompt
        """
        # Format prediction as string
        prediction_str = json.dumps(prediction, indent=2)
        
        # Format sources as string
        sources_str = ""
        
        for i, source in enumerate(sources):
            sources_str += f"Source {i+1} - {source['element']}:\n"
            sources_str += f"{source['text']}\n\n"
        
        # Create prompt based on level
        if level == "beginner":
            prompt = f"""
# Explanation Request - Beginner Level

## Astrological Market Prediction
```json
{prediction_str}
```

## Key Elements
{', '.join(elements)}

## Relevant Sources
{sources_str}

## Task
Explain this astrological market prediction in simple terms for someone with no background in astrology or financial markets.
Your explanation should:
1. Avoid technical jargon
2. Use simple analogies to explain astrological concepts
3. Clearly explain why the prediction is what it is
4. Provide a straightforward interpretation of what actions to take

Focus on making the explanation accessible and practical.
"""
        
        elif level == "advanced":
            prompt = f"""
# Explanation Request - Advanced Level

## Astrological Market Prediction
```json
{prediction_str}
```

## Key Elements
{', '.join(elements)}

## Relevant Sources
{sources_str}

## Task
Provide a detailed technical explanation of this astrological market prediction for someone with advanced knowledge of both astrology and financial markets.
Your explanation should:
1. Include technical astrological terminology
2. Reference specific astrological principles and theories
3. Explain the mathematical or geometrical relationships between celestial bodies
4. Provide detailed reasoning for each aspect of the prediction
5. Discuss historical precedents for similar configurations

Focus on depth, precision, and technical accuracy.
"""
        
        else:  # intermediate
            prompt = f"""
# Explanation Request - Intermediate Level

## Astrological Market Prediction
```json
{prediction_str}
```

## Key Elements
{', '.join(elements)}

## Relevant Sources
{sources_str}

## Task
Explain this astrological market prediction for someone with basic knowledge of astrology and financial markets.
Your explanation should:
1. Use appropriate astrological terminology but explain key concepts
2. Connect astrological factors to market behavior
3. Provide clear reasoning for the prediction
4. Balance technical detail with accessibility

Focus on providing a clear understanding of the astrological reasoning behind the prediction.
"""
        
        return prompt
    
    def explain_concept(self, 
                       concept: str,
                       level: str = "intermediate",
                       include_sources: bool = True) -> Dict[str, Any]:
        """
        Generate an explanation for an astrological concept.
        
        Args:
            concept: Concept to explain
            level: Explanation level
            include_sources: Whether to include sources
            
        Returns:
            Explanation
        """
        # Validate level
        if level not in self.explanation_levels:
            level = "intermediate"
        
        # Create cache key
        cache_key = f"concept_{concept}_{level}_{include_sources}"
        
        # Check cache
        if cache_key in self.explanation_cache:
            return self.explanation_cache[cache_key]
        
        # Get relevant sources if requested
        sources = []
        
        if include_sources and self.knowledge_base:
            # Search knowledge base
            results = self.knowledge_base.search(
                query=concept,
                top_k=3
            )
            
            # Add to sources
            for result in results:
                sources.append({
                    "element": concept,
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "doc_id": result["doc_id"]
                })
        
        # Create prompt
        prompt = self._create_concept_prompt(concept, level, sources)
        
        # Generate explanation
        explanation_text = self.llm_provider.generate(prompt)
        
        # Create explanation
        explanation = {
            "concept": concept,
            "explanation": explanation_text,
            "level": level,
            "sources": sources,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache explanation
        self.explanation_cache[cache_key] = explanation
        
        return explanation
    
    def _create_concept_prompt(self, 
                              concept: str,
                              level: str,
                              sources: List[Dict[str, Any]]) -> str:
        """
        Create concept explanation prompt.
        
        Args:
            concept: Concept to explain
            level: Explanation level
            sources: Relevant sources
            
        Returns:
            Prompt
        """
        # Format sources as string
        sources_str = ""
        
        for i, source in enumerate(sources):
            sources_str += f"Source {i+1}:\n"
            sources_str += f"{source['text']}\n\n"
        
        # Create prompt based on level
        if level == "beginner":
            prompt = f"""
# Concept Explanation Request - Beginner Level

## Astrological Concept
{concept}

## Relevant Sources
{sources_str}

## Task
Explain this astrological concept in simple terms for someone with no background in astrology or financial markets.
Your explanation should:
1. Avoid technical jargon
2. Use simple analogies to explain the concept
3. Provide examples of how this concept applies to financial markets
4. Make the explanation accessible and practical

Focus on making the explanation accessible and practical.
"""
        
        elif level == "advanced":
            prompt = f"""
# Concept Explanation Request - Advanced Level

## Astrological Concept
{concept}

## Relevant Sources
{sources_str}

## Task
Provide a detailed technical explanation of this astrological concept for someone with advanced knowledge of both astrology and financial markets.
Your explanation should:
1. Include technical astrological terminology
2. Reference specific astrological principles and theories
3. Explain the mathematical or geometrical relationships if applicable
4. Discuss historical development of this concept
5. Provide detailed examples of how this concept applies to financial markets
6. Discuss varying interpretations or schools of thought

Focus on depth, precision, and technical accuracy.
"""
        
        else:  # intermediate
            prompt = f"""
# Concept Explanation Request - Intermediate Level

## Astrological Concept
{concept}

## Relevant Sources
{sources_str}

## Task
Explain this astrological concept for someone with basic knowledge of astrology and financial markets.
Your explanation should:
1. Use appropriate astrological terminology but explain key terms
2. Connect the concept to market behavior
3. Provide clear examples
4. Balance technical detail with accessibility

Focus on providing a clear understanding of the concept and its application to financial markets.
"""
        
        return prompt
