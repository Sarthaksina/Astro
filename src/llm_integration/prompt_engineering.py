#!/usr/bin/env python
# Cosmic Market Oracle - Prompt Engineering Framework

"""
Specialized Prompt Engineering Framework for the Cosmic Market Oracle.

This module implements a framework for creating and managing prompts
for astrological interpretation and market analysis.
"""

import os
import json
import re
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from pathlib import Path
from datetime import datetime
import string
from jinja2 import Template

from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("prompt_engineering")


class PromptTemplate:
    """
    Template for generating prompts.
    """
    
    def __init__(self, 
                 template: str,
                 name: str,
                 description: str = "",
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a prompt template.
        
        Args:
            template: Template string with placeholders
            name: Template name
            description: Template description
            metadata: Additional metadata
        """
        self.template = template
        self.name = name
        self.description = description
        self.metadata = metadata or {}
        self.jinja_template = Template(template)
    
    def format(self, **kwargs) -> str:
        """
        Format the template with values.
        
        Args:
            **kwargs: Values for placeholders
            
        Returns:
            Formatted prompt
        """
        return self.jinja_template.render(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert template to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "template": self.template,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """
        Create template from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            Prompt template
        """
        return cls(
            template=data["template"],
            name=data["name"],
            description=data.get("description", ""),
            metadata=data.get("metadata", {})
        )


class PromptLibrary:
    """
    Library of prompt templates for different astrological scenarios.
    """
    
    def __init__(self, base_path: str = "data/prompts"):
        """
        Initialize the prompt library.
        
        Args:
            base_path: Path to store prompt templates
        """
        self.base_path = Path(base_path)
        
        # Create directory
        os.makedirs(self.base_path, exist_ok=True)
        
        # Template cache
        self.templates = {}
        
        # Load templates
        self._load_templates()
    
    def _load_templates(self):
        """Load templates from disk."""
        for template_file in self.base_path.glob("*.json"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                
                template = PromptTemplate.from_dict(template_data)
                self.templates[template.name] = template
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
        
        logger.info(f"Loaded {len(self.templates)} prompt templates")
    
    def add_template(self, template: PromptTemplate) -> str:
        """
        Add a template to the library.
        
        Args:
            template: Prompt template
            
        Returns:
            Template name
        """
        # Save template
        self.templates[template.name] = template
        
        # Save to disk
        template_file = self.base_path / f"{template.name.replace(' ', '_')}.json"
        
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(template.to_dict(), f, indent=2)
        
        logger.info(f"Added template {template.name}")
        
        return template.name
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get template by name.
        
        Args:
            name: Template name
            
        Returns:
            Prompt template or None if not found
        """
        return self.templates.get(name)
    
    def delete_template(self, name: str) -> bool:
        """
        Delete template by name.
        
        Args:
            name: Template name
            
        Returns:
            True if template was deleted, False otherwise
        """
        if name not in self.templates:
            return False
        
        # Remove template
        del self.templates[name]
        
        # Remove file
        template_file = self.base_path / f"{name.replace(' ', '_')}.json"
        
        if template_file.exists():
            os.remove(template_file)
        
        logger.info(f"Deleted template {name}")
        
        return True
    
    def list_templates(self, 
                       filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List templates in the library.
        
        Args:
            filter_metadata: Filter templates by metadata
            
        Returns:
            List of templates
        """
        results = []
        
        for name, template in self.templates.items():
            # Apply metadata filter
            if filter_metadata:
                skip = False
                
                for key, value in filter_metadata.items():
                    if key not in template.metadata or template.metadata[key] != value:
                        skip = True
                        break
                
                if skip:
                    continue
            
            # Add to results
            results.append({
                "name": name,
                "description": template.description,
                "metadata": template.metadata
            })
        
        return results
    
    def create_astrological_prompt(self, 
                                  template_name: str,
                                  planetary_data: Dict[str, Any],
                                  market_data: Optional[Dict[str, Any]] = None,
                                  additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a prompt for astrological interpretation.
        
        Args:
            template_name: Template name
            planetary_data: Planetary data
            market_data: Market data
            additional_context: Additional context
            
        Returns:
            Formatted prompt
        """
        # Get template
        template = self.get_template(template_name)
        
        if not template:
            raise ValueError(f"Template {template_name} not found")
        
        # Prepare context
        context = {
            "planetary_data": planetary_data,
            "market_data": market_data or {},
            "timestamp": datetime.now().isoformat(),
            **additional_context or {}
        }
        
        # Format template
        prompt = template.format(**context)
        
        return prompt


# Initialize default templates
def create_default_templates() -> List[PromptTemplate]:
    """
    Create default prompt templates.
    
    Returns:
        List of default templates
    """
    templates = []
    
    # General astrological analysis
    general_analysis = PromptTemplate(
        template="""
# Astrological Market Analysis

## Context
- Date: {{ timestamp }}
- Market: {{ market_data.get('symbol', 'Unknown') }}
- Current Price: {{ market_data.get('price', 'Unknown') }}
- Recent Trend: {{ market_data.get('trend', 'Unknown') }}

## Planetary Positions
{% for planet, position in planetary_data.items() %}
- {{ planet }}: {{ position.get('sign') }} at {{ position.get('degrees') }}° ({{ position.get('retrograde', False) and 'Retrograde' or 'Direct' }})
{% endfor %}

## Aspects
{% for aspect in planetary_data.get('aspects', []) %}
- {{ aspect.get('planet1') }} {{ aspect.get('aspect_type') }} {{ aspect.get('planet2') }} (Orb: {{ aspect.get('orb') }}°)
{% endfor %}

## Request
Based on the astrological data above, please provide:
1. An analysis of the current astrological influences on this market
2. Potential support and resistance levels based on planetary positions
3. Key dates to watch in the coming month based on upcoming transits
4. Overall market sentiment from an astrological perspective

Please be specific and detailed in your analysis, focusing on the most significant astrological factors.
""",
        name="General Astrological Analysis",
        description="Template for general astrological market analysis",
        metadata={"type": "analysis", "complexity": "medium"}
    )
    templates.append(general_analysis)
    
    # Predictive analysis
    predictive_analysis = PromptTemplate(
        template="""
# Astrological Market Prediction

## Context
- Date: {{ timestamp }}
- Market: {{ market_data.get('symbol', 'Unknown') }}
- Current Price: {{ market_data.get('price', 'Unknown') }}
- Time Horizon: {{ additional_context.get('time_horizon', '1 month') }}

## Planetary Positions
{% for planet, position in planetary_data.items() %}
- {{ planet }}: {{ position.get('sign') }} at {{ position.get('degrees') }}° ({{ position.get('retrograde', False) and 'Retrograde' or 'Direct' }})
{% endfor %}

## Upcoming Transits
{% for transit in planetary_data.get('upcoming_transits', []) %}
- {{ transit.get('date') }}: {{ transit.get('description') }}
{% endfor %}

## Request
Based on the astrological data above, please provide:
1. A detailed price prediction for the specified time horizon
2. Key turning points and their astrological triggers
3. Confidence level in the prediction (low/medium/high) with reasoning
4. Risk factors that could alter the prediction

Focus on quantifiable predictions where possible, and explain the astrological reasoning behind each prediction.
""",
        name="Predictive Astrological Analysis",
        description="Template for predictive astrological market analysis",
        metadata={"type": "prediction", "complexity": "high"}
    )
    templates.append(predictive_analysis)
    
    # Financial astrology explanation
    explanation = PromptTemplate(
        template="""
# Financial Astrology Explanation

## Context
- Astrological Factor: {{ additional_context.get('factor', 'Unknown') }}
- Market: {{ market_data.get('symbol', 'Unknown') }}
- Audience: {{ additional_context.get('audience', 'General') }}

## Request
Please explain the following financial astrology concept in detail:

{{ additional_context.get('factor', 'Unknown') }}

Your explanation should include:
1. The historical basis for this astrological factor in financial markets
2. How this factor is calculated or determined
3. Historical examples of this factor's influence on markets
4. How traders and investors can use this information

Tailor your explanation to the specified audience level, using appropriate terminology and depth.
""",
        name="Financial Astrology Explanation",
        description="Template for explaining financial astrology concepts",
        metadata={"type": "education", "complexity": "variable"}
    )
    templates.append(explanation)
    
    # Market regime analysis
    regime_analysis = PromptTemplate(
        template="""
# Astrological Market Regime Analysis

## Context
- Date: {{ timestamp }}
- Market: {{ market_data.get('symbol', 'Unknown') }}
- Current Regime: {{ market_data.get('current_regime', 'Unknown') }}

## Planetary Positions
{% for planet, position in planetary_data.items() %}
- {{ planet }}: {{ position.get('sign') }} at {{ position.get('degrees') }}° ({{ position.get('retrograde', False) and 'Retrograde' or 'Direct' }})
{% endfor %}

## Major Aspects
{% for aspect in planetary_data.get('major_aspects', []) %}
- {{ aspect.get('description') }}
{% endfor %}

## Request
Based on the astrological data above, please analyze:
1. The current market regime from an astrological perspective
2. When the next regime change is likely to occur
3. The expected characteristics of the next regime
4. How traders should adapt their strategies for the current and upcoming regimes

Focus on the major planetary cycles and aspects that define market regimes, particularly the outer planets.
""",
        name="Astrological Market Regime Analysis",
        description="Template for analyzing market regimes using astrology",
        metadata={"type": "regime", "complexity": "high"}
    )
    templates.append(regime_analysis)
    
    return templates
