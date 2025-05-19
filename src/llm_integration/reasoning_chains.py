#!/usr/bin/env python
# Cosmic Market Oracle - Reasoning Chains

"""
Market-specific Reasoning Chains for the Cosmic Market Oracle.

This module implements specialized reasoning chains for complex astrological
market analysis, enabling structured and transparent reasoning.
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime
import re
from pathlib import Path

from src.llm_integration.retrieval_augmentation import LLMProvider
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("reasoning_chains")


class ReasoningStep:
    """
    Step in a reasoning chain.
    """
    
    def __init__(self, 
                 name: str,
                 description: str,
                 prompt_template: str,
                 output_parser: Optional[callable] = None):
        """
        Initialize a reasoning step.
        
        Args:
            name: Step name
            description: Step description
            prompt_template: Prompt template for the step
            output_parser: Function to parse the output
        """
        self.name = name
        self.description = description
        self.prompt_template = prompt_template
        self.output_parser = output_parser
    
    def execute(self, 
               llm_provider: LLMProvider,
               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the reasoning step.
        
        Args:
            llm_provider: LLM provider
            context: Context for the step
            
        Returns:
            Updated context
        """
        # Format prompt
        prompt = self.prompt_template.format(**context)
        
        # Generate response
        response = llm_provider.generate(prompt)
        
        # Parse output if parser is provided
        if self.output_parser:
            parsed_output = self.output_parser(response)
        else:
            parsed_output = response
        
        # Update context
        result = {
            "step_name": self.name,
            "step_description": self.description,
            "prompt": prompt,
            "raw_output": response,
            "parsed_output": parsed_output,
            "timestamp": datetime.now().isoformat()
        }
        
        return result


class ReasoningChain:
    """
    Chain of reasoning steps for market analysis.
    """
    
    def __init__(self, 
                 name: str,
                 description: str,
                 steps: List[ReasoningStep],
                 llm_provider: LLMProvider):
        """
        Initialize a reasoning chain.
        
        Args:
            name: Chain name
            description: Chain description
            steps: List of reasoning steps
            llm_provider: LLM provider
        """
        self.name = name
        self.description = description
        self.steps = steps
        self.llm_provider = llm_provider
    
    def execute(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the reasoning chain.
        
        Args:
            initial_context: Initial context
            
        Returns:
            Final context with all step results
        """
        context = initial_context.copy()
        context["chain_name"] = self.name
        context["chain_description"] = self.description
        context["steps"] = []
        
        # Execute each step
        for step in self.steps:
            try:
                # Execute step
                step_result = step.execute(self.llm_provider, context)
                
                # Add to context
                context[step.name] = step_result["parsed_output"]
                context["steps"].append(step_result)
                
                logger.info(f"Executed step {step.name}")
            
            except Exception as e:
                logger.error(f"Error executing step {step.name}: {e}")
                
                # Add error to context
                error_result = {
                    "step_name": step.name,
                    "step_description": step.description,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                
                context["steps"].append(error_result)
                break
        
        # Add final timestamp
        context["completion_timestamp"] = datetime.now().isoformat()
        
        return context


# Output parsers
def parse_json_output(output: str) -> Dict[str, Any]:
    """
    Parse JSON output from LLM.
    
    Args:
        output: LLM output
        
    Returns:
        Parsed JSON
    """
    # Extract JSON
    json_match = re.search(r'```json\n(.*?)\n```', output, re.DOTALL)
    
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find any JSON-like structure
        json_match = re.search(r'(\{.*\})', output, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # Use the entire output
            json_str = output
    
    try:
        # Parse JSON
        return json.loads(json_str)
    except json.JSONDecodeError:
        logger.error(f"Error parsing JSON: {json_str}")
        return {"error": "Failed to parse JSON", "raw_output": output}


def parse_list_output(output: str) -> List[str]:
    """
    Parse list output from LLM.
    
    Args:
        output: LLM output
        
    Returns:
        Parsed list
    """
    # Extract list items
    items = []
    
    for line in output.split('\n'):
        # Check for list markers
        if re.match(r'^\s*[-*]\s+', line):
            # Remove list marker
            item = re.sub(r'^\s*[-*]\s+', '', line).strip()
            
            if item:
                items.append(item)
        elif re.match(r'^\s*\d+\.\s+', line):
            # Remove numbered list marker
            item = re.sub(r'^\s*\d+\.\s+', '', line).strip()
            
            if item:
                items.append(item)
    
    return items


# Predefined reasoning chains
def create_market_analysis_chain(llm_provider: LLMProvider) -> ReasoningChain:
    """
    Create a market analysis reasoning chain.
    
    Args:
        llm_provider: LLM provider
        
    Returns:
        Market analysis reasoning chain
    """
    # Step 1: Identify key astrological factors
    identify_factors = ReasoningStep(
        name="identify_factors",
        description="Identify key astrological factors affecting the market",
        prompt_template="""
# Astrological Factor Identification

## Context
- Date: {date}
- Market: {market_symbol}
- Current Price: {current_price}

## Planetary Positions
{planetary_positions}

## Task
Identify the 5 most significant astrological factors currently affecting this market.
For each factor, provide:
1. The planets involved
2. The type of aspect or configuration
3. The market sector or theme it's likely to influence
4. The expected strength of influence (1-10)

Format your response as a JSON array of objects with the following structure:
```json
[
  {
    "factor": "Jupiter-Saturn square",
    "planets": ["Jupiter", "Saturn"],
    "type": "Square",
    "influence": "Banking sector, regulatory changes",
    "strength": 8
  }
]
```
""",
        output_parser=parse_json_output
    )
    
    # Step 2: Analyze historical patterns
    analyze_patterns = ReasoningStep(
        name="analyze_patterns",
        description="Analyze historical patterns for identified factors",
        prompt_template="""
# Historical Pattern Analysis

## Context
- Date: {date}
- Market: {market_symbol}
- Current Price: {current_price}

## Identified Factors
{identify_factors}

## Task
For each of the identified astrological factors, analyze historical patterns when similar configurations occurred.
For each factor, provide:
1. At least 2 historical examples (dates or periods)
2. The market behavior during those periods
3. The similarity to current conditions (1-10)
4. The reliability of this pattern (1-10)

Format your response as a JSON array of objects with the following structure:
```json
[
  {
    "factor": "Jupiter-Saturn square",
    "historical_examples": [
      {
        "period": "May-June 2016",
        "market_behavior": "Market declined 5% then rebounded sharply",
        "similarity": 7,
        "reliability": 8
      }
    ]
  }
]
```
""",
        output_parser=parse_json_output
    )
    
    # Step 3: Forecast market movements
    forecast_movements = ReasoningStep(
        name="forecast_movements",
        description="Forecast market movements based on astrological factors",
        prompt_template="""
# Market Movement Forecast

## Context
- Date: {date}
- Market: {market_symbol}
- Current Price: {current_price}

## Identified Factors
{identify_factors}

## Historical Patterns
{analyze_patterns}

## Task
Based on the identified astrological factors and historical patterns, forecast the likely market movements for the next 30 days.
Provide:
1. The expected price direction (up, down, sideways)
2. Potential percentage change
3. Key dates for trend changes
4. Confidence level (1-10)
5. Risk factors that could alter the forecast

Format your response as a JSON object with the following structure:
```json
{
  "direction": "up",
  "percentage_change": "3-5%",
  "key_dates": [
    {
      "date": "2023-06-15",
      "event": "Mercury-Jupiter trine",
      "expected_impact": "Positive announcement, potential 1-2% gain"
    }
  ],
  "confidence": 7,
  "risk_factors": [
    "Unexpected Saturn retrograde effects could dampen growth",
    "Venus-Mars square on June 10 may increase volatility"
  ]
}
```
""",
        output_parser=parse_json_output
    )
    
    # Step 4: Generate trading recommendations
    generate_recommendations = ReasoningStep(
        name="generate_recommendations",
        description="Generate trading recommendations based on forecast",
        prompt_template="""
# Trading Recommendations

## Context
- Date: {date}
- Market: {market_symbol}
- Current Price: {current_price}

## Market Forecast
{forecast_movements}

## Task
Based on the market forecast, generate specific trading recommendations.
Provide:
1. Entry points (price levels and/or dates)
2. Exit points (price levels and/or dates)
3. Stop-loss levels
4. Position sizing recommendations
5. Alternative scenarios to watch for

Format your response as a JSON object with the following structure:
```json
{
  "primary_recommendation": "Buy",
  "entry_points": [
    {
      "price": "$50.25",
      "date": "Around June 15",
      "condition": "After Mercury-Jupiter trine"
    }
  ],
  "exit_points": [
    {
      "price": "$53.50",
      "date": "Before June 28",
      "condition": "Before Mars-Saturn opposition"
    }
  ],
  "stop_loss": "$48.75",
  "position_sizing": "Moderate (5% of portfolio)",
  "alternative_scenarios": [
    {
      "scenario": "If price breaks below $49.00 before entry",
      "action": "Wait for stabilization around $47.50"
    }
  ]
}
```
""",
        output_parser=parse_json_output
    )
    
    # Step 5: Summarize analysis
    summarize_analysis = ReasoningStep(
        name="summarize_analysis",
        description="Summarize the complete astrological market analysis",
        prompt_template="""
# Astrological Market Analysis Summary

## Context
- Date: {date}
- Market: {market_symbol}
- Current Price: {current_price}

## Key Astrological Factors
{identify_factors}

## Historical Patterns
{analyze_patterns}

## Market Forecast
{forecast_movements}

## Trading Recommendations
{generate_recommendations}

## Task
Provide a comprehensive summary of the astrological market analysis, integrating all the information above.
The summary should be well-structured, clear, and actionable for traders.
Include:
1. Overall market outlook
2. Key astrological influences
3. Specific trading strategy
4. Risk management considerations
5. Important dates to watch

Your summary should be detailed yet concise, focusing on the most significant insights from the analysis.
""",
        output_parser=None  # No parsing needed for the summary
    )
    
    # Create chain
    chain = ReasoningChain(
        name="Astrological Market Analysis",
        description="Comprehensive astrological analysis of market conditions and trading recommendations",
        steps=[
            identify_factors,
            analyze_patterns,
            forecast_movements,
            generate_recommendations,
            summarize_analysis
        ],
        llm_provider=llm_provider
    )
    
    return chain


def create_market_timing_chain(llm_provider: LLMProvider) -> ReasoningChain:
    """
    Create a market timing reasoning chain.
    
    Args:
        llm_provider: LLM provider
        
    Returns:
        Market timing reasoning chain
    """
    # Step 1: Identify current market phase
    identify_phase = ReasoningStep(
        name="identify_phase",
        description="Identify the current market phase based on astrological cycles",
        prompt_template="""
# Market Phase Identification

## Context
- Date: {date}
- Market: {market_symbol}
- Current Price: {current_price}

## Planetary Positions
{planetary_positions}

## Major Cycles
{major_cycles}

## Task
Identify the current market phase based on astrological cycles.
Analyze:
1. The current phase of each major planetary cycle
2. The interaction between different cycles
3. The historical precedent for the current configuration
4. The expected duration of the current phase

Format your response as a JSON object with the following structure:
```json
{
  "current_phase": "Accumulation",
  "cycle_analysis": [
    {
      "cycle": "Jupiter-Saturn cycle",
      "current_phase": "Early expansion",
      "phase_start": "2020-12-21",
      "phase_end": "2023-05-15",
      "historical_precedent": "Similar to 2000-2003 period"
    }
  ],
  "cycle_interactions": [
    "Jupiter-Saturn cycle in early phase while Uranus-Neptune in late phase creates market uncertainty"
  ],
  "expected_duration": "6-8 months",
  "confidence": 7
}
```
""",
        output_parser=parse_json_output
    )
    
    # Step 2: Identify key turning points
    identify_turning_points = ReasoningStep(
        name="identify_turning_points",
        description="Identify key market turning points based on upcoming transits",
        prompt_template="""
# Market Turning Points

## Context
- Date: {date}
- Market: {market_symbol}
- Current Price: {current_price}

## Current Market Phase
{identify_phase}

## Upcoming Transits
{upcoming_transits}

## Task
Identify potential market turning points based on upcoming astrological transits.
For each turning point, provide:
1. The date or date range
2. The astrological configuration causing it
3. The expected market reaction
4. The confidence level (1-10)
5. Historical precedents for similar configurations

Format your response as a JSON array of objects with the following structure:
```json
[
  {
    "date_range": "June 15-20, 2023",
    "configuration": "Mercury-Jupiter trine combined with Mars entering Leo",
    "expected_reaction": "Significant upward movement, potential breakout",
    "confidence": 8,
    "historical_precedents": [
      {
        "date": "August 2019",
        "market_reaction": "5% rally over 3 days"
      }
    ]
  }
]
```
""",
        output_parser=parse_json_output
    )
    
    # Step 3: Generate timing strategy
    generate_timing_strategy = ReasoningStep(
        name="generate_timing_strategy",
        description="Generate a market timing strategy based on turning points",
        prompt_template="""
# Market Timing Strategy

## Context
- Date: {date}
- Market: {market_symbol}
- Current Price: {current_price}

## Current Market Phase
{identify_phase}

## Turning Points
{identify_turning_points}

## Task
Generate a detailed market timing strategy based on the identified market phase and turning points.
Provide:
1. Specific entry and exit dates
2. The rationale for each timing decision
3. Risk management considerations
4. Alternative scenarios to watch for
5. Confirmation indicators to validate the astrological timing

Format your response as a JSON object with the following structure:
```json
{
  "primary_strategy": "Time entry for June 16-17 during Mercury-Jupiter trine",
  "timing_decisions": [
    {
      "action": "Buy",
      "date_range": "June 16-17, 2023",
      "rationale": "Mercury-Jupiter trine historically correlates with positive market announcements",
      "confirmation_indicators": [
        "Increased trading volume",
        "Break above 50-day moving average"
      ]
    }
  ],
  "risk_management": {
    "stop_timing": "Exit if no positive movement by June 20",
    "alternative_entry": "Wait for Mars entering Leo on June 21 if initial entry is missed"
  },
  "alternative_scenarios": [
    {
      "scenario": "If Mercury retrograde begins before entry",
      "adjusted_strategy": "Delay entry until retrograde ends"
    }
  ]
}
```
""",
        output_parser=parse_json_output
    )
    
    # Step 4: Summarize timing analysis
    summarize_timing = ReasoningStep(
        name="summarize_timing",
        description="Summarize the market timing analysis",
        prompt_template="""
# Market Timing Analysis Summary

## Context
- Date: {date}
- Market: {market_symbol}
- Current Price: {current_price}

## Market Phase
{identify_phase}

## Turning Points
{identify_turning_points}

## Timing Strategy
{generate_timing_strategy}

## Task
Provide a comprehensive summary of the market timing analysis, integrating all the information above.
The summary should be well-structured, clear, and actionable for traders.
Include:
1. Current market phase and cycle position
2. Key upcoming turning points
3. Specific timing recommendations
4. Risk management timing considerations
5. Calendar of important dates

Your summary should be detailed yet concise, focusing on the most significant timing insights from the analysis.
""",
        output_parser=None  # No parsing needed for the summary
    )
    
    # Create chain
    chain = ReasoningChain(
        name="Astrological Market Timing",
        description="Precise market timing analysis based on astrological cycles and transits",
        steps=[
            identify_phase,
            identify_turning_points,
            generate_timing_strategy,
            summarize_timing
        ],
        llm_provider=llm_provider
    )
    
    return chain


def create_aspect_analysis_chain(llm_provider: LLMProvider) -> ReasoningChain:
    """
    Create an aspect analysis reasoning chain.
    
    Args:
        llm_provider: LLM provider
        
    Returns:
        Aspect analysis reasoning chain
    """
    # Step 1: Analyze current aspects
    analyze_aspects = ReasoningStep(
        name="analyze_aspects",
        description="Analyze current planetary aspects and their market implications",
        prompt_template="""
# Planetary Aspect Analysis

## Context
- Date: {date}
- Market: {market_symbol}
- Current Price: {current_price}

## Current Aspects
{current_aspects}

## Task
Analyze the current planetary aspects and their implications for the market.
For each significant aspect, provide:
1. The planets involved
2. The type of aspect
3. The exact degree and orb
4. The market sectors likely to be affected
5. The expected influence (positive, negative, neutral)
6. The strength of influence (1-10)

Format your response as a JSON array of objects with the following structure:
```json
[
  {
    "planets": ["Sun", "Jupiter"],
    "aspect": "Trine",
    "degrees": "120°",
    "orb": "2.3°",
    "market_sectors": ["Technology", "Growth stocks"],
    "influence": "Positive",
    "strength": 7,
    "interpretation": "Favorable for growth and expansion, especially in technology sector"
  }
]
```
""",
        output_parser=parse_json_output
    )
    
    # Step 2: Analyze aspect patterns
    analyze_patterns = ReasoningStep(
        name="analyze_patterns",
        description="Analyze aspect patterns and configurations",
        prompt_template="""
# Aspect Pattern Analysis

## Context
- Date: {date}
- Market: {market_symbol}
- Current Price: {current_price}

## Current Aspects
{analyze_aspects}

## Task
Identify and analyze significant aspect patterns and configurations in the current chart.
Look for:
1. Grand trines, T-squares, grand crosses, yods, etc.
2. Stelliums or concentrations in particular signs
3. Void of course moon or other special lunar conditions
4. Mutual receptions or other planetary relationships

For each pattern, provide:
1. The type of pattern
2. The planets and signs involved
3. The market implications
4. Historical examples of similar patterns

Format your response as a JSON array of objects with the following structure:
```json
[
  {
    "pattern": "Grand Trine",
    "planets": ["Jupiter", "Saturn", "Neptune"],
    "signs": ["Taurus", "Virgo", "Capricorn"],
    "market_implications": "Stability in earth-related sectors: real estate, commodities, banking",
    "historical_examples": [
      {
        "date": "September 2018",
        "market_behavior": "Steady growth in real estate sector"
      }
    ]
  }
]
```
""",
        output_parser=parse_json_output
    )
    
    # Step 3: Forecast aspect changes
    forecast_changes = ReasoningStep(
        name="forecast_changes",
        description="Forecast changes in aspects over the next 30 days",
        prompt_template="""
# Aspect Change Forecast

## Context
- Date: {date}
- Market: {market_symbol}
- Current Price: {current_price}

## Current Aspects
{analyze_aspects}

## Aspect Patterns
{analyze_patterns}

## Task
Forecast the significant changes in planetary aspects over the next 30 days and their market implications.
For each significant change, provide:
1. The date of the aspect
2. The planets involved
3. The type of aspect forming or separating
4. The expected market impact
5. Recommended trading actions

Format your response as a JSON array of objects with the following structure:
```json
[
  {
    "date": "2023-06-15",
    "planets": ["Mercury", "Jupiter"],
    "aspect": "Trine",
    "status": "Forming",
    "market_impact": "Positive news likely to boost market sentiment",
    "trading_recommendation": "Consider entering long positions 1-2 days before this aspect"
  }
]
```
""",
        output_parser=parse_json_output
    )
    
    # Step 4: Summarize aspect analysis
    summarize_aspects = ReasoningStep(
        name="summarize_aspects",
        description="Summarize the aspect analysis and its trading implications",
        prompt_template="""
# Aspect Analysis Summary

## Context
- Date: {date}
- Market: {market_symbol}
- Current Price: {current_price}

## Current Aspects
{analyze_aspects}

## Aspect Patterns
{analyze_patterns}

## Upcoming Aspect Changes
{forecast_changes}

## Task
Provide a comprehensive summary of the planetary aspect analysis and its trading implications.
The summary should be well-structured, clear, and actionable for traders.
Include:
1. The most significant current aspects
2. Important aspect patterns
3. Key upcoming aspect changes
4. Specific trading recommendations based on aspects
5. Risk factors to watch

Your summary should be detailed yet concise, focusing on the most significant insights from the aspect analysis.
""",
        output_parser=None  # No parsing needed for the summary
    )
    
    # Create chain
    chain = ReasoningChain(
        name="Astrological Aspect Analysis",
        description="Detailed analysis of planetary aspects and their market implications",
        steps=[
            analyze_aspects,
            analyze_patterns,
            forecast_changes,
            summarize_aspects
        ],
        llm_provider=llm_provider
    )
    
    return chain
