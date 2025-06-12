#!/usr/bin/env python
# Cosmic Market Oracle - Report Generation

"""
Report Generation Module for the Cosmic Market Oracle.

This module implements automated report generation with natural language summaries
for astrological market analysis.
"""

import os
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.llm_integration.retrieval_augmentation import LLMProvider
from src.llm_integration.output_formats import OutputFormatter, TemplateLibrary
from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("report_generation")


class ReportGenerator:
    """
    Generator for astrological market analysis reports.
    """
    
    from .constants import DEFAULT_REPORT_DIR

    def __init__(self, 
                 llm_provider: LLMProvider,
                 output_formatter: OutputFormatter,
                 output_dir: str = DEFAULT_REPORT_DIR):
        """
        Initialize the report generator.
        
        Args:
            llm_provider: LLM provider
            output_formatter: Output formatter
            output_dir: Directory to save reports
        """
        self.llm_provider = llm_provider
        self.output_formatter = output_formatter
        self.output_dir = Path(output_dir)
        self.template_library = TemplateLibrary()
        
        # Create directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_report(self, 
                       report_type: str,
                       market_data: Dict[str, Any],
                       planetary_data: Dict[str, Any],
                       analysis_results: Optional[Dict[str, Any]] = None,
                       additional_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a report.
        
        Args:
            report_type: Type of report (trader, executive, technical)
            market_data: Market data
            planetary_data: Planetary data
            analysis_results: Analysis results
            additional_context: Additional context
            
        Returns:
            Generated report
        """
        # Prepare context
        context = {
            "report_type": report_type,
            "market_data": market_data,
            "planetary_data": planetary_data,
            "analysis_results": analysis_results or {},
            "additional_context": additional_context or {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Generate report content
        report_content = self._generate_report_content(context)
        
        # Format report
        formatted_report = self._format_report(report_content, report_type)
        
        # Save report
        report_path = self._save_report(formatted_report, report_type, market_data.get("symbol", "unknown"))
        
        # Prepare result
        result = {
            "report_type": report_type,
            "content": formatted_report,
            "path": report_path,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _generate_report_content(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate report content.
        
        Args:
            context: Report context
            
        Returns:
            Report content
        """
        # Create prompt based on report type
        prompt = self._create_report_prompt(context)
        
        # Generate content
        response = self.llm_provider.generate(prompt)
        
        # Parse content
        try:
            # Try to parse as JSON
            content = json.loads(response)
        except json.JSONDecodeError:
            # Use text as is
            content = {
                "report_text": response,
                "sections": [{"title": "Report", "content": response}]
            }
        
        return content
    
    def _create_report_prompt(self, context: Dict[str, Any]) -> str:
        """
        Create report prompt.
        
        Args:
            context: Report context
            
        Returns:
            Prompt
        """
        report_type = context["report_type"]
        market_data = context["market_data"]
        planetary_data = context["planetary_data"]
        analysis_results = context["analysis_results"]
        
        # Format market data
        market_data_str = json.dumps(market_data, indent=2)
        
        # Format planetary data
        planetary_data_str = json.dumps(planetary_data, indent=2)
        
        # Format analysis results
        analysis_results_str = json.dumps(analysis_results, indent=2)
        
        # Create prompt based on report type
        if report_type == "trader":
            prompt = f"""
# Trader Report Generation

## Market Data
```json
{market_data_str}
```

## Planetary Data
```json
{planetary_data_str}
```

## Analysis Results
```json
{analysis_results_str}
```

## Task
Generate a comprehensive trading report based on astrological analysis.
The report should include:
1. Key astrological factors affecting the market
2. Market forecast (direction, expected change, confidence)
3. Key dates to watch
4. Trading recommendations (entry points, exit points, stop loss)
5. Risk factors

Format your response as a JSON object with the following structure:
```json
{{
  "title": "Astrological Trading Report: [Market Symbol]",
  "date": "[Current Date]",
  "market_symbol": "[Market Symbol]",
  "current_price": "[Current Price]",
  "key_factors": "[Bulleted list of key astrological factors]",
  "forecast_direction": "[Up/Down/Sideways]",
  "forecast_change": "[Expected percentage change]",
  "forecast_confidence": "[1-10]",
  "key_dates": "[Bulleted list of key dates with events]",
  "primary_strategy": "[Brief strategy description]",
  "entry_points": "[Bulleted list of entry points]",
  "exit_points": "[Bulleted list of exit points]",
  "stop_loss": "[Stop loss level]",
  "risk_factors": "[Bulleted list of risk factors]"
}}
```
"""
        
        elif report_type == "executive":
            prompt = f"""
# Executive Summary Generation

## Market Data
```json
{market_data_str}
```

## Planetary Data
```json
{planetary_data_str}
```

## Analysis Results
```json
{analysis_results_str}
```

## Task
Generate an executive summary of astrological market analysis.
The summary should be concise, focused on key insights, and actionable.
Include:
1. Overall market outlook
2. Key insights from astrological analysis
3. Recommended actions
4. Risk assessment

Format your response as a JSON object with the following structure:
```json
{{
  "title": "Executive Summary: [Market Symbol]",
  "date": "[Current Date]",
  "market_symbol": "[Market Symbol]",
  "market_outlook": "[Brief paragraph on overall market outlook]",
  "key_insights": "[3-5 bullet points of key insights]",
  "recommended_actions": "[3-5 bullet points of recommended actions]",
  "risk_assessment": "[Brief paragraph on risks]"
}}
```
"""
        
        elif report_type == "technical":
            prompt = f"""
# Technical Analysis with Astrological Overlay

## Market Data
```json
{market_data_str}
```

## Planetary Data
```json
{planetary_data_str}
```

## Analysis Results
```json
{analysis_results_str}
```

## Task
Generate a technical analysis report with astrological overlay.
The report should combine traditional technical analysis with astrological insights.
Include:
1. Technical indicators and their current readings
2. Astrological correlations with technical patterns
3. Support and resistance levels (both technical and astrological)
4. Confluence analysis where technical and astrological factors align
5. Trading strategy based on the combined analysis

Format your response as a JSON object with the following structure:
```json
{{
  "title": "Technical Analysis with Astrological Overlay: [Market Symbol]",
  "date": "[Current Date]",
  "market_symbol": "[Market Symbol]",
  "current_price": "[Current Price]",
  "technical_indicators": "[List of technical indicators with readings]",
  "astrological_correlations": "[List of correlations between astrological factors and technical patterns]",
  "astrological_support": "[List of support levels based on astrological factors]",
  "astrological_resistance": "[List of resistance levels based on astrological factors]",
  "technical_support": "[List of support levels based on technical analysis]",
  "technical_resistance": "[List of resistance levels based on technical analysis]",
  "confluence_analysis": "[Analysis of areas where technical and astrological factors align]",
  "trading_strategy": "[Detailed trading strategy based on the combined analysis]"
}}
```
"""
        
        else:
            # Default to general report
            prompt = f"""
# Astrological Market Analysis Report

## Market Data
```json
{market_data_str}
```

## Planetary Data
```json
{planetary_data_str}
```

## Analysis Results
```json
{analysis_results_str}
```

## Task
Generate a general astrological market analysis report.
The report should provide a comprehensive analysis of the market from an astrological perspective.
Include any relevant information that would be useful for understanding the market's current and future state.

Format your response as a JSON object with appropriate fields for the report content.
"""
        
        return prompt
    
    def _format_report(self, content: Dict[str, Any], report_type: str) -> str:
        """
        Format report content.
        
        Args:
            content: Report content
            report_type: Report type
            
        Returns:
            Formatted report
        """
        # Get template based on report type
        if report_type == "trader":
            template = self.template_library.get_trader_report_template()
        elif report_type == "executive":
            template = self.template_library.get_executive_summary_template()
        elif report_type == "technical":
            template = self.template_library.get_technical_analysis_template()
        else:
            # Default to trader report
            template = self.template_library.get_trader_report_template()
        
        # Add generation timestamp
        content["generation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format report
        try:
            return self.output_formatter.format_as_markdown(content, template)
        except Exception as e:
            logger.error(f"Error formatting report: {e}")
            return f"Error formatting report: {e}\n\nRaw content:\n{json.dumps(content, indent=2)}"
    
    def _save_report(self, report: str, report_type: str, market_symbol: str) -> str:
        """
        Save report to file.
        
        Args:
            report: Formatted report
            report_type: Report type
            market_symbol: Market symbol
            
        Returns:
            Path to saved report
        """
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_{market_symbol}_{timestamp}.md"
        
        # Save report
        report_path = self.output_dir / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Saved report to {report_path}")
        
        return str(report_path)
    
    def generate_periodic_report(self, 
                               period: str,
                               market_data: Dict[str, Any],
                               planetary_data: Dict[str, Any],
                               report_type: str = "trader") -> Dict[str, Any]:
        """
        Generate a periodic report.
        
        Args:
            period: Report period (daily, weekly, monthly)
            market_data: Market data
            planetary_data: Planetary data
            report_type: Report type
            
        Returns:
            Generated report
        """
        # Add period to context
        additional_context = {
            "period": period,
            "report_title": f"{period.capitalize()} {report_type.capitalize()} Report"
        }
        
        # Generate report
        return self.generate_report(
            report_type=report_type,
            market_data=market_data,
            planetary_data=planetary_data,
            additional_context=additional_context
        )
    
    def generate_multi_market_report(self, 
                                   markets: List[Dict[str, Any]],
                                   planetary_data: Dict[str, Any],
                                   report_type: str = "executive") -> Dict[str, Any]:
        """
        Generate a report covering multiple markets.
        
        Args:
            markets: List of market data dictionaries
            planetary_data: Planetary data
            report_type: Report type
            
        Returns:
            Generated report
        """
        # Create combined market data
        combined_market_data = {
            "markets": markets,
            "num_markets": len(markets),
            "symbols": [m.get("symbol", "unknown") for m in markets]
        }
        
        # Add multi-market context
        additional_context = {
            "multi_market": True,
            "report_title": f"Multi-Market {report_type.capitalize()} Report"
        }
        
        # Generate report
        return self.generate_report(
            report_type=report_type,
            market_data=combined_market_data,
            planetary_data=planetary_data,
            additional_context=additional_context
        )
