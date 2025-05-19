#!/usr/bin/env python
# Cosmic Market Oracle - Output Formats

"""
Specialized Output Formats for the Cosmic Market Oracle.

This module implements various output formats for different stakeholder needs,
from technical traders to executive summaries.
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

from src.utils.logger import setup_logger

# Configure logging
logger = setup_logger("output_formats")


class OutputFormatter:
    """
    Formatter for LLM outputs to different formats.
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize the output formatter.
        
        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
    
    def format_as_json(self, data: Dict[str, Any]) -> str:
        """
        Format data as JSON.
        
        Args:
            data: Data to format
            
        Returns:
            Formatted JSON string
        """
        return json.dumps(data, indent=2)
    
    def format_as_markdown(self, data: Dict[str, Any], template: str) -> str:
        """
        Format data as Markdown.
        
        Args:
            data: Data to format
            template: Markdown template
            
        Returns:
            Formatted Markdown string
        """
        try:
            return template.format(**data)
        except KeyError as e:
            logger.error(f"Missing key in data: {e}")
            return f"Error formatting markdown: {e}"
    
    def format_as_html(self, data: Dict[str, Any], template: str) -> str:
        """
        Format data as HTML.
        
        Args:
            data: Data to format
            template: HTML template
            
        Returns:
            Formatted HTML string
        """
        try:
            return template.format(**data)
        except KeyError as e:
            logger.error(f"Missing key in data: {e}")
            return f"<p>Error formatting HTML: {e}</p>"
    
    def format_as_csv(self, data: List[Dict[str, Any]]) -> str:
        """
        Format data as CSV.
        
        Args:
            data: List of dictionaries to format
            
        Returns:
            Formatted CSV string
        """
        try:
            df = pd.DataFrame(data)
            return df.to_csv(index=False)
        except Exception as e:
            logger.error(f"Error formatting CSV: {e}")
            return f"Error formatting CSV: {e}"
    
    def format_as_chart(self, 
                       data: Dict[str, Any], 
                       chart_type: str = "line",
                       title: str = "",
                       save_path: Optional[str] = None) -> str:
        """
        Format data as chart.
        
        Args:
            data: Data to format
            chart_type: Type of chart (line, bar, scatter)
            title: Chart title
            save_path: Path to save chart
            
        Returns:
            Path to saved chart
        """
        try:
            plt.figure(figsize=(10, 6))
            
            if chart_type == "line":
                for key, values in data.items():
                    if isinstance(values, list) and all(isinstance(v, (int, float)) for v in values):
                        plt.plot(values, label=key)
            
            elif chart_type == "bar":
                plt.bar(list(data.keys()), list(data.values()))
            
            elif chart_type == "scatter":
                for key, values in data.items():
                    if isinstance(values, list) and len(values) == 2:
                        plt.scatter(values[0], values[1], label=key)
            
            plt.title(title)
            plt.legend()
            plt.tight_layout()
            
            # Save chart
            if not save_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = self.output_dir / f"chart_{timestamp}.png"
            
            plt.savefig(save_path)
            plt.close()
            
            return str(save_path)
        
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return f"Error creating chart: {e}"
    
    def save_output(self, 
                   data: Any, 
                   filename: str, 
                   format_type: str = "json") -> str:
        """
        Save output to file.
        
        Args:
            data: Data to save
            filename: Filename
            format_type: Format type (json, md, html, csv)
            
        Returns:
            Path to saved file
        """
        try:
            # Create full path
            if not filename.endswith(f".{format_type}"):
                filename = f"{filename}.{format_type}"
            
            file_path = self.output_dir / filename
            
            # Save based on format
            if format_type == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            
            elif format_type in ["md", "markdown"]:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(data)
            
            elif format_type == "html":
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(data)
            
            elif format_type == "csv":
                if isinstance(data, str):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(data)
                else:
                    df = pd.DataFrame(data)
                    df.to_csv(file_path, index=False)
            
            else:
                logger.error(f"Unsupported format type: {format_type}")
                return f"Error: Unsupported format type: {format_type}"
            
            logger.info(f"Saved output to {file_path}")
            
            return str(file_path)
        
        except Exception as e:
            logger.error(f"Error saving output: {e}")
            return f"Error saving output: {e}"


# Predefined templates
class TemplateLibrary:
    """
    Library of output templates.
    """
    
    @staticmethod
    def get_trader_report_template() -> str:
        """
        Get template for trader report.
        
        Returns:
            Markdown template
        """
        return """
# Astrological Market Analysis: {market_symbol}

**Date:** {date}
**Current Price:** {current_price}

## Key Astrological Factors

{key_factors}

## Market Forecast

**Direction:** {forecast_direction}
**Expected Change:** {forecast_change}
**Confidence:** {forecast_confidence}/10

### Key Dates

{key_dates}

## Trading Recommendations

**Primary Strategy:** {primary_strategy}

### Entry Points

{entry_points}

### Exit Points

{exit_points}

### Stop Loss

{stop_loss}

## Risk Factors

{risk_factors}

---
*Generated by Cosmic Market Oracle on {generation_timestamp}*
"""
    
    @staticmethod
    def get_executive_summary_template() -> str:
        """
        Get template for executive summary.
        
        Returns:
            Markdown template
        """
        return """
# Executive Summary: {market_symbol}

**Date:** {date}

## Market Outlook

{market_outlook}

## Key Insights

{key_insights}

## Recommended Actions

{recommended_actions}

## Risk Assessment

{risk_assessment}

---
*Generated by Cosmic Market Oracle on {generation_timestamp}*
"""
    
    @staticmethod
    def get_technical_analysis_template() -> str:
        """
        Get template for technical analysis.
        
        Returns:
            Markdown template
        """
        return """
# Technical Analysis with Astrological Overlay: {market_symbol}

**Date:** {date}
**Current Price:** {current_price}

## Technical Indicators

{technical_indicators}

## Astrological Correlations

{astrological_correlations}

## Support and Resistance Levels

**Astrological Support Levels:**
{astrological_support}

**Astrological Resistance Levels:**
{astrological_resistance}

**Technical Support Levels:**
{technical_support}

**Technical Resistance Levels:**
{technical_resistance}

## Confluence Analysis

{confluence_analysis}

## Trading Strategy

{trading_strategy}

---
*Generated by Cosmic Market Oracle on {generation_timestamp}*
"""
    
    @staticmethod
    def get_html_dashboard_template() -> str:
        """
        Get template for HTML dashboard.
        
        Returns:
            HTML template
        """
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Cosmic Market Oracle: {market_symbol}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ background-color: #f9f9f9; padding: 15px; margin: 15px 0; border-radius: 5px; }}
        .forecast {{ display: flex; justify-content: space-between; }}
        .forecast-card {{ background-color: white; padding: 15px; border-radius: 5px; width: 30%; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .neutral {{ color: orange; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #2c3e50; color: white; }}
        .footer {{ text-align: center; margin-top: 30px; font-size: 12px; color: #777; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Cosmic Market Oracle</h1>
            <h2>{market_symbol} - {date}</h2>
            <p>Current Price: {current_price}</p>
        </div>
        
        <div class="section">
            <h2>Market Forecast</h2>
            <div class="forecast">
                <div class="forecast-card">
                    <h3>Short-term (1-7 days)</h3>
                    <p class="{short_term_class}">{short_term_forecast}</p>
                </div>
                <div class="forecast-card">
                    <h3>Medium-term (1-4 weeks)</h3>
                    <p class="{medium_term_class}">{medium_term_forecast}</p>
                </div>
                <div class="forecast-card">
                    <h3>Long-term (1-3 months)</h3>
                    <p class="{long_term_class}">{long_term_forecast}</p>
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Key Astrological Factors</h2>
            <table>
                <tr>
                    <th>Factor</th>
                    <th>Influence</th>
                    <th>Strength</th>
                </tr>
                {astrological_factors_rows}
            </table>
        </div>
        
        <div class="section">
            <h2>Key Dates</h2>
            <table>
                <tr>
                    <th>Date</th>
                    <th>Event</th>
                    <th>Expected Impact</th>
                </tr>
                {key_dates_rows}
            </table>
        </div>
        
        <div class="section">
            <h2>Trading Recommendations</h2>
            {trading_recommendations}
        </div>
        
        <div class="footer">
            Generated by Cosmic Market Oracle on {generation_timestamp}
        </div>
    </div>
</body>
</html>
"""
