"""
Expert Validation Framework for the Cosmic Market Oracle.

This module implements a framework for domain experts to validate and refine
astrological features, providing a systematic approach to incorporate expert
knowledge into the feature engineering process.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Any, Callable, Union
from datetime import datetime, timedelta
import json
# import logging # Removed
from dataclasses import dataclass, field
import os
from src.utils.logger import get_logger # Added
from pathlib import Path

# Import from the centralized feature_definitions module
from .feature_definitions import FeatureDefinition
from .feature_generator import FeatureGenerator

# Configure logging
# logging.basicConfig( # Removed
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = get_logger(__name__) # Changed


@dataclass
class ExpertFeedback:
    """Feedback from a domain expert on an astrological feature."""
    feature_name: str
    expert_name: str
    relevance_score: float  # 0-10 scale
    reliability_score: float  # 0-10 scale
    market_impact: str  # 'bullish', 'bearish', 'volatile', 'neutral'
    time_frames: List[str]  # e.g., ['daily', 'weekly', 'monthly']
    comments: str
    suggested_improvements: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class ExpertValidationFramework:
    """Framework for domain expert validation of astrological features."""
    
    def __init__(self, feature_generator: FeatureGenerator, 
                validation_data_path: str = "data/expert_validation"):
        """
        Initialize the expert validation framework.
        
        Args:
            feature_generator: Feature generator instance
            validation_data_path: Path to store validation data
        """
        self.feature_generator = feature_generator
        self.validation_data_path = validation_data_path
        
        # Create validation data directory if it doesn't exist
        os.makedirs(validation_data_path, exist_ok=True)
        
        # Initialize feedback database
        self.feedback_database = self._load_feedback_database()
        
        # Initialize expert registry
        self.expert_registry = self._load_expert_registry()
        
        # Initialize validated features
        self.validated_features = {}
    
    def _load_feedback_database(self) -> Dict[str, List[ExpertFeedback]]:
        """
        Load the expert feedback database.
        
        Returns:
            Dictionary of expert feedback by feature name
        """
        feedback_file = os.path.join(self.validation_data_path, "feedback_database.json")
        
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, 'r') as f:
                    feedback_data = json.load(f)
                
                # Convert JSON to ExpertFeedback objects
                feedback_database = {}
                
                for feature_name, feedbacks in feedback_data.items():
                    feedback_database[feature_name] = []
                    
                    for feedback in feedbacks:
                        # Convert timestamp string to datetime
                        timestamp = datetime.fromisoformat(feedback["timestamp"])
                        
                        # Create ExpertFeedback object
                        expert_feedback = ExpertFeedback(
                            feature_name=feedback["feature_name"],
                            expert_name=feedback["expert_name"],
                            relevance_score=feedback["relevance_score"],
                            reliability_score=feedback["reliability_score"],
                            market_impact=feedback["market_impact"],
                            time_frames=feedback["time_frames"],
                            comments=feedback["comments"],
                            suggested_improvements=feedback.get("suggested_improvements"),
                            timestamp=timestamp
                        )
                        
                        feedback_database[feature_name].append(expert_feedback)
                
                return feedback_database
            except Exception as e:
                logger.error(f"Error loading feedback database: {e}")
                return {}
        else:
            return {}
    
    def _save_feedback_database(self):
        """Save the expert feedback database."""
        feedback_file = os.path.join(self.validation_data_path, "feedback_database.json")
        
        try:
            # Convert ExpertFeedback objects to JSON-serializable dict
            feedback_data = {}
            
            for feature_name, feedbacks in self.feedback_database.items():
                feedback_data[feature_name] = []
                
                for feedback in feedbacks:
                    feedback_dict = {
                        "feature_name": feedback.feature_name,
                        "expert_name": feedback.expert_name,
                        "relevance_score": feedback.relevance_score,
                        "reliability_score": feedback.reliability_score,
                        "market_impact": feedback.market_impact,
                        "time_frames": feedback.time_frames,
                        "comments": feedback.comments,
                        "timestamp": feedback.timestamp.isoformat()
                    }
                    
                    if feedback.suggested_improvements:
                        feedback_dict["suggested_improvements"] = feedback.suggested_improvements
                    
                    feedback_data[feature_name].append(feedback_dict)
            
            with open(feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=2)
            
            logger.info(f"Saved feedback database to {feedback_file}")
        except Exception as e:
            logger.error(f"Error saving feedback database: {e}")
    
    def _load_expert_registry(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the expert registry.
        
        Returns:
            Dictionary of expert information by expert name
        """
        registry_file = os.path.join(self.validation_data_path, "expert_registry.json")
        
        if os.path.exists(registry_file):
            try:
                with open(registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading expert registry: {e}")
                return {}
        else:
            return {}
    
    def _save_expert_registry(self):
        """Save the expert registry."""
        registry_file = os.path.join(self.validation_data_path, "expert_registry.json")
        
        try:
            with open(registry_file, 'w') as f:
                json.dump(self.expert_registry, f, indent=2)
            
            logger.info(f"Saved expert registry to {registry_file}")
        except Exception as e:
            logger.error(f"Error saving expert registry: {e}")
    
    def register_expert(self, name: str, credentials: str, expertise_areas: List[str], 
                       contact_info: Optional[str] = None):
        """
        Register a domain expert.
        
        Args:
            name: Expert name
            credentials: Expert credentials
            expertise_areas: Areas of expertise
            contact_info: Contact information (optional)
        """
        self.expert_registry[name] = {
            "credentials": credentials,
            "expertise_areas": expertise_areas,
            "registration_date": datetime.now().isoformat()
        }
        
        if contact_info:
            self.expert_registry[name]["contact_info"] = contact_info
        
        self._save_expert_registry()
        logger.info(f"Registered expert: {name}")
    
    def add_expert_feedback(self, feedback: ExpertFeedback):
        """
        Add expert feedback on a feature.
        
        Args:
            feedback: Expert feedback
        """
        # Check if expert is registered
        if feedback.expert_name not in self.expert_registry:
            logger.warning(f"Expert {feedback.expert_name} is not registered")
        
        # Add feedback to database
        if feedback.feature_name not in self.feedback_database:
            self.feedback_database[feedback.feature_name] = []
        
        self.feedback_database[feedback.feature_name].append(feedback)
        
        # Save feedback database
        self._save_feedback_database()
        
        logger.info(f"Added feedback from {feedback.expert_name} for feature {feedback.feature_name}")
        
        # Update validated features
        self._update_validated_features(feedback.feature_name)
    
    def _update_validated_features(self, feature_name: str):
        """
        Update validated features based on expert feedback.
        
        Args:
            feature_name: Feature name to update
        """
        if feature_name not in self.feedback_database:
            return
        
        feedbacks = self.feedback_database[feature_name]
        
        if not feedbacks:
            return
        
        # Calculate average scores
        relevance_scores = [f.relevance_score for f in feedbacks]
        reliability_scores = [f.reliability_score for f in feedbacks]
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        avg_reliability = sum(reliability_scores) / len(reliability_scores)
        
        # Determine consensus market impact
        impact_counts = {}
        for f in feedbacks:
            impact = f.market_impact
            impact_counts[impact] = impact_counts.get(impact, 0) + 1
        
        consensus_impact = max(impact_counts.items(), key=lambda x: x[1])[0]
        
        # Determine consensus time frames
        time_frame_counts = {}
        for f in feedbacks:
            for tf in f.time_frames:
                time_frame_counts[tf] = time_frame_counts.get(tf, 0) + 1
        
        consensus_time_frames = [tf for tf, count in time_frame_counts.items() 
                               if count > len(feedbacks) / 2]
        
        # Update validated features
        self.validated_features[feature_name] = {
            "relevance_score": avg_relevance,
            "reliability_score": avg_reliability,
            "market_impact": consensus_impact,
            "time_frames": consensus_time_frames,
            "feedback_count": len(feedbacks),
            "last_updated": datetime.now().isoformat()
        }
        
        logger.info(f"Updated validated feature: {feature_name}")
    
    def generate_validation_report(self, feature_name: str) -> Dict[str, Any]:
        """
        Generate a validation report for a feature.
        
        Args:
            feature_name: Feature name
            
        Returns:
            Validation report
        """
        if feature_name not in self.feedback_database:
            return {
                "feature_name": feature_name,
                "status": "not_validated",
                "message": "No expert feedback available for this feature"
            }
        
        feedbacks = self.feedback_database[feature_name]
        
        if not feedbacks:
            return {
                "feature_name": feature_name,
                "status": "not_validated",
                "message": "No expert feedback available for this feature"
            }
        
        # Get validated feature data
        validated_data = self.validated_features.get(feature_name, {})
        
        # Create report
        report = {
            "feature_name": feature_name,
            "status": "validated",
            "feedback_count": len(feedbacks),
            "experts": [f.expert_name for f in feedbacks],
            "relevance_score": validated_data.get("relevance_score"),
            "reliability_score": validated_data.get("reliability_score"),
            "market_impact": validated_data.get("market_impact"),
            "time_frames": validated_data.get("time_frames"),
            "last_updated": validated_data.get("last_updated"),
            "comments": [f.comments for f in feedbacks],
            "suggested_improvements": [f.suggested_improvements for f in feedbacks if f.suggested_improvements]
        }
        
        # Determine validation status
        avg_score = (validated_data.get("relevance_score", 0) + validated_data.get("reliability_score", 0)) / 2
        
        if avg_score >= 7.5:
            report["validation_level"] = "high"
        elif avg_score >= 5.0:
            report["validation_level"] = "medium"
        else:
            report["validation_level"] = "low"
        
        return report
    
    def get_all_validated_features(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all validated features.
        
        Returns:
            Dictionary of validated features
        """
        return self.validated_features
    
    def get_highly_validated_features(self) -> Dict[str, Dict[str, Any]]:
        """
        Get highly validated features.
        
        Returns:
            Dictionary of highly validated features
        """
        return {
            name: data for name, data in self.validated_features.items()
            if (data.get("relevance_score", 0) + data.get("reliability_score", 0)) / 2 >= 7.5
        }
    
    def export_validation_data(self, output_file: str):
        """
        Export validation data to a JSON file.
        
        Args:
            output_file: Output file path
        """
        export_data = {
            "validated_features": self.validated_features,
            "expert_count": len(self.expert_registry),
            "total_feedback_count": sum(len(feedbacks) for feedbacks in self.feedback_database.values()),
            "export_date": datetime.now().isoformat()
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported validation data to {output_file}")
        except Exception as e:
            logger.error(f"Error exporting validation data: {e}")
    
    def refine_feature_based_on_feedback(self, feature_name: str) -> Optional[FeatureDefinition]:
        """
        Refine a feature based on expert feedback.
        
        Args:
            feature_name: Feature name
            
        Returns:
            Refined feature definition or None if feature not found
        """
        if feature_name not in self.feedback_database:
            logger.warning(f"No feedback available for feature {feature_name}")
            return None
        
        feedbacks = self.feedback_database[feature_name]
        
        if not feedbacks:
            logger.warning(f"No feedback available for feature {feature_name}")
            return None
        
        # Get original feature definition
        if feature_name in self.feature_generator.feature_catalog:
            original_def = self.feature_generator.feature_catalog[feature_name]
        else:
            logger.warning(f"Feature {feature_name} not found in feature catalog")
            return None
        
        # Create refined feature definition
        refined_def = copy.deepcopy(original_def)
        refined_def.name = f"{original_def.name}_refined"
        refined_def.description = f"{original_def.description} (refined based on expert feedback)"
        
        # Apply suggested improvements
        for feedback in feedbacks:
            if feedback.suggested_improvements:
                # Log the suggested improvement
                logger.info(f"Applying suggested improvement from {feedback.expert_name}: {feedback.suggested_improvements}")
                
                # In a real system, we would parse and apply the suggested improvements
                # For now, we just note them in the description
                refined_def.description += f"\nSuggested by {feedback.expert_name}: {feedback.suggested_improvements}"
        
        # Add to feature catalog
        self.feature_generator.add_feature_to_catalog(refined_def)
        
        return refined_def


class ExpertValidationUI:
    """User interface for expert validation of astrological features."""
    
    def __init__(self, validation_framework: ExpertValidationFramework):
        """
        Initialize the expert validation UI.
        
        Args:
            validation_framework: Expert validation framework
        """
        self.validation_framework = validation_framework
    
    def generate_feature_validation_form(self, feature_def: FeatureDefinition, 
                                       output_file: Optional[str] = None) -> str:
        """
        Generate a validation form for a feature.
        
        Args:
            feature_def: Feature definition
            output_file: Output file path (optional)
            
        Returns:
            Validation form as HTML string
        """
        # Create HTML form
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Expert Validation Form: {feature_def.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .form-group {{ margin-bottom: 15px; }}
                label {{ display: block; font-weight: bold; }}
                input, select, textarea {{ width: 100%; padding: 8px; }}
                .rating {{ display: flex; }}
                .rating input {{ width: auto; margin-right: 10px; }}
                .submit-btn {{ padding: 10px 20px; background-color: #4CAF50; color: white; border: none; }}
            </style>
        </head>
        <body>
            <h1>Expert Validation Form</h1>
            <h2>Feature: {feature_def.name}</h2>
            <p><strong>Description:</strong> {feature_def.description}</p>
            <p><strong>Type:</strong> {feature_def.feature_type}</p>
            <p><strong>Parameters:</strong> {json.dumps(feature_def.parameters, indent=2)}</p>
            
            <form id="validationForm">
                <input type="hidden" name="feature_name" value="{feature_def.name}">
                
                <div class="form-group">
                    <label for="expert_name">Expert Name:</label>
                    <input type="text" id="expert_name" name="expert_name" required>
                </div>
                
                <div class="form-group">
                    <label for="relevance_score">Relevance Score (0-10):</label>
                    <div class="rating">
                        <input type="range" id="relevance_score" name="relevance_score" min="0" max="10" step="0.5" value="5">
                        <span id="relevance_display">5</span>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="reliability_score">Reliability Score (0-10):</label>
                    <div class="rating">
                        <input type="range" id="reliability_score" name="reliability_score" min="0" max="10" step="0.5" value="5">
                        <span id="reliability_display">5</span>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="market_impact">Market Impact:</label>
                    <select id="market_impact" name="market_impact" required>
                        <option value="bullish">Bullish</option>
                        <option value="bearish">Bearish</option>
                        <option value="volatile">Volatile</option>
                        <option value="neutral" selected>Neutral</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label>Time Frames:</label>
                    <div>
                        <input type="checkbox" id="tf_daily" name="time_frames" value="daily">
                        <label for="tf_daily">Daily</label>
                    </div>
                    <div>
                        <input type="checkbox" id="tf_weekly" name="time_frames" value="weekly">
                        <label for="tf_weekly">Weekly</label>
                    </div>
                    <div>
                        <input type="checkbox" id="tf_monthly" name="time_frames" value="monthly">
                        <label for="tf_monthly">Monthly</label>
                    </div>
                    <div>
                        <input type="checkbox" id="tf_quarterly" name="time_frames" value="quarterly">
                        <label for="tf_quarterly">Quarterly</label>
                    </div>
                    <div>
                        <input type="checkbox" id="tf_yearly" name="time_frames" value="yearly">
                        <label for="tf_yearly">Yearly</label>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="comments">Comments:</label>
                    <textarea id="comments" name="comments" rows="4" required></textarea>
                </div>
                
                <div class="form-group">
                    <label for="suggested_improvements">Suggested Improvements (optional):</label>
                    <textarea id="suggested_improvements" name="suggested_improvements" rows="4"></textarea>
                </div>
                
                <button type="submit" class="submit-btn">Submit Validation</button>
            </form>
            
            <script>
                // Update range display values
                document.getElementById('relevance_score').addEventListener('input', function() {
                    document.getElementById('relevance_display').textContent = this.value;
                });
                
                document.getElementById('reliability_score').addEventListener('input', function() {
                    document.getElementById('reliability_display').textContent = this.value;
                });
                
                // Form submission
                document.getElementById('validationForm').addEventListener('submit', function(e) {
                    e.preventDefault();
                    
                    // Get form data
                    const formData = new FormData(this);
                    const timeFrames = [];
                    
                    // Get selected time frames
                    document.querySelectorAll('input[name="time_frames"]:checked').forEach(function(checkbox) {
                        timeFrames.push(checkbox.value);
                    });
                    
                    // Create feedback object
                    const feedback = {
                        feature_name: formData.get('feature_name'),
                        expert_name: formData.get('expert_name'),
                        relevance_score: parseFloat(formData.get('relevance_score')),
                        reliability_score: parseFloat(formData.get('reliability_score')),
                        market_impact: formData.get('market_impact'),
                        time_frames: timeFrames,
                        comments: formData.get('comments'),
                        suggested_improvements: formData.get('suggested_improvements') || null,
                        timestamp: new Date().toISOString()
                    };
                    
                    // In a real application, this would be sent to a server
                    console.log('Validation submitted:', feedback);
                    alert('Validation submitted successfully!');
                });
            </script>
        </body>
        </html>
        """
        
        # Save to file if output file specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(html)
                
                logger.info(f"Generated validation form saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving validation form: {e}")
        
        return html
    
    def generate_validation_dashboard(self, output_file: Optional[str] = None) -> str:
        """
        Generate a dashboard for viewing validation results.
        
        Args:
            output_file: Output file path (optional)
            
        Returns:
            Dashboard as HTML string
        """
        # Get validated features
        validated_features = self.validation_framework.get_all_validated_features()
        
        # Create HTML dashboard
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Expert Validation Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .high { color: green; }
                .medium { color: orange; }
                .low { color: red; }
            </style>
        </head>
        <body>
            <h1>Expert Validation Dashboard</h1>
            <p>Total validated features: """ + str(len(validated_features)) + """</p>
            
            <h2>Validation Results</h2>
            <table>
                <tr>
                    <th>Feature Name</th>
                    <th>Relevance Score</th>
                    <th>Reliability Score</th>
                    <th>Market Impact</th>
                    <th>Time Frames</th>
                    <th>Feedback Count</th>
                    <th>Last Updated</th>
                </tr>
        """
        
        # Add rows for each validated feature
        for name, data in validated_features.items():
            # Determine validation level
            avg_score = (data.get("relevance_score", 0) + data.get("reliability_score", 0)) / 2
            validation_class = "high" if avg_score >= 7.5 else ("medium" if avg_score >= 5.0 else "low")
            
            html += f"""
                <tr class="{validation_class}">
                    <td>{name}</td>
                    <td>{data.get("relevance_score", 0):.1f}</td>
                    <td>{data.get("reliability_score", 0):.1f}</td>
                    <td>{data.get("market_impact", "unknown")}</td>
                    <td>{", ".join(data.get("time_frames", []))}</td>
                    <td>{data.get("feedback_count", 0)}</td>
                    <td>{data.get("last_updated", "")}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        # Save to file if output file specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(html)
                
                logger.info(f"Generated validation dashboard saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving validation dashboard: {e}")
        
        return html


# Example usage
if __name__ == "__main__":
    import os
    from src.astro_engine.astronomical_calculator import AstronomicalCalculator
    
    # Initialize calculator
    calculator = AstronomicalCalculator()
    
    # Initialize feature generator
    feature_generator = FeatureGenerator(calculator)
    
    # Generate sample feature
    feature_def = feature_generator.generate_feature_definition(
        feature_type="planetary_position",
        transformation_type="sine_transform"
    )
    
    # Add to catalog
    feature_generator.add_feature_to_catalog(feature_def)
    
    # Initialize expert validation framework
    validation_framework = ExpertValidationFramework(feature_generator)
    
    # Register expert
    validation_framework.register_expert(
        name="Dr. Jane Smith",
        credentials="Ph.D. in Financial Astrology, 15 years experience",
        expertise_areas=["Vedic Astrology", "Market Cycles", "Planetary Economics"]
    )
    
    # Create sample feedback
    feedback = ExpertFeedback(
        feature_name=feature_def.name,
        expert_name="Dr. Jane Smith",
        relevance_score=8.5,
        reliability_score=7.0,
        market_impact="bullish",
        time_frames=["daily", "weekly"],
        comments="This feature shows strong correlation with market momentum phases.",
        suggested_improvements="Consider adding Jupiter-Saturn cycle phase as a modulating factor."
    )
    
    # Add feedback
    validation_framework.add_expert_feedback(feedback)
    
    # Generate validation report
    report = validation_framework.generate_validation_report(feature_def.name)
    print(f"Validation report for {feature_def.name}:")
    print(json.dumps(report, indent=2))
    
    # Initialize UI
    validation_ui = ExpertValidationUI(validation_framework)
    
    # Generate validation form
    output_dir = "data/expert_validation"
    os.makedirs(output_dir, exist_ok=True)
    
    form_file = os.path.join(output_dir, f"{feature_def.name}_validation_form.html")
    validation_ui.generate_feature_validation_form(feature_def, form_file)
    
    # Generate dashboard
    dashboard_file = os.path.join(output_dir, "validation_dashboard.html")
    validation_ui.generate_validation_dashboard(dashboard_file)
    
    print(f"Generated validation form: {form_file}")
    print(f"Generated validation dashboard: {dashboard_file}")
