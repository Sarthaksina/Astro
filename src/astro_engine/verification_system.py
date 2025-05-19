# Cosmic Market Oracle - Verification System Module

"""
This module provides verification functionality for the Vedic Astronomical Computation Engine
by comparing calculated planetary positions with historical almanac data.

It helps ensure the accuracy and reliability of the astrological calculations by
validating against trusted historical sources.
"""

import os
import json
import datetime
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .planetary_positions import PlanetaryCalculator, SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, RAHU, KETU


class VerificationSystem:
    """Verification system for the Vedic Astronomical Computation Engine."""

    def __init__(self, planetary_calculator: PlanetaryCalculator, reference_data_path: str = None):
        """
        Initialize the verification system.
        
        Args:
            planetary_calculator: Instance of PlanetaryCalculator for calculations
            reference_data_path: Path to reference data files (almanacs)
        """
        self.calculator = planetary_calculator
        
        # Set default reference data path if not provided
        if reference_data_path is None:
            # Determine the project root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "../.."))
            self.reference_data_path = os.path.join(project_root, "data", "reference", "almanacs")
        else:
            self.reference_data_path = reference_data_path
            
        # Dictionary to map planet names to their IDs
        self.planet_map = {
            "sun": SUN,
            "moon": MOON,
            "mercury": MERCURY,
            "venus": VENUS,
            "mars": MARS,
            "jupiter": JUPITER,
            "saturn": SATURN,
            "rahu": RAHU,
            "ketu": KETU
        }
        
        # Load reference data
        self.reference_data = self._load_reference_data()
        
    def _load_reference_data(self) -> Dict:
        """
        Load reference data from almanac files.
        
        Returns:
            Dictionary containing reference data
        """
        reference_data = {}
        
        # Load historical positions from JSON file
        historical_positions_file = os.path.join(self.reference_data_path, "historical_positions.json")
        if os.path.exists(historical_positions_file):
            with open(historical_positions_file, 'r') as f:
                data = json.load(f)
                reference_data["historical_positions"] = data["data"]
        
        return reference_data
    
    def verify_planetary_positions(self, tolerance: float = 1.0) -> Dict:
        """
        Verify planetary position calculations against historical almanac data.
        
        Args:
            tolerance: Maximum allowed difference in degrees (default: 1.0)
            
        Returns:
            Dictionary with verification results
        """
        if not self.reference_data.get("historical_positions"):
            return {
                "success": False,
                "error": "No reference data available for verification"
            }
        
        results = {
            "success": True,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "details": []
        }
        
        # Verify each date in the reference data
        for entry in self.reference_data["historical_positions"]:
            date = entry["date"]
            date_results = {
                "date": date,
                "planets": []
            }
            
            # Get calculated positions for this date
            calculated_positions = self.calculator.get_all_planets(date)
            
            # Compare each planet's position
            for planet_name, reference_data in entry.items():
                if planet_name == "date":
                    continue
                
                planet_id = self.planet_map.get(planet_name)
                if planet_id is None:
                    continue
                
                # Skip if planet not in calculated positions
                if planet_id not in calculated_positions:
                    continue
                
                calculated = calculated_positions[planet_id]
                
                # Compare longitude
                if "longitude" in reference_data and "longitude" in calculated:
                    results["total_tests"] += 1
                    
                    ref_longitude = reference_data["longitude"]
                    calc_longitude = calculated["longitude"]
                    
                    # Handle cases near 0/360 boundary
                    diff = min(
                        abs(ref_longitude - calc_longitude),
                        abs(ref_longitude - calc_longitude + 360),
                        abs(ref_longitude - calc_longitude - 360)
                    )
                    
                    passed = diff <= tolerance
                    if passed:
                        results["passed_tests"] += 1
                    else:
                        results["failed_tests"] += 1
                    
                    planet_result = {
                        "planet": planet_name,
                        "property": "longitude",
                        "reference": ref_longitude,
                        "calculated": calc_longitude,
                        "difference": diff,
                        "passed": passed
                    }
                    date_results["planets"].append(planet_result)
                
                # Compare retrograde status if available
                if "is_retrograde" in reference_data and "is_retrograde" in calculated:
                    results["total_tests"] += 1
                    
                    ref_retrograde = reference_data["is_retrograde"]
                    calc_retrograde = calculated["is_retrograde"]
                    
                    passed = ref_retrograde == calc_retrograde
                    if passed:
                        results["passed_tests"] += 1
                    else:
                        results["failed_tests"] += 1
                    
                    planet_result = {
                        "planet": planet_name,
                        "property": "is_retrograde",
                        "reference": ref_retrograde,
                        "calculated": calc_retrograde,
                        "passed": passed
                    }
                    date_results["planets"].append(planet_result)
                
                # Compare nakshatra if available
                if "nakshatra" in reference_data and "nakshatra" in calculated:
                    results["total_tests"] += 1
                    
                    ref_nakshatra = reference_data["nakshatra"]
                    calc_nakshatra = calculated["nakshatra"]
                    
                    passed = ref_nakshatra == calc_nakshatra
                    if passed:
                        results["passed_tests"] += 1
                    else:
                        results["failed_tests"] += 1
                    
                    planet_result = {
                        "planet": planet_name,
                        "property": "nakshatra",
                        "reference": ref_nakshatra,
                        "calculated": calc_nakshatra,
                        "passed": passed
                    }
                    date_results["planets"].append(planet_result)
            
            results["details"].append(date_results)
        
        # Calculate accuracy percentage
        if results["total_tests"] > 0:
            results["accuracy"] = (results["passed_tests"] / results["total_tests"]) * 100
        else:
            results["accuracy"] = 0
            
        # Update overall success flag
        results["success"] = results["accuracy"] >= 95  # Require at least 95% accuracy
        
        return results
    
    def generate_verification_report(self, results: Dict = None, output_path: str = None) -> str:
        """
        Generate a detailed verification report.
        
        Args:
            results: Verification results (if None, will run verification)
            output_path: Path to save the report (if None, will not save)
            
        Returns:
            Path to the saved report or report content as string
        """
        if results is None:
            results = self.verify_planetary_positions()
        
        # Create report content
        report = []
        report.append("# Vedic Astronomical Computation Engine Verification Report")
        report.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overall results
        report.append("## Summary")
        report.append(f"- Overall Status: {'PASSED' if results['success'] else 'FAILED'}")
        report.append(f"- Accuracy: {results['accuracy']:.2f}%")
        report.append(f"- Total Tests: {results['total_tests']}")
        report.append(f"- Passed Tests: {results['passed_tests']}")
        report.append(f"- Failed Tests: {results['failed_tests']}\n")
        
        # Detailed results by date
        report.append("## Detailed Results")
        
        for date_result in results["details"]:
            report.append(f"### Date: {date_result['date']}")
            
            # Create a table for each date
            report.append("| Planet | Property | Reference | Calculated | Difference | Status |")
            report.append("|--------|----------|-----------|------------|------------|--------|")
            
            for planet_result in date_result["planets"]:
                planet = planet_result["planet"].capitalize()
                property_name = planet_result["property"]
                reference = planet_result["reference"]
                calculated = planet_result["calculated"]
                
                # Format difference if present
                if "difference" in planet_result:
                    difference = f"{planet_result['difference']:.4f}°"
                else:
                    difference = "N/A"
                
                status = "✅" if planet_result["passed"] else "❌"
                
                report.append(f"| {planet} | {property_name} | {reference} | {calculated} | {difference} | {status} |")
            
            report.append("")  # Empty line after each date
        
        # Recommendations section
        report.append("## Recommendations")
        
        if results["success"]:
            report.append("- The Vedic Astronomical Computation Engine is performing with high accuracy.")
            report.append("- Continue to monitor for any deviations in future updates.")
        else:
            report.append("- The engine requires calibration to improve accuracy.")
            report.append("- Focus on planets with highest error margins first.")
            
            # Identify problematic planets
            problem_planets = {}
            for date_result in results["details"]:
                for planet_result in date_result["planets"]:
                    if not planet_result["passed"]:
                        planet = planet_result["planet"]
                        if planet not in problem_planets:
                            problem_planets[planet] = 0
                        problem_planets[planet] += 1
            
            if problem_planets:
                report.append("\nPlanets requiring attention:")
                for planet, count in sorted(problem_planets.items(), key=lambda x: x[1], reverse=True):
                    report.append(f"- {planet.capitalize()}: {count} failed tests")
        
        # Join report lines
        report_content = "\n".join(report)
        
        # Save report if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_content)
            return output_path
        
        return report_content
    
    def visualize_accuracy(self, results: Dict = None, output_path: str = None) -> str:
        """
        Create visualizations of verification accuracy.
        
        Args:
            results: Verification results (if None, will run verification)
            output_path: Path to save the visualization (if None, will not save)
            
        Returns:
            Path to the saved visualization or None
        """
        if results is None:
            results = self.verify_planetary_positions()
        
        # Extract data for visualization
        planets = set()
        planet_errors = {}
        
        for date_result in results["details"]:
            for planet_result in date_result["planets"]:
                if planet_result["property"] == "longitude":
                    planet = planet_result["planet"]
                    planets.add(planet)
                    
                    if planet not in planet_errors:
                        planet_errors[planet] = []
                    
                    if "difference" in planet_result:
                        planet_errors[planet].append(planet_result["difference"])
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Box plot of errors by planet
        plt.subplot(2, 1, 1)
        box_data = [planet_errors[planet] for planet in sorted(planets) if planet in planet_errors]
        plt.boxplot(box_data, labels=[p.capitalize() for p in sorted(planets) if p in planet_errors])
        plt.ylabel("Error (degrees)")
        plt.title("Planetary Position Calculation Errors")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Overall accuracy pie chart
        plt.subplot(2, 2, 3)
        plt.pie(
            [results["passed_tests"], results["failed_tests"]], 
            labels=["Passed", "Failed"],
            autopct='%1.1f%%',
            colors=['#4CAF50', '#F44336'],
            startangle=90
        )
        plt.title("Overall Test Results")
        
        # Average error by planet bar chart
        plt.subplot(2, 2, 4)
        avg_errors = {planet: np.mean(errors) for planet, errors in planet_errors.items()}
        planets_sorted = sorted(avg_errors.keys(), key=lambda p: avg_errors[p])
        
        plt.bar(
            [p.capitalize() for p in planets_sorted],
            [avg_errors[p] for p in planets_sorted],
            color='#2196F3'
        )
        plt.ylabel("Avg. Error (degrees)")
        plt.title("Average Error by Planet")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save figure if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
            return output_path
        
        plt.close()
        return None
    
    def add_reference_data(self, data: Dict, file_name: str = "custom_reference_data.json") -> bool:
        """
        Add custom reference data for verification.
        
        Args:
            data: Dictionary containing reference data
            file_name: Name of the file to save the data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate data structure
            if "data" not in data:
                data = {"data": data}
            
            # Ensure directory exists
            os.makedirs(self.reference_data_path, exist_ok=True)
            
            # Save data to file
            file_path = os.path.join(self.reference_data_path, file_name)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            # Reload reference data
            self.reference_data = self._load_reference_data()
            
            return True
        except Exception as e:
            print(f"Error adding reference data: {e}")
            return False
    
    def compare_with_external_almanac(self, almanac_data: Dict, tolerance: float = 1.0) -> Dict:
        """
        Compare calculations with external almanac data not stored in the reference files.
        
        Args:
            almanac_data: Dictionary with almanac data in the same format as reference data
            tolerance: Maximum allowed difference in degrees
            
        Returns:
            Dictionary with comparison results
        """
        results = {
            "success": True,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "details": []
        }
        
        # Process each date in the almanac data
        for entry in almanac_data:
            date = entry["date"]
            date_results = {
                "date": date,
                "planets": []
            }
            
            # Get calculated positions for this date
            calculated_positions = self.calculator.get_all_planets(date)
            
            # Compare each planet's position
            for planet_name, reference_data in entry.items():
                if planet_name == "date":
                    continue
                
                planet_id = self.planet_map.get(planet_name)
                if planet_id is None:
                    continue
                
                # Skip if planet not in calculated positions
                if planet_id not in calculated_positions:
                    continue
                
                calculated = calculated_positions[planet_id]
                
                # Compare longitude
                if "longitude" in reference_data and "longitude" in calculated:
                    results["total_tests"] += 1
                    
                    ref_longitude = reference_data["longitude"]
                    calc_longitude = calculated["longitude"]
                    
                    # Handle cases near 0/360 boundary
                    diff = min(
                        abs(ref_longitude - calc_longitude),
                        abs(ref_longitude - calc_longitude + 360),
                        abs(ref_longitude - calc_longitude - 360)
                    )
                    
                    passed = diff <= tolerance
                    if passed:
                        results["passed_tests"] += 1
                    else:
                        results["failed_tests"] += 1
                    
                    planet_result = {
                        "planet": planet_name,
                        "property": "longitude",
                        "reference": ref_longitude,
                        "calculated": calc_longitude,
                        "difference": diff,
                        "passed": passed
                    }
                    date_results["planets"].append(planet_result)
            
            results["details"].append(date_results)
        
        # Calculate accuracy percentage
        if results["total_tests"] > 0:
            results["accuracy"] = (results["passed_tests"] / results["total_tests"]) * 100
        else:
            results["accuracy"] = 0
            
        # Update overall success flag
        results["success"] = results["accuracy"] >= 95  # Require at least 95% accuracy
        
        return results


def verify_engine_accuracy(calculator: PlanetaryCalculator = None, 
                          reference_data_path: str = None,
                          generate_report: bool = True,
                          report_path: str = None) -> Dict:
    """
    Convenience function to verify the accuracy of the Vedic Astronomical Computation Engine.
    
    Args:
        calculator: PlanetaryCalculator instance (if None, will create a new one)
        reference_data_path: Path to reference data
        generate_report: Whether to generate a detailed report
        report_path: Path to save the report
        
    Returns:
        Dictionary with verification results
    """
    # Create calculator if not provided
    if calculator is None:
        calculator = PlanetaryCalculator()
    
    # Create verification system
    verification_system = VerificationSystem(calculator, reference_data_path)
    
    # Run verification
    results = verification_system.verify_planetary_positions()
    
    # Generate report if requested
    if generate_report:
        if report_path is None:
            # Determine the project root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, "../.."))
            report_path = os.path.join(project_root, "reports", "verification_report.md")
        
        verification_system.generate_verification_report(results, report_path)
        
        # Generate visualization
        vis_path = os.path.splitext(report_path)[0] + "_visualization.png"
        verification_system.visualize_accuracy(results, vis_path)
    
    return results
