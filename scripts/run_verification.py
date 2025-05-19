#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Vedic Astronomical Computation Engine Verification Script

This script runs the verification system to compare the engine's calculations
with historical almanac data, generating a detailed report and visualizations.

Usage:
    python run_verification.py [--tolerance TOLERANCE] [--report-dir REPORT_DIR]

Options:
    --tolerance TOLERANCE    Maximum allowed difference in degrees (default: 1.0)
    --report-dir REPORT_DIR  Directory to save the verification report (default: reports)
"""

import os
import sys
import argparse
import datetime
from src.trading.unified_mcts import MCTS, MCTSPredictor
from src.trading.modular_hierarchical_rl import ModularHierarchicalRLAgent, MCTSStrategicPlanner, PPOTacticalExecutor
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Check Python version compatibility
if sys.version_info.major != 3 or sys.version_info.minor < 10:
    print("Warning: This script is optimized for Python 3.10. You are using Python {}.{}.".format(
        sys.version_info.major, sys.version_info.minor))
    print("Some features may not work correctly.")

from src.astro_engine.planetary_positions import PlanetaryCalculator
from src.astro_engine.verification_system import VerificationSystem


def main():
    """Run the verification system and generate reports."""
    parser = argparse.ArgumentParser(description="Verify the Vedic Astronomical Computation Engine")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="Maximum allowed difference in degrees (default: 1.0)")
    parser.add_argument("--report-dir", type=str, default=os.path.join(project_root, "reports"),
                        help="Directory to save the verification report")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Vedic Astronomical Computation Engine Verification")
    print("=" * 80)
    print(f"Starting verification process at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using tolerance: {args.tolerance}°")
    print()
    
    # Create calculator
    print("Initializing Planetary Calculator...")
    calculator = PlanetaryCalculator()
    
    # Create verification system
    print("Setting up Verification System...")
    verification_system = VerificationSystem(calculator)
    
    # Create report directory if it doesn't exist
    os.makedirs(args.report_dir, exist_ok=True)
    
    # Run verification
    print("Running verification against historical almanac data...")
    results = verification_system.verify_planetary_positions(tolerance=args.tolerance)
    
    # Print summary results
    print("\nVerification Results:")
    print(f"- Overall Status: {'PASSED' if results['success'] else 'FAILED'}")
    print(f"- Accuracy: {results.get('accuracy', 0):.2f}%")
    print(f"- Total Tests: {results['total_tests']}")
    print(f"- Passed Tests: {results['passed_tests']}")
    print(f"- Failed Tests: {results['failed_tests']}")
    
    # Generate report
    report_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(args.report_dir, f"verification_report_{report_timestamp}.md")
    vis_path = os.path.join(args.report_dir, f"verification_visualization_{report_timestamp}.png")
    
    print("\nGenerating detailed report...")
    verification_system.generate_verification_report(results, report_path)
    
    print("Generating visualizations...")
    verification_system.visualize_accuracy(results, vis_path)
    
    print("\nVerification completed!")
    print(f"Report saved to: {report_path}")
    print(f"Visualization saved to: {vis_path}")
    
    # Return success status
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
