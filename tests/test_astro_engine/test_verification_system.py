"""
Tests for the Vedic Astronomical Computation Engine Verification System.

This module tests the functionality of the verification system that compares
calculated planetary positions with historical almanac data.
"""

import os
import sys
import pytest
import json
import tempfile
from datetime import datetime

# Add the src directory to the path so we can import the modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.astro_engine.planetary_positions import PlanetaryCalculator, SUN, MOON, MERCURY, VENUS, MARS, JUPITER, SATURN, RAHU, KETU
from src.astro_engine.verification_system import VerificationSystem, verify_engine_accuracy


@pytest.fixture
def planetary_calculator():
    """Create a PlanetaryCalculator instance for testing."""
    return PlanetaryCalculator()


@pytest.fixture
def verification_system(planetary_calculator):
    """Create a VerificationSystem instance for testing."""
    return VerificationSystem(planetary_calculator)


@pytest.fixture
def sample_reference_data():
    """Sample reference data for testing."""
    return {
        "data": [
            {
                "date": "2000-01-01",
                "sun": {
                    "longitude": 280.55,
                    "nakshatra": 20
                },
                "moon": {
                    "longitude": 124.32,
                    "nakshatra": 9
                }
            }
        ]
    }


def test_verification_system_initialization(verification_system):
    """Test VerificationSystem initialization."""
    assert verification_system is not None
    assert verification_system.calculator is not None
    assert verification_system.reference_data is not None


def test_load_reference_data(verification_system):
    """Test loading reference data."""
    reference_data = verification_system._load_reference_data()
    
    # Check that reference data is a dictionary
    assert isinstance(reference_data, dict)
    
    # Check that historical_positions is present
    assert "historical_positions" in reference_data


def test_verify_planetary_positions(verification_system):
    """Test verification of planetary positions."""
    results = verification_system.verify_planetary_positions()
    
    # Check that results contain expected keys
    assert "success" in results
    assert "total_tests" in results
    assert "passed_tests" in results
    assert "failed_tests" in results
    assert "details" in results
    
    # Check that details is a list
    assert isinstance(results["details"], list)


def test_generate_verification_report(verification_system):
    """Test generation of verification report."""
    # Run verification
    results = verification_system.verify_planetary_positions()
    
    # Generate report
    report = verification_system.generate_verification_report(results)
    
    # Check that report is a string
    assert isinstance(report, str)
    
    # Check that report contains expected sections
    assert "# Vedic Astronomical Computation Engine Verification Report" in report
    assert "## Summary" in report
    assert "## Detailed Results" in report
    assert "## Recommendations" in report


def test_generate_verification_report_with_output(verification_system):
    """Test generation of verification report with output file."""
    # Run verification
    results = verification_system.verify_planetary_positions()
    
    # Create temporary file for report
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
        report_path = tmp.name
    
    try:
        # Generate report
        output_path = verification_system.generate_verification_report(results, report_path)
        
        # Check that output path is correct
        assert output_path == report_path
        
        # Check that file exists
        assert os.path.exists(report_path)
        
        # Check that file contains report
        with open(report_path, 'r') as f:
            content = f.read()
            assert "# Vedic Astronomical Computation Engine Verification Report" in content
    finally:
        # Clean up
        if os.path.exists(report_path):
            os.unlink(report_path)


def test_add_reference_data(verification_system, sample_reference_data):
    """Test adding custom reference data."""
    # Create temporary directory for reference data
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Set reference data path
        verification_system.reference_data_path = tmp_dir
        
        # Add reference data
        success = verification_system.add_reference_data(sample_reference_data, "test_data.json")
        
        # Check that data was added successfully
        assert success
        
        # Check that file exists
        file_path = os.path.join(tmp_dir, "test_data.json")
        assert os.path.exists(file_path)
        
        # Check that file contains correct data
        with open(file_path, 'r') as f:
            data = json.load(f)
            assert "data" in data
            assert len(data["data"]) == 1
            assert data["data"][0]["date"] == "2000-01-01"


def test_compare_with_external_almanac(verification_system):
    """Test comparison with external almanac data."""
    # Create sample almanac data
    almanac_data = [
        {
            "date": "2000-01-01",
            "sun": {
                "longitude": 280.55
            },
            "moon": {
                "longitude": 124.32
            }
        }
    ]
    
    # Compare with almanac data
    results = verification_system.compare_with_external_almanac(almanac_data)
    
    # Check that results contain expected keys
    assert "success" in results
    assert "total_tests" in results
    assert "passed_tests" in results
    assert "failed_tests" in results
    assert "details" in results
    
    # Check that details is a list
    assert isinstance(results["details"], list)


def test_verify_engine_accuracy():
    """Test the convenience function for verifying engine accuracy."""
    # Create temporary directory for report
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = os.path.join(tmp_dir, "verification_report.md")
        
        # Run verification
        results = verify_engine_accuracy(
            generate_report=True,
            report_path=report_path
        )
        
        # Check that results contain expected keys
        assert "success" in results
        assert "total_tests" in results
        assert "passed_tests" in results
        assert "failed_tests" in results
        
        # Check that report was generated
        assert os.path.exists(report_path)


def test_edge_case_empty_reference_data():
    """Test handling of empty reference data."""
    # Create calculator
    calculator = PlanetaryCalculator()
    
    # Create verification system with empty reference data path
    with tempfile.TemporaryDirectory() as tmp_dir:
        verification_system = VerificationSystem(calculator, tmp_dir)
        
        # Run verification
        results = verification_system.verify_planetary_positions()
        
        # Check that verification fails gracefully
        assert "success" in results
        assert results["success"] is False
        assert "error" in results


def test_edge_case_invalid_date_format():
    """Test handling of invalid date format in reference data."""
    # Create calculator
    calculator = PlanetaryCalculator()
    
    # Create verification system
    verification_system = VerificationSystem(calculator)
    
    # Create sample almanac data with invalid date
    almanac_data = [
        {
            "date": "invalid-date",
            "sun": {
                "longitude": 280.55
            }
        }
    ]
    
    # Compare with almanac data (should not raise exception)
    try:
        verification_system.compare_with_external_almanac(almanac_data)
        assert True  # Test passes if no exception is raised
    except Exception:
        assert False  # Test fails if exception is raised


def test_integration_full_verification_workflow(verification_system):
    """Test the full verification workflow."""
    # Run verification
    results = verification_system.verify_planetary_positions()
    
    # Generate report
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = os.path.join(tmp_dir, "verification_report.md")
        vis_path = os.path.join(tmp_dir, "verification_visualization.png")
        
        # Generate report
        verification_system.generate_verification_report(results, report_path)
        
        # Generate visualization
        verification_system.visualize_accuracy(results, vis_path)
        
        # Check that files exist
        assert os.path.exists(report_path)
        assert os.path.exists(vis_path)
