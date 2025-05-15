# Cosmic Market Oracle - Tests for ThunderCompute GPU Manager Module

import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from infrastructure.cloud_gpu.thunder_compute_manager import (
    ThunderComputeManager,
    find_optimal_instance,
    calculate_cost_estimate,
    monitor_instance_utilization
)


@pytest.fixture
def mock_thunder_api_key():
    """Set up a mock ThunderCompute API key for testing."""
    original_key = os.environ.get('THUNDER_COMPUTE_API_KEY')
    os.environ['THUNDER_COMPUTE_API_KEY'] = 'test_thunder_api_key'
    yield
    if original_key:
        os.environ['THUNDER_COMPUTE_API_KEY'] = original_key
    else:
        del os.environ['THUNDER_COMPUTE_API_KEY']


@pytest.fixture
def thunder_manager(mock_thunder_api_key):
    """Create a ThunderComputeManager instance for testing."""
    return ThunderComputeManager()


class TestThunderComputeManager:
    """Tests for the ThunderComputeManager class."""
    
    def test_initialization(self, thunder_manager):
        """Test initialization of the ThunderComputeManager."""
        assert thunder_manager.api_key == 'test_thunder_api_key'
        assert thunder_manager.api_url == 'https://api.thundercompute.com/v1'
    
    @patch('infrastructure.cloud_gpu.thunder_compute_manager.requests.get')
    def test_list_available_instances(self, mock_get, thunder_manager):
        """Test listing available GPU instances on ThunderCompute."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'instances': [
                {
                    'id': 'tc-1',
                    'gpu_type': 'RTX 4090',
                    'gpu_count': 1,
                    'hourly_rate': 0.45,
                    'availability': 'available',
                    'performance_score': 95,
                    'memory_gb': 24
                },
                {
                    'id': 'tc-2',
                    'gpu_type': 'RTX 4090',
                    'gpu_count': 1,
                    'hourly_rate': 0.42,
                    'availability': 'available',
                    'performance_score': 92,
                    'memory_gb': 24
                },
                {
                    'id': 'tc-3',
                    'gpu_type': 'A100',
                    'gpu_count': 1,
                    'hourly_rate': 1.20,
                    'availability': 'available',
                    'performance_score': 98,
                    'memory_gb': 80
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Call the function
        instances = thunder_manager.list_available_instances(gpu_type='RTX 4090')
        
        # Verify the API was called correctly
        mock_get.assert_called_once()
        
        # Verify the returned instances
        assert len(instances) == 2  # Only the RTX 4090 instances
        assert instances[0]['id'] == 'tc-1'
        assert instances[1]['id'] == 'tc-2'
        
        # Test with no GPU type filter
        mock_get.reset_mock()
        mock_get.return_value = mock_response
        all_instances = thunder_manager.list_available_instances()
        assert len(all_instances) == 3  # All instances
    
    def test_find_optimal_instance(self):
        """Test finding the optimal instance based on price and performance."""
        # Sample instances
        instances = [
            {
                'id': 'tc-1',
                'hourly_rate': 0.45,
                'performance_score': 95,
                'memory_gb': 24
            },
            {
                'id': 'tc-2',
                'hourly_rate': 0.42,
                'performance_score': 92,
                'memory_gb': 24
            },
            {
                'id': 'tc-3',
                'hourly_rate': 0.40,
                'performance_score': 85,  # Lower performance
                'memory_gb': 24
            }
        ]
        
        # Call the function with default weights
        optimal = find_optimal_instance(instances)
        
        # The second instance should be optimal (good balance of price and performance)
        assert optimal['id'] == 'tc-2'
        
        # Test with different weights (prioritize performance)
        optimal_performance = find_optimal_instance(instances, performance_weight=0.8, price_weight=0.2)
        assert optimal_performance['id'] == 'tc-1'  # Best performance
        
        # Test with different weights (prioritize price)
        optimal_cheap = find_optimal_instance(instances, performance_weight=0.2, price_weight=0.8)
        assert optimal_cheap['id'] == 'tc-3'  # Cheapest
    
    def test_calculate_cost_estimate(self):
        """Test calculation of cost estimates for ThunderCompute instances."""
        # Sample instance
        instance = {'hourly_rate': 0.45}
        
        # Test with default parameters
        cost_24h = calculate_cost_estimate(instance, hours=24)
        assert cost_24h == 0.45 * 24
        
        # Test with different duration
        cost_48h = calculate_cost_estimate(instance, hours=48)
        assert cost_48h == 0.45 * 48
        
        # Test with discount for longer duration
        cost_with_discount = calculate_cost_estimate(instance, hours=720, long_term_discount=0.2)  # 30 days
        assert cost_with_discount < 0.45 * 720  # Should be discounted
        assert cost_with_discount == 0.45 * 720 * 0.8  # 20% discount
    
    @patch('infrastructure.cloud_gpu.thunder_compute_manager.requests.post')
    def test_create_instance(self, mock_post, thunder_manager):
        """Test creating a GPU instance on ThunderCompute."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'success': True,
            'instance': {
                'id': 'tc-12345',
                'gpu_type': 'RTX 4090',
                'gpu_count': 1,
                'hourly_rate': 0.45,
                'status': 'starting',
                'image': 'pytorch/pytorch:latest',
                'disk_gb': 100,
                'ssh_port': 22,
                'ssh_host': '5.6.7.8',
                'created_at': '2023-01-01T12:00:00Z'
            }
        }
        mock_post.return_value = mock_response
        
        # Call the function
        instance = thunder_manager.create_instance(
            instance_type='RTX 4090',
            image='pytorch/pytorch:latest',
            disk_gb=100
        )
        
        # Verify the API was called correctly
        mock_post.assert_called_once()
        
        # Verify the returned instance info
        assert instance is not None
        assert instance['id'] == 'tc-12345'
        assert instance['gpu_type'] == 'RTX 4090'
        assert instance['ssh_host'] == '5.6.7.8'
        assert instance['ssh_port'] == 22
    
    @patch('infrastructure.cloud_gpu.thunder_compute_manager.requests.get')
    def test_get_instance_status(self, mock_get, thunder_manager):
        """Test getting the status of a GPU instance on ThunderCompute."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'success': True,
            'instance': {
                'id': 'tc-12345',
                'gpu_type': 'RTX 4090',
                'gpu_count': 1,
                'hourly_rate': 0.45,
                'status': 'running',
                'image': 'pytorch/pytorch:latest',
                'disk_gb': 100,
                'ssh_port': 22,
                'ssh_host': '5.6.7.8',
                'created_at': '2023-01-01T12:00:00Z',
                'gpu_utilization': 0.65,
                'memory_utilization': 0.45
            }
        }
        mock_get.return_value = mock_response
        
        # Call the function
        status = thunder_manager.get_instance_status('tc-12345')
        
        # Verify the API was called correctly
        mock_get.assert_called_once()
        
        # Verify the returned status
        assert status is not None
        assert status['id'] == 'tc-12345'
        assert status['status'] == 'running'
        assert status['gpu_utilization'] == 0.65
        assert 'created_at' in status
    
    @patch('infrastructure.cloud_gpu.thunder_compute_manager.requests.delete')
    def test_stop_instance(self, mock_delete, thunder_manager):
        """Test stopping a GPU instance on ThunderCompute."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {'success': True}
        mock_delete.return_value = mock_response
        
        # Call the function
        success = thunder_manager.stop_instance('tc-12345')
        
        # Verify the API was called correctly
        mock_delete.assert_called_once()
        
        # Verify the result
        assert success is True
        
        # Test with API failure
        mock_delete.reset_mock()
        mock_response.json.return_value = {'success': False, 'error': 'Error stopping instance'}
        mock_delete.return_value = mock_response
        
        failure = thunder_manager.stop_instance('tc-12345')
        assert failure is False
    
    @patch('infrastructure.cloud_gpu.thunder_compute_manager.requests.get')
    def test_monitor_instance_utilization(self, mock_get):
        """Test monitoring GPU utilization of an instance."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'success': True,
            'instance': {
                'id': 'tc-12345',
                'status': 'running',
                'gpu_utilization': 0.05  # Low utilization
            }
        }
        mock_get.return_value = mock_response
        
        # Call the function
        manager = ThunderComputeManager()
        utilization = monitor_instance_utilization(manager, 'tc-12345', threshold=0.1)
        
        # Verify the API was called correctly
        mock_get.assert_called_once()
        
        # Verify the result (should be below threshold)
        assert utilization < 0.1
        
        # Test with high utilization
        mock_get.reset_mock()
        mock_response.json.return_value['instance']['gpu_utilization'] = 0.85  # High utilization
        mock_get.return_value = mock_response
        
        utilization_high = monitor_instance_utilization(manager, 'tc-12345', threshold=0.1)
        assert utilization_high > 0.1
