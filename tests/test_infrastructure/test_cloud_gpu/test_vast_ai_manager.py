# Cosmic Market Oracle - Tests for VAST.ai GPU Manager Module

import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from infrastructure.cloud_gpu.vast_ai_manager import (
    VASTAIManager,
    find_optimal_instance,
    calculate_bid_price,
    monitor_instance_utilization
)


@pytest.fixture
def mock_vast_api_key():
    """Set up a mock VAST.ai API key for testing."""
    original_key = os.environ.get('VAST_AI_API_KEY')
    os.environ['VAST_AI_API_KEY'] = 'test_vast_api_key'
    yield
    if original_key:
        os.environ['VAST_AI_API_KEY'] = original_key
    else:
        del os.environ['VAST_AI_API_KEY']


@pytest.fixture
def vast_manager(mock_vast_api_key):
    """Create a VASTAIManager instance for testing."""
    return VASTAIManager()


class TestVASTAIManager:
    """Tests for the VASTAIManager class."""
    
    def test_initialization(self, vast_manager):
        """Test initialization of the VASTAIManager."""
        assert vast_manager.api_key == 'test_vast_api_key'
        assert vast_manager.api_url == 'https://console.vast.ai/api/v0'
    
    @patch('infrastructure.cloud_gpu.vast_ai_manager.requests.get')
    def test_list_available_instances(self, mock_get, vast_manager):
        """Test listing available GPU instances on VAST.ai."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'offers': [
                {
                    'id': 1,
                    'cuda_max_good': 12,
                    'gpu_name': 'RTX 4090',
                    'num_gpus': 1,
                    'dph_total': 0.4,
                    'reliability2': 0.98,
                    'dlperf': 35.5,
                    'verified': True
                },
                {
                    'id': 2,
                    'cuda_max_good': 12,
                    'gpu_name': 'RTX 4090',
                    'num_gpus': 1,
                    'dph_total': 0.38,
                    'reliability2': 0.95,
                    'dlperf': 34.8,
                    'verified': True
                },
                {
                    'id': 3,
                    'cuda_max_good': 11,
                    'gpu_name': 'RTX 3090',
                    'num_gpus': 1,
                    'dph_total': 0.25,
                    'reliability2': 0.99,
                    'dlperf': 28.3,
                    'verified': True
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Call the function
        instances = vast_manager.list_available_instances(gpu_type='RTX 4090')
        
        # Verify the API was called correctly
        mock_get.assert_called_once()
        
        # Verify the returned instances
        assert len(instances) == 2  # Only the RTX 4090 instances
        assert instances[0]['id'] == 1
        assert instances[1]['id'] == 2
        
        # Test with no GPU type filter
        mock_get.reset_mock()
        mock_get.return_value = mock_response
        all_instances = vast_manager.list_available_instances()
        assert len(all_instances) == 3  # All instances
    
    def test_find_optimal_instance(self):
        """Test finding the optimal instance based on price and reliability."""
        # Sample instances
        instances = [
            {
                'id': 1,
                'dph_total': 0.4,
                'reliability2': 0.98,
                'dlperf': 35.5,
                'verified': True
            },
            {
                'id': 2,
                'dph_total': 0.38,
                'reliability2': 0.95,
                'dlperf': 34.8,
                'verified': True
            },
            {
                'id': 3,
                'dph_total': 0.35,
                'reliability2': 0.85,  # Lower reliability
                'dlperf': 34.0,
                'verified': True
            }
        ]
        
        # Call the function with default weights
        optimal = find_optimal_instance(instances)
        
        # The second instance should be optimal (good balance of price and reliability)
        assert optimal['id'] == 2
        
        # Test with different weights (prioritize reliability)
        optimal_reliable = find_optimal_instance(instances, reliability_weight=0.8, price_weight=0.2)
        assert optimal_reliable['id'] == 1  # Most reliable
        
        # Test with different weights (prioritize price)
        optimal_cheap = find_optimal_instance(instances, reliability_weight=0.2, price_weight=0.8)
        assert optimal_cheap['id'] == 3  # Cheapest
    
    def test_calculate_bid_price(self):
        """Test calculation of bid price based on market conditions."""
        # Sample instance with market price
        instance = {'dph_total': 0.4}
        
        # Test with default parameters
        bid = calculate_bid_price(instance)
        assert bid > instance['dph_total']  # Should be higher than market price
        assert bid <= instance['dph_total'] * 1.2  # But not too high
        
        # Test with custom parameters
        bid_aggressive = calculate_bid_price(instance, premium_factor=1.3)
        assert bid_aggressive > bid  # Should be higher with higher premium
        
        # Test with minimum and maximum constraints
        bid_constrained = calculate_bid_price(instance, min_price=0.5, max_price=0.6)
        assert bid_constrained == 0.5  # Should be at least min_price
        
        # Test with very high market price
        expensive_instance = {'dph_total': 2.0}
        bid_expensive = calculate_bid_price(expensive_instance, max_price=1.5)
        assert bid_expensive == 1.5  # Should be capped at max_price
    
    @patch('infrastructure.cloud_gpu.vast_ai_manager.requests.post')
    def test_create_instance(self, mock_post, vast_manager):
        """Test creating a GPU instance on VAST.ai."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'success': True,
            'new_contract': {
                'id': 12345,
                'machine_id': 1,
                'gpu_count': 1,
                'gpu_name': 'RTX 4090',
                'dph_total': 0.42,
                'image_uuid': 'docker-image-123',
                'disk_space_gb': 100,
                'ssh_port': 22,
                'ssh_host': '1.2.3.4'
            }
        }
        mock_post.return_value = mock_response
        
        # Call the function
        instance = vast_manager.create_instance(
            machine_id=1,
            image='pytorch/pytorch:latest',
            disk_space_gb=100,
            bid_price=0.42
        )
        
        # Verify the API was called correctly
        mock_post.assert_called_once()
        
        # Verify the returned instance info
        assert instance is not None
        assert instance['id'] == 12345
        assert instance['machine_id'] == 1
        assert instance['gpu_name'] == 'RTX 4090'
        assert instance['ssh_host'] == '1.2.3.4'
        assert instance['ssh_port'] == 22
    
    @patch('infrastructure.cloud_gpu.vast_ai_manager.requests.get')
    def test_get_instance_status(self, mock_get, vast_manager):
        """Test getting the status of a GPU instance on VAST.ai."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'success': True,
            'contract': {
                'id': 12345,
                'machine_id': 1,
                'gpu_count': 1,
                'gpu_name': 'RTX 4090',
                'dph_total': 0.42,
                'image_uuid': 'docker-image-123',
                'disk_space_gb': 100,
                'ssh_port': 22,
                'ssh_host': '1.2.3.4',
                'status': 'running',
                'start_date': '2023-01-01T12:00:00Z',
                'end_date': None,
                'gpu_utilization': 0.75
            }
        }
        mock_get.return_value = mock_response
        
        # Call the function
        status = vast_manager.get_instance_status(12345)
        
        # Verify the API was called correctly
        mock_get.assert_called_once()
        
        # Verify the returned status
        assert status is not None
        assert status['id'] == 12345
        assert status['status'] == 'running'
        assert status['gpu_utilization'] == 0.75
        assert 'start_date' in status
    
    @patch('infrastructure.cloud_gpu.vast_ai_manager.requests.put')
    def test_stop_instance(self, mock_put, vast_manager):
        """Test stopping a GPU instance on VAST.ai."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {'success': True}
        mock_put.return_value = mock_response
        
        # Call the function
        success = vast_manager.stop_instance(12345)
        
        # Verify the API was called correctly
        mock_put.assert_called_once()
        
        # Verify the result
        assert success is True
        
        # Test with API failure
        mock_put.reset_mock()
        mock_response.json.return_value = {'success': False, 'msg': 'Error stopping instance'}
        mock_put.return_value = mock_response
        
        failure = vast_manager.stop_instance(12345)
        assert failure is False
    
    @patch('infrastructure.cloud_gpu.vast_ai_manager.requests.get')
    def test_monitor_instance_utilization(self, mock_get):
        """Test monitoring GPU utilization of an instance."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'success': True,
            'contract': {
                'id': 12345,
                'status': 'running',
                'gpu_utilization': 0.05  # Low utilization
            }
        }
        mock_get.return_value = mock_response
        
        # Call the function
        manager = VASTAIManager()
        utilization = monitor_instance_utilization(manager, 12345, threshold=0.1)
        
        # Verify the API was called correctly
        mock_get.assert_called_once()
        
        # Verify the result (should be below threshold)
        assert utilization < 0.1
        
        # Test with high utilization
        mock_get.reset_mock()
        mock_response.json.return_value['contract']['gpu_utilization'] = 0.85  # High utilization
        mock_get.return_value = mock_response
        
        utilization_high = monitor_instance_utilization(manager, 12345, threshold=0.1)
        assert utilization_high > 0.1
