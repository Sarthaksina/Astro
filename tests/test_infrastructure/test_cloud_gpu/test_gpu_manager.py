# Cosmic Market Oracle - Tests for GPU Manager Module

import pytest
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from infrastructure.cloud_gpu.gpu_manager import GPUInstanceManager


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    config_content = """
    provider: vast.ai
    api_key: test_api_key
    instances_dir: ./test_instances
    instance_types:
      rtx4090:
        vast_ai_id: 123
        min_bid: 0.3
        max_bid: 0.5
      a100:
        vast_ai_id: 456
        min_bid: 1.0
        max_bid: 1.5
    """
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
        f.write(config_content)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    os.unlink(config_path)


@pytest.fixture
def mock_instances_dir():
    """Create a temporary directory for instance info files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def gpu_manager(temp_config_file, mock_instances_dir):
    """Create a GPUInstanceManager instance for testing."""
    # Patch the config to use our test instances directory
    with patch('infrastructure.cloud_gpu.gpu_manager.yaml.safe_load') as mock_yaml_load:
        mock_yaml_load.return_value = {
            'provider': 'vast.ai',
            'api_key': 'test_api_key',
            'instances_dir': mock_instances_dir,
            'instance_types': {
                'rtx4090': {
                    'vast_ai_id': 123,
                    'min_bid': 0.3,
                    'max_bid': 0.5
                },
                'a100': {
                    'vast_ai_id': 456,
                    'min_bid': 1.0,
                    'max_bid': 1.5
                }
            }
        }
        
        manager = GPUInstanceManager(config_path=temp_config_file)
        yield manager


class TestGPUInstanceManager:
    """Tests for the GPUInstanceManager class."""
    
    def test_initialization(self, gpu_manager):
        """Test initialization of the GPUInstanceManager."""
        assert gpu_manager.provider == 'vast.ai'
        assert gpu_manager.api_key == 'test_api_key'
        assert isinstance(gpu_manager.config, dict)
        assert 'instance_types' in gpu_manager.config
        assert 'rtx4090' in gpu_manager.config['instance_types']
    
    def test_save_and_get_instance_info(self, gpu_manager):
        """Test saving and retrieving instance information."""
        # Sample instance info
        instance_id = 'test-instance-123'
        instance_info = {
            'id': instance_id,
            'provider': 'vast.ai',
            'instance_type': 'rtx4090',
            'status': 'running',
            'start_time': '2023-01-01T12:00:00',
            'job': 'training',
            'cost_per_hour': 0.4
        }
        
        # Save the instance info
        gpu_manager._save_instance_info(instance_id, instance_info)
        
        # Retrieve the instance info
        retrieved_info = gpu_manager._get_instance_info(instance_id)
        
        # Verify the retrieved info matches the original
        assert retrieved_info == instance_info
        
        # Test retrieving non-existent instance
        non_existent_info = gpu_manager._get_instance_info('non-existent-id')
        assert non_existent_info is None
    
    @patch('infrastructure.cloud_gpu.gpu_manager.requests.post')
    def test_start_instance_vast_ai(self, mock_post, gpu_manager):
        """Test starting a GPU instance on VAST.ai."""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'success': True,
            'new_contract': {
                'id': 12345,
                'machine_id': 'vast-machine-123',
                'gpu_count': 1,
                'gpu_name': 'RTX 4090',
                'per_gpu_cost': 0.4
            }
        }
        mock_post.return_value = mock_response
        
        # Call the function
        with patch('infrastructure.cloud_gpu.gpu_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value.isoformat.return_value = '2023-01-01T12:00:00'
            
            instance_id = gpu_manager.start_instance('rtx4090', 'training', duration=24)
        
        # Verify the instance ID is returned
        assert instance_id is not None
        assert isinstance(instance_id, str)
        
        # Verify the API was called correctly
        mock_post.assert_called_once()
        
        # Verify instance info was saved
        instance_info = gpu_manager._get_instance_info(instance_id)
        assert instance_info is not None
        assert instance_info['instance_type'] == 'rtx4090'
        assert instance_info['job'] == 'training'
        assert instance_info['status'] == 'running'
    
    @patch('infrastructure.cloud_gpu.gpu_manager.requests.post')
    def test_stop_instance_vast_ai(self, mock_post, gpu_manager):
        """Test stopping a GPU instance on VAST.ai."""
        # Create a test instance first
        instance_id = 'test-instance-456'
        instance_info = {
            'id': instance_id,
            'provider': 'vast.ai',
            'instance_type': 'rtx4090',
            'status': 'running',
            'start_time': '2023-01-01T12:00:00',
            'job': 'training',
            'cost_per_hour': 0.4,
            'vast_contract_id': 12345  # Needed for the API call
        }
        gpu_manager._save_instance_info(instance_id, instance_info)
        
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {'success': True}
        mock_post.return_value = mock_response
        
        # Call the function
        success = gpu_manager.stop_instance(instance_id)
        
        # Verify the result
        assert success is True
        
        # Verify the API was called correctly
        mock_post.assert_called_once()
        
        # Verify instance info was updated
        updated_info = gpu_manager._get_instance_info(instance_id)
        assert updated_info is not None
        assert updated_info['status'] == 'stopped'
    
    def test_get_instance_status(self, gpu_manager):
        """Test getting the status of a GPU instance."""
        # Create a test instance first
        instance_id = 'test-instance-789'
        instance_info = {
            'id': instance_id,
            'provider': 'vast.ai',
            'instance_type': 'rtx4090',
            'status': 'running',
            'start_time': '2023-01-01T12:00:00',
            'job': 'training',
            'cost_per_hour': 0.4,
            'ip_address': '1.2.3.4',
            'ssh_port': 22
        }
        gpu_manager._save_instance_info(instance_id, instance_info)
        
        # Call the function
        status = gpu_manager.get_instance_status(instance_id)
        
        # Verify the status contains the expected information
        assert status is not None
        assert status['id'] == instance_id
        assert status['status'] == 'running'
        assert status['job'] == 'training'
        assert 'uptime_hours' in status
        assert 'estimated_cost' in status
        
        # Test with non-existent instance
        non_existent_status = gpu_manager.get_instance_status('non-existent-id')
        assert non_existent_status == {'error': 'Instance not found'}
    
    @patch('infrastructure.cloud_gpu.gpu_manager.time.sleep')
    @patch('infrastructure.cloud_gpu.gpu_manager.GPUInstanceManager.get_instance_status')
    @patch('infrastructure.cloud_gpu.gpu_manager.GPUInstanceManager.stop_instance')
    def test_monitor_instances(self, mock_stop, mock_status, mock_sleep, gpu_manager, mock_instances_dir):
        """Test monitoring of GPU instances."""
        # Create some test instances
        instance1_id = 'test-instance-1'
        instance1_info = {
            'id': instance1_id,
            'provider': 'vast.ai',
            'instance_type': 'rtx4090',
            'status': 'running',
            'start_time': '2023-01-01T10:00:00',  # 3+ hours ago
            'job': 'training',
            'cost_per_hour': 0.4
        }
        
        instance2_id = 'test-instance-2'
        instance2_info = {
            'id': instance2_id,
            'provider': 'vast.ai',
            'instance_type': 'a100',
            'status': 'running',
            'start_time': '2023-01-01T12:55:00',  # Just started
            'job': 'inference',
            'cost_per_hour': 1.2
        }
        
        # Save the instance info files
        Path(mock_instances_dir).mkdir(exist_ok=True)
        with open(Path(mock_instances_dir) / f"{instance1_id}.json", 'w') as f:
            json.dump(instance1_info, f)
        with open(Path(mock_instances_dir) / f"{instance2_id}.json", 'w') as f:
            json.dump(instance2_info, f)
        
        # Mock the status responses
        def mock_get_status(instance_id):
            if instance_id == instance1_id:
                return {
                    'id': instance1_id,
                    'status': 'running',
                    'gpu_utilization': 0.05,  # Low utilization
                    'uptime_hours': 3.0
                }
            else:
                return {
                    'id': instance2_id,
                    'status': 'running',
                    'gpu_utilization': 0.8,  # High utilization
                    'uptime_hours': 0.1
                }
        
        mock_status.side_effect = mock_get_status
        
        # Mock the sleep function to exit after one iteration
        mock_sleep.side_effect = [None, KeyboardInterrupt]
        
        # Call the function with a low threshold to trigger stopping for instance1
        try:
            gpu_manager.monitor_instances(threshold=0.1, interval=60)
        except KeyboardInterrupt:
            pass  # Expected to exit after second iteration
        
        # Verify that stop_instance was called for the low-utilization instance
        mock_stop.assert_called_once_with(instance1_id)
        
        # Verify sleep was called with the correct interval
        mock_sleep.assert_called_with(60)
