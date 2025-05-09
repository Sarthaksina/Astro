import pytest
import torch
import numpy as np
from src.models.model_factory import ModelFactory, create_model_from_config, get_optimizer, get_scheduler
from src.models.time_series import AttentionBiLSTM, TemporalConvNet, WaveNetModel
from src.models.transformers import AstroEconomicTransformer, AstroEventDetectionTransformer

@pytest.fixture
def base_config():
    """Base configuration fixture for testing"""
    return {
        'device': 'cpu',  # Use CPU for testing
        'model_type': 'lstm',
        'lstm_config': {
            'input_dim': 32,
            'hidden_dim': 64,
            'num_layers': 2,
            'output_dim': 1,
            'dropout': 0.1
        }
    }

@pytest.fixture
def model_factory(base_config):
    """Model factory fixture"""
    return ModelFactory(base_config)

def test_model_factory_initialization(base_config):
    """Test ModelFactory initialization"""
    factory = ModelFactory(base_config)
    assert factory.model_type == 'lstm'
    assert factory.device == 'cpu'
    assert factory.config == base_config

@pytest.mark.parametrize('model_type', [
    'lstm', 'tcn', 'wavenet', 'transformer', 'event_transformer'
])
def test_create_model_types(base_config, model_type):
    """Test creation of different model types"""
    config = base_config.copy()
    config['model_type'] = model_type
    
    # Add necessary config for each model type
    if model_type == 'tcn':
        config['tcn_config'] = {
            'input_dim': 32,
            'num_channels': [64, 128],
            'kernel_size': 3,
            'dropout': 0.1
        }
    elif model_type == 'wavenet':
        config['wavenet_config'] = {
            'input_dim': 32,
            'residual_channels': 32,
            'skip_channels': 32,
            'dilation_layers': 4,
            'output_dim': 1
        }
    elif model_type == 'transformer':
        config['transformer_config'] = {
            'd_model': 64,
            'num_heads': 4,
            'num_layers': 2,
            'd_ff': 128,
            'max_seq_len': 100,
            'dropout': 0.1,
            'num_market_features': 5,
            'num_astro_features': 10,
            'output_dim': 1
        }
    elif model_type == 'event_transformer':
        config['event_transformer_config'] = {
            'd_model': 64,
            'num_heads': 4,
            'num_layers': 2,
            'num_astro_events': 5,
            'max_seq_len': 100,
            'dropout': 0.1,
            'num_features': 15
        }
    
    factory = ModelFactory(config)
    model = factory.create_model()
    
    # Test model type
    if model_type == 'lstm':
        assert isinstance(model, AttentionBiLSTM)
    elif model_type == 'tcn':
        assert isinstance(model, TemporalConvNet)
    elif model_type == 'wavenet':
        assert isinstance(model, WaveNetModel)
    elif model_type == 'transformer':
        assert isinstance(model, AstroEconomicTransformer)
    elif model_type == 'event_transformer':
        assert isinstance(model, AstroEventDetectionTransformer)

def test_invalid_model_type(base_config):
    """Test handling of invalid model type"""
    config = base_config.copy()
    config['model_type'] = 'invalid_type'
    factory = ModelFactory(config)
    
    with pytest.raises(ValueError, match="Unknown model type: invalid_type"):
        factory.create_model()

def test_create_model_from_config(base_config):
    """Test create_model_from_config convenience function"""
    model = create_model_from_config(base_config)
    assert isinstance(model, AttentionBiLSTM)
    assert model.input_dim == base_config['lstm_config']['input_dim']

@pytest.mark.parametrize('optimizer_type', ['adam', 'adamw', 'sgd'])
def test_get_optimizer(base_config, optimizer_type):
    """Test optimizer creation"""
    config = base_config.copy()
    config['optimizer_config'] = {
        'type': optimizer_type,
        'learning_rate': 0.001,
        'weight_decay': 0.01
    }
    
    model = create_model_from_config(config)
    optimizer = get_optimizer(model, config)
    
    if optimizer_type == 'adam':
        assert isinstance(optimizer, torch.optim.Adam)
    elif optimizer_type == 'adamw':
        assert isinstance(optimizer, torch.optim.AdamW)
    elif optimizer_type == 'sgd':
        assert isinstance(optimizer, torch.optim.SGD)

def test_invalid_optimizer_type(base_config):
    """Test handling of invalid optimizer type"""
    config = base_config.copy()
    config['optimizer_config'] = {'type': 'invalid_type'}
    model = create_model_from_config(config)
    
    with pytest.raises(ValueError, match="Unknown optimizer type: invalid_type"):
        get_optimizer(model, config)

@pytest.mark.parametrize('scheduler_type', ['step', 'cosine', 'plateau'])
def test_get_scheduler(base_config, scheduler_type):
    """Test scheduler creation"""
    config = base_config.copy()
    config['optimizer_config'] = {'type': 'adam', 'learning_rate': 0.001}
    config['scheduler_config'] = {'type': scheduler_type}
    
    if scheduler_type == 'step':
        config['scheduler_config'].update({'step_size': 10, 'gamma': 0.1})
    elif scheduler_type == 'cosine':
        config['scheduler_config'].update({'T_max': 100, 'eta_min': 0})
    elif scheduler_type == 'plateau':
        config['scheduler_config'].update({'patience': 5, 'factor': 0.1})
    
    model = create_model_from_config(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    if scheduler_type == 'step':
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    elif scheduler_type == 'cosine':
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    elif scheduler_type == 'plateau':
        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

def test_invalid_scheduler_type(base_config):
    """Test handling of invalid scheduler type"""
    config = base_config.copy()
    config['optimizer_config'] = {'type': 'adam', 'learning_rate': 0.001}
    config['scheduler_config'] = {'type': 'invalid_type'}
    
    model = create_model_from_config(config)
    optimizer = get_optimizer(model, config)
    
    with pytest.raises(ValueError, match="Unknown scheduler type: invalid_type"):
        get_scheduler(optimizer, config)

def test_model_to_device(base_config):
    """Test model is correctly moved to specified device"""
    config = base_config.copy()
    factory = ModelFactory(config)
    model = factory.create_model()
    
    # Check model device
    assert next(model.parameters()).device.type == 'cpu'

def test_model_output_shapes(base_config):
    """Test model output shapes"""
    config = base_config.copy()
    factory = ModelFactory(config)
    model = factory.create_model()
    
    # Create dummy input
    batch_size = 4
    seq_len = 10
    input_dim = config['lstm_config']['input_dim']
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Get model output
    output = model(x)
    
    # Check output shape
    expected_output_dim = config['lstm_config']['output_dim']
    assert output.shape == (batch_size, expected_output_dim)