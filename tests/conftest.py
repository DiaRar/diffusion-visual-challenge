"""Pytest configuration and fixtures"""

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_pipeline():
    """Create a mock diffusion pipeline"""
    mock_pipe = Mock()
    mock_pipe.scheduler = Mock()
    mock_pipe.scheduler.config = Mock()
    mock_pipe.vae = Mock()
    mock_pipe.to = Mock(return_value=mock_pipe)
    mock_pipe.enable_attention_slicing = Mock()
    mock_pipe.enable_model_cpu_offload = Mock()
    return mock_pipe


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory as a Path object"""
    return tmp_path
