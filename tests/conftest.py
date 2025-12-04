"""Pytest configuration and fixtures"""

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_pipeline():
    """Create a mock diffusion pipeline with a dict-like scheduler config"""
    mock_pipe = Mock()
    mock_pipe.scheduler = Mock()
    # Make config behave like a dict
    mock_pipe.scheduler.config = {
        "algorithm_type": "dpmsolver++",
        "solver_order": 1,
        "solver_type": "midpoint",
        "prediction_type": "epsilon",
        "use_karras_sigmas": False,
        "use_lu_lambdas": False,
    }
    mock_pipe.vae = Mock()
    mock_pipe.to = Mock(return_value=mock_pipe)
    mock_pipe.enable_attention_slicing = Mock()
    mock_pipe.enable_model_cpu_offload = Mock()
    return mock_pipe


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory as a Path object"""
    return tmp_path
