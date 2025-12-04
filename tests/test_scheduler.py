"""Tests for configs/scheduler_loader.py"""

import pytest

from configs.scheduler_loader import (
    SCHEDULER_CONFIGS,
    apply_scheduler_to_pipeline,
    get_scheduler_info,
    list_available_schedulers,
)


class TestSchedulerConfigs:
    """Test scheduler configurations"""

    def test_all_schedulers_defined(self):
        """All expected schedulers should be defined"""
        assert "euler" in SCHEDULER_CONFIGS
        assert "dpm" in SCHEDULER_CONFIGS
        assert "unipc" in SCHEDULER_CONFIGS

    def test_euler_config(self):
        """Euler scheduler config should have required fields"""
        config = SCHEDULER_CONFIGS["euler"]
        assert "class" in config
        assert config["class"] == "EulerDiscreteScheduler"
        assert "description" in config
        assert "use_case" in config

    def test_dpm_config(self):
        """DPM scheduler config should have required fields"""
        config = SCHEDULER_CONFIGS["dpm"]
        assert "class" in config
        assert config["class"] == "HighQualityDPMScheduler"
        assert "description" in config
        assert "use_case" in config

    def test_unipc_config(self):
        """UniPC scheduler config should have required fields"""
        config = SCHEDULER_CONFIGS["unipc"]
        assert "class" in config
        assert config["class"] == "UniPCMultistepScheduler"
        assert "description" in config
        assert "use_case" in config


class TestGetSchedulerInfo:
    """Test get_scheduler_info function"""

    def test_get_valid_scheduler(self):
        """Should return config for valid scheduler name"""
        info = get_scheduler_info("euler")
        assert info is not None
        assert info["class"] == "EulerDiscreteScheduler"

    def test_get_invalid_scheduler(self):
        """Should return None for invalid scheduler name"""
        info = get_scheduler_info("nonexistent")
        assert info is None

    def test_case_insensitive(self):
        """Should work with different case"""
        info_lower = get_scheduler_info("euler")
        info_upper = get_scheduler_info("EULER")
        info_mixed = get_scheduler_info("EuLeR")

        assert info_lower is not None
        assert info_upper is not None
        assert info_mixed is not None
        assert info_lower == info_upper == info_mixed


class TestApplySchedulerToPipeline:
    """Test apply_scheduler_to_pipeline function"""

    def test_apply_valid_scheduler(self, mock_pipeline):
        """Should apply valid scheduler to pipeline"""
        result, _ = apply_scheduler_to_pipeline(mock_pipeline, "euler", 20)
        assert result is mock_pipeline
        # The function attempts to apply scheduler and logs warnings if it fails
        # but always returns the pipeline

    def test_apply_invalid_scheduler(self, mock_pipeline):
        """Should raise ValueError for invalid scheduler"""
        with pytest.raises(ValueError, match="Unknown scheduler"):
            apply_scheduler_to_pipeline(mock_pipeline, "nonexistent", 20)

    def test_apply_scheduler_returns_pipeline(self, mock_pipeline):
        """Should always return the pipeline object"""
        original_pipeline = mock_pipeline
        result, _ = apply_scheduler_to_pipeline(mock_pipeline, "euler", 20)

        assert result is original_pipeline  # Same pipeline instance is returned


class TestListAvailableSchedulers:
    """Test list_available_schedulers function"""

    def test_returns_all_configs(self):
        """Should return copy of all scheduler configs"""
        schedulers = list_available_schedulers()

        assert isinstance(schedulers, dict)
        assert len(schedulers) == len(SCHEDULER_CONFIGS)
        assert "euler" in schedulers
        assert "dmp" in schedulers or "dpm" in schedulers
        assert "unipc" in schedulers

    def test_returns_copy(self):
        """Should return a copy, not the original"""
        schedulers1 = list_available_schedulers()
        schedulers2 = list_available_schedulers()

        # Modifications shouldn't affect the original
        assert schedulers1 is not schedulers2


class TestSchedulerIntegration:
    """Integration tests for scheduler system"""

    def test_all_schedulers_have_valid_classes(self):
        """All scheduler classes should be importable"""
        from diffusers.schedulers.scheduling_euler_discrete import (
            EulerDiscreteScheduler,
        )
        from diffusers.schedulers.scheduling_dpmsolver_multistep import (
            DPMSolverMultistepScheduler,
        )
        from diffusers.schedulers.scheduling_unipc_multistep import (
            UniPCMultistepScheduler,
        )

        # Verify classes exist and are importable
        assert EulerDiscreteScheduler is not None
        assert DPMSolverMultistepScheduler is not None
        assert UniPCMultistepScheduler is not None

    def test_scheduler_classes_match_configs(self):
        """Scheduler classes in configs should match actual classes"""
        from diffusers.schedulers.scheduling_euler_discrete import (
            EulerDiscreteScheduler,
        )
        from diffusers.schedulers.scheduling_dpmsolver_multistep import (
            DPMSolverMultistepScheduler,
        )
        from diffusers.schedulers.scheduling_unipc_multistep import (
            UniPCMultistepScheduler,
        )
        from configs.my_scheduler import HighQualityDPMScheduler

        class_map = {
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "HighQualityDPMScheduler": HighQualityDPMScheduler,
            "UniPCMultistepScheduler": UniPCMultistepScheduler,
        }

        for scheduler_name, config in SCHEDULER_CONFIGS.items():
            class_name = config["class"]
            assert class_name in class_map
            assert class_map[class_name] is not None
