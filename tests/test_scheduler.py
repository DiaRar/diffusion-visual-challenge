"""Tests for the simplified scheduler system"""

import pytest


class TestBestHQScheduler:
    """Test the high-quality scheduler class"""

    def test_scheduler_class_exists(self):
        """The SDXLAnime_BestHQScheduler class should exist and be importable"""
        from configs.schedulers.high_scheduler import SDXLAnime_BestHQScheduler
        from diffusers.schedulers.scheduling_dpmsolver_multistep import (
            DPMSolverMultistepScheduler,
        )

        # Should be a subclass of DPMSolverMultistepScheduler
        assert issubclass(SDXLAnime_BestHQScheduler, DPMSolverMultistepScheduler)

    def test_scheduler_has_hooks(self):
        """The scheduler should have hook methods"""
        from configs.schedulers.high_scheduler import SDXLAnime_BestHQScheduler
        import torch

        scheduler = SDXLAnime_BestHQScheduler.from_config(
            {"algorithm_type": "dpmsolver++", "solver_order": 1}
        )

        # Should have hook methods
        assert hasattr(scheduler, "hook_before_step")
        assert hasattr(scheduler, "hook_after_step")

        # Hooks should return the input unchanged
        model_output = torch.randn(1, 4, 8, 8)
        timestep = 10
        sample = torch.randn(1, 4, 8, 8)

        result = scheduler.hook_before_step(model_output, timestep, sample)
        assert torch.equal(result, model_output)

        result = scheduler.hook_after_step(sample, timestep, sample)
        assert torch.equal(result, sample)


class TestApplyBestHQScheduler:
    """Test the apply_best_hq_scheduler function"""

    def test_function_exists(self):
        """The apply_best_hq_scheduler function should exist"""
        from configs.schedulers.high_scheduler import apply_best_hq_scheduler
        assert callable(apply_best_hq_scheduler)

    def test_apply_scheduler_modifies_pipeline(self, mock_pipeline):
        """Applying scheduler should modify the pipeline's scheduler"""
        from configs.schedulers.high_scheduler import (
            apply_best_hq_scheduler,
            SDXLAnime_BestHQScheduler,
        )

        original_scheduler = mock_pipeline.scheduler
        result = apply_best_hq_scheduler(mock_pipeline, use_karras_sigmas=True)

        # Should return the same pipeline
        assert result is mock_pipeline

        # Scheduler should be changed to our custom scheduler
        assert mock_pipeline.scheduler is not original_scheduler
        assert isinstance(mock_pipeline.scheduler, SDXLAnime_BestHQScheduler)

    def test_use_karras_sigmas_option(self, mock_pipeline):
        """The use_karras_sigmas parameter should be respected"""
        from configs.schedulers.high_scheduler import apply_best_hq_scheduler

        # Apply with karras sigmas
        apply_best_hq_scheduler(mock_pipeline, use_karras_sigmas=True)
        assert mock_pipeline.scheduler.config.use_karras_sigmas is True

        # Reset and apply without
        from diffusers import StableDiffusionXLPipeline
        mock_pipeline.scheduler = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype="float16",
        ).scheduler

        apply_best_hq_scheduler(mock_pipeline, use_karras_sigmas=False)
        assert mock_pipeline.scheduler.config.use_karras_sigmas is False

    def test_use_lu_lambdas_option(self, mock_pipeline):
        """The use_lu_lambdas parameter should be respected"""
        from configs.schedulers.high_scheduler import apply_best_hq_scheduler

        # Reset scheduler config
        mock_pipeline.scheduler.config = {
            "algorithm_type": "dpmsolver++",
            "solver_order": 1,
            "solver_type": "midpoint",
            "prediction_type": "epsilon",
            "use_karras_sigmas": False,
            "use_lu_lambdas": False,
        }

        # Explicitly set karras to False since it defaults to True
        apply_best_hq_scheduler(mock_pipeline, use_lu_lambdas=True, use_karras_sigmas=False)
        assert mock_pipeline.scheduler.config.use_lu_lambdas is True

        # Reset and apply without
        from diffusers import StableDiffusionXLPipeline
        mock_pipeline.scheduler = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype="float16",
        ).scheduler

        apply_best_hq_scheduler(mock_pipeline, use_lu_lambdas=False, use_karras_sigmas=True)
        assert mock_pipeline.scheduler.config.use_lu_lambdas is False

    def test_mutual_exclusivity(self, mock_pipeline):
        """use_karras_sigmas and use_lu_lambdas should be mutually exclusive"""
        from configs.schedulers.high_scheduler import apply_best_hq_scheduler

        with pytest.raises(ValueError, match="Choose only one"):
            apply_best_hq_scheduler(
                mock_pipeline, use_karras_sigmas=True, use_lu_lambdas=True
            )


class TestFilteredInitKwargs:
    """Test the _filtered_init_kwargs_for_dpmsolver helper function"""

    def test_function_exists(self):
        """The _filtered_init_kwargs_for_dpmsolver function should exist"""
        from configs.schedulers.high_scheduler import (
            _filtered_init_kwargs_for_dpmsolver,
        )
        assert callable(_filtered_init_kwargs_for_dpmsolver)

    def test_filters_kwargs_correctly(self):
        """Should filter out invalid kwargs for DPMSolverMultistepScheduler"""
        from configs.schedulers.high_scheduler import (
            _filtered_init_kwargs_for_dpmsolver,
        )

        # Create a config with valid and invalid kwargs
        config = {
            "algorithm_type": "dpmsolver++",
            "solver_order": 1,
            "invalid_param": "should_be_removed",
            "another_invalid": 123,
        }

        filtered = _filtered_init_kwargs_for_dpmsolver(config)

        # Should only contain valid params
        assert "algorithm_type" in filtered
        assert "solver_order" in filtered
        assert "invalid_param" not in filtered
        assert "another_invalid" not in filtered

    def test_preserves_valid_kwargs(self):
        """Should preserve all valid kwargs"""
        from configs.schedulers.high_scheduler import (
            _filtered_init_kwargs_for_dpmsolver,
        )

        # Valid config
        config = {
            "algorithm_type": "dpmsolver++",
            "solver_order": 1,
            "solver_type": "midpoint",
            "prediction_type": "epsilon",
        }

        filtered = _filtered_init_kwargs_for_dpmsolver(config)

        for key in config:
            assert key in filtered
            assert filtered[key] == config[key]
