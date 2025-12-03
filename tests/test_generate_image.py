"""Tests for infer/generate_image.py"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

import pytest
import torch

# Add project root to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from infer.generate_image import (
    _export_run_metadata,
    _generate_run_id,
    _get_git_hash,
    _get_package_version,
    _get_package_versions,
    generate_single_image,
    parse_args,
)


class TestRunIdGeneration:
    """Test run ID generation"""

    def test_generate_run_id_with_different_params(self):
        """Run IDs should differ with different parameters"""
        run_id_1 = _generate_run_id(seed=123, prompt="test prompt", profile_name="768_long")
        run_id_2 = _generate_run_id(seed=456, prompt="test prompt", profile_name="768_long")
        run_id_3 = _generate_run_id(seed=123, prompt="different prompt", profile_name="768_long")
        run_id_4 = _generate_run_id(seed=123, prompt="test prompt", profile_name="1024_hq")

        assert run_id_1 != run_id_2
        assert run_id_1 != run_id_3
        assert run_id_1 != run_id_4

    def test_generate_run_id_format(self):
        """Run ID should contain timestamp, seed, prompt hash, and profile"""
        run_id = _generate_run_id(seed=123, prompt="test prompt", profile_name="768_long")

        assert "123" in run_id
        # Prompt is hashed, so check for hex hash pattern
        import re
        assert re.search(r'[0-9a-f]{8}', run_id)  # MD5 hash is 8 chars in our implementation
        assert "768_long" in run_id


class TestMetadataExport:
    """Test metadata export functionality"""

    def test_export_run_metadata_basic(self):
        """Basic metadata export should work"""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = _generate_run_id(seed=123, prompt="test", profile_name="smoke")
            out_path = Path(tmpdir) / "test.json"

            result = _export_run_metadata(
                run_id=run_id,
                seed=123,
                prompt="test prompt",
                profile_name="smoke",
                scheduler_mode="euler",
                num_steps=10,
                backbone="sdxl",
                out_path=str(out_path),
                negative_prompt="negative",
                vae_fp32_decode=True,
            )

            assert result.exists()
            with open(result) as f:
                metadata = json.load(f)

            assert metadata["run_id"] == run_id
            assert metadata["seed"] == 123
            assert metadata["prompt"] == "test prompt"
            assert metadata["profile"] == "smoke"
            assert metadata["scheduler"] == "euler"
            assert metadata["num_steps"] == 10
            assert metadata["backbone"] == "sdxl"
            assert metadata["negative_prompt"] == "negative"
            assert metadata["vae_fp32_decode"] is True

    def test_export_run_metadata_package_versions(self):
        """Metadata should include package versions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_id = _generate_run_id(seed=123, prompt="test", profile_name="smoke")
            out_path = Path(tmpdir) / "test.json"

            result = _export_run_metadata(
                run_id=run_id,
                seed=123,
                prompt="test prompt",
                profile_name="smoke",
                scheduler_mode="euler",
                num_steps=10,
                backbone="sdxl",
                out_path=str(out_path),
                negative_prompt=None,
                vae_fp32_decode=False,
            )

            with open(result) as f:
                metadata = json.load(f)

            # Check for package versions
            assert "package_versions" in metadata
            assert isinstance(metadata["package_versions"], dict)

            # GPU info
            assert "gpu_name" in metadata


class TestPackageVersion:
    """Test package version detection"""

    def test_get_package_version_valid(self):
        """Should return version for valid package"""
        version = _get_package_version("torch")
        assert version is not None
        assert isinstance(version, str)
        assert len(version) > 0

    def test_get_package_version_invalid(self):
        """Should return None for invalid package"""
        version = _get_package_version("nonexistent_package_xyz_123")
        assert version is None

    def test_get_package_versions_multiple(self):
        """Should return versions for multiple packages"""
        versions = _get_package_versions(["torch", "nonexistent"])
        assert "torch" in versions
        assert versions["torch"] is not None
        assert "nonexistent" in versions
        assert versions["nonexistent"] is None


class TestArgumentParsing:
    """Test argument parsing"""

    def test_parse_args_minimal(self):
        """Should parse minimal required arguments"""
        test_args = [
            "--prompt",
            "test prompt",
            "--out",
            "test.png",
        ]

        with patch("sys.argv", ["prog"] + test_args):
            args = parse_args()
            assert args.prompt == "test prompt"
            assert args.out == "test.png"
            assert args.profile == "smoke"
            assert args.seed == 123

    def test_parse_args_full(self):
        """Should parse all arguments"""
        test_args = [
            "--prompt",
            "test prompt",
            "--seed",
            "456",
            "--profile",
            "768_long",
            "--scheduler",
            "dpm",
            "--out",
            "test.png",
            "--backbone",
            "sdxl",
            "--steps",
            "30",
            "--negative-prompt",
            "bad prompt",
            "--vae-fp32-decode",
        ]

        with patch("sys.argv", ["prog"] + test_args):
            args = parse_args()
            assert args.prompt == "test prompt"
            assert args.seed == 456
            assert args.profile == "768_long"
            assert args.scheduler == "dpm"
            assert args.out == "test.png"
            assert args.backbone == "sdxl"
            assert args.steps == 30
            assert args.negative_prompt == "bad prompt"
            assert args.vae_fp32_decode is True

    def test_parse_args_forbidden_args(self):
        """Should reject forbidden arguments"""
        forbidden_args = [
            "--refiner",
            "/path/to/refiner",
        ]

        with patch("sys.argv", ["prog", "--prompt", "test"] + forbidden_args):
            with pytest.raises(SystemExit):
                parse_args()

    def test_parse_args_controlnet_allowed(self):
        """ControlNet should now be allowed"""
        test_args = [
            "--prompt",
            "test prompt",
            "--controlnet",
            "/path/to/controlnet",
        ]

        with patch("sys.argv", ["prog"] + test_args):
            args = parse_args()
            assert args.controlnet == "/path/to/controlnet"

    def test_parse_args_controlnet_images(self):
        """ControlNet images should be allowed"""
        test_args = [
            "--prompt",
            "test prompt",
            "--controlnet-images",
            "/path/to/image.jpg",
        ]

        with patch("sys.argv", ["prog"] + test_args):
            args = parse_args()
            assert args.controlnet_images == "/path/to/image.jpg"

    def test_parse_args_torch_compile_allowed(self):
        """torch_compile flag should be allowed"""
        test_args = [
            "--prompt",
            "test prompt",
            "--torch-compile",
        ]

        with patch("sys.argv", ["prog"] + test_args):
            args = parse_args()
            assert args.torch_compile is True

    def test_parse_args_with_profile_lcm(self):
        """Should accept LCM profile with CFG 1.7"""
        test_args = [
            "--prompt",
            "test prompt",
            "--profile",
            "768_lcm",
        ]

        with patch("sys.argv", ["prog"] + test_args):
            args = parse_args()
            assert args.profile == "768_lcm"


class TestSmokeTest:
    """Test smoke test functionality"""

    @patch("infer.generate_image.generate_single_image")
    def test_smoke_flag_sets_default_path(self, mock_generate):
        """Smoke flag should work without explicit out path"""
        test_args = [
            "--prompt",
            "smoke test",
            "--smoke",
        ]

        with patch("sys.argv", ["prog"] + test_args):
            from infer.generate_image import main

            # Mock generate_single_image to avoid actual generation
            mock_generate.return_value = Path("/fake/path.png")

            # Run main
            main()

            # Verify generate_single_image was called
            assert mock_generate.called
            call_kwargs = mock_generate.call_args.kwargs
            assert call_kwargs["prompt"] == "smoke test"
            assert call_kwargs["seed"] == 123  # Default smoke test seed


class TestInputValidation:
    """Test input validation"""

    @patch("infer.generate_image._get_pipeline")
    def test_empty_prompt_raises_error(self, mock_get_pipeline):
        """Empty prompt should raise ValueError"""
        with pytest.raises(ValueError, match="Prompt must be a non-empty string"):
            generate_single_image(
                prompt="",
                backbone="sdxl",
                profile_name="smoke",
            )

    @patch("infer.generate_image._get_pipeline")
    def test_invalid_backbone_raises_error(self, mock_get_pipeline):
        """Invalid backbone should raise ValueError"""
        with pytest.raises(ValueError, match="Backbone must be 'sdxl' or 'sd2'"):
            generate_single_image(
                prompt="test prompt",
                backbone="invalid",
                profile_name="smoke",
            )



class TestProfileSystem:
    """Test profile configuration integration"""

    @patch("infer.generate_image._get_pipeline")
    def test_smoke_profile_uses_correct_settings(self, mock_get_pipeline):
        """Smoke profile should use 128x128, 4 steps"""
        # This test verifies the profile system works
        # Full integration test would require actual model loading
        from configs.profiles import get_profile

        profile = get_profile("smoke")
        assert profile.height == 128
        assert profile.width == 128
        assert profile.num_inference_steps == 4

    @patch("infer.generate_image._get_pipeline")
    def test_768_long_profile_uses_correct_settings(self, mock_get_pipeline):
        """768_long profile should use 768x768, 22 steps"""
        from configs.profiles import get_profile

        profile = get_profile("768_long")
        assert profile.height == 768
        assert profile.width == 768
        assert profile.num_inference_steps == 22

    def test_invalid_profile_raises_error(self):
        """Invalid profile name should raise ValueError"""
        from configs.profiles import get_profile

        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("invalid_profile")


class TestSchedulerSystem:
    """Test scheduler configuration integration"""

    def test_get_scheduler_info_valid(self):
        """Should return info for valid scheduler"""
        from configs.scheduler_loader import get_scheduler_info

        info = get_scheduler_info("euler")
        assert info is not None
        assert "class" in info
        assert info["class"] == "EulerDiscreteScheduler"

    def test_get_scheduler_info_invalid(self):
        """Should return None for invalid scheduler"""
        from configs.scheduler_loader import get_scheduler_info

        info = get_scheduler_info("invalid_scheduler")
        assert info is None
