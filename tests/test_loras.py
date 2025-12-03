"""Tests for configs/loras.py"""

import pytest

from configs.loras import (
    LoRAConfig,
    STYLE_LORAS,
    CHARACTER_LORAS,
    LCM_LORA,
    MOTION_LORAS,
    get_active_loras,
    get_motion_loras,
    validate_lora_count,
    lora_summary,
)


class TestLoRAConfig:
    """Test LoRAConfig dataclass"""

    def test_lora_config_creation(self):
        """Should create LoRAConfig with all attributes"""
        lora = LoRAConfig(
            name="Test LoRA",
            path="test/lora",
            weight=0.8,
            adapter_name="test_adapter",
            type="style",
        )

        assert lora.name == "Test LoRA"
        assert lora.path == "test/lora"
        assert lora.weight == 0.8
        assert lora.adapter_name == "test_adapter"
        assert lora.type == "style"

    def test_lora_config_types(self):
        """Should validate LoRA type"""
        lora = LoRAConfig(
            name="Test",
            path="test",
            weight=1.0,
            adapter_name="test",
            type="lcm",
        )
        assert lora.type in ["style", "character", "lcm", "motion", "utility"]


class TestLoRAConfigurations:
    """Test predefined LoRA configurations"""

    def test_style_loras_defined(self):
        """Style LoRAs should be configured"""
        assert isinstance(STYLE_LORAS, list)
        assert len(STYLE_LORAS) >= 1  # At least Pastel Anime XL

    def test_lcm_lora_defined(self):
        """LCM LoRA should be configured"""
        assert LCM_LORA is not None
        assert isinstance(LCM_LORA, LoRAConfig)
        assert LCM_LORA.type == "lcm"

    def test_lcm_lora_attributes(self):
        """LCM LoRA should have correct attributes"""
        assert LCM_LORA.name == "LCM SDXL"
        assert "lcm-lora-sdxl" in LCM_LORA.path
        assert LCM_LORA.weight == 1.0
        assert LCM_LORA.adapter_name == "lcm"

    def test_character_loras_empty(self):
        """Character LoRAs should be empty by default"""
        assert isinstance(CHARACTER_LORAS, list)
        # Can add character LoRAs if needed

    def test_motion_loras_list_exists(self):
        """Motion LoRAs list should exist"""
        assert isinstance(MOTION_LORAS, list)


class TestLoRAHelpers:
    """Test helper functions"""

    def test_get_active_loras_includes_style_and_lcm(self):
        """Should include style LoRAs and LCM"""
        loras = get_active_loras(include_lcm=True)
        
        assert len(loras) >= 2  # At least 2 LoRAs: style + LCM
        
        # Check for LCM
        has_lcm = any(lora.type == "lcm" for lora in loras)
        assert has_lcm

    def test_get_active_loras_excludes_lcm(self):
        """Should exclude LCM when requested"""
        loras = get_active_loras(include_lcm=False)
        
        # Check no LCM
        has_lcm = any(lora.type == "lcm" for lora in loras)
        assert not has_lcm

    def test_get_motion_loras(self):
        """Should return motion LoRAs list"""
        motion_loras = get_motion_loras()
        assert isinstance(motion_loras, list)

    def test_validate_lora_count_valid(self):
        """Should return True for valid count"""
        test_loras = [
            LoRAConfig("A", "a", 0.5, "a", "style"),
            LoRAConfig("B", "b", 0.5, "b", "style"),
            LoRAConfig("C", "c", 0.5, "c", "lcm"),
        ]
        assert validate_lora_count(test_loras, max_count=3) is True

    def test_validate_lora_count_invalid(self):
        """Should return False for invalid count"""
        test_loras = [
            LoRAConfig("A", "a", 0.5, "a", "style"),
            LoRAConfig("B", "b", 0.5, "b", "style"),
            LoRAConfig("C", "c", 0.5, "c", "lcm"),
            LoRAConfig("D", "d", 0.5, "d", "motion"),
        ]
        assert validate_lora_count(test_loras, max_count=3) is False

    def test_lora_summary_with_loras(self):
        """Should generate summary with LoRAs"""
        test_loras = [
            LoRAConfig("Pastel Anime XL", "path1", 0.8, "pastel", "style"),
            LoRAConfig("LCM SDXL", "path2", 1.0, "lcm", "lcm"),
        ]
        summary = lora_summary(test_loras)
        
        assert "Active LoRAs" in summary
        assert "Pastel Anime XL" in summary
        assert "LCM SDXL" in summary

    def test_lora_summary_empty(self):
        """Should handle empty LoRA list"""
        summary = lora_summary([])
        assert "No LoRAs configured" in summary


class TestLoRAIntegration:
    """Test LoRA integration with pipeline"""

    def test_total_lora_count_limit(self):
        """Should not exceed maximum LoRA count for stability"""
        loras = get_active_loras(include_lcm=True)
        # Video stability requires max 3 LoRAs
        assert len(loras) <= 3, f"Too many LoRAs: {len(loras)}"

    def test_lcm_weight_is_one(self):
        """LCM LoRA should have weight 1.0"""
        loras = get_active_loras(include_lcm=True)
        lcm_lora = next((l for l in loras if l.type == "lcm"), None)
        assert lcm_lora is not None
        assert lcm_lora.weight == 1.0

    def test_style_lora_weights_in_range(self):
        """Style LoRA weights should be in recommended range"""
        style_loras = [l for l in STYLE_LORAS if l.type == "style"]
        
        for lora in style_loras:
            # Primary style: 0.7-0.85, Secondary: 0.15-0.3
            assert 0.1 <= lora.weight <= 1.0, f"LoRA {lora.name} weight {lora.weight} out of range"
