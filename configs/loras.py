"""
LoRA configurations for SDXL anime pipeline.

Based on PIPELINE.md specifications:
- Primary Style LoRA: 0.7-0.85 weight
- Secondary Flat-Color LoRA: 0.15-0.3 weight
- Optional Character LoRA: 0.6-0.8 weight
- LCM LoRA: 1.0 weight (for fast sampling)

Do not exceed 3 total LoRAs for video stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LoRAConfig:
    """Single LoRA configuration.

    Attributes:
        name: Human-readable name for logging
        path: HuggingFace model ID or local path to .safetensors
        weight: LoRA weight/scale (0.0-2.0)
        adapter_name: Unique identifier for this adapter in the pipeline
        type: LoRA type for categorization
    """
    name: str
    path: str
    weight: float
    adapter_name: str
    type: Literal["style", "character", "lcm", "motion", "utility"]


# =============================================================================
# LoRA Presets - Edit these to change which LoRAs are loaded
# =============================================================================

# Style LoRAs - anime visual style
STYLE_LORAS: list[LoRAConfig] = [
    # Primary anime style - uncomment one of these:
    # LoRAConfig(
    #     name="Pastel Anime XL",
    #     path="Linaqruf/pastel-anime-xl-lora",  # HuggingFace ID
    #     weight=0.8,
    #     adapter_name="pastel_anime",
    #     type="style",
    # ),
    # LoRAConfig(
    #     name="Anime Flat Color XL",
    #     path="2vXpSwA7/anime-flat-color-xl",
    #     weight=0.25,
    #     adapter_name="flat_color",
    #     type="style",
    # ),
]

# Character LoRAs - identity consistency (use max 1)
CHARACTER_LORAS: list[LoRAConfig] = [
    # Add character-specific LoRA here if needed
    # LoRAConfig(
    #     name="Character Name",
    #     path="path/to/character.safetensors",
    #     weight=0.7,
    #     adapter_name="character",
    #     type="character",
    # ),
]

# LCM LoRA - for fast 4-6 step sampling
LCM_LORA: LoRAConfig | None = None
# Uncomment to enable LCM fast sampling:
# LCM_LORA = LoRAConfig(
#     name="LCM SDXL",
#     path="latent-consistency/lcm-lora-sdxl",
#     weight=1.0,
#     adapter_name="lcm",
#     type="lcm",
# )

# Motion LoRAs - for AnimateDiff (camera pan, zoom, etc.)
MOTION_LORAS: list[LoRAConfig] = [
    # Add motion LoRAs for video generation
    # LoRAConfig(
    #     name="Camera Pan",
    #     path="path/to/pan.safetensors",
    #     weight=1.0,
    #     adapter_name="motion_pan",
    #     type="motion",
    # ),
]


def get_active_loras(include_lcm: bool = True) -> list[LoRAConfig]:
    """Get all active LoRA configurations.

    Args:
        include_lcm: Whether to include LCM LoRA if configured

    Returns:
        List of active LoRA configurations
    """
    loras: list[LoRAConfig] = []

    # Add style LoRAs
    loras.extend(STYLE_LORAS)

    # Add character LoRAs (should be max 1)
    loras.extend(CHARACTER_LORAS)

    # Add LCM if enabled
    if include_lcm and LCM_LORA is not None:
        loras.append(LCM_LORA)

    return loras


def get_motion_loras() -> list[LoRAConfig]:
    """Get motion LoRAs for AnimateDiff."""
    return MOTION_LORAS.copy()


def validate_lora_count(loras: list[LoRAConfig], max_count: int = 3) -> bool:
    """Validate that LoRA count doesn't exceed maximum for stability.

    Args:
        loras: List of LoRA configurations
        max_count: Maximum allowed LoRAs (default 3 for video stability)

    Returns:
        True if count is valid
    """
    return len(loras) <= max_count


def lora_summary(loras: list[LoRAConfig]) -> str:
    """Generate a summary string of active LoRAs for logging."""
    if not loras:
        return "No LoRAs configured"

    lines = [f"Active LoRAs ({len(loras)}):"]
    for lora in loras:
        lines.append(f"  - {lora.name} ({lora.type}): weight={lora.weight}")
    return "\n".join(lines)
