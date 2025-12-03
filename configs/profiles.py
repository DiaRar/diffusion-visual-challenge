"""
Profile configurations for inference.

Based on FINAL.md specifications for different resolution and quality presets.
Each profile defines optimal settings for specific use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Profile:
    """Profile configuration for inference.

    Attributes:
        name: Profile identifier
        height: Image height in pixels
        width: Image width in pixels
        num_inference_steps: Default number of denoising steps
        guidance_scale: CFG scale (higher = more adherence to prompt)
        description: Human-readable description of the profile
    """

    name: str
    height: int
    width: int
    num_inference_steps: int
    guidance_scale: float
    description: str


# Predefined profiles
PROFILES: Final[dict[str, Profile]] = {
    "smoke": Profile(
        name="smoke",
        height=128,
        width=128,
        num_inference_steps=4,
        guidance_scale=6.0,
        description="Smoke test profile: 128x128, 4 steps, CFG 6.0 - Fast testing",
    ),
    "768_long": Profile(
        name="768_long",
        height=768,
        width=768,
        num_inference_steps=22,
        guidance_scale=6.0,
        description="Default profile: 768x768, 22 steps, CFG 6.0 - Best balance",
    ),
    "1024_hq": Profile(
        name="1024_hq",
        height=1024,
        width=1024,
        num_inference_steps=26,
        guidance_scale=6.0,
        description="High quality profile: 1024x1024, 26 steps, CFG 6.0 - Maximum detail",
    ),
    # LCM fast sampling profiles (use with LCM LoRA)
    "768_lcm": Profile(
        name="768_lcm",
        height=768,
        width=768,
        num_inference_steps=5,
        guidance_scale=1.7,
        description="LCM fast profile: 768x768, 5 steps, CFG 1.7 - Use with LCM LoRA",
    ),
    "1024_lcm": Profile(
        name="1024_lcm",
        height=1024,
        width=1024,
        num_inference_steps=6,
        guidance_scale=1.7,
        description="LCM HQ profile: 1024x1024, 6 steps, CFG 1.7 - Use with LCM LoRA",
    ),
}


def get_profile(name: str) -> Profile:
    """
    Get profile configuration by name.

    Args:
        name: Profile identifier

    Returns:
        Profile configuration

    Raises:
        ValueError: If profile name is not found
    """
    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES.keys()))
        raise ValueError(f"Unknown profile: '{name}'. Available profiles: {available}")
    return PROFILES[name]


def list_profiles() -> str:
    """List all available profiles with their descriptions."""
    lines = [f"{profile.name}: {profile.description}" for profile in PROFILES.values()]
    return "\n".join(lines)


def validate_profile(name: str, height: int, width: int) -> bool:
    """
    Validate if dimensions are supported by a profile.

    Args:
        name: Profile name to validate against
        height: Image height
        width: Image width

    Returns:
        True if dimensions are valid for the profile
    """
    try:
        profile = get_profile(name)
        return profile.height == height and profile.width == width
    except ValueError:
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="List available profiles")
    _ = parser.add_argument(  # type: ignore[assignment] (Action return not needed)
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    args = parser.parse_args()  # noqa: F841 (used in if statement below)

    if args.format == "json":  # pyright: ignore[reportAny]
        import json

        # Convert to dict for JSON serialization
        profiles_dict = {
            name: {
                "name": profile.name,
                "height": profile.height,
                "width": profile.width,
                "num_inference_steps": profile.num_inference_steps,
                "guidance_scale": profile.guidance_scale,
                "description": profile.description,
            }
            for name, profile in PROFILES.items()
        }
        print(json.dumps(profiles_dict, indent=2))
    else:
        print("Available Profiles:")
        print("=" * 80)
        print(list_profiles())
        print("=" * 80)
