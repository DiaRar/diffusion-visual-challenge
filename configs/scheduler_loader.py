"""
Scheduler configuration loader for SDXL/SD2.

Uses stock diffusers schedulers for maximum compatibility.
Optimized for performance and code quality.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)

# Scheduler configurations
SCHEDULER_CONFIGS: dict[str, dict[str, str]] = {
    "euler": {
        "class": "EulerDiscreteScheduler",
        "description": "Fast and stable for most use cases",
        "use_case": "General-purpose image generation",
    },
    "dpm": {
        "class": "DPMSolverMultistepScheduler",
        "description": "High-quality sampling for detailed images",
        "use_case": "Final renders, maximum fidelity",
    },
    "unipc": {
        "class": "UniPCMultistepScheduler",
        "description": "Alternative solver with good quality-speed balance",
        "use_case": "Balanced speed and quality",
    },
    "edm_dpm": {
        "class": "EDMDPMSolverMultistepScheduler",
        "description": "EDM variant of DPM solver for improved sampling",
        "use_case": "High-quality EDM-based sampling",
    },
}


def get_scheduler_info(scheduler_name: str) -> dict[str, str] | None:
    """
    Get scheduler information by name.

    Args:
        scheduler_name: Name of scheduler ("euler", "dpm", or "unipc")

    Returns:
        Scheduler configuration dict or None if not found
    """
    return SCHEDULER_CONFIGS.get(scheduler_name.lower())


def apply_scheduler_to_pipeline(
    pipeline: "DiffusionPipeline",
    scheduler_name: str,
    _total_steps: int = 20,  # noqa: F841 (kept for API compatibility)
) -> tuple["DiffusionPipeline", None]:
    """
    Apply scheduler configuration to a Diffusers pipeline.

    Args:
        pipeline: Diffusers pipeline to configure
        scheduler_name: Name of scheduler to use
        total_steps: Number of inference steps (parameter retained for API compatibility)

    Returns:
        Tuple of (pipeline, None)

    Raises:
        ValueError: If scheduler_name is not supported
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name not in SCHEDULER_CONFIGS:
        raise ValueError(
            (
                f"Unknown scheduler: {scheduler_name}. "
                f"Available options: {list(SCHEDULER_CONFIGS.keys())}"
            )
        )

    try:
        scheduler_info = SCHEDULER_CONFIGS[scheduler_name]
        scheduler_class_name = scheduler_info["class"]

        # Import diffusers schedulers
        from diffusers.schedulers.scheduling_dpmsolver_multistep import (
            DPMSolverMultistepScheduler,
        )
        from diffusers.schedulers.scheduling_edm_dpmsolver_multistep import (
            EDMDPMSolverMultistepScheduler,
        )
        from diffusers.schedulers.scheduling_euler_discrete import (
            EulerDiscreteScheduler,
        )
        from diffusers.schedulers.scheduling_unipc_multistep import (
            UniPCMultistepScheduler,
        )

        # Map class names to actual classes
        scheduler_map: dict[str, type] = {
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "EDMDPMSolverMultistepScheduler": EDMDPMSolverMultistepScheduler,
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "UniPCMultistepScheduler": UniPCMultistepScheduler,
        }

        if scheduler_class_name not in scheduler_map:
            raise ValueError(f"No mapping for scheduler class: {scheduler_class_name}")

        scheduler_class = scheduler_map[scheduler_class_name]

        # Create and apply scheduler with optimizations
        scheduler: object = scheduler_class.from_config(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            pipeline.scheduler.config,  # pyright: ignore[reportAny]
            use_karras_sigmas=True,
        )

        pipeline.scheduler = scheduler
        logger.info(f"✓ Applied {scheduler_name} scheduler")

        return pipeline, None

    except Exception as e:
        logger.warning(
            (
                f"Could not apply scheduler '{scheduler_name}': {e}. "
                "Using default scheduler."
            )
        )
        return pipeline, None


def list_available_schedulers() -> dict[str, dict[str, str]]:
    """Get all available schedulers information."""
    return SCHEDULER_CONFIGS.copy()


if __name__ == "__main__":
    # Print all available schedulers
    import argparse

    parser = argparse.ArgumentParser(description="List available schedulers")
    _ = parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    args = parser.parse_args()  # noqa: F841 (used in if statement below)

    schedulers = list_available_schedulers()

    if cast(str, args.format) == "json":
        import json

        print(json.dumps(schedulers, indent=2))
    else:
        print("\nAvailable Schedulers:")
        print("=" * 70)

        for name, info in schedulers.items():
            print(f"\n{name} ({info['class']})")
            print(f"  Description: {info['description']}")
            print(f"  Use case: {info['use_case']}")

        print("\n" + "=" * 70)
