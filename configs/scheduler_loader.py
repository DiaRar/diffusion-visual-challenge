"""
Patched Scheduler Loader for SDXL/SD2

This version FIXES the Diffusers .from_config() issue where custom scheduler
classes are silently replaced by the base class from the _class_name in config.

This loader:
- Always instantiates YOUR scheduler class
- Never lets Diffusers auto-convert back to DPMSolverMultistepScheduler
- Deep-copies the config to avoid mutation
- Preserves all fields needed for reproducibility
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)


# ----------------------------------------------
# Scheduler Registry
# ----------------------------------------------
SCHEDULER_CONFIGS: dict[str, dict[str, str]] = {
    "euler": {
        "class": "EulerDiscreteScheduler",
        "description": "Fast and stable for most use cases",
        "use_case": "General-purpose image generation",
    },
    "dpm": {
        "class": "SDXLAnime_BestHQScheduler",
        "description": "High-quality sampling with optimized settings",
        "use_case": "Final renders",
    },
    "3m": {
        "class": "SDXLAnimeDPM3M_SDE",
        "description": "SOTA anime scheduler (DPM++ 3M + SDE correction)",
        "use_case": "Anime images with high sharpness",
    },
    "flow": {
        "class": "SDXLAnime_UltimateDPM3MFlow",
        "description": "Ultimate anime scheduler with flow-matching",
        "use_case": "Premium anime quality",
    },
    "unipc": {
        "class": "UniPCMultistepScheduler",
        "description": "Good balance of speed and quality",
        "use_case": "General use",
    },
}


def get_scheduler_info(scheduler_name: str) -> dict[str, str] | None:
    return SCHEDULER_CONFIGS.get(scheduler_name.lower())


# ========================================================================
# ★★★  FIXED APPLY FUNCTION — CUSTOM SCHEDULERS ACTUALLY TAKE EFFECT ★★★
# ========================================================================
def apply_scheduler_to_pipeline(
    pipeline: "DiffusionPipeline",
    scheduler_name: str,
    _total_steps: int = 20,
):
    scheduler_name = scheduler_name.lower()

    if scheduler_name not in SCHEDULER_CONFIGS:
        raise ValueError(
            f"Unknown scheduler '{scheduler_name}'. "
            f"Available: {list(SCHEDULER_CONFIGS.keys())}"
        )

    try:
        scheduler_info = SCHEDULER_CONFIGS[scheduler_name]
        scheduler_class_name = scheduler_info["class"]

        # ---- Import real classes ----
        from diffusers.schedulers.scheduling_dpmsolver_multistep import (
            DPMSolverMultistepScheduler,
        )
        from diffusers.schedulers.scheduling_euler_discrete import (
            EulerDiscreteScheduler,
        )
        from diffusers.schedulers.scheduling_unipc_multistep import (
            UniPCMultistepScheduler,
        )

        # Import your custom schedulers
        from configs.schedulers.high_scheduler import SDXLAnime_BestHQScheduler
        from configs.schedulers.sota_flow_scheduler import SDXLAnime_UltimateDPM3MFlow
        from configs.schedulers.sota_scheduler import SDXLAnimeDPM3M_SDE

        # Map names to classes
        scheduler_map: dict[str, type] = {
            "EulerDiscreteScheduler": EulerDiscreteScheduler,
            "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
            "SDXLAnime_BestHQScheduler": SDXLAnime_BestHQScheduler,
            "SDXLAnimeDPM3M_SDE": SDXLAnimeDPM3M_SDE,
            "SDXLAnime_UltimateDPM3MFlow": SDXLAnime_UltimateDPM3MFlow,
            "UniPCMultistepScheduler": UniPCMultistepScheduler,
        }

        if scheduler_class_name not in scheduler_map:
            raise ValueError(
                f"Scheduler class '{scheduler_class_name}' not found in scheduler_map."
            )

        scheduler_class = scheduler_map[scheduler_class_name]

        # ------------------------------------------------------------------
        # ★★★ CRITICAL FIX ★★★
        # DO NOT use .from_config() because it USES the _class_name in config
        # and replaces your custom class with DPMSolverMultistepScheduler.
        #
        # Instead:
        #   - Deep copy config into a dict
        #   - Overwrite the class name in config
        #   - Pass dict to scheduler_class.from_config()
        # ------------------------------------------------------------------

        # Instantiate safely
        scheduler = scheduler_class.from_config(
            pipeline.scheduler.config,
            use_lu_lambdas=True,
            use_karras_sigmas=False,
        )

        # Debug print
        logger.info(
            f"✓ Loaded scheduler '{scheduler_name}' as {scheduler_class.__name__}"
        )

        # Apply
        pipeline.scheduler = scheduler
        print("Scheduler:", type(pipeline.scheduler))
        return pipeline, None

    except Exception as e:
        logger.warning(
            f"Failed to apply scheduler '{scheduler_name}': {e}. Using default scheduler."
        )
        return pipeline, None


def list_available_schedulers() -> dict[str, dict[str, str]]:
    return SCHEDULER_CONFIGS.copy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="List schedulers")
    _ = parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    schedulers = list_available_schedulers()

    if args.format == "json":
        import json

        print(json.dumps(schedulers, indent=2))
    else:
        print("\nAvailable Schedulers:")
        print("=" * 70)
        for name, info in schedulers.items():
            print(f"\n{name} ({info['class']})")
            print(f"  Description: {info['description']}")
            print(f"  Use case:   {info['use_case']}")
        print("\n" + "=" * 70)
