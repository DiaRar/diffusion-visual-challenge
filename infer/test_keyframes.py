"""
Test script for keyframe generation and control map extraction.

Usage:
    python infer/test_keyframes.py --prompt "your prompt here" --seed 123
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from infer.keyframes import generate_keyframe_with_maps


def main() -> None:
    """Main entry point for testing keyframe generation."""
    parser = argparse.ArgumentParser(
        description="Test keyframe generation and control map extraction"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for keyframe generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed (default: 123)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="768_long",
        choices=["smoke", "768_long", "1024_hq", "768_lcm", "1024_lcm"],
        help="Profile to use (default: 768_long)",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of inference steps (overrides profile)",
    )
    parser.add_argument(
        "--use-custom-vae",
        action="store_true",
        help="Use custom VAE",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Skip pose map extraction",
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Skip depth map extraction",
    )
    parser.add_argument(
        "--no-edge",
        action="store_true",
        help="Skip edge map extraction",
    )

    args = parser.parse_args()

    try:
        logger.info("=" * 80)
        logger.info("Testing Keyframe Generation + Control Map Extraction")
        logger.info("=" * 80)

        # Generate keyframe and extract control maps
        keyframe, maps, saved_paths = generate_keyframe_with_maps(
            prompt=args.prompt,
            seed=args.seed,
            profile_name=args.profile,
            extract_pose=not args.no_pose,
            extract_depth=not args.no_depth,
            extract_edge=not args.no_edge,
            use_custom_vae=args.use_custom_vae,
            negative_prompt=args.negative_prompt,
            num_steps=args.steps,
        )

        logger.info("=" * 80)
        logger.info("✓ Keyframe generation and control map extraction complete!")
        logger.info("=" * 80)
        logger.info(f"Keyframe size: {keyframe.size}")
        logger.info(f"Extracted maps: {len(saved_paths)}")
        for map_type, path in saved_paths.items():
            logger.info(f"  - {map_type}: {path}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()

