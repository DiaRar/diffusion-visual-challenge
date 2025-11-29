#!/usr/bin/env python3
"""
Generate evaluation grids for comparing base SDXL vs LoRA results.

Creates visual grids with multiple prompts and seeds for human evaluation.
Outputs CSV template for scoring results.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from infer.generate_image import generate_single_image

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_prompt_suite(prompt_suite_path: str) -> dict[str, Any]:
    """Load prompt suite from JSON file."""
    with open(prompt_suite_path, "r") as f:
        return json.load(f)


def generate_grid_for_prompt(
    prompt_data: dict[str, Any],
    output_dir: Path,
    backbone: str,
    profile: str,
    scheduler: str,
    lora_path: str | None,
    lora_scale: float,
    seed_variations: list[int],
    negative_prompt: str | None = None,
) -> list[Path]:
    """
    Generate images for a single prompt with multiple seeds.

    Returns list of generated image paths.
    """
    all_images: list[Path] = []
    prompt_text = prompt_data["prompt"]

    for seed in seed_variations:
        out_path = output_dir / f"{prompt_data.get('id', 'prompt')}_seed_{seed}.png"

        try:
            generate_single_image(
                prompt=prompt_text,
                backbone=backbone,
                profile_name=profile,
                scheduler_mode=scheduler,
                seed=seed,
                out_path=str(out_path),
                num_steps=None,
                negative_prompt=negative_prompt or prompt_data.get("negative_prompt"),
                lora_path=lora_path,
                lora_scale=lora_scale,
                vae_fp32_decode=False,
            )
            all_images.append(out_path)
            logger.info(f"Generated: {out_path}")
        except Exception as e:
            logger.error(f"Failed to generate image for seed {seed}: {e}")
            continue

    return all_images


def create_comparison_grid(
    image_paths: list[Path],
    output_grid_path: Path,
    title: str | None = None,
) -> None:
    """
    Create a grid image from multiple individual images.

    Args:
        image_paths: List of image file paths (assumed to be same size)
        output_grid_path: Where to save the grid image
        title: Optional title for the grid
    """
    if not image_paths:
        logger.warning("No images to create grid from")
        return

    # Load images
    images = [Image.open(p) for p in image_paths if p.exists()]
    if not images:
        logger.warning("No valid images found")
        return

    # Assume all images are same size
    img_width, img_height = images[0].size

    # Grid dimensions: number of columns = number of images (stacked vertically)
    grid_width = img_width
    grid_height = img_height * len(images)

    # Create grid image
    grid = Image.new("RGB", (grid_width, grid_height), color="white")
    draw = ImageDraw.Draw(grid)

    # Try to load a font
    try:
        # Default font size based on image height
        font_size = max(12, img_height // 20)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Paste images
    for idx, img in enumerate(images):
        y_offset = idx * img_height
        grid.paste(img, (0, y_offset))

        # Add seed label
        seed = image_paths[idx].stem.split("_seed_")[-1]
        label = f"Seed: {seed}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Draw text background for readability
        padding = 5
        draw.rectangle(
            [
                padding,
                y_offset + padding,
                padding + text_width + 2 * padding,
                y_offset + text_height + 2 * padding,
            ],
            fill="black",
        )
        # Draw text
        draw.text(
            (padding + padding, y_offset + padding),
            label,
            fill="white",
            font=font,
        )

    # Add title if provided
    if title:
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width = title_bbox[2] - title_bbox[0]
        title_x = (grid_width - title_width) // 2
        draw.text((title_x, 5), title, fill="black", font=font)

    grid.save(output_grid_path)
    logger.info(f"Grid saved to: {output_grid_path}")


def create_evaluation_csv(
    prompt_suite: dict[str, Any],
    output_csv_path: Path,
    base_dir: Path,
    lora_name: str | None,
) -> None:
    """
    Create CSV template for human evaluation of generated images.

    Args:
        prompt_suite: Loaded prompt suite
        output_csv_path: Where to save the CSV
        base_dir: Base directory containing generated images
        lora_name: Name of LoRA for column headers
    """
    categories = prompt_suite["categories"]
    seed_variations = prompt_suite["testing_protocol"]["seed_variations"]

    with open(output_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        base_col = "Base SDXL" if lora_name is None else f"Base SDXL ({lora_name})"
        lora_col = f"LoRA ({lora_name})" if lora_name else "LoRA"

        writer.writerow(
            [
                "Category",
                "Prompt ID",
                "Prompt",
                "Seed",
                base_col,
                lora_col,
                "Notes",
            ]
        )

        # Write data rows
        for category_name, category_data in categories.items():
            prompts = category_data["prompts"]
            for prompt in prompts:
                prompt_id = prompt.get("id", prompt["prompt"][:30])
                for seed in seed_variations:
                    writer.writerow(
                        [
                            category_name,
                            prompt_id,
                            prompt["prompt"][:100],
                            seed,
                            "",
                            "",
                            "",
                        ]
                    )

    logger.info(f"Evaluation CSV saved to: {output_csv_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate evaluation grids for base SDXL vs LoRA comparison"
    )
    parser.add_argument(
        "--prompt-suite",
        type=str,
        default="configs/prompt_suite.json",
        help="Path to prompt suite JSON (default: configs/prompt_suite.json)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="sdxl",
        choices=["sdxl", "sd2"],
        help="Model backbone (default: sdxl)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="768_long",
        choices=["smoke", "768_long", "1024_hq"],
        help="Profile to use (default: 768_long)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="dpm",
        choices=["euler", "dpm", "unipc"],
        help="Scheduler (default: dpm)",
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="Path to LoRA weights (optional, for comparison)",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=0.9,
        help="LoRA scale (default: 0.9)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/eval_grids",
        help="Output directory (default: outputs/eval_grids)",
    )
    parser.add_argument(
        "--create-csv",
        action="store_true",
        help="Also create CSV template for evaluation",
    )

    args = parser.parse_args()

    # Load prompt suite
    try:
        prompt_suite = load_prompt_suite(args.prompt_suite)
    except Exception as e:
        logger.error(f"Failed to load prompt suite: {e}")
        sys.exit(1)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get test protocol settings
    seed_variations = prompt_suite["testing_protocol"]["seed_variations"]

    # Generate images for each category and prompt
    lora_name = Path(args.lora).stem if args.lora else None

    all_image_paths: list[Path] = []

    for category_name, category_data in prompt_suite["categories"].items():
        logger.info(f"\nProcessing category: {category_name}")

        category_output_dir = output_dir / category_name
        category_output_dir.mkdir(exist_ok=True)

        for prompt_idx, prompt_data in enumerate(category_data["prompts"]):
            prompt_id = prompt_data.get("id", f"prompt_{prompt_idx}")

            # Generate with base model (and LoRA if provided)
            negative_prompt = prompt_data.get("negative_prompt")

            # Generate images
            image_paths = generate_grid_for_prompt(
                prompt_data=prompt_data,
                output_dir=category_output_dir,
                backbone=args.backbone,
                profile=args.profile,
                scheduler=args.scheduler,
                lora_path=args.lora,
                lora_scale=args.lora_scale,
                seed_variations=seed_variations,
                negative_prompt=negative_prompt,
            )

            all_image_paths.extend(image_paths)

            # Create grid for this prompt
            grid_path = category_output_dir / f"{prompt_id}_grid.png"
            prompt_preview = prompt_data["prompt"][:50]
            create_comparison_grid(
                image_paths=image_paths,
                output_grid_path=grid_path,
                title=f"{prompt_preview}...",
            )

    # Create CSV template if requested
    if args.create_csv:
        csv_path = output_dir / "evaluation_template.csv"
        create_evaluation_csv(
            prompt_suite=prompt_suite,
            output_csv_path=csv_path,
            base_dir=output_dir,
            lora_name=lora_name,
        )

    logger.info(f"\n✓ Evaluation grid generation complete!")
    logger.info(f"  Total images: {len(all_image_paths)}")
    logger.info(f"  Output directory: {output_dir}")
    if args.create_csv:
        logger.info(f"  CSV template: {csv_path}")


if __name__ == "__main__":
    main()
