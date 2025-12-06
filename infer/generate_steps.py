"""
Generate and save all sampler step images (plus final) and assemble an MP4.

Usage example:
python infer/generate_steps.py \
  --prompt "anime girl" \
  --seed 123 \
  --profile 1024_hq \
  --out-folder outputs
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import cast

import torch
from PIL import Image

# Ensure project root on sys.path for local imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports from the main generation script
from infer.generate_image import (  # type: ignore
    _PIPELINE_CACHE,
    _export_run_metadata,
    _generate_run_id,
    _get_pipeline,
)


logger = logging.getLogger(__name__)


def decode_latents(pipeline, latents: torch.Tensor) -> Image.Image:
    """Decode SDXL latents to a PIL image using the pipeline VAE in FP32."""
    vae = pipeline.vae
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure FP32 for stable decode and align devices
    vae.to(device=device, dtype=torch.float32)
    latents = latents.to(device=device, dtype=torch.float32, non_blocking=True)

    # Unscale latents (SDXL-specific scaling)
    latents = latents / vae.config.scaling_factor

    with torch.no_grad():
        decoded = vae.decode(latents).sample

    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = decoded.cpu().permute(0, 2, 3, 1).float().numpy()

    from diffusers.image_processor import VaeImageProcessor

    return VaeImageProcessor.numpy_to_pil(decoded)[0]


def load_profile(profile_name: str):
    from configs.profiles import get_profile

    return get_profile(profile_name)


def load_loras(pipeline, include_lcm: bool = True) -> list:
    from configs.loras import get_active_loras, lora_summary

    active_loras = get_active_loras(include_lcm=include_lcm)
    if not active_loras:
        return []

    logger.info(lora_summary(active_loras))
    try:
        # Clear existing LoRAs if supported
        try:
            if hasattr(pipeline, "unload_lora_weights"):
                pipeline.unload_lora_weights()
        except Exception as e:
            logger.warning(f"Could not unload previous LoRA weights: {e}")

        adapter_names = []
        adapter_weights = []

        for lora_cfg in active_loras:
            logger.info(f"Loading LoRA: {lora_cfg.name}")
            load_kwargs = {"adapter_name": lora_cfg.adapter_name}
            if lora_cfg.weight_name is not None:
                load_kwargs["weight_name"] = lora_cfg.weight_name

            pipeline.load_lora_weights(lora_cfg.path, **load_kwargs)
            adapter_names.append(lora_cfg.adapter_name)
            adapter_weights.append(lora_cfg.weight)

        pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
        logger.info(f"✓ Loaded {len(active_loras)} LoRA(s)")
    except Exception as e:
        logger.error(f"Failed to load LoRAs: {e}")
        raise RuntimeError(f"LoRA loading failed: {e}") from e

    return active_loras


def apply_scheduler(pipeline, scheduler_mode: str, has_lcm: bool):
    mode = scheduler_mode.lower()

    if has_lcm:
        # LCM path always forces LCMScheduler for fast sampling
        logger.info("LCM LoRA detected - using LCMScheduler")
        try:
            from diffusers import LCMScheduler

            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        except Exception as e:
            logger.error(f"Failed to apply LCMScheduler: {e}")
            raise RuntimeError(f"LCMScheduler initialization failed: {e}") from e
        return pipeline

    if mode == "dpm":
        from configs.schedulers.high_scheduler import apply_best_hq_scheduler

        pipeline = apply_best_hq_scheduler(pipeline, use_karras_sigmas=True)
        logger.info("✓ Applied high-quality scheduler (DPM++ 2M)")
        return pipeline

    if mode == "unipc":
        from diffusers import UniPCMultistepScheduler

        pipeline.scheduler = UniPCMultistepScheduler.from_config(
            pipeline.scheduler.config
        )
        logger.info("✓ Applied UniPCMultistepScheduler")
        return pipeline

    if mode in {"dpmpp_sde", "sde"}:
        from diffusers import DPMSolverSDEScheduler

        pipeline.scheduler = DPMSolverSDEScheduler.from_config(
            pipeline.scheduler.config
        )
        logger.info("✓ Applied DPMSolverSDEScheduler (DPM++ SDE)")
        return pipeline

    if mode in {"euler_a", "euler-ancestral"}:
        from diffusers import EulerAncestralDiscreteScheduler

        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipeline.scheduler.config
        )
        logger.info("✓ Applied EulerAncestralDiscreteScheduler")
        return pipeline

    raise ValueError(f"Unsupported scheduler mode: {scheduler_mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and save sampler step images + MP4"
    )
    parser.add_argument("--prompt", required=True, type=str)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--profile",
        type=str,
        default="smoke",
        choices=["smoke", "768_long", "1024_hq", "768_lcm", "1024_lcm"],
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="dpm",
        choices=["dpm", "unipc", "euler_a", "dpmpp_sde", "sde"],
        help=(
            "Scheduler mode (dpm, unipc, euler_a, dpmpp_sde/sde). "
            "LCM LoRA forces LCMScheduler."
        ),
    )
    parser.add_argument("--out-folder", type=str, default="outputs")
    parser.add_argument("--steps", type=int, default=None, help="Override steps")
    parser.add_argument("--negative-prompt", type=str, default=None)
    parser.add_argument("--backbone", type=str, default="sdxl", choices=["sdxl"])
    parser.add_argument("--use-custom-vae", action="store_true")
    parser.add_argument("--fps", type=int, default=16, help="FPS for MP4 (default: 16)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Free any previous cached pipelines on GPU to avoid OOM when rerunning
    _PIPELINE_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    profile = load_profile(args.profile)
    actual_steps = args.steps if args.steps is not None else profile.num_inference_steps

    pipeline = _get_pipeline(args.backbone, use_custom_vae=bool(args.use_custom_vae))

    active_loras = load_loras(pipeline, include_lcm=True)
    has_lcm = any(lora.type == "lcm" for lora in active_loras)
    pipeline = apply_scheduler(pipeline, args.scheduler, has_lcm)

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # Collect latents at every step
    step_latents: list[torch.Tensor] = []

    def callback(step: int, timestep: int, latents: torch.Tensor) -> None:
        # store on CPU to reduce GPU memory pressure
        step_latents.append(latents.detach().to("cpu"))

    run_id = _generate_run_id(args.seed, args.prompt, args.profile)

    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sched_tag = args.scheduler.lower()
    base_out = Path(args.out_folder) / f"{sched_tag}_gen_steps_{timestamp}"
    base_out.mkdir(parents=True, exist_ok=True)

    # Export metadata (reuse the existing helper)
    _export_run_metadata(
        run_id=run_id,
        seed=args.seed,
        prompt=args.prompt,
        profile_name=args.profile,
        scheduler_mode=args.scheduler,
        num_steps=actual_steps,
        backbone=args.backbone,
        out_path=str(base_out / "final.png"),
        negative_prompt=args.negative_prompt,
        vae_fp32_decode=True,
        use_custom_vae=bool(args.use_custom_vae),
    )

    pipeline_kwargs = {
        "prompt": args.prompt,
        "height": profile.height,
        "width": profile.width,
        "num_inference_steps": actual_steps,
        "guidance_scale": profile.guidance_scale,
        "generator": generator,
        "negative_prompt": args.negative_prompt,
        "output_type": "latent",
        "callback": callback,
        "callback_steps": 1,
    }

    logger.info("Generating latents with step callbacks...")
    with torch.no_grad():
        result = pipeline(**pipeline_kwargs)

    final_latents = result.images

    # Decode and save each step
    logger.info(f"Decoding and saving {len(step_latents)} step images...")
    for idx, lat in enumerate(step_latents):
        img = decode_latents(pipeline, lat)
        img.save(base_out / f"step_{idx:03d}.png")

    # Decode final (ensures parity with pipeline output)
    final_img = decode_latents(pipeline, final_latents)
    final_path = base_out / "final.png"
    final_img.save(final_path)
    logger.info(f"✓ Final image saved to: {final_path}")

    # Assemble MP4 using ffmpeg
    mp4_path = base_out / "steps.mp4"
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(base_out / "step_%03d.png"),
        "-pix_fmt",
        "yuv420p",
        str(mp4_path),
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"✓ MP4 saved to: {mp4_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr.decode('utf-8', errors='ignore')}")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()

