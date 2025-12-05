"""
Generate single images using SDXL/SD2 with stock diffusers.

Optimized for:
- Performance: Pipeline caching, lazy imports, memory efficiency
- Code Quality: Type hints, error handling, validation, logging
- Stability: Robust FP32 VAE decoding to prevent green tint/artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from PIL import Image

# Suppress warnings for cleaner output (TODO: re-enable when ControlNet integration is stable)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

if TYPE_CHECKING:
    from diffusers import StableDiffusionXLPipeline

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pipeline cache for performance
_PIPELINE_CACHE: dict[str, "StableDiffusionXLPipeline"] = {}


def _get_git_hash() -> str:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        logger.warning("Could not get git hash, using 'unknown'")
        return "unknown"


def _generate_run_id(seed: int, prompt: str, profile_name: str) -> str:
    """Generate a unique run ID based on seed, prompt, and profile."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    content = f"{timestamp}_{seed}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}_{profile_name}"
    return content


def _get_package_version(package_name: str) -> str | None:
    """Get the version of a package."""
    try:
        import importlib.metadata as metadata
    except ImportError:
        try:
            import importlib_metadata as metadata
        except ImportError:
            return None
    try:
        return metadata.version(package_name)
    except Exception:
        return None


def _get_package_versions(package_names: list[str]) -> dict[str, str | None]:
    """Get versions of multiple packages."""
    return {name: _get_package_version(name) for name in package_names}


def _export_run_metadata(
    run_id: str,
    seed: int,
    prompt: str,
    profile_name: str,
    scheduler_mode: str,
    num_steps: int,
    backbone: str,
    out_path: str,
    negative_prompt: str | None,
    vae_fp32_decode: bool,
    use_custom_vae: bool,
) -> Path:
    """
    Export run metadata to JSON for reproducibility.

    Args:
        run_id: Unique run identifier
        seed: Random seed used
        prompt: Prompt used for generation
        profile_name: Profile name
        scheduler_mode: Scheduler used
        num_steps: Number of inference steps
        backbone: Model backbone used
        out_path: Output image path
        negative_prompt: Negative prompt used (if any)
        vae_fp32_decode: Whether fp32 decode was used
        use_custom_vae: Whether custom VAE was used

    Returns:
        Path to the exported run.json file
    """
    # Create runs directory
    runs_dir = PROJECT_ROOT / "outputs" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Collect comprehensive metadata for reproducibility
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "git_hash": _get_git_hash(),
        "seed": seed,
        "prompt": prompt,
        "profile": profile_name,
        "scheduler": scheduler_mode,
        "num_steps": num_steps,
        "backbone": backbone,
        "out_path": str(out_path),
        "negative_prompt": negative_prompt,
        "vae_fp32_decode": vae_fp32_decode,
        "use_custom_vae": use_custom_vae,
        "precision": "fp16_mixed",
        "compile_enabled": False,
        "gpu_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "diffusers_version": _get_package_version("diffusers"),
        "package_versions": _get_package_versions(
            ["torch", "torchvision", "diffusers", "transformers", "safetensors"]
        ),
    }

    # Save to file
    run_json_path = runs_dir / f"{run_id}.json"
    with open(run_json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Run metadata exported to: {run_json_path}")
    return run_json_path


def _use_local_unet(pipeline: "StableDiffusionXLPipeline") -> "StableDiffusionXLPipeline":
    """
    Replace the pipeline UNet with the local implementation from `unets/unet_2d.py`.
    Keeps weights and device/dtype identical to the originally loaded UNet.
    """
    try:
        from unets.unet_2d import UNet2DConditionModel
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Local UNet import failed, keeping default: {exc}")
        return pipeline

    base_unet = pipeline.unet
    device = next(base_unet.parameters()).device
    dtype = next(base_unet.parameters()).dtype

    try:
        local_unet_or_tuple = UNet2DConditionModel.from_config(base_unet.config)
        if isinstance(local_unet_or_tuple, tuple):
            local_unet: UNet2DConditionModel = local_unet_or_tuple[0]
        else:
            local_unet = cast(UNet2DConditionModel, local_unet_or_tuple)
        _ = local_unet.load_state_dict(base_unet.state_dict())
        local_unet.to(device=device, dtype=dtype)
        pipeline.unet = local_unet
        logger.info("✓ Swapped pipeline UNet with local `unets.unet_2d.UNet2DConditionModel`")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Failed to swap to local UNet, using default: {exc}")

    return pipeline


def _get_pipeline(
    backbone: str, use_custom_vae: bool = False
) -> "StableDiffusionXLPipeline":
    """
    Get or create a cached diffusion pipeline for SDXL.

    Args:
        backbone: Model backbone (only "sdxl" supported)
        use_custom_vae: Whether to use the custom VAE or default SDXL VAE
    """
    backbone_key = backbone.lower()
    vae_key = "custom_vae" if use_custom_vae else "default_vae"
    cache_key = f"{backbone_key}_{vae_key}_pipeline"

    if cache_key in _PIPELINE_CACHE:
        logger.info(f"Using cached {backbone.upper()} pipeline ({vae_key})")
        return _PIPELINE_CACHE[cache_key]

    # CONSTRAINT CHECK: Only allow SDXL
    ALLOWED_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

    if backbone_key != "sdxl":
        raise RuntimeError(
            f"Unauthorized backbone '{backbone}'. Only SDXL is supported. "
            f"No refiner models allowed."
        )

    logger.info(f"Loading {backbone.upper()} pipeline from HuggingFace Diffusers...")
    logger.info(f"✓ Constraint check passed: Using approved model {ALLOWED_MODEL}")

    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    pipeline = None

    if use_custom_vae:
        vae_path = "../vae/g9_5++.safetensors"
        logger.info(f"Loading custom VAE from: {vae_path}")

        # 1. Force VAE to load in FP32
        vae = AutoencoderKL.from_single_file(
            str(vae_path),
            use_safetensors=True,
            torch_dtype=torch.float32,
        )

        # 2. Force strict SDXL scaling factor
        vae.config.scaling_factor = 0.13025

        # Load SDXL pipeline with custom VAE
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            ALLOWED_MODEL,
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    else:
        logger.info("Using default SDXL VAE...")
        # Load SDXL pipeline with default VAE
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            ALLOWED_MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )

    # 3. Move pipeline to CUDA
    # Note: We do NOT pass torch.float16 to .to() here, to avoid downcasting the VAE
    _ = pipeline.to("cuda")

    # 4. Explicitly ensure VAE stays in Float32 (for both custom and default)
    # This is critical for the robust manual decode strategy
    pipeline.vae.to(dtype=torch.float32)

    # 4.5 Swap in local UNet implementation for any project-specific patches
    pipeline = _use_local_unet(pipeline)

    # Try to enable CPU offload if accelerate is available
    try:
        pipeline.enable_model_cpu_offload()
    except (ImportError, AttributeError):
        logger.info("CPU offload not available. Skipping optimization.")

    # Cache the pipeline
    _PIPELINE_CACHE[cache_key] = pipeline
    logger.info(f"✓ {backbone.upper()} pipeline loaded and cached (VAE: Float32)")

    return pipeline


def generate_single_image(
    prompt: str,
    backbone: str = "sdxl",
    profile_name: str = "smoke",
    scheduler_mode: str = "dpm",
    seed: int = 123,
    out_path: str = "outputs/test.png",
    num_steps: int | None = None,
    negative_prompt: str | None = None,
    vae_fp32_decode: bool = False,
    controlnet_path: str | None = None,
    controlnet_images: str | None = None,
    controlnet_type: str | None = None,  # "pose", "depth", "canny", "lineart"
    control_image: Image.Image | None = None,  # Direct PIL Image input
    controlnet_conditioning_scale: float = 0.8,  # How strongly to follow control map
    torch_compile: bool = False,
    use_custom_vae: bool = False,
) -> Path:
    """
    Generate a single image from a prompt.
    Implements manual FP32 decoding to fix green-tint artifacts.
    """
    logger.info(f"Generating image with prompt: {prompt[:50]}...")

    # Validate inputs
    if not prompt:
        raise ValueError("Prompt must be a non-empty string")

    if backbone.lower() != "sdxl":
        raise ValueError("Backbone must be 'sdxl'")

    # Import configurations
    from configs.profiles import get_profile

    try:
        profile = get_profile(profile_name)
    except ValueError as e:
        logger.error(f"Invalid profile: {e}")
        raise

    # Determine actual steps
    actual_steps = num_steps if num_steps is not None else profile.num_inference_steps

    # Load LoRAs from config
    from configs.loras import get_active_loras, lora_summary
    from configs.schedulers.high_scheduler import apply_best_hq_scheduler

    active_loras = get_active_loras(include_lcm=True)
    pipeline = _get_pipeline(backbone, use_custom_vae=use_custom_vae)

    # Check if LCM LoRA is loaded
    has_lcm = any(lora.type == "lcm" for lora in active_loras)

    # Apply scheduler
    if has_lcm:
        logger.info("LCM LoRA detected - using LCMScheduler")
        try:
            from diffusers import LCMScheduler

            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        except Exception as e:
            logger.error(f"Failed to apply LCMScheduler: {e}")
            raise RuntimeError(f"LCMScheduler initialization failed: {e}") from e
    else:
        pipeline = apply_best_hq_scheduler(pipeline, use_karras_sigmas=True)
        logger.info(f"✓ Applied high-quality scheduler (mode: {scheduler_mode})")

    # Generate run ID and export metadata
    run_id = _generate_run_id(seed, prompt, profile_name)
    _ = _export_run_metadata(
        run_id=run_id,
        seed=seed,
        prompt=prompt,
        profile_name=profile_name,
        scheduler_mode=scheduler_mode,
        num_steps=actual_steps,
        backbone=backbone,
        out_path=out_path,
        negative_prompt=negative_prompt,
        vae_fp32_decode=vae_fp32_decode,
        use_custom_vae=use_custom_vae,
    )

    # Generate image
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Load all configured LoRAs
    # Load all configured LoRAs
    if active_loras:
        logger.info(lora_summary(active_loras))
        try:
            # Clear existing LoRAs using Diffusers only
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

    # Apply ControlNet if provided
    control_image_prepared = None
    if control_image is not None or controlnet_images is not None:
        if controlnet_type is None:
            raise ValueError(
                "controlnet_type must be specified when using control images. "
                "Options: 'pose', 'depth', 'canny', 'lineart'"
            )

        logger.info(f"Loading ControlNet: {controlnet_type}")

        # Import ControlNet loader
        from infer.controlnet_loader import (
            load_controlnet,
            create_controlnet_pipeline,
            prepare_control_image,
        )

        # Load ControlNet model (device managed by enable_model_cpu_offload)
        controlnet = load_controlnet(
            controlnet_type=controlnet_type,
            torch_dtype=torch.float16,
        )

        # Convert pipeline to ControlNet pipeline
        pipeline = create_controlnet_pipeline(pipeline, controlnet)

        # Prepare control image
        if control_image is not None:
            # Direct PIL Image provided
            control_image_prepared = prepare_control_image(
                control_image, width=profile.width, height=profile.height
            )
        elif controlnet_images is not None:
            # Path provided - load image
            control_image_path = Path(controlnet_images)
            if not control_image_path.exists():
                raise FileNotFoundError(
                    f"Control image not found: {control_image_path}"
                )
            control_image_prepared = Image.open(control_image_path)
            control_image_prepared = prepare_control_image(
                control_image_prepared, width=profile.width, height=profile.height
            )

        logger.info(
            f"✓ ControlNet enabled: {controlnet_type} "
            f"(conditioning_scale={controlnet_conditioning_scale})"
        )

    # Enable torch.compile if requested
    if torch_compile and hasattr(pipeline, "unet"):
        try:
            pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")
        except Exception as e:
            logger.warning(f"Failed to enable torch.compile: {e}")

    try:
        # 1. GENERATE LATENTS ONLY
        # We stop the pipeline from decoding to avoid the FP16/FP32 mismatch error
        # and the "green tint" artifact.
        logger.info("Generating latents (FP16)...")
        with torch.no_grad():
            # Prepare pipeline call arguments
            pipeline_kwargs = {
                "prompt": prompt,
                "height": profile.height,
                "width": profile.width,
                "num_inference_steps": actual_steps,
                "guidance_scale": profile.guidance_scale,
                "generator": generator,
                "negative_prompt": negative_prompt,
                "output_type": "latent",  # <--- CRITICAL: Get latents, do not decode yet
            }

            # Add control image if ControlNet is enabled
            if control_image_prepared is not None:
                pipeline_kwargs["image"] = control_image_prepared
                pipeline_kwargs["controlnet_conditioning_scale"] = (
                    controlnet_conditioning_scale
                )
                logger.info(
                    f"Using ControlNet conditioning (scale={controlnet_conditioning_scale})"
                )

            result = pipeline(**pipeline_kwargs)
            latents = result.images  # This is the latent tensor [Batch, C, H, W]

        # 2. MANUAL ROBUST DECODE
        # We manually decode in strict FP32
        logger.info("Decoding latents in strict FP32...")
        with torch.no_grad():
            # Ensure VAE is in Float32
            pipeline.vae.to(dtype=torch.float32)

            # Cast latents to Float32 to match VAE weights
            # This fixes: "Input type (c10::Half) and bias type (float) should be the same"
            latents = latents.to(dtype=torch.float32)

            # Unscale latents (Required for SDXL VAE)
            latents = latents / pipeline.vae.config.scaling_factor

            # Decode
            decoded_image = pipeline.vae.decode(latents).sample

            # Post-process (Clamp and convert to PIL)
            # We do this manually to avoid any hidden auto-casting in VaeImageProcessor
            decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
            # Permute to [Batch, Height, Width, Channels]
            decoded_image = decoded_image.cpu().permute(0, 2, 3, 1).float().numpy()

            # Convert to PIL
            from diffusers.image_processor import VaeImageProcessor

            image = VaeImageProcessor.numpy_to_pil(decoded_image)[0]

        # Save image
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_file)

        logger.info(f"✓ Generated image saved to: {out_file}")
        logger.info(
            (
                f"  Profile: {profile_name}, Steps: {actual_steps}, "
                f"CFG: {profile.guidance_scale}, Scheduler: {scheduler_mode}"
            )
        )

        return out_file

    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        raise RuntimeError(f"Failed to generate image: {e}") from e


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate single image with SDXL using high-quality scheduler"
    )
    _ = parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for image generation",
    )
    _ = parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed (default: 123)",
    )
    _ = parser.add_argument(
        "--profile",
        type=str,
        default="smoke",
        choices=["smoke", "768_long", "1024_hq", "768_lcm", "1024_lcm"],
        help="Profile to use (default: smoke)",
    )
    _ = parser.add_argument(
        "--scheduler",
        type=str,
        default="dpm",
        choices=["dpm"],
        help="Scheduler to use (default: dpm - high-quality DPM++ 2M)",
    )
    _ = parser.add_argument(
        "--out",
        type=str,
        default="outputs/test.png",
        help="Output path (default: outputs/test.png)",
    )
    _ = parser.add_argument(
        "--backbone",
        type=str,
        default="sdxl",
        choices=["sdxl"],
        help="Backbone to use (default: sdxl)",
    )
    _ = parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of inference steps (overrides profile)",
    )
    _ = parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Negative prompt to avoid certain features",
    )
    _ = parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke test (deterministic, validates pipeline)",
    )
    _ = parser.add_argument(
        "--vae-fp32-decode",
        action="store_true",
        help="Decode VAE output in fp32 to avoid fp16 artifacts",
    )
    _ = parser.add_argument(
        "--use-custom-vae",
        action="store_true",
        help="Use custom VAE (xlVAEC_g952) instead of default SDXL VAE",
    )

    # Add support for adapters (ControlNet)
    _ = parser.add_argument(
        "--controlnet",
        type=str,
        default=None,
        help="Path to ControlNet model or model ID (deprecated - use --controlnet-type)",
    )
    _ = parser.add_argument(
        "--controlnet-type",
        type=str,
        default=None,
        choices=["pose", "depth", "canny", "lineart"],
        help="Type of ControlNet to use: 'pose', 'depth', 'canny', or 'lineart'",
    )
    _ = parser.add_argument(
        "--controlnet-images",
        type=str,
        default=None,
        help="Path to conditioning image for ControlNet (pose/depth/edge map)",
    )
    _ = parser.add_argument(
        "--controlnet-scale",
        type=float,
        default=0.8,
        help="ControlNet conditioning scale (0.0-2.0, default: 0.8). Higher = stronger control",
    )
    _ = parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for UNet acceleration",
    )

    # Note: --refiner is still forbidden
    forbidden_args = ["--refiner", "--vae-override", "--vae_override"]

    # Check for forbidden arguments at parse time
    import sys

    for arg in forbidden_args:
        if arg in sys.argv:
            parser.error(
                f"Forbidden argument '{arg}' detected. "
                "Only SDXL/SD2 base models allowed. "
                "No refiner models permitted."
            )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Handle smoke test mode
    if hasattr(args, "smoke") and args.smoke:  # pyright: ignore[reportAny]
        logger.info("Running smoke test...")
        out_path = args.out  # pyright: ignore[reportAny]
    else:
        out_path = args.out  # pyright: ignore[reportAny]

    try:
        _ = generate_single_image(
            prompt=cast(str, args.prompt),
            backbone=cast(str, args.backbone),
            profile_name=cast(str, args.profile),
            scheduler_mode=cast(str, args.scheduler),
            seed=cast(int, args.seed),
            out_path=out_path,  # pyright: ignore[reportAny]
            num_steps=cast(int | None, args.steps),
            negative_prompt=cast(str | None, args.negative_prompt),
            vae_fp32_decode=cast(bool, args.vae_fp32_decode),
            controlnet_path=cast(str | None, args.controlnet),
            controlnet_images=cast(str | None, args.controlnet_images),
            controlnet_type=cast(str | None, getattr(args, "controlnet_type", None)),
            controlnet_conditioning_scale=cast(
                float, getattr(args, "controlnet_scale", 0.8)
            ),
            torch_compile=cast(bool, args.torch_compile),
            use_custom_vae=cast(bool, args.use_custom_vae),
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
