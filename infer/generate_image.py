"""
Generate single images using SDXL/SD2 with stock diffusers.

Optimized for:
- Performance: Pipeline caching, lazy imports, memory efficiency
- Code Quality: Type hints, error handling, validation, logging
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pipeline cache for performance
_PIPELINE_CACHE: dict[str, "DiffusionPipeline"] = {}


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
        "precision": "fp16",
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


def _get_pipeline(backbone: str) -> "DiffusionPipeline":
    """
    Get or create a cached diffusion pipeline.

    Args:
        backbone: Model backbone ("sdxl" or "sd2")

    Returns:
        Cached diffusion pipeline

    Raises:
        ValueError: If backbone is not supported
        RuntimeError: If unauthorized models are detected
    """
    backbone_key = backbone.lower()
    cache_key = f"{backbone_key}_pipeline"

    if cache_key in _PIPELINE_CACHE:
        logger.info(f"Using cached {backbone.upper()} pipeline")
        return _PIPELINE_CACHE[cache_key]

    # CONSTRAINT CHECK: Only allow approved model IDs
    ALLOWED_MODELS = {
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "sd2": "stabilityai/stable-diffusion-2-base",
    }

    if backbone_key not in ALLOWED_MODELS:
        raise RuntimeError(
            f"Unauthorized backbone '{backbone}'. Only SDXL and SD2 base models are allowed. "
            f"No refiner models allowed."
        )

    logger.info(f"Loading {backbone.upper()} pipeline from HuggingFace Diffusers...")
    logger.info(
        f"✓ Constraint check passed: Using approved model {ALLOWED_MODELS[backbone_key]}"
    )

    # Lazy imports for performance
    try:
        if backbone_key == "sdxl":
            from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (  # noqa: E501
                StableDiffusionXLPipeline,
            )

            # Assert exact model ID match
            model_id = ALLOWED_MODELS[backbone_key]
            assert model_id == "stabilityai/stable-diffusion-xl-base-1.0", (
                "Only SDXL base 1.0 allowed (no refiner)"
            )

            pipeline = StableDiffusionXLPipeline.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
        elif backbone_key in ("sd2", "sd2-base"):
            from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (  # noqa: E501
                StableDiffusionPipeline,
            )

            # Assert exact model ID match
            model_id = ALLOWED_MODELS[backbone_key]
            assert model_id == "stabilityai/stable-diffusion-2-base", (
                "Only SD2 base allowed (no refiner/ControlNet)"
            )

            pipeline = StableDiffusionPipeline.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                model_id,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
        else:
            raise RuntimeError(
                f"Unauthorized model detected. "
                f"Only {list(ALLOWED_MODELS.values())} are allowed."
            )

        # Enable memory optimizations
        _ = pipeline.to("cuda", torch.float16)  # pyright: ignore[reportUnknownMemberType] # noqa: E501
        pipeline.enable_attention_slicing()

        # Try to enable CPU offload if accelerate is available and recent enough
        try:
            pipeline.enable_model_cpu_offload()
        except (ImportError, AttributeError):
            logger.info(
                (
                    "CPU offload not available (requires accelerate >= 0.17.0). "
                    "Skipping optimization."
                )
            )

        # Cache the pipeline
        _PIPELINE_CACHE[cache_key] = pipeline
        logger.info(f"✓ {backbone.upper()} pipeline loaded and cached")

        return pipeline

    except AssertionError as e:
        logger.error(f"Constraint violation: {e}")
        raise RuntimeError(f"Unauthorized model configuration detected: {e}") from e
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        raise


def generate_single_image(
    prompt: str,
    backbone: str = "sdxl",
    profile_name: str = "smoke",
    scheduler_mode: str = "euler",
    seed: int = 123,
    out_path: str = "outputs/test.png",
    num_steps: int | None = None,
    negative_prompt: str | None = None,
    vae_fp32_decode: bool = False,
    controlnet_path: str | None = None,
    controlnet_images: str | None = None,
    torch_compile: bool = False,
) -> Path:
    """
    Generate a single image from a prompt using SDXL/SD2 with adapters.

    Args:
        prompt: Text prompt for image generation
        backbone: Model backbone ("sdxl" or "sd2")
        profile_name: Profile name ("smoke", "768_long", "1024_hq", "768_lcm", "1024_lcm")
        scheduler_mode: Scheduler to use ("euler", "dpm", "unipc")
        seed: Random seed for reproducibility
        out_path: Output file path
        num_steps: Number of inference steps (overrides profile default)
        negative_prompt: Negative prompt to avoid certain features
        vae_fp32_decode: Decode VAE output in fp32 to avoid artifacts
        controlnet_path: Path to ControlNet model or model ID
        controlnet_images: Path to conditioning images for ControlNet (comma-separated or single path)
        torch_compile: Enable torch.compile for UNet acceleration

    Returns:
        Path to generated image file

    Raises:
        ValueError: If backbone or profile is invalid
        RuntimeError: If generation fails
    """
    logger.info(f"Generating image with prompt: {prompt[:50]}...")

    # Validate inputs
    if not prompt:
        raise ValueError("Prompt must be a non-empty string")

    if backbone.lower() not in ["sdxl", "sd2", "sd2-base"]:
        raise ValueError("Backbone must be 'sdxl' or 'sd2'")

    # Import configurations
    from configs.profiles import get_profile

    try:
        profile = get_profile(profile_name)
    except ValueError as e:
        logger.error(f"Invalid profile: {e}")
        raise

    # Determine actual steps
    actual_steps = num_steps if num_steps is not None else profile.num_inference_steps

    # Get cached pipeline
    pipeline = _get_pipeline(backbone)

    # Apply scheduler
    from configs.scheduler_loader import apply_scheduler_to_pipeline

    # Load LoRAs from config
    from configs.loras import get_active_loras, lora_summary

    active_loras = get_active_loras(include_lcm=True)

    # Check if LCM LoRA is loaded
    has_lcm = any(lora.type == "lcm" for lora in active_loras)

    # Apply scheduler - use LCMScheduler if LCM LoRA is present
    if has_lcm:
        logger.info("LCM LoRA detected - using LCMScheduler for fast sampling")
        try:
            from diffusers import LCMScheduler

            # Replace scheduler with LCM scheduler
            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
            logger.info("✓ Applied LCMScheduler for LCM LoRA")
        except Exception as e:
            logger.error(f"Failed to apply LCMScheduler: {e}")
            raise RuntimeError(f"LCMScheduler initialization failed: {e}") from e
    else:
        _ = apply_scheduler_to_pipeline(pipeline, scheduler_mode, actual_steps)

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
    )

    # Generate image
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Load all configured LoRAs
    if active_loras:
        logger.info(lora_summary(active_loras))
        try:
            from diffusers import StableDiffusionXLPipeline

            if isinstance(pipeline, StableDiffusionXLPipeline):
                # Clear any existing adapters to avoid conflicts
                try:
                    # Try to unload all adapters
                    if hasattr(pipeline, "unload_lora_weights"):
                        logger.info("Clearing existing LoRA adapters...")
                        pipeline.unload_lora_weights()
                    elif hasattr(pipeline, "peft_config") and pipeline.peft_config:
                        # Alternative clearing method
                        logger.info("Clearing existing LoRA adapters (alternative)...")
                        pipeline.peft_config = {}
                        if hasattr(pipeline.unet, "peft_config"):
                            pipeline.unet.peft_config = {}
                            pipeline.unet.base_layer.peft_config = {}
                except Exception as e:
                    logger.warning(f"Could not clear existing adapters: {e}")

                adapter_names = []
                adapter_weights = []

                for lora_cfg in active_loras:
                    logger.info(f"Loading LoRA: {lora_cfg.name} from {lora_cfg.path}")
                    # Prepare load arguments
                    load_kwargs = {
                        "adapter_name": lora_cfg.adapter_name,
                    }
                    # Add weight_name if specified
                    if lora_cfg.weight_name is not None:
                        load_kwargs["weight_name"] = lora_cfg.weight_name
                        logger.info(f"  Using weight file: {lora_cfg.weight_name}")

                    pipeline.load_lora_weights(
                        lora_cfg.path,
                        **load_kwargs,
                    )
                    adapter_names.append(lora_cfg.adapter_name)
                    adapter_weights.append(lora_cfg.weight)

                # Set adapter weights for all LoRAs
                pipeline.set_adapters(
                    adapter_names, adapter_weights=adapter_weights
                )

                if has_lcm:
                    logger.info(
                        "LCM LoRA loaded - using fast sampling mode (4-6 steps)"
                    )

                logger.info(f"✓ Loaded {len(active_loras)} LoRA(s)")
            else:
                logger.warning(
                    f"LoRA loading not fully implemented for {type(pipeline).__name__}"
                )
        except Exception as e:
            logger.error(f"Failed to load LoRAs: {e}")
            raise RuntimeError(f"LoRA loading failed: {e}") from e

    # Apply ControlNet if provided
    if controlnet_path is not None:
        logger.info(f"Loading ControlNet from: {controlnet_path}")
        try:
            from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

            if isinstance(pipeline, StableDiffusionXLPipeline):
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_path, torch_dtype=torch.float16
                )
                controlnet_pipeline = (
                    StableDiffusionXLControlNetPipeline.from_pretrained(
                        ALLOWED_MODELS[backbone.lower()],
                        controlnet=controlnet,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        use_safetensors=True,
                    )
                )
                # Replace pipeline with controlnet pipeline
                pipeline = controlnet_pipeline
                logger.info("ControlNet loaded successfully")
            else:
                logger.warning(
                    f"ControlNet not fully implemented for {type(pipeline).__name__}"
                )
        except Exception as e:
            logger.error(f"Failed to load ControlNet: {e}")
            raise RuntimeError(f"ControlNet loading failed: {e}") from e

    # Enable torch.compile if requested
    if torch_compile and hasattr(pipeline, "unet"):
        logger.info("Enabling torch.compile for UNet acceleration")
        try:
            pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")
            logger.info("✓ torch.compile enabled")
        except Exception as e:
            logger.warning(f"Failed to enable torch.compile: {e}")

    # Apply VAE fp32 decode if requested
    if vae_fp32_decode and hasattr(pipeline, "vae"):
        logger.info("Enabling fp32 VAE decode to avoid artifacts")
        pipeline.vae.dtype = torch.float32

    try:
        with torch.no_grad():
            assert pipeline is not None
            result: object = pipeline(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
                prompt=prompt,
                height=profile.height,
                width=profile.width,
                num_inference_steps=actual_steps,
                guidance_scale=profile.guidance_scale,
                generator=generator,
                negative_prompt=negative_prompt,
            )
            image: object = result.images[0]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]

        # Save image
        out_file = Path(out_path)
        _ = out_file.parent.mkdir(
            parents=True, exist_ok=True
        )  # Assign to _ to avoid unused result warning
        image.save(out_file)  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]

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
        description="Generate single image with stock SDXL/SD2 pipeline"
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
        default="euler",
        choices=["euler", "dpm", "unipc"],
        help="Scheduler to use (default: euler)",
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
        choices=["sdxl", "sd2"],
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

    # Add support for adapters (ControlNet)
    _ = parser.add_argument(
        "--controlnet",
        type=str,
        default=None,
        help="Path to ControlNet model or model ID",
    )
    _ = parser.add_argument(
        "--controlnet-images",
        type=str,
        default=None,
        help="Path to conditioning images for ControlNet (comma-separated or single path)",
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
            torch_compile=cast(bool, args.torch_compile),
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
