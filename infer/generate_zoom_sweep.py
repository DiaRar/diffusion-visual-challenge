"""
Generate 10 images with zoom LoRA weight sweep from -1 to 1.

Optimized: Loads pipeline and LoRAs once, only updates adapter weights.

Usage:
    python infer/generate_zoom_sweep.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.loras import get_active_loras, lora_summary, LoRAConfig
from configs.profiles import get_profile
from configs.schedulers.high_scheduler import apply_best_hq_scheduler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_pipeline_with_loras():
    """Load SDXL pipeline with all LoRAs once."""
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    ALLOWED_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

    logger.info("Loading SDXL pipeline...")

    # Load custom VAE in FP32
    vae_path = "../vae/g9_5++.safetensors"
    vae = AutoencoderKL.from_single_file(
        str(vae_path),
        use_safetensors=True,
        torch_dtype=torch.float32,
    )
    vae.config.scaling_factor = 0.13025  # type: ignore[union-attr]

    # Load pipeline
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        ALLOWED_MODEL,
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    _ = pipeline.to("cuda")
    _ = pipeline.vae.to(dtype=torch.float32)

    try:
        pipeline.enable_model_cpu_offload()
    except (ImportError, AttributeError):
        pass

    # Apply scheduler
    pipeline = apply_best_hq_scheduler(pipeline, use_karras_sigmas=True)

    # Load all LoRAs once
    active_loras = get_active_loras(include_lcm=False)
    logger.info(lora_summary(active_loras))

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

    return pipeline, adapter_names, active_loras


def generate_with_zoom_weight(
    pipeline,
    adapter_names: list[str],
    active_loras: list[LoRAConfig],
    zoom_weight: float,
    prompt: str,
    negative_prompt: str,
    seed: int,
    profile_name: str,
    num_steps: int,
    out_path: str,
) -> Path:
    """Generate a single image with specified zoom weight."""
    from diffusers.image_processor import VaeImageProcessor

    profile = get_profile(profile_name)

    # Build new weights list, replacing zoom adapter weight
    new_weights = []
    for lora in active_loras:
        if lora.adapter_name == "zoom":
            new_weights.append(zoom_weight)
        else:
            new_weights.append(lora.weight)

    # Update adapter weights (fast - no reloading)
    pipeline.set_adapters(adapter_names, adapter_weights=new_weights)

    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Generate latents
    with torch.no_grad():
        result = pipeline(
            prompt=prompt,
            height=profile.height,
            width=profile.width,
            num_inference_steps=num_steps,
            guidance_scale=profile.guidance_scale,
            generator=generator,
            negative_prompt=negative_prompt,
            output_type="latent",
        )
        latents = result.images

    # Decode in FP32
    with torch.no_grad():
        pipeline.vae.to(dtype=torch.float32)
        latents = latents.to(dtype=torch.float32)
        latents = latents / pipeline.vae.config.scaling_factor
        decoded_image = pipeline.vae.decode(latents).sample
        decoded_image = (decoded_image / 2 + 0.5).clamp(0, 1)
        decoded_image = decoded_image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = VaeImageProcessor.numpy_to_pil(decoded_image)[0]

    # Save
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_file)

    return out_file


def main() -> None:
    """Generate 10 images with zoom weight sweep from -1 to 1."""

    # Configuration
    prompt = (
        "anime girl with blue hair, beautiful detailed eyes, cel shading, "
        "clean lines, masterpiece, cinematic volumetric god-rays, "
        "subsurface glow diffusion, balanced color harmony, 8k quality, "
        "perfect composition, beautiful godess"
    )
    negative_prompt = "blurry, low quality, bad anatomy"
    seed = 123
    profile = "1024_hq"
    steps = 20

    # Generate 10 weights uniformly from -1 to 1
    zoom_weights: list[float] = [float(x) for x in np.linspace(-1, 1, 31)]

    print(f"Generating {len(zoom_weights)} images with zoom weights:")
    print(f"  {zoom_weights}")
    print("=" * 60)

    # Load pipeline and LoRAs ONCE
    print("\n[SETUP] Loading pipeline and LoRAs (one-time)...")
    pipeline, adapter_names, active_loras = load_pipeline_with_loras()
    print("✓ Pipeline ready!\n")

    for i, zw in enumerate(zoom_weights):
        print(f"[{i+1}/10] Generating with zoom_weight = {zw:.3f}...")

        weight_str = f"{zw:+.2f}".replace(".", "p").replace("-", "neg").replace("+", "pos")
        out_path = f"outputs/zoom_sweep/zoom_{i:02d}_w{weight_str}.png"

        try:
            _ = generate_with_zoom_weight(
                pipeline=pipeline,
                adapter_names=adapter_names,
                active_loras=active_loras,
                zoom_weight=zw,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                profile_name=profile,
                num_steps=steps,
                out_path=out_path,
            )
            print(f"  ✓ Saved: {out_path}")
        except Exception as e:
            print(f"  ✗ Failed for weight {zw}: {e}")
            raise

    print("\n" + "=" * 60)
    print("✓ All 10 images generated successfully!")
    print(f"  Output directory: outputs/zoom_sweep/")


if __name__ == "__main__":
    main()
