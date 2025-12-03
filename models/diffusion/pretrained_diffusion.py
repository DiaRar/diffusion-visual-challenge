"""
PretrainedDiffusion: Wrapper for diffusers SDXL/SD2 with custom extensions.

This module provides a diffusers-based backbone that can be extended with:
- AnimeLoRA for style adaptation
- TemporalModule for video generation
- Control branches (edge, trajectory)

Per FINAL.md, the pretrained SDXL/SD2 weights are used as the generative backbone,
with additional modules trained on top.

Note: This uses stock diffusers models from HuggingFace for maximum compatibility.
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING

import torch
from PIL import Image

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

logger = logging.getLogger(__name__)


class PretrainedDiffusion:
    """
    Diffusers-based SDXL/SD2 pipeline with custom extensions.

    This class wraps the pretrained diffusers models and provides hooks for:
    - LoRA injection for style adaptation
    - Temporal attention for video generation
    - Control signal injection

    Usage:
        pipeline = PretrainedDiffusion(backbone="sdxl")
        pipeline.load_pretrained()

        # Generate image
        image = pipeline.sample(prompt="anime girl", seed=42)
    """

    MODEL_IDS: dict[str, str] = {
        "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        "sd2": "stabilityai/stable-diffusion-2-base",
    }

    backbone: str
    device: str
    dtype: torch.dtype
    pipeline: DiffusionPipeline | None

    def __init__(
        self,
        backbone: str = "sdxl",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """
        Initialize the pretrained diffusion pipeline.

        Args:
            backbone: Model backbone ("sdxl" or "sd2")
            device: Device to run on (default: "cuda")
            dtype: Data type (default: torch.float16)
        """
        self.backbone = backbone.lower()
        self.device = device
        self.dtype = dtype

        # Validate backbone
        if self.backbone not in self.MODEL_IDS:
            raise ValueError(
                f"Unsupported backbone: {self.backbone}. "
                f"Valid options: {list(self.MODEL_IDS.keys())}"  # pyright: ignore[reportImplicitStringConcatenation]
            )

        self.pipeline = None

    def load_pretrained(self) -> None:
        """Load pretrained models from HuggingFace."""
        logger.info(f"Loading {self.backbone.upper()} pipeline from HuggingFace...")

        try:
            if self.backbone == "sdxl":
                from diffusers import AutoencoderKL
                from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (  # noqa: E501
                    StableDiffusionXLPipeline,
                )

                vae = AutoencoderKL.from_pretrained(
                    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
                )

                self.pipeline = StableDiffusionXLPipeline.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                    self.MODEL_IDS[self.backbone],
                    torch_dtype=self.dtype,
                    variant="fp16",
                    use_safetensors=True,
                    vae=vae,
                )

            elif self.backbone == "sd2":
                from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (  # noqa: E501
                    StableDiffusionPipeline,
                )

                self.pipeline = StableDiffusionPipeline.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
                    self.MODEL_IDS[self.backbone],
                    torch_dtype=self.dtype,
                    variant="fp16",
                    use_safetensors=True,
                )

            # Move to device and enable optimizations
            assert self.pipeline is not None
            _ = self.pipeline.to(self.device, self.dtype)  # pyright: ignore[reportUnknownMemberType]
            self.pipeline.enable_attention_slicing()
            self.pipeline.enable_model_cpu_offload()

            logger.info(f"✓ Loaded {self.backbone.upper()} pipeline successfully")

        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            raise

    def sample(
        self,
        prompt: str,
        height: int = 768,
        width: int = 768,
        num_inference_steps: int = 20,
        guidance_scale: float = 6.0,
        seed: int | None = None,
        negative_prompt: str | None = None,
    ) -> Image.Image:
        """
        Generate an image from a prompt.

        Args:
            prompt: Text prompt for image generation
            height: Image height in pixels (default: 768)
            width: Image width in pixels (default: 768)
            num_inference_steps: Number of denoising steps (default: 20)
            guidance_scale: CFG scale, higher = more adherence to prompt (default: 6.0)
            seed: Random seed for reproducibility (default: None)
            negative_prompt: Negative prompt to avoid features (default: None)

        Returns:
            Generated PIL Image

        Raises:
            RuntimeError: If pipeline not loaded
            ValueError: If prompt is invalid
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pretrained() first.")

        if not prompt:
            raise ValueError("Prompt must be a non-empty string")

        try:
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)

            with torch.no_grad():
                assert self.pipeline is not None
                result = self.pipeline(  # pyright: ignore[reportUnknownVariableType, reportCallIssue]
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    negative_prompt=negative_prompt,
                )

            assert hasattr(result, "images") and result.images, (  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                "Pipeline result missing images"
            )
            images: list[Image.Image] = result.images  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            return images[0]  # pyright: ignore[reportUnknownVariableType]

        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            raise RuntimeError(f"Failed to generate image: {e}") from e

    def load_lora(self, lora_path: str) -> None:
        """
        Load a LoRA adapter for style customization.

        Args:
            lora_path: Path to LoRA weights file

        Note:
            - Requires diffusers >= 0.21.0
            - LoRA is automatically applied in subsequent generation calls
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not loaded. Call load_pretrained() first.")

        try:
            _ = self.pipeline.load_lora_weights(lora_path)  # pyright: ignore[reportAny]
            logger.info(f"✓ Loaded LoRA from {lora_path}")

        except Exception as e:
            logger.error(f"Failed to load LoRA: {e}")
            raise

    def cleanup(self) -> None:
        """Clean up GPU memory."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        _ = gc.collect()
        _ = torch.cuda.empty_cache()
        logger.info("✓ Cleaned up GPU memory")


def create_pipeline(
    backbone: str = "sdxl",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> PretrainedDiffusion:
    """
    Create and load a pretrained diffusion pipeline.

    Args:
        backbone: Model backbone ("sdxl" or "sd2")
        device: Device to run on
        dtype: Data type

    Returns:
        Loaded PretrainedDiffusion pipeline
    """
    pipeline = PretrainedDiffusion(
        backbone=backbone,
        device=device,
        dtype=dtype,
    )
    pipeline.load_pretrained()
    return pipeline


if __name__ == "__main__":
    # Example usage
    pipeline = create_pipeline(backbone="sdxl")

    # Generate image
    image = pipeline.sample(
        prompt="anime girl, beautiful, detailed",
        height=512,
        width=512,
        num_inference_steps=20,
        seed=42,
    )

    # Save image
    image.save("outputs/test_pipeline.png")
    logger.info("✓ Image saved to outputs/test_pipeline.png")
