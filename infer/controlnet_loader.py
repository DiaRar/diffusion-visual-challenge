"""
ControlNet loading and management for SDXL pipeline.

Supports OpenPose, Depth, and Canny/LineArt ControlNets for video stability.
Used by both generate_image.py and generate_video.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from PIL import Image

if TYPE_CHECKING:
    from diffusers import (
        ControlNetModel,
        MultiControlNetModel,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLPipeline,
    )

logger = logging.getLogger(__name__)

# ControlNet model IDs for SDXL
# Using xinsir/controlnet-union-sdxl-1.0 - the best SDXL ControlNet model
# It's a multi-control model supporting: depth, canny, softedge, lineart, openpose, etc.
# For video stability, DEPTH and CANNY (edge) are the most reliable control types.
CONTROLNET_MODELS = {
    "depth": "xinsir/controlnet-union-sdxl-1.0",   # BEST for video - maintains spatial structure
    "canny": "xinsir/controlnet-union-sdxl-1.0",   # BEST for edges - preserves outlines/details
    "edge": "xinsir/controlnet-union-sdxl-1.0",    # Alias for canny
    "lineart": "xinsir/controlnet-union-sdxl-1.0", # Good for anime line art
    "pose": "xinsir/controlnet-union-sdxl-1.0",    # Less reliable for video (pose detection can fail)
}

# Preferred control types for video generation (in order of reliability)
PREFERRED_VIDEO_CONTROL_TYPES = ["depth", "canny"]

# Cache for loaded ControlNet models
_CONTROLNET_CACHE: dict[str, "ControlNetModel"] = {}


def load_controlnet(
    controlnet_type: str,
    torch_dtype: torch.dtype = torch.float16,
) -> "ControlNetModel":
    """
    Load a ControlNet model by type.

    Args:
        controlnet_type: Type of ControlNet ("pose", "depth", "canny", "lineart")
        torch_dtype: Data type for model weights (default: float16)

    Returns:
        Loaded ControlNetModel

    Note:
        Device placement is handled by enable_model_cpu_offload() in the pipeline,
        not at model load time.

    Raises:
        ValueError: If controlnet_type is not supported
        ImportError: If diffusers is not available
    """
    if controlnet_type not in CONTROLNET_MODELS:
        raise ValueError(
            f"Unsupported controlnet_type: {controlnet_type}. "
            f"Supported: {list(CONTROLNET_MODELS.keys())}"
        )

    # Check cache (device not needed since we use CPU offload)
    cache_key = f"{controlnet_type}_{torch_dtype}"
    if cache_key in _CONTROLNET_CACHE:
        logger.info(f"Using cached ControlNet: {controlnet_type}")
        return _CONTROLNET_CACHE[cache_key]

    try:
        from diffusers import ControlNetModel
    except ImportError as e:
        raise ImportError(
            "diffusers is required for ControlNet. Install with: pip install diffusers"
        ) from e

    model_id = CONTROLNET_MODELS[controlnet_type]
    logger.info(f"Loading ControlNet: {controlnet_type} from {model_id}")

    try:
        # Suppress warnings during loading
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            controlnet = ControlNetModel.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
            )
    except Exception as e:
        error_msg = (
            f"Failed to load ControlNet model '{model_id}'. "
            f"Error: {e}\n\n"
            f"Try: xinsir/controlnet-union-sdxl-1.0 (union model with better SDXL compatibility)"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e

    # Don't move to device here - let enable_model_cpu_offload() handle it
    _CONTROLNET_CACHE[cache_key] = controlnet
    logger.info(f"✓ Loaded ControlNet: {controlnet_type}")

    return controlnet


def create_controlnet_pipeline(
    base_pipeline: "StableDiffusionXLPipeline",
    controlnet: "ControlNetModel",
) -> "StableDiffusionXLControlNetPipeline":
    """
    Convert a base SDXL pipeline to a ControlNet pipeline.
    Uses enable_model_cpu_offload() for automatic device management.

    Args:
        base_pipeline: Base StableDiffusionXLPipeline
        controlnet: Loaded ControlNetModel

    Returns:
        StableDiffusionXLControlNetPipeline with ControlNet attached
    """
    try:
        from diffusers import StableDiffusionXLControlNetPipeline
    except ImportError as e:
        raise ImportError(
            "diffusers is required for ControlNet. Install with: pip install diffusers"
        ) from e

    logger.info("Creating ControlNet pipeline from base pipeline...")

    # Suppress warnings during pipeline creation
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        
        # Create ControlNet pipeline from base pipeline components
        controlnet_pipeline = StableDiffusionXLControlNetPipeline(
            vae=base_pipeline.vae,
            text_encoder=base_pipeline.text_encoder,
            text_encoder_2=base_pipeline.text_encoder_2,
            tokenizer=base_pipeline.tokenizer,
            tokenizer_2=base_pipeline.tokenizer_2,
            unet=base_pipeline.unet,
            controlnet=controlnet,
            scheduler=base_pipeline.scheduler,
        )

    # Use enable_model_cpu_offload() for automatic device management
    # This handles device placement better than manual .to(device)
    try:
        controlnet_pipeline.enable_model_cpu_offload()
        logger.info("✓ ControlNet pipeline created with CPU offload")
    except Exception:
        # Fallback to manual device placement if CPU offload fails
        logger.warning("CPU offload failed, using manual device placement")
        controlnet_pipeline = controlnet_pipeline.to("cuda")

    # Copy VAE dtype settings (keep VAE in float32 for decoding)
    if hasattr(base_pipeline.vae, "dtype"):
        controlnet_pipeline.vae.to(dtype=base_pipeline.vae.dtype)

    return controlnet_pipeline


def load_multi_controlnet(
    controlnet_types: list[str],
    torch_dtype: torch.dtype = torch.float16,
) -> "ControlNetModel | MultiControlNetModel | None":
    """
    Load multiple ControlNet models for video generation.

    Since we use the union model, this returns a single model that handles
    all control types. For separate models, it would return MultiControlNetModel.

    Args:
        controlnet_types: List of control types to load (e.g., ["pose", "depth", "canny"])
        torch_dtype: Data type for model weights (default: float16)

    Returns:
        ControlNetModel, MultiControlNetModel, or None if no types specified
    """
    if not controlnet_types:
        return None

    # Filter to valid types
    valid_types = [t for t in controlnet_types if t in CONTROLNET_MODELS]
    if not valid_types:
        logger.warning(f"No valid controlnet types in: {controlnet_types}")
        return None

    # Since we use the union model, we only need to load it once
    # (all types point to the same model)
    controlnet = load_controlnet(valid_types[0], torch_dtype=torch_dtype)
    logger.info(f"✓ Loaded union ControlNet for types: {valid_types}")

    return controlnet


def prepare_control_image(
    control_image: Image.Image,
    width: int,
    height: int,
) -> Image.Image:
    """
    Prepare control image for ControlNet conditioning.

    Args:
        control_image: PIL Image (pose, depth, or edge map)
        width: Target width
        height: Target height

    Returns:
        Resized control image
    """
    if control_image.size != (width, height):
        logger.debug(f"Resizing control image from {control_image.size} to ({width}, {height})")
        control_image = control_image.resize(
            (width, height), Image.Resampling.LANCZOS
        )
    return control_image


def prepare_control_images_for_video(
    pose: Image.Image | None,
    depth: Image.Image | None,
    edge: Image.Image | None,
    num_frames: int,
    width: int,
    height: int,
) -> list[Image.Image] | None:
    """
    Prepare control images for video generation by tiling across frames.

    For the union model, we use a single control image.
    Priority: DEPTH > EDGE (canny) > pose
    
    Depth and edge are preferred over pose because:
    - Depth: Most stable for video, maintains spatial structure reliably
    - Edge: Preserves outlines and fine details
    - Pose: Can fail on cropped/partial body images

    Args:
        pose: Pose control map (optional) - least preferred
        depth: Depth control map (optional) - MOST preferred
        edge: Edge/canny control map (optional) - second preferred
        num_frames: Number of frames to generate
        width: Target width
        height: Target height

    Returns:
        List of control images (one per frame), or None if no maps provided
    """
    # Select the best available control map
    # Priority: depth > edge > pose (depth and edge are most reliable for video)
    control_map = None
    control_type = None

    if depth is not None:
        control_map = depth
        control_type = "depth"
        logger.info("✓ Using DEPTH map (best for video stability)")
    elif edge is not None:
        control_map = edge
        control_type = "edge"
        logger.info("✓ Using EDGE map (good for preserving outlines)")
    elif pose is not None:
        control_map = pose
        control_type = "pose"
        logger.warning("⚠ Using POSE map (less reliable - consider using depth/edge instead)")

    if control_map is None:
        logger.warning("No control maps provided for video generation")
        return None

    # Resize to target dimensions
    control_map = prepare_control_image(control_map, width, height)
    logger.info(f"Tiling {control_type} control map for {num_frames} frames")

    # Tile the same control image across all frames
    return [control_map] * num_frames

