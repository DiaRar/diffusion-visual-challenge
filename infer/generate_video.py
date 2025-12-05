"""
Generate anime videos with SDXL + AnimateDiff under contest constraints.

Key features:
- SDXL base 1.0 only (no refiner)
- AnimateDiff motion adapter (default SDXL v1.5)
- Style LoRAs reuse from image pipeline (<=3 total)
- HQ DPM++ 2M scheduler (or LCM if LoRA present)
- Deterministic seeding, segment chaining with latent carryover
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Any

from diffusers import AnimateDiffSDXLPipeline, ControlNetModel, MultiControlNetModel, AutoencoderKL
import numpy as np
import torch
from PIL import Image

# Add project root to path for imports before local imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# Local imports (after sys.path)
from infer.keyframes import generate_keyframe_with_maps
from configs.schedulers.high_scheduler import apply_best_hq_scheduler


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Pipeline cache for performance
_PIPELINE_CACHE: dict[str, "AnimateDiffSDXLPipeline"] = {}

# Defaults per user selection (highest quality setup)
DEFAULT_MOTION_ADAPTER = "guoyww/animatediff-motion-adapter-v1-5-0-sdxl"
DEFAULT_FPS = 12
DEFAULT_SEGMENT_LENGTH = 16
DEFAULT_TOTAL_FRAMES = 32  # two segments chained
DEFAULT_CN_OPENPOSE = "thibaud/controlnet-openpose-sdxl-1.0"
DEFAULT_CN_DEPTH = "diffusers/controlnet-depth-sdxl-1.0"
DEFAULT_CN_CANNY = "diffusers/controlnet-canny-sdxl-1.0"


# --------------------------------------------------------------------------
# Shared helpers (mirrors infer/generate_image.py where applicable)
# --------------------------------------------------------------------------
def _get_git_hash() -> str:
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    content = f"{timestamp}_{seed}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}_{profile_name}"
    return content


def _get_package_version(package_name: str) -> str | None:
    try:
        import importlib.metadata as metadata
    except ImportError:
        try:
            import importlib_metadata as metadata  # type: ignore
        except ImportError:
            return None
    try:
        return metadata.version(package_name)
    except Exception:
        return None


def _get_package_versions(package_names: list[str]) -> dict[str, str | None]:
    return {name: _get_package_version(name) for name in package_names}


def _load_controlnets(
    openpose_id: str | None,
    depth_id: str | None,
    canny_id: str | None,
) -> ControlNetModel | MultiControlNetModel | None:
    controlnets: list[ControlNetModel] = []
    for cid in [openpose_id, depth_id, canny_id]:
        if cid:
            logger.info(f"Loading ControlNet: {cid}")
            cn = ControlNetModel.from_pretrained(cid, torch_dtype=torch.float16, use_safetensors=True)
            controlnets.append(cn)
    if not controlnets:
        return None
    if len(controlnets) == 1:
        return controlnets[0]
    return MultiControlNetModel(controlnets)


# Force fp16 modules
def _align_module_dtypes(pipeline: AnimateDiffSDXLPipeline) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp16_targets = ["unet", "text_encoder", "text_encoder_2", "controlnet", "motion_adapter", "vae"]
    for name in fp16_targets:
        mod = getattr(pipeline, name, None)
        if mod is not None and callable(getattr(mod, "to", None)):
            try:
                mod.to(device=device, dtype=torch.float16)
            except Exception:
                pass


# Helper to tile control maps per frame
def _prepare_control_images(
    pose: Image.Image | None,
    depth: Image.Image | None,
    edge: Image.Image | None,
    num_frames: int,
) -> list[list[Image.Image]]:
    control_images: list[list[Image.Image]] = []
    for img in [pose, depth, edge]:
        if img is not None:
            control_images.append([img] * num_frames)
    return control_images


def _export_run_metadata(
    run_id: str,
    seed: int,
    prompt: str,
    profile_name: str,
    scheduler_mode: str,
    num_steps: int,
    backbone: str,
    out_dir: Path,
    negative_prompt: str | None,
    motion_adapter: str,
    num_frames: int,
    fps: int,
    segment_length: int,
    vae_fp32_decode: bool,
    use_custom_vae: bool,
) -> Path:
    runs_dir = PROJECT_ROOT / "outputs" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

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
        "negative_prompt": negative_prompt,
        "motion_adapter": motion_adapter,
        "num_frames": num_frames,
        "fps": fps,
        "segment_length": segment_length,
        "vae_fp32_decode": vae_fp32_decode,
        "use_custom_vae": use_custom_vae,
        "precision": "fp16_mixed",
        "compile_enabled": False,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,  # type: ignore[attr-defined]
        "torch_version": torch.__version__,
        "diffusers_version": _get_package_version("diffusers"),
        "package_versions": _get_package_versions(
            ["torch", "torchvision", "diffusers", "transformers", "safetensors"]
        ),
        "out_dir": str(out_dir),
    }

    run_json_path = runs_dir / f"{run_id}.json"
    with open(run_json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Run metadata exported to: {run_json_path}")
    return run_json_path


# --------------------------------------------------------------------------
# Pipeline setup
# --------------------------------------------------------------------------
def _load_motion_adapter(adapter_id: str):
    from diffusers import MotionAdapter

    logger.info(f"Loading motion adapter: {adapter_id}")
    adapter = MotionAdapter.from_pretrained(adapter_id, torch_dtype=torch.float16)
    if torch.cuda.is_available():
        adapter.to(device="cuda", dtype=torch.float16)
    return adapter


def _get_pipeline(
    backbone: str,
    motion_adapter_id: str,
    use_custom_vae: bool = False,
    controlnet: ControlNetModel | MultiControlNetModel | None = None,
) -> "AnimateDiffSDXLPipeline":
    """
    Get or create a cached AnimateDiff pipeline for SDXL.
    """
    backbone_key = backbone.lower()
    vae_key = "custom_vae" if use_custom_vae else "default_vae"
    cn_key = "cn" if controlnet is not None else "no-cn"
    cache_key = f"{backbone_key}_{vae_key}_{motion_adapter_id}_{cn_key}"

    if cache_key in _PIPELINE_CACHE:
        logger.info(f"Using cached {backbone.upper()} AnimateDiff pipeline ({vae_key})")
        return _PIPELINE_CACHE[cache_key]

    ALLOWED_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
    if backbone_key != "sdxl":
        raise RuntimeError(
            f"Unauthorized backbone '{backbone}'. Only SDXL is supported. No refiner models allowed."
        )

    logger.info(f"Loading {backbone.upper()} AnimateDiff pipeline...")
    logger.info(f"✓ Constraint check passed: Using approved model {ALLOWED_MODEL}")

    from diffusers import AutoencoderKL

    motion_adapter = _load_motion_adapter(motion_adapter_id)

    pipeline_kwargs = dict(
        pretrained_model_name_or_path=ALLOWED_MODEL,
        motion_adapter=motion_adapter,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    if use_custom_vae:
        vae_path = "../vae/g9_5++.safetensors"
        logger.info(f"Loading custom VAE (fp32 decode) from: {vae_path}")
        vae = AutoencoderKL.from_single_file(
            str(vae_path),
            use_safetensors=True,
            torch_dtype=torch.float32,
        )
        vae.config.scaling_factor = 0.13025  # type: ignore[assignment]
        pipeline_kwargs["vae"] = vae

    if controlnet is not None:
        pipeline_kwargs["controlnet"] = controlnet

    pipeline = AnimateDiffSDXLPipeline.from_pretrained(**pipeline_kwargs)

    _ = pipeline.to("cuda")
    pipeline.enable_vae_slicing()
    pipeline.enable_attention_slicing()
    pipeline = apply_best_hq_scheduler(pipeline, use_karras_sigmas=True, use_lu_lambdas=False)

    _PIPELINE_CACHE[cache_key] = pipeline
    logger.info("✓ AnimateDiff pipeline loaded and cached")
    return pipeline


# --------------------------------------------------------------------------
# Latent helpers
# --------------------------------------------------------------------------
def _get_device(pipeline: Any) -> torch.device:
    if hasattr(pipeline, "_execution_device"):
        # type: ignore[attr-defined]
        return pipeline._execution_device  # pyright: ignore
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_latents(
    pipeline: Any,
    num_frames: int,
    height: int,
    width: int,
    generator: torch.Generator,
    carryover_latent: torch.Tensor | None,
) -> torch.Tensor:
    device = _get_device(pipeline)
    dtype = pipeline.unet.dtype
    channels = pipeline.unet.in_channels
    latent_h = height // pipeline.vae_scale_factor if hasattr(pipeline, "vae_scale_factor") else height // 8
    latent_w = width // pipeline.vae_scale_factor if hasattr(pipeline, "vae_scale_factor") else width // 8

    latents = torch.randn(
        (1, channels, num_frames, latent_h, latent_w),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    if carryover_latent is not None:
        latents[:, :, 0] = carryover_latent.to(device=device, dtype=dtype)

    return latents


def _encode_frame_to_latent(
    pipeline: Any,
    frame: Image.Image,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Encode a PIL frame back to latent space to serve as carryover for next segment.
    """
    device = _get_device(pipeline)

    np_frame = np.array(frame).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np_frame).permute(2, 0, 1).unsqueeze(0).to(device).half()
    tensor = tensor * 2.0 - 1.0  # scale to [-1, 1]

    with torch.no_grad():
        pipeline.vae.to(dtype=torch.float16)
        _align_module_dtypes(pipeline)
        encoded = pipeline.vae.encode(tensor).latent_dist.sample(generator=generator)
        encoded = encoded * pipeline.vae.config.scaling_factor

    return encoded.to(dtype=pipeline.unet.dtype)


# --------------------------------------------------------------------------
# Video generation
# --------------------------------------------------------------------------
def _apply_loras(pipeline: Any) -> None:
    from configs.loras import get_active_loras, get_motion_loras, lora_summary

    active_loras = get_active_loras(include_lcm=True)
    motion_loras = get_motion_loras()

    all_loras = active_loras + motion_loras
    if not all_loras:
        return

    logger.info(lora_summary(all_loras))
    try:
        if hasattr(pipeline, "unload_lora_weights"):
            pipeline.unload_lora_weights()
    except Exception as e:
        logger.warning(f"Could not unload previous LoRA weights: {e}")

    adapter_names: list[str] = []
    adapter_weights: list[float] = []

    for lora_cfg in all_loras:
        logger.info(f"Loading LoRA: {lora_cfg.name}")
        load_kwargs: dict[str, str] = {"adapter_name": lora_cfg.adapter_name}
        if lora_cfg.weight_name is not None:
            load_kwargs["weight_name"] = lora_cfg.weight_name

        pipeline.load_lora_weights(lora_cfg.path, **load_kwargs)
        adapter_names.append(lora_cfg.adapter_name)
        adapter_weights.append(lora_cfg.weight)

    pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
    logger.info(f"✓ Loaded {len(all_loras)} LoRA(s)")


def _apply_scheduler(pipeline: Any, has_lcm: bool):
    if has_lcm:
        logger.info("LCM LoRA detected - switching to LCMScheduler")
        try:
            from diffusers import LCMScheduler

            pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        except Exception as e:
            logger.error(f"Failed to apply LCMScheduler: {e}")
            raise RuntimeError(f"LCMScheduler initialization failed: {e}") from e
    else:
        from configs.schedulers.high_scheduler import apply_best_hq_scheduler

        pipeline = apply_best_hq_scheduler(pipeline, use_karras_sigmas=True)
        logger.info("✓ Applied high-quality DPM++ 2M scheduler")
    return pipeline


def generate_video(
    prompt: str,
    backbone: str = "sdxl",
    profile_name: str = "768_long",
    scheduler_mode: str = "dpm",
    seed: int = 123,
    out_dir: str = "outputs/videos",
    num_steps: int | None = None,
    negative_prompt: str | None = None,
    motion_adapter_id: str = DEFAULT_MOTION_ADAPTER,
    num_frames: int = DEFAULT_TOTAL_FRAMES,
    fps: int = DEFAULT_FPS,
    segment_length: int = DEFAULT_SEGMENT_LENGTH,
    torch_compile: bool = False,
    use_custom_vae: bool = False,
    make_mp4: bool = False,
    use_controlnet: bool = True,
    controlnet_openpose: str | None = DEFAULT_CN_OPENPOSE,
    controlnet_depth: str | None = DEFAULT_CN_DEPTH,
    controlnet_canny: str | None = DEFAULT_CN_CANNY,
    controlnet_scale: float = 1.0,
) -> Path:
    if not prompt:
        raise ValueError("Prompt must be a non-empty string")
    if backbone.lower() != "sdxl":
        raise ValueError("Backbone must be 'sdxl'")
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    if segment_length < 1:
        raise ValueError("segment_length must be >= 1")

    from configs.profiles import get_profile
    from configs.loras import get_active_loras

    profile = get_profile(profile_name)
    actual_steps = num_steps if num_steps is not None else profile.num_inference_steps

    controlnet_models = None
    control_images: list[list[Image.Image]] | None = None

    if use_controlnet:
        controlnet_models = _load_controlnets(
            openpose_id=controlnet_openpose,
            depth_id=controlnet_depth,
            canny_id=controlnet_canny,
        )
        if controlnet_models is None:
            logger.warning("ControlNet enabled but no models loaded; proceeding without ControlNet.")
        else:
            keyframe_image, maps, _ = generate_keyframe_with_maps(
                prompt=prompt,
                seed=seed,
                profile_name=profile_name,
                use_custom_vae=use_custom_vae,
                backbone=backbone,
            )
            control_images = _prepare_control_images(
                pose=maps.pose,
                depth=maps.depth,
                edge=maps.edge,
                num_frames=num_frames,
            )
            if not control_images:
                logger.warning("No control maps extracted; proceeding without ControlNet conditioning.")

    pipeline = _get_pipeline(
        backbone=backbone,
        motion_adapter_id=motion_adapter_id,
        use_custom_vae=use_custom_vae,
        controlnet=controlnet_models,
    )
    _align_module_dtypes(pipeline)

    active_loras = get_active_loras(include_lcm=True)
    has_lcm = any(lora.type == "lcm" for lora in active_loras)
    pipeline = _apply_scheduler(pipeline, has_lcm=has_lcm)

    # LoRAs (style + optional motion)
    _apply_loras(pipeline)
    _align_module_dtypes(pipeline)
    # Force attention processors (LoRA layers) to fp16
    for proc in pipeline.unet.attn_processors.values():
        if hasattr(proc, "to"):
            try:
                proc.to(dtype=torch.float16, device=next(pipeline.unet.parameters()).device)
            except Exception:
                pass

    # Optional torch.compile for UNet
    if torch_compile and hasattr(pipeline, "unet"):
        try:
            pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")
            logger.info("✓ Enabled torch.compile on UNet")
        except Exception as e:
            logger.warning(f"Failed to enable torch.compile: {e}")

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

    run_id = _generate_run_id(seed, prompt, profile_name)
    out_base = Path(out_dir) / run_id
    frames_dir = out_base / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    _ = _export_run_metadata(
        run_id=run_id,
        seed=seed,
        prompt=prompt,
        profile_name=profile_name,
        scheduler_mode=scheduler_mode,
        num_steps=actual_steps,
        backbone=backbone,
        out_dir=frames_dir,
        negative_prompt=negative_prompt,
        motion_adapter=motion_adapter_id,
        num_frames=num_frames,
        fps=fps,
        segment_length=segment_length,
        vae_fp32_decode=True,
        use_custom_vae=use_custom_vae,
    )

    frames: list[Image.Image] = []
    carryover_latent: torch.Tensor | None = None
    remaining = num_frames

    segment_idx = 0
    while remaining > 0:
        current_len = min(segment_length, remaining)
        segment_idx += 1
        logger.info(f"Generating segment {segment_idx} with {current_len} frame(s)...")

        latents = _prepare_latents(
            pipeline=pipeline,
            num_frames=current_len,
            height=profile.height,
            width=profile.width,
            generator=generator,
            carryover_latent=carryover_latent,
        )

        with torch.no_grad():
            # Final dtype/device alignment to avoid mixed float/half biases
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for name in ["unet", "motion_adapter", "text_encoder", "text_encoder_2", "controlnet", "vae"]:
                mod = getattr(pipeline, name, None)
                if mod is not None and callable(getattr(mod, "to", None)):
                    mod.to(device=device, dtype=torch.float16)

            pipeline_kwargs = dict(
                prompt=prompt,
                height=profile.height,
                width=profile.width,
                num_frames=current_len,
                num_inference_steps=actual_steps,
                guidance_scale=profile.guidance_scale,
                generator=generator,
                negative_prompt=negative_prompt,
                output_type="pil",
                latents=latents,
            )
            if control_images:
                segment_imgs = [imgs[:current_len] for imgs in control_images]
                scales = [controlnet_scale] * len(segment_imgs)
                pipeline_kwargs.update(
                    controlnet_conditioning_image=segment_imgs,  # type: ignore[arg-type]
                    controlnet_conditioning_scale=scales,  # type: ignore[arg-type]
                )

            result = pipeline(**pipeline_kwargs)

        # AnimateDiffPipelineOutput.frames is list[List[PIL]] shaped [batch][frames]
        segment_frames: List[Image.Image] = []
        if hasattr(result, "frames"):
            raw_frames = result.frames
            if isinstance(raw_frames, list) and len(raw_frames) > 0 and isinstance(raw_frames[0], list):
                segment_frames = raw_frames[0]
            elif isinstance(raw_frames, list):
                segment_frames = raw_frames  # fallback
        else:
            segment_frames = result[0] if isinstance(result, (list, tuple)) else []

        if not segment_frames:
            raise RuntimeError("No frames returned by AnimateDiff pipeline.")

        frames.extend(segment_frames)
        remaining -= current_len

        # Prepare carryover latent from last frame
        last_frame = segment_frames[-1]
        carryover_latent = _encode_frame_to_latent(
            pipeline=pipeline, frame=last_frame, generator=generator
        )

    # Save frames
    for idx, frame in enumerate(frames):
        frame_path = frames_dir / f"frame_{idx:04d}.png"
        frame.save(frame_path)

    logger.info(f"✓ Saved {len(frames)} frames to {frames_dir}")

    video_path: Path | None = None
    if make_mp4:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin is None:
            logger.warning("ffmpeg not found; skipping mp4 assembly.")
        else:
            video_path = out_base / "video.mp4"
            cmd = [
                ffmpeg_bin,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(frames_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(video_path),
            ]
            try:
                subprocess.run(cmd, check=True)
                logger.info(f"✓ MP4 saved to: {video_path}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to assemble mp4: {e}")

    logger.info(
        f"Run complete. Frames: {len(frames)}, FPS: {fps}, Steps: {actual_steps}, "
        f"Scheduler: {scheduler_mode}, Motion adapter: {motion_adapter_id}"
    )

    return frames_dir


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate anime video with SDXL + AnimateDiff (contest-compliant)"
    )
    _ = parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt for video generation",
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
        default="768_long",
        choices=["smoke", "768_long", "1024_hq", "768_lcm", "1024_lcm"],
        help="Profile to use (default: 768_long)",
    )
    _ = parser.add_argument(
        "--scheduler",
        type=str,
        default="dpm",
        choices=["dpm"],
        help="Scheduler to use (default: dpm - high-quality DPM++ 2M)",
    )
    _ = parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/videos",
        help="Output directory for frames (default: outputs/videos)",
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
        "--motion-adapter",
        type=str,
        default=DEFAULT_MOTION_ADAPTER,
        help="Motion adapter model ID or path (default: SDXL AnimateDiff v1.5)",
    )
    _ = parser.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_TOTAL_FRAMES,
        help="Total number of frames to generate (default: 32)",
    )
    _ = parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Frames per second for mp4 export (default: 12)",
    )
    _ = parser.add_argument(
        "--segment-length",
        type=int,
        default=DEFAULT_SEGMENT_LENGTH,
        help="Frames per segment (default: 16, chains segments automatically)",
    )
    _ = parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Enable torch.compile for UNet acceleration",
    )
    _ = parser.add_argument(
        "--use-custom-vae",
        action="store_true",
        help="Use custom VAE (xlVAEC_g952) instead of default SDXL VAE",
    )
    _ = parser.add_argument(
        "--mp4",
        action="store_true",
        help="Assemble frames into video.mp4 using ffmpeg (if available)",
    )
    _ = parser.add_argument(
        "--use-controlnet",
        action="store_true",
        default=True,
        help="Enable ControlNet conditioning with internal maps (default: on)",
    )
    _ = parser.add_argument(
        "--no-controlnet",
        action="store_false",
        dest="use_controlnet",
        help="Disable ControlNet conditioning",
    )
    _ = parser.add_argument(
        "--controlnet-openpose",
        type=str,
        default=DEFAULT_CN_OPENPOSE,
        help="ControlNet OpenPose model id (default: SDXL openpose)",
    )
    _ = parser.add_argument(
        "--controlnet-depth",
        type=str,
        default=DEFAULT_CN_DEPTH,
        help="ControlNet Depth model id (default: SDXL depth)",
    )
    _ = parser.add_argument(
        "--controlnet-canny",
        type=str,
        default=DEFAULT_CN_CANNY,
        help="ControlNet Canny/LineArt model id (default: SDXL canny)",
    )
    _ = parser.add_argument(
        "--controlnet-scale",
        type=float,
        default=1.0,
        help="ControlNet conditioning scale (applied to all loaded CNs)",
    )

    forbidden_args = ["--refiner", "--vae-override", "--vae_override"]
    for arg in forbidden_args:
        if arg in sys.argv:
            parser.error(
                f"Forbidden argument '{arg}' detected. Only SDXL base model allowed. No refiner models permitted."
            )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        _ = generate_video(
            prompt=args.prompt,
            backbone=args.backbone,
            profile_name=args.profile,
            scheduler_mode=args.scheduler,
            seed=args.seed,
            out_dir=args.out_dir,
            num_steps=args.steps,
            negative_prompt=args.negative_prompt,
            motion_adapter_id=args.motion_adapter,
            num_frames=args.num_frames,
            fps=args.fps,
            segment_length=args.segment_length,
            torch_compile=args.torch_compile,
            use_custom_vae=args.use_custom_vae,
            make_mp4=args.mp4,
            use_controlnet=args.use_controlnet,
            controlnet_openpose=args.controlnet_openpose,
            controlnet_depth=args.controlnet_depth,
            controlnet_canny=args.controlnet_canny,
            controlnet_scale=args.controlnet_scale,
        )
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()

