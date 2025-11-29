# Diffusion Visual Challenge — Anime Video (SDXL + Adapters)

Generate high-quality anime images and videos using SDXL 1.0 as the base, extended with pretrained adapters:
- **LoRA**: Style adaptation (anime aesthetics)
- **ControlNet**: Structure guidance (pose, depth, lineart)
- **AnimateDiff**: Motion generation for video
- **LCM LoRA**: Fast sampling (4-6 steps)

This repository follows the contest constraints described in `docs/FINAL.md`.

## Quick Start
- Prerequisites:
  - NVIDIA GPU with drivers compatible with CUDA 12.1 (or 11.8).
  - `ffmpeg` installed for high-quality MP4 export (optional; falls back to `imageio`).
  - Recommended Python: 3.12.
- Setup (uv):
  - `uv venv .venv --python 3.12`
  - `source .venv/bin/activate`
  - `uv pip install -r requirements.txt`
  - If your server uses CUDA 11.8, change the first line of `requirements.txt` to `--index-url https://download.pytorch.org/whl/cu118` and reinstall.
- Smoke run (deterministic):
  - `uv run python infer/generate_image.py --smoke --prompt "smoke test" --seed 123 --profile smoke --out outputs/test.png`
  - Verify reproducibility by regenerating with the same seed and profile
  - Test adapters: `uv run python infer/generate_image.py --lora /path/to/lora --smoke --prompt "smoke test" --seed 123`

## Repository Structure
- `models/` — model components (UNet/VAE/temporal/control/mask adapters).
- `train/` — training scripts for LoRA, temporal, control, layers, alignment.
- `infer/` — inference scripts and CLI (`generate_video.py`).
- `data/` — datasets and manifests (see compliance).
- `configs/` — configuration files (VRAM, profiles).
- `scripts/` — utilities like `export_video.sh` (PNG → MP4).
- `reports/` — result galleries and ablations.
- `docs/` — specs and planning (`TODO.md`, `FINAL.md`).

## Inference CLI

### Basic Image Generation (aligned to `docs/FINAL.md`)
```bash
python infer/generate_image.py \
  --prompt "anime character with blue hair" \
  --seed 123 \
  --profile 768_long \
  --scheduler dpm \
  --out outputs/image_001.png
```

### With Adapters (LoRA, ControlNet, LCM LoRA)
```bash
# Style LoRA for anime adaptation
python infer/generate_image.py \
  --lora /path/to/anime_style_lora.safetensors \
  --lora-scale 0.8 \
  --prompt "anime character with blue hair" \
  --profile 768_long \
  --out outputs/anime_001.png

# ControlNet for pose guidance
python infer/generate_image.py \
  --controlnet lllyasviel/sd-controlnet-openpose \
  --controlnet-images pose.jpg \
  --prompt "anime character in this pose" \
  --profile 768_long \
  --out outputs/posed_001.png

# LCM LoRA for fast sampling (4-6 steps)
python infer/generate_image.py \
  --lcm-lora /path/to/lcm_lora.safetensors \
  --steps 5 \
  --prompt "anime character with blue hair" \
  --out outputs/fast_001.png

# Performance optimization with torch.compile
python infer/generate_image.py \
  --torch-compile \
  --prompt "anime character" \
  --profile 768_long \
  --out outputs/optimized_001.png
```

### Video Generation (with AnimateDiff)
```bash
# Coming soon: infer/generate_video.py with AnimateDiff motion module
# Example (TBD):
python infer/generate_video.py \
  --motion-module /path/to/animatdiff_motion.safetensors \
  --prompt "anime character running" \
  --num-frames 64 \
  --fps 8 \
  --out outputs/video_001.mp4
```

## Progress
- ✅ **Day 0**: Basic image generation with stock SDXL/SD2
- ✅ **Day 1**: LoRA loading and adapter support (LoRA, ControlNet, LCM LoRA)
- ✅ **Day 1**: torch.compile integration for performance
- [ ] AnimateDiff video generation (Day 2-3)
- [ ] ControlNet integration with video (Day 3-4)
- [ ] End-to-end anime video pipeline (Day 4-7)

## Deterministic Logging
- Basic logging in place for seed, profile, steps, CFG, and scheduler settings
- TODO: Add `run.json` export with full metadata (git hash, compile flags, etc.)
- See implementation in `infer/generate_image.py`.

## Video Export
- Prefer `scripts/export_video.sh` (requires `ffmpeg`).
- Frame naming pattern: `%05d.png` under `outputs/<run>_frames/`.
- If `ffmpeg` is unavailable, falls back to `imageio`-based export.

## Compliance
- Dataset manifest JSONL must include: `{path, source_url, license, split}`.
- **Allowed**: SDXL 1.0 base with pretrained adapters (LoRA, ControlNet, AnimateDiff)
- **Forbidden**: SDXL refiner models, other diffusion backbones (SD1.5, SD2.1), commercial models
- All adapter usage logged in `outputs/runs/*.json` for reproducibility
- Example manifest: `data/manifest.example.jsonl`

## Profiles and Roadmap
- Follow stages and acceptance gates in `docs/TODO.md`.

### Completed
- ✅ **Day 0**: Basic image generation with stock SDXL/SD2
- ✅ **Day 1**: Pretrained adapter support (LoRA, ControlNet, LCM LoRA)
- ✅ **Day 1**: torch.compile integration for performance

### Upcoming
- **Day 2**: ControlNet integration with video (AnimateDiff)
- **Day 3**: Video generation with temporal coherence
- **Day 4-7**: End-to-end anime video pipeline with adapters

### Performance Features
- ✅ torch.compile for UNet acceleration (10-20% speedup)
- ✅ LCM LoRA for fast sampling (4-6 steps vs 20+ steps)
- [ ] Memory optimization with SDPA + bf16
- [ ] UNet tiling for large resolutions

## Troubleshooting
- CUDA wheels:
  - `requirements.txt` pins `cu121` by default; switch to `cu118` if needed.
- Missing `ffmpeg`:
  - Install via your package manager or rely on fallback export.
- Python version:
  - Use 3.12 for broad wheel availability and stable installs.

## License and Citations
- Track dataset licenses in manifests.
- Keep references in the final write-up per `docs/FINAL.md`.
