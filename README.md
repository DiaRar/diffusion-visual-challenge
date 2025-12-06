# Diffusion Visual Challenge — Anime Video (SDXL + Adapters)

Generate high-quality anime images and videos using SDXL 1.0 as the base, extended with pretrained adapters:
- **LoRA**: Style adaptation (anime aesthetics)
- **ControlNet**: Structure guidance (pose, depth, lineart)
- **AnimateDiff**: Motion generation for video
- **LCM LoRA**: Fast sampling (4-6 steps)

**Key Features**:
- ✅ **Single-scheduler architecture**: Automatic selection between DPM++ 2M (HQ) and LCMScheduler (fast LCM mode)
- ✅ **No manual scheduler configuration required**
- ✅ **67 unit tests passing** with full integration testing

This repository follows the contest constraints described in `docs/FINAL.md`.

**Current Status**: Day 1 (Image Generation) complete - see [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) for details.

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
  - **Note**: Scheduler is automatically selected - no configuration needed

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
  --out outputs/image_001.png
```

**Note**: Scheduler is automatically selected based on configuration:
- **High-Quality Mode**: Uses DPM++ 2M with Karras sigmas (26 steps, CFG 6.0)
- **Fast Mode** (with LCM): Uses LCMScheduler (4-6 steps, CFG 1.7)

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

**Automatic Scheduler Selection**: The system automatically chooses the optimal scheduler:
- **DPM++ 2M** (Karras sigmas) for high-quality generation (20-30 steps, CFG 6.0)
- **LCMScheduler** when LCM LoRA is enabled (4-6 steps, CFG 1.7)

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

### ✅ Completed
- ✅ **Day 0**: Basic image generation with stock SDXL/SD2
- ✅ **Day 1**: LoRA loading and adapter support (LoRA, ControlNet, LCM LoRA)
- ✅ **Day 1**: torch.compile integration for performance
- ✅ **Day 1**: Configuration-based LoRA system (configs/loras.py)
- ✅ **Day 1**: Multi-profile system (smoke, 768_long, 768_lcm, etc.)
- ✅ **Day 1**: 67 unit tests passing
- ✅ **Day 1**: Integration tests with actual image generation

### 🔄 In Progress / Upcoming
- [ ] AnimateDiff video generation (Day 2-3)
- [ ] ControlNet integration with video (Day 3-4)
- [ ] End-to-end anime video pipeline (Day 4-7)

See [docs/IMPLEMENTATION_STATUS.md](docs/IMPLEMENTATION_STATUS.md) for detailed progress report.

## LoRA Configuration and Testing

The system uses **configuration-based LoRA loading** (not CLI flags). LoRAs are configured in `configs/loras.py`.

### Active LoRAs (Current)

**Style LoRAs:**
- ✅ **Pastel Anime XL** (weight 0.8) - Main anime style

**Fast Sampling:**
- ⚡ **LCM SDXL** (weight 1.0) - Disabled by default for testing

### How to Enable/Disable LCM

Edit `configs/loras.py` line 75:

**To ENABLE LCM (Fast 4-6 steps):**
```python
LCM_LORA: LoRAConfig | None = LoRAConfig(
    name="LCM SDXL",
    path="latent-consistency/lcm-lora-sdxl",
    weight=1.0,
    adapter_name="lcm",
    type="lcm",
)
```

**To DISABLE LCM (High Quality):**
```python
LCM_LORA: LoRAConfig | None = None
```

### Test Commands

**With LCM (Fast Generation):**
```bash
uv run python infer/generate_image.py \
  --prompt "anime girl with blue hair, beautiful detailed eyes, cel shading, clean lines, masterpiece" \
  --seed 123 \
  --profile 768_lcm \
  --negative_prompt "blurry, low quality, bad anatomy" \
  --out outputs/test_with_lcm.png
```
- Expected: ~5 seconds, 5 steps, CFG 1.7 (uses LCMScheduler)

**Without LCM (High Quality):**
```bash
uv run python infer/generate_image.py \
  --prompt "anime girl with blue hair, beautiful detailed eyes, cel shading, clean lines, masterpiece" \
  --seed 123 \
  --profile 1024_hq \
  --negative_prompt "blurry, low quality, bad anatomy" \
  --out outputs/test_without_lcm.png
```
- Expected: ~20-30 seconds, 26 steps, CFG 6.0 (uses DPM++ 2M with Karras sigmas)

### Verify LoRA is Active

Look for this in the logs:
```
Active LoRAs (1):
  - Pastel Anime XL (style): weight=0.8
```

If LCM is enabled:
```
LCM LoRA loaded - using fast sampling mode (4-6 steps)
```

### Adding More LoRAs

Edit `configs/loras.py`:

```python
STYLE_LORAS: list[LoRAConfig] = [
    LoRAConfig(
        name="Pastel Anime XL",
        path="Linaqruf/pastel-anime-xl-lora",
        weight=0.8,
        adapter_name="pastel_anime",
        type="style",
    ),
    # Add more style LoRAs here (max 3 total for video stability)
    LoRAConfig(
        name="Your LoRA Name",
        path="username/your-lora",
        weight=0.5,
        adapter_name="your_adapter",
        type="style",
    ),
]
```

**Note:** Maximum 3 LoRAs total for video stability (2 style + 1 LCM recommended).

## Deterministic Logging
- Basic logging in place for seed, profile, steps, CFG, and scheduler settings
- Full `run.json` export with metadata (git hash, package versions, etc.)
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



```bash
cd /root/diffusion-visual-challenge

ACCELERATE_DISABLE_RICH=1 \
.venv/bin/accelerate launch \
  --num_processes=1 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no \
  scripts/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --train_data_dir sdxl_anime_lora_dataset \
  --image_column image --caption_column text \
  --resolution 1024 --center_crop --random_flip \
  --train_batch_size 2 --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 --lr_scheduler cosine --lr_warmup_steps 500 \
  --max_train_steps 10000 \
  --snr_gamma 5.0 --noise_offset 0.03 \
  --mixed_precision bf16 \
  --checkpointing_steps 1000 \
  --output_dir outputs/sdxl_anime_lora \
  --rank 64 --gradient_checkpointing \
  --seed 42 \
  --validation_prompt "masterpiece, best quality, anime girl portrait" \
  --num_validation_images 1 --validation_epochs 500
  ```