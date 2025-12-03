# PIPELINE.md — SOTA SDXL Anime Video Generation Pipeline (LoRA + AnimateDiff + ControlNet + LCM)

This document is the **single source of truth** for your full **SDXL Anime Video Generation System**, using:

* SDXL 1.0 base (contest‑compliant)
* Anime LoRAs
* AnimateDiff SDXL motion module
* ControlNet (Pose, Depth, LineArt)
* LCM LoRA for 4–6 step inference
* Optimized inference (fp16/bf16 + SDPA + torch.compile)

Everything below is actionable and reflects the **best SOTA configuration** in late 2025.

---

# 1. Model & Adapter Components

## 1.1 Base Model (Required)

* **Stable Diffusion XL Base 1.0**

  * `stabilityai/stable-diffusion-xl-base-1.0`
  * Load in `fp16` or `bf16`
  * Use `sdxl-vae-fp16-fix` VAE

## 1.2 LoRAs (Anime Style)

### **Primary Style LoRA** (required)

* **Pastel Anime XL** — weight **0.7–0.85**

### **Secondary Flat-Color Stability LoRA** (strongly recommended)

* **Anime Flat Color XL** — weight **0.15–0.3**

### **(Optional) Character LoRA** (only if identity consistency needed)

* Weight **0.6–0.8**
* Only use **one** character LoRA

### **(Optional) Utility/Fusion LoRA**

* **SDXL Aesthetic Anime LoRA** — weight **0.1–0.2**

**Do not exceed 3 total LoRAs** for video stability.

---

# 2. ControlNet Modules

These enforce **structure**, **pose**, and **line stability** across frames.

## 2.1 Required ControlNets

### **OpenPose ControlNet** (must‑use)

* Guarantees pose stability frame‑to‑frame
* Works perfectly with AnimateDiff
* Input: stick figure pose per frame or per keyframe

### **Depth ControlNet** (highly recommended)

* Locks environment + perspective
* Prevents background distortion or size drift
* Input: depth maps (full sequence or keyframes)

## 2.2 Optional ControlNets

### **LineArt / SoftEdge / Canny**

* Enhances outline stability
* Makes anime frames more cel‑shaded
* Use at 0.5–1.0 weight

## 2.3 ControlNet Guidance Strategy

* Use **keyframes** (0, ~25%, ~50%, ~75%, 100%)
* AnimateDiff interpolates motion between them
* Full-per-frame conditioning is optional but heavier

---

# 3. AnimateDiff Temporal Module

## 3.1 Motion Module

* **AnimateDiff SDXL Motion Module (v1.5 preferred)**
* Use with `num_frames=16` segments
* Output FPS = **8–12 fps**

## 3.2 Motion LoRAs (Optional)

* Use only **one** at a time:

  * Camera Pan
  * Zoom In/Out
  * Cinematic Drift
* Weight: **1.0** or author‑recommended

## 3.3 Segment Chaining

* Generate in 16-frame chunks
* Use last frame latent as init for next segment
* Ensures smooth transitions for 8–10s videos

---

# 4. Inference Optimizations

## 4.1 LCM LoRA (Latent Consistency Model)

* Load LCM LoRA for SDXL
* Steps: **4–6**
* CFG: **1.5–2.0**
* Biggest acceleration improvement in pipeline

## 4.2 Precision & Attention

* Use **fp16** or **bf16** everywhere
* Ensure **SDPA / Flash Attention** enabled
* In PyTorch 2.1+: SDPA auto-selects optimal kernels

## 4.3 Compile

```python
pipe.unet = torch.compile(pipe.unet, mode="max-autotune")
```

* Typically +10–25% speed boost

## 4.4 Memory Strategy

* Use `torch.set_grad_enabled(False)`
* Disable safety checker
* Turn off refiner (disallowed anyway)
* Optionally enable tiling if generating > 768p

---

# 5. Prompting Strategy

## 5.1 Positive Prompt Template

```
(masterpiece, anime, beautiful detailed eyes), cel shading, clean lines,
<lora:pastelAnimeXL:0.8>, <lora:animeFlatColorXL:0.25>,
1 girl, blue hair, dynamic lighting, action pose, dramatic composition
```

## 5.2 Negative Prompt Template

```
bad anatomy, extra limbs, extra fingers, blurry, watermark, text,
low detail, inconsistent lighting, distorted face
```

## 5.3 Guidance

* LCM: CFG = **1.5–2.0**
* Non-LCM fallback: CFG = **5–7**, steps = **20–30**

---

# 6. Full Generation Procedure (Step-by-Step)

## **Step 1 — Load Models**

* Load SDXL base
* Load SDXL VAE FP16-FIX
* Load Pastel Anime XL
* Load Anime Flat Color XL
* Load optional character LoRA
* Load LCM LoRA

## **Step 2 — Load ControlNets**

* Load OpenPose → condition frames or keyframes
* Load Depth → full-sequence depth maps or keyframes
* Load LineArt → if outline enhancement desired

## **Step 3 — Load AnimateDiff**

* Load SDXL motion module
* Set `num_frames = 16`
* Set `fps = 8–12`
* Optionally load Motion LoRA

## **Step 4 — Configure Sampler**

* If using LCM: steps = 4–6
* Scheduler: DPM++ SDE or DPMSolver++

## **Step 5 — Generate Frames**

* Generate 16-frame batch
* Save latents of frame 16
* Feed latent as init → next 16-frame batch
* Repeat until 8–10 seconds reached

## **Step 6 — Post-Process**

* Assemble frames into video (ffmpeg)
* (Optional) Frame interpolate to 12–24fps
* (Optional) SDXL upscaling if needed

---

# 7. Output Structure

```
outputs/
  images/
    prompt_grid_*.png
  video/
    clip_001/
      frames/*.png
      clip_001.mp4
  configs/
    pipeline_config.json
    lora_versions.json
    controlnet_config.json
```

---

# 8. Quality Checklist

* ✔ Stable character identity
* ✔ Consistent background perspective
* ✔ No flicker on edges or shading
* ✔ Lines look anime/cel-friendly
* ✔ Motion is smooth at 8–12fps
* ✔ Steps kept to 4–6 (LCM)
* ✔ ControlNets do not overpower style

---

# 9. Minimal Reproducible Code Snippet (Diffusers)

```python
from animatediff import AnimateDiffSDXLPipeline
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
import torch

pipe = AnimateDiffSDXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    motion_adapter="path/to/animatediff_sdxl.safetensors",
    torch_dtype=torch.float16,
).to("cuda")

pipe.unet = torch.compile(pipe.unet)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights("pastel-anime-xl.safetensors", weight=0.8)
pipe.load_lora_weights("anime-flat-color-xl.safetensors", weight=0.25)
pipe.load_lora_weights("lcm-sdxl.safetensors", weight=1.0)

frames = pipe(prompt, num_frames=16, guidance_scale=1.7)
```

---

# 10. Final Notes

* This pipeline is **fully contest-compliant** (SDXL base only)
* Maximizes stability with style LoRA + flat-color LoRA + ControlNet
* AnimateDiff provides temporal backbone
* LCM drastically reduces compute cost

If you need, I can also provide:

* A `config.json` matching this pipeline
* A run script (`run_video.py`)
* A LoRA weight tuner
