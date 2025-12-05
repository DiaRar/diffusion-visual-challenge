# FINAL.md — SOTA SDXL Anime Video Pipeline (Contest-Compliant, Current Impl)

This document is the **single source of truth** for the current **implemented** anime pipeline under the **CS492C Visual Generation Contest constraints**.

> **CRITICAL RULE UPDATE (Slack Clarifications):**
> *   **SDXL 1.0** is the enforced backbone in code (SD2 is allowed by rules but not wired in this repo).
> *   **No External Visual Input at Inference**: You cannot use existing images, sketches, or manually drawn pose maps as input to ControlNet/img2img.
> *   **Internal Generation Allowed**: If your model generates a conditioning signal (e.g., text-to-image -> pose extraction), you CAN use that signal to guide further generation.
> *   **AnimateDiff Allowed**: Explicitly permitted as an "additional neural network for guidance".
> *   **Post-Processing Allowed**: Basic cropping, zooming, scaling of final output is permitted.
> *   **Data Restrictions**: Fine-tuning allowed only with FREE data (no paid assets).

---

# 1. System Goals (as implemented)

### **Primary Goal**

*   Produce **high‑quality anime images** at 768×768 (default) and 1024×1024 (HQ).
*   SDXL + Anime LoRAs; LCM is **available but disabled by default** until quality is vetted.

### **Secondary Goal (Stretch)**

*   Produce **8–10 second 768p anime videos**
*   SDXL + AnimateDiff Motion Module (to be integrated)
*   **Self-generated** ControlNet guidance (no external images) — loader not yet wired
*   Minimal flicker, consistent identity, stable backgrounds

---

# 2. Architectural Components

## 2.1 Base Model (Required)

*   **Stable Diffusion XL Base 1.0** (only; SD2 not yet supported in code)
*   Precision: pipeline runs in `torch.float16`; VAE kept in **float32** for decoding
*   VAE options:
    * Default SDXL VAE (fp16 weights, decoded in fp32)
    * Optional custom VAE `../vae/g10.safetensors` (loaded fp32, scaling 0.13025)
*   **No refiner models permitted** (blocked at CLI)

## 2.2 LoRAs (Anime Style)

### Active Style LoRAs (configs/loras.py)
1.  **Pastel Anime XL** — 0.8 (adapter `pastel_anime`)
2.  **Ani40 Stabilizer** — 0.4 (adapter `ani_stabilizer`)
3.  **Noob** — 0.2 (adapter `noob`)

### Optional (configured but empty/disabled)
*   Character LoRA slot — none loaded
*   Motion LoRAs — none loaded
*   LCM LoRA — **configured but disabled** (`LCM_LORA = None`)

> Rule: keep **≤3 concurrent LoRAs** for stability (currently at 3 style adapters).

---

# 3. ControlNets (Strict Compliance)

## Planned ControlNet modules:
*   **OpenPose** (pose stability, motion structure)
*   **Depth** (background + perspective stability)
*   **LineArt / Canny** (outline stabilization)

## CRITICAL: Conditioning Rules
**ALL ControlNet conditioning must be internally generated.**

**✅ Allowed:**
*   Text-to-Image generation -> Extract Pose/Depth -> Use as ControlNet input for Video.
*   Text-to-Image generation -> Extract Canny edges -> Use as ControlNet input.
*   Using AnimateDiff with text prompts only (no ControlNet).

**❌ NOT Allowed:**
*   Inputting an external image (e.g., from Google Images, ArtStation) to guide generation.
*   Inputting a hand-drawn sketch or stick figure.
*   Inputting a manually created/edited depth map or pose map.
*   Using commercial tools (ChatGPT, Midjourney) to generate the input.

---

# 4. AnimateDiff Temporal Module

### Required
*   **AnimateDiff Motion Module** (SDXL or SD2 version)

### Parameters (planned defaults)
*   `num_frames=16` per segment
*   FPS: **8–12 fps**
*   Chaining: pass last-frame latent → next segment

### Optional
*   **Motion LoRA** (pan, zoom, drift) - limit to one at a time.

---

# 5. Scheduler & LCM

*   **Default (enabled):** High-quality DPM++ 2M (custom `SDXLAnime_BestHQScheduler`)
    * Karras sigmas, solver_order=3 midpoint, no refiner
    * Profiles:
        * `768_long`: 22 steps, CFG 6.0
        * `1024_hq`: 90 steps, CFG 6.5
*   **Fast path (optional):** LCM LoRA + LCMScheduler (4–6 steps, CFG 1.7) — currently **off by default** until quality sign-off.

---

# 6. Inference Optimizations (implemented)

*   Manual FP32 VAE decode to eliminate green-tint artifacts
*   Pipeline caching (per VAE selection)
*   Optional `--torch-compile` (`torch.compile(unet, mode="reduce-overhead")`)
*   SDPA / Flash Attention via PyTorch 2.x
*   Deterministic seeding
*   CPU offload attempted when available
*   Safety checker disabled

---

# 7. Prompting Strategy

### Positive Prompt Template (matches active LoRAs)
```
(masterpiece, anime, clean cel-shading), detailed eyes, vivid colors,
<lora:pastelAnimeXL:0.8>, <lora:ani40_stabilizer:0.4>, <lora:noob:0.2>,
1 girl, flowing hair, dynamic pose, dramatic lighting
```

### Negative Prompt Template
```
bad anatomy, extra limbs, two heads, blurry, low detail, watermark,
text, distorted face, unstable shading
```

---

# 8. Full Pipeline Procedure (Compliant Video)

## **Step 1 — (Optional) Generate Internal Reference**
*   Use Text-to-Image (SDXL/SD2) to generate a starting frame or keyframe.
*   **Rule**: This image MUST be generated by your model, not imported.
*   Scheduler is automatically selected (DPM++ 2M for HQ, LCMScheduler if LCM enabled).

## **Step 2 — (Planned) Extract Control Signal**
*   If using ControlNet: Run OpenPose/Depth/LineArt preprocessors on the *generated* image from Step 1.
*   Use this internal map to guide the video generation (no external inputs).

## **Step 3 — Initialize AnimateDiff**
*   Load Motion Model
*   Apply LoRAs + LCM
*   Set num_frames=16

## **Step 4 — Generate Segment**
*   Generate 16 frames.
*   Use last latent as context for next segment (smooth transitions).
*   Same automatic scheduler selection as image generation.

## **Step 5 — Post-Process**
*   Save frames → assemble to MP4 (ffmpeg).
*   Basic cropping/scaling allowed.
*   Frame interpolation (e.g., RIFE) allowed if open-source model.

---

# 9. Submission Requirements

### Deliverables
*   **Poster**: PDF, max 10MB.
*   **Source Code & Data**: Must reproduce results.
*   **Write-up**: PDF, max 4 pages. Must include:
    *   Project Title, Names, IDs
    *   Technical & Artistic description
    *   Reproduction steps
    *   **Citations**: List ALL pretrained checkpoints (SDXL, LoRAs, ControlNets, AnimateDiff) with links.

### Presentation
*   Online via Gather.town (Dec 8 or Dec 10).
*   Mandatory attendance.

---

# 10. Definition of Done

*   Reproducible SDXL pipeline (SD2 optional but not yet implemented).
*   1–3 high-quality anime stills.
*   8–10 second stable anime video.
*   **Full compliance with contest rules (No external inputs).**
