# FINAL.md — SOTA SDXL Anime Video Pipeline (Contest-Compliant)

This document is the **single source of truth** for your entire anime video generation system under the **CS492C Visual Generation Contest constraints**.

**Only SDXL 1.0 is allowed as a diffusion backbone.**
LoRAs, ControlNets, AnimateDiff, Motion LoRAs, LCM, performance optimizations are allowed.
All ControlNet conditioning must be **internally generated**, not externally supplied.

---

# 1. System Goals

### **Primary Goal**

* Produce **high‑quality anime images** (768×768 or 1024×1024)
* SDXL + Anime LoRAs + LCM (4–6 steps)

### **Secondary Goal (Stretch)**

* Produce **8–10 second 768p anime videos**
* SDXL + AnimateDiff Motion Module
* Self-generated ControlNet guidance
* Minimal flicker, consistent identity, stable backgrounds

---

# 2. Architectural Components

## 2.1 Base Model (required)

* **Stable Diffusion XL Base 1.0**
* Use: `torch.float16` or `bfloat16`
* VAE: **sdxl-vae-fp16-fix** (cleanest, most stable decoding)

---

## 2.2 LoRAs (Anime Style)

### Required LoRAs

1. **Pastel Anime XL** — Main style backbone (weight 0.7–0.85)
2. **Anime Flat Color XL** — Flat-color, stable-cel look (weight 0.15–0.3)

### Optional LoRAs

* **Character LoRA** — if identity stability needed (weight 0.6–0.8)
* **Aesthetic Anime LoRA** — small polish (weight 0.1–0.2 max)

> Use **max 3 LoRAs** simultaneously to avoid interference.

---

# 3. ControlNets (Contest-Compliant)

## ControlNet modules included:

* **OpenPose** (pose stability, motion structure)
* **Depth** (background + perspective stability)
* **LineArt / Canny** (outline stabilization)

## ALL ControlNet conditioning must be internally generated from SDXL outputs

**Allowed:**

* Generate keyframe → extract pose/depth/edges → use in ControlNet.
* Generate multiple self-keyframes → sparse conditioning.

**NOT Allowed:**

* External images, sketches, stick figures.
* Downloaded depth maps.
* Manually drawn guides.

---

# 4. AnimateDiff Temporal Module

### Required

* **AnimateDiff Motion Module for SDXL** (v1.5 preferred)

### Parameters

* `num_frames=16` per segment
* FPS: **8–12 fps**
* Chaining: pass last-frame latent → next segment

### Optional

* **Motion LoRA** (pan, zoom, drift)
* Limit to **one** at a time

---

# 5. LCM Acceleration (Latent Consistency Model)

* Load LCM LoRA for SDXL
* Steps: **4–6**
* CFG: **1.5–2.0**
* Scheduler: DPM++ SDE or DPMSolver++

This achieves **10×–20× speedup** while maintaining excellent anime-style sharpness.

---

# 6. Inference Optimizations

* Use SDPA / Flash Attention (PyTorch 2.x)
* Enable `torch.compile(unet, mode="max-autotune")`
* fp16 / bf16 everywhere
* Disable safety checker
* Use deterministic seeding
* Optional tiling for >768p

---

# 7. Prompting Strategy

### Positive Prompt Template

```
(masterpiece, anime, clean cel-shading), detailed eyes, vivid colors,
<lora:pastelAnimeXL:0.8>, <lora:animeFlatColorXL:0.25>,
1 girl, flowing hair, dynamic pose, dramatic lighting
```

### Negative Prompt Template

```
bad anatomy, extra limbs, two heads, blurry, low detail, watermark,
text, distorted face, unstable shading
```

---

# 8. Full Pipeline Procedure

## **Step 1 — Generate Keyframe(s)**

* Use SDXL + LoRAs (no ControlNet)
* Generate 1–3 keyframes describing major beats of the motion

## **Step 2 — Extract Internal Control Maps**

For each keyframe:

* Pose Map: OpenPose preprocessor
* Depth Map: depth estimator
* LineArt/Canny Map: edge extractor

## **Step 3 — Initialize AnimateDiff**

* Load SDXL Motion Model
* Apply LoRAs + LCM
* Apply ControlNet(s) with extracted internal maps
* Set num_frames=16

## **Step 4 — Generate Segment (16 frames)**

* Save last latent
* Continue next segment with same config

## **Step 5 — Post-Process**

* Save frames → assemble to mp4
* Optional frame interpolation
* Optional SDXL upscaling (legal because still SDXL)

---

# 9. Directory Structure

```
pipeline/
  configs/
  keyframes/
  controlmaps/
  segments/
  outputs/
    images/
    video/
```

---

# 10. Quality Checklist

* Clear, stable anime style
* Identity does not drift
* No frame flicker
* Background remains stable
* Lines do not wobble
* Temporal motion is smooth
* Steps kept to 4–6 (LCM)
* All ControlNet maps internally generated

---

# 11. Definition of Done

* Reproducible SDXL-only pipeline
* 1–3 high-quality anime stills
* 8–10 second stable anime video
* Full compliance with contest rules
* Documented seeds, configs, adapter versions
