**FINAL.md — SOTA SDXL Anime Video Pipeline (LoRA, AnimateDiff, ControlNet)**

## 0) Hard Constraints (Contest Compliance)

### 0.1 Allowed Components

* Generative base: **SDXL 1.0 (`stabilityai/stable-diffusion-xl-base-1.0`) only**
* **Pretrained adapters allowed:**

  * LoRAs (visual + motion)
  * ControlNets (pose, depth, lineart, etc.)
  * Temporal adapters (AnimateDiff motion modules)
* **Inference-time optimizations allowed**: LCM LoRA, fp16/bf16, flash attention, torch.compile

### 0.2 Forbidden

* No pretrained diffusion models besides SDXL 1.0
* No commercial or closed-source models or tools
* No direct use of refiner, SD1.5, SD2, other backbone checkpoints

---

## 1) Goals and Deliverables

### Deliverable A (Required)

* High-quality anime stills (768x768+)
* SDXL base with style LoRA
* Visual + style prompt suite

### Deliverable B (Stretch)

* 8–10s 768p anime video (64+ frames @ 8–12 fps)
* Smooth motion with AnimateDiff
* ControlNet-guided composition
* Temporal consistency

---

## 2) Core Architecture

### 2.1 Style Adaptation (LoRA)

* Load **anime style LoRA** (e.g., AnimeProduction, Cybernella AestheticAnime LoRA)
* Optional: Character-specific LoRA for identity consistency
* Use 1–2 LoRAs max; apply at 0.5–0.8 scale

### 2.2 Structure Control (ControlNet)

* Load ControlNets for:

  * **OpenPose** (pose guidance)
  * **Depth** (background coherence)
  * **LineArt or Edge** (outline preservation)
* Use **sparse keyframe conditioning** if needed

### 2.3 Motion Adapter (AnimateDiff)

* Load AnimateDiff SDXL motion module
* Set `num_frames=16`, `fps=8` per segment
* Optional: Motion LoRAs (e.g. zoom, pan, run)

### 2.4 LoRA+LCM for Speed

* Load **LCM LoRA** for SDXL
* Set sampler steps = 4–6
* CFG = 1.5–2.0

---

## 3) Inference Optimizations

* Use `torch.compile(unet)`
* Enable flash attention (SDPA)
* Use `torch_dtype=torch.float16` or bf16
* Seed locking for deterministic runs

---

## 4) Data and Prompting

### 4.1 Prompt Style

* Use anime tags + style keywords
* Include trigger tokens for LoRA (if needed)
* Use consistent character + scene description
* Negative prompt: "lowres, watermark, bad anatomy, blur"

### 4.2 Prompt Suite

* ~30 diverse prompts across portrait, action, background
* Evaluate for line clarity, coherence, color, face stability

---

## 5) Generation Protocol

### Image

* 768x768, DPM++ 2M or DPM Solver++
* 28 steps (non-LCM), 4–6 steps (LCM)
* CFG: 5–7 (non-LCM), 1.5–2 (LCM)

### Video

* AnimateDiff + ControlNet
* 16–64 frames (chunked)
* 8–12 fps output
* Optional: frame interpolation post-process

---

## 6) Output Packaging

* Save outputs to `outputs/`
* For each prompt:

  * image grid
  * video (mp4)
  * config + metadata (json)
* Include LoRA and adapter versions used

---

## 7) Evaluation

* Visual: composition, color, fidelity, linework
* Motion: coherence, flicker, character consistency
* Style: anime strength, LoRA effectiveness
* Performance: inference speed per frame

---

**Definition of Done**

* At least 1 reproducible image and video pipeline
* Uses only SDXL base + adapters
* Reproducible from prompt, seed, and config
