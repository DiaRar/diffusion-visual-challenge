# TODO.md — Week Plan for 3‑Person Team (Contest‑Compliant, SOTA)

This TODO reflects everything allowed under contest rules:

* **SDXL base only**
* LoRAs, ControlNets, AnimateDiff, Motion LoRAs allowed
* **All ControlNet maps must be internally generated**
* No external pose/depth/edge sources

---

# Team Roles

**P1 — Temporal pipeline / AnimateDiff**
**P2 — ControlNet maps + keyframe pipeline**
**P3 — LoRA integration + performance optimizations**

---

# Day 1 — Base Pipeline + LoRA Integration (P3)

* [ ] Load SDXL 1.0 base in fp16/bf16
* [ ] Integrate **sdxl-vae-fp16-fix**
* [ ] Implement LoRA loader
* [ ] Add primary LoRAs:

  * Pastel Anime XL (0.7–0.85)
  * Anime Flat Color XL (0.15–0.3)
* [ ] Add optional character LoRA support
* [ ] Add LCM LoRA loader (4–6 steps, CFG 1.5–2)
* [ ] Add flags: `--lora_config`, `--lora_weights`, `--lcm`
* [ ] Add deterministic seed handler
* [ ] Validate single-frame output quality

---

# Day 2 — ControlNet Integration (P2) — Contest‑Legal

### Core Loader

* [ ] Implement ControlNet loading for:

  * OpenPose
  * Depth
  * LineArt / Canny

### Internal Conditioning Map Generator (legal)

* [ ] Add `generate_keyframe(prompt, seed)` helper
* [ ] Add `extract_control_maps(frame)`:

  * pose_map = openpose(frame)
  * depth_map = depth_estimator(frame)
  * edge_map = canny/lineart(frame)
* [ ] Ensure **NO external image paths** accepted
* [ ] Enforce `--internal_controlnet_only` (reject external files)

### Sparse / Keyframe Conditioning

* [ ] Implement SparseCtrl-compatible structure
* [ ] Support keyframe-only guidance (frame 0, K, 2K…N)
* [ ] Interpolation handled via AnimateDiff motion module

### Shape Validation

* [ ] Auto-resize control maps → SDXL resolution
* [ ] Auto-resize → latent space (128×128)
* [ ] Pre-inference validation errors for mismatched shapes

---

# Day 3 — AnimateDiff Integration (P1)

* [ ] Integrate SDXL AnimateDiff Motion Module (v1.5)
* [ ] Add CLI flags:

  * `--num_frames`
  * `--fps`
  * `--segment_length`
  * `--motion_lora`
* [ ] Integrate Motion LoRA support (only one at a time)
* [ ] Build function: `generate_segment(keyframe_maps, segment_len)`
* [ ] Add latent-carryover between segments (frame 16 → next init)
* [ ] Validate 16-frame generation end-to-end

---

# Day 4 — Performance Optimization (P3)

* [ ] Enable `torch.compile(unet, mode="max-autotune")`
* [ ] Enforce SDPA / Flash Attention
* [ ] Use fp16 / bf16 end-to-end
* [ ] Add UNet tiling support (optional)
* [ ] Benchmark speeds:

  * Single-frame
  * 16-frame segment
  * Full 8–10s video
* [ ] Document VRAM usage for A100 20GB

---

# Day 5 — Keyframe & Control Pipeline (P2)

* [ ] Implement `generate_keyframes_from_prompt(prompt, n_keyframes)`
* [ ] Extract internal control maps for each keyframe
* [ ] Save to structured folder:

  ```
  keyframes/
    frame_000.png
    frame_001.png
  controlmaps/
    frame_000_pose.png
    frame_000_depth.png
    frame_000_edge.png
  ```
* [ ] Add JSON registry: `controlmaps/index.json`
* [ ] Validate quality + stability from self-generated maps

---

# Day 6 — Full Video Pipeline Assembly (P1 + P2 + P3)

* [ ] Build `run_video.py`:

  * Generate keyframes
  * Extract control maps (internal)
  * Generate segments with AnimateDiff
  * Chain segments
  * Save frames
* [ ] Add automatic ffmpeg export to mp4
* [ ] Add optional frame interpolation (if legal)
* [ ] Add SDXL upscaling option (also legal)
* [ ] Run full 8–10s generation test

---

# Day 7 — Reproducibility & Final Report

* [ ] Create `reproduce.sh`:

  * installs requirements
  * downloads SDXL + LoRAs + CNs
  * runs full pipeline
* [ ] Export seeds, configs, LoRA versions
* [ ] Add `RESULTS.md` with:

  * best images
  * motion analysis
  * flicker score
  * performance benchmarks
* [ ] Tag final commit as contest submission checkpoint

---

# Definition of Done

* Fully working SDXL‑only pipeline
* All ControlNet maps internally generated
* High‑quality anime stills
* Stable 8–10s anime video output
* Reproducible seeds + configs
* Contest‑compliant and SOTA optimized
