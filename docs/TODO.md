**TODO.md — SOTA Anime Video Pipeline (Week Plan for 3-person Team)**

## A) Setup & Compliance

* [x] Enforce SDXL base-only constraint (no alt diffusion backbones)
* [x] Enable LoRA loading with `--lora`, `--lora_scale`
* [x] Add block flags for forbidden inputs: `--refiner`, `--vae_override`, `--external_diffusion`
* [x] Log `run_manifest.json`: args, seed, GPU, git hash, dtype, compile flags
* [x] Enable `--torch_compile`, flash attention, bf16/fp16 mode

---

## B) Prompt Suite and Validation

* [x] Create `configs/prompt_suite.json` (~30 prompts, varied scenes)
* [x] Create `scripts/eval_grid.py` to evaluate prompt outputs
* [ ] Add scoring rubric (face, style, flicker, structure)
* [ ] Reject commits regressing prompt grid results

---

## Day 1: LoRA Inference & Setup (P3)

* [x] Load SDXL with LoRA (style + optional char LoRA)
* [x] Load LCM LoRA and adjust steps (4–6) and CFG (1.5–2)
* [x] Add support for torch.compile, SDPA, bf16
* [ ] Verify reproducibility (same seed = same frames)
* [ ] Add deterministic logging (`seed`, `adapter_versions`, etc)

---

## Day 2: ControlNet Integration (P2)

* [ ] Add ControlNet loader: OpenPose, Depth, Lineart
* [ ] Support `--controlnet_images` per frame or segment
* [ ] Enable sparse conditioning (SparseCtrl-compatible)
* [ ] Add fallback to keyframe-only guidance if full seq too large
* [ ] Auto-check shape compatibility with SDXL latents

---

## Day 3: AnimateDiff Integration (P1)

* [ ] Load AnimateDiff SDXL motion module
* [ ] Add CLI `--motion_module`, `--num_frames`, `--fps`
* [ ] Support segment chaining: 16-frame chunks w/ overlap
* [ ] Optional: Add motion LoRA support (camera pan, zoom)

---

## Day 4: Performance

* [ ] Support LCM LoRA sampling (4–6 steps, fast sampler)
* [ ] Enable torch.compile UNet
* [ ] Optimize VRAM: SDPA + bf16 + UNet tiling
* [ ] Benchmark single frame, 16-frame, and 64-frame generation times

---

## Day 5: Regression Harness (P3)

* [ ] Add `reports/val/step_XXXX/` dumps per checkpoint
* [ ] Include prompt suite results: images, videos, metadata
* [ ] Compare against past runs using hashes and frame metrics

---

## Day 6: Output & Evaluation

* [ ] Run final 8–10s video generation
* [ ] Export frames and compile to mp4
* [ ] Generate prompt grid for final showcase
* [ ] Write `reports/RESULTS.md` summary: performance, visuals, motion

---

## Day 7: Reproducibility Package

* [ ] `scripts/reproduce.sh`: full env + infer run
* [ ] Include all adapter IDs, seeds, CLI args
* [ ] Add markdown: How to run and modify

---

## Stretch (if time)

* [ ] Add post-process frame interpolation tool
* [ ] Add GUI for visualizing prompt → video mapping
* [ ] Merge LCM + Motion LoRA for one-click config

---

**Team Roles:**

* P1 (Lead): AnimateDiff, video, motion LoRAs
* P2: ControlNet adapters, frame prep
* P3: Inference infra, LoRA loading, performance

**Definition of Done:**

* At least one 8–10s anime video at 768p
* Uses SDXL base + adapters only
* Reproducible pipeline: CLI, configs, prompt, seed
