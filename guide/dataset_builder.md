# DATASET_BUILDER.md

## SDXL Anime LoRA Dataset Builder

This document defines a **complete pipeline** for building a dataset used for training an **SDXL Anime LoRA using Diffusers**.

You will:

1. Download an anime dataset (HuggingFace / Danbooru-derived)
2. Filter for **SFW**, high-resolution images
3. Build captions/prompts from tags
4. Export final format:

```
sdxl_anime_lora_dataset/
  images/00000000.png
  captions/00000000.txt
  ...
```

This format is directly usable with `TRAIN_SDXL_LORA.py`.

---

## 0. Requirements

```bash
uv venv .venv
source .venv/bin/activate
uv pip install datasets pillow tqdm
```

---

## 1. Dataset Source

Recommended primary dataset:

| Dataset                                               | Why                                       |
| ----------------------------------------------------- | ----------------------------------------- |
| CaptionEmporium/anime-caption-danbooru-2021-sfw-5m-hq | Already SFW-filtered + tagged + captioned |
| (Optional) Anime Face dataset                         | boosts eye/face clarity                   |

If using HF dataset directly, you will not need manual Danbooru downloads.

---

## 2. Create dataset_builder.py

This script downloads, filters, tags, captions, and exports usable data.

```python
# dataset_builder.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ================================
# CONFIG
# ================================
DATASET_NAME = "CaptionEmporium/anime-caption-danbooru-2021-sfw-5m-hq"
DATASET_SPLIT = "train"

OUT_ROOT = Path("./sdxl_anime_lora_dataset")
IMAGES_DIR = OUT_ROOT / "images"
CAPTIONS_DIR = OUT_ROOT / "captions"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)

MIN_RES = 768
MIN_AR, MAX_AR = 0.7, 1.4
ALLOWED_RATINGS = {"general", "safe"}

IMPORTANT_TAGS = {
    "1girl","1boy","solo","looking_at_viewer","smile","long_hair","short_hair",
    "twintails","school_uniform","blue_eyes","green_eyes","brown_eyes","red_eyes",
    "blonde_hair","blue_hair","brown_hair","black_hair","pink_hair",
    "upper_body","full_body","outdoors","indoors","night","day"
}

# ================================
# FILTER CHECKS
# ================================
def is_good(example: Dict[str, Any]) -> bool:
    w, h = example.get("width"), example.get("height")
    if w is None or h is None: return False
    if min(w, h) < MIN_RES: return False
    ar = w / h
    if not (MIN_AR <= ar <= MAX_AR): return False
    rating = example.get("rating")
    if rating and rating not in ALLOWED_RATINGS: return False
    tags_str = example.get("danbooru_tags", "")
    if any(t in tags_str for t in ["comic","meme","screenshot"]): return False
    return True

# ================================
# PROMPT BUILDER
# ================================
def build_prompt(example: Dict[str, Any]) -> str:
    tags = example.get("danbooru_tags", "").split()
    selected = [t.replace("_"," ") for t in tags if t in IMPORTANT_TAGS]
    base_cap = (example.get("caption") or "").strip()
    prompt = ["masterpiece","best quality","anime illustration"]
    if selected: prompt += selected
    if base_cap: prompt.append(base_cap)
    used=set(); final=[]
    for p in prompt:
        if p not in used:
            used.add(p); final.append(p)
    return ", ".join(final)

# ================================
# BUILD + EXPORT
# ================================
def main():
    print(f"Loading dataset {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    ds_filtered = ds.filter(is_good, num_proc=8)
    print(f"Filtered: {len(ds)} → {len(ds_filtered)}")

    idx=0
    for ex in tqdm(ds_filtered,desc="Exporting"):
        img=ex["image"].convert("RGB")
        w,h=img.size; ms=min(w,h)
        left=(w-ms)//2; top=(h-ms)//2
        img=img.crop((left,top,left+ms,top+ms)).resize((1024,1024))

        prompt=build_prompt(ex)
        img.save(IMAGES_DIR/f"{idx:08d}.png")
        with open(CAPTIONS_DIR/f"{idx:08d}.txt","w") as f: f.write(prompt)
        idx+=1
    print("DONE → Dataset ready for LoRA training.")

if __name__=="__main__": main()
```

---

## 3. Run

```bash
uv run python dataset_builder.py
```

Dataset output structure:

```
sdxl_anime_lora_dataset/
  images/*.png
  captions/*.txt
```

Use this with `TRAIN_SDXL_LORA.py` to train your SDXL Anime LoRA.
