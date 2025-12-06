"""
Streaming dataset builder for SDXL Anime LoRA.

Mirrors `guide/dataset_builder.md` with a practical `--max-samples` control so we
can cap the subset size (default 30k). Filters to SFW, 768px+, aspect ratio
0.7–1.4, center-crops to square, resizes to 1024, and exports paired
`images/*.png` and `captions/*.txt`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Any, Dict, Iterable, List, Optional

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# ================================
# CONFIG DEFAULTS
# ================================
DATASET_NAME = "CaptionEmporium/anime-caption-danbooru-2021-sfw-5m-hq"
DATASET_SPLIT = "train"

OUT_ROOT = Path("./sdxl_anime_lora_dataset")
IMAGES_DIR = OUT_ROOT / "images"
CAPTIONS_DIR = OUT_ROOT / "captions"

MIN_RES = 768
MIN_AR, MAX_AR = 0.7, 1.4
ALLOWED_RATINGS = {"general", "safe"}

IMPORTANT_TAGS = {
    "1girl",
    "1boy",
    "solo",
    "looking_at_viewer",
    "smile",
    "long_hair",
    "short_hair",
    "twintails",
    "school_uniform",
    "blue_eyes",
    "green_eyes",
    "brown_eyes",
    "red_eyes",
    "blonde_hair",
    "blue_hair",
    "brown_hair",
    "black_hair",
    "pink_hair",
    "upper_body",
    "full_body",
    "outdoors",
    "indoors",
    "night",
    "day",
}


def _get_image_size(example: Dict[str, Any]) -> Optional[tuple[int, int]]:
    """Get width/height from metadata or the PIL image itself."""
    w, h = example.get("width"), example.get("height")
    if w is None or h is None:
        img = example.get("image")
        if isinstance(img, Image.Image):
            w, h = img.size
    if w is None or h is None:
        return None
    return int(w), int(h)


def is_good(example: Dict[str, Any]) -> bool:
    """Filtering criteria."""
    wh = _get_image_size(example)
    if wh is None:
        return False
    w, h = wh
    if min(w, h) < MIN_RES:
        return False
    ar = w / h
    if not (MIN_AR <= ar <= MAX_AR):
        return False
    rating = example.get("rating")
    if rating and rating not in ALLOWED_RATINGS:
        return False
    tags_str = example.get("danbooru_tags", "") or ""
    if tags_str and any(t in tags_str for t in ["comic", "meme", "screenshot"]):
        return False
    return True


def _extract_tags(example: Dict[str, Any]) -> List[str]:
    """Extract tag list from known fields."""
    # 1) direct danbooru tags
    raw = (example.get("danbooru_tags") or "").split()
    if raw:
        return [t.replace("_", " ") for t in raw if t in IMPORTANT_TAGS]

    # 2) wd_swinv2_tagger_v3_tags JSON
    wd_json = example.get("wd_swinv2_tagger_v3_tags")
    if wd_json:
        try:
            data = json.loads(wd_json)
            general = data.get("general", {}) or {}
            # keep top tags above 0.35 prob
            sorted_tags = sorted(general.items(), key=lambda x: x[1], reverse=True)
            return [
                t.replace("_", " ")
                for t, score in sorted_tags
                if score >= 0.35 and t in IMPORTANT_TAGS
            ]
        except Exception:
            pass

    return []


def _extract_caption(example: Dict[str, Any]) -> str:
    """Pick a caption/text field in priority order."""
    for key in [
        "caption",
        "caption_llava_34b_no_tags_short",
        "caption_llava_34b_no_tags",
        "caption_llava_34b",
        "caption_cogvlm",
        "text",
        "mldanbooru_tag_caption",
    ]:
        val = example.get(key)
        if val:
            return str(val).strip()
    return ""


def build_prompt(example: Dict[str, Any]) -> str:
    """Construct prompt from tags + caption."""
    selected = _extract_tags(example)
    base_cap = _extract_caption(example)
    prompt_parts = ["masterpiece", "best quality", "anime illustration"]
    if selected:
        prompt_parts += selected
    if base_cap:
        prompt_parts.append(base_cap)

    used = set()
    final: list[str] = []
    for p in prompt_parts:
        if p not in used:
            used.add(p)
            final.append(p)
    return ", ".join(final)


def export_samples(
    ds_stream: Iterable[Dict[str, Any]],
    max_samples: int,
) -> None:
    """Iterate streaming dataset, filter, and export images/captions."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    CAPTIONS_DIR.mkdir(parents=True, exist_ok=True)

    kept = 0
    for ex in tqdm(ds_stream, total=max_samples, desc="Exporting", unit="img"):
        if kept >= max_samples:
            break
        if not is_good(ex):
            continue

        img: Image.Image = ex["image"].convert("RGB")
        w, h = img.size
        ms = min(w, h)
        left = (w - ms) // 2
        top = (h - ms) // 2
        img = img.crop((left, top, left + ms, top + ms)).resize((1024, 1024))

        prompt = build_prompt(ex)
        img.save(IMAGES_DIR / f"{kept:08d}.png")
        with open(CAPTIONS_DIR / f"{kept:08d}.txt", "w") as f:
            f.write(prompt)

        kept += 1

    print(f"DONE → kept {kept} samples into {OUT_ROOT}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SDXL Anime LoRA dataset subset")
    _ = parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NAME,
        help="HF dataset name or path",
    )
    _ = parser.add_argument(
        "--split",
        type=str,
        default=DATASET_SPLIT,
        help="Dataset split",
    )
    _ = parser.add_argument(
        "--max-samples",
        type=int,
        default=30_000,
        help="Number of filtered samples to export",
    )
    _ = parser.add_argument(
        "--out-root",
        type=str,
        default=str(OUT_ROOT),
        help="Output root directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global OUT_ROOT, IMAGES_DIR, CAPTIONS_DIR
    OUT_ROOT = Path(args.out_root)
    IMAGES_DIR = OUT_ROOT / "images"
    CAPTIONS_DIR = OUT_ROOT / "captions"

    print(f"Loading dataset {args.dataset} split={args.split} (streaming)...")
    ds_stream = load_dataset(
        args.dataset,
        split=args.split,
        streaming=True,
    )

    export_samples(ds_stream, max_samples=args.max_samples)


if __name__ == "__main__":
    main()

