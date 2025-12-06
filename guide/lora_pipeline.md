# **SDXL Anime LoRA Training Pipeline**

Fully compliant with KAIST Visual Generation Contest rules
– *Only SDXL used as base model (allowed)*
– *No external pretrained generators (LoRA trained from scratch)*

---

## **0. Objective**

Train a **high-quality SDXL Anime LoRA** that produces AnimagineXL-level anime images using **Diffusers**, based on:

| Component       | Choice                                         |
| --------------- | ---------------------------------------------- |
| Model           | `stabilityai/stable-diffusion-xl-base-1.0`     |
| Dataset         | SFW Clean Danbooru subset (tagged + captioned) |
| Resolution      | 1024×1024                                      |
| GPU             | A100 20GB / A100 80GB                          |
| Expected output | `sdxl_anime_lora.safetensors`                  |

---

## **1. Dataset Acquisition**

Recommended primary dataset — **public, curated, tagged, SFW**:

| Dataset                                               | Why                              |
| ----------------------------------------------------- | -------------------------------- |
| CaptionEmporium/anime-caption-danbooru-2021-sfw-5m-hq | Large, captioned, SFW-filtered   |
| (Optional Booster) Anime Face Dataset                 | improves eyes & face consistency |

### Example Loader

```
from datasets import load_dataset
ds = load_dataset("CaptionEmporium/anime-caption-danbooru-2021-sfw-5m-hq", split="train")
```

---

## **2. Curation Pipeline**

Filtering criteria:

* Resolution min 768px
* Aspect ratio 0.7 to 1.4
* Only SFW/general-rated content

```
def is_good(example):
    w, h = example["width"], example["height"]
    ar = w / h
    if min(w, h) < 768: return False
    if not (0.7 <= ar <= 1.4): return False
    if example.get("rating") not in ["general", "safe"]: return False
    return True

ds_filtered = ds.filter(is_good, num_proc=8)
```

### Caption + Prompt Construction

```
IMPORTANT_TAGS = {
 "1girl","1boy","solo","looking_at_viewer","smile","long_hair","short_hair",
 "school_uniform","twintails","blue_eyes","brown_hair","blonde_hair",
 "upper_body","full_body","outdoors","indoors","night","day"
}

def build_prompt(example):
    tags = example["danbooru_tags"].split(" ")
    selected = [t.replace("_", " ") for t in tags if t in IMPORTANT_TAGS]
    base_caption = example.get("caption") or ""
    prompt = "masterpiece, best quality"
    if selected: prompt += ", " + ", ".join(selected)
    if base_caption: prompt += ", " + base_caption
    return {"prompt": prompt}

ds_curated = ds_filtered.map(build_prompt, num_proc=8)
```

### Export to Disk

```
from tqdm import tqdm
import os
os.makedirs("./images", exist_ok=True)
os.makedirs("./captions", exist_ok=True)

for i, ex in enumerate(tqdm(ds_curated)):
    ex["image"].save(f"images/{i:08d}.png")
    with open(f"captions/{i:08d}.txt","w") as f: f.write(ex["prompt"])
```

---

## **3. LoRA Training (Diffusers)**

Install environment:

```
uv venv .venv
source .venv/bin/activate
uv pip install diffusers transformers accelerate safetensors datasets bitsandbytes xformers
```

### Training Command

```
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export DATA_DIR="./"

accelerate launch examples/text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --train_data_dir ./images \
  --caption_column prompt \
  --resolution 1024 \
  --center_crop \
  --random_flip \
  --train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --use_8bit_adam \
  --learning_rate 1e-4 \
  --lr_scheduler cosine \
  --lr_warmup_steps 500 \
  --snr_gamma 5.0 \
  --noise_offset 0.03 \
  --max_train_steps 18000 \
  --lora_rank 64 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --output_dir outputs/sdxl_anime_lora
```

---

## **4. Inference**

```
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("outputs/sdxl_anime_lora")

img = pipe("masterpiece, best quality, anime girl").images[0]
img.save("sample.png")
```

---

## Finished

You are now ready to generate your SDXL Anime LoRA.
