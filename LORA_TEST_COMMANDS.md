# LoRA Testing Commands

## Quick Test (Copy-Paste Ready)

### With LCM (Fast - 5 steps)
```bash
python infer/generate_image.py --prompt "anime girl with blue hair, beautiful detailed eyes, cel shading, clean lines, masterpiece" --seed 123 --profile 768_lcm --scheduler dpm --negative-prompt "blurry, low quality, bad anatomy" --out outputs/test_with_lcm.png
```

### Without LCM (High Quality - 26 steps)
```bash
python infer/generate_image.py --prompt "anime girl with blue hair, beautiful detailed eyes, cel shading, clean lines, masterpiece" --seed 123 --profile 1024_hq --scheduler dpm --negative-prompt "blurry, low quality, bad anatomy" --out outputs/test_without_lcm.png
```

## Verify LoRA is Active

Look for this in the output:
```
Active LoRAs (1):
  - Pastel Anime XL (style): weight=0.8
```

## Expected Results

- **With LCM**: ~5 seconds, 5 steps, CFG 1.7
- **Without LCM**: ~20-30 seconds, 26 steps, CFG 6.0

Both will have Pastel Anime XL LoRA applied for anime styling.
