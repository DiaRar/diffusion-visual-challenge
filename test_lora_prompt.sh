#!/bin/bash

# Test prompt to verify LoRA integration is working
# This will generate an image with all active LoRAs

echo "Testing LoRA Integration..."
echo "Active LoRAs:"
echo "  - Pastel Anime XL (style)"
echo "  - LCM SDXL (fast sampling)"
echo ""

uv run python infer/generate_image.py \
  --prompt "anime girl with blue hair, beautiful detailed eyes, cel shading, clean lines, masterpiece" \
  --seed 123 \
  --profile 768_lcm \
  --scheduler dpm \
  --out outputs/test_lora_active.png \
  --torch-compile

echo ""
echo "✓ Check outputs/test_lora_active.png for results"
echo "Expected: Anime-styled image with Pastel Anime XL LoRA applied + fast LCM sampling"
