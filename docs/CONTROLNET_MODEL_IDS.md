# ControlNet Model IDs for SDXL

If you get a "Repository Not Found" error, the model IDs in `infer/controlnet_loader.py` may need to be updated.

## Finding Correct Model IDs

1. Go to https://huggingface.co/models
2. Search for: `controlnet sdxl openpose` (or `depth`, `canny`, `lineart`)
3. Look for models that are:
   - Compatible with SDXL (not SD 1.5)
   - Published by trusted organizations
   - Have recent activity

## Common Model ID Patterns

Try these patterns (update in `infer/controlnet_loader.py`):

```python
CONTROLNET_MODELS = {
    "pose": "thibaud/controlnet-openpose-sdxl-1.0",  # Try this first
    "depth": "thibaud/controlnet-depth-sdxl-1.0",
    "canny": "thibaud/controlnet-canny-sdxl-1.0",
    "lineart": "thibaud/controlnet-lineart-sdxl-1.0",
}
```

Or if that doesn't work, try:
- `diffusers/controlnet-openpose-sdxl-1.0` (if diffusers org has them)
- Check the official ControlNet repository for SDXL support

## Alternative: Use SD 1.5 ControlNet (Not Recommended)

SD 1.5 ControlNet models exist but won't work with SDXL:
- `lllyasviel/sd-controlnet-openpose` (SD 1.5 only)
- `lllyasviel/sd-controlnet-depth` (SD 1.5 only)

**These won't work with your SDXL pipeline!**

## Quick Test

To test if a model ID is correct, try loading it directly:

```python
from diffusers import ControlNetModel

# Test if this model exists
try:
    model = ControlNetModel.from_pretrained("thibaud/controlnet-openpose-sdxl-1.0")
    print("✓ Model ID is correct!")
except Exception as e:
    print(f"✗ Model ID failed: {e}")
    print("Try searching HuggingFace for the correct ID")
```

## Current Status

The code currently uses `thibaud/controlnet-*-sdxl-1.0` pattern. If this fails, update the `CONTROLNET_MODELS` dictionary in `infer/controlnet_loader.py` with the correct IDs you find on HuggingFace.

