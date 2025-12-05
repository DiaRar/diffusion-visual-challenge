# ControlNet Implementation Guide: Reducing Flicker in Video Generation

## 🎯 How ControlNet Reduces Flicker & Artifacts

### The Problem: AnimateDiff Alone

When you generate video frames with **AnimateDiff only**, each frame is generated somewhat independently:

```
Frame 0: "anime girl standing" → Model generates: pose A, background X
Frame 1: "anime girl standing" → Model generates: pose B, background Y  ❌ Different!
Frame 2: "anime girl standing" → Model generates: pose C, background Z  ❌ Different!
```

**Result:** Flicker, pose drift, background changes, inconsistent lighting

### The Solution: ControlNet + AnimateDiff

ControlNet **locks structure** at keyframes, while AnimateDiff **interpolates motion**:

```
Keyframe 0: Extract pose/depth/edge maps
Frame 0-15: AnimateDiff interpolates motion
            ControlNet locks: pose structure ✓, depth structure ✓, edge structure ✓
```

**Result:** Smooth motion with stable structure = no flicker!

### What Each Control Map Fixes

| Control Map | What It Locks | Prevents |
|------------|---------------|----------|
| **Pose** | Body position, limb angles | Pose drift, arm/leg flicker |
| **Depth** | Foreground/background separation | Background drift, size changes |
| **Edge/Canny** | Character outline, shape | Shape flicker, outline inconsistency |

---

## 🚀 Implementation

### Step 1: Generate Keyframe & Extract Maps

You already have this! Use your existing function:

```python
from infer.keyframes import generate_keyframe_with_maps

# Generate keyframe and extract all control maps
keyframe, maps, saved_paths = generate_keyframe_with_maps(
    prompt="anime girl with blue hair, standing pose",
    seed=123,
    profile_name="768_long",
)
```

This gives you:
- `maps.pose` - Pose stick figure
- `maps.depth` - Depth map (grayscale)
- `maps.edge` - Edge/lineart map

### Step 2: Generate Image with ControlNet

Now use those maps to guide generation:

```python
from infer.keyframes import generate_with_control_maps

# Generate new image with same pose
new_image = generate_with_control_maps(
    prompt="anime girl with red hair, different outfit",
    control_maps=maps,
    controlnet_type="pose",  # Use pose map
    seed=456,  # Different seed = different style
    controlnet_conditioning_scale=0.8,  # How strongly to follow map
)
```

**Result:** New image has different hair/outfit, but **same pose structure**!

### Step 3: CLI Usage

You can also use it from command line:

```bash
# First, extract control maps (you already have this)
python infer/test_keyframes.py

# Then generate with ControlNet
python infer/generate_image.py \
    --prompt "anime girl, red hair, same pose" \
    --controlnet-type pose \
    --controlnet-images outputs/controlmaps/keyframe_XXX_pose.png \
    --controlnet-scale 0.8 \
    --profile 768_long \
    --seed 456
```

---

## 🎬 How This Connects to Video Generation

### The Full Video Pipeline (Future: Day 3-6)

```python
# 1. Generate keyframes (Day 5)
keyframes = generate_keyframes_from_prompt(
    prompt="anime girl walking",
    n_keyframes=4,
    seeds=[123, 456, 789, 1011]
)

# 2. Extract control maps for each keyframe (YOU HAVE THIS ✅)
all_maps = []
for kf in keyframes:
    _, maps, _ = generate_keyframe_with_maps(...)  # Your function!
    all_maps.append(maps)

# 3. Generate video segments with AnimateDiff + ControlNet (Day 3)
for i, maps in enumerate(all_maps):
    segment = generate_segment_with_controlnet(
        prompt="anime girl walking",
        control_maps=maps,  # Your extracted maps!
        num_frames=16,
        controlnet_type="pose",
        controlnet_conditioning_scale=0.8
    )
    # AnimateDiff interpolates motion
    # ControlNet locks structure at keyframe
```

### Why This Works

**AnimateDiff** handles:
- Temporal consistency (smooth motion between frames)
- Motion interpolation (walking animation)

**ControlNet** handles:
- Structural consistency (pose stays locked)
- Background stability (depth map prevents drift)
- Outline stability (edge map prevents shape flicker)

**Together:** Smooth motion + stable structure = **no flicker!**

---

## 📊 ControlNet Conditioning Scale

The `controlnet_conditioning_scale` parameter controls how strongly ControlNet enforces structure:

| Scale | Effect | Use Case |
|-------|--------|----------|
| **0.0** | No control (ignores map) | Not useful |
| **0.5** | Weak control | Subtle guidance, more creative freedom |
| **0.8** | **Recommended** | Good balance (default) |
| **1.0** | Strong control | Strict structure matching |
| **1.5+** | Very strong | May cause artifacts, over-constraining |

**For video:** Start with `0.8`, adjust if needed:
- Too much flicker? Increase to `1.0`
- Too rigid/stiff? Decrease to `0.6`

---

## 🧪 Testing Your Implementation

Run the test script:

```bash
python infer/test_controlnet.py
```

This will:
1. Generate a keyframe
2. Extract control maps
3. Generate a new image using the pose map
4. Verify the pose matches (different style, same structure)

**Expected output:**
- New image has different colors/outfit (different seed)
- But **same pose structure** (thanks to ControlNet!)

---

## 🔧 Troubleshooting

### "ControlNet model not found"

ControlNet models are downloaded automatically on first use. They're large (~1-2GB each):
- `diffusers/controlnet-openpose-sdxl-1.0`
- `diffusers/controlnet-depth-sdxl-1.0`
- `diffusers/controlnet-canny-sdxl-1.0`

First run will download them (may take a few minutes).

### "Control map not available"

Make sure you extracted the maps first:
```python
maps = extract_control_maps(keyframe_image)
# Check: maps.pose, maps.depth, maps.edge
```

### Images don't match structure

- Increase `controlnet_conditioning_scale` (try 1.0)
- Check that control map is correct (visualize it)
- Ensure control map matches target resolution

---

## 🎯 Next Steps for Video Generation

Now that ControlNet is implemented:

1. **Day 3: AnimateDiff Integration**
   - Load AnimateDiff motion module
   - Create `generate_segment()` that uses your control maps
   - Test 16-frame generation

2. **Day 5: Keyframe Pipeline**
   - Generate multiple keyframes
   - Extract maps for each (you have this!)
   - Chain them together

3. **Day 6: Full Video Pipeline**
   - Combine AnimateDiff + ControlNet
   - Generate full 8-10s video
   - Export to MP4

**The control maps you extract are the "glue" that makes everything stable!** 🚀

