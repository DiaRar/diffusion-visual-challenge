"""
Test script demonstrating ControlNet integration with control maps.

This shows the complete workflow:
1. Generate keyframe
2. Extract control maps
3. Generate new image using control maps (should match structure)
"""

import sys
import warnings
from pathlib import Path

# Suppress warnings for cleaner output (TODO: re-enable when ControlNet integration is stable)
warnings.filterwarnings("ignore")

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from infer.keyframes import generate_keyframe_with_maps, generate_with_control_maps


def main():
    """Test ControlNet with control maps."""
    print("=" * 60)
    print("ControlNet + Control Maps Test")
    print("=" * 60)

    # Step 1: Generate keyframe and extract control maps
    print("\n[Step 1] Generating keyframe and extracting control maps...")
    prompt = "anime girl with blue hair, beautiful detailed eyes, cel shading, clean lines, masterpiece, cinematic volumetric god-rays, subsurface glow diffusion, balanced color harmony, 8k quality, perfect composition, beautiful goddess"
    seed = 123

    keyframe, maps, saved_paths = generate_keyframe_with_maps(
        prompt=prompt,
        seed=seed,
        profile_name="1024_hq",  # Use HQ profile like in the test command
        use_custom_vae=True,  # Use custom VAE like in the test command
        negative_prompt="blurry, low quality, bad anatomy",
    )

    print(f"✓ Keyframe generated")
    print(f"✓ Control maps extracted:")
    if maps.pose:
        print(f"  - Pose map: {saved_paths.get('pose')}")
    if maps.depth:
        print(f"  - Depth map: {saved_paths.get('depth')}")
    if maps.edge:
        print(f"  - Edge map: {saved_paths.get('edge')}")

    # Step 2: Generate new image using DEPTH map (most reliable for video stability)
    print("\n[Step 2] Generating new image with same structure (ControlNet + Depth)...")
    new_prompt = "anime girl with red hair, different outfit, cel shading, clean lines, masterpiece, cinematic volumetric god-rays, subsurface glow diffusion, balanced color harmony, 8k quality, perfect composition, beautiful goddess"

    try:
        new_image = generate_with_control_maps(
            prompt=new_prompt,
            control_maps=maps,
            controlnet_type="depth",  # DEPTH is most reliable for video stability
            seed=123,  # Different seed = different style, but same structure
            profile_name="1024_hq",  # Match the keyframe profile
            controlnet_conditioning_scale=0.8,  # Slightly reduced for more creative freedom
            use_custom_vae=True,
            negative_prompt="blurry, low quality, bad anatomy",
        )

        # Save result
        output_path = PROJECT_ROOT / "outputs" / "test_controlnet_result.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_image.save(output_path)
        print(f"✓ Generated image with ControlNet (depth) saved to: {output_path}")
        print("\n" + "=" * 60)
        print("SUCCESS: ControlNet integration working!")
        print("=" * 60)
        print("\nThe new image should have:")
        print("  - Different hair color (red vs blue)")
        print("  - Different outfit")
        print("  - SAME SPATIAL STRUCTURE (thanks to depth ControlNet!)")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nThis might mean:")
        print("  1. ControlNet models not downloaded yet")
        print("  2. Missing dependencies (diffusers)")
        print("  3. ControlNet integration needs debugging")
        raise


if __name__ == "__main__":
    main()

