"""Integration tests for image generation pipeline"""

import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from infer.generate_image import generate_single_image


def test_generate_image_with_smoke_profile():
    """Test generating image with smoke profile"""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_smoke.png"
        
        result = generate_single_image(
            prompt="anime girl, blue hair",
            backbone="sdxl",
            profile_name="smoke",
            scheduler_mode="euler",
            seed=123,
            out_path=str(out_path),
        )
        
        assert result.exists()
        assert result.suffix == ".png"
        print(f"✓ Generated smoke test image: {result}")


def test_generate_image_with_lcm_profile():
    """Test generating image with LCM profile (uses configured LoRAs)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_lcm.png"
        
        result = generate_single_image(
            prompt="anime character, cel shading, beautiful detailed eyes",
            backbone="sdxl",
            profile_name="768_lcm",
            scheduler_mode="dpm",
            seed=456,
            out_path=str(out_path),
        )
        
        assert result.exists()
        assert result.suffix == ".png"
        print(f"✓ Generated LCM image with LoRAs: {result}")


def test_generate_image_with_768_long():
    """Test generating image with 768_long profile"""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_768.png"
        
        result = generate_single_image(
            prompt="anime boy, dynamic pose, flowing hair",
            backbone="sdxl",
            profile_name="768_long",
            scheduler_mode="dpm",
            seed=789,
            out_path=str(out_path),
            negative_prompt="blurry, low detail",
        )
        
        assert result.exists()
        assert result.suffix == ".png"
        print(f"✓ Generated 768_long image: {result}")


def test_generate_image_with_torch_compile():
    """Test generating image with torch.compile enabled"""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_compile.png"
        
        result = generate_single_image(
            prompt="anime scene, dramatic lighting",
            backbone="sdxl",
            profile_name="smoke",
            scheduler_mode="euler",
            seed=999,
            out_path=str(out_path),
            torch_compile=True,
        )
        
        assert result.exists()
        assert result.suffix == ".png"
        print(f"✓ Generated image with torch.compile: {result}")


if __name__ == "__main__":
    print("\nRunning integration tests...")
    print("=" * 80)
    
    try:
        test_generate_image_with_smoke_profile()
        test_generate_image_with_lcm_profile()
        test_generate_image_with_768_long()
        test_generate_image_with_torch_compile()
        
        print("=" * 80)
        print("✓ All integration tests passed!")
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
