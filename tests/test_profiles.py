"""Tests for configs/profiles.py"""

import pytest

from configs.profiles import (
    PROFILES,
    get_profile,
    list_profiles,
    validate_profile,
    Profile,
)


class TestProfile:
    """Test Profile dataclass"""

    def test_profile_creation(self):
        """Should create Profile with all attributes"""
        profile = Profile(
            name="test",
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=7.5,
            description="Test profile",
        )

        assert profile.name == "test"
        assert profile.height == 512
        assert profile.width == 512
        assert profile.num_inference_steps == 20
        assert profile.guidance_scale == 7.5
        assert profile.description == "Test profile"

    def test_profile_frozen(self):
        """Profile should be immutable (frozen)"""
        profile = Profile(
            name="test",
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=7.5,
            description="Test profile",
        )

        with pytest.raises(AttributeError):
            profile.height = 768


class TestProfiles:
    """Test predefined profiles"""

    def test_smoke_profile_exists(self):
        """Smoke profile should be defined"""
        profile = PROFILES["smoke"]
        assert profile.name == "smoke"
        assert profile.height == 128
        assert profile.width == 128
        assert profile.num_inference_steps == 4

    def test_768_long_profile_exists(self):
        """768_long profile should be defined"""
        profile = PROFILES["768_long"]
        assert profile.name == "768_long"
        assert profile.height == 768
        assert profile.width == 768
        assert profile.num_inference_steps == 22

    def test_1024_hq_profile_exists(self):
        """1024_hq profile should be defined"""
        profile = PROFILES["1024_hq"]
        assert profile.name == "1024_hq"
        assert profile.height == 1024
        assert profile.width == 1024
        assert profile.num_inference_steps == 26


class TestGetProfile:
    """Test get_profile function"""

    def test_get_valid_profile(self):
        """Should return profile for valid name"""
        profile = get_profile("smoke")
        assert profile is not None
        assert profile.name == "smoke"

    def test_get_invalid_profile(self):
        """Should raise ValueError for invalid name"""
        with pytest.raises(ValueError) as exc_info:
            get_profile("nonexistent")

        assert "Unknown profile" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)

    def test_get_profile_available_list(self):
        """Error message should list available profiles"""
        with pytest.raises(ValueError) as exc_info:
            get_profile("invalid")

        error_msg = str(exc_info.value)
        assert "smoke" in error_msg
        assert "768_long" in error_msg
        assert "1024_hq" in error_msg


class TestListProfiles:
    """Test list_profiles function"""

    def test_list_profiles_returns_string(self):
        """Should return a string with all profiles"""
        result = list_profiles()
        assert isinstance(result, str)
        assert "smoke" in result
        assert "768_long" in result
        assert "1024_hq" in result

    def test_list_profiles_format(self):
        """Should format profiles correctly"""
        result = list_profiles()
        lines = result.split("\n")
        assert len(lines) >= 3  # At least one line per profile


class TestValidateProfile:
    """Test validate_profile function"""

    def test_validate_correct_dimensions(self):
        """Should return True for matching dimensions"""
        assert validate_profile("smoke", 128, 128) is True
        assert validate_profile("768_long", 768, 768) is True

    def test_validate_incorrect_dimensions(self):
        """Should return False for non-matching dimensions"""
        assert validate_profile("smoke", 768, 768) is False
        assert validate_profile("768_long", 128, 128) is False

    def test_validate_invalid_profile(self):
        """Should return False for invalid profile name"""
        assert validate_profile("nonexistent", 512, 512) is False

    def test_validate_partial_match(self):
        """Should return False if only height matches"""
        assert validate_profile("smoke", 128, 256) is False
        assert validate_profile("smoke", 256, 128) is False
