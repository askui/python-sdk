"""Tests for LLM image utility functions."""

import logging

import pytest
from PIL import Image

from askui.utils.llm_image_utils import (
    compute_contained_size,
    compute_patch_optimized_size,
    count_image_tokens,
    resize_and_pad_image,
    resize_image,
)


class TestCountImageTokens:
    def test_exact_patches(self) -> None:
        # 56x56 with patch_size=28 → 2x2 = 4 tokens
        assert count_image_tokens(56, 56, patch_size=28) == 4

    def test_single_patch(self) -> None:
        assert count_image_tokens(28, 28, patch_size=28) == 1

    def test_partial_patches_round_up(self) -> None:
        # 30x30 with patch_size=28 → ceil(30/28) * ceil(30/28) = 2*2 = 4
        assert count_image_tokens(30, 30, patch_size=28) == 4

    def test_known_anthropic_value(self) -> None:
        # 1568x1568 with patch_size=28 → 56*56 = 3136
        assert count_image_tokens(1568, 1568, patch_size=28) == 3136

    def test_rectangular(self) -> None:
        # 1024x768 with patch_size=28 → ceil(1024/28)*ceil(768/28) = 37*28 = 1036
        assert count_image_tokens(1024, 768, patch_size=28) == 37 * 28


class TestComputePatchOptimizedSize:
    def test_small_image_unchanged(self) -> None:
        # A small image that fits within all constraints is returned as-is
        w, h = compute_patch_optimized_size(200, 100)
        assert w == 200
        assert h == 100

    def test_respects_max_edge(self) -> None:
        w, h = compute_patch_optimized_size(3000, 2000, max_edge=1568)
        assert w <= 1568
        assert h <= 1568

    def test_respects_max_tokens(self) -> None:
        w, h = compute_patch_optimized_size(
            1920, 1080, max_edge=1568, max_tokens=1568, patch_size=28
        )
        tokens = count_image_tokens(w, h, patch_size=28)
        assert tokens <= 1568

    def test_preserves_aspect_ratio(self) -> None:
        w, h = compute_patch_optimized_size(1920, 1080)
        original_ratio = 1920 / 1080
        result_ratio = w / h
        assert abs(original_ratio - result_ratio) / original_ratio < 0.02

    def test_invalid_dimensions_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_patch_optimized_size(0, 100)

    def test_openai_params(self) -> None:
        w, h = compute_patch_optimized_size(
            1920, 1080, max_edge=2048, max_tokens=1536, patch_size=32
        )
        tokens = count_image_tokens(w, h, patch_size=32)
        assert tokens <= 1536
        assert w <= 2048
        assert h <= 2048


class TestComputeContainedSize:
    def test_already_fits(self) -> None:
        assert compute_contained_size(800, 600, 1024, 768) == (800, 600)

    def test_exact_match(self) -> None:
        assert compute_contained_size(1024, 768, 1024, 768) == (1024, 768)

    def test_landscape_too_wide(self) -> None:
        w, h = compute_contained_size(2048, 768, 1024, 768)
        assert w <= 1024
        assert h <= 768

    def test_portrait_too_tall(self) -> None:
        w, h = compute_contained_size(768, 2048, 1024, 768)
        assert w <= 1024
        assert h <= 768

    def test_preserves_aspect_ratio(self) -> None:
        w, h = compute_contained_size(1920, 1080, 1024, 768)
        original_ratio = 1920 / 1080
        result_ratio = w / h
        assert abs(original_ratio - result_ratio) / original_ratio < 0.02

    def test_invalid_dimensions_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_contained_size(0, 100)


class TestResizeImage:
    def test_correct_dimensions(self) -> None:
        img = Image.new("RGB", (1920, 1080))
        result = resize_image(img, (1024, 576))
        assert result.size == (1024, 576)

    def test_no_op_when_same_size(self) -> None:
        img = Image.new("RGB", (1024, 768))
        result = resize_image(img, (1024, 768))
        assert result is img  # Same object, no copy

    def test_aspect_ratio_warning_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        img = Image.new("RGB", (1920, 1080))
        with caplog.at_level(logging.WARNING):
            resize_image(img, (1024, 768))
        assert "Aspect ratio change" in caplog.text

    def test_no_warning_when_ratio_preserved(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        img = Image.new("RGB", (1920, 1080))
        with caplog.at_level(logging.WARNING):
            resize_image(img, (960, 540))
        assert "Aspect ratio change" not in caplog.text


class TestResizeAndPadImage:
    def test_correct_dimensions(self) -> None:
        img = Image.new("RGB", (1920, 1080))
        result = resize_and_pad_image(img, (1024, 768))
        assert result.size == (1024, 768)

    def test_preserves_aspect_ratio_with_padding(self) -> None:
        img = Image.new("RGB", (1080, 2400), color=(255, 0, 0))
        result = resize_and_pad_image(img, (1024, 768))
        assert result.size == (1024, 768)
        # Check that some padding exists (black pixels at edges)
        left_pixel = result.getpixel((0, 0))
        assert left_pixel == (0, 0, 0)  # Black padding
