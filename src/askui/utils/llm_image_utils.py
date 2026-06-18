"""Image utilities for LLM vision model preprocessing.

Functions for computing optimal image sizes based on patch-based token budgets
and resizing images for VLM consumption.
"""

import logging
import math

from PIL import Image

logger = logging.getLogger(__name__)


def count_image_tokens(width: int, height: int, patch_size: int = 28) -> int:
    """Count the number of tokens an image will consume in a patch-based VLM.

    Each non-overlapping ``patch_size x patch_size`` square maps to one token.

    Args:
        width (int): Image width in pixels.
        height (int): Image height in pixels.
        patch_size (int): Side length of a single patch in pixels.

    Returns:
        int: Number of image tokens.
    """
    patches_w = math.ceil(width / patch_size)
    patches_h = math.ceil(height / patch_size)
    return patches_w * patches_h


def compute_patch_optimized_size(
    width: int,
    height: int,
    max_edge: int = 1568,
    max_tokens: int = 1568,
    patch_size: int = 28,
) -> tuple[int, int]:
    """Compute the largest aspect-preserving size within a patch-based token budget.

    Uses binary search to find the biggest scale factor such that:
    - Neither dimension exceeds ``max_edge``.
    - ``count_image_tokens(w, h, patch_size) <= max_tokens``.

    Args:
        width (int): Original image width.
        height (int): Original image height.
        max_edge (int): Maximum allowed dimension (width or height).
        max_tokens (int): Maximum allowed number of image tokens.
        patch_size (int): Patch size used by the model.

    Returns:
        tuple[int, int]: Target ``(width, height)``.
    """
    if width <= 0 or height <= 0:
        error_msg = f"Image dimensions must be positive, got {width}x{height}"
        raise ValueError(error_msg)

    # If already within all constraints, return as-is
    if (
        width <= max_edge
        and height <= max_edge
        and count_image_tokens(width, height, patch_size) <= max_tokens
    ):
        return width, height

    # Clamp to max_edge first
    scale = min(max_edge / width, max_edge / height, 1.0)

    # Binary search for largest scale that fits within token budget
    lo, hi = 0.0, scale
    for _ in range(50):
        mid = (lo + hi) / 2
        w = max(1, int(width * mid))
        h = max(1, int(height * mid))
        if count_image_tokens(w, h, patch_size) <= max_tokens:
            lo = mid
        else:
            hi = mid

    result_w = max(1, int(width * lo))
    result_h = max(1, int(height * lo))
    return result_w, result_h


def compute_contained_size(
    width: int,
    height: int,
    max_width: int = 1024,
    max_height: int = 768,
) -> tuple[int, int]:
    """Compute the largest aspect-preserving size contained within max bounds.

    If the image already fits, returns its original dimensions.

    Args:
        width (int): Original image width.
        height (int): Original image height.
        max_width (int): Maximum allowed width.
        max_height (int): Maximum allowed height.

    Returns:
        tuple[int, int]: Target ``(width, height)``.
    """
    if width <= 0 or height <= 0:
        error_msg = f"Image dimensions must be positive, got {width}x{height}"
        raise ValueError(error_msg)

    if width <= max_width and height <= max_height:
        return width, height

    scale = min(max_width / width, max_height / height)
    return max(1, int(width * scale)), max(1, int(height * scale))


def resize_image(image: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Resize an image to exact ``target_size`` using LANCZOS resampling.

    Logs a warning if the aspect ratio changes by more than 1%.

    Args:
        image (Image.Image): Source image.
        target_size (tuple[int, int]): Target ``(width, height)``.

    Returns:
        Image.Image: Resized image.
    """
    if image.size == target_size:
        return image

    src_ratio = image.width / image.height
    dst_ratio = target_size[0] / target_size[1]
    if abs(src_ratio - dst_ratio) / max(src_ratio, dst_ratio) > 0.01:
        logger.warning(
            "Aspect ratio change during resize: %.3f -> %.3f",
            src_ratio,
            dst_ratio,
        )

    return image.resize(target_size, Image.Resampling.LANCZOS)


def compute_patch_optimized_image(
    image: Image.Image,
    max_edge: int = 1568,
    max_tokens: int = 1568,
    patch_size: int = 28,
) -> Image.Image:
    """Resize an image to its patch-optimized size.

    Convenience wrapper that combines `compute_patch_optimized_size` and
    `resize_image` into a single call.

    Args:
        image (Image.Image): Source image.
        max_edge (int): Maximum allowed dimension (width or height).
        max_tokens (int): Maximum allowed number of image tokens.
        patch_size (int): Patch size used by the model.

    Returns:
        Image.Image: Resized image.
    """
    target = compute_patch_optimized_size(
        image.width,
        image.height,
        max_edge=max_edge,
        max_tokens=max_tokens,
        patch_size=patch_size,
    )
    return resize_image(image, target)


def downscale_image(
    image: Image.Image,
    max_dimension: int = 2000,
) -> Image.Image:
    """Downscale an image so its longest side does not exceed `max_dimension`.

    Convenience wrapper around `compute_contained_size()` and `resize_image()`.
    Unlike ``scale_image_to_fit()`` from `askui.utils.image_utils`, this does
    **not** add black padding — the output keeps its natural dimensions.

    Preserves the original aspect ratio. Images that are already
    within the limit are returned unchanged.

    Args:
        image (Image.Image): The PIL Image to downscale.
        max_dimension (int, optional): Maximum allowed size for the longest side.
            Defaults to `2000`.

    Returns:
        Image.Image: The downscaled image, or the original if no scaling was needed.
    """
    target = compute_contained_size(
        image.width, image.height, max_width=max_dimension, max_height=max_dimension
    )
    return resize_image(image, target)


def resize_and_pad_image(
    image: Image.Image,
    target_size: tuple[int, int],
) -> Image.Image:
    """Resize preserving aspect ratio, then center on a padded canvas.

    Equivalent to the legacy ``scale_image_to_fit`` behaviour.

    Args:
        image (Image.Image): Source image.
        target_size (tuple[int, int]): Canvas ``(width, height)``.

    Returns:
        Image.Image: Image centered on a ``target_size`` canvas.
    """
    from askui.utils.image_utils import scale_image_to_fit

    return scale_image_to_fit(image, target_size)
