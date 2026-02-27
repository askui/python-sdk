"""Visual validation utilities for cache execution.

This module provides utilities for visual validation of cached trajectories:
- Image hashing functions (perceptual hash, average hash)
- Hamming distance computation
- Region extraction from images
- Screenshot extraction from message history
"""

import logging
from typing import TYPE_CHECKING, Any

import imagehash
from PIL import Image

if TYPE_CHECKING:
    from askui.models.shared.agent_message_param import MessageParam

logger = logging.getLogger(__name__)


def compute_phash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute perceptual hash (pHash) of an image.

    Uses DCT-based perceptual hashing which is robust to scaling, aspect ratio
    changes, and minor modifications.

    Args:
        image: PIL Image to hash
        hash_size: Size of the hash (default: 8, produces 64-bit hash)

    Returns:
        String representation of the hash (hex format)
    """
    phash = imagehash.phash(image, hash_size=hash_size)
    return str(phash)


def compute_ahash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute average hash (aHash) of an image.

    Average hash is faster but less robust than perceptual hash.
    Good for detecting exact duplicates or very similar images.

    Args:
        image: PIL Image to hash
        hash_size: Size of the hash (default: 8, produces 64-bit hash)

    Returns:
        String representation of the hash (hex format)
    """
    ahash = imagehash.average_hash(image, hash_size=hash_size)
    return str(ahash)


def compute_hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two image hashes.

    The Hamming distance is the number of bit positions in which the two
    hashes differ. A distance of 0 means the images are identical (or very
    similar). Larger distances indicate more visual difference.

    Typical thresholds:
    - 0-5: Nearly identical images
    - 6-10: Similar images with minor differences
    - 11+: Different images

    Args:
        hash1: First hash string (hex format)
        hash2: Second hash string (hex format)

    Returns:
        Hamming distance (number of differing bits)

    Raises:
        ValueError: If hashes have different lengths
    """
    if len(hash1) != len(hash2):
        msg = f"Hashes must have same length. Got {len(hash1)} and {len(hash2)}"
        raise ValueError(msg)

    # Convert hex strings to imagehash objects
    ihash1 = imagehash.hex_to_hash(hash1)
    ihash2 = imagehash.hex_to_hash(hash2)

    # Compute Hamming distance
    return ihash1 - ihash2


def extract_region(
    image: Image.Image,
    action_input: dict[str, Any],
    region_size: int = 50,
) -> Image.Image:
    """Extract a square region around an action coordinate.

    Extracts a square region centered on the coordinate specified in the
    action input. Handles edge cases where the region would extend beyond
    image boundaries by clipping to valid bounds.

    Args:
        image: PIL Image to extract region from
        action_input: Action input dict containing 'coordinate' key with [x, y]
        region_size: Size of the square region to extract (default: 50 pixels)

    Returns:
        Extracted region as PIL Image (may be smaller than region_size if
        near image edges)
    """
    coordinate = action_input.get("coordinate")
    if not coordinate:
        msg = f"No coordinate found in action_input: {action_input}"
        logger.warning(msg)
        return image

    x, y = coordinate
    width, height = image.size

    # Calculate region bounds (centered on coordinate) and clip to valid bounds
    half_size = region_size // 2
    left = max(0, x - half_size)
    top = max(0, y - half_size)
    right = min(width, x + half_size)
    bottom = min(height, y + half_size)

    # Handle edge case where coordinates are completely out of bounds
    # In this case, return an empty or minimal region
    if left >= right or top >= bottom:
        # Return minimal 1x1 region from top-left corner
        return image.crop((0, 0, min(1, width), min(1, height)))

    # Extract and return region
    return image.crop((left, top, right, bottom))


def find_recent_screenshot(
    messages: list["MessageParam"],
    from_index: int | None = None,
) -> Image.Image | None:
    """Extract most recent screenshot from message history.

    Looks backwards through message history for the most recent tool result
    containing an image block (screenshot). This is used during both recording
    and validation to extract the "before" state screenshot.

    Args:
        messages: Message history to search through
        from_index: Optional index to start searching backwards from.
                   If None, starts from end of list.

    Returns:
        PIL Image from most recent screenshot, or None if not found
    """
    start_idx = from_index if from_index is not None else len(messages) - 1

    # Look backwards from start index
    for i in range(start_idx, -1, -1):
        message = messages[i]
        if message.role != "user":
            continue

        # Check if message content is a list of blocks
        if isinstance(message.content, str):
            continue

        # Look for tool result blocks with images
        for block in message.content:
            if block.type == "tool_result":
                # Check for image blocks within tool result
                if isinstance(block.content, list):
                    for content_item in block.content:
                        if content_item.type == "image":
                            # Found screenshot - decode and return
                            from askui.utils.image_utils import base64_to_image

                            # Only base64 images have data attribute
                            if hasattr(content_item.source, "data"):
                                return base64_to_image(content_item.source.data)

    return None


def get_validation_coordinate(tool_input: dict[str, Any]) -> tuple[int, int] | None:
    """Extract the coordinate for visual validation from tool input.

    Args:
        tool_input: Tool input dictionary

    Returns:
        (x, y) coordinate tuple or None if not applicable

    For click actions, returns the click coordinate.
    For type actions, returns the coordinate of the text input field.
    """

    def try_pair(x_val: Any, y_val: Any) -> tuple[int, int] | None:
        x = _safe_int(x_val)
        y = _safe_int(y_val)
        if x is None or y is None:
            return None
        return (x, y)

    if "coordinate" in tool_input:
        coord = tool_input["coordinate"]
        if isinstance(coord, list) and len(coord) == 2:
            result = try_pair(coord[0], coord[1])
            if result is not None:
                return result

    if "x" in tool_input and "y" in tool_input:
        result = try_pair(tool_input["x"], tool_input["y"])
        if result is not None:
            return result

    if "x1" in tool_input and "y1" in tool_input:
        result = try_pair(tool_input["x1"], tool_input["y1"])
        if result is not None:
            return result

    return None


def _safe_int(value: Any) -> int | None:
    """Try converting value to int, return None if not possible."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
