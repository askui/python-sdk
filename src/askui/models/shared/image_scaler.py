"""Image scaler types used by VLM providers."""

from abc import ABC, abstractmethod

from PIL import Image

from askui.utils.llm_image_utils import (
    compute_contained_size,
    compute_patch_optimized_image,
    resize_image,
)


class ImageScaler(ABC):
    """Base class for image scalers used by VLM providers.

    Subclasses implement ``__call__`` to preprocess a screenshot
    before it is sent to a model.
    """

    @abstractmethod
    def __call__(self, image: Image.Image) -> Image.Image:
        """Scale ``image`` for model consumption."""


class PatchOptimizedImageScaler(ImageScaler):
    """Image scaler that fits images within a patch-based token budget.

    Uses `compute_patch_optimized_image()` under the hood: the image
    is aspect-preserving scaled so that neither dimension exceeds
    ``max_edge`` and the total patch count stays within ``max_tokens``.

    Args:
        max_edge (int): Maximum allowed dimension (width or height).
        max_tokens (int): Maximum allowed number of image tokens.
        patch_size (int): Side length of a single patch in pixels.
    """

    def __init__(
        self,
        max_edge: int = 1568,
        max_tokens: int = 1568,
        patch_size: int = 28,
    ) -> None:
        self._max_edge = max_edge
        self._max_tokens = max_tokens
        self._patch_size = patch_size

    def __call__(self, image: Image.Image) -> Image.Image:
        """Scale ``image`` to fit within the configured token budget."""
        return compute_patch_optimized_image(
            image,
            max_edge=self._max_edge,
            max_tokens=self._max_tokens,
            patch_size=self._patch_size,
        )


class ContainedImageScaler(ImageScaler):
    """Image scaler that fits images within maximum width/height bounds.

    Preserves aspect ratio. Images already within the bounds are
    returned unchanged.

    Args:
        max_width (int): Maximum allowed width.
        max_height (int): Maximum allowed height.
    """

    def __init__(
        self,
        max_width: int = 1024,
        max_height: int = 768,
    ) -> None:
        self._max_width = max_width
        self._max_height = max_height

    def __call__(self, image: Image.Image) -> Image.Image:
        """Scale ``image`` to fit within the configured bounds."""
        target = compute_contained_size(
            image.width,
            image.height,
            max_width=self._max_width,
            max_height=self._max_height,
        )
        return resize_image(image, target)
