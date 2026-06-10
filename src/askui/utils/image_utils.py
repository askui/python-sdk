import base64
import binascii
import io
import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

from PIL import Image, ImageDraw, UnidentifiedImageError
from PIL import Image as PILImage
from pydantic import ConfigDict, RootModel


def image_to_data_url(image: PILImage.Image) -> str:
    """Convert a PIL Image to a data URL.

    Args:
        image (PILImage.Image): The PIL Image to convert.

    Returns:
        str: A data URL string in the format "data:image/png;base64,..."
    """
    return f"data:image/png;base64,{image_to_base64(image=image, format_='PNG')}"


def base64_to_image(base64_string: str) -> Image.Image:
    """Convert a base64 string to a PIL Image.

    Args:
        base64_string (str): The base64 encoded image string.

    Returns:
        Image.Image: A PIL Image object.

    Raises:
        ValueError: If the base64 string is invalid or the image cannot be decoded.
    """
    try:
        image_bytes = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_bytes))
    except (binascii.Error, UnidentifiedImageError) as e:
        error_msg = f"Could not convert base64 string to image: {e}"
        raise ValueError(error_msg) from e


def data_url_to_image(data_url: str) -> Image.Image:
    """Convert a data URL to a PIL Image.

    Args:
        data_url (str): The data URL string to convert.

    Returns:
        Image.Image: A PIL Image object.

    Raises:
        ValueError: If the data URL is invalid or the data URL data cannot be decoded
            or the image cannot be decoded.
    """
    try:
        data_url_data = data_url.split(",")[1]
        while len(data_url_data) % 4 != 0:
            data_url_data += "="
        return base64_to_image(data_url_data)
    except (IndexError, ValueError) as e:
        error_msg = f"Could not convert data URL to image: {e}"
        raise ValueError(error_msg) from e


def draw_point_on_image(
    image: Image.Image, x: int, y: int, size: int = 3
) -> Image.Image:
    """Draw a red point at the specified x,y coordinates on a copy of the input image.

    Args:
        image (Image.Image): The PIL Image to draw on.
        x (int): The x-coordinate for the point.
        y (int): The y-coordinate for the point.
        size (int, optional): The size of the point in pixels. Defaults to `3`.

    Returns:
        Image.Image: A new PIL Image with the point drawn.
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.ellipse([x - size, y - size, x + size, y + size], fill="red")
    return img_copy


def image_to_base64(
    image: Union[pathlib.Path, Image.Image], format_: Literal["PNG", "JPEG"] = "PNG"
) -> str:
    """Convert an image to a base64 string.

    Args:
        image (Union[pathlib.Path, Image.Image]): The image to convert, either a PIL Image or a file path.
        format_ (Literal["PNG", "JPEG"], optional): The image format to use. Defaults to `"PNG"`.

    Returns:
        str: A base64 encoded string of the image.

    Raises:
        ValueError: If the image cannot be encoded or the format is unsupported.
    """
    image_bytes: bytes | None = None
    if isinstance(image, Image.Image):
        with io.BytesIO() as buffer:
            image.save(buffer, format=format_)
            image_bytes = buffer.getvalue()
    else:
        with Path.open(image, "rb") as file:
            image_bytes = file.read()

    return base64.b64encode(image_bytes).decode("utf-8")


def _calc_center_offset(
    image_size: tuple[int, int],
    container_size: tuple[int, int],
) -> tuple[int, int]:
    """Calculate the offset to center the image in the container.

    If the image is larger than the container, the offset will be negative.

    Args:
        image_size (tuple[int, int]): The size of the image to center (width, height).
        container_size (tuple[int, int]): The size of the container to center the image in (width, height).

    Returns:
        tuple[int, int]: The offset to center the image in the container.
    """
    return (
        (container_size[0] - image_size[0]) // 2,
        (container_size[1] - image_size[1]) // 2,
    )


@dataclass
class ScalingResults:
    """Results of scaling calculations.

    Args:
        factor (float): The scaling factor applied.
        size (tuple[int, int]): The resulting size (width, height).
    """

    factor: float
    size: tuple[int, int]


def _calculate_scaling_for_fit(
    original_size: tuple[int, int],
    target_size: tuple[int, int],
) -> ScalingResults:
    """Calculate the scaling factor and size of an image to fit within target size while maintaining aspect ratio.

    If the image is larger than the target size, the scaling factor will be less than 1.

    Args:
        original_size (tuple[int, int]): The size of the original image (width, height).
        target_size (tuple[int, int]): The target size to fit the image into (width, height).

    Returns:
        ScalingResults: The scaling factor and resulting size.

    Raises:
        ValueError: If the original size or target size is not positive.
    """
    if original_size[0] <= 0 or original_size[1] <= 0:
        error_msg = f"Size must have positive width and height: {original_size}"
        raise ValueError(error_msg)

    if target_size[0] <= 0 or target_size[1] <= 0:
        error_msg = f"Target size must have positive width and height: {target_size}"
        raise ValueError(error_msg)

    aspect_ratio = original_size[0] / original_size[1]
    target_aspect_ratio = target_size[0] / target_size[1]
    if target_aspect_ratio > aspect_ratio:
        factor = target_size[1] / original_size[1]
        width = max(1, int(original_size[0] * factor))  # Ensure minimum width of 1
        height = target_size[1]
    else:
        factor = target_size[0] / original_size[0]
        width = target_size[0]
        height = max(1, int(original_size[1] * factor))  # Ensure minimum height of 1
    return ScalingResults(factor=factor, size=(width, height))


def _center_image_in_background(
    image: Image.Image,
    background_size: tuple[int, int],
    background_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Center an image in a background image.

    Args:
        image (Image.Image): The image to center.
        background_size (tuple[int, int]): The size of the background (width, height).
        background_color (tuple[int, int, int], optional): The background color. Defaults to `(0, 0, 0)`.

    Returns:
        Image.Image: A new image with the input image centered on the background.
    """
    background = Image.new("RGB", background_size, background_color)
    offset = _calc_center_offset(image.size, background_size)
    background.paste(image, offset)
    return background


def scale_image_to_fit(
    image: Image.Image,
    target_size: tuple[int, int],
) -> Image.Image:
    """Scale an image to fit within specified size while maintaining aspect ratio.

    Use black padding to fill the remaining space.

    Args:
        image (Image.Image): The PIL Image to scale.
        target_size (tuple[int, int]): The target size to fit the image into (width, height).

    Returns:
        Image.Image: A new PIL Image that fits within the specified size.
    """
    scaling_results = _calculate_scaling_for_fit(image.size, target_size)
    scaled_image = image.resize(scaling_results.size, Image.Resampling.LANCZOS)
    return _center_image_in_background(scaled_image, target_size)


def downscale_image(
    image: Image.Image,
    max_dimension: int = 2000,
) -> Image.Image:
    """Downscale an image so its longest side does not exceed `max_dimension`.

    Preserves the original aspect ratio. Images that are already
    within the limit are returned unchanged.

    Args:
        image (Image.Image): The PIL Image to downscale.
        max_dimension (int, optional): Maximum allowed size for the longest side. Defaults to `2000`.

    Returns:
        Image.Image: The downscaled image, or the original if no scaling was needed.
    """
    longest_side = max(image.width, image.height)
    if longest_side <= max_dimension:
        return image
    scale_factor = max_dimension / longest_side
    new_size = (
        int(image.width * scale_factor),
        int(image.height * scale_factor),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _scale_coordinates(
    coordinates: tuple[int, int],
    offset: tuple[int, int],
    factor: float,
    inverse: bool,
) -> tuple[int, int]:
    """Scale coordinates based on scaling factor and offset.

    Args:
        coordinates (tuple[int, int]): The coordinates to scale.
        offset (tuple[int, int]): The offset to apply.
        factor (float): The scaling factor.
        inverse (bool): Whether to apply inverse scaling.

    Returns:
        tuple[int, int]: The scaled coordinates.
    """
    if inverse:
        result = (
            (coordinates[0] - offset[0]) / factor,
            (coordinates[1] - offset[1]) / factor,
        )
    else:
        result = (
            (coordinates[0]) * factor + offset[0],
            (coordinates[1]) * factor + offset[1],
        )
    return (int(result[0]), int(result[1]))


def _check_coordinates_in_bounds(
    coordinates: tuple[float, float],
    bounds: tuple[int, int],
) -> None:
    """Check if coordinates are within bounds.

    Args:
        coordinates (tuple[float, float]): The coordinates to check.
        bounds (tuple[int, int]): The bounds (width, height).

    Raises:
        ValueError: If coordinates are out of bounds.
    """
    if (
        coordinates[0] < 0
        or coordinates[1] < 0
        or coordinates[0] > bounds[0]
        or coordinates[1] > bounds[1]
    ):
        error_msg = f"Coordinates {coordinates[0]}, {coordinates[1]} are out of bounds"
        raise ValueError(error_msg)


def scale_coordinates(
    coordinates: tuple[int, int],
    original_size: tuple[int, int],
    target_size: tuple[int, int],
    inverse: bool = False,
    check_coordinates_in_bounds: bool = True,
) -> tuple[int, int]:
    """Scale coordinates between original and scaled image sizes.

    Args:
        coordinates (tuple[int, int]): The coordinates to scale.
        original_size (tuple[int, int]): The original image size (width, height).
        target_size (tuple[int, int]): The target size (width, height).
        inverse (bool, optional): Whether to scale from target to original. Defaults to `False`.
        check_coordinates_in_bounds (bool, optional): Whether to check if the scaled coordinates are in bounds. Defaults to `True`.

    Returns:
        tuple[int, int]: The scaled coordinates.

    Raises:
        ValueError: If the scaled coordinates are out of bounds.
    """
    scaling_results = _calculate_scaling_for_fit(original_size, target_size)
    offset = _calc_center_offset(scaling_results.size, target_size)
    result = _scale_coordinates(coordinates, offset, scaling_results.factor, inverse)
    if check_coordinates_in_bounds:
        _check_coordinates_in_bounds(result, original_size if inverse else target_size)
    return result


class ImageSource(RootModel):
    """A class that represents an image source and provides methods to convert it to different formats.

    The class can be initialized with:
    - A PIL Image object
    - A file path (str or pathlib.Path)
    - A data URL string

    Attributes:
        root (PILImage.Image): The underlying PIL Image object.

    Args:
        root (Img): The image source to load from.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    root: PILImage.Image

    def to_data_url(self) -> str:
        """Convert the image to a data URL.

        Returns:
            str: A data URL string in the format `"data:image/png;base64,..."`
        """
        return image_to_data_url(image=self.root)

    def to_base64(self) -> str:
        """Convert the image to a base64 string.

        Returns:
            str: A base64 encoded string of the image.
        """
        return image_to_base64(image=self.root)

    def to_bytes(self) -> bytes:
        """Convert the image to bytes.

        Returns:
            bytes: The image as bytes.
        """
        img_byte_arr = io.BytesIO()
        self.root.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()


__all__ = [
    "image_to_data_url",
    "data_url_to_image",
    "draw_point_on_image",
    "base64_to_image",
    "image_to_base64",
    "scale_image_to_fit",
    "scale_coordinates",
    "ScalingResults",
    "ImageSource",
]
