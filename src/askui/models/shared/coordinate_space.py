from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


def _common_prompt_lines() -> list[str]:
    return ["* Coordinate origin is the top-left corner (0, 0)"]


class VlmCoordinateSpace(BaseModel, ABC):
    """Abstract base for VLM coordinate conventions.

    Each subclass describes one coordinate grid a VLM may emit and knows
    how to map those coordinates back to pixel space and how to render
    the matching prompt section.
    """

    @property
    def maps_to_screenshot_pixels(self) -> bool:
        """Whether model coordinates are absolute pixels in the screenshot image.

        When ``True``, coordinates need padding-aware inverse scaling
        (screenshot space to device space).  When ``False``, coordinates
        are in a normalised grid and map directly to device resolution.
        """
        return False

    @abstractmethod
    def map_to_target(
        self, x: float, y: float, target_resolution: tuple[int, int]
    ) -> tuple[int, int]:
        """Map model coordinates to pixel coordinates in *target_resolution*."""

    @abstractmethod
    def build_prompt_section(self) -> str:
        """Build prompt text describing coordinate bounds for the model."""


class PixelCoordinateSpace(VlmCoordinateSpace):
    """Identity mapping -- coordinates already in pixel space.

    Used by Anthropic/Claude which emit coordinates matching the
    screenshot resolution.
    """

    @property
    def maps_to_screenshot_pixels(self) -> bool:
        return True

    def map_to_target(
        self,
        x: float,
        y: float,
        target_resolution: tuple[int, int],  # noqa: ARG002
    ) -> tuple[int, int]:
        return int(x), int(y)

    def build_prompt_section(self) -> str:
        lines = _common_prompt_lines()
        lines.append(
            "* Coordinates are in pixel space matching the screenshot dimensions"
        )
        return "\n".join(lines)


class ScaledCoordinateSpace(VlmCoordinateSpace):
    """Integer grid (e.g. 1000x1000 for Qwen). Linear scaling."""

    width: int = Field(gt=0, description="Width of the coordinate grid")
    height: int = Field(gt=0, description="Height of the coordinate grid")

    def map_to_target(
        self, x: float, y: float, target_resolution: tuple[int, int]
    ) -> tuple[int, int]:
        tw, th = target_resolution
        return int(x * tw / self.width), int(y * th / self.height)

    def build_prompt_section(self) -> str:
        lines = _common_prompt_lines()
        lines.append(
            f"* Emit coordinates in a {self.width}x{self.height} "
            f"normalised grid: 0 <= x < {self.width}, "
            f"0 <= y < {self.height}"
        )
        return "\n".join(lines)


class NormalizedCoordinateSpace(VlmCoordinateSpace):
    """0.0-1.0 float grid (Kimi). No fields."""

    def map_to_target(
        self, x: float, y: float, target_resolution: tuple[int, int]
    ) -> tuple[int, int]:
        tw, th = target_resolution
        return int(x * tw), int(y * th)

    def build_prompt_section(self) -> str:
        lines = _common_prompt_lines()
        lines.append(
            "* Emit coordinates as normalised floats: 0.0 <= x <= 1.0, 0.0 <= y <= 1.0"
        )
        return "\n".join(lines)
