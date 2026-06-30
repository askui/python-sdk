import base64
from io import BufferedReader, BytesIO
from pathlib import Path

from pydantic import ConfigDict, RootModel

# Anthropic and OpenAI both reject PDFs larger than 32 MB. We guard against this
# at the source so the caller gets a clear error instead of an opaque provider
# 400/413 once the (base64-inflated) request is sent.
# See https://docs.anthropic.com/en/docs/build-with-claude/pdf-support
MAX_PDF_SIZE_BYTES = 32 * 1024 * 1024

# Fallback file name used when a PDF has no associated path (e.g. loaded from
# raw bytes), since some providers require a file name for document parts.
DEFAULT_PDF_FILENAME = "document.pdf"


class PdfTooLargeError(ValueError):
    """Raised when a PDF exceeds the maximum size supported by the model."""

    def __init__(self, size_bytes: int, max_size_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_size_bytes = max_size_bytes
        super().__init__(
            f"PDF is {size_bytes} bytes, which exceeds the maximum supported "
            f"size of {max_size_bytes} bytes (~{max_size_bytes // (1024 * 1024)} "
            "MB). Reduce the file size or split the document."
        )


class PdfSource(RootModel):
    """A class that represents a PDF source.
    It provides methods to convert it to different formats.

    The class can be initialized with:
    - A file path (str or pathlib.Path)

    Attributes:
        root (bytes): The underlying PDF bytes.

    Args:
        root (Pdf): The PDF source to load from.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    root: bytes | Path

    @property
    def reader(self) -> BufferedReader | BytesIO:
        if isinstance(self.root, Path):
            return self.root.open("rb")
        return BytesIO(self.root)

    @property
    def filename(self) -> str | None:
        """The file name of the PDF when loaded from a path, otherwise `None`."""
        if isinstance(self.root, Path):
            return self.root.name
        return None

    @property
    def size_bytes(self) -> int:
        """The size of the PDF in bytes (without reading it fully into memory)."""
        if isinstance(self.root, Path):
            return self.root.stat().st_size
        return len(self.root)

    def validate_size(self, max_size_bytes: int = MAX_PDF_SIZE_BYTES) -> None:
        """Raise `PdfTooLargeError` if the PDF exceeds `max_size_bytes`.

        Args:
            max_size_bytes (int, optional): The maximum allowed size in bytes.
                Defaults to `MAX_PDF_SIZE_BYTES`.

        Raises:
            PdfTooLargeError: If the PDF is larger than `max_size_bytes`.
        """
        size = self.size_bytes
        if size > max_size_bytes:
            raise PdfTooLargeError(size, max_size_bytes)

    def to_bytes(self) -> bytes:
        """Read the PDF source into bytes.

        Returns:
            bytes: The PDF as bytes.

        Raises:
            PdfTooLargeError: If the PDF exceeds `MAX_PDF_SIZE_BYTES`.
        """
        self.validate_size()
        with self.reader as reader:
            return reader.read()

    def to_base64(self) -> str:
        """Convert the PDF to a base64 string.

        Returns:
            str: A base64 encoded string of the PDF.

        Raises:
            PdfTooLargeError: If the PDF exceeds `MAX_PDF_SIZE_BYTES`.
        """
        return base64.b64encode(self.to_bytes()).decode("utf-8")

    def to_data_url(self) -> str:
        """Convert the PDF to a data URL.

        Returns:
            str: A data URL string in the format
                `"data:application/pdf;base64,..."`.

        Raises:
            PdfTooLargeError: If the PDF exceeds `MAX_PDF_SIZE_BYTES`.
        """
        return f"data:application/pdf;base64,{self.to_base64()}"


__all__ = [
    "DEFAULT_PDF_FILENAME",
    "MAX_PDF_SIZE_BYTES",
    "PdfSource",
    "PdfTooLargeError",
]
