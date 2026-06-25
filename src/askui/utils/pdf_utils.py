import base64
from io import BufferedReader, BytesIO
from pathlib import Path

from pydantic import ConfigDict, RootModel


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

    def to_bytes(self) -> bytes:
        """Read the PDF source into bytes.

        Returns:
            bytes: The PDF as bytes.
        """
        with self.reader as reader:
            return reader.read()

    def to_base64(self) -> str:
        """Convert the PDF to a base64 string.

        Returns:
            str: A base64 encoded string of the PDF.
        """
        return base64.b64encode(self.to_bytes()).decode("utf-8")

    def to_data_url(self) -> str:
        """Convert the PDF to a data URL.

        Returns:
            str: A data URL string in the format
                `"data:application/pdf;base64,..."`.
        """
        return f"data:application/pdf;base64,{self.to_base64()}"


__all__ = [
    "PdfSource",
]
