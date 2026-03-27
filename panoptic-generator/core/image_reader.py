"""Read georeferenced raster images (GeoTIFF, ENVI)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from rasterio.transform import Affine, rowcol, xy


class ImageReader:
    """Wraps rasterio to provide a clean interface for reading georeferenced rasters."""

    def __init__(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        try:
            self._dataset = rasterio.open(path)
        except rasterio.errors.RasterioIOError as e:
            raise ValueError(f"Not a valid raster file: {path}") from e
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    @property
    def width(self) -> int:
        return self._dataset.width

    @property
    def height(self) -> int:
        return self._dataset.height

    @property
    def band_count(self) -> int:
        return self._dataset.count

    @property
    def band_names(self) -> list[str]:
        descriptions = self._dataset.descriptions
        return [
            desc if desc else f"Band {i + 1}"
            for i, desc in enumerate(descriptions)
        ]

    @property
    def crs(self) -> CRS | None:
        return self._dataset.crs

    @property
    def transform(self) -> Affine:
        return self._dataset.transform

    @property
    def bounds(self) -> BoundingBox:
        return self._dataset.bounds

    @property
    def pixel_size(self) -> tuple[float, float]:
        t = self.transform
        return (abs(t.a), abs(t.e))

    @property
    def nodata(self) -> float | None:
        return self._dataset.nodata

    def read_window(
        self,
        row: int,
        col: int,
        height: int,
        width: int,
        bands: list[int] | None = None,
    ) -> np.ndarray:
        """
        Read a rectangular window. Never reads the full image.

        Args:
            row, col: top-left pixel.
            height, width: window size in pixels.
            bands: 1-based band indices. None = all bands.

        Returns:
            np.ndarray with shape (bands, height, width), dtype float32.
        """
        if row < 0 or col < 0:
            raise ValueError(f"Window origin must be non-negative, got ({row}, {col})")
        if row + height > self.height or col + width > self.width:
            raise ValueError(
                f"Window ({row}, {col}, {height}, {width}) exceeds image bounds "
                f"({self.height}, {self.width})"
            )

        window = rasterio.windows.Window(col, row, width, height)
        indexes = bands if bands else None
        return self._dataset.read(indexes=indexes, window=window).astype(np.float32)

    def read_overview(
        self,
        max_size: int = 2000,
        bands: list[int] | None = None,
    ) -> np.ndarray:
        """
        Read a downsampled version for preview/thumbnail.

        Args:
            max_size: max dimension (height or width) of output.
            bands: 1-based band indices. None = first 3 (or all if < 3).

        Returns:
            np.ndarray with shape (bands, out_height, out_width), dtype float32.
        """
        if bands is None:
            bands = list(range(1, min(self.band_count, 3) + 1))

        factor = max(self.height, self.width) / max_size
        if factor <= 1.0:
            return self._dataset.read(indexes=bands).astype(np.float32)

        out_h = max(1, int(self.height / factor))
        out_w = max(1, int(self.width / factor))
        return self._dataset.read(
            indexes=bands, out_shape=(len(bands), out_h, out_w)
        ).astype(np.float32)

    def geo_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """Geographic coordinates → pixel (row, col)."""
        row, col = rowcol(self.transform, x, y)
        return (int(row), int(col))

    def pixel_to_geo(self, row: int, col: int) -> tuple[float, float]:
        """Pixel (row, col) → geographic coordinates (center of pixel)."""
        x, y = xy(self.transform, row, col)
        return (x, y)

    def close(self) -> None:
        """Close the underlying rasterio dataset."""
        self._dataset.close()

    def __enter__(self) -> ImageReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
