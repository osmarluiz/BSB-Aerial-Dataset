"""Define where to extract tiles: point shapefiles or sliding window."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterator

import geopandas as gpd
from rasterio.crs import CRS
from rasterio.transform import Affine, rowcol

from core.models import TileLocation
from core.utils import ensure_same_crs

logger = logging.getLogger(__name__)


class PointSampler:
    """Sample tiles centered on point shapefile locations."""

    def __init__(
        self,
        shapefiles: dict[str, Path],
        tile_height: int,
        tile_width: int,
        image_transform: Affine,
        image_crs: CRS,
        image_height: int,
        image_width: int,
    ) -> None:
        if "train" not in shapefiles:
            raise ValueError("At least 'train' shapefile is required")

        self._shapefiles = shapefiles
        self._tile_h = tile_height
        self._tile_w = tile_width
        self._transform = image_transform
        self._image_crs = image_crs
        self._img_h = image_height
        self._img_w = image_width

    def generate(self) -> Iterator[TileLocation]:
        tile_id = 1
        half_h = self._tile_h // 2
        half_w = self._tile_w // 2

        for split, shp_path in self._shapefiles.items():
            gdf = gpd.read_file(shp_path)
            gdf = ensure_same_crs(gdf, self._image_crs, f"{split} point shapefile")

            # Detect and warn about duplicate points
            coords = [(geom.x, geom.y) for geom in gdf.geometry]
            seen: set[tuple[int, int]] = set()
            duplicates = 0

            for geom in gdf.geometry:
                x, y = geom.x, geom.y
                r, c = rowcol(self._transform, x, y)
                r, c = int(r), int(c)

                pixel_key = (r, c)
                if pixel_key in seen:
                    duplicates += 1
                    continue
                seen.add(pixel_key)

                # Compute top-left of tile centered on this point
                top = r - half_h
                left = c - half_w

                # Check bounds
                if top < 0 or left < 0:
                    logger.warning(
                        "Point at (%s, %s) too close to image edge, skipping", x, y
                    )
                    continue
                if top + self._tile_h > self._img_h or left + self._tile_w > self._img_w:
                    logger.warning(
                        "Point at (%s, %s) too close to image edge, skipping", x, y
                    )
                    continue

                yield TileLocation(
                    id=tile_id,
                    row=top,
                    col=left,
                    height=self._tile_h,
                    width=self._tile_w,
                    split=split,
                )
                tile_id += 1

            if duplicates > 0:
                logger.warning(
                    "Skipped %d duplicate points in %s shapefile",
                    duplicates,
                    split,
                )


class SlidingWindowSampler:
    """Sample tiles with a sliding window over the entire image."""

    def __init__(
        self,
        tile_height: int,
        tile_width: int,
        stride_y: int,
        stride_x: int,
        image_height: int,
        image_width: int,
        split_ratios: dict[str, float] | None = None,
        seed: int = 42,
    ) -> None:
        self._tile_h = tile_height
        self._tile_w = tile_width
        self._stride_y = stride_y
        self._stride_x = stride_x
        self._img_h = image_height
        self._img_w = image_width
        self._split_ratios = split_ratios
        self._seed = seed

    @property
    def estimated_count(self) -> int:
        """Compute tile count mathematically (without generating)."""
        rows = max(0, (self._img_h - self._tile_h) // self._stride_y + 1)
        cols = max(0, (self._img_w - self._tile_w) // self._stride_x + 1)
        return rows * cols

    def generate(self) -> Iterator[TileLocation]:
        # Collect all valid positions
        positions: list[tuple[int, int]] = []
        for row in range(0, self._img_h - self._tile_h + 1, self._stride_y):
            for col in range(0, self._img_w - self._tile_w + 1, self._stride_x):
                positions.append((row, col))

        if not positions:
            logger.warning("No valid tile positions found")
            return

        # Assign splits
        splits = self._assign_splits(positions)

        # Re-sort by position for spatial locality (better I/O performance)
        indexed = list(zip(positions, splits))
        indexed.sort(key=lambda x: x[0])

        for tile_id, ((row, col), split) in enumerate(indexed, start=1):
            yield TileLocation(
                id=tile_id,
                row=row,
                col=col,
                height=self._tile_h,
                width=self._tile_w,
                split=split,
            )

    def _assign_splits(
        self, positions: list[tuple[int, int]]
    ) -> list[str]:
        """Assign splits based on ratios with seeded shuffle."""
        n = len(positions)

        if self._split_ratios is None:
            return ["train"] * n

        # Shuffle indices for random assignment
        rng = random.Random(self._seed)
        indices = list(range(n))
        rng.shuffle(indices)

        splits = [""] * n
        offset = 0
        sorted_keys = sorted(self._split_ratios.keys())

        for i, key in enumerate(sorted_keys):
            if i == len(sorted_keys) - 1:
                count = n - offset  # Last split gets the remainder
            else:
                count = int(n * self._split_ratios[key])

            for j in range(offset, min(offset + count, n)):
                splits[indices[j]] = key
            offset += count

        return splits
