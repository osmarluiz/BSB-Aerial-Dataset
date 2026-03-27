"""Load annotations from shapefile (vector) or pre-rasterized masks (raster)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from rasterio.crs import CRS
from rasterio.transform import Affine
from shapely.geometry import box

from core.models import CategoryInfo, MaskBundle
from core.utils import decode_panoptic_array, ensure_same_crs

logger = logging.getLogger(__name__)


# ── Category Auto-Discovery ─────────────────────────────────────────────────


@dataclass
class DiscoveredCategory:
    """Result of auto-discovering categories from a shapefile."""

    original_value: str | int
    suggested_id: int
    suggested_name: str
    count: int


def discover_categories(
    shapefile_path: Path, class_column: str
) -> list[DiscoveredCategory]:
    """
    Read unique values from class_column and build a suggested mapping.

    Works for both numeric and string columns.
    """
    gdf = gpd.read_file(shapefile_path)
    if class_column not in gdf.columns:
        raise ValueError(
            f"Column '{class_column}' not found. "
            f"Available: {list(gdf.columns)}"
        )

    value_counts = gdf[class_column].value_counts()
    unique_values = sorted(value_counts.index, key=str)

    result = []
    for i, val in enumerate(unique_values):
        is_numeric = isinstance(val, (int, float, np.integer, np.floating))
        suggested_id = int(val) if is_numeric else i + 1
        suggested_name = str(val) if not is_numeric else f"class_{int(val)}"

        result.append(
            DiscoveredCategory(
                original_value=val,
                suggested_id=suggested_id,
                suggested_name=suggested_name,
                count=int(value_counts[val]),
            )
        )

    return result


# ── Vector Annotation Source ─────────────────────────────────────────────────


class VectorAnnotationSource:
    """Annotation from polygon shapefile — rasterizes on the fly."""

    def __init__(
        self,
        shapefile_path: Path,
        class_column: str,
        categories: list[CategoryInfo],
        image_transform: Affine,
        image_crs: CRS,
        image_shape: tuple[int, int],
    ) -> None:
        self._transform = image_transform
        self._image_shape = image_shape
        self._categories = categories
        self._thing_ids = {c.id for c in categories if c.is_thing}

        # Read and validate shapefile
        gdf = gpd.read_file(shapefile_path)
        if class_column not in gdf.columns:
            raise ValueError(
                f"Column '{class_column}' not found in shapefile. "
                f"Available: {list(gdf.columns)}"
            )

        # CRS handling
        if image_crs is None:
            raise ValueError("Image has no CRS defined")
        gdf = ensure_same_crs(gdf, image_crs, "annotation shapefile")

        # Handle string vs numeric class column
        col_dtype = gdf[class_column].dtype
        if col_dtype == object or col_dtype.name == "category":
            # String column → map to numeric IDs
            name_to_id = {c.name: c.id for c in categories}
            unmapped = set(gdf[class_column].unique()) - set(name_to_id.keys())
            if unmapped:
                logger.warning(
                    "Values not found in categories (will be skipped): %s", unmapped
                )
            gdf["_category_id"] = gdf[class_column].map(name_to_id)
            gdf = gdf.dropna(subset=["_category_id"])
            gdf["_category_id"] = gdf["_category_id"].astype(int)
        else:
            # Numeric column
            gdf["_category_id"] = gdf[class_column].astype(int)
            invalid = gdf[gdf["_category_id"] < 1]
            if len(invalid) > 0:
                logger.warning(
                    "Dropping %d features with category_id < 1", len(invalid)
                )
                gdf = gdf[gdf["_category_id"] >= 1]

        # Drop empty/null geometries
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()

        self._gdf = gdf
        self._sindex = gdf.sindex  # R-tree spatial index

    def get_categories(self) -> list[CategoryInfo]:
        return list(self._categories)

    def get_masks(
        self, row: int, col: int, height: int, width: int
    ) -> MaskBundle:
        """
        Single rasterization pass producing both panoptic and semantic masks.

        Uses R-tree spatial index for fast geometry lookup.
        Rasterizes stuff first, then things on top (smaller things last).
        """
        # Tile bounds in geographic coordinates
        tile_transform = rasterio.transform.from_bounds(
            *self._tile_geo_bounds(row, col, height, width),
            width,
            height,
        )
        tile_box = box(*self._tile_geo_bounds(row, col, height, width))

        # Query spatial index for candidates (O(log n))
        candidate_idx = list(self._sindex.query(tile_box))
        if not candidate_idx:
            return MaskBundle(
                panoptic=np.zeros((height, width), dtype=np.int32),
                semantic=np.zeros((height, width), dtype=np.uint8),
            )

        candidates = self._gdf.iloc[candidate_idx].copy()

        # Clip to tile bounds
        candidates = candidates.clip(tile_box)
        candidates = candidates[
            ~candidates.geometry.is_empty & candidates.geometry.notna()
        ]

        if candidates.empty:
            return MaskBundle(
                panoptic=np.zeros((height, width), dtype=np.int32),
                semantic=np.zeros((height, width), dtype=np.uint8),
            )

        # Separate stuff and things
        stuff = candidates[~candidates["_category_id"].isin(self._thing_ids)]
        things = candidates[candidates["_category_id"].isin(self._thing_ids)]

        panoptic = np.zeros((height, width), dtype=np.int32)
        semantic = np.zeros((height, width), dtype=np.uint8)

        # Use tile-local transform for rasterization
        tile_t = self._tile_transform(row, col)

        # Rasterize stuff first (category_id as both panoptic and semantic ID)
        if not stuff.empty:
            shapes_stuff = [
                (geom, int(cat_id))
                for geom, cat_id in zip(stuff.geometry, stuff["_category_id"])
            ]
            rasterio.features.rasterize(
                shapes_stuff, out=semantic, transform=tile_t, dtype="uint8"
            )
            # For stuff, panoptic ID = category_id (one ID per stuff class per tile)
            rasterio.features.rasterize(
                shapes_stuff, out=panoptic, transform=tile_t, dtype="int32"
            )

        # Rasterize things on top, each instance with a unique ID
        # Sort by area descending so smaller objects render on top
        if not things.empty:
            things = things.copy()
            things["_area"] = things.geometry.area
            things = things.sort_values("_area", ascending=False)

            instance_id = max(c.id for c in self._categories) + 1
            for _, feat in things.iterrows():
                geom = feat.geometry
                cat_id = int(feat["_category_id"])

                # Semantic: burn category_id
                rasterio.features.rasterize(
                    [(geom, cat_id)], out=semantic, transform=tile_t, dtype="uint8"
                )
                # Panoptic: burn unique instance_id
                rasterio.features.rasterize(
                    [(geom, instance_id)],
                    out=panoptic,
                    transform=tile_t,
                    dtype="int32",
                )
                instance_id += 1

        return MaskBundle(panoptic=panoptic, semantic=semantic)

    def _tile_geo_bounds(
        self, row: int, col: int, height: int, width: int
    ) -> tuple[float, float, float, float]:
        """Compute geographic bounds (left, bottom, right, top) for a tile window."""
        t = self._transform
        left = t.c + col * t.a
        top = t.f + row * t.e
        right = left + width * t.a
        bottom = top + height * t.e
        return (min(left, right), min(top, bottom), max(left, right), max(top, bottom))

    def _tile_transform(self, row: int, col: int) -> Affine:
        """Compute affine transform for a tile window."""
        t = self._transform
        return Affine(t.a, t.b, t.c + col * t.a, t.d, t.e, t.f + row * t.e)

    def close(self) -> None:
        pass  # GeoDataFrame is in-memory, nothing to close


# ── Raster Annotation Source ─────────────────────────────────────────────────


class RasterAnnotationSource:
    """Annotation from pre-rasterized mask images (legacy mode)."""

    def __init__(
        self,
        semantic_path: Path,
        panoptic_path: Path | None,
        categories: list[CategoryInfo],
    ) -> None:
        self._categories = categories
        self._semantic_ds = rasterio.open(semantic_path)

        self._panoptic_ds = None
        if panoptic_path:
            self._panoptic_ds = rasterio.open(panoptic_path)
            if (
                self._panoptic_ds.height != self._semantic_ds.height
                or self._panoptic_ds.width != self._semantic_ds.width
            ):
                raise ValueError(
                    "Semantic and panoptic rasters must have the same dimensions"
                )

    def get_categories(self) -> list[CategoryInfo]:
        return list(self._categories)

    def get_masks(
        self, row: int, col: int, height: int, width: int
    ) -> MaskBundle:
        window = rasterio.windows.Window(col, row, width, height)

        # Semantic mask: band 1
        semantic = (
            self._semantic_ds.read(1, window=window).astype(np.uint8)
        )

        # Panoptic mask
        if self._panoptic_ds is not None:
            if self._panoptic_ds.count >= 3:
                # RGB-encoded: decode R + G*256 + B*65536
                rgb = self._panoptic_ds.read([1, 2, 3], window=window)
                rgb = np.moveaxis(rgb, 0, -1)  # (3, H, W) → (H, W, 3)
                panoptic = decode_panoptic_array(rgb)
            else:
                # Single-band: use values directly
                panoptic = self._panoptic_ds.read(1, window=window).astype(np.int32)
        else:
            # No panoptic raster: generate from semantic using connected components
            panoptic = self._generate_panoptic_from_semantic(semantic)

        return MaskBundle(panoptic=panoptic, semantic=semantic)

    def _generate_panoptic_from_semantic(
        self, semantic: np.ndarray
    ) -> np.ndarray:
        """Generate panoptic IDs from semantic mask using connected components."""
        thing_ids = {c.id for c in self._categories if c.is_thing}
        panoptic = np.zeros_like(semantic, dtype=np.int32)
        next_id = max(c.id for c in self._categories) + 1

        for cat_id in np.unique(semantic):
            if cat_id == 0:
                continue
            binary = (semantic == cat_id).astype(np.uint8)

            if cat_id in thing_ids:
                # Each connected component gets a unique ID
                num_labels, labels = cv2.connectedComponents(binary)
                for label in range(1, num_labels):
                    panoptic[labels == label] = next_id
                    next_id += 1
            else:
                # Stuff: one ID per category
                panoptic[binary == 1] = int(cat_id)

        return panoptic

    def close(self) -> None:
        self._semantic_ds.close()
        if self._panoptic_ds is not None:
            self._panoptic_ds.close()

    def __enter__(self) -> RasterAnnotationSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
