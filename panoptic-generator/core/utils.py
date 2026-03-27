"""Shared utility functions: panoptic ID encoding, CRS handling, normalization."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio.crs import CRS

from core.models import CategoryInfo

logger = logging.getLogger(__name__)


# ── Panoptic ID Encoding (scalar) ───────────────────────────────────────────


def encode_panoptic_id(instance_id: int) -> tuple[int, int, int]:
    """Encode integer ID → (R, G, B) for COCO panoptic mask PNG."""
    r = instance_id % 256
    g = (instance_id >> 8) % 256
    b = (instance_id >> 16) % 256
    return (r, g, b)


def decode_panoptic_id(r: int, g: int, b: int) -> int:
    """Decode (R, G, B) → integer ID."""
    return r + g * 256 + b * 65536


# ── Panoptic ID Encoding (vectorized) ───────────────────────────────────────


def encode_panoptic_array(id_array: np.ndarray) -> np.ndarray:
    """Vectorized: (H, W) int32 → (H, W, 3) uint8 RGB."""
    rgb = np.zeros((*id_array.shape, 3), dtype=np.uint8)
    rgb[:, :, 0] = id_array % 256
    rgb[:, :, 1] = (id_array >> 8) % 256
    rgb[:, :, 2] = (id_array >> 16) % 256
    return rgb


def decode_panoptic_array(rgb_array: np.ndarray) -> np.ndarray:
    """Vectorized: (H, W, 3) uint8 RGB → (H, W) int32."""
    return (
        rgb_array[:, :, 0].astype(np.int32)
        + rgb_array[:, :, 1].astype(np.int32) * 256
        + rgb_array[:, :, 2].astype(np.int32) * 65536
    )


# ── CRS Handling ─────────────────────────────────────────────────────────────


def ensure_same_crs(
    gdf: gpd.GeoDataFrame, target_crs: CRS, source_name: str = "shapefile"
) -> gpd.GeoDataFrame:
    """
    Ensure a GeoDataFrame is in the target CRS. Reprojects if needed.

    Raises ValueError if gdf has no CRS defined.
    """
    if gdf.crs is None:
        raise ValueError(
            f"{source_name} has no CRS defined. "
            "Please set a coordinate reference system."
        )

    if gdf.crs != target_crs:
        logger.warning(
            "Reprojecting %s from %s to %s", source_name, gdf.crs, target_crs
        )
        return gdf.to_crs(target_crs)

    return gdf


# ── Category I/O ─────────────────────────────────────────────────────────────


def load_categories(path: Path) -> list[CategoryInfo]:
    """Load categories from YAML or JSON file."""
    text = path.read_text(encoding="utf-8")

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required to load .yaml files") from e
        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"Unsupported category file format: {path.suffix}")

    raw_categories = data if isinstance(data, list) else data.get("categories", [])

    categories = []
    seen_ids: set[int] = set()
    for item in raw_categories:
        cat_id = int(item["id"])
        if cat_id < 1:
            raise ValueError(f"Category ID must be >= 1, got {cat_id}")
        if cat_id in seen_ids:
            raise ValueError(f"Duplicate category ID: {cat_id}")
        seen_ids.add(cat_id)

        categories.append(
            CategoryInfo(
                id=cat_id,
                name=str(item["name"]),
                is_thing=bool(item.get("is_thing", True)),
                supercategory=str(item.get("supercategory", "")),
            )
        )

    return categories


def save_categories(categories: list[CategoryInfo], path: Path) -> None:
    """Save categories to YAML or JSON file."""
    data = {
        "categories": [
            {
                "id": c.id,
                "name": c.name,
                "is_thing": c.is_thing,
                "supercategory": c.supercategory,
            }
            for c in categories
        ]
    }

    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML is required to save .yaml files") from e
        path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    elif path.suffix == ".json":
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported category file format: {path.suffix}")


# ── Normalization ────────────────────────────────────────────────────────────


def normalize_to_uint8(
    array: np.ndarray,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
) -> np.ndarray:
    """Normalize float array to uint8 using percentile stretch."""
    low = np.percentile(array, percentile_low)
    high = np.percentile(array, percentile_high)
    if high - low < 1e-6:
        return np.zeros(array.shape, dtype=np.uint8)
    clipped = np.clip(array, low, high)
    return ((clipped - low) / (high - low) * 255).astype(np.uint8)
