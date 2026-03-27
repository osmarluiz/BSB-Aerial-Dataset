"""Incremental dataset statistics accumulator."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from core.models import CategoryInfo, TileData


@dataclass
class CategoryStats:
    category_id: int = 0
    name: str = ""
    is_thing: bool = True
    total_pixels: int = 0
    total_instances: int = 0
    tile_count: int = 0
    area_sum: int = 0
    min_area: int = 999_999_999
    max_area: int = 0

    @property
    def mean_area(self) -> float:
        return self.area_sum / self.total_instances if self.total_instances > 0 else 0.0


@dataclass
class SplitStats:
    split: str = ""
    num_tiles: int = 0
    total_segments: int = 0
    skipped_tiles: int = 0


@dataclass
class DatasetStats:
    tile_size: tuple[int, int] = (0, 0)
    splits: dict[str, SplitStats] = field(default_factory=dict)
    categories: dict[int, CategoryStats] = field(default_factory=dict)
    total_tiles: int = 0
    total_background_pixels: int = 0
    total_pixels: int = 0

    @property
    def background_ratio(self) -> float:
        return (
            self.total_background_pixels / self.total_pixels
            if self.total_pixels > 0
            else 0.0
        )


class StatsAccumulator:
    """Accumulates statistics tile-by-tile during streaming."""

    def __init__(self, categories: list[CategoryInfo]) -> None:
        self._cat_stats: dict[int, CategoryStats] = {}
        for c in categories:
            self._cat_stats[c.id] = CategoryStats(
                category_id=c.id, name=c.name, is_thing=c.is_thing
            )

        self._split_stats: dict[str, SplitStats] = defaultdict(
            lambda: SplitStats()
        )
        self._total_tiles = 0
        self._total_bg = 0
        self._total_px = 0
        self._tile_size: tuple[int, int] = (0, 0)

    def update(self, tile: TileData) -> None:
        """Update stats with one tile."""
        split = tile.location.split
        ss = self._split_stats[split]
        ss.split = split
        ss.num_tiles += 1
        self._total_tiles += 1

        if self._tile_size == (0, 0):
            self._tile_size = (tile.location.height, tile.location.width)

        if tile.masks is None:
            return

        semantic = tile.masks.semantic
        total_px = semantic.size
        self._total_px += total_px

        # Background pixels
        bg_px = int((semantic == 0).sum())
        self._total_bg += bg_px

        # Per-category pixel counts using bincount (fast)
        max_cat = max(self._cat_stats.keys()) if self._cat_stats else 0
        counts = np.bincount(semantic.ravel(), minlength=max_cat + 1)

        present: set[int] = set()
        for cat_id, cs in self._cat_stats.items():
            if cat_id < len(counts) and counts[cat_id] > 0:
                cs.total_pixels += int(counts[cat_id])
                present.add(cat_id)
                cs.tile_count += 1

        # Per-segment stats
        for seg in tile.segments:
            cat_id = seg.category_id
            if cat_id not in self._cat_stats:
                continue
            cs = self._cat_stats[cat_id]
            cs.total_instances += 1
            cs.area_sum += seg.area
            cs.min_area = min(cs.min_area, seg.area)
            cs.max_area = max(cs.max_area, seg.area)

        ss.total_segments += len(tile.segments)

    def record_skipped(self, split: str) -> None:
        self._split_stats[split].skipped_tiles += 1

    def finalize(self) -> DatasetStats:
        # Fix min_area for categories with 0 instances
        for cs in self._cat_stats.values():
            if cs.total_instances == 0:
                cs.min_area = 0

        return DatasetStats(
            tile_size=self._tile_size,
            splits=dict(self._split_stats),
            categories=dict(self._cat_stats),
            total_tiles=self._total_tiles,
            total_background_pixels=self._total_bg,
            total_pixels=self._total_px,
        )


# ── Formatters ───────────────────────────────────────────────────────────────


def stats_to_dict(stats: DatasetStats) -> dict:
    """Convert to plain dict for JSON serialization."""
    return {
        "total_tiles": stats.total_tiles,
        "tile_size": list(stats.tile_size),
        "background_ratio": round(stats.background_ratio, 4),
        "splits": {
            name: {
                "num_tiles": ss.num_tiles,
                "total_segments": ss.total_segments,
                "skipped_tiles": ss.skipped_tiles,
            }
            for name, ss in stats.splits.items()
        },
        "categories": {
            str(cid): {
                "name": cs.name,
                "is_thing": cs.is_thing,
                "total_pixels": cs.total_pixels,
                "total_instances": cs.total_instances,
                "tile_count": cs.tile_count,
                "mean_area": round(cs.mean_area, 1),
                "min_area": cs.min_area,
                "max_area": cs.max_area,
            }
            for cid, cs in stats.categories.items()
        },
    }


def stats_to_text(stats: DatasetStats) -> str:
    """Human-readable summary text."""
    lines: list[str] = []
    h, w = stats.tile_size
    lines.append("=" * 60)
    lines.append("Dataset Summary")
    lines.append("=" * 60)
    lines.append(f"Total tiles: {stats.total_tiles} ({h}x{w})")

    splits_str = ", ".join(
        f"{name}={ss.num_tiles}" for name, ss in sorted(stats.splits.items())
    )
    lines.append(f"Splits: {splits_str}")

    skipped = sum(ss.skipped_tiles for ss in stats.splits.values())
    if skipped > 0:
        lines.append(f"Skipped: {skipped} tiles")

    lines.append(f"Background: {stats.background_ratio:.1%}")
    lines.append("")

    # Category table
    lines.append(
        f"{'ID':>4}  {'Name':<20}  {'Type':<6}  {'Pixels%':>8}  "
        f"{'Instances':>10}  {'Tiles':>6}"
    )
    lines.append("-" * 60)

    total_annotated = stats.total_pixels - stats.total_background_pixels
    if total_annotated <= 0:
        total_annotated = 1  # avoid division by zero

    for cid in sorted(stats.categories.keys()):
        cs = stats.categories[cid]
        pct = cs.total_pixels / total_annotated * 100 if total_annotated > 0 else 0
        type_str = "thing" if cs.is_thing else "stuff"
        inst_str = str(cs.total_instances) if cs.is_thing else "-"
        lines.append(
            f"{cid:>4}  {cs.name:<20}  {type_str:<6}  {pct:>7.1f}%  "
            f"{inst_str:>10}  {cs.tile_count:>6}"
        )

    lines.append("=" * 60)
    return "\n".join(lines)
