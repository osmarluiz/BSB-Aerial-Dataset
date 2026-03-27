"""Shared data classes, enums, and protocols used across all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, Flag, auto
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Protocol

import numpy as np

if TYPE_CHECKING:
    pass


# ── Enums ────────────────────────────────────────────────────────────────────


class SamplingMode(Enum):
    POINTS = "points"
    SLIDING_WINDOW = "sliding_window"


class OutputMode(Flag):
    SEMANTIC = auto()
    INSTANCE = auto()
    PANOPTIC = auto()
    ALL = SEMANTIC | INSTANCE | PANOPTIC


class AnnotationFormat(Enum):
    COCO_PANOPTIC = "coco_panoptic"
    COCO_INSTANCE = "coco_instance"
    YOLO_DETECT = "yolo_detect"
    YOLO_SEG = "yolo_seg"
    PASCAL_VOC = "pascal_voc"


class ImageFormat(Enum):
    TIFF = "tiff"
    PNG = "png"


# ── Data Classes ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class CategoryInfo:
    """A single annotation category."""

    id: int  # 1-based (0 = background)
    name: str
    is_thing: bool  # True = instance (thing), False = stuff
    supercategory: str = ""


@dataclass(slots=True)
class TileLocation:
    """Where a tile lives in the source image."""

    id: int  # Sequential, 1-based
    row: int  # Top-left row (pixels)
    col: int  # Top-left col (pixels)
    height: int  # Tile height (pixels)
    width: int  # Tile width (pixels)
    split: str  # "train", "val", or "test"


@dataclass(slots=True)
class SegmentInfo:
    """Metadata for a single segment (instance or stuff region) within a tile."""

    instance_id: int  # Unique within tile
    category_id: int  # 1-based
    area: int  # Pixel count
    bbox: tuple[int, int, int, int]  # (x, y, w, h) — COCO convention: x=col, y=row
    contour: list[list[int]] | None  # [[x1,y1,...], ...] or None for stuff


@dataclass(slots=True)
class MaskBundle:
    """All masks for a single tile, derived from ONE rasterization pass."""

    panoptic: np.ndarray  # (H, W) int32 — unique instance IDs (0=background)
    semantic: np.ndarray  # (H, W) uint8 — category IDs (0=background)

    def get_instance(self, thing_category_ids: set[int]) -> np.ndarray:
        """Derive instance mask: keep panoptic IDs only for thing categories."""
        thing_mask = np.isin(self.semantic, list(thing_category_ids))
        return np.where(thing_mask, self.panoptic, 0).astype(np.int32)


@dataclass(slots=True)
class TileData:
    """Extracted data for a single tile."""

    location: TileLocation
    image: np.ndarray  # (bands, H, W) float32
    masks: MaskBundle | None  # None if no annotation source
    segments: list[SegmentInfo]


@dataclass(slots=True)
class FileOutput:
    """A file that a builder wants written to disk."""

    relative_path: Path
    content: str | bytes
    encoding: str = "utf-8"


# ── Protocols ────────────────────────────────────────────────────────────────


class AnnotationSource(Protocol):
    """Interface for annotation data providers."""

    def get_categories(self) -> list[CategoryInfo]: ...
    def get_masks(self, row: int, col: int, height: int, width: int) -> MaskBundle: ...
    def close(self) -> None: ...


class Sampler(Protocol):
    """Interface for tile location generators."""

    def generate(self) -> Iterator[TileLocation]: ...


class AnnotationBuilder(Protocol):
    """Accumulator interface: add tiles one by one, finalize at the end."""

    def add_tile(
        self, location: TileLocation, segments: list[SegmentInfo]
    ) -> None: ...
    def finalize(self, categories: list[CategoryInfo]) -> list[FileOutput]: ...


# ── Pipeline Listener ────────────────────────────────────────────────────────


class PipelineListener:
    """Base listener with no-op defaults. Subclass and override what you need."""

    def on_stage_start(self, name: str) -> None:
        pass

    def on_stage_end(self, name: str) -> None:
        pass

    def on_progress(self, current: int, total: int, message: str) -> None:
        pass

    def on_warning(self, message: str) -> None:
        pass

    def on_error(self, message: str) -> None:
        pass
