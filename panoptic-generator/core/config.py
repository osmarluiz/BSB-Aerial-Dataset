"""Pydantic configuration models with validation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

from core.models import (
    AnnotationFormat,
    CategoryInfo,
    ImageFormat,
    OutputMode,
)
from core.utils import load_categories


# ── Annotation Config (discriminated union) ──────────────────────────────────


class ShapefileAnnotationConfig(BaseModel):
    mode: Literal["shapefile"] = "shapefile"
    shapefile_path: Path
    class_column: str = "class_id"


class RasterAnnotationConfig(BaseModel):
    mode: Literal["raster"] = "raster"
    semantic_path: Path
    panoptic_path: Path | None = None


AnnotationConfig = ShapefileAnnotationConfig | RasterAnnotationConfig


# ── Sampling Config (discriminated union) ────────────────────────────────────


class PointSamplingConfig(BaseModel):
    mode: Literal["points"] = "points"
    shapefiles: dict[str, Path]

    @field_validator("shapefiles")
    @classmethod
    def must_have_train(cls, v: dict[str, Path]) -> dict[str, Path]:
        if "train" not in v:
            raise ValueError("At least 'train' shapefile is required")
        return v


class SlidingWindowConfig(BaseModel):
    mode: Literal["sliding_window"] = "sliding_window"
    stride_y: int | None = None
    stride_x: int | None = None
    split_ratios: dict[str, float] = {"train": 0.7, "val": 0.15, "test": 0.15}
    seed: int = 42

    @field_validator("split_ratios")
    @classmethod
    def ratios_must_sum_to_one(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return v


SamplingConfig = PointSamplingConfig | SlidingWindowConfig


# ── Extraction Config ────────────────────────────────────────────────────────


class ExtractionConfig(BaseModel):
    bands: list[int] | None = None
    output_mode: OutputMode = OutputMode.ALL
    min_segment_area: int = 10
    min_annotated_ratio: float = 0.0
    max_nodata_ratio: float = 1.0

    @field_validator("bands")
    @classmethod
    def bands_must_be_positive(cls, v: list[int] | None) -> list[int] | None:
        if v and any(b < 1 for b in v):
            raise ValueError("Band indices must be >= 1 (1-based)")
        return v

    @field_validator("min_annotated_ratio", "max_nodata_ratio")
    @classmethod
    def ratio_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Ratio must be between 0.0 and 1.0")
        return v


# ── Output Config ────────────────────────────────────────────────────────────


class OutputConfig(BaseModel):
    output_dir: Path = Path("./output")
    image_format: ImageFormat = ImageFormat.TIFF
    annotation_formats: list[AnnotationFormat] = [AnnotationFormat.COCO_PANOPTIC]
    create_stuff_masks: bool = True
    generate_stats: bool = True

    @field_validator("annotation_formats")
    @classmethod
    def at_least_one_format(
        cls, v: list[AnnotationFormat],
    ) -> list[AnnotationFormat]:
        if not v:
            raise ValueError("At least one annotation format is required")
        return v


# ── Top-Level Config ─────────────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    image_path: Path
    annotation: AnnotationConfig
    sampling: SamplingConfig
    extraction: ExtractionConfig = ExtractionConfig()
    output: OutputConfig = OutputConfig()

    categories: list[CategoryInfo] | None = None
    category_config_path: Path | None = None

    tile_height: int = 512
    tile_width: int = 512

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def validate_and_resolve(self) -> "PipelineConfig":
        # Load categories from file if not provided inline
        if self.categories is None and self.category_config_path is not None:
            self.categories = load_categories(self.category_config_path)

        # Raster mode requires categories
        if isinstance(self.annotation, RasterAnnotationConfig) and not self.categories:
            raise ValueError(
                "Categories must be provided when using raster annotation mode"
            )

        # Tile dimensions must be positive
        if self.tile_height <= 0 or self.tile_width <= 0:
            raise ValueError("Tile dimensions must be > 0")

        # Default strides to tile size
        if isinstance(self.sampling, SlidingWindowConfig):
            if self.sampling.stride_y is None:
                self.sampling.stride_y = self.tile_height
            if self.sampling.stride_x is None:
                self.sampling.stride_x = self.tile_width

        return self

    def save(self, path: Path) -> None:
        """Serialize to YAML file."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML required for config save") from e

        data = self.model_dump(mode="json")
        path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        """Load from YAML file."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError("PyYAML required for config load") from e

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)
