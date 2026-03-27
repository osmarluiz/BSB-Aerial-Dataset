"""Orchestrate the full streaming pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from core.annotation_source import (
    RasterAnnotationSource,
    VectorAnnotationSource,
)
from core.builders import create_builder
from core.config import (
    PipelineConfig,
    PointSamplingConfig,
    RasterAnnotationConfig,
    ShapefileAnnotationConfig,
    SlidingWindowConfig,
)
from core.dataset_writer import DatasetWriter
from core.image_reader import ImageReader
from core.models import (
    CategoryInfo,
    PipelineListener,
    TileLocation,
)
from core.sampler import PointSampler, SlidingWindowSampler
from core.stats import (
    DatasetStats,
    StatsAccumulator,
    stats_to_dict,
    stats_to_text,
)
from core.tile_extractor import TileExtractor

logger = logging.getLogger(__name__)


class Pipeline:
    """Streaming pipeline: extracts, writes, and accumulates stats one tile at a time."""

    def __init__(
        self,
        config: PipelineConfig,
        listener: PipelineListener | None = None,
    ) -> None:
        self._config = config
        self._listener = listener or PipelineListener()
        self._builders: list = []

    def run(self) -> DatasetStats | None:
        """Execute the full streaming pipeline. Returns stats if enabled."""
        image_reader = None
        annotation_source = None

        try:
            # Stage 1: Load image
            image_reader = self._load_image()

            # Stage 2: Load annotations
            annotation_source = self._load_annotations(image_reader)

            # Stage 3: Get categories
            categories = (
                annotation_source.get_categories()
                if annotation_source
                else self._config.categories or []
            )

            # Stage 4: Generate tile locations + count
            total_tiles = self._estimate_tile_count(image_reader)
            locations = self._generate_tile_locations(image_reader)

            # Stage 5+6+7: Streaming core
            stats = self._process_tiles(
                image_reader, annotation_source, categories, locations, total_tiles
            )

            # Stage 8: Write annotation files
            self._write_annotations(categories)

            # Stage 9: Write stats
            if stats and self._config.output.generate_stats:
                self._write_stats(stats)

            self._listener.on_stage_end("Pipeline complete")
            return stats

        finally:
            if image_reader:
                image_reader.close()
            if annotation_source:
                annotation_source.close()

    def _load_image(self) -> ImageReader:
        self._listener.on_stage_start("Loading image")
        reader = ImageReader(self._config.image_path)
        self._listener.on_stage_end(
            f"Image: {reader.width}x{reader.height}, {reader.band_count} bands"
        )
        return reader

    def _load_annotations(self, img: ImageReader):
        self._listener.on_stage_start("Loading annotations")
        cfg = self._config

        if isinstance(cfg.annotation, ShapefileAnnotationConfig):
            if img.crs is None:
                raise ValueError("Image has no CRS — cannot use shapefile annotations")

            source = VectorAnnotationSource(
                shapefile_path=cfg.annotation.shapefile_path,
                class_column=cfg.annotation.class_column,
                categories=cfg.categories or [],
                image_transform=img.transform,
                image_crs=img.crs,
                image_shape=(img.height, img.width),
            )
        elif isinstance(cfg.annotation, RasterAnnotationConfig):
            source = RasterAnnotationSource(
                semantic_path=cfg.annotation.semantic_path,
                panoptic_path=cfg.annotation.panoptic_path,
                categories=cfg.categories or [],
            )
        else:
            raise ValueError(f"Unknown annotation config type: {type(cfg.annotation)}")

        self._listener.on_stage_end("Annotations loaded")
        return source

    def _generate_tile_locations(
        self, img: ImageReader
    ) -> Iterator[TileLocation]:
        self._listener.on_stage_start("Generating tile locations")
        cfg = self._config

        if isinstance(cfg.sampling, PointSamplingConfig):
            if img.crs is None:
                raise ValueError("Image has no CRS — cannot use point shapefiles")
            sampler = PointSampler(
                shapefiles=cfg.sampling.shapefiles,
                tile_height=cfg.tile_height,
                tile_width=cfg.tile_width,
                image_transform=img.transform,
                image_crs=img.crs,
                image_height=img.height,
                image_width=img.width,
            )
        elif isinstance(cfg.sampling, SlidingWindowConfig):
            sampler = SlidingWindowSampler(
                tile_height=cfg.tile_height,
                tile_width=cfg.tile_width,
                stride_y=cfg.sampling.stride_y or cfg.tile_height,
                stride_x=cfg.sampling.stride_x or cfg.tile_width,
                image_height=img.height,
                image_width=img.width,
                split_ratios=cfg.sampling.split_ratios,
                seed=cfg.sampling.seed,
            )
        else:
            raise ValueError(f"Unknown sampling config: {type(cfg.sampling)}")

        self._listener.on_stage_end("Tile locations generated")
        return sampler.generate()

    def _estimate_tile_count(self, img: ImageReader) -> int:
        """Estimate total tiles without consuming the generator."""
        cfg = self._config
        if isinstance(cfg.sampling, SlidingWindowConfig):
            stride_y = cfg.sampling.stride_y or cfg.tile_height
            stride_x = cfg.sampling.stride_x or cfg.tile_width
            rows = max(0, (img.height - cfg.tile_height) // stride_y + 1)
            cols = max(0, (img.width - cfg.tile_width) // stride_x + 1)
            return rows * cols
        elif isinstance(cfg.sampling, PointSamplingConfig):
            # Count points in all shapefiles
            import geopandas as gpd

            total = 0
            for shp_path in cfg.sampling.shapefiles.values():
                gdf = gpd.read_file(shp_path)
                total += len(gdf)
            return total
        return 0

    def _process_tiles(
        self,
        img: ImageReader,
        source,
        categories: list[CategoryInfo],
        locations: Iterator[TileLocation],
        total_tiles: int,
    ) -> DatasetStats | None:
        """Streaming core: extract → write → accumulate, one tile at a time."""
        cfg = self._config
        self._listener.on_stage_start("Processing tiles")

        # Create extractor
        extractor = TileExtractor(
            image_reader=img,
            annotation_source=source,
            categories=categories,
            bands=cfg.extraction.bands,
            output_mode=cfg.extraction.output_mode,
            min_segment_area=cfg.extraction.min_segment_area,
            min_annotated_ratio=cfg.extraction.min_annotated_ratio,
            max_nodata_ratio=cfg.extraction.max_nodata_ratio,
        )

        # Create writer
        writer = DatasetWriter(
            output_dir=cfg.output.output_dir,
            image_format=cfg.output.image_format,
            output_mode=cfg.extraction.output_mode,
            create_stuff_masks=cfg.output.create_stuff_masks,
            categories=categories,
        )

        # Create builders
        self._builders = [
            create_builder(fmt) for fmt in cfg.output.annotation_formats
        ]

        # Create stats accumulator
        stats_acc = StatsAccumulator(categories) if cfg.output.generate_stats else None

        # Stream tiles
        current = 0
        for tile_data in extractor.extract_tiles(
            locations,
            on_warning=self._listener.on_warning,
            total=total_tiles,
        ):
            current += 1
            self._listener.on_progress(current, total_tiles, "Processing tiles")

            # Write to disk
            writer.write_tile(tile_data)

            # Accumulate annotations (metadata only)
            for builder in self._builders:
                builder.add_tile(tile_data.location, tile_data.segments)

            # Accumulate stats
            if stats_acc:
                stats_acc.update(tile_data)

        self._listener.on_stage_end(f"Processed {current} tiles")
        return stats_acc.finalize() if stats_acc else None

    def _write_annotations(self, categories: list[CategoryInfo]) -> None:
        self._listener.on_stage_start("Writing annotations")
        from core.dataset_writer import DatasetWriter

        all_files = []
        for builder in self._builders:
            all_files.extend(builder.finalize(categories))

        # Reuse writer just for annotations
        writer = DatasetWriter(output_dir=self._config.output.output_dir)
        writer.write_annotations(all_files)
        self._listener.on_stage_end(f"Wrote {len(all_files)} annotation files")

    def _write_stats(self, stats: DatasetStats) -> None:
        output_dir = self._config.output.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        stats_path = output_dir / "stats.json"
        stats_path.write_text(
            json.dumps(stats_to_dict(stats), indent=2), encoding="utf-8"
        )

        # Text summary
        summary = stats_to_text(stats)
        logger.info("\n%s", summary)
        self._listener.on_stage_end(f"Stats written to {stats_path}")
