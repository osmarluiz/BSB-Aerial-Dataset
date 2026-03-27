"""Command-line interface for Panoptic Generator."""

from __future__ import annotations

from pathlib import Path

import click

from core.config import (
    AnnotationConfig,
    ExtractionConfig,
    OutputConfig,
    PipelineConfig,
    PointSamplingConfig,
    RasterAnnotationConfig,
    ShapefileAnnotationConfig,
    SlidingWindowConfig,
)
from core.models import (
    AnnotationFormat,
    ImageFormat,
    OutputMode,
    PipelineListener,
)
from core.pipeline import Pipeline
from core.stats import stats_to_text


class CLIListener(PipelineListener):
    """Print stages to console with optional tqdm progress."""

    def __init__(self) -> None:
        self._pbar = None

    def on_stage_start(self, name: str) -> None:
        if self._pbar:
            self._pbar.close()
            self._pbar = None
        click.echo(f"\n>> {name}")

    def on_stage_end(self, name: str) -> None:
        if self._pbar:
            self._pbar.close()
            self._pbar = None

    def on_progress(self, current: int, total: int, message: str) -> None:
        if self._pbar is None:
            try:
                from tqdm import tqdm

                self._pbar = tqdm(total=total, desc=message, unit="tile")
            except ImportError:
                pass
        if self._pbar:
            self._pbar.n = current
            self._pbar.refresh()
        elif current % 100 == 0 or current == total:
            click.echo(f"   {current}/{total} {message}")

    def on_warning(self, message: str) -> None:
        click.echo(f"   Warning: {message}", err=True)

    def on_error(self, message: str) -> None:
        click.echo(f"   ERROR: {message}", err=True)


OUTPUT_MODE_MAP = {
    "semantic": OutputMode.SEMANTIC,
    "instance": OutputMode.INSTANCE,
    "panoptic": OutputMode.PANOPTIC,
    "all": OutputMode.ALL,
}


def _build_config(**kwargs) -> PipelineConfig:
    """Convert flat CLI kwargs to nested PipelineConfig."""
    # Annotation config
    if kwargs.get("annotation_shapefile"):
        annotation: AnnotationConfig = ShapefileAnnotationConfig(
            shapefile_path=kwargs["annotation_shapefile"],
            class_column=kwargs.get("class_column", "class_id"),
        )
    elif kwargs.get("annotation_semantic"):
        annotation = RasterAnnotationConfig(
            semantic_path=kwargs["annotation_semantic"],
            panoptic_path=kwargs.get("annotation_panoptic"),
        )
    else:
        raise click.UsageError(
            "Either --annotation-shapefile or --annotation-semantic is required"
        )

    # Sampling config
    if kwargs.get("sampling") == "points":
        shapefiles: dict[str, Path] = {}
        if kwargs.get("train_shape"):
            shapefiles["train"] = kwargs["train_shape"]
        if kwargs.get("val_shape"):
            shapefiles["val"] = kwargs["val_shape"]
        if kwargs.get("test_shape"):
            shapefiles["test"] = kwargs["test_shape"]
        if not shapefiles:
            raise click.UsageError("At least --train-shape is required for points mode")
        sampling = PointSamplingConfig(shapefiles=shapefiles)
    else:
        t, v, te = kwargs.get("split_ratios", (0.7, 0.15, 0.15))
        sampling = SlidingWindowConfig(
            stride_y=kwargs.get("stride_y"),
            stride_x=kwargs.get("stride_x"),
            split_ratios={"train": t, "val": v, "test": te},
            seed=kwargs.get("seed", 42),
        )

    # Bands
    bands = list(kwargs.get("bands", ())) or None

    # Formats
    formats_raw = kwargs.get("formats", ("coco_panoptic",))
    formats = [AnnotationFormat(f) for f in formats_raw]

    return PipelineConfig(
        image_path=kwargs["image"],
        annotation=annotation,
        sampling=sampling,
        tile_height=kwargs.get("tile_height", 512),
        tile_width=kwargs.get("tile_width", 512),
        category_config_path=kwargs.get("categories"),
        extraction=ExtractionConfig(
            bands=bands,
            output_mode=OUTPUT_MODE_MAP[kwargs.get("output_mode", "all")],
            min_segment_area=kwargs.get("min_segment_area", 10),
            min_annotated_ratio=kwargs.get("min_annotated_ratio", 0.0),
        ),
        output=OutputConfig(
            output_dir=kwargs.get("output_dir", Path("./output")),
            image_format=ImageFormat(kwargs.get("image_format", "tiff")),
            annotation_formats=formats,
            generate_stats=kwargs.get("stats", True),
        ),
    )


@click.group()
def cli():
    """Panoptic Generator — Dataset builder for remote sensing segmentation."""


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path))
@click.option("--image", type=click.Path(exists=True, path_type=Path))
@click.option("--annotation-shapefile", type=click.Path(exists=True, path_type=Path))
@click.option("--class-column", default="class_id")
@click.option("--annotation-semantic", type=click.Path(exists=True, path_type=Path))
@click.option("--annotation-panoptic", type=click.Path(exists=True, path_type=Path))
@click.option("--categories", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--sampling",
    type=click.Choice(["points", "sliding_window"]),
    default="points",
)
@click.option("--train-shape", type=click.Path(exists=True, path_type=Path))
@click.option("--val-shape", type=click.Path(exists=True, path_type=Path))
@click.option("--test-shape", type=click.Path(exists=True, path_type=Path))
@click.option("--tile-height", default=512, type=int)
@click.option("--tile-width", default=512, type=int)
@click.option("--stride-y", default=None, type=int)
@click.option("--stride-x", default=None, type=int)
@click.option("--split-ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15))
@click.option("--seed", default=42, type=int)
@click.option("--bands", multiple=True, type=int)
@click.option(
    "--output-mode",
    type=click.Choice(["semantic", "instance", "panoptic", "all"]),
    default="all",
)
@click.option("--output-dir", default="./output", type=click.Path(path_type=Path))
@click.option("--image-format", type=click.Choice(["tiff", "png"]), default="tiff")
@click.option(
    "--formats",
    multiple=True,
    type=click.Choice(
        ["coco_panoptic", "coco_instance", "yolo_detect", "yolo_seg", "pascal_voc"]
    ),
    default=["coco_panoptic"],
)
@click.option("--min-segment-area", default=10, type=int)
@click.option("--min-annotated-ratio", default=0.0, type=float)
@click.option("--stats/--no-stats", default=True)
def generate(**kwargs):
    """Generate a segmentation dataset."""
    if kwargs.get("config_path"):
        config = PipelineConfig.load(kwargs["config_path"])
    else:
        if not kwargs.get("image"):
            raise click.UsageError("--image is required (or use --config)")
        config = _build_config(**kwargs)

    listener = CLIListener()
    pipeline = Pipeline(config, listener=listener)

    try:
        stats = pipeline.run()
        if stats:
            click.echo("\n" + stats_to_text(stats))
        click.echo(f"\nDataset written to: {config.output.output_dir}")
    except Exception as e:
        click.echo(f"\nError: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option("--image", type=click.Path(exists=True, path_type=Path))
@click.option("--shapefile", type=click.Path(exists=True, path_type=Path))
def info(image: Path | None, shapefile: Path | None):
    """Inspect an image or shapefile."""
    if image:
        from core.image_reader import ImageReader

        with ImageReader(image) as img:
            click.echo(f"File: {image}")
            click.echo(f"Size: {img.width} x {img.height} px")
            click.echo(f"Bands: {img.band_count} ({', '.join(img.band_names)})")
            click.echo(f"CRS: {img.crs}")
            px, py = img.pixel_size
            click.echo(f"Pixel size: {px:.4f} x {py:.4f}")
            if img.nodata is not None:
                click.echo(f"NoData: {img.nodata}")

    if shapefile:
        import geopandas as gpd

        gdf = gpd.read_file(shapefile)
        click.echo(f"File: {shapefile}")
        click.echo(f"Features: {len(gdf)}")
        click.echo(f"Geometry: {gdf.geometry.geom_type.unique().tolist()}")
        click.echo(f"CRS: {gdf.crs}")
        click.echo(f"Columns: {', '.join(gdf.columns.tolist())}")
