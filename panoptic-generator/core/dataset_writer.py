"""Write tiles, masks, and annotation files to disk."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio

from core.models import (
    CategoryInfo,
    FileOutput,
    ImageFormat,
    MaskBundle,
    OutputMode,
    TileData,
)
from core.utils import encode_panoptic_array, normalize_to_uint8


class DatasetWriter:
    """
    Streaming writer: write_tile() is called per-tile,
    write_annotations() at the end.
    """

    def __init__(
        self,
        output_dir: Path,
        image_format: ImageFormat = ImageFormat.TIFF,
        output_mode: OutputMode = OutputMode.ALL,
        create_stuff_masks: bool = True,
        categories: list[CategoryInfo] | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._image_fmt = image_format
        self._mode = output_mode
        self._stuff = create_stuff_masks
        self._categories = categories or []
        self._thing_ids = {c.id for c in self._categories if c.is_thing}
        self._created_dirs: set[Path] = set()

    def write_tile(self, tile: TileData) -> None:
        """Write a single tile's image and masks to disk."""
        split = tile.location.split
        tile_id = tile.location.id
        ext = self._image_fmt.value

        # Image
        img_dir = self._output_dir / f"image_{split}"
        self._ensure_dir(img_dir)
        self._write_image(tile.image, img_dir / f"{tile_id}.{ext}")

        if tile.masks is None:
            return

        # Semantic mask
        if OutputMode.SEMANTIC in self._mode:
            cls_dir = self._output_dir / f"class_{split}"
            self._ensure_dir(cls_dir)
            self._write_mask_grayscale(
                tile.masks.semantic, cls_dir / f"{tile_id}.png"
            )

        # Panoptic mask
        if OutputMode.PANOPTIC in self._mode:
            pan_dir = self._output_dir / f"panoptic_{split}"
            self._ensure_dir(pan_dir)
            self._write_mask_rgb(tile.masks.panoptic, pan_dir / f"{tile_id}.png")

            # Stuff-only mask
            if self._stuff:
                stuff_dir = self._output_dir / f"panoptic_stuff_{split}"
                self._ensure_dir(stuff_dir)
                stuff_mask = self._derive_stuff_mask(tile.masks)
                self._write_mask_rgb(
                    stuff_mask, stuff_dir / f"{tile_id}.png"
                )

    def write_annotations(self, file_outputs: list[FileOutput]) -> None:
        """Write all annotation files produced by builders."""
        for fo in file_outputs:
            full_path = self._output_dir / fo.relative_path
            self._ensure_dir(full_path.parent)

            if isinstance(fo.content, bytes):
                full_path.write_bytes(fo.content)
            else:
                full_path.write_text(fo.content, encoding=fo.encoding)

    def _ensure_dir(self, path: Path) -> None:
        if path not in self._created_dirs:
            path.mkdir(parents=True, exist_ok=True)
            self._created_dirs.add(path)

    def _write_image(self, image: np.ndarray, path: Path) -> None:
        """Write image tile to disk."""
        bands, height, width = image.shape

        if self._image_fmt == ImageFormat.TIFF:
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                count=bands,
                height=height,
                width=width,
                dtype=image.dtype,
            ) as dst:
                dst.write(image)
        else:
            # PNG: normalize to uint8, use first 3 bands for RGB
            if bands >= 3:
                rgb = np.stack(
                    [normalize_to_uint8(image[i]) for i in range(3)], axis=-1
                )
            elif bands == 1:
                gray = normalize_to_uint8(image[0])
                rgb = np.stack([gray, gray, gray], axis=-1)
            else:
                ch0 = normalize_to_uint8(image[0])
                ch1 = normalize_to_uint8(image[1]) if bands > 1 else ch0
                rgb = np.stack([ch0, ch1, ch0], axis=-1)

            cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    @staticmethod
    def _write_mask_grayscale(mask: np.ndarray, path: Path) -> None:
        """Write semantic mask as single-channel PNG."""
        cv2.imwrite(str(path), mask)

    @staticmethod
    def _write_mask_rgb(panoptic_mask: np.ndarray, path: Path) -> None:
        """Write panoptic mask as RGB PNG using panoptic ID encoding."""
        rgb = encode_panoptic_array(panoptic_mask)
        cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    def _derive_stuff_mask(self, masks: MaskBundle) -> np.ndarray:
        """Zero out thing pixels from panoptic mask."""
        stuff = masks.panoptic.copy()
        stuff[np.isin(masks.semantic, list(self._thing_ids))] = 0
        return stuff
