"""Extract image + masks for each tile. Streaming: yields one TileData at a time."""

from __future__ import annotations

import logging
from typing import Callable, Iterator

import cv2
import numpy as np
from scipy import ndimage

from core.image_reader import ImageReader
from core.models import (
    AnnotationSource,
    CategoryInfo,
    MaskBundle,
    OutputMode,
    SegmentInfo,
    TileData,
    TileLocation,
)

logger = logging.getLogger(__name__)


class TileExtractor:
    """Streaming tile extractor — yields one TileData at a time."""

    def __init__(
        self,
        image_reader: ImageReader,
        annotation_source: AnnotationSource | None,
        categories: list[CategoryInfo] | None = None,
        bands: list[int] | None = None,
        output_mode: OutputMode = OutputMode.ALL,
        min_segment_area: int = 10,
        min_annotated_ratio: float = 0.0,
        max_nodata_ratio: float = 1.0,
    ) -> None:
        self._reader = image_reader
        self._source = annotation_source
        self._categories = categories or []
        self._bands = bands
        self._mode = output_mode
        self._min_area = min_segment_area
        self._min_annotated = min_annotated_ratio
        self._max_nodata = max_nodata_ratio
        self._nodata = image_reader.nodata
        self._thing_ids = {c.id for c in self._categories if c.is_thing}

    def extract_tiles(
        self,
        locations: Iterator[TileLocation],
        on_progress: Callable[[int, int], None] | None = None,
        on_warning: Callable[[str], None] | None = None,
        total: int | None = None,
    ) -> Iterator[TileData]:
        """
        Yield TileData one at a time. Skips failed tiles with a warning.

        Args:
            locations: tile locations to extract.
            on_progress: callback(current, total) after each tile.
            on_warning: callback(message) for non-fatal issues.
            total: total number of tiles (for progress reporting).
        """
        current = 0
        for loc in locations:
            current += 1
            try:
                tile = self._extract_one(loc, on_warning)
            except Exception as e:
                msg = f"Tile {loc.id} failed: {e}, skipping"
                logger.warning(msg)
                if on_warning:
                    on_warning(msg)
                continue

            if tile is not None:
                yield tile

            if on_progress and total:
                on_progress(current, total)

    def _extract_one(
        self,
        loc: TileLocation,
        on_warning: Callable[[str], None] | None = None,
    ) -> TileData | None:
        """Extract a single tile. Returns None if tile should be skipped."""
        # 1. Read image window
        image = self._reader.read_window(
            loc.row, loc.col, loc.height, loc.width, self._bands
        )

        # 2. Check NoData ratio
        if self._nodata is not None and self._max_nodata < 1.0:
            nodata_ratio = np.isclose(image, self._nodata).any(axis=0).mean()
            if nodata_ratio > self._max_nodata:
                msg = f"Tile {loc.id} has {nodata_ratio:.1%} NoData, skipping"
                if on_warning:
                    on_warning(msg)
                return None

        # 3. Get masks (single rasterization)
        masks = None
        if self._source is not None:
            masks = self._source.get_masks(loc.row, loc.col, loc.height, loc.width)

        # 4. Check annotation ratio (no double rasterization — masks already computed)
        if masks is not None and self._min_annotated > 0:
            ratio = np.count_nonzero(masks.semantic) / masks.semantic.size
            if ratio < self._min_annotated:
                msg = f"Tile {loc.id} below threshold ({ratio:.2%}), skipping"
                if on_warning:
                    on_warning(msg)
                return None

        # 5. Extract segments
        segments = self._extract_segments(masks) if masks is not None else []

        return TileData(
            location=loc, image=image, masks=masks, segments=segments
        )

    def _extract_segments(self, masks: MaskBundle) -> list[SegmentInfo]:
        """Analyze masks to extract per-segment metadata."""
        panoptic = masks.panoptic
        semantic = masks.semantic

        unique_ids = np.unique(panoptic)
        unique_ids = unique_ids[unique_ids != 0]  # skip background

        if len(unique_ids) == 0:
            return []

        # Use find_objects for fast bounding slice lookup
        max_id = unique_ids.max()
        slices = ndimage.find_objects(panoptic.astype(np.int32), max_label=max_id)

        segments: list[SegmentInfo] = []
        need_contours = bool(self._mode & (OutputMode.INSTANCE | OutputMode.PANOPTIC))

        for uid in unique_ids:
            uid = int(uid)
            slice_idx = uid - 1  # find_objects is 0-indexed
            if slice_idx < 0 or slice_idx >= len(slices) or slices[slice_idx] is None:
                # Fallback: compute manually
                binary = panoptic == uid
            else:
                # Work within the bounding slice (much faster for large tiles)
                row_slice, col_slice = slices[slice_idx]
                sub_panoptic = panoptic[row_slice, col_slice]
                binary_sub = sub_panoptic == uid
                area = int(binary_sub.sum())

                if area < self._min_area:
                    continue

                # Category from semantic mask
                sub_semantic = semantic[row_slice, col_slice]
                cat_id = int(sub_semantic[binary_sub][0])

                # Bbox in full-tile coordinates (COCO: x, y, w, h)
                rows_offset = row_slice.start
                cols_offset = col_slice.start
                local_rows, local_cols = np.where(binary_sub)
                x = int(local_cols.min()) + cols_offset
                y = int(local_rows.min()) + rows_offset
                w = int(local_cols.ptp()) + 1
                h = int(local_rows.ptp()) + 1

                # Contours (only for thing classes)
                contour = None
                if need_contours and cat_id in self._thing_ids:
                    contour = self._extract_contours(binary_sub, cols_offset, rows_offset)

                segments.append(
                    SegmentInfo(
                        instance_id=uid,
                        category_id=cat_id,
                        area=area,
                        bbox=(x, y, w, h),
                        contour=contour,
                    )
                )
                continue

            # Fallback path (when find_objects slice is unavailable)
            binary = panoptic == uid
            area = int(binary.sum())
            if area < self._min_area:
                continue

            cat_id = int(semantic[binary][0])
            rows_all, cols_all = np.where(binary)
            x = int(cols_all.min())
            y = int(rows_all.min())
            w = int(cols_all.ptp()) + 1
            h = int(rows_all.ptp()) + 1

            contour = None
            if need_contours and cat_id in self._thing_ids:
                contour = self._extract_contours(binary, 0, 0)

            segments.append(
                SegmentInfo(
                    instance_id=uid,
                    category_id=cat_id,
                    area=area,
                    bbox=(x, y, w, h),
                    contour=contour,
                )
            )

        return segments

    @staticmethod
    def _extract_contours(
        binary_mask: np.ndarray, col_offset: int = 0, row_offset: int = 0
    ) -> list[list[int]] | None:
        """
        Extract polygon contours using cv2.findContours.
        Returns [[x1,y1,x2,y2,...], ...] in full-tile coordinates.
        """
        mask_u8 = (binary_mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        result = []
        for cnt in contours:
            if len(cnt) < 3:
                continue
            flat: list[int] = []
            for point in cnt.reshape(-1, 2):
                flat.append(int(point[0]) + col_offset)
                flat.append(int(point[1]) + row_offset)
            if len(flat) >= 6:
                result.append(flat)

        return result if result else None
