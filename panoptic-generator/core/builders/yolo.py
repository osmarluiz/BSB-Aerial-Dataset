"""YOLO detection and segmentation format builder."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from core.builders import register_builder
from core.models import (
    AnnotationFormat,
    CategoryInfo,
    FileOutput,
    SegmentInfo,
    TileLocation,
)


class _YOLOBuilderBase:
    """Shared logic for YOLO detect and segment formats."""

    def __init__(self, use_segmentation: bool = False) -> None:
        self._use_seg = use_segmentation
        self._labels: dict[str, dict[str, str]] = defaultdict(dict)
        self._splits: set[str] = set()

    def add_tile(
        self, location: TileLocation, segments: list[SegmentInfo]
    ) -> None:
        self._splits.add(location.split)
        lines: list[str] = []
        tile_w = location.width
        tile_h = location.height

        for seg in segments:
            if seg.contour is None:
                continue

            class_id = seg.category_id - 1  # YOLO is 0-based

            if self._use_seg and seg.contour:
                # Segmentation: normalized polygon points
                points = seg.contour[0]  # Use first contour
                parts: list[str] = []
                for i in range(0, len(points), 2):
                    px = points[i] / tile_w
                    py = points[i + 1] / tile_h
                    parts.append(f"{px:.6f}")
                    parts.append(f"{py:.6f}")
                lines.append(f"{class_id} {' '.join(parts)}")
            else:
                # Detection: normalized center + size
                x, y, w, h = seg.bbox
                cx = (x + w / 2) / tile_w
                cy = (y + h / 2) / tile_h
                nw = w / tile_w
                nh = h / tile_h
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        self._labels[location.split][f"{location.id}.txt"] = "\n".join(lines)

    def finalize(self, categories: list[CategoryInfo]) -> list[FileOutput]:
        outputs: list[FileOutput] = []

        # Label files
        for split, labels in self._labels.items():
            for filename, content in labels.items():
                outputs.append(
                    FileOutput(
                        relative_path=Path(f"labels/{split}/{filename}"),
                        content=content,
                    )
                )

        # data.yaml
        thing_cats = [c for c in categories if c.is_thing]
        yaml_lines = []
        for split in sorted(self._splits):
            yaml_lines.append(f"{split}: images/{split}")
        yaml_lines.append(f"nc: {len(thing_cats)}")
        yaml_lines.append(
            "names: [" + ", ".join(f'"{c.name}"' for c in thing_cats) + "]"
        )

        outputs.append(
            FileOutput(
                relative_path=Path("data.yaml"),
                content="\n".join(yaml_lines) + "\n",
            )
        )

        return outputs


@register_builder(AnnotationFormat.YOLO_DETECT)
class YOLODetectBuilder(_YOLOBuilderBase):
    def __init__(self) -> None:
        super().__init__(use_segmentation=False)


@register_builder(AnnotationFormat.YOLO_SEG)
class YOLOSegBuilder(_YOLOBuilderBase):
    def __init__(self) -> None:
        super().__init__(use_segmentation=True)
