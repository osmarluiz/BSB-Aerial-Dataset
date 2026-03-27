"""Pascal VOC XML format builder."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

from core.builders import register_builder
from core.models import (
    AnnotationFormat,
    CategoryInfo,
    FileOutput,
    SegmentInfo,
    TileLocation,
)


@register_builder(AnnotationFormat.PASCAL_VOC)
class PascalVOCBuilder:
    """Accumulates bounding box annotations, produces Pascal VOC XML files."""

    def __init__(self) -> None:
        self._annotations: dict[str, dict[int, str]] = defaultdict(dict)
        self._imagesets: dict[str, list[int]] = defaultdict(list)
        self._categories: dict[int, str] = {}  # id → name (populated at finalize)

    def add_tile(
        self, location: TileLocation, segments: list[SegmentInfo]
    ) -> None:
        split = location.split
        self._imagesets[split].append(location.id)

        root = Element("annotation")

        filename_el = SubElement(root, "filename")
        filename_el.text = f"{location.id}.tiff"

        size_el = SubElement(root, "size")
        SubElement(size_el, "width").text = str(location.width)
        SubElement(size_el, "height").text = str(location.height)
        SubElement(size_el, "depth").text = "3"

        for seg in segments:
            # VOC only includes things with bounding boxes
            if seg.contour is None:
                continue

            obj = SubElement(root, "object")
            # Name will be resolved at finalize; use category_id for now
            SubElement(obj, "name").text = f"__cat_{seg.category_id}__"
            SubElement(obj, "difficult").text = "0"

            bbox = SubElement(obj, "bndbox")
            x, y, w, h = seg.bbox
            SubElement(bbox, "xmin").text = str(x)
            SubElement(bbox, "ymin").text = str(y)
            SubElement(bbox, "xmax").text = str(x + w)
            SubElement(bbox, "ymax").text = str(y + h)

        xml_str = parseString(tostring(root, encoding="unicode")).toprettyxml(
            indent="  "
        )
        # Remove extra XML declaration from toprettyxml
        lines = xml_str.split("\n")
        if lines and lines[0].startswith("<?xml"):
            xml_str = "\n".join(lines[1:])

        self._annotations[split][location.id] = xml_str

    def finalize(self, categories: list[CategoryInfo]) -> list[FileOutput]:
        cat_map = {c.id: c.name for c in categories if c.is_thing}
        outputs: list[FileOutput] = []

        for split, annots in self._annotations.items():
            for tile_id, xml_str in annots.items():
                # Replace placeholder names with actual category names
                for cat_id, cat_name in cat_map.items():
                    xml_str = xml_str.replace(f"__cat_{cat_id}__", cat_name)

                outputs.append(
                    FileOutput(
                        relative_path=Path(
                            f"annotations_voc/{split}/{tile_id}.xml"
                        ),
                        content=xml_str,
                    )
                )

        # ImageSets
        for split, ids in self._imagesets.items():
            content = "\n".join(str(i) for i in sorted(ids)) + "\n"
            outputs.append(
                FileOutput(
                    relative_path=Path(f"ImageSets/{split}.txt"),
                    content=content,
                )
            )

        return outputs
