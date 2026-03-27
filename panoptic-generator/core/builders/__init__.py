"""Annotation builder registry. New formats register via @register_builder decorator."""

from __future__ import annotations

from core.models import AnnotationFormat

BUILDER_REGISTRY: dict[AnnotationFormat, type] = {}


def register_builder(fmt: AnnotationFormat):
    """Decorator to register a builder class for an annotation format."""

    def decorator(cls: type) -> type:
        BUILDER_REGISTRY[fmt] = cls
        return cls

    return decorator


def create_builder(fmt: AnnotationFormat, **kwargs):
    """Instantiate a registered builder."""
    if fmt not in BUILDER_REGISTRY:
        raise ValueError(
            f"No builder registered for {fmt.value}. "
            f"Available: {[f.value for f in BUILDER_REGISTRY]}"
        )
    return BUILDER_REGISTRY[fmt](**kwargs)


# Import builder modules so they self-register
from core.builders import coco_panoptic as _cp  # noqa: E402, F401
from core.builders import coco_instance as _ci  # noqa: E402, F401
from core.builders import yolo as _yl  # noqa: E402, F401
from core.builders import pascal_voc as _pv  # noqa: E402, F401
