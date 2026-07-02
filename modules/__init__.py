"""
Pipeline modules: segmentation, text extraction, shape/arrow handling, XML merge.
See project README for pipeline overview and config.
"""

import importlib
import warnings

from .base import BaseProcessor, ProcessingContext
from .data_types import (
    ElementInfo,
    BoundingBox,
    ProcessingResult,
    XMLFragment,
    LayerLevel,
    get_layer_level,
)

_LAZY_EXPORTS = {
    "XMLMerger": (".xml_merger", "XMLMerger"),
    "IconPictureProcessor": (".icon_picture_processor", "IconPictureProcessor"),
    "BasicShapeProcessor": (".basic_shape_processor", "BasicShapeProcessor"),
    "MetricEvaluator": (".metric_evaluator", "MetricEvaluator"),
    "RefinementProcessor": (".refinement_processor", "RefinementProcessor"),
    "VLMElementRefiner": (".vlm_element_refiner", "VLMElementRefiner"),
    "VLMLayoutRefiner": (".vlm_layout_refiner", "VLMLayoutRefiner"),
    "VLMExportValidator": (".vlm_export_validator", "VLMExportValidator"),
    "TextRestorer": (".text.restorer", "TextRestorer"),
}


def __getattr__(name: str):
    """Lazily import heavy optional processors only when requested."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    try:
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
    except Exception as exc:
        warnings.warn(f"{name} unavailable (missing deps): {exc}.")
        value = None
    globals()[name] = value
    return value


__all__ = [
    "BaseProcessor",
    "ProcessingContext",
    "ElementInfo",
    "BoundingBox",
    "ProcessingResult",
    "XMLFragment",
    "LayerLevel",
    "get_layer_level",
    "TextRestorer",
    "XMLMerger",
    "IconPictureProcessor",
    "BasicShapeProcessor",
    "MetricEvaluator",
    "RefinementProcessor",
    "VLMElementRefiner",
    "VLMLayoutRefiner",
    "VLMExportValidator",
]
