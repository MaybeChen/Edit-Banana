"""VLM-based refinement for segmentation element classifications.

The refiner crops each selected segmentation element from the source image, sends the
local crop plus segmentation's current candidate type to the shared VLM client, and only
commits the VLM result when the returned confidence is above the configured
threshold.  It writes the normalized standard element type and line style back to
``ElementInfo`` for downstream processors.
"""

from __future__ import annotations

import base64
import io
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

from PIL import Image

from ..base import BaseProcessor, ProcessingContext
from ..data_types import ElementInfo, ProcessingResult, get_layer_level
from .client import OpenAICompatibleVLMClient
from .schemas import ELEMENT_CLASSIFICATION_SCHEMA

STANDARD_TYPES = {
    "rectangle",
    "rounded_rectangle",
    "diamond",
    "ellipse",
    "circle",
    "triangle",
    "hexagon",
    "parallelogram",
    "cylinder",
    "cloud",
    "actor",
    "title_bar",
    "section_panel",
    "icon",
    "picture",
    "logo",
    "chart",
    "function_graph",
    "arrow",
    "line",
    "connector",
    "text",
    "unknown",
}

TYPE_ALIASES = {
    "bitmap": "picture",
    "container": "section_panel",
    "edge": "connector",
    "flowline": "connector",
    "image": "picture",
    "panel": "section_panel",
    "photo": "picture",
    "round_rectangle": "rounded_rectangle",
    "rounded_rectangle": "rounded_rectangle",
    "rounded_rect": "rounded_rectangle",
    "shape": "rectangle",
    "arrow_line": "arrow",
    "connection": "connector",
}

LINE_STYLE_ALIASES = {
    "dash": "dashed",
    "dashed": "dashed",
    "dot": "dotted",
    "dotted": "dotted",
    "solid": "solid",
    "none": None,
    "null": None,
    "unknown": None,
    "": None,
}


class VLMElementRefiner(BaseProcessor):
    """Refine segmentation element types with localized VLM classification."""

    CONFUSING_TYPES = {
        "unknown",
        "image",
        "picture",
        "icon",
        "shape",
        "rectangle",
        "rounded_rectangle",
        "circle",
        "ellipse",
        "cylinder",
        "arrow",
        "connector",
        "line",
        "section_panel",
        "title_bar",
        "container",
    }

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        super().__init__(None)
        self.vlm_config = dict(config or {})
        self.low_confidence_threshold = float(self.vlm_config.get("low_confidence_threshold", 0.65))
        self.override_confidence_threshold = float(
            self.vlm_config.get("override_confidence_threshold", self.vlm_config.get("confidence_threshold", 0.75))
        )
        self.large_container_area_ratio = float(self.vlm_config.get("large_container_area_ratio", 0.18))
        self.max_elements = int(self.vlm_config.get("max_elements", 40))
        self.crop_padding = int(self.vlm_config.get("crop_padding", 8))
        self.use_masked_crop = bool(self.vlm_config.get("use_masked_crop", True))

    def process(self, context: ProcessingContext) -> ProcessingResult:
        self._log("Refining segmentation element types with VLM")
        if not context.image_path or not os.path.exists(context.image_path):
            return ProcessingResult(success=False, elements=context.elements, error_message="Invalid image path")

        client = self._get_vlm_client(context)
        if client is None:
            self._log("Skipped: VLM client is not configured")
            return ProcessingResult(
                success=True,
                elements=context.elements,
                canvas_width=context.canvas_width,
                canvas_height=context.canvas_height,
                metadata={"processed_count": 0, "updated_count": 0, "skipped_reason": "vlm_not_configured"},
            )

        processed = 0
        updated = 0
        with Image.open(context.image_path) as img:
            image = img.convert("RGB")
            canvas_width = context.canvas_width or image.width
            canvas_height = context.canvas_height or image.height
            canvas_area = max(1, canvas_width * canvas_height)
            targets = self._select_targets(context.elements, canvas_area)[: self.max_elements]
            for elem in targets:
                try:
                    crop_data_url = self._crop_as_data_url(image, elem)
                    output = client.classify(crop_data_url, self._build_prompt(elem), ELEMENT_CLASSIFICATION_SCHEMA)
                    if self._apply_vlm_output(elem, output):
                        updated += 1
                    processed += 1
                except Exception as exc:
                    elem.processing_notes.append(f"VLM refine failed: {exc}")
                    self._log(f"Element {elem.id} VLM refine failed: {exc}")

        self._log(f"Done: processed={processed}, updated={updated}")
        return ProcessingResult(
            success=True,
            elements=context.elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={"processed_count": processed, "updated_count": updated, "confidence_threshold": self.override_confidence_threshold},
        )

    def _get_vlm_client(self, context: ProcessingContext) -> Optional[OpenAICompatibleVLMClient]:
        shared = context.shared_models.get("vlm_client")
        if shared is not None:
            return shared
        client = OpenAICompatibleVLMClient(self.vlm_config)
        return client if client.available else None

    def _select_targets(self, elements: Iterable[ElementInfo], canvas_area: int) -> List[ElementInfo]:
        selected: List[ElementInfo] = []
        for elem in elements:
            elem_type = _normalize_type(elem.element_type)
            area_ratio = elem.bbox.area / canvas_area
            low_conf = elem.score > 0 and elem.score < self.low_confidence_threshold
            confusing = elem_type in self.CONFUSING_TYPES
            large_container = area_ratio >= self.large_container_area_ratio or elem_type in {"section_panel", "title_bar"}
            if low_conf or confusing or large_container:
                selected.append(elem)
        selected.sort(key=lambda e: (not (e.score > 0 and e.score < self.low_confidence_threshold), -e.bbox.area))
        return selected

    def _crop_as_data_url(self, image: Image.Image, elem: ElementInfo) -> str:
        x1 = max(0, int(elem.bbox.x1) - self.crop_padding)
        y1 = max(0, int(elem.bbox.y1) - self.crop_padding)
        x2 = min(image.width, int(elem.bbox.x2) + self.crop_padding)
        y2 = min(image.height, int(elem.bbox.y2) + self.crop_padding)
        crop = image.crop((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)))
        if self.use_masked_crop and elem.mask is not None:
            crop = _apply_mask_to_crop(crop, elem.mask, (x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _build_prompt(self, elem: ElementInfo) -> str:
        standard_types = ", ".join(sorted(STANDARD_TYPES))
        return (
            "Classify this cropped element from a diagram/UI screenshot. "
            "Use the current segmentation candidate type as context, but correct it when the crop shows another standard type. "
            "Return STRICT JSON only with schema: "
            '{"element_type": string, "line_style": string|null, "confidence": number, "reason": string}. '
            f"element_type must be one of: {standard_types}. "
            "line_style must be solid, dashed, dotted, or null. "
            "Prefer connector for a line without arrowheads; arrow for a directed line; "
            "section_panel/title_bar for grouping panels; picture for photo-like raster content; icon for symbolic graphics. "
            f"Current segmentation candidate type: {elem.element_type}; bbox: {elem.bbox.to_list()}; segmentation score: {elem.score}."
        )

    def _apply_vlm_output(self, elem: ElementInfo, output: Dict[str, Any]) -> bool:
        confidence = _safe_float(output.get("confidence"), 0.0)
        new_type = _normalize_type(str(output.get("element_type", "unknown")))
        line_style = _normalize_line_style(output.get("line_style"))
        reason = str(output.get("reason", "")).strip()

        if confidence < self.override_confidence_threshold:
            elem.processing_notes.append(
                f"VLM refine skipped: confidence={confidence:.3f} below threshold={self.override_confidence_threshold:.3f}"
            )
            return False
        if new_type not in STANDARD_TYPES or new_type == "unknown":
            elem.processing_notes.append(f"VLM refine skipped: invalid type={new_type!r}, confidence={confidence:.3f}")
            return False

        old_type = elem.element_type
        old_line_style = elem.line_style
        elem.element_type = new_type
        elem.layer_level = get_layer_level(new_type)
        elem.line_style = line_style

        note = f"VLM refined type: {old_type} -> {new_type}, confidence={confidence:.3f}"
        if old_line_style != line_style:
            note += f", line_style={old_line_style} -> {line_style}"
        if reason:
            note += f" ({reason[:160]})"
        elem.processing_notes.append(note)
        return old_type != new_type or old_line_style != line_style


def _normalize_type(value: str) -> str:
    normalized = (value or "unknown").strip().lower().replace("-", "_")
    normalized = re.sub(r"\s+", " ", normalized).replace(" ", "_")
    normalized = TYPE_ALIASES.get(normalized, normalized)
    return normalized


def _normalize_line_style(value: Any) -> Optional[str]:
    if value is None:
        return None
    return LINE_STYLE_ALIASES.get(str(value).strip().lower(), None)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _apply_mask_to_crop(crop: Image.Image, mask: Any, crop_box: tuple[int, int, int, int]) -> Image.Image:
    """White out pixels outside the segmentation mask when mask dimensions are usable."""
    try:
        import numpy as np

        x1, y1, x2, y2 = crop_box
        mask_array = np.asarray(mask)
        if mask_array.ndim > 2:
            mask_array = mask_array.squeeze()
        mask_crop = mask_array[y1:y2, x1:x2]
        if mask_crop.shape[:2] != (crop.height, crop.width):
            return crop
        crop_array = np.asarray(crop).copy()
        crop_array[mask_crop <= 0] = 255
        return Image.fromarray(crop_array)
    except Exception:
        return crop
