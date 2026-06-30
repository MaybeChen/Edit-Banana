"""VLM-based element type refinement.

This processor runs after SAM3 segmentation and before raster/vector processors.
It crops uncertain or frequently-confused element regions, asks a unified VLM
client for strict JSON, and maps the answer back to the project's standard
ElementInfo fields.
"""

import base64
import io
import os
import re
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image

from .base import BaseProcessor, ProcessingContext
from .data_types import ElementInfo, ProcessingResult, get_layer_level


STANDARD_TYPES = {
    "icon",
    "picture",
    "rectangle",
    "rounded rectangle",
    "circle",
    "ellipse",
    "cylinder",
    "arrow",
    "connector",
    "container",
    "diamond",
    "triangle",
    "hexagon",
    "parallelogram",
    "cloud",
    "actor",
    "line",
    "text",
    "unknown",
}

TYPE_ALIASES = {
    "image": "picture",
    "photo": "picture",
    "bitmap": "picture",
    "rounded_rectangle": "rounded rectangle",
    "round rectangle": "rounded rectangle",
    "rounded rect": "rounded rectangle",
    "rect": "rectangle",
    "box": "rectangle",
    "panel": "container",
    "section_panel": "container",
    "title_bar": "container",
    "group": "container",
    "flowline": "connector",
    "connection": "connector",
    "edge": "connector",
    "arrow line": "arrow",
}

LINE_STYLE_ALIASES = {
    "dash": "dashed",
    "dashed": "dashed",
    "dot": "dotted",
    "dotted": "dotted",
    "solid": "solid",
    "none": None,
    "unknown": None,
}


from .vlm.client import OpenAICompatibleVLMClient
from .vlm.schemas import ELEMENT_CLASSIFICATION_SCHEMA


class VLMElementRefiner(BaseProcessor):
    """Refine SAM3 element classes with a VLM for uncertain/confusing regions."""

    CONFUSING_TYPES = {
        "unknown", "image", "picture", "icon", "shape", "rectangle", "rounded_rectangle",
        "rounded rectangle", "circle", "ellipse", "cylinder", "arrow", "connector", "line",
        "section_panel", "title_bar", "container",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(None)
        self.vlm_config = config or {}
        self.low_confidence_threshold = float(self.vlm_config.get("low_confidence_threshold", 0.65))
        self.large_container_area_ratio = float(self.vlm_config.get("large_container_area_ratio", 0.18))
        self.max_elements = int(self.vlm_config.get("max_elements", 40))
        self.crop_padding = int(self.vlm_config.get("crop_padding", 8))

    def process(self, context: ProcessingContext) -> ProcessingResult:
        self._log("Refining SAM3 element types with VLM")
        if not context.image_path or not os.path.exists(context.image_path):
            return ProcessingResult(success=False, elements=context.elements, error_message="Invalid image path")

        client = self._get_vlm_client(context)
        if client is None:
            self._log("Skipped: VLM client is not configured")
            return ProcessingResult(success=True, elements=context.elements, canvas_width=context.canvas_width, canvas_height=context.canvas_height, metadata={"processed_count": 0, "skipped_reason": "vlm_not_configured"})

        with Image.open(context.image_path) as img:
            image = img.convert("RGB")
            canvas_area = max(1, (context.canvas_width or image.width) * (context.canvas_height or image.height))
            targets = self._select_targets(context.elements, canvas_area)[: self.max_elements]
            processed = 0
            updated = 0
            for elem in targets:
                try:
                    crop_data_url = self._crop_as_data_url(image, elem)
                    vlm_output = client.classify(crop_data_url, self._build_prompt(elem), ELEMENT_CLASSIFICATION_SCHEMA)
                    if self._apply_vlm_output(elem, vlm_output):
                        updated += 1
                    processed += 1
                except Exception as exc:
                    elem.processing_notes.append(f"VLM refine failed: {exc}")
                    self._log(f"Element {elem.id} VLM refine failed: {exc}")

        self._log(f"Done: processed={processed}, updated={updated}")
        return ProcessingResult(success=True, elements=context.elements, canvas_width=context.canvas_width, canvas_height=context.canvas_height, metadata={"processed_count": processed, "updated_count": updated})

    def _get_vlm_client(self, context: ProcessingContext):
        shared = context.shared_models.get("vlm_client")
        if shared is not None:
            return shared
        client = OpenAICompatibleVLMClient(self.vlm_config)
        return client if client.available else None

    def _select_targets(self, elements: Iterable[ElementInfo], canvas_area: int) -> List[ElementInfo]:
        selected = []
        for elem in elements:
            elem_type = _normalize_type(elem.element_type)
            area_ratio = elem.bbox.area / canvas_area
            low_conf = elem.score and elem.score < self.low_confidence_threshold
            confusing = elem_type in self.CONFUSING_TYPES
            large_container = area_ratio >= self.large_container_area_ratio or elem_type in {"container", "section_panel", "title_bar"}
            if low_conf or confusing or large_container:
                selected.append(elem)
        selected.sort(key=lambda e: (not (e.score and e.score < self.low_confidence_threshold), -e.bbox.area))
        return selected

    def _crop_as_data_url(self, image: Image.Image, elem: ElementInfo) -> str:
        x1 = max(0, int(elem.bbox.x1) - self.crop_padding)
        y1 = max(0, int(elem.bbox.y1) - self.crop_padding)
        x2 = min(image.width, int(elem.bbox.x2) + self.crop_padding)
        y2 = min(image.height, int(elem.bbox.y2) + self.crop_padding)
        crop = image.crop((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _build_prompt(self, elem: ElementInfo) -> str:
        return (
            "Classify the cropped diagram/UI element. Return STRICT JSON only, no markdown. "
            "Schema: {\"element_type\": string, \"line_style\": string|null, \"confidence\": number, \"reason\": string}. "
            "element_type must be one of: icon, picture, rectangle, rounded rectangle, circle, ellipse, "
            "cylinder, arrow, connector, container, diamond, triangle, hexagon, parallelogram, cloud, actor, line, text, unknown. "
            "line_style must be solid, dashed, dotted, or null. Prefer connector for lines without arrowheads; "
            "prefer container for large grouping panels/cards; prefer picture for photo-like raster content; prefer icon for small symbolic graphics. "
            f"Current SAM3 type: {elem.element_type}; bbox: {elem.bbox.to_list()}; score: {elem.score}."
        )

    def _apply_vlm_output(self, elem: ElementInfo, output: Dict[str, Any]) -> bool:
        new_type = _normalize_type(str(output.get("element_type", "unknown")))
        if new_type not in STANDARD_TYPES or new_type == "unknown":
            return False
        old_type = elem.element_type
        changed = new_type != _normalize_type(old_type)
        elem.element_type = new_type
        elem.layer_level = get_layer_level(new_type)
        line_style = _normalize_line_style(output.get("line_style"))
        if line_style:
            elem.line_style = line_style
        reason = str(output.get("reason", "")).strip()
        conf = output.get("confidence")
        note = f"VLM refined type: {old_type} -> {new_type}"
        if line_style:
            note += f", line_style={line_style}"
        if conf is not None:
            note += f", confidence={conf}"
        if reason:
            note += f" ({reason[:160]})"
        elem.processing_notes.append(note)
        return changed or bool(line_style)


def _normalize_type(value: str) -> str:
    value = (value or "unknown").strip().lower().replace("-", "_")
    value = re.sub(r"\s+", " ", value).replace("_", " ")
    return TYPE_ALIASES.get(value, value)


def _normalize_line_style(value: Any) -> Optional[str]:
    if value is None:
        return None
    return LINE_STYLE_ALIASES.get(str(value).strip().lower())
