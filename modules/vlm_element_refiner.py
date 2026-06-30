"""VLM-based element type refinement.

This processor runs after SAM3 segmentation and before raster/vector processors.
It crops uncertain or frequently-confused element regions, asks a unified VLM
client for strict JSON, and maps the answer back to the project's standard
ElementInfo fields.
"""

import base64
import io
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import requests
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


class OpenAICompatibleVLMClient:
    """Small OpenAI-compatible chat-completions client for image classification."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        mode = self.config.get("mode", "api")
        self.base_url = (self.config.get("local_base_url") if mode == "local" else self.config.get("base_url")) or ""
        self.api_key = (self.config.get("local_api_key") if mode == "local" else self.config.get("api_key")) or ""
        self.model = (self.config.get("local_model") if mode == "local" else self.config.get("model")) or ""
        self.timeout = float(self.config.get("timeout", 60))
        self.max_tokens = int(self.config.get("max_tokens", 512))
        self.proxy = self.config.get("proxy") or None
        self.ca_cert_path = self.config.get("ca_cert_path", True)

    @property
    def available(self) -> bool:
        return bool(
            self.base_url
            and self.model
            and self.api_key
            and "YOUR_" not in self.api_key
            and "YOUR_" not in self.model
            and "YOUR_" not in self.base_url
        )

    def classify(self, image_data_url: str, prompt: str) -> Dict[str, Any]:
        if not self.available:
            raise RuntimeError("VLM client is not configured")

        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
        verify = self.ca_cert_path if self.ca_cert_path not in (False, "false", "False") else False
        response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout, proxies=proxies, verify=verify)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return _parse_json_object(content)


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
                    vlm_output = client.classify(crop_data_url, self._build_prompt(elem))
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


def _parse_json_object(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = str(content).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))
