"""Optional VLM enhancement passes for the image-to-diagram pipeline.

The enhancer is deliberately conservative: every VLM call is behind config
switches and failures fall back to the existing OCR/SAM3/CV pipeline.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ProcessingContext
from .data_types import BoundingBox, ElementInfo
from .sam3_info_extractor import PromptGroup
from .vlm_client import create_vlm_client_from_config
from prompts.vlm_structure import VLM_CONNECTOR_PROMPT, VLM_PAGE_REGIONS_PROMPT, VLM_REGION_ELEMENTS_PROMPT, VLM_STRUCTURE_PROMPT


CANONICAL_TYPES = {
    "icon",
    "picture",
    "logo",
    "chart",
    "rectangle",
    "rounded rectangle",
    "rounded_rectangle",
    "circle",
    "ellipse",
    "cylinder",
    "diamond",
    "triangle",
    "hexagon",
    "container",
    "arrow",
    "connector",
    "line",
    "text",
}




class VLMStructureMixin:
    def recognize_region_elements(self, context: ProcessingContext, regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stage 2: recognize elements inside known regions."""
        ocr_context = self._summarize_ocr_context(context, max_blocks=self.vlm_only_ocr_anchor_max_blocks)
        compact_regions = self._summarize_vlm_regions(regions, max_items=self.vlm_only_region_max_items)
        prompt = (
            VLM_REGION_ELEMENTS_PROMPT
            + "\n\n已识别粗区域 JSON："
            + json.dumps(compact_regions, ensure_ascii=False, separators=(",", ":"))
            + "\nOCR文本锚点 JSON（如果存在，text 元素应优先复用这些文字和位置；列表可能已截断）："
            + json.dumps(ocr_context, ensure_ascii=False, separators=(",", ":"))
            + "\n只输出 {\"elements\":[...]}，不要复述输入 JSON。"
        )
        try:
            data = self._parse_json_response_with_debug(
                self.client.analyze_image(context.image_path, prompt),
                "vlm_region_elements",
            )
        except Exception as exc:
            print(f"[VLMEnhancer] VLM region element recognition skipped: {exc}", flush=True)
            return {"recognized": False, "items": [], "error": str(exc)}
        items = self._vlm_structure_items(data)
        result = {"recognized": bool(items), "items": items, "raw_count": len(items)}
        self._print_json("vlm_region_elements recognized", {"items": items})
        self._write_artifact(context, "vlm_region_elements.json", result)
        return result

    def recognize_connectors(self, context: ProcessingContext) -> Dict[str, Any]:
        """Stage 4: connector-only recognition for VLM-only and staged flows."""
        if not context.elements:
            return {"updated": 0, "added": 0, "edges": [], "changes": []}
        threshold = float(self.thresholds.get("vlm_connector_confidence", self.thresholds.get("layout_refine_confidence", 0.70)))
        prompt = (
            VLM_CONNECTOR_PROMPT
            + "已有元素列表："
            + json.dumps(self._summarize_elements(context.elements), ensure_ascii=False)
        )
        try:
            data = self._parse_json_response_with_debug(
                self.client.analyze_image(context.image_path, prompt),
                "vlm_connectors",
            )
        except Exception as exc:
            print(f"[VLMEnhancer] VLM connector recognition skipped: {exc}", flush=True)
            return {"updated": 0, "added": 0, "edges": [], "changes": [], "error": str(exc)}
        edges = data.get("edges", []) if isinstance(data.get("edges"), list) else data.get("elements", [])
        if not isinstance(edges, list):
            edges = []
        by_id = {elem.id: elem for elem in context.elements}
        next_id = max(by_id.keys(), default=-1) + 1
        changes = []
        added = []
        for edge in edges:
            if not isinstance(edge, dict) or self._confidence(edge) < threshold:
                continue
            elem = by_id.get(edge.get("id"))
            if elem is None:
                bbox = self._extract_vlm_bbox(edge)
                normalized_bbox = self._normalize_bbox(bbox, 1000, 1000) if bbox is not None else None
                if normalized_bbox is None:
                    continue
                elem = ElementInfo(
                    id=next_id,
                    element_type=self._sanitize_type(edge.get("type")) or "connector",
                    bbox=BoundingBox.from_list(self._scale_normalized_bbox(normalized_bbox, context.canvas_width, context.canvas_height)),
                    score=self._confidence(edge),
                    source_prompt="vlm_connectors",
                )
                next_id += 1
                context.elements.append(elem)
                by_id[elem.id] = elem
                added.append(elem.id)
            before = elem.to_dict()
            self._apply_vlm_structure_attributes(elem, edge, context.canvas_width, context.canvas_height)
            new_type = self._sanitize_type(edge.get("type"))
            if new_type in {"arrow", "connector", "line"}:
                elem.element_type = new_type
            if edge.get("source_id") is not None:
                elem.source_id = int(edge.get("source_id"))
            if edge.get("target_id") is not None:
                elem.target_id = int(edge.get("target_id"))
            elem.processing_notes.append("vlm_connectors")
            after = elem.to_dict()
            if before != after:
                changes.append({"id": elem.id, "from": before, "to": after, "confidence": self._confidence(edge)})
        result = {"updated": len(changes), "added": len(added), "edges": edges, "changes": changes, "added_ids": added}
        self._print_json("vlm_connectors applied", result)
        self._write_artifact(context, "vlm_connectors.json", result)
        return result

    def recognize_structure(self, context: ProcessingContext) -> Dict[str, Any]:
        """Ask VLM to recognize the editable page structure without SAM3 candidates."""
        if not self.enabled or self.client is None:
            return {"recognized": False, "elements": [], "error": "multimodal VLM is disabled"}
        threshold = float(self.thresholds.get("vlm_structure_confidence", 0.60))
        prompt = VLM_STRUCTURE_PROMPT
        try:
            data = self._parse_json_response_with_debug(self.client.analyze_image(context.image_path, prompt), "vlm_structure")
        except Exception as exc:
            print(f"[VLMEnhancer] VLM-only structure recognition skipped: {exc}", flush=True)
            return {"recognized": False, "elements": [], "error": str(exc)}
        if not data:
            return {"recognized": False, "elements": [], "error": "empty or invalid VLM JSON response"}

        items = self._vlm_structure_items(data)
        elements: List[ElementInfo] = []
        for item in items:
            if not isinstance(item, dict) or self._confidence(item) < threshold:
                continue
            new_type = self._vlm_structure_element_type(item)
            bbox = self._extract_vlm_bbox(item)
            if not new_type or bbox is None:
                continue
            normalized_bbox = self._normalize_bbox(bbox, 1000, 1000)
            if normalized_bbox is None:
                continue
            pixel_bbox = self._scale_normalized_bbox(normalized_bbox, context.canvas_width, context.canvas_height)
            elem = ElementInfo(
                id=len(elements),
                element_type=new_type,
                bbox=BoundingBox.from_list(pixel_bbox),
                score=self._confidence(item),
                source_prompt="vlm_structure",
            )
            semantic_type = item.get("semantic_type") or item.get("original_type") or item.get("type") or item.get("subtype")
            elem.semantic_type = str(semantic_type).strip() if semantic_type else None
            strategy = item.get("reconstruction_strategy") or item.get("strategy")
            elem.reconstruction_strategy = str(strategy).strip() if strategy else None
            setattr(elem, "vlm_item", item)
            self._apply_vlm_structure_attributes(elem, item, context.canvas_width, context.canvas_height)
            elem.processing_notes.append("vlm_structure")
            elements.append(elem)

        context.elements = elements
        result = {
            "recognized": True,
            "count": len(elements),
            "raw_count": len(items),
            "dropped_count": max(0, len(items) - len(elements)),
            "items": items,
            "elements": [elem.to_dict() for elem in elements],
        }
        self._print_json("vlm_structure recognized", result["elements"])
        self._write_artifact(context, "vlm_structure.json", result)
        return result

    def _build_vlm_elements_from_items(
        self,
        items: List[Dict[str, Any]],
        context: ProcessingContext,
        source_prompt: str = "vlm_structure",
    ) -> List[ElementInfo]:
        """Convert VLM JSON items into ElementInfo objects with semantic metadata."""
        threshold = float(self.thresholds.get("vlm_structure_confidence", 0.60))
        elements: List[ElementInfo] = []
        for item in items:
            if not isinstance(item, dict) or self._confidence(item) < threshold:
                continue
            new_type = self._vlm_structure_element_type(item)
            bbox = self._extract_vlm_bbox(item)
            if not new_type or bbox is None:
                continue
            normalized_bbox = self._normalize_bbox(bbox, 1000, 1000)
            if normalized_bbox is None:
                continue
            elem = ElementInfo(
                id=len(elements),
                element_type=new_type,
                bbox=BoundingBox.from_list(self._scale_normalized_bbox(normalized_bbox, context.canvas_width, context.canvas_height)),
                score=self._confidence(item),
                source_prompt=source_prompt,
            )
            semantic_type = (
                item.get("semantic_type")
                or item.get("original_type")
                or item.get("type")
                or item.get("subtype")
            )
            elem.semantic_type = str(semantic_type).strip() if semantic_type else None
            strategy = item.get("reconstruction_strategy") or item.get("strategy")
            elem.reconstruction_strategy = str(strategy).strip() if strategy else None
            setattr(elem, "vlm_item", item)
            self._apply_vlm_structure_attributes(elem, item, context.canvas_width, context.canvas_height)
            elem.processing_notes.append(source_prompt)
            elements.append(elem)
        return elements

    @staticmethod
    def _vlm_structure_items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Accept common VLM wrappers so valid results are not silently ignored."""
        background = data.get("background")
        prefix = []
        if isinstance(background, dict):
            background_item = dict(background)
            background_item.setdefault("type", "background")
            background_item.setdefault("bbox", {"x": 0, "y": 0, "width": 1000, "height": 1000})
            background_item.setdefault("confidence", 1.0)
            prefix.append(background_item)
        for key in ("elements", "page_elements", "objects", "shapes", "items"):
            values = data.get(key)
            items = VLMEnhancer._coerce_vlm_item_collection(values)
            if items:
                return prefix + items
        page = data.get("page") or data.get("diagram") or data.get("structure")
        if isinstance(page, dict):
            for key in ("elements", "page_elements", "objects", "shapes", "items"):
                values = page.get(key)
                items = VLMEnhancer._coerce_vlm_item_collection(values)
                if items:
                    return prefix + items
        return prefix

    @staticmethod
    def _coerce_vlm_item_collection(values: Any) -> List[Dict[str, Any]]:
        """Accept list or id-keyed dict element collections from VLM output."""
        if isinstance(values, list):
            return [item for item in values if isinstance(item, dict)]
        if isinstance(values, dict):
            items = []
            for key, value in values.items():
                if not isinstance(value, dict):
                    continue
                item = dict(value)
                item.setdefault("id", key)
                items.append(item)
            return items
        return []

    def _vlm_structure_element_type(self, item: Dict[str, Any]) -> Optional[str]:
        raw_type = item.get("type") or item.get("element_type") or item.get("shape_type") or item.get("kind")
        subtype = str(item.get("subtype") or item.get("shape_type") or "").strip().lower().replace("_", " ")
        direct = self._sanitize_type(raw_type)
        if direct and direct != "text":
            return direct

        normalized = str(raw_type or "").strip().lower().replace("_", " ")
        if normalized in {
            "background",
            "container",
            "card",
            "panel",
            "section",
            "header",
            "footer",
            "sidebar",
            "main content",
            "main_content",
            "container group",
            "container_group",
            "card group",
            "card_group",
            "table region",
            "table_region",
            "diagram region",
            "diagram_region",
            "complex visual region",
            "complex_visual_region",
        }:
            return "container"
        if normalized in {"image region", "image_region"}:
            return "picture"
        if normalized in {"icon logo region", "icon_logo_region"}:
            return "logo"
        if normalized in {"chart region", "chart_region"}:
            return "chart"
        if normalized == "shape":
            shape_aliases = {
                "rounded rectangle": "rounded rectangle",
                "circle": "circle",
                "ellipse": "ellipse",
                "triangle": "triangle",
                "diamond": "diamond",
                "hexagon": "hexagon",
                "pill": "rounded rectangle",
                "parallelogram": "parallelogram",
            }
            return shape_aliases.get(subtype, "rectangle")
        if normalized in {"line", "arrow", "icon", "logo", "chart"}:
            return normalized
        if normalized in {"image", "complex visual", "complex_visual", "unknown"}:
            return "picture"
        if normalized in {"table", "diagram", "decoration", "group"}:
            return "container" if normalized in {"table", "diagram", "group"} else "rectangle"
        if normalized == "text":
            return "text"
        return None

    @staticmethod
    def _extract_vlm_bbox(item: Dict[str, Any]) -> Optional[List[Any]]:
        """Normalize common bbox/geometry formats returned by VLMs."""
        bbox = item.get("bbox") or item.get("bounding_box") or item.get("box")
        if isinstance(bbox, list) and len(bbox) == 4:
            return bbox
        if isinstance(bbox, dict):
            return VLMEnhancer._bbox_from_dict(bbox)
        geometry = item.get("geometry") or item.get("position") or item.get("rect")
        if isinstance(geometry, dict):
            return VLMEnhancer._bbox_from_dict(geometry)
        coords = [item.get(key) for key in ("x1", "y1", "x2", "y2")]
        if all(value is not None for value in coords):
            return coords
        xywh = [item.get(key) for key in ("x", "y", "width", "height")]
        if all(value is not None for value in xywh):
            try:
                x, y, width, height = [float(value) for value in xywh]
            except (TypeError, ValueError):
                return None
            return [x, y, x + width, y + height]
        return None

    @staticmethod
    def _scale_normalized_bbox(bbox: List[int], canvas_width: int, canvas_height: int) -> List[int]:
        width = max(1, int(canvas_width or 1000))
        height = max(1, int(canvas_height or 1000))
        x1, y1, x2, y2 = bbox
        return [
            int(round(x1 * width / 1000)),
            int(round(y1 * height / 1000)),
            int(round(x2 * width / 1000)),
            int(round(y2 * height / 1000)),
        ]

    @staticmethod
    def _bbox_from_dict(data: Dict[str, Any]) -> Optional[List[Any]]:
        if all(key in data for key in ("x1", "y1", "x2", "y2")):
            return [data["x1"], data["y1"], data["x2"], data["y2"]]
        if all(key in data for key in ("left", "top", "right", "bottom")):
            return [data["left"], data["top"], data["right"], data["bottom"]]
        if all(key in data for key in ("x", "y", "width", "height")):
            try:
                x = float(data["x"])
                y = float(data["y"])
                width = float(data["width"])
                height = float(data["height"])
            except (TypeError, ValueError):
                return None
            return [x, y, x + width, y + height]
        return None

    @staticmethod
    def _normalize_bbox(bbox: List[Any], canvas_width: int, canvas_height: int) -> Optional[List[int]]:
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        except (TypeError, ValueError):
            return None
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        max_x = max(1, int(canvas_width or max(x2, 1)))
        max_y = max(1, int(canvas_height or max(y2, 1)))
        x1 = max(0, min(max_x, x1))
        x2 = max(0, min(max_x, x2))
        y1 = max(0, min(max_y, y1))
        y2 = max(0, min(max_y, y2))
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            return None
        return [x1, y1, x2, y2]

    def _apply_vlm_structure_attributes(
        self,
        elem: ElementInfo,
        item: Dict[str, Any],
        canvas_width: int = 1000,
        canvas_height: int = 1000,
    ) -> None:
        semantic_type = item.get("semantic_type") or item.get("original_type") or item.get("subtype")
        if semantic_type:
            elem.semantic_type = str(semantic_type).strip()
        strategy = item.get("reconstruction_strategy") or item.get("strategy")
        if strategy:
            elem.reconstruction_strategy = str(strategy).strip()
        style = item.get("style") if isinstance(item.get("style"), dict) else {}
        text_style_keys = {
            "content",
            "text",
            "label",
            "value",
            "font_family",
            "font_size",
            "font_size_estimate",
            "font_weight",
            "font_style",
            "font_color",
            "text_align",
            "vertical_align",
        }
        if str(elem.element_type or "").lower() == "text":
            flattened = dict(getattr(elem, "vlm_item", {}) or item)
            for key in text_style_keys:
                if key not in flattened and key in item:
                    flattened[key] = item.get(key)
                if key not in flattened and key in style:
                    flattened[key] = style.get(key)
            setattr(elem, "vlm_item", flattened)
        stroke_value = item.get("stroke_color") or item.get("stroke") or style.get("stroke_color") or style.get("stroke")
        stroke_style = stroke_value if isinstance(stroke_value, dict) else {}
        line_style = self._normalize_line_style(
            item.get("line_style")
            or style.get("line_style")
            or style.get("dash")
            or stroke_style.get("dash")
        )
        if line_style:
            elem.line_style = line_style
        arrow_heads = self._normalize_arrow_heads(item.get("arrow_heads") or item.get("end_arrow") or item.get("arrowhead"))
        if not arrow_heads:
            arrow_heads = self._arrow_heads_from_head_tail(item.get("arrow_head"), item.get("arrow_tail"))
        if arrow_heads:
            elem.arrow_heads = arrow_heads
        arrow_style = str(item.get("arrow_style") or item.get("curve") or "").strip().lower()
        if arrow_style in {"curved", "curve", "arc"} or item.get("is_curved") is True:
            elem.arrow_style = "curved"
        fill_value = item.get("fill_color") or item.get("fill") or style.get("fill_color") or style.get("fill")
        fill = self._valid_hex_color(self._color_value(fill_value))
        stroke = self._valid_hex_color(self._color_value(stroke_value))
        if fill:
            elem.fill_color = fill
        if str(fill_value or "").strip().lower() == "none":
            elem.fill_color = "none"
        if stroke:
            elem.stroke_color = stroke
        try:
            stroke_width = item.get("stroke_width", style.get("stroke_width"))
            if stroke_width is None and isinstance(stroke_value, dict):
                stroke_width = stroke_value.get("width")
            if stroke_width is not None:
                elem.stroke_width = max(1, min(12, int(round(float(stroke_width)))))
        except (TypeError, ValueError):
            pass
        point_aliases = (
            ("arrow_start", "start_point", "start"),
            ("arrow_end", "end_point", "end"),
        )
        for field_name, primary, fallback in point_aliases:
            point = item.get(primary) or item.get(fallback)
            normalized_point = self._normalize_point(point)
            if normalized_point is not None:
                setattr(
                    elem,
                    field_name,
                    (
                        normalized_point[0] * max(1, int(canvas_width or 1000)) / 1000,
                        normalized_point[1] * max(1, int(canvas_height or 1000)) / 1000,
                    ),
                )

    @staticmethod
    def _normalize_point(point: Any) -> Optional[tuple]:
        if isinstance(point, dict):
            point = [point.get("x"), point.get("y")]
        if isinstance(point, list) and len(point) >= 2:
            try:
                x = max(0.0, min(1000.0, float(point[0])))
                y = max(0.0, min(1000.0, float(point[1])))
                return x, y
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _color_value(value: Any) -> Any:
        if isinstance(value, dict):
            return value.get("color")
        return value

    @staticmethod
    def _arrow_heads_from_head_tail(head: Any, tail: Any) -> Optional[str]:
        head_text = str(head or "").strip().lower()
        tail_text = str(tail or "").strip().lower()
        has_head = head_text not in {"", "none", "null", "no"}
        has_tail = tail_text not in {"", "none", "null", "no"}
        if has_head and has_tail:
            return "both"
        if has_head:
            return "end"
        if has_tail:
            return "start"
        return None

