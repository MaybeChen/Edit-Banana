"""Optional VLM enhancement passes for the image-to-diagram pipeline.

The enhancer is deliberately conservative: every VLM call is behind config
switches and failures fall back to the existing OCR/CV pipeline.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .base import ProcessingContext
from .vlm_client import create_vlm_client_from_config
from prompts.vlm_structure import build_vlm_page_regions_prompt

from .vlm_enhancer_core import VLMCoreMixin
from .vlm_enhancer_structure import VLMStructureMixin
from .vlm_enhancer_refinement import VLMRefinementMixin
from .vlm_enhancer_text import VLMTextPromptMixin

PAGE_REGION_TYPES = {
    "background",
    "header",
    "footer",
    "sidebar",
    "main_content",
    "container_group",
    "card_group",
    "image_region",
    "icon_logo_region",
    "table_region",
    "chart_region",
    "diagram_region",
    "complex_visual_region",
}

PAGE_REGION_TYPE_ALIASES = {
    "title": "header",
    "page_title": "header",
    "top_bar": "header",
    "left_panel": "sidebar",
    "left_sidebar": "sidebar",
    "navigation": "sidebar",
    "body": "main_content",
    "content": "main_content",
    "center": "main_content",
    "canvas": "diagram_region",
    "diagram": "diagram_region",
    "flowchart": "diagram_region",
    "agent_diagram": "diagram_region",
    "card_list": "card_group",
    "cards": "card_group",
    "steps": "card_group",
    "key_issues": "card_group",
    "requirements": "card_group",
    "summary": "footer",
    "bottom_bar": "footer",
}


class VLMEnhancer(VLMCoreMixin, VLMTextPromptMixin, VLMStructureMixin, VLMRefinementMixin):
    """Runs optional VLM prompt, element, region, layout, and export passes."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.root_config = config or {}
        self.config = (self.root_config.get("multimodal") or {})
        self.enabled = bool(self.config.get("enabled", False))
        default_use_for = {
            "prompt_planning": False,
            "text_style": True,
            "segmentation_refine": True,
            "element_refine": False,
            "region_refine": False,
            "layout_refine": False,
            "element_attributes": True,
            "export_validate": True,
        }
        self.use_for = {**default_use_for, **(self.config.get("use_for") or {})}
        self.thresholds = self.config.get("thresholds") or {}
        self.max_elements = int(self.config.get("max_elements_for_vlm", 80) or 80)
        self.vlm_only_region_max_items = int(self.config.get("vlm_only_region_max_items", 20) or 20)
        self.vlm_only_ocr_anchor_max_blocks = int(self.config.get("vlm_only_ocr_anchor_max_blocks", 30) or 30)
        self.client = create_vlm_client_from_config(self.root_config) if self.enabled else None

    def _enabled_for(self, feature: str) -> bool:
        return bool(self.enabled and self.use_for.get(feature, False) and self.client is not None)

    def recognize_structure_staged(self, context: ProcessingContext) -> Dict[str, Any]:
        """Recognize VLM-only structure with coarse-to-fine passes.

        The staged path keeps the first call focused on large regions, the
        second call focused on elements inside those regions, and a connector
        pass focused only on arrows/lines. This is intentionally narrower than
        the legacy all-in-one prompt and gives downstream validation clearer
        artifacts to inspect.
        """
        if not self.enabled or self.client is None:
            return {"recognized": False, "elements": [], "error": "multimodal VLM is disabled"}

        region_result = self.recognize_page_regions(context)
        regions = region_result.get("regions", [])
        planner_result = self.recognize_planner_page_structure(context, regions)
        items = list(regions) + list(planner_result.get("items", []))
        elements = self._build_vlm_elements_from_items(items, context, source_prompt="planner_vlm")
        context.elements = elements

        connector_result = self.recognize_connectors(context)
        result = {
            "recognized": True,
            "count": len(context.elements),
            "raw_count": len(items),
            "dropped_count": max(0, len(items) - len(elements)),
            "regions": regions,
            "items": items,
            "planner_stages": planner_result.get("stages", {}),
            "elements": [elem.to_dict() for elem in context.elements],
            "connector_changes": connector_result.get("changes", []),
        }
        self._print_json("vlm_structure_staged recognized", result)
        self._write_artifact(context, "vlm_structure.json", result)
        return result

    def recognize_page_regions(self, context: ProcessingContext) -> Dict[str, Any]:
        """Stage 1: recognize only page skeleton and major layout regions."""
        threshold = float(self.thresholds.get("vlm_region_confidence", self.thresholds.get("vlm_structure_confidence", 0.60)))
        vlm_image_path = self._prepare_vlm_layout_image(context)
        if not self.enabled or self.client is None:
            result = {
                "recognized": False,
                "regions": [],
                "error": "multimodal VLM is disabled",
                "coordinate_system": "pixel_original",
                "vlm_image_path": vlm_image_path,
                "original_image_path": context.image_path,
                "original_size": {"width": context.canvas_width, "height": context.canvas_height},
            }
            overlay_path = self._save_vlm_page_regions_overlay(context, [])
            result["overlay_path"] = overlay_path
            context.intermediate_results["vlm_page_regions_overlay"] = overlay_path
            self._write_artifact(context, "vlm_page_regions.json", result)
            return result
        try:
            vlm_size = context.intermediate_results.get("vlm_layout_image_size") or {"width": context.canvas_width, "height": context.canvas_height}
            prompt = build_vlm_page_regions_prompt(vlm_size.get("width", context.canvas_width), vlm_size.get("height", context.canvas_height))
            data = self._parse_json_response_with_debug(
                self.client.analyze_image(vlm_image_path, prompt),
                "vlm_page_regions",
            )
        except Exception as exc:
            result = {
                "recognized": False,
                "regions": [],
                "error": str(exc),
                "coordinate_system": "pixel_original",
                "vlm_image_path": vlm_image_path,
                "original_image_path": context.image_path,
                "original_size": {"width": context.canvas_width, "height": context.canvas_height},
            }
            overlay_path = self._save_vlm_page_regions_overlay(context, [])
            result["overlay_path"] = overlay_path
            context.intermediate_results["vlm_page_regions_overlay"] = overlay_path
            self._write_artifact(context, "vlm_page_regions.json", result)
            print(f"[VLMEnhancer] VLM page region recognition skipped: {exc}", flush=True)
            return result
        raw_regions = data.get("regions", []) if isinstance(data.get("regions"), list) else []
        regions = []
        dropped_regions = []
        min_area = float(self.thresholds.get("vlm_region_min_area", 12000) or 12000)
        max_regions = int(self.config.get("vlm_page_region_max_items", 15) or 15)
        for idx, item in enumerate(raw_regions):
            if not isinstance(item, dict):
                dropped_regions.append({"index": idx, "reason": "not_object"})
                continue
            if self._confidence(item) < threshold:
                dropped_regions.append({"index": idx, "reason": "low_confidence", "confidence": self._confidence(item)})
                continue
            region_type = self._normalize_page_region_type(item.get("type"))
            if not region_type:
                dropped_regions.append({"index": idx, "reason": "unsupported_type", "type": item.get("type")})
                continue
            bbox = self._extract_vlm_bbox(item)
            pixel_bbox = self._vlm_bbox_to_original_pixel_dict(bbox, context) if bbox is not None else None
            if pixel_bbox is None:
                dropped_regions.append({"index": idx, "reason": "invalid_bbox"})
                continue
            width = pixel_bbox["width"]
            height = pixel_bbox["height"]
            area = width * height
            if area < min_area and region_type != "header":
                dropped_regions.append({"index": idx, "reason": "too_small", "type": region_type, "area": area})
                continue
            region = dict(item)
            region["type"] = region_type
            region.setdefault("id", f"region_{idx + 1:03d}")
            region["bbox"] = pixel_bbox
            region["pixel_bbox"] = pixel_bbox
            region["coordinate_system"] = "pixel_original"
            region["area"] = area
            regions.append(region)
        regions, duplicate_drops = self._drop_near_duplicate_page_regions(regions)
        dropped_regions.extend(duplicate_drops)
        if len(regions) > max_regions:
            regions.sort(key=lambda region: region.get("area", 0), reverse=True)
            dropped_regions.extend(
                {"id": region.get("id"), "reason": "over_max_regions", "area": region.get("area", 0)}
                for region in regions[max_regions:]
            )
            regions = regions[:max_regions]
        regions.sort(key=lambda region: (region["pixel_bbox"]["y"], region["pixel_bbox"]["x"]))
        result = {
            "recognized": bool(regions),
            "page_aspect_ratio_estimate": data.get("page_aspect_ratio_estimate"),
            "layout_pattern": data.get("layout_pattern"),
            "page_structure": data.get("page_structure"),
            "regions": regions,
            "reading_order": data.get("reading_order", []),
            "raw_count": len(raw_regions),
            "dropped_count": len(dropped_regions),
            "dropped_regions": dropped_regions,
            "coordinate_system": "pixel_original",
            "vlm_image_path": vlm_image_path,
            "original_image_path": context.image_path,
            "original_size": {"width": context.canvas_width, "height": context.canvas_height},
        }
        self._print_json("vlm_page_regions recognized", {"items": regions})
        self._write_artifact(context, "vlm_page_regions.json", result)
        overlay_path = self._save_vlm_page_regions_overlay(context, regions)
        result["overlay_path"] = overlay_path
        context.intermediate_results["vlm_page_regions_overlay"] = overlay_path
        self._write_artifact(context, "vlm_page_regions.json", result)
        return result

    @classmethod
    def _drop_near_duplicate_page_regions(cls, regions: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Drop regions whose bbox adds no new crop area over an earlier region."""
        kept: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        for region in regions:
            duplicate_of = next(
                (candidate for candidate in kept if cls._page_region_bbox_iou(region, candidate) >= 0.96),
                None,
            )
            if duplicate_of is not None:
                dropped.append({
                    "id": region.get("id"),
                    "type": region.get("type"),
                    "reason": "near_duplicate_bbox",
                    "duplicate_of": duplicate_of.get("id"),
                })
                continue
            kept.append(region)
        return kept, dropped

    @staticmethod
    def _page_region_bbox_iou(first: Dict[str, Any], second: Dict[str, Any]) -> float:
        a = first.get("pixel_bbox") or first.get("bbox") or {}
        b = second.get("pixel_bbox") or second.get("bbox") or {}
        ax1 = float(a.get("x", 0) or 0)
        ay1 = float(a.get("y", 0) or 0)
        ax2 = ax1 + float(a.get("width", 0) or 0)
        ay2 = ay1 + float(a.get("height", 0) or 0)
        bx1 = float(b.get("x", 0) or 0)
        by1 = float(b.get("y", 0) or 0)
        bx2 = bx1 + float(b.get("width", 0) or 0)
        by2 = by1 + float(b.get("height", 0) or 0)
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        intersection = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0

    def _prepare_vlm_layout_image(self, context: ProcessingContext) -> str:
        """Record original image size and use the real image for pixel-coordinate VLM calls."""
        from PIL import Image
        os.makedirs(context.output_dir, exist_ok=True)
        with Image.open(context.image_path) as img:
            width, height = img.size
        context.canvas_width = context.canvas_width or width
        context.canvas_height = context.canvas_height or height
        context.intermediate_results["original_image_path"] = context.image_path
        context.intermediate_results["original_size"] = {"width": width, "height": height}
        context.intermediate_results["vlm_layout_image_path"] = context.image_path
        context.intermediate_results["vlm_layout_image_size"] = {"width": width, "height": height}
        return context.image_path

    @staticmethod
    def _normalize_page_region_type(value: Any) -> Optional[str]:
        normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        normalized = PAGE_REGION_TYPE_ALIASES.get(normalized, normalized)
        return normalized if normalized in PAGE_REGION_TYPES else None

    @staticmethod
    def _normalized_bbox_to_pixel_dict(bbox: Dict[str, Any], width: int, height: int) -> Dict[str, int]:
        x = float(bbox.get("x", 0) or 0)
        y = float(bbox.get("y", 0) or 0)
        bbox_width = float(bbox.get("width", 0) or 0)
        bbox_height = float(bbox.get("height", 0) or 0)
        x1 = round(x * width / 1000)
        y1 = round(y * height / 1000)
        x2 = round((x + bbox_width) * width / 1000)
        y2 = round((y + bbox_height) * height / 1000)
        return {"x": x1, "y": y1, "width": max(0, x2 - x1), "height": max(0, y2 - y1)}

    def _vlm_bbox_to_original_pixel_dict(self, bbox: List[Any], context: ProcessingContext) -> Optional[Dict[str, int]]:
        pixel_bbox = self._normalize_bbox(bbox, context.canvas_width, context.canvas_height)
        if pixel_bbox is None:
            return None
        x1, y1, x2, y2 = pixel_bbox
        return {"x": x1, "y": y1, "width": max(0, x2 - x1), "height": max(0, y2 - y1)}

    def _save_vlm_page_regions_overlay(self, context: ProcessingContext, regions: List[Dict[str, Any]]) -> str:
        """Draw layout regions on the original image using real pixel coordinates."""
        from PIL import Image, ImageDraw, ImageFont
        output_path = os.path.join(context.output_dir, "vlm_page_regions_overlay.png")
        with Image.open(context.image_path) as img:
            canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for region in regions:
            pixel_bbox = region.get("pixel_bbox") or region.get("bbox") or self._normalized_bbox_to_pixel_dict(region.get("bbox", {}), canvas.width, canvas.height)
            x1, y1 = int(pixel_bbox["x"]), int(pixel_bbox["y"])
            x2, y2 = x1 + int(pixel_bbox["width"]), y1 + int(pixel_bbox["height"])
            draw.rectangle([x1, y1, x2, y2], outline="#ff3b30", width=3)
            label = f"{region.get('id')}:{region.get('type')} {float(region.get('confidence', 0)):.2f}"
            draw.text((max(0, x1), max(0, y1 - 12)), label, fill="#ff3b30", font=font)
        canvas.save(output_path)
        return output_path

