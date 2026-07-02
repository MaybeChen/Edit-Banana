"""Optional VLM enhancement passes for the image-to-diagram pipeline.

The enhancer is deliberately conservative: every VLM call is behind config
switches and failures fall back to the existing OCR/SAM3/CV pipeline.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from .base import ProcessingContext
from .vlm_client import create_vlm_client_from_config
from prompts.vlm_structure import VLM_PAGE_REGIONS_PROMPT

from .vlm_enhancer_core import VLMCoreMixin
from .vlm_enhancer_structure import VLMStructureMixin
from .vlm_enhancer_refinement import VLMRefinementMixin
from .vlm_enhancer_text import VLMTextPromptMixin


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
        element_result = self.recognize_region_elements(context, regions)
        items = list(regions) + list(element_result.get("items", []))
        elements = self._build_vlm_elements_from_items(items, context, source_prompt="vlm_region_elements")
        context.elements = elements

        connector_result = self.recognize_connectors(context)
        result = {
            "recognized": True,
            "count": len(context.elements),
            "raw_count": len(items),
            "dropped_count": max(0, len(items) - len(elements)),
            "regions": regions,
            "items": items,
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
        try:
            data = self._parse_json_response_with_debug(
                self.client.analyze_image(vlm_image_path, VLM_PAGE_REGIONS_PROMPT),
                "vlm_page_regions",
            )
        except Exception as exc:
            print(f"[VLMEnhancer] VLM page region recognition skipped: {exc}", flush=True)
            return {"recognized": False, "regions": [], "error": str(exc)}
        raw_regions = data.get("regions", []) if isinstance(data.get("regions"), list) else []
        regions = []
        for idx, item in enumerate(raw_regions):
            if not isinstance(item, dict) or self._confidence(item) < threshold:
                continue
            bbox = self._extract_vlm_bbox(item)
            normalized_bbox = self._normalize_bbox(bbox, 1000, 1000) if bbox is not None else None
            if normalized_bbox is None:
                continue
            region = dict(item)
            region.setdefault("id", f"region_{idx + 1:03d}")
            region["bbox"] = {"x": normalized_bbox[0], "y": normalized_bbox[1], "width": normalized_bbox[2] - normalized_bbox[0], "height": normalized_bbox[3] - normalized_bbox[1]}
            region["pixel_bbox"] = self._normalized_bbox_to_pixel_dict(region["bbox"], context.canvas_width, context.canvas_height)
            regions.append(region)
        result = {
            "recognized": bool(regions),
            "page_aspect_ratio_estimate": data.get("page_aspect_ratio_estimate"),
            "layout_pattern": data.get("layout_pattern"),
            "page_structure": data.get("page_structure"),
            "regions": regions,
            "reading_order": data.get("reading_order", []),
            "raw_count": len(raw_regions),
            "coordinate_system": "normalized_0_1000",
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

    def _prepare_vlm_layout_image(self, context: ProcessingContext) -> str:
        """Save original-size metadata and create the <=2048 long-edge image for VLM."""
        from PIL import Image
        os.makedirs(context.output_dir, exist_ok=True)
        with Image.open(context.image_path) as img:
            width, height = img.size
            context.canvas_width = context.canvas_width or width
            context.canvas_height = context.canvas_height or height
            long_edge = max(width, height)
            if long_edge <= 2048:
                vlm_path = context.image_path
                vlm_size = {"width": width, "height": height}
            else:
                scale = 2048 / float(long_edge)
                resized = img.convert("RGB").resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)
                vlm_path = os.path.join(context.output_dir, "vlm_layout_thumbnail.png")
                resized.save(vlm_path)
                vlm_size = {"width": resized.width, "height": resized.height}
        context.intermediate_results["original_image_path"] = context.image_path
        context.intermediate_results["original_size"] = {"width": width, "height": height}
        context.intermediate_results["vlm_layout_image_path"] = vlm_path
        context.intermediate_results["vlm_layout_image_size"] = vlm_size
        return vlm_path

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

    def _save_vlm_page_regions_overlay(self, context: ProcessingContext, regions: List[Dict[str, Any]]) -> str:
        """Draw layout regions on the original image using real pixel coordinates."""
        from PIL import Image, ImageDraw, ImageFont
        output_path = os.path.join(context.output_dir, "vlm_page_regions_overlay.png")
        with Image.open(context.image_path) as img:
            canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for region in regions:
            pixel_bbox = region.get("pixel_bbox") or self._normalized_bbox_to_pixel_dict(region.get("bbox", {}), canvas.width, canvas.height)
            x1, y1 = int(pixel_bbox["x"]), int(pixel_bbox["y"])
            x2, y2 = x1 + int(pixel_bbox["width"]), y1 + int(pixel_bbox["height"])
            draw.rectangle([x1, y1, x2, y2], outline="#ff3b30", width=3)
            label = f"{region.get('id')}:{region.get('type')} {float(region.get('confidence', 0)):.2f}"
            draw.text((max(0, x1), max(0, y1 - 12)), label, fill="#ff3b30", font=font)
        canvas.save(output_path)
        return output_path

