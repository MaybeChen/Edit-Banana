"""Auxiliary Pipeline methods for VLM-only recognition and PPTX export."""

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

from .base import ProcessingContext
from .data_types import ElementInfo, BoundingBox, LayerLevel, get_layer_level


class PipelineVLMAndExportMixin:
    def _process_image_vlm_only(self, context: ProcessingContext, img_output_dir: str, img_stem: str) -> Optional[str]:
        """Run the independent VLM-only pipeline.

        This path skips SAM3, but can optionally use OCR text anchors so VLM is
        responsible for structure/semantics while OCR keeps text content stable.
        """
        print("\n[1] VLM-only OCR anchors...")
        self._initialize_canvas_from_image(context)
        ocr_anchor_blocks = self._extract_vlm_only_ocr_anchors(context, img_output_dir)
        if ocr_anchor_blocks:
            context.intermediate_results['ocr_text_blocks'] = ocr_anchor_blocks
            print(f"   OCR anchors: {len(ocr_anchor_blocks)}")
        else:
            print("   OCR anchors: disabled or unavailable")

        print("\n[2] VLM-only staged structure recognition...")
        structure_result = self.vlm_enhancer.recognize_structure_staged(context)
        structure_path = os.path.join(img_output_dir, "vlm_structure_result.json")
        self._save_json(structure_path, structure_result)
        context.intermediate_results['vlm_structure_result_json'] = structure_path
        if not structure_result.get("recognized"):
            raise Exception(f"VLM-only recognition failed: {structure_result.get('error', 'no structured result')}")

        text_blocks = ocr_anchor_blocks or self._extract_vlm_text_blocks(context)
        context.intermediate_results['vlm_text_blocks'] = text_blocks
        context.intermediate_results['ocr_text_blocks'] = text_blocks
        text_path = os.path.join(img_output_dir, "vlm_text_result.json")
        self._save_json(
            text_path,
            {"text_blocks": text_blocks, "source": "ocr_anchors" if ocr_anchor_blocks else "vlm_structure_text_elements"},
        )
        context.intermediate_results['vlm_text_result_json'] = text_path
        print(f"   VLM text blocks: {len(text_blocks)}")

        crop_result = self._crop_vlm_only_image_elements(context, img_output_dir)
        print(f"   VLM image crops: {crop_result.get('cropped', 0)} cropped")

        overlay_path = self._save_vlm_structure_overlay(context, img_output_dir)
        context.intermediate_results['vlm_structure_overlay'] = overlay_path
        print(f"   VLM structure overlay: {overlay_path}")
        if structure_result.get("raw_count", 0) and not context.elements:
            print("   Warning: VLM returned raw items but none passed schema/bbox validation")
        print(f"   VLM-only elements: {len(context.elements)}")

        print("\n[3] VLM element attribute enrichment...")
        attr_result = self.vlm_enhancer.enrich_element_attributes(context)
        attr_path = os.path.join(img_output_dir, "vlm_element_attributes.json")
        self._save_json(attr_path, {"elements": [e.to_dict() for e in context.elements], "changes": attr_result.get("changes", [])})
        context.intermediate_results['vlm_element_attributes_json'] = attr_path
        print(f"   VLM element attributes: {attr_result.get('updated', 0)} updated")

        print("\n[4] Direct PPTX generation + VLM quality loop...")
        pptx_path = self._run_pptx_quality_loop(context, text_blocks, img_output_dir, img_stem)
        print(f"\n{'='*60}\nDone.\n{'='*60}")
        return pptx_path

    def _vlm_only_use_ocr_anchors(self) -> bool:
        recognition_cfg = self.config.get("recognition") or {}
        multimodal_cfg = self.config.get("multimodal") or {}
        return bool(
            recognition_cfg.get(
                "use_ocr_anchors",
                multimodal_cfg.get("vlm_only_use_ocr_anchors", True),
            )
        )

    def _extract_vlm_only_ocr_anchors(self, context: ProcessingContext, output_dir: str) -> List[Dict[str, Any]]:
        """Run OCR as optional text anchors for VLM-only structure recognition."""
        if not self._vlm_only_use_ocr_anchors() or self.text_restorer is None:
            return []
        try:
            text_blocks = self.text_restorer.process_image(context.image_path)
            for idx, block in enumerate(text_blocks):
                block.setdefault("id", idx)
            if hasattr(self.text_restorer, "save_ocr_artifacts"):
                context.intermediate_results['ocr_result_json'] = self.text_restorer.save_ocr_artifacts(output_dir, context.image_path)
            else:
                self._save_json(os.path.join(output_dir, "ocr_result.json"), {"text_blocks": text_blocks})
            if hasattr(self.text_restorer, "save_ocr_overlay"):
                context.intermediate_results['ocr_overlay'] = self.text_restorer.save_ocr_overlay(output_dir, context.image_path)
            return text_blocks
        except Exception as exc:
            print(f"   OCR anchors failed: {exc}")
            return []

    @staticmethod
    def _crop_vlm_only_image_elements(context: ProcessingContext, output_dir: str) -> Dict[str, Any]:
        """Crop image/icon/logo/chart boxes in VLM-only mode for image fallback."""
        from PIL import Image
        image_types = {"icon", "picture", "logo", "chart", "function_graph"}
        crop_dir = os.path.join(output_dir, "vlm_crops")
        cropped = []
        try:
            with Image.open(context.image_path) as img:
                rgba = img.convert("RGBA")
                os.makedirs(crop_dir, exist_ok=True)
                for elem in context.elements:
                    if str(elem.element_type or "").lower() not in image_types or elem.base64:
                        continue
                    bbox = elem.bbox
                    if bbox.width <= 1 or bbox.height <= 1:
                        continue
                    crop_box = (
                        max(0, int(bbox.x1)),
                        max(0, int(bbox.y1)),
                        min(rgba.width, int(bbox.x2)),
                        min(rgba.height, int(bbox.y2)),
                    )
                    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                        continue
                    crop_path = os.path.join(crop_dir, f"element_{elem.id}.png")
                    rgba.crop(crop_box).save(crop_path)
                    with open(crop_path, "rb") as f:
                        elem.base64 = base64.b64encode(f.read()).decode("ascii")
                    elem.reconstruction_strategy = elem.reconstruction_strategy or "cropped_image"
                    elem.processing_notes.append("vlm_only_bbox_crop")
                    cropped.append({"id": elem.id, "path": crop_path})
        except Exception as exc:
            return {"cropped": len(cropped), "items": cropped, "error": str(exc)}
        context.intermediate_results['vlm_image_crops'] = cropped
        return {"cropped": len(cropped), "items": cropped}

    @staticmethod
    def _extract_vlm_text_blocks(context: ProcessingContext) -> List[Dict[str, Any]]:
        """Convert VLM structure ``text`` elements into exporter text blocks."""
        text_blocks: List[Dict[str, Any]] = []
        for elem in context.elements:
            if str(elem.element_type or "").lower() != "text":
                continue
            meta = getattr(elem, "vlm_item", {}) or {}
            bbox = elem.bbox
            content = (
                meta.get("content")
                or meta.get("text")
                or meta.get("label")
                or meta.get("value")
                or ""
            )
            if not str(content).strip():
                continue
            text_blocks.append(
                {
                    "id": len(text_blocks),
                    "source_element_id": elem.id,
                    "text": str(content).strip(),
                    "geometry": {
                        "x": bbox.x1,
                        "y": bbox.y1,
                        "width": bbox.width,
                        "height": bbox.height,
                    },
                    "font_family": meta.get("font_family") or "Microsoft YaHei",
                    "font_size": meta.get("font_size_estimate") or meta.get("font_size") or 12,
                    "font_weight": meta.get("font_weight") or "normal",
                    "font_style": meta.get("font_style") or "normal",
                    "font_color": meta.get("font_color") or "#000000",
                    "text_align": meta.get("text_align") or "center",
                    "vertical_align": meta.get("vertical_align") or "middle",
                    "confidence": elem.score,
                }
            )
        return text_blocks

    @staticmethod
    def _save_json(path: str, data: dict) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def _recognition_mode(self) -> str:
        recognition_cfg = self.config.get("recognition") or {}
        mode = recognition_cfg.get("mode") or (self.config.get("multimodal") or {}).get("recognition_mode") or "sam3_first"
        return str(mode).strip().lower().replace("-", "_")

    @staticmethod
    def _initialize_canvas_from_image(context: ProcessingContext) -> None:
        from PIL import Image
        with Image.open(context.image_path) as img:
            context.canvas_width, context.canvas_height = img.size

    @staticmethod
    def _save_vlm_structure_overlay(context: ProcessingContext, output_dir: str) -> str:
        """Save a debug image with VLM-only recognized elements annotated."""
        from PIL import Image, ImageDraw, ImageFont
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "vlm_structure_overlay.png")
        with Image.open(context.image_path) as img:
            canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        palette = {
            "container": "#00a6ff",
            "rounded rectangle": "#ff5c8a",
            "rounded_rectangle": "#ff5c8a",
            "rectangle": "#ff5c8a",
            "cylinder": "#a855f7",
            "arrow": "#2563eb",
            "connector": "#2563eb",
            "line": "#2563eb",
            "icon": "#22c55e",
            "picture": "#22c55e",
            "logo": "#22c55e",
            "chart": "#22c55e",
        }
        for elem in context.elements:
            elem_type = str(elem.element_type or "").lower()
            color = palette.get(elem_type, "#f97316")
            bbox = elem.bbox
            if elem_type in {"arrow", "connector", "line"}:
                start = elem.arrow_start or (bbox.x1, (bbox.y1 + bbox.y2) / 2)
                end = elem.arrow_end or (bbox.x2, (bbox.y1 + bbox.y2) / 2)
                draw.line([start, end], fill=color, width=max(2, int(elem.stroke_width or 1)))
                draw.rectangle(bbox.to_list(), outline=color, width=1)
            else:
                draw.rectangle(bbox.to_list(), outline=color, width=2)
            label = f"{elem.id}:{elem.element_type}"
            if elem_type == "text":
                meta = getattr(elem, "vlm_item", {}) or {}
                content = str(meta.get("content") or meta.get("text") or meta.get("label") or "")[:24]
                if content:
                    label = f"{label}:{content}"
            text_xy = (max(0, bbox.x1), max(0, bbox.y1 - 12))
            draw.text(text_xy, label, fill=color, font=font)
        canvas.save(output_path)
        return output_path

    def _run_pptx_quality_loop(self, context: ProcessingContext, text_blocks: List[dict], output_dir: str, img_stem: str) -> Optional[str]:
        pptx_path = None
        max_rounds = int((self.config.get("multimodal") or {}).get("max_quality_rounds", 3) or 3)
        quality_threshold = float((self.config.get("multimodal") or {}).get("quality_threshold", 90) or 90)
        for round_idx in range(1, max_rounds + 1):
            pptx_elements = self._prepare_pptx_elements(context)
            pptx_path = self._export_pptx_direct(context, pptx_elements, text_blocks, output_dir, img_stem)
            context.intermediate_results['pptx_output'] = pptx_path
            print(f"   Round {round_idx} PPTX: {pptx_path}")
            validation = self.vlm_enhancer.validate_pptx_export(context, pptx_path, round_idx)
            score = float(validation.get("score", 100) or 0)
            print(f"   Round {round_idx} quality score: {score:.1f}/100")
            if score >= quality_threshold or round_idx >= max_rounds:
                break
            repairs = self.vlm_enhancer.apply_export_repairs(context, validation)
            if repairs.get("updated", 0) == 0 and repairs.get("added", 0) == 0:
                print("   No structured repairs returned; stopping quality loop")
                break
        return pptx_path

    def _prepare_pptx_elements(self, context: ProcessingContext) -> List[ElementInfo]:
        """Prepare recognized elements for direct PPTX export."""
        prepared = []
        for elem in context.elements:
            elem_type = elem.element_type.lower()

            if elem_type == "text":
                # VLM-only text elements are converted to text_blocks and exported
                # as native PPT text boxes; do not also render shape placeholders.
                continue

            if elem_type in {'arrow', 'line', 'connector'}:
                if self._is_border_like_connector(elem, context):
                    elem.element_type = "container"
                    elem.fill_color = "none"
                    elem.stroke_color = elem.stroke_color or "#000000"
                    elem.stroke_width = max(1, int(elem.stroke_width or 1))
                    elem.layer_level = LayerLevel.BACKGROUND.value
                else:
                    if self._is_duplicate_line_fragment(elem, context.elements):
                        elem.processing_notes.append("Skipped duplicate line fragment contained by an arrow")
                        continue
                    if not elem.arrow_start or not elem.arrow_end:
                        elem.arrow_start, elem.arrow_end = self._infer_edge_points(elem, context.elements)
                    inferred_heads = self._infer_arrow_heads_from_polygon(elem)
                    if inferred_heads and elem.arrow_heads in {None, "none"}:
                        elem.arrow_heads = inferred_heads
                    if (
                        self._recognition_mode() != "vlm_only"
                        and elem.line_style not in {"dashed", "dotted"}
                        and self._edge_looks_dashed(elem, context)
                    ):
                        elem.line_style = "dashed"
                    if self._edge_looks_curved(elem):
                        elem.arrow_style = "curved"
                    elem.layer_level = LayerLevel.ARROW.value

            elif self._is_background_like_element(elem, context):
                # Large panels/frames are often detected by SAM3 as generic image or
                # shape prompts. Export them as transparent background containers so
                # their borders remain visible without covering inner content.
                elem.element_type = "container"
                elem.fill_color = "none"
                elem.stroke_color = elem.stroke_color or "#000000"
                elem.stroke_width = max(1, int(elem.stroke_width or 1))
                elem.layer_level = LayerLevel.BACKGROUND.value

            elif elem_type in {'icon', 'picture', 'logo', 'chart', 'function_graph'}:
                if elem.base64:
                    elem.layer_level = LayerLevel.IMAGE.value
                else:
                    # If VLM-only bbox cropping failed, export a visible placeholder
                    # box instead of silently dropping the image-like element.
                    elem.element_type = "rectangle"
                    elem.fill_color = "none"
                    elem.stroke_color = elem.stroke_color or "#000000"
                    elem.stroke_width = max(1, int(elem.stroke_width or 1))
                    elem.layer_level = LayerLevel.BASIC_SHAPE.value

            else:
                semantic_type = str(getattr(elem, "semantic_type", "") or "").lower()
                if semantic_type in {"table", "chart", "diagram", "group"} and elem_type == "container":
                    elem.reconstruction_strategy = elem.reconstruction_strategy or f"{semantic_type}_candidate"
                if elem_type in {"rectangle", "rounded rectangle", "rounded_rectangle", "container", "cylinder"}:
                    elem.fill_color = "none"
                else:
                    elem.fill_color = elem.fill_color or "#ffffff"
                elem.stroke_color = elem.stroke_color or "#000000"
                elem.stroke_width = max(1, int(elem.stroke_width or 1))
                elem.layer_level = LayerLevel.BASIC_SHAPE.value

            prepared.append(elem)

        if self._recognition_mode() != "vlm_only":
            prepared.extend(self._detect_missing_divider_lines(context, prepared))
        return prepared



    @staticmethod
    def _is_border_like_connector(elem: ElementInfo, context: ProcessingContext) -> bool:
        """Detect connector candidates that are actually container/panel borders."""
        elem_type = elem.element_type.lower()
        if elem_type not in {"connector", "line"}:
            return False
        if elem.arrow_heads not in {None, "none"}:
            return False
        if not elem.bbox:
            return False
        canvas_area = max(1, int((context.canvas_width or 0) * (context.canvas_height or 0)))
        area_ratio = elem.bbox.area / canvas_area
        # A true connector is usually thin. A large two-dimensional connector box is
        # typically a misclassified frame/container boundary, such as a grouped panel.
        return bool(elem.bbox.width >= 40 and elem.bbox.height >= 40 and area_ratio >= 0.025)

    @staticmethod
    def _edge_looks_curved(elem: ElementInfo) -> bool:
        if elem.element_type.lower() not in {"arrow", "connector", "line"} or not elem.polygon:
            return False
        if not elem.bbox or min(elem.bbox.width, elem.bbox.height) < 20:
            return False
        return len(elem.polygon) >= 5

    def _infer_arrow_heads_from_polygon(self, elem: ElementInfo) -> Optional[str]:
        """Infer start/end/both arrowheads from sharp polygon tips near endpoints."""
        if elem.element_type.lower() not in {"arrow", "connector", "line"} or not elem.polygon:
            return None
        if not elem.arrow_start or not elem.arrow_end:
            return None
        points = [(float(p[0]), float(p[1])) for p in elem.polygon if len(p) >= 2]
        if len(points) < 4:
            return None

        sharp_points = []
        for point in points:
            angle = self._polygon_angle_at_point(points, point)
            if angle is not None and angle <= 75:
                sharp_points.append(point)
        if not sharp_points:
            return None

        def near(candidate, endpoint):
            import math
            tolerance = max(10.0, min(elem.bbox.width, elem.bbox.height) * 0.35)
            return math.hypot(candidate[0] - endpoint[0], candidate[1] - endpoint[1]) <= tolerance

        start_has = any(near(point, elem.arrow_start) for point in sharp_points)
        end_has = any(near(point, elem.arrow_end) for point in sharp_points)
        if start_has and end_has:
            return "both"
        if start_has:
            return "start"
        if end_has:
            return "end"
        return None

    @staticmethod
    def _polygon_angle_at_point(points, point) -> Optional[float]:
        import math
        try:
            idx = points.index(point)
        except ValueError:
            return None
        prev_p = points[(idx - 1) % len(points)]
        next_p = points[(idx + 1) % len(points)]
        v1 = (prev_p[0] - point[0], prev_p[1] - point[1])
        v2 = (next_p[0] - point[0], next_p[1] - point[1])
        len1 = math.hypot(*v1)
        len2 = math.hypot(*v2)
        if len1 <= 0 or len2 <= 0:
            return None
        cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)))
        return math.degrees(math.acos(cos_a))

    def _detect_missing_divider_lines(self, context: ProcessingContext, existing_elements: List[ElementInfo]) -> List[ElementInfo]:
        """Use lightweight image analysis to recover long horizontal divider lines."""
        if not context.image_path:
            return []
        from PIL import Image
        import numpy as np
        with Image.open(context.image_path) as img:
            gray = np.array(img.convert("L"))

        dark = gray < 110
        height, width = dark.shape
        candidates = []
        for y in range(height):
            if y < 8 or y > height - 8:
                continue
            xs = np.flatnonzero(dark[y])
            if len(xs) < width * 0.35:
                continue
            runs = self._continuous_runs(xs)
            for x1, x2 in runs:
                if x2 - x1 >= width * 0.35:
                    candidates.append((x1, y, x2, y + 1))

        merged = self._merge_horizontal_line_candidates(candidates)
        new_elements = []
        next_id = max([e.id for e in existing_elements], default=-1) + 1
        for x1, y1, x2, y2 in merged:
            if self._overlaps_existing_line([x1, y1, x2, y2], existing_elements):
                continue
            elem = ElementInfo(
                id=next_id + len(new_elements),
                element_type="line",
                bbox=BoundingBox(int(x1), int(y1), int(x2), int(max(y2, y1 + 2))),
                score=0.8,
                source_prompt="cv_horizontal_divider",
                stroke_color="#000000",
                stroke_width=1,
                line_style="solid",
                layer_level=LayerLevel.ARROW.value,
            )
            cy = (elem.bbox.y1 + elem.bbox.y2) / 2
            elem.arrow_start = (elem.bbox.x1, cy)
            elem.arrow_end = (elem.bbox.x2, cy)
            elem.arrow_heads = "none"
            elem.processing_notes.append("cv_horizontal_divider:add")
            new_elements.append(elem)
        return new_elements

    @staticmethod
    def _continuous_runs(indices):
        runs = []
        start = int(indices[0]) if len(indices) else 0
        prev = start
        for value in indices[1:]:
            value = int(value)
            if value > prev + 1:
                runs.append((start, prev + 1))
                start = value
            prev = value
        if len(indices):
            runs.append((start, prev + 1))
        return runs

    @staticmethod
    def _merge_horizontal_line_candidates(candidates):
        if not candidates:
            return []
        candidates = sorted(candidates, key=lambda box: (box[1], box[0]))
        merged = []
        for box in candidates:
            x1, y1, x2, y2 = box
            if not merged or y1 - merged[-1][3] > 2 or x1 > merged[-1][2] + 8:
                merged.append([x1, y1, x2, y2])
            else:
                merged[-1][0] = min(merged[-1][0], x1)
                merged[-1][1] = min(merged[-1][1], y1)
                merged[-1][2] = max(merged[-1][2], x2)
                merged[-1][3] = max(merged[-1][3], y2)
        return merged

    @staticmethod
    def _overlaps_existing_line(box, elements) -> bool:
        x1, y1, x2, y2 = box
        for elem in elements:
            if elem.element_type.lower() not in {"line", "connector", "arrow"}:
                continue
            bx1, by1, bx2, by2 = elem.bbox.to_list()
            if x2 <= bx1 or bx2 <= x1 or y2 <= by1 or by2 <= y1:
                continue
            inter = (min(x2, bx2) - max(x1, bx1)) * (min(y2, by2) - max(y1, by1))
            if inter / max(1, (x2 - x1) * max(1, y2 - y1)) > 0.5:
                return True
        return False

    @staticmethod
    def _is_background_like_element(elem: ElementInfo, context: ProcessingContext) -> bool:
        elem_type = elem.element_type.lower()
        if elem_type in {'section_panel', 'title_bar', 'container', 'background', 'panel'}:
            return True
        canvas_area = max(1, int((context.canvas_width or 0) * (context.canvas_height or 0)))
        return bool(elem.bbox and elem.bbox.area / canvas_area >= 0.12)

    @staticmethod
    def _export_pptx_direct(context: ProcessingContext, elements: List[ElementInfo], text_blocks: List[dict], output_dir: str, img_stem: str) -> Optional[str]:
        from pptx_exporter import (
            export_elements_to_pptx,
            is_pptx_export_available,
            missing_pptx_dependency_message,
        )
        if not is_pptx_export_available():
            print(f"   PPTX skipped: {missing_pptx_dependency_message()}")
            return None
        pptx_path = os.path.join(output_dir, f"{img_stem}.pptx")
        return export_elements_to_pptx(
            elements=elements,
            text_blocks=text_blocks,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            pptx_path=pptx_path,
        )

    def _is_duplicate_line_fragment(self, elem, elements) -> bool:
        """Skip short line detections that are already part of an arrow mask."""
        if elem.element_type.lower() != "line":
            return False
        line_box = elem.bbox.to_list()
        line_area = max(1, elem.bbox.area)
        for other in elements:
            if other is elem or other.element_type.lower() not in {"arrow", "connector"}:
                continue
            other_box = other.bbox.to_list()
            x1 = max(line_box[0], other_box[0])
            y1 = max(line_box[1], other_box[1])
            x2 = min(line_box[2], other_box[2])
            y2 = min(line_box[3], other_box[3])
            if x2 <= x1 or y2 <= y1:
                continue
            if ((x2 - x1) * (y2 - y1)) / line_area >= 0.65:
                return True
        return False

    def _edge_looks_dashed(self, elem, context: ProcessingContext = None) -> bool:
        """Heuristically detect dashed/dotted connectors from the source crop.

        SAM3 only returns element type and geometry, so dashed source lines need a
        lightweight pixel check before PPTX export. We sample the long axis
        of thin line/connector detections and mark it dashed when ink occupancy is
        broken into several separated runs.
        """
        if elem.element_type.lower() not in {"arrow", "line", "connector"}:
            return False
        if context is None or not getattr(context, "image_path", None):
            return False
        bbox = elem.bbox
        if max(bbox.width, bbox.height) < 40:
            return False
        try:
            from PIL import Image
            import numpy as np
            with Image.open(context.image_path) as img:
                gray = img.convert("L")
                x1, y1, x2, y2 = map(int, bbox.to_list())
                pad = 3
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(gray.width, x2 + pad)
                y2 = min(gray.height, y2 + pad)
                crop = np.array(gray.crop((x1, y1, x2, y2)))
            if crop.size == 0:
                return False
            ink = crop < 180
            profile = ink.any(axis=0) if bbox.width >= bbox.height else ink.any(axis=1)
            runs = []
            run = 0
            for value in profile:
                if bool(value):
                    run += 1
                elif run:
                    runs.append(run)
                    run = 0
            if run:
                runs.append(run)
            if len(runs) < 3:
                return False
            coverage = sum(runs) / max(1, len(profile))
            return coverage < 0.72
        except Exception:
            return False

    def _infer_edge_points(self, elem, elements=None) -> tuple:
        """Infer connector endpoints from the arrow/line geometry.

        Arrows are intentionally exported as coordinate-based edges instead of
        binding to source/target elements. This avoids incorrect attachments when
        relationship inference is noisy and preserves the source image geometry.
        """
        bbox = elem.bbox
        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2
        elem_type = elem.element_type.lower()

        if elem_type in {"line", "connector"} or not elem.polygon:
            if bbox.width >= bbox.height:
                return (bbox.x1, cy), (bbox.x2, cy)
            return (cx, bbox.y1), (cx, bbox.y2)

        points = [(float(p[0]), float(p[1])) for p in elem.polygon if len(p) >= 2]
        if len(points) < 2:
            if bbox.width >= bbox.height:
                return (bbox.x1, cy), (bbox.x2, cy)
            return (cx, bbox.y1), (cx, bbox.y2)

        tip = self._find_sharpest_polygon_point(points)
        start = max(points, key=lambda p: (p[0] - tip[0]) ** 2 + (p[1] - tip[1]) ** 2)
        return start, tip

    def _find_sharpest_polygon_point(self, points):
        """Find the most likely arrow head tip as the sharpest polygon vertex."""
        if len(points) < 3:
            return points[-1]
        import math
        best_point = points[0]
        best_angle = 360.0
        n = len(points)
        for i, point in enumerate(points):
            prev_p = points[(i - 1) % n]
            next_p = points[(i + 1) % n]
            v1 = (prev_p[0] - point[0], prev_p[1] - point[1])
            v2 = (next_p[0] - point[0], next_p[1] - point[1])
            len1 = math.hypot(*v1)
            len2 = math.hypot(*v2)
            if len1 <= 0 or len2 <= 0:
                continue
            cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)))
            angle = math.degrees(math.acos(cos_a))
            if angle < best_angle:
                best_angle = angle
                best_point = point
        return best_point
