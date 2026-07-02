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
from .sam3_config import PromptGroup
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




class VLMRefinementMixin:
    def refine_elements(self, context: ProcessingContext) -> Dict[str, Any]:
        """Ask VLM to correct SAM3 element types and line styles."""
        if not self._enabled_for("element_refine") or not context.elements:
            return {"updated": 0, "items": []}
        threshold = float(self.thresholds.get("element_refine_confidence", 0.75))
        prompt = (
            "你是图表元素类型校正器。根据图片和候选元素列表，输出可直接覆盖原识别结果的结构化增量。"
            + self._json_only_instruction(
                '{"elements":[{"id":1,"type":"connector","line_style":"dashed","confidence":0.92}]}'
            )
            + "只返回需要修改的元素；不确定就不要返回该元素。"
            "type 只能选 icon,picture,rectangle,rounded rectangle,circle,ellipse,cylinder,diamond,triangle,hexagon,container,arrow,connector,line。"
            "line_style 只能是 solid、dashed 或 null。"
            f"候选元素：{json.dumps(self._summarize_elements(context.elements), ensure_ascii=False)}"
        )
        try:
            data = self._parse_json_response_with_debug(self.client.analyze_image(context.image_path, prompt), "element_refine")
        except Exception as exc:
            print(f"[VLMEnhancer] element refine skipped: {exc}", flush=True)
            return {"updated": 0, "items": []}
        updates = data.get("elements", []) if isinstance(data.get("elements"), list) else []
        by_id = {elem.id: elem for elem in context.elements}
        applied = []
        changes = []
        for item in updates:
            if not isinstance(item, dict) or self._confidence(item) < threshold:
                continue
            elem = by_id.get(item.get("id"))
            new_type = self._sanitize_type(item.get("type"))
            if not elem or not new_type:
                continue
            old_type = elem.element_type
            old_line_style = elem.line_style
            if new_type:
                elem.element_type = new_type
            line_style = item.get("line_style")
            if line_style in {"solid", "dashed"}:
                elem.line_style = line_style
            elem.processing_notes.append(f"vlm_element_refine: {old_type}->{elem.element_type}")
            applied.append(item)
            if old_type != elem.element_type or old_line_style != elem.line_style:
                changes.append(
                    {
                        "id": elem.id,
                        "type": {"from": old_type, "to": elem.element_type},
                        "line_style": {"from": old_line_style, "to": elem.line_style},
                        "confidence": self._confidence(item),
                    }
                )
        result = {"updated": len(changes), "items": applied, "changes": changes}
        self._print_json("element_refine applied", changes)
        self._write_artifact(context, "vlm_element_refine.json", result)
        return result

    def refine_regions(self, context: ProcessingContext) -> Dict[str, Any]:
        """Ask VLM to convert bad regions into structured fallback elements."""
        if not self._enabled_for("region_refine"):
            return {"added": 0, "items": []}
        regions = context.intermediate_results.get("bad_regions") or []
        if not regions:
            return {"added": 0, "items": []}
        threshold = float(self.thresholds.get("region_refine_confidence", 0.70))
        prompt = (
            "你是图表漏检区域结构化识别器。根据图片和 bad_regions，补充可落地到原识别结果的元素。"
            + self._json_only_instruction(
                '{"elements":[{"type":"rectangle","bbox":[10,20,100,120],"line_style":"solid","confidence":0.86}]}'
            )
            + "bbox 必须是原图像素坐标 [x1,y1,x2,y2]；如果不确定，不要返回该元素。bad_regions="
            + json.dumps(regions[:20], ensure_ascii=False)
        )
        try:
            data = self._parse_json_response_with_debug(self.client.analyze_image(context.image_path, prompt), "region_refine")
        except Exception as exc:
            print(f"[VLMEnhancer] region refine skipped: {exc}", flush=True)
            return {"added": 0, "items": []}
        items = data.get("elements", []) if isinstance(data.get("elements"), list) else []
        new_elements: List[ElementInfo] = []
        next_id = max([e.id for e in context.elements], default=-1) + 1
        for item in items:
            if not isinstance(item, dict) or self._confidence(item) < threshold:
                continue
            new_type = self._sanitize_type(item.get("type"))
            bbox = item.get("bbox")
            if not new_type or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            elem = ElementInfo(
                id=next_id + len(new_elements),
                element_type=new_type,
                bbox=BoundingBox.from_list([int(v) for v in bbox]),
                score=self._confidence(item),
                source_prompt="vlm_region_refine",
            )
            if item.get("line_style") in {"solid", "dashed"}:
                elem.line_style = item.get("line_style")
            elem.processing_notes.append("vlm_region_refine")
            new_elements.append(elem)
        context.elements.extend(new_elements)
        result = {"added": len(new_elements), "items": [e.to_dict() for e in new_elements]}
        self._print_json("region_refine added", result["items"])
        self._write_artifact(context, "vlm_region_refine.json", result)
        return result

    def refine_layout(self, context: ProcessingContext) -> Dict[str, Any]:
        """Ask VLM to improve connector direction, endpoints, and dashed styles."""
        if not self._enabled_for("layout_refine") or not context.elements:
            return {"updated": 0, "edges": []}
        threshold = float(self.thresholds.get("layout_refine_confidence", 0.70))
        prompt = (
            "你是图表连接关系分析器。根据图片和元素列表，输出可直接更新原连接线元素的结构化增量。"
            + self._json_only_instruction(
                '{"edges":[{"id":3,"source_id":10,"target_id":11,"line_style":"dashed","type":"connector","confidence":0.9}]}'
            )
            + "只返回需要修改的连接线；source_id/target_id 必须来自元素列表，无法判断则为 null。"
            f"元素列表：{json.dumps(self._summarize_elements(context.elements), ensure_ascii=False)}"
        )
        try:
            data = self._parse_json_response_with_debug(self.client.analyze_image(context.image_path, prompt), "layout_refine")
        except Exception as exc:
            print(f"[VLMEnhancer] layout refine skipped: {exc}", flush=True)
            return {"updated": 0, "edges": []}
        edges = data.get("edges", []) if isinstance(data.get("edges"), list) else []
        by_id = {elem.id: elem for elem in context.elements}
        applied = []
        changes = []
        for edge in edges:
            if not isinstance(edge, dict) or self._confidence(edge) < threshold:
                continue
            elem = by_id.get(edge.get("id"))
            if not elem:
                continue
            before = {
                "type": elem.element_type,
                "line_style": elem.line_style,
                "source_id": elem.source_id,
                "target_id": elem.target_id,
            }
            edge_type = self._sanitize_type(edge.get("type"))
            if edge_type in {"arrow", "connector", "line"}:
                elem.element_type = edge_type
            if edge.get("line_style") in {"solid", "dashed"}:
                elem.line_style = edge.get("line_style")
            if edge.get("source_id") is not None:
                elem.source_id = int(edge.get("source_id"))
            if edge.get("target_id") is not None:
                elem.target_id = int(edge.get("target_id"))
            elem.processing_notes.append("vlm_layout_refine")
            applied.append(edge)
            after = {
                "type": elem.element_type,
                "line_style": elem.line_style,
                "source_id": elem.source_id,
                "target_id": elem.target_id,
            }
            if before != after:
                changes.append({"id": elem.id, "from": before, "to": after, "confidence": self._confidence(edge)})
        result = {"updated": len(changes), "edges": applied, "changes": changes}
        self._print_json("layout_refine applied", changes)
        self._write_artifact(context, "vlm_layout_refine.json", result)
        return result



    @staticmethod
    def _summarize_ocr_context(context: ProcessingContext, max_blocks: int = 60) -> List[Dict[str, Any]]:
        """Return compact OCR text anchors for VLM segmentation/attribute passes."""
        blocks = []
        if hasattr(context, "intermediate_results"):
            blocks = context.intermediate_results.get("ocr_text_blocks") or []
        summary = []
        for idx, block in enumerate(blocks[:max_blocks]):
            geo = block.get("geometry") or {}
            summary.append(
                {
                    "id": block.get("id", idx),
                    "text": block.get("text"),
                    "bbox": [
                        int(geo.get("x", 0)),
                        int(geo.get("y", 0)),
                        int(geo.get("x", 0) + geo.get("width", 0)),
                        int(geo.get("y", 0) + geo.get("height", 0)),
                    ],
                }
            )
        return summary

    @staticmethod
    def _summarize_vlm_regions(regions: List[Dict[str, Any]], max_items: int = 20) -> List[Dict[str, Any]]:
        """Keep region context compact before injecting it into a VLM prompt."""
        summary = []
        for idx, region in enumerate((regions or [])[:max_items]):
            if not isinstance(region, dict):
                continue
            summary.append(
                {
                    "id": region.get("id", f"r{idx}"),
                    "type": region.get("type"),
                    "semantic_type": region.get("semantic_type") or region.get("subtype") or region.get("type"),
                    "bbox": region.get("bbox") or region.get("bounding_box") or region.get("box"),
                    "confidence": region.get("confidence"),
                }
            )
        return summary

    @staticmethod
    def _image_for_stage(context: ProcessingContext, preferred_key: str) -> str:
        """Use ID-labelled overlays for object mapping when available."""
        if hasattr(context, "intermediate_results"):
            path = context.intermediate_results.get(preferred_key)
            if path and os.path.exists(path):
                return path
        return context.image_path

    def refine_segmentation(self, context: ProcessingContext) -> Dict[str, Any]:
        """Use VLM to add/correct coarse SAM3 segmentation objects before attribute enrichment."""
        if not self._enabled_for("segmentation_refine") or not context.elements:
            return {"updated": 0, "added": 0, "items": [], "changes": []}
        threshold = float(self.thresholds.get("segmentation_refine_confidence", 0.70))
        ocr_context = self._summarize_ocr_context(context)
        prompt = (
            "你是图表分割结果增补和纠错器。SAM3 已输出候选对象，请基于带对象ID的标注图、原图视觉内容和 OCR 文本锚点补充漏检对象并纠正明显类型错误。"
            + self._json_only_instruction(
                '{"elements":[{"id":2,"type":"container","bbox":[10,20,200,120],"action":"update","confidence":0.86},{"type":"arrow","bbox":[1,2,30,40],"action":"add","confidence":0.82}]}'
            )
            + "action 只能是 update 或 add；update 必须带已有 id；add 必须给 bbox。type 只能选 icon,picture,rectangle,rounded rectangle,circle,ellipse,cylinder,diamond,triangle,hexagon,container,arrow,connector,line。"
            "重点检查：容器内部的小模块/表格/卡片是否漏检；水平或垂直分割线是否漏检；带顶部椭圆的数据库/分层共享块应为 cylinder 而不是 rounded rectangle；外层分组边框/区域框应为 container 而不是 connector。"
            f"OCR文本锚点：{json.dumps(ocr_context, ensure_ascii=False)}；SAM3对象：{json.dumps(self._summarize_elements(context.elements), ensure_ascii=False)}"
        )
        try:
            data = self._parse_json_response_with_debug(self.client.analyze_image(self._image_for_stage(context, "sam3_visualization"), prompt), "segmentation_refine")
        except Exception as exc:
            print(f"[VLMEnhancer] segmentation refine skipped: {exc}", flush=True)
            return {"updated": 0, "added": 0, "items": [], "changes": []}
        items = data.get("elements", []) if isinstance(data.get("elements"), list) else []
        by_id = {elem.id: elem for elem in context.elements}
        next_id = max(by_id.keys(), default=-1) + 1
        changes = []
        added = []
        for item in items:
            if not isinstance(item, dict) or self._confidence(item) < threshold:
                continue
            action = str(item.get("action") or "update").lower()
            new_type = self._sanitize_type(item.get("type"))
            if not new_type:
                continue
            if action == "add":
                bbox = item.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    continue
                elem = ElementInfo(
                    id=next_id,
                    element_type=new_type,
                    bbox=BoundingBox.from_list([int(v) for v in bbox]),
                    score=self._confidence(item),
                    source_prompt="vlm_segmentation_refine",
                )
                next_id += 1
                elem.processing_notes.append("vlm_segmentation_refine:add")
                context.elements.append(elem)
                added.append(elem.to_dict())
                continue
            elem = by_id.get(item.get("id"))
            if not elem:
                continue
            before = {"type": elem.element_type, "bbox": elem.bbox.to_list()}
            elem.element_type = new_type
            bbox = item.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                elem.bbox = BoundingBox.from_list([int(v) for v in bbox])
            elem.processing_notes.append("vlm_segmentation_refine:update")
            after = {"type": elem.element_type, "bbox": elem.bbox.to_list()}
            if before != after:
                changes.append({"id": elem.id, "from": before, "to": after, "confidence": self._confidence(item)})
        result = {"updated": len(changes), "added": len(added), "items": items, "changes": changes, "added_items": added}
        self._print_json("segmentation_refine applied", {"changes": changes, "added": added})
        self._write_artifact(context, "vlm_segmentation_refine.json", result)
        return result

    def enrich_element_attributes(self, context: ProcessingContext) -> Dict[str, Any]:
        """Ask VLM to add export attributes to each SAM3/VLM element."""
        if not self._enabled_for("element_attributes") or not context.elements:
            return {"updated": 0, "items": [], "changes": []}
        threshold = float(self.thresholds.get("element_attribute_confidence", 0.65))
        ocr_context = self._summarize_ocr_context(context)
        prompt = (
            "你是图表元素属性识别器。请基于带对象ID的标注图、原图视觉内容、OCR 文本锚点和对象列表，为每个 SAM3 对象补充 PPTX/图形导出所需属性。"
            + self._json_only_instruction(
                '{"elements":[{"id":1,"type":"arrow","line_style":"dashed","arrow_heads":"end","source_id":2,"target_id":3,"fill_color":"#ffffff","stroke_color":"#333333","stroke_width":2,"corner_radius":8,"confidence":0.9}]}'
            )
            + "只返回需要补充/修正的属性。arrow_heads 只能是 none/start/end/both；双向箭头必须返回 arrow_heads=both；单向箭头 source_id 到 target_id 必须表示箭头方向；如果方向不确定则只给 arrow_heads，不要猜 source_id/target_id；line_style 只能是 solid/dashed/dotted/null；颜色必须是 #RRGGBB。"
            "重点纠正：右侧分层共享区域的外框/分组框应为 container；带顶部椭圆的高层共享/中层共享/底层共享块应为 cylinder；底部分割线应为 line/connector；共享状态容器内部的模块/表格/小矩形应保留为 shape/container。"
            f"OCR文本锚点：{json.dumps(ocr_context, ensure_ascii=False)}；对象列表：{json.dumps(self._summarize_elements(context.elements), ensure_ascii=False)}"
        )
        try:
            data = self._parse_json_response_with_debug(self.client.analyze_image(self._image_for_stage(context, "sam3_vlm_refined_visualization"), prompt), "element_attributes")
        except Exception as exc:
            print(f"[VLMEnhancer] element attributes skipped: {exc}", flush=True)
            return {"updated": 0, "items": [], "changes": []}
        items = data.get("elements", []) if isinstance(data.get("elements"), list) else []
        by_id = {elem.id: elem for elem in context.elements}
        changes = []
        for item in items:
            if not isinstance(item, dict) or self._confidence(item) < threshold:
                continue
            elem = by_id.get(item.get("id"))
            if not elem:
                continue
            before = {
                "type": elem.element_type,
                "line_style": elem.line_style,
                "arrow_heads": elem.arrow_heads,
                "fill_color": elem.fill_color,
                "stroke_color": elem.stroke_color,
                "stroke_width": elem.stroke_width,
                "corner_radius": elem.corner_radius,
                "source_id": elem.source_id,
                "target_id": elem.target_id,
            }
            new_type = self._sanitize_type(item.get("type"))
            if new_type:
                elem.element_type = new_type
            line_style = self._normalize_line_style(item.get("line_style"))
            if line_style:
                elem.line_style = line_style
            arrow_heads = self._normalize_arrow_heads(item.get("arrow_heads"))
            if arrow_heads:
                elem.arrow_heads = arrow_heads
            fill = self._valid_hex_color(item.get("fill_color"))
            stroke = self._valid_hex_color(item.get("stroke_color"))
            if fill:
                elem.fill_color = fill
            if stroke:
                elem.stroke_color = stroke
            try:
                if item.get("stroke_width") is not None:
                    elem.stroke_width = max(1, min(12, int(round(float(item.get("stroke_width"))))))
            except (TypeError, ValueError):
                pass
            try:
                if item.get("corner_radius") is not None:
                    elem.corner_radius = max(0.0, min(100.0, float(item.get("corner_radius"))))
            except (TypeError, ValueError):
                pass
            if item.get("source_id") is not None:
                elem.source_id = int(item.get("source_id"))
            if item.get("target_id") is not None:
                elem.target_id = int(item.get("target_id"))
            elem.processing_notes.append("vlm_element_attributes")
            after = {
                "type": elem.element_type,
                "line_style": elem.line_style,
                "arrow_heads": elem.arrow_heads,
                "fill_color": elem.fill_color,
                "stroke_color": elem.stroke_color,
                "stroke_width": elem.stroke_width,
                "corner_radius": elem.corner_radius,
                "source_id": elem.source_id,
                "target_id": elem.target_id,
            }
            if before != after:
                changes.append({"id": elem.id, "from": before, "to": after, "confidence": self._confidence(item)})
        result = {"updated": len(changes), "items": items, "changes": changes, "elements": [e.to_dict() for e in context.elements]}
        self._print_json("element_attributes applied", changes)
        self._write_artifact(context, "vlm_element_attributes.json", result)
        return result

    def validate_pptx_export(self, context: ProcessingContext, pptx_path: str, round_index: int = 1) -> Dict[str, Any]:
        """Score final PPTX against the original image and request structured repair hints."""
        if not self._enabled_for("export_validate") or not pptx_path:
            return {"checked": False, "score": 100, "pass": True}
        prompt = (
            "你是图表 PPTX 导出质量评估器。请对比原图和当前元素结构，评估最终 PPTX 还原质量。"
            + self._json_only_instruction(
                '{"score":92,"pass":true,"issues":[{"severity":"medium","description":"missing dashed connector","element_id":3}],"repairs":{"elements":[{"id":3,"line_style":"dashed","confidence":0.9}]}}'
            )
            + "score 为 0-100；低于90时 pass=false，并在 repairs.elements 中给出可直接应用到元素的修补增量。"
            f"pptx_output={pptx_path}; round={round_index}; elements={json.dumps(self._summarize_elements(context.elements), ensure_ascii=False)}"
        )
        try:
            data = self._parse_json_response_with_debug(self.client.analyze_image(context.image_path, prompt), "pptx_validate")
        except Exception as exc:
            print(f"[VLMEnhancer] PPTX validation skipped: {exc}", flush=True)
            return {"checked": False, "score": 100, "pass": True, "error": str(exc)}
        score = float(data.get("score", 100 if data.get("pass") else 0) or 0)
        result = {"checked": True, "score": score, "pass": bool(data.get("pass", score >= 90)), "report": data}
        self._print_json("pptx_validate report", data)
        self._write_artifact(context, f"vlm_pptx_validation_round_{round_index}.json", result)
        return result

    def apply_export_repairs(self, context: ProcessingContext, report: Dict[str, Any]) -> Dict[str, Any]:
        """Apply conservative structured repairs returned by validate_pptx_export."""
        repairs = (report.get("report") or {}).get("repairs") if isinstance(report, dict) else None
        if not isinstance(repairs, dict):
            return {"updated": 0, "added": 0, "changes": []}
        items = repairs.get("elements", []) if isinstance(repairs.get("elements"), list) else []
        by_id = {elem.id: elem for elem in context.elements}
        changes = []
        added = []
        next_id = max(by_id.keys(), default=-1) + 1
        for item in items:
            if not isinstance(item, dict) or self._confidence(item, 0.8) < 0.55:
                continue
            elem = by_id.get(item.get("id"))
            if elem is None and isinstance(item.get("bbox"), list) and len(item.get("bbox")) == 4:
                new_type = self._sanitize_type(item.get("type")) or "rectangle"
                elem = ElementInfo(
                    id=next_id,
                    element_type=new_type,
                    bbox=BoundingBox.from_list([int(v) for v in item.get("bbox")]),
                    score=self._confidence(item),
                    source_prompt="vlm_export_repair",
                )
                next_id += 1
                context.elements.append(elem)
                added.append(elem.id)
            if elem is None:
                continue
            before = elem.to_dict()
            new_type = self._sanitize_type(item.get("type"))
            if new_type:
                elem.element_type = new_type
            line_style = self._normalize_line_style(item.get("line_style"))
            if line_style:
                elem.line_style = line_style
            arrow_heads = self._normalize_arrow_heads(item.get("arrow_heads"))
            if arrow_heads:
                elem.arrow_heads = arrow_heads
            for attr in ("fill_color", "stroke_color"):
                color = self._valid_hex_color(item.get(attr))
                if color:
                    setattr(elem, attr, color)
            if item.get("source_id") is not None:
                elem.source_id = int(item.get("source_id"))
            if item.get("target_id") is not None:
                elem.target_id = int(item.get("target_id"))
            elem.xml_fragment = None
            elem.processing_notes.append("vlm_export_repair")
            after = elem.to_dict()
            if before != after:
                changes.append({"id": elem.id, "from": before, "to": after})
        result = {"updated": len(changes), "added": len(added), "changes": changes}
        self._print_json("export_repair applied", result)
        self._write_artifact(context, "vlm_export_repairs.json", result)
        return result

    def validate_export(self, context: ProcessingContext, output_path: str) -> Dict[str, Any]:
        """Ask VLM for a final export quality report before PPTX conversion."""
        if not self._enabled_for("export_validate") or not output_path:
            return {"checked": False}
        prompt = (
            "你是图表导出质量审查器。根据原图、元素摘要和 draw.io 输出路径信息，结构化列出质量问题。"
            + self._json_only_instruction(
                '{"issues":[{"severity":"medium","description":"missing dashed connector","element_id":3}],"pass":false}'
            )
            + "issues 中 description 要简短明确；没有问题返回 {\"issues\":[],\"pass\":true}。"
            f"drawio_output={output_path}; elements={json.dumps(self._summarize_elements(context.elements), ensure_ascii=False)}"
        )
        try:
            data = self._parse_json_response_with_debug(self.client.analyze_image(context.image_path, prompt), "export_validate")
        except Exception as exc:
            print(f"[VLMEnhancer] export validation skipped: {exc}", flush=True)
            return {"checked": False, "error": str(exc)}
        result = {"checked": True, "report": data}
        self._print_json("export_validate report", data)
        self._write_artifact(context, "vlm_export_validation.json", result)
        return result
