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


class VLMEnhancer:
    """Runs optional VLM prompt, element, region, layout, and export passes."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.root_config = config or {}
        self.config = (self.root_config.get("multimodal") or {})
        self.enabled = bool(self.config.get("enabled", False))
        self.use_for = self.config.get("use_for") or {}
        self.thresholds = self.config.get("thresholds") or {}
        self.max_elements = int(self.config.get("max_elements_for_vlm", 80) or 80)
        self.client = create_vlm_client_from_config(self.root_config) if self.enabled else None

    def _enabled_for(self, feature: str) -> bool:
        return bool(self.enabled and self.use_for.get(feature, False) and self.client is not None)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Extract text content from common chat-completion or direct JSON responses."""
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            if isinstance(response.get("content"), str):
                return response["content"]
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
                text = choices[0].get("text") if isinstance(choices[0], dict) else None
                if isinstance(text, str):
                    return text
            if any(key in response for key in ("image", "shape", "arrow", "background", "elements", "edges")):
                return json.dumps(response, ensure_ascii=False)
        return ""

    @classmethod
    def _parse_json_response(cls, response: Any) -> Dict[str, Any]:
        """Parse JSON returned by VLM, including fenced Markdown JSON."""
        text = cls._extract_response_text(response).strip()
        if not text:
            return {}
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                    return parsed if isinstance(parsed, dict) else {}
                except json.JSONDecodeError:
                    return {}
        return {}

    @classmethod
    def _parse_json_response_with_debug(cls, response: Any, stage: str) -> Dict[str, Any]:
        """Parse VLM JSON and log non-structured responses for prompt tuning."""
        parsed = cls._parse_json_response(response)
        if parsed:
            return parsed
        text = cls._extract_response_text(response).strip()
        if text:
            preview = text[:1000] + ("...<truncated>" if len(text) > 1000 else "")
            print(f"[VLMEnhancer] {stage} returned non-JSON response: {preview}", flush=True)
        return {}

    @staticmethod
    def _json_only_instruction(schema: str) -> str:
        """Return strict output instructions shared by VLM enhancement prompts."""
        return (
            "输出必须是一个可被 json.loads 直接解析的 JSON 对象。"
            "不要输出 Markdown 代码块，不要输出解释、分析、口语化描述或额外文本。"
            "如果没有可修改内容，也必须返回符合 schema 的空数组/空报告。"
            f"JSON schema 示例：{schema}"
        )

    @staticmethod
    def _element_summary(elem: ElementInfo) -> Dict[str, Any]:
        return {
            "id": elem.id,
            "type": elem.element_type,
            "bbox": elem.bbox.to_list(),
            "score": elem.score,
            "source_prompt": elem.source_prompt,
            "line_style": elem.line_style,
            "source_id": elem.source_id,
            "target_id": elem.target_id,
        }

    def _summarize_elements(self, elements: List[ElementInfo]) -> List[Dict[str, Any]]:
        return [self._element_summary(elem) for elem in elements[: self.max_elements]]

    @staticmethod
    def _sanitize_type(element_type: str) -> Optional[str]:
        normalized = str(element_type or "").strip().lower().replace("_", " ")
        aliases = {
            "rounded_rectangle": "rounded rectangle",
            "rounded rect": "rounded rectangle",
            "database": "cylinder",
            "database cylinder": "cylinder",
            "dotted connector": "connector",
            "dashed connector": "connector",
            "dashed line": "connector",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in CANONICAL_TYPES else None

    @staticmethod
    def _confidence(item: Dict[str, Any], default: float = 1.0) -> float:
        try:
            return float(item.get("confidence", default))
        except (TypeError, ValueError):
            return default

    def _write_artifact(self, context: ProcessingContext, name: str, data: Dict[str, Any]) -> Optional[str]:
        output_dir = getattr(context, "output_dir", "") or ""
        if not output_dir:
            return None
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        context.intermediate_results[name] = path
        return path

    def plan_prompts(self, image_path: str) -> Dict[str, List[str]]:
        """Ask VLM for image-specific SAM3 prompt additions."""
        if not self._enabled_for("prompt_planning"):
            return {}
        prompt = (
            "你是图表元素识别提示词规划器。请观察图片，为 SAM3 分割补充英文 prompt。"
            + self._json_only_instruction(
                '{"image":["icon"],"shape":["rounded rectangle"],"arrow":["dashed connector line"],"background":["outer panel"]}'
            )
            + "键只能是 image、shape、arrow、background；每个值必须是英文 prompt 字符串数组。"
            "重点包含图标、卡片、圆柱数据库、虚线/实线连接器、容器面板。"
        )
        try:
            data = self._parse_json_response_with_debug(self.client.analyze_image(image_path, prompt), "prompt_planning")
        except Exception as exc:
            print(f"[VLMEnhancer] prompt planning skipped: {exc}", flush=True)
            return {}
        planned: Dict[str, List[str]] = {}
        for key in ("image", "shape", "arrow", "background"):
            values = data.get(key, [])
            if isinstance(values, list):
                planned[key] = [str(v).strip() for v in values if str(v).strip()]
        return planned

    def apply_prompt_plan(self, extractor: Any, image_path: str, output_dir: str = "") -> Dict[str, List[str]]:
        planned = self.plan_prompts(image_path)
        if not planned:
            return {}
        mapping = {
            "image": PromptGroup.IMAGE,
            "shape": PromptGroup.BASIC_SHAPE,
            "arrow": PromptGroup.ARROW,
            "background": PromptGroup.BACKGROUND,
        }
        for key, prompts in planned.items():
            group = mapping.get(key)
            if group and prompts:
                extractor.add_prompts_to_group(group, prompts)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "vlm_prompts.json"), "w", encoding="utf-8") as f:
                json.dump(planned, f, ensure_ascii=False, indent=2)
        return planned

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
        for item in updates:
            if not isinstance(item, dict) or self._confidence(item) < threshold:
                continue
            elem = by_id.get(item.get("id"))
            new_type = self._sanitize_type(item.get("type"))
            if not elem or not new_type:
                continue
            old_type = elem.element_type
            if new_type:
                elem.element_type = new_type
            line_style = item.get("line_style")
            if line_style in {"solid", "dashed"}:
                elem.line_style = line_style
            elem.processing_notes.append(f"vlm_element_refine: {old_type}->{elem.element_type}")
            applied.append(item)
        result = {"updated": len(applied), "items": applied}
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
        for edge in edges:
            if not isinstance(edge, dict) or self._confidence(edge) < threshold:
                continue
            elem = by_id.get(edge.get("id"))
            if not elem:
                continue
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
        result = {"updated": len(applied), "edges": applied}
        self._write_artifact(context, "vlm_layout_refine.json", result)
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
        self._write_artifact(context, "vlm_export_validation.json", result)
        return result
