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
            if any(key in response for key in ("image", "shape", "arrow", "background", "elements", "edges", "text_blocks", "score", "repairs")):
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
            cls._print_json(f"{stage} parsed", parsed)
            return parsed
        text = cls._extract_response_text(response).strip()
        coerced = cls._parse_and_coerce_stage_json(text, stage)
        if coerced:
            cls._print_json(f"{stage} parsed", coerced)
            return coerced
        if text:
            preview = text[:1000] + ("...<truncated>" if len(text) > 1000 else "")
            print(f"[VLMEnhancer] {stage} returned non-JSON response: {preview}", flush=True)
        return {}

    @classmethod
    def _parse_and_coerce_stage_json(cls, text: str, stage: str) -> Dict[str, Any]:
        """Parse JSON that is valid but not wrapped in the expected top-level object."""
        if not text:
            return {}
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
            if not match:
                return {}
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                return {}
        if isinstance(parsed, dict):
            return parsed
        if not isinstance(parsed, list):
            return {}
        if stage in {"element_refine", "region_refine", "segmentation_refine", "element_attributes", "export_repair"}:
            return {"elements": parsed}
        if stage == "layout_refine":
            return {"edges": parsed}
        if stage == "text_style":
            return {"text_blocks": parsed}
        if stage in {"export_validate", "pptx_validate"}:
            return {"issues": parsed, "pass": False}
        return {}

    @staticmethod
    def _print_json(label: str, data: Any, max_chars: int = 4000) -> None:
        """Print bounded structured VLM data, not raw provider responses."""
        text = json.dumps(data, ensure_ascii=False)
        if len(text) > max_chars:
            text = text[:max_chars] + "...<truncated>"
        print(f"[VLMEnhancer] {label}: {text}", flush=True)

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


    @staticmethod
    def _valid_hex_color(value: Any) -> Optional[str]:
        text = str(value or "").strip()
        if re.fullmatch(r"#[0-9a-fA-F]{6}", text):
            return text.lower()
        return None

    @staticmethod
    def _normalize_line_style(value: Any) -> Optional[str]:
        text = str(value or "").strip().lower()
        if text in {"solid", "dashed", "dotted"}:
            return text
        if text in {"null", "none", ""}:
            return None
        return None

    @staticmethod
    def _normalize_arrow_heads(value: Any) -> Optional[str]:
        text = str(value or "").strip().lower().replace("_", " ")
        aliases = {
            "no": "none",
            "none": "none",
            "start": "start",
            "source": "start",
            "end": "end",
            "target": "end",
            "single": "end",
            "one way": "end",
            "both": "both",
            "double": "both",
            "bidirectional": "both",
            "two way": "both",
        }
        return aliases.get(text)

    @staticmethod
    def _summarize_text_blocks(text_blocks: List[Dict[str, Any]], max_blocks: int = 80) -> List[Dict[str, Any]]:
        summary = []
        for idx, block in enumerate(text_blocks[:max_blocks]):
            geometry = block.get("geometry") or {}
            summary.append(
                {
                    "id": block.get("id", idx),
                    "text": block.get("text"),
                    "geometry": geometry,
                    "confidence": block.get("confidence"),
                    "font_size": block.get("font_size"),
                    "font_color": block.get("font_color"),
                    "font_family": block.get("font_family"),
                    "font_weight": block.get("font_weight"),
                    "font_style": block.get("font_style"),
                }
            )
        return summary

    def enrich_text_styles(self, context: ProcessingContext, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Ask VLM to supplement OCR text style attributes while keeping OCR text/bboxes as anchors."""
        if not self._enabled_for("text_style") or not text_blocks:
            return {"updated": 0, "text_blocks": text_blocks, "changes": []}
        threshold = float(self.thresholds.get("text_style_confidence", 0.65))
        for idx, block in enumerate(text_blocks):
            block.setdefault("id", idx)
        prompt = (
            "你是图中文字样式识别器。OCR 已经给出文字和位置，请只补充/修正每个文字块的样式属性。"
            + self._json_only_instruction(
                '{"text_blocks":[{"id":0,"font_family":"Microsoft YaHei","font_size":18,"font_color":"#333333","font_weight":"normal|bold","font_style":"normal|italic","confidence":0.9}]}'
            )
            + "不要改文字内容和坐标；只返回需要补充或修正样式的文字块。font_color 必须是 #RRGGBB；不确定的字段省略。"
            f"OCR文字块：{json.dumps(self._summarize_text_blocks(text_blocks, self.max_elements), ensure_ascii=False)}"
        )
        try:
            data = self._parse_json_response_with_debug(self.client.analyze_image(context.image_path, prompt), "text_style")
        except Exception as exc:
            print(f"[VLMEnhancer] text style skipped: {exc}", flush=True)
            return {"updated": 0, "text_blocks": text_blocks, "changes": []}
        updates = data.get("text_blocks", []) if isinstance(data.get("text_blocks"), list) else []
        by_id = {block.get("id", idx): block for idx, block in enumerate(text_blocks)}
        changes = []
        allowed_weights = {"normal", "bold"}
        allowed_styles = {"normal", "italic"}
        for item in updates:
            if not isinstance(item, dict) or self._confidence(item) < threshold:
                continue
            block = by_id.get(item.get("id"))
            if block is None:
                continue
            before = {k: block.get(k) for k in ("font_family", "font_size", "font_color", "font_weight", "font_style")}
            if item.get("font_family"):
                block["font_family"] = str(item.get("font_family"))
            try:
                if item.get("font_size") is not None:
                    size = float(item.get("font_size"))
                    if 4 <= size <= 96:
                        block["font_size"] = size
            except (TypeError, ValueError):
                pass
            color = self._valid_hex_color(item.get("font_color"))
            if color:
                block["font_color"] = color
            weight = str(item.get("font_weight", "")).lower()
            if weight in allowed_weights:
                block["font_weight"] = weight
                block["is_bold"] = weight == "bold"
            style = str(item.get("font_style", "")).lower()
            if style in allowed_styles:
                block["font_style"] = style
                block["is_italic"] = style == "italic"
            after = {k: block.get(k) for k in before}
            if before != after:
                changes.append({"id": block.get("id"), "from": before, "to": after, "confidence": self._confidence(item)})
        result = {"updated": len(changes), "text_blocks": text_blocks, "changes": changes}
        self._print_json("text_style applied", changes)
        self._write_artifact(context, "vlm_text_styles.json", result)
        return result

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
        if planned:
            self._print_json("prompt_planning additions", planned)
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
        self._print_json(
            "prompt_planning applied",
            {key: len(prompts) for key, prompts in planned.items() if prompts},
        )
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "vlm_prompts.json"), "w", encoding="utf-8") as f:
                json.dump(planned, f, ensure_ascii=False, indent=2)
        return planned

    def recognize_structure(self, context: ProcessingContext) -> Dict[str, Any]:
        """Ask VLM to recognize the editable page structure without SAM3 candidates."""
        if not self.enabled or self.client is None:
            return {"recognized": False, "elements": [], "error": "multimodal VLM is disabled"}
        threshold = float(self.thresholds.get("vlm_structure_confidence", 0.60))
        canvas = {"width": context.canvas_width, "height": context.canvas_height}
        ocr_context = self._summarize_ocr_context(context, self.max_elements)
        prompt = (
            "你是页面结构等价识别器。请直接根据原图和 OCR 文本锚点识别可编辑 PPTX 结构，不要依赖 SAM3。"
            + self._json_only_instruction(
                '{"elements":[{"type":"container","bbox":[10,20,300,180],"fill_color":"none","stroke_color":"#333333","stroke_width":1,"line_style":"solid","arrow_heads":"none","arrow_start":[10,20],"arrow_end":[300,20],"confidence":0.9}]}'
            )
            + "只输出非文本图形元素；文本已经由 OCR 单独导出，不要为普通文字创建 text 元素。"
            "type 只能选 rectangle,rounded rectangle,circle,ellipse,cylinder,diamond,triangle,hexagon,container,arrow,connector,line,icon,picture,logo,chart。"
            "bbox 必须是原图像素坐标 [x1,y1,x2,y2]。容器/背景框用 container；普通卡片用 rounded rectangle 或 rectangle；数据库/顶部椭圆分层块用 cylinder。"
            "箭头和线条需要给 arrow_start/arrow_end 坐标；arrow_heads 只能是 none/start/end/both；line_style 只能是 solid/dashed/dotted/null；弧形箭头设置 arrow_style=curved。"
            "不要输出文字笔画、图标内部装饰线、截图选中框控制点或仅由 OCR 字符组成的伪元素。"
            f"画布：{json.dumps(canvas, ensure_ascii=False)}；OCR文本锚点：{json.dumps(ocr_context, ensure_ascii=False)}"
        )
        try:
            data = self._parse_json_response_with_debug(self.client.analyze_image(context.image_path, prompt), "vlm_structure")
        except Exception as exc:
            print(f"[VLMEnhancer] VLM-only structure recognition skipped: {exc}", flush=True)
            return {"recognized": False, "elements": [], "error": str(exc)}

        items = self._vlm_structure_items(data)
        elements: List[ElementInfo] = []
        for item in items:
            if not isinstance(item, dict) or self._confidence(item) < threshold:
                continue
            new_type = self._sanitize_type(
                item.get("type")
                or item.get("element_type")
                or item.get("shape_type")
                or item.get("kind")
            )
            bbox = self._extract_vlm_bbox(item)
            if not new_type or bbox is None:
                continue
            normalized_bbox = self._normalize_bbox(bbox, context.canvas_width, context.canvas_height)
            if normalized_bbox is None:
                continue
            elem = ElementInfo(
                id=len(elements),
                element_type=new_type,
                bbox=BoundingBox.from_list(normalized_bbox),
                score=self._confidence(item),
                source_prompt="vlm_structure",
            )
            self._apply_vlm_structure_attributes(elem, item)
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

    @staticmethod
    def _vlm_structure_items(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Accept common VLM wrappers so valid results are not silently ignored."""
        for key in ("elements", "page_elements", "objects", "shapes", "items"):
            values = data.get(key)
            if isinstance(values, list):
                return [item for item in values if isinstance(item, dict)]
        page = data.get("page") or data.get("diagram") or data.get("structure")
        if isinstance(page, dict):
            for key in ("elements", "page_elements", "objects", "shapes", "items"):
                values = page.get(key)
                if isinstance(values, list):
                    return [item for item in values if isinstance(item, dict)]
        return []

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

    def _apply_vlm_structure_attributes(self, elem: ElementInfo, item: Dict[str, Any]) -> None:
        style = item.get("style") if isinstance(item.get("style"), dict) else {}
        line_style = self._normalize_line_style(item.get("line_style") or style.get("line_style") or style.get("dash"))
        if line_style:
            elem.line_style = line_style
        arrow_heads = self._normalize_arrow_heads(item.get("arrow_heads") or item.get("end_arrow") or item.get("arrowhead"))
        if arrow_heads:
            elem.arrow_heads = arrow_heads
        arrow_style = str(item.get("arrow_style") or item.get("curve") or "").strip().lower()
        if arrow_style in {"curved", "curve", "arc"} or item.get("is_curved") is True:
            elem.arrow_style = "curved"
        fill_value = item.get("fill_color") or item.get("fill") or style.get("fill_color") or style.get("fill")
        stroke_value = item.get("stroke_color") or item.get("stroke") or style.get("stroke_color") or style.get("stroke")
        fill = self._valid_hex_color(fill_value)
        stroke = self._valid_hex_color(stroke_value)
        if fill:
            elem.fill_color = fill
        if str(fill_value or "").strip().lower() == "none":
            elem.fill_color = "none"
        if stroke:
            elem.stroke_color = stroke
        try:
            stroke_width = item.get("stroke_width", style.get("stroke_width"))
            if stroke_width is not None:
                elem.stroke_width = max(1, min(12, int(round(float(stroke_width)))))
        except (TypeError, ValueError):
            pass
        for attr, field_name in (("arrow_start", "arrow_start"), ("arrow_end", "arrow_end")):
            point = item.get(attr) or item.get("start" if attr == "arrow_start" else "end")
            if isinstance(point, list) and len(point) >= 2:
                try:
                    setattr(elem, field_name, (float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    pass

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
