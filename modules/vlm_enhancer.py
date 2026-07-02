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


from .vlm_enhancer_structure import VLMStructureMixin
from .vlm_enhancer_refinement import VLMRefinementMixin

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


class VLMEnhancer(VLMStructureMixin, VLMRefinementMixin):
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
                    if isinstance(content, list):
                        parts = []
                        for part in content:
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                parts.append(part["text"])
                        if parts:
                            return "\n".join(parts)
                text = choices[0].get("text") if isinstance(choices[0], dict) else None
                if isinstance(text, str):
                    return text
            if any(key in response for key in ("image", "shape", "arrow", "background", "elements", "edges", "text_blocks", "score", "repairs")):
                return json.dumps(response, ensure_ascii=False)
        return ""

    @classmethod
    def _parse_json_response(cls, response: Any) -> Dict[str, Any]:
        """Parse JSON returned by VLM, including fenced Markdown JSON."""
        text = cls._normalize_json_text(cls._extract_response_text(response))
        if not text:
            return {}
        parsed = cls._loads_json_object(text)
        if isinstance(parsed, dict):
            return parsed
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            parsed = cls._loads_json_object(match.group(0))
            if isinstance(parsed, dict):
                return parsed
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
            print(
                f"[VLMEnhancer] {stage} returned non-JSON response: "
                + json.dumps(
                    {
                        "chars": len(text),
                        "parse_error": True,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return {}

    @classmethod
    def _parse_and_coerce_stage_json(cls, text: str, stage: str) -> Dict[str, Any]:
        """Parse JSON that is valid but not wrapped in the expected top-level object."""
        text = cls._normalize_json_text(text)
        if not text:
            return {}
        parsed = cls._loads_json_object(text)
        if parsed is None:
            match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
            if not match:
                return {}
            parsed = cls._loads_json_object(match.group(1))
            if parsed is None:
                return {}
        if isinstance(parsed, dict):
            return parsed
        if not isinstance(parsed, list):
            return {}
        if stage in {"element_refine", "region_refine", "segmentation_refine", "element_attributes", "export_repair", "vlm_region_elements"}:
            return {"elements": parsed}
        if stage == "vlm_page_regions":
            return {"regions": parsed}
        if stage in {"layout_refine", "vlm_connectors"}:
            return {"edges": parsed}
        if stage == "text_style":
            return {"text_blocks": parsed}
        if stage in {"export_validate", "pptx_validate"}:
            return {"issues": parsed, "pass": False}
        return {}

    @staticmethod
    def _normalize_json_text(text: str) -> str:
        """Normalize common VLM JSON wrappers before parsing.

        Some providers return Markdown fences with literal ``\\n`` sequences
        instead of real newlines (for example `````json\\n{...}\\n`````), which
        looks readable in logs but fails the normal fenced-JSON stripping path.
        """
        normalized = str(text or "").strip()
        if not normalized:
            return ""
        if "\\n" in normalized and "\n" not in normalized:
            normalized = normalized.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
        if normalized.startswith("```"):
            normalized = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", normalized, flags=re.IGNORECASE)
            normalized = re.sub(r"\s*```\s*$", "", normalized)
        return normalized.strip()

    @staticmethod
    def _loads_json_object(text: str) -> Any:
        """Load JSON with a conservative repair pass for provider line-wrap noise."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            repaired = VLMEnhancer._repair_dirty_json_text(text)
            if repaired == text:
                return None
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return None

    @staticmethod
    def _repair_dirty_json_text(text: str) -> str:
        """Repair common invalid JSON produced by VLM line wrapping.

        Examples seen in provider output include keys split as
        ``"font_\nfamily"`` and escaped newlines before the next key such as
        ``{\\nn "enabled": false}``. This keeps value strings intact while
        normalizing only key tokens and escaped newlines that precede a key.
        """
        repaired = str(text or "")
        repaired = re.sub(r"\\n+\s*(?=\")", "\n", repaired)

        def fix_key(match: re.Match) -> str:
            key = re.sub(r"[\r\n]\s*", "", match.group(1))
            return f'"{key}":'

        previous = None
        key_pattern = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"\s*:', flags=re.DOTALL)
        while previous != repaired:
            previous = repaired
            repaired = key_pattern.sub(fix_key, repaired)
        return repaired

    @staticmethod
    def _print_json(label: str, data: Any) -> None:
        """Print only PPTX-relevant VLM data instead of the full raw response."""
        text = json.dumps(VLMEnhancer._pptx_log_payload(data), ensure_ascii=False)
        print(f"[VLMEnhancer] {label}: {text}", flush=True)

    @staticmethod
    def _pptx_log_payload(data: Any, max_items: int = 30) -> Any:
        """Keep logs focused on fields used by PPTX reconstruction."""
        if isinstance(data, list):
            return {
                "count": len(data),
                "items": [VLMEnhancer._pptx_log_item(item) for item in data[:max_items]],
                "truncated": len(data) > max_items,
            }
        if not isinstance(data, dict):
            return data

        payload: Dict[str, Any] = {}
        for key in ("score", "pass", "updated", "added", "count", "raw_count", "dropped_count", "recognized", "error"):
            if key in data:
                payload[key] = data[key]

        for key in ("elements", "text_blocks", "changes", "items", "issues"):
            values = data.get(key)
            if isinstance(values, list):
                payload[key] = {
                    "count": len(values),
                    "items": [VLMEnhancer._pptx_log_item(item) for item in values[:max_items]],
                    "truncated": len(values) > max_items,
                }
            elif isinstance(values, dict):
                compact_values = VLMEnhancer._coerce_vlm_item_collection(values)
                payload[key] = {
                    "count": len(compact_values),
                    "items": [VLMEnhancer._pptx_log_item(item) for item in compact_values[:max_items]],
                    "truncated": len(compact_values) > max_items,
                }

        repairs = data.get("repairs")
        if isinstance(repairs, dict):
            payload["repairs"] = VLMEnhancer._pptx_log_payload(repairs, max_items=max_items)

        if "background" in data:
            payload["background"] = VLMEnhancer._pptx_log_item(data["background"])
        if "reconstruction_summary" in data and isinstance(data["reconstruction_summary"], dict):
            summary = data["reconstruction_summary"]
            payload["reconstruction_summary"] = {
                key: summary.get(key)
                for key in (
                    "native_text_count",
                    "native_shape_count",
                    "native_line_count",
                    "cropped_image_count",
                    "image_fallback_count",
                    "overall_reconstruction_confidence",
                )
                if key in summary
            }

        return payload or {"type": type(data).__name__, "keys": sorted(data.keys())[:20]}

    @staticmethod
    def _pptx_log_item(item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        compact: Dict[str, Any] = {}
        for key in (
            "id",
            "element_id",
            "source_element_id",
            "type",
            "element_type",
            "subtype",
            "bbox",
            "content",
            "text",
            "font_size",
            "font_size_estimate",
            "font_color",
            "text_align",
            "fill_color",
            "stroke_color",
            "stroke_width",
            "line_style",
            "arrow_heads",
            "source_id",
            "target_id",
            "confidence",
            "score",
            "severity",
            "description",
            "action",
        ):
            if key in item:
                compact[key] = item[key]
        style = item.get("style")
        if isinstance(style, dict):
            for key in ("font_size", "font_size_estimate", "font_color", "text_align", "fill", "stroke"):
                if key in style:
                    compact.setdefault(key, style[key])
        return compact or {"keys": sorted(item.keys())[:12]}

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
            "semantic_type": getattr(elem, "semantic_type", None),
            "reconstruction_strategy": getattr(elem, "reconstruction_strategy", None),
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

