"""Shared VLM enhancer parsing, logging, and normalization helpers."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from .base import ProcessingContext
from .data_types import ElementInfo

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


class VLMCoreMixin:
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
            if any(key in response for key in ("image", "shape", "arrow", "background", "regions", "page", "layout_pattern", "page_structure", "elements", "edges", "text_blocks", "score", "repairs")):
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
            repaired = VLMCoreMixin._repair_dirty_json_text(text)
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
        text = json.dumps(VLMCoreMixin._pptx_log_payload(data), ensure_ascii=False)
        print(f"[VLMEnhancer] {label}: {text}", flush=True)

    @staticmethod
    def _pptx_log_payload(data: Any, max_items: int = 30) -> Any:
        """Keep logs focused on fields used by PPTX reconstruction."""
        if isinstance(data, list):
            return {
                "count": len(data),
                "items": [VLMCoreMixin._pptx_log_item(item) for item in data[:max_items]],
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
                    "items": [VLMCoreMixin._pptx_log_item(item) for item in values[:max_items]],
                    "truncated": len(values) > max_items,
                }
            elif isinstance(values, dict):
                compact_values = VLMCoreMixin._coerce_vlm_item_collection(values)
                payload[key] = {
                    "count": len(compact_values),
                    "items": [VLMCoreMixin._pptx_log_item(item) for item in compact_values[:max_items]],
                    "truncated": len(compact_values) > max_items,
                }

        repairs = data.get("repairs")
        if isinstance(repairs, dict):
            payload["repairs"] = VLMCoreMixin._pptx_log_payload(repairs, max_items=max_items)

        if "background" in data:
            payload["background"] = VLMCoreMixin._pptx_log_item(data["background"])
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

