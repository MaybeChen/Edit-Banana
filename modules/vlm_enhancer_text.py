"""Prompt-planning and text-style VLM enhancement passes."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from .base import ProcessingContext


class VLMTextPromptMixin:
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
        """Ask VLM for image-specific segmentation prompt additions."""
        if not self._enabled_for("prompt_planning"):
            return {}
        prompt = (
            "你是图表元素识别提示词规划器。请观察图片，为 segmentation 分割补充英文 prompt。"
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
        from .segmentation_info_extractor import PromptGroup

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
