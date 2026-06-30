"""VLM prompt planner for SAM3 prompt groups.

Given a full input image, ask an OpenAI-compatible VLM to propose concise SAM3
text prompts for four groups: image, shape, arrow, and background.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Mapping, Optional

from .vlm.client import OpenAICompatibleVLMClient
from .vlm.schemas import PROMPT_GROUPS_SCHEMA


PROMPT_GROUP_KEYS = ("image", "shape", "arrow", "background")


class VLMPromptPlanner:
    """Plan image-specific SAM3 prompts with an OpenAI-compatible VLM."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config = dict(config or {})
        self.client = OpenAICompatibleVLMClient(self.config)
        self.client.max_tokens = int(self.config.get("max_tokens", 1024))

    @property
    def available(self) -> bool:
        return self.client.available

    def plan(self, image_path: str, max_per_group: int = 8) -> Dict[str, List[str]]:
        """Return dynamic prompts grouped as image/shape/arrow/background."""
        if not self.available:
            raise RuntimeError("VLM prompt planner is not configured")
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        response = self.client.analyze_image(image_path, self._build_prompt(max_per_group), PROMPT_GROUPS_SCHEMA)
        return self._normalize_groups(response, max_per_group=max_per_group)

    def _build_prompt(self, max_per_group: int) -> str:
        return (
            "You are planning text prompts for SAM3 segmentation on a diagram image. "
            "Return STRICT JSON only, no markdown. Schema: "
            "{\"image\":[string],\"shape\":[string],\"arrow\":[string],\"background\":[string]}. "
            f"Each array must contain at most {max_per_group} short English noun phrases. "
            "Use prompts that describe visible visual objects, not OCR text. "
            "image: icons, logos, photos, embedded charts, symbolic pictures. "
            "shape: diagram nodes such as rectangles, rounded rectangles, circles, diamonds, cylinders. "
            "arrow: arrows, connector lines, dashed/dotted connectors. "
            "background: grouping panels, containers, swimlanes, frames, section backgrounds. "
            "Prefer concrete phrases likely to help object segmentation."
        )

    def _normalize_groups(self, data: Mapping[str, Any], max_per_group: int) -> Dict[str, List[str]]:
        normalized: Dict[str, List[str]] = {}
        for key in PROMPT_GROUP_KEYS:
            values = data.get(key, []) if isinstance(data, Mapping) else []
            if isinstance(values, str):
                values = [values]
            prompts: List[str] = []
            seen = set()
            for value in values or []:
                prompt = re.sub(r"\s+", " ", str(value)).strip().strip("-•,.;")
                if not prompt or prompt in seen:
                    continue
                seen.add(prompt)
                prompts.append(prompt)
                if len(prompts) >= max_per_group:
                    break
            normalized[key] = prompts
        return normalized
