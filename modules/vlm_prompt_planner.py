"""VLM prompt planner for SAM3 prompt groups.

Given a full input image, ask an OpenAI-compatible VLM to propose concise SAM3
text prompts for four groups: image, shape, arrow, and background.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional

import requests
from PIL import Image

PROMPT_GROUP_KEYS = ("image", "shape", "arrow", "background")


class VLMPromptPlanner:
    """Plan image-specific SAM3 prompts with an OpenAI-compatible VLM."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config = dict(config or {})
        mode = self.config.get("mode", "api")
        self.base_url = (self.config.get("local_base_url") if mode == "local" else self.config.get("base_url")) or ""
        self.api_key = (self.config.get("local_api_key") if mode == "local" else self.config.get("api_key")) or ""
        self.model = (self.config.get("local_model") if mode == "local" else self.config.get("model")) or ""
        self.timeout = float(self.config.get("timeout", 60))
        self.max_tokens = int(self.config.get("max_tokens", 1024))
        self.proxy = self.config.get("proxy") or None
        self.ca_cert_path = self.config.get("ca_cert_path", True)

    @property
    def available(self) -> bool:
        return bool(
            self.base_url
            and self.model
            and self.api_key
            and "YOUR_" not in self.base_url
            and "YOUR_" not in self.model
            and "YOUR_" not in self.api_key
        )

    def plan(self, image_path: str, max_per_group: int = 8) -> Dict[str, List[str]]:
        """Return dynamic prompts grouped as image/shape/arrow/background."""
        if not self.available:
            raise RuntimeError("VLM prompt planner is not configured")
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        data_url = self._image_as_data_url(image_path)
        payload = self._build_payload(data_url, max_per_group=max_per_group)
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
        verify = self.ca_cert_path if self.ca_cert_path not in (False, "false", "False") else False
        response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout, proxies=proxies, verify=verify)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return self._normalize_groups(_parse_json_object(content), max_per_group=max_per_group)

    def _build_payload(self, image_data_url: str, max_per_group: int) -> Dict[str, Any]:
        prompt = (
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
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

    def _image_as_data_url(self, image_path: str) -> str:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            buf = io.BytesIO()
            image.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

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


def _parse_json_object(content: Any) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    text = str(content).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))
