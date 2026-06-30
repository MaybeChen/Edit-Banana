"""Unified OpenAI-compatible VLM client.

All business modules should call this client instead of making HTTP requests
inline. The client supports configured API and local modes, JSON-only VLM
responses, parsing, schema validation, retry, timeout, and safe degradation.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any, Dict, Mapping, Optional

import requests
from PIL import Image

from .prompts import require_json
from .schemas import JsonSchema, validate_json_schema


class VLMClientError(RuntimeError):
    """Raised when a VLM call fails after retries or fallback."""


class OpenAICompatibleVLMClient:
    """OpenAI-compatible chat-completions client for vision-language models."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        raw = dict(config or {})
        self.config = dict(raw.get("multimodal") or raw)
        self.mode = str(self.config.get("mode", "api") or "api").lower()
        if self.mode not in {"api", "local"}:
            self.mode = "api"
        self.base_url = (self.config.get("local_base_url") if self.mode == "local" else self.config.get("base_url")) or ""
        self.api_key = (self.config.get("local_api_key") if self.mode == "local" else self.config.get("api_key")) or ""
        self.model = (self.config.get("local_model") if self.mode == "local" else self.config.get("model")) or ""
        self.timeout = float(self.config.get("timeout", 60))
        self.max_tokens = int(self.config.get("max_tokens", 512))
        self.retries = int(self.config.get("retries", 2))
        self.retry_backoff = float(self.config.get("retry_backoff", 0.5))
        self.proxy = self.config.get("proxy") or None
        self.ca_cert_path = self.config.get("ca_cert_path", True)

    @property
    def available(self) -> bool:
        values = (self.base_url, self.model, self.api_key)
        return all(values) and not any("YOUR_" in str(value) for value in values)

    def analyze_image(self, image_path: str, prompt: str, schema: Optional[JsonSchema] = None) -> Dict[str, Any]:
        """Analyze an image and return a parsed, schema-validated JSON object."""
        if not self.available:
            raise VLMClientError("VLM client is not configured")
        image_data_url = self._image_as_data_url(image_path)
        payload = self._build_payload(image_data_url, prompt)
        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                content = self._post_chat_completion(payload)
                data = parse_json_object(content)
                return validate_json_schema(data, schema)
            except Exception as exc:  # retry parsing/schema/network/server failures
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(self.retry_backoff * (2 ** attempt))
        raise VLMClientError(f"VLM analysis failed: {last_error}") from last_error

    def classify(self, image_data_url: str, prompt: str, schema: Optional[JsonSchema] = None) -> Dict[str, Any]:
        """Backward-compatible helper for existing callers that pass data URLs."""
        if image_data_url.startswith("data:image/"):
            return self._analyze_data_url(image_data_url, prompt, schema)
        return self.analyze_image(image_data_url, prompt, schema)

    def _analyze_data_url(self, image_data_url: str, prompt: str, schema: Optional[JsonSchema]) -> Dict[str, Any]:
        payload = self._build_payload(image_data_url, prompt)
        last_error: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                return validate_json_schema(parse_json_object(self._post_chat_completion(payload)), schema)
            except Exception as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(self.retry_backoff * (2 ** attempt))
        raise VLMClientError(f"VLM analysis failed: {last_error}") from last_error

    def _build_payload(self, image_data_url: str, prompt: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": require_json(prompt)}, {"type": "image_url", "image_url": {"url": image_data_url}}]}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }

    def _post_chat_completion(self, payload: Mapping[str, Any]) -> Any:
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None
        verify = self.ca_cert_path if self.ca_cert_path not in (False, "false", "False") else False
        response = requests.post(endpoint, headers=headers, json=payload, timeout=self.timeout, proxies=proxies, verify=verify)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _image_as_data_url(self, image_path: str) -> str:
        if image_path.startswith("data:image/"):
            return image_path
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            buf = io.BytesIO()
            image.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def parse_json_object(content: Any) -> Dict[str, Any]:
    """Parse a JSON object from VLM content, including fenced/text-wrapped JSON."""
    if isinstance(content, dict):
        return content
    text = str(content).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("VLM response must be a JSON object")
    return data
