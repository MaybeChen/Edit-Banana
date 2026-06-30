"""VLM service client utilities.

The project can use a custom multimodal service configured under
``multimodal`` in config.yaml.  The service is expected to expose an
OpenAI-compatible chat-completions style endpoint, or ``base_url`` may point
directly at the service endpoint.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


_PLACEHOLDER_VALUES = {
    "",
    "YOUR_API_KEY_HERE",
    "YOUR_BASE_URL_HERE",
    "your-vlm-model-name",
    "YOUR_X_HW_ID_HERE",
    "YOUR_X_HW_APPKEY_HERE",
}


class VLMClient:
    """Small HTTP client for configured VLM services."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.mode = self.config.get("mode", "api")
        if self.mode == "local":
            self.base_url = self.config.get("local_base_url") or self.config.get("base_url") or ""
            self.model = self.config.get("local_model") or self.config.get("model") or ""
            self.api_key = self.config.get("local_api_key") or self.config.get("api_key") or ""
        else:
            self.base_url = self.config.get("base_url") or ""
            self.model = self.config.get("model") or ""
            self.api_key = self.config.get("api_key") or ""

        self.x_hw_id = self.config.get("x_hw_id") or ""
        self.x_hw_appkey = self.config.get("x_hw_appkey") or ""
        self.timeout = int(self.config.get("timeout", 60) or 60)
        self.max_tokens = int(self.config.get("max_tokens", 4000) or 4000)

    def _request_url(self) -> str:
        """Resolve the request URL from configured base_url."""
        base_url = (self.base_url or "").rstrip("/")
        if not base_url or base_url in _PLACEHOLDER_VALUES:
            raise ValueError("multimodal.base_url is not configured")
        if base_url.endswith("/chat/completions") or base_url.endswith("/v1/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    @staticmethod
    def _mask_secret(value: str) -> str:
        """Mask secret values before logging."""
        value = str(value or "")
        if not value:
            return ""
        if len(value) <= 8:
            return "****"
        return f"{value[:4]}...{value[-4:]}"

    def _headers(self) -> Dict[str, str]:
        """Build headers required by the custom VLM service."""
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key not in _PLACEHOLDER_VALUES:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.x_hw_id and self.x_hw_id not in _PLACEHOLDER_VALUES:
            headers["X-HW-ID"] = str(self.x_hw_id)
        if self.x_hw_appkey and self.x_hw_appkey not in _PLACEHOLDER_VALUES:
            headers["X-HW-APPKEY"] = str(self.x_hw_appkey)
        return headers

    def _safe_headers_for_log(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Return headers with sensitive values masked for logs."""
        safe = dict(headers)
        for key in ("Authorization", "X-HW-ID", "X-HW-APPKEY"):
            if key in safe:
                safe[key] = self._mask_secret(safe[key])
        return safe

    @staticmethod
    def _summarize_content_part(part: Any) -> Any:
        """Summarize multimodal content without dumping base64 image data."""
        if not isinstance(part, dict):
            return part
        if part.get("type") == "image_url":
            image_url = part.get("image_url") or {}
            url = str(image_url.get("url", ""))
            if url.startswith("data:"):
                prefix = url.split(",", 1)[0]
                return {"type": "image_url", "image_url": {"url": f"{prefix},<base64 omitted>"}}
        if part.get("type") == "text":
            text = str(part.get("text", ""))
            if len(text) > 500:
                part = dict(part)
                part["text"] = f"{text[:500]}...<truncated {len(text) - 500} chars>"
        return part

    @classmethod
    def _summarize_messages_for_log(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Summarize messages for request logs."""
        summarized = []
        for message in messages:
            msg = dict(message)
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = [cls._summarize_content_part(part) for part in content]
            elif isinstance(content, str) and len(content) > 500:
                msg["content"] = f"{content[:500]}...<truncated {len(content) - 500} chars>"
            summarized.append(msg)
        return summarized

    @staticmethod
    def _image_to_data_url(image_path: str) -> str:
        """Encode a local image as a data URL for multimodal chat payloads."""
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def request(self, messages: List[Dict[str, Any]], **overrides: Any) -> Dict[str, Any]:
        """Send a request to the configured VLM service."""
        model = overrides.pop("model", None) or self.model
        if not model or model in _PLACEHOLDER_VALUES:
            raise ValueError("multimodal.model is not configured")

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": overrides.pop("max_tokens", self.max_tokens),
        }
        payload.update(overrides)

        url = self._request_url()
        headers = self._headers()
        log_payload = dict(payload)
        log_payload["messages"] = self._summarize_messages_for_log(messages)
        print(
            "[VLMClient] request "
            + json.dumps(
                {
                    "url": url,
                    "model": model,
                    "timeout": self.timeout,
                    "headers": self._safe_headers_for_log(headers),
                    "payload": log_payload,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def chat(self, messages: List[Dict[str, Any]], **overrides: Any) -> Dict[str, Any]:
        """Backward-compatible alias for request()."""
        return self.request(messages, **overrides)

    def analyze_image(self, image_path: str, prompt: str, **overrides: Any) -> Dict[str, Any]:
        """Send a single image plus text prompt to the VLM service."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": self._image_to_data_url(image_path)}},
                ],
            }
        ]
        return self.chat(messages, **overrides)


def create_vlm_client_from_config(config: Dict[str, Any]) -> VLMClient:
    """Create a VLMClient from the root project config."""
    return VLMClient((config or {}).get("multimodal") or {})
