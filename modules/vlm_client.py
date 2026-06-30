"""VLM service client utilities.

The project can use a custom multimodal service configured under
``multimodal`` in config.yaml.  The service is expected to expose an
OpenAI-compatible chat-completions style endpoint, or ``base_url`` may point
directly at the service endpoint.
"""

from __future__ import annotations

import base64
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

        response = requests.post(
            self._request_url(),
            json=payload,
            headers=self._headers(),
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
