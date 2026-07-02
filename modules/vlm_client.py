"""VLM service client utilities.

The project can use a custom multimodal service configured under
``multimodal`` in config.yaml.  The service is expected to expose an
OpenAI-compatible chat-completions style endpoint, or ``base_url`` may point
directly at the service endpoint.
"""

from __future__ import annotations

import base64
import hashlib
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
        self.endpoint_path = self.config.get("endpoint_path", "chat/completions")
        self.base_url_is_endpoint = bool(self.config.get("base_url_is_endpoint", False))
        self.image_url_format = str(self.config.get("image_url_format", "raw_base64") or "raw_base64")
        self.image_content_order = str(self.config.get("image_content_order", "text_first") or "text_first")
        self.verify_ssl = self.config.get("verify_ssl", True)
        self.ca_cert_path = self.config.get("ca_cert_path")
        self.proxy = self.config.get("proxy") or ""
        self.log_response = bool(self.config.get("log_response", True))
        self.response_log_chars = int(self.config.get("response_log_chars", 0) or 0)
        self.request_text_log_chars = int(self.config.get("request_text_log_chars", 1200) or 1200)

    def _request_url(self) -> str:
        """Resolve the request URL from configured base_url.

        Set multimodal.base_url_is_endpoint=true to use base_url exactly as
        configured (matching a known-good Postman URL, including any trailing
        slash). Otherwise endpoint_path is appended unless base_url already
        points to a chat-completions endpoint.
        """
        raw_base_url = self.base_url or ""
        base_url = raw_base_url.rstrip("/")
        if not base_url or base_url in _PLACEHOLDER_VALUES:
            raise ValueError("multimodal.base_url is not configured")
        if self.base_url_is_endpoint:
            return raw_base_url
        if base_url.endswith("/chat/completions") or base_url.endswith("/v1/chat/completions"):
            return raw_base_url
        endpoint_path = str(self.endpoint_path or "").strip("/")
        if not endpoint_path:
            return raw_base_url
        return f"{base_url}/{endpoint_path}"

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

    @classmethod
    def _summarize_content_part(cls, part: Any, text_log_chars: int = 1200) -> Any:
        """Summarize multimodal content without dumping base64 image data."""
        if not isinstance(part, dict):
            return part
        if part.get("type") == "image_url":
            image_url = part.get("image_url")
            if isinstance(image_url, dict):
                url = str(image_url.get("url", ""))
                if url.startswith("data:"):
                    prefix = url.split(",", 1)[0]
                    return {"type": "image_url", "image_url": {"url": f"{prefix},<base64 omitted>"}}
                if len(url) > 80:
                    return {"type": "image_url", "image_url": {"url": "<image omitted>"}}
            elif isinstance(image_url, str):
                if image_url.startswith("data:"):
                    prefix = image_url.split(",", 1)[0]
                    return {"type": "image_url", "image_url": f"{prefix},<base64 omitted>"}
                if len(image_url) > 80:
                    return {"type": "image_url", "image_url": "<base64 omitted>"}
        if part.get("type") == "text":
            text = str(part.get("text") or "")
            limit = max(0, int(text_log_chars or 0))
            preview = text if limit == 0 or len(text) <= limit else text[:limit]
            return {
                "type": "text",
                "text_preview": preview,
                "text_chars": len(text),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
                "truncated": bool(limit and len(text) > limit),
            }
        return part

    @classmethod
    def _summarize_messages_for_log(cls, messages: List[Dict[str, Any]], text_log_chars: int = 1200) -> List[Dict[str, Any]]:
        """Summarize messages for request logs."""
        summarized = []
        for message in messages:
            msg = dict(message)
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = [cls._summarize_content_part(part, text_log_chars) for part in content]
            summarized.append(msg)
        return summarized

    def _summarize_response_for_log(self, result: Any, raw_text: str) -> Dict[str, Any]:
        """Log response metadata and compact PPTX-related fields only."""
        summary: Dict[str, Any] = {"body_chars": len(raw_text or "")}
        content = self._extract_response_content_for_log(result)
        if content:
            parsed = self._try_parse_json_content(content)
            if isinstance(parsed, dict):
                summary["pptx"] = self._pptx_fields_for_log(parsed)
            else:
                summary["parse_error"] = True
                summary["content_chars"] = len(content)
        return summary

    @staticmethod
    def _extract_response_content_for_log(result: Any) -> str:
        if not isinstance(result, dict):
            return ""
        if isinstance(result.get("content"), str):
            return result["content"]
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            message = first.get("message") if isinstance(first, dict) else {}
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "\n".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and isinstance(part.get("text"), str)
                    )
            text = first.get("text") if isinstance(first, dict) else None
            if isinstance(text, str):
                return text
        return ""

    @staticmethod
    def _try_parse_json_content(content: str) -> Any:
        text = str(content or "").strip()
        if "\\n" in text and "\n" not in text:
            text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
        if text.startswith("```"):
            import re

            text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
        try:
            return json.loads(text)
        except Exception:
            repaired = VLMClient._repair_dirty_json_text(text)
            if repaired == text:
                return None
            try:
                return json.loads(repaired)
            except Exception:
                return None

    @staticmethod
    def _repair_dirty_json_text(text: str) -> str:
        import re

        repaired = str(text or "")
        repaired = re.sub(r"\\n+\s*(?=\")", "\n", repaired)

        def fix_key(match: Any) -> str:
            key = re.sub(r"[\r\n]\s*", "", match.group(1))
            return f'"{key}":'

        previous = None
        key_pattern = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"\s*:', flags=re.DOTALL)
        while previous != repaired:
            previous = repaired
            repaired = key_pattern.sub(fix_key, repaired)
        return repaired

    @staticmethod
    def _pptx_fields_for_log(data: Dict[str, Any], max_items: int = 30) -> Dict[str, Any]:
        def compact_item(item: Any) -> Any:
            if not isinstance(item, dict):
                return item
            return {
                key: item[key]
                for key in (
                    "id",
                    "type",
                    "subtype",
                    "bbox",
                    "content",
                    "text",
                    "font_size",
                    "font_size_estimate",
                    "font_color",
                    "fill_color",
                    "stroke_color",
                    "stroke_width",
                    "line_style",
                    "confidence",
                )
                if key in item
            }

        summary: Dict[str, Any] = {}
        if isinstance(data.get("background"), dict):
            summary["background"] = compact_item(data["background"])
        elements = data.get("elements")
        if isinstance(elements, dict):
            elements = [dict(value, id=key) if isinstance(value, dict) and "id" not in value else value for key, value in elements.items()]
        if isinstance(elements, list):
            summary["elements"] = {
                "count": len(elements),
                "items": [compact_item(item) for item in elements[:max_items]],
                "truncated": len(elements) > max_items,
            }
        text_blocks = data.get("text_blocks")
        if isinstance(text_blocks, list):
            summary["text_blocks"] = {
                "count": len(text_blocks),
                "items": [compact_item(item) for item in text_blocks[:max_items]],
                "truncated": len(text_blocks) > max_items,
            }
        if isinstance(data.get("reconstruction_summary"), dict):
            summary["reconstruction_summary"] = data["reconstruction_summary"]
        return summary

    @staticmethod
    def _image_to_base64(image_path: str) -> str:
        """Encode a local image as a plain base64 string."""
        return base64.b64encode(Path(image_path).read_bytes()).decode("ascii")

    @staticmethod
    def _image_to_data_url(image_path: str) -> str:
        """Encode a local image as a data URL for OpenAI-style payloads."""
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _image_content_part(self, image_path: str) -> Dict[str, Any]:
        """Build the image content part expected by the configured service."""
        image_format = self.image_url_format.lower().replace("-", "_")
        if image_format in {"raw_base64", "base64", "plain_base64"}:
            return {"type": "image_url", "image_url": self._image_to_base64(image_path)}
        if image_format in {"data_url", "data_url_string"}:
            return {"type": "image_url", "image_url": self._image_to_data_url(image_path)}
        if image_format in {"openai", "openai_data_url", "object", "object_data_url"}:
            return {"type": "image_url", "image_url": {"url": self._image_to_data_url(image_path)}}
        raise ValueError(
            "multimodal.image_url_format must be one of raw_base64, data_url_string, or openai_data_url"
        )

    def _request_verify(self) -> Any:
        """Resolve TLS verification behavior for requests."""
        if self.ca_cert_path and self.ca_cert_path not in (False, "false", "False"):
            return self.ca_cert_path
        return bool(self.verify_ssl)

    def _request_proxies(self) -> Optional[Dict[str, str]]:
        """Resolve optional proxy configuration."""
        if not self.proxy:
            return None
        return {"http": str(self.proxy), "https": str(self.proxy)}

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
        log_payload["messages"] = self._summarize_messages_for_log(messages, self.request_text_log_chars)
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
            verify=self._request_verify(),
            proxies=self._request_proxies(),
        )
        if response.status_code >= 400:
            print(
                "[VLMClient] error "
                + json.dumps(
                    {
                        "status_code": response.status_code,
                        "url": response.url,
                        "body_chars": len(response.text),
                        "body_preview": response.text[:800],
                        "body_truncated": len(response.text) > 800,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        response.raise_for_status()
        result = response.json()
        if self.log_response:
            response_summary = self._summarize_response_for_log(result, response.text)
            print(
                "[VLMClient] response "
                + json.dumps(
                    {
                        "status_code": response.status_code,
                        "url": response.url,
                        **response_summary,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        return result

    def chat(self, messages: List[Dict[str, Any]], **overrides: Any) -> Dict[str, Any]:
        """Backward-compatible alias for request()."""
        return self.request(messages, **overrides)

    def analyze_image(self, image_path: str, prompt: str, **overrides: Any) -> Dict[str, Any]:
        """Send a single image plus text prompt to the VLM service."""
        text_part = {"type": "text", "text": prompt}
        image_part = self._image_content_part(image_path)
        if self.image_content_order.lower() in {"image_first", "image-url-first", "image_url_first"}:
            content = [image_part, text_part]
        else:
            content = [text_part, image_part]
        messages = [{"role": "user", "content": content}]
        return self.chat(messages, **overrides)


def create_vlm_client_from_config(config: Dict[str, Any]) -> VLMClient:
    """Create a VLMClient from the root project config."""
    return VLMClient((config or {}).get("multimodal") or {})
