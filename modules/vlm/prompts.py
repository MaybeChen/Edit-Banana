"""Shared VLM prompt helpers."""

JSON_ONLY_SUFFIX = "Return STRICT JSON only, no markdown or explanatory text."


def require_json(prompt: str) -> str:
    """Append the project-wide JSON-only instruction when it is absent."""
    text = (prompt or "").strip()
    if "json" not in text.lower():
        return f"{text}\n{JSON_ONLY_SUFFIX}".strip()
    return text
