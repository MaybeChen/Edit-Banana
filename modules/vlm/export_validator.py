"""VLM-assisted draw.io export validation and safe XML repair.

This module compares three inputs before PPTX export:

1. the original source image,
2. an optional draw.io preview rendering, and
3. a compact XML element summary.

The VLM returns a structured difference report. Only obvious, deterministic fixes
are applied automatically; all remaining issues are persisted for manual review.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Mapping, Optional, Tuple

from PIL import Image, ImageDraw

from .client import OpenAICompatibleVLMClient
from .schemas import EXPORT_VALIDATION_SCHEMA

# Conservative aliases that are safe to rewrite to draw.io's cylinder shape when
# the VLM explicitly identifies an existing element as a cylinder.
CYLINDER_AUTO_FIX_CURRENT_TYPES = {
    "rectangle",
    "rounded rectangle",
    "rounded_rectangle",
    "round rectangle",
    "roundedrect",
}

DASH_PATTERNS = {
    "dashed": "3 3",
    "dotted": "1 4",
}


class VLMExportValidator:
    """Validate merged draw.io XML immediately before PowerPoint export."""

    def __init__(self, config: Optional[Mapping[str, Any]] = None):
        self.config = dict(config or {})
        self.max_elements = int(self.config.get("export_validator_max_elements", 120))
        configured_tokens = int(self.config.get("max_tokens", 512))
        self.max_tokens = int(self.config.get("export_validator_max_tokens", max(1024, configured_tokens)))

    def validate_before_pptx(
        self,
        drawio_xml_path: str,
        original_image_path: str,
        preview_image_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run VLM validation, apply safe XML repairs, and save review JSON."""
        output_dir = output_dir or os.path.dirname(os.path.abspath(drawio_xml_path))
        os.makedirs(output_dir, exist_ok=True)

        artifact: Dict[str, Any] = {
            "drawio_xml_path": drawio_xml_path,
            "original_image_path": original_image_path,
            "preview_image_path": preview_image_path,
            "element_summary": [],
            "vlm_response": None,
            "auto_fixes": [],
            "warnings": [],
            "skipped_reason": None,
        }

        tree = ET.parse(drawio_xml_path)
        root = tree.getroot()
        artifact["element_summary"] = self._summarize_xml(root)

        client = self._get_vlm_client()
        if client is None:
            artifact["skipped_reason"] = "vlm_not_configured"
            artifact["warnings"].append({"type": "validator_skipped", "reason": "VLM client is not configured"})
            return self._finish(output_dir, artifact, tree, drawio_xml_path, save_xml=False)

        old_max_tokens = getattr(client, "max_tokens", None)
        try:
            if old_max_tokens is not None:
                client.max_tokens = max(int(old_max_tokens), self.max_tokens)
            image_url = self._comparison_image_as_data_url(original_image_path, preview_image_path)
            response = client.classify(image_url, self._build_prompt(artifact["element_summary"]), EXPORT_VALIDATION_SCHEMA)
            artifact["vlm_response"] = response
            fixes, warnings = self._apply_safe_fixes(root, response)
            artifact["auto_fixes"] = fixes
            artifact["warnings"].extend(warnings)
            return self._finish(output_dir, artifact, tree, drawio_xml_path, save_xml=bool(fixes))
        except Exception as exc:
            artifact["skipped_reason"] = f"vlm_failed: {exc}"
            artifact["warnings"].append({"type": "validator_failed", "reason": str(exc)})
            return self._finish(output_dir, artifact, tree, drawio_xml_path, save_xml=False)
        finally:
            if old_max_tokens is not None:
                client.max_tokens = old_max_tokens

    def _get_vlm_client(self) -> Optional[OpenAICompatibleVLMClient]:
        client = OpenAICompatibleVLMClient(self.config)
        return client if client.available else None

    def _summarize_xml(self, root: ET.Element) -> List[Dict[str, Any]]:
        """Build a compact, VLM-friendly summary of visible XML elements."""
        items: List[Dict[str, Any]] = []
        for cell in root.iter("mxCell"):
            cell_id = cell.get("id")
            if not cell_id or cell_id in {"0", "1"}:
                continue
            geometry = cell.find("mxGeometry")
            if geometry is None:
                continue
            style = self._parse_style(cell.get("style", ""))
            item: Dict[str, Any] = {
                "id": cell_id,
                "kind": "edge" if cell.get("edge") == "1" else "vertex",
                "value": re.sub(r"<[^>]+>", "", cell.get("value", ""))[:80],
                "style": {k: style[k] for k in sorted(style) if k in {"shape", "rounded", "dashed", "dashPattern", "startArrow", "endArrow", "fillColor", "strokeColor"}},
                "geometry": {k: geometry.get(k) for k in ("x", "y", "width", "height", "relative") if geometry.get(k) is not None},
            }
            points = [{"as": p.get("as"), "x": p.get("x"), "y": p.get("y")} for p in geometry.iter("mxPoint")]
            if points:
                item["points"] = points
            items.append(item)
            if len(items) >= self.max_elements:
                break
        return items

    def _build_prompt(self, summary: List[Dict[str, Any]]) -> str:
        schema_hint = {
            "missing_elements": [{"expected_type": "string", "location": "string", "confidence": 0.0, "reason": "string"}],
            "wrong_types": [{"element_id": "string", "current_type": "string", "expected_type": "string", "confidence": 0.0, "reason": "string"}],
            "wrong_line_styles": [{"element_id": "string", "expected_style": "solid|dashed|dotted", "needs_arrow": False, "confidence": 0.0, "reason": "string"}],
            "layout_mismatch": [{"element_id": "string", "confidence": 0.0, "reason": "string"}],
        }
        return (
            "Compare the left/original diagram with the right draw.io preview and the XML summary. "
            "Report only visible differences that matter for PPTX export. Return STRICT JSON matching this shape: "
            f"{json.dumps(schema_hint, ensure_ascii=False)}. Only reference existing XML element IDs when possible. "
            "Do not suggest uncertain auto-repairs; put uncertain observations in layout_mismatch. "
            "Element summary JSON: " + json.dumps(summary, ensure_ascii=False)
        )

    def _comparison_image_as_data_url(self, original_path: str, preview_path: Optional[str]) -> str:
        with Image.open(original_path) as original_img:
            original = original_img.convert("RGB")
            if preview_path and os.path.exists(preview_path):
                with Image.open(preview_path) as preview_img:
                    preview = preview_img.convert("RGB").resize(original.size)
                combined = Image.new("RGB", (original.width * 2, original.height + 28), "white")
                combined.paste(original, (0, 28))
                combined.paste(preview, (original.width, 28))
                draw = ImageDraw.Draw(combined)
                draw.text((8, 8), "original", fill=(0, 0, 0))
                draw.text((original.width + 8, 8), "draw.io preview", fill=(0, 0, 0))
            else:
                combined = original.copy()
            buf = io.BytesIO()
            combined.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _apply_safe_fixes(self, root: ET.Element, response: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Apply conservative, rule-based repairs and return fixes/warnings."""
        cells = {cell.get("id"): cell for cell in root.iter("mxCell") if cell.get("id")}
        fixes: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []

        for issue in self._as_list(response.get("wrong_types")):
            element_id = str(issue.get("element_id") or issue.get("id") or "")
            expected = str(issue.get("expected_type") or "").strip().lower()
            current = str(issue.get("current_type") or "").strip().lower()
            cell = cells.get(element_id)
            if cell is not None and cell.get("edge") != "1" and expected == "cylinder" and current in CYLINDER_AUTO_FIX_CURRENT_TYPES:
                style = self._parse_style(cell.get("style", ""))
                style.pop("rounded", None)
                style["shape"] = "cylinder"
                cell.set("style", self._format_style(style))
                fixes.append({"element_id": element_id, "action": "set_shape", "shape": "cylinder", "reason": issue.get("reason")})
            else:
                warnings.append({"type": "wrong_type", **issue})

        for issue in self._as_list(response.get("wrong_line_styles")):
            element_id = str(issue.get("element_id") or issue.get("id") or "")
            expected = str(issue.get("expected_style") or issue.get("line_style") or "").strip().lower()
            cell = cells.get(element_id)
            if cell is None or cell.get("edge") != "1":
                warnings.append({"type": "wrong_line_style", **issue})
                continue
            style = self._parse_style(cell.get("style", ""))
            changed = False
            if expected == "solid":
                removed_dashed = style.pop("dashed", None) is not None
                removed_pattern = style.pop("dashPattern", None) is not None
                changed = removed_dashed or removed_pattern
            elif expected in DASH_PATTERNS:
                if style.get("dashed") != "1" or style.get("dashPattern") != DASH_PATTERNS[expected]:
                    style["dashed"] = "1"
                    style["dashPattern"] = DASH_PATTERNS[expected]
                    changed = True
            if bool(issue.get("needs_arrow")) and style.get("endArrow", "none") in {"", "none"}:
                style["endArrow"] = "classic"
                changed = True
            if changed:
                cell.set("style", self._format_style(style))
                fixes.append({"element_id": element_id, "action": "update_line_style", "expected_style": expected, "needs_arrow": bool(issue.get("needs_arrow")), "reason": issue.get("reason")})
            else:
                warnings.append({"type": "wrong_line_style", **issue})

        for key in ("missing_elements", "layout_mismatch"):
            for issue in self._as_list(response.get(key)):
                warnings.append({"type": key, **issue})
        return fixes, warnings

    def _finish(self, output_dir: str, artifact: Dict[str, Any], tree: ET.ElementTree, xml_path: str, save_xml: bool) -> Dict[str, Any]:
        if save_xml:
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        review_path = os.path.join(output_dir, "vlm_export_review.json")
        with open(review_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False, indent=2)
        # Backward-compatible key/name used by existing pipeline logs.
        warnings_path = os.path.join(output_dir, "vlm_export_warnings.json")
        if warnings_path != review_path:
            with open(warnings_path, "w", encoding="utf-8") as f:
                json.dump(artifact, f, ensure_ascii=False, indent=2)
        artifact["review_path"] = review_path
        artifact["warnings_path"] = warnings_path
        return artifact

    def _parse_style(self, style: str) -> Dict[str, str]:
        parsed: Dict[str, str] = {}
        for part in style.split(";"):
            if not part:
                continue
            if "=" in part:
                key, value = part.split("=", 1)
                parsed[key] = value
            else:
                parsed[part] = ""
        return parsed

    def _format_style(self, style: Dict[str, str]) -> str:
        return ";".join(f"{key}={value}" if value != "" else key for key, value in style.items()) + ";"

    def _as_list(self, value: Any) -> List[Dict[str, Any]]:
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
