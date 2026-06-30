"""VLM-assisted validation and repair before PPTX export.

The validator compares the source image, an optional draw.io rendered preview,
and a compact summary of the merged XML. A configured VLM returns structured
mismatches; deterministic rules then apply safe XML fixes before the PPTX
exporter consumes the file. Issues that are not safe to fix are written to
``vlm_export_warnings.json`` for manual review.
"""

import base64
import io
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from .vlm.client import OpenAICompatibleVLMClient
from .vlm.schemas import EXPORT_VALIDATION_SCHEMA


AUTO_FIX_TYPES = {
    "rounded rectangle": "cylinder",
    "rounded_rectangle": "cylinder",
    "round rectangle": "cylinder",
    "rectangle": "cylinder",
}
EDGE_TYPES = {"connector", "arrow", "line", "edge"}


class VLMExportValidator:
    """Validate merged draw.io XML immediately before PowerPoint export."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_elements = int(self.config.get("export_validator_max_elements", 120))
        self.max_tokens = int(self.config.get("export_validator_max_tokens", max(1024, int(self.config.get("max_tokens", 512)))))

    def validate_before_pptx(
        self,
        drawio_xml_path: str,
        original_image_path: str,
        preview_image_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run VLM validation, apply safe XML repairs, and persist warnings."""
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
        summary = self._summarize_xml(root)
        artifact["element_summary"] = summary

        client = self._get_vlm_client()
        if client is None:
            artifact["skipped_reason"] = "vlm_not_configured"
            artifact["warnings"].append({"type": "validator_skipped", "reason": "VLM client is not configured"})
            return self._finish(output_dir, artifact, tree, drawio_xml_path, save_xml=False)

        try:
            image_url = self._comparison_image_as_data_url(original_image_path, preview_image_path)
            old_max_tokens = getattr(client, "max_tokens", None)
            if old_max_tokens is not None:
                client.max_tokens = max(int(old_max_tokens), self.max_tokens)
            response = client.classify(image_url, self._build_prompt(summary), EXPORT_VALIDATION_SCHEMA)
            if old_max_tokens is not None:
                client.max_tokens = old_max_tokens
            artifact["vlm_response"] = response
            fixes, warnings = self._apply_safe_fixes(root, response)
            artifact["auto_fixes"] = fixes
            artifact["warnings"].extend(warnings)
            return self._finish(output_dir, artifact, tree, drawio_xml_path, save_xml=bool(fixes))
        except Exception as exc:
            artifact["skipped_reason"] = f"vlm_failed: {exc}"
            artifact["warnings"].append({"type": "validator_failed", "reason": str(exc)})
            return self._finish(output_dir, artifact, tree, drawio_xml_path, save_xml=False)

    def _get_vlm_client(self):
        client = OpenAICompatibleVLMClient(self.config)
        return client if client.available else None

    def _summarize_xml(self, root: ET.Element) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for cell in root.iter("mxCell"):
            cell_id = cell.get("id")
            if cell_id in {"0", "1"}:
                continue
            style = self._parse_style(cell.get("style", ""))
            geometry = cell.find("mxGeometry")
            if geometry is None:
                continue
            item = {
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
        return (
            "Compare the original diagram with the draw.io/PPTX export preview and current XML summary. "
            "Return STRICT JSON only with schema: {\"missing_elements\":[...],\"wrong_types\":[{\"element_id\":str,\"current_type\":str,\"expected_type\":str,\"confidence\":number,\"reason\":str}],"
            "\"wrong_line_styles\":[{\"element_id\":str,\"expected_style\":\"solid|dashed|dotted\",\"needs_arrow\":bool,\"confidence\":number,\"reason\":str}],"
            "\"layout_mismatch\":[...]}. Only reference existing XML element IDs when possible. "
            "Element summary JSON: " + json.dumps(summary, ensure_ascii=False)
        )

    def _comparison_image_as_data_url(self, original_path: str, preview_path: Optional[str]) -> str:
        with Image.open(original_path) as original:
            original = original.convert("RGB")
            if preview_path and os.path.exists(preview_path):
                with Image.open(preview_path) as preview:
                    preview = preview.convert("RGB").resize(original.size)
                combined = Image.new("RGB", (original.width * 2, original.height + 28), "white")
                combined.paste(original, (0, 28))
                combined.paste(preview, (original.width, 28))
                draw = ImageDraw.Draw(combined)
                draw.text((8, 8), "original", fill=(0, 0, 0))
                draw.text((original.width + 8, 8), "draw.io preview", fill=(0, 0, 0))
            else:
                combined = original
            buf = io.BytesIO()
            combined.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _apply_safe_fixes(self, root: ET.Element, response: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        cells = {cell.get("id"): cell for cell in root.iter("mxCell") if cell.get("id")}
        fixes: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []

        for issue in self._as_list(response.get("wrong_types")):
            element_id = str(issue.get("element_id") or issue.get("id") or "")
            expected = str(issue.get("expected_type") or "").strip().lower()
            current = str(issue.get("current_type") or "").strip().lower()
            cell = cells.get(element_id)
            if cell is not None and expected == "cylinder" and current in AUTO_FIX_TYPES:
                style = self._parse_style(cell.get("style", ""))
                style.pop("rounded", None)
                style["shape"] = "cylinder"
                cell.set("style", self._format_style(style))
                fix = {"element_id": element_id, "action": "set_shape", "shape": "cylinder", "reason": issue.get("reason")}
                fixes.append(fix)
            else:
                warnings.append({"type": "wrong_type", **issue})

        for issue in self._as_list(response.get("wrong_line_styles")):
            element_id = str(issue.get("element_id") or issue.get("id") or "")
            expected = str(issue.get("expected_style") or issue.get("line_style") or "").strip().lower()
            cell = cells.get(element_id)
            if cell is not None and cell.get("edge") == "1":
                style = self._parse_style(cell.get("style", ""))
                changed = False
                if expected in {"dashed", "dotted"}:
                    style["dashed"] = "1"
                    style.setdefault("dashPattern", "1 4" if expected == "dotted" else "3 3")
                    changed = True
                if bool(issue.get("needs_arrow")) and style.get("endArrow", "none") == "none":
                    style["endArrow"] = "classic"
                    changed = True
                if changed:
                    cell.set("style", self._format_style(style))
                    fixes.append({"element_id": element_id, "action": "update_line_style", "expected_style": expected, "needs_arrow": bool(issue.get("needs_arrow"))})
                else:
                    warnings.append({"type": "wrong_line_style", **issue})
            else:
                warnings.append({"type": "wrong_line_style", **issue})

        for key in ("missing_elements", "layout_mismatch"):
            for issue in self._as_list(response.get(key)):
                warnings.append({"type": key[:-1] if key.endswith("s") else key, **issue})
        return fixes, warnings

    def _finish(self, output_dir: str, artifact: Dict[str, Any], tree: ET.ElementTree, xml_path: str, save_xml: bool) -> Dict[str, Any]:
        if save_xml:
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        warnings_path = os.path.join(output_dir, "vlm_export_warnings.json")
        with open(warnings_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False, indent=2)
        artifact["warnings_path"] = warnings_path
        return artifact

    def _parse_style(self, style: str) -> Dict[str, str]:
        parsed: Dict[str, str] = {}
        for part in style.split(";"):
            if not part:
                continue
            if "=" in part:
                k, v = part.split("=", 1)
                parsed[k] = v
            else:
                parsed[part] = ""
        return parsed

    def _format_style(self, style: Dict[str, str]) -> str:
        return ";".join(f"{k}={v}" if v != "" else k for k, v in style.items()) + ";"

    def _as_list(self, value: Any) -> List[Dict[str, Any]]:
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
