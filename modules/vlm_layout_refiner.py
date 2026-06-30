"""VLM-based layout relationship refinement.

This processor asks a configured VLM to infer diagram edge relationships from the
full image plus the current element inventory. It does not create new elements;
it only annotates existing arrow/connector/line elements with source/target and
style hints used later by XML generation.
"""

import base64
import io
import json
import os
import re
from typing import Any, Dict, List, Optional

from PIL import Image

from .base import BaseProcessor, ProcessingContext
from .data_types import ElementInfo, ProcessingResult
from .vlm_element_refiner import OpenAICompatibleVLMClient

EDGE_TYPES = {"arrow", "connector", "line"}
LINE_STYLES = {"solid", "dashed", "dotted"}
ARROW_STYLES = {"none", "classic", "open", "block"}


class VLMLayoutRefiner(BaseProcessor):
    """Infer edge source/target relationships with a VLM before XML generation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(None)
        self.vlm_config = config or {}
        self.max_elements = int(self.vlm_config.get("layout_max_elements", self.vlm_config.get("max_elements", 80)))
        self.max_tokens = int(self.vlm_config.get("layout_max_tokens", max(1024, int(self.vlm_config.get("max_tokens", 512)))))

    def process(self, context: ProcessingContext) -> ProcessingResult:
        self._log("Refining edge layout relationships with VLM")
        artifact: Dict[str, Any] = {
            "image_path": context.image_path,
            "elements": self._collect_elements(context),
            "vlm_response": None,
            "applied_relations": [],
            "skipped_reason": None,
        }

        if not context.image_path or not os.path.exists(context.image_path):
            artifact["skipped_reason"] = "invalid_image_path"
            path = self._save_artifact(context, artifact)
            return ProcessingResult(success=False, elements=context.elements, error_message="Invalid image path", metadata={"vlm_layout_json": path, "skipped_reason": artifact["skipped_reason"]})

        client = self._get_vlm_client(context)
        if client is None:
            artifact["skipped_reason"] = "vlm_not_configured"
            path = self._save_artifact(context, artifact)
            self._log("Skipped: VLM client is not configured")
            return ProcessingResult(success=True, elements=context.elements, canvas_width=context.canvas_width, canvas_height=context.canvas_height, metadata={"vlm_layout_json": path, "processed_count": 0, "updated_count": 0, "skipped_reason": artifact["skipped_reason"]})

        try:
            data_url = self._image_as_data_url(context.image_path)
            prompt = self._build_prompt(artifact["elements"])
            # Keep the existing client shape but allow a larger response budget for relationship JSON.
            old_max_tokens = getattr(client, "max_tokens", None)
            if old_max_tokens is not None:
                client.max_tokens = max(int(old_max_tokens), self.max_tokens)
            response = client.classify(data_url, prompt)
            if old_max_tokens is not None:
                client.max_tokens = old_max_tokens
            artifact["vlm_response"] = response
            applied = self._apply_response(context.elements, response)
            artifact["applied_relations"] = applied
            path = self._save_artifact(context, artifact)
            self._log(f"Done: updated={len(applied)}")
            return ProcessingResult(success=True, elements=context.elements, canvas_width=context.canvas_width, canvas_height=context.canvas_height, metadata={"vlm_layout_json": path, "processed_count": len(self._edge_elements(context.elements)), "updated_count": len(applied)})
        except Exception as exc:
            artifact["skipped_reason"] = f"vlm_failed: {exc}"
            path = self._save_artifact(context, artifact)
            self._log(f"VLM layout refine failed: {exc}")
            return ProcessingResult(success=True, elements=context.elements, canvas_width=context.canvas_width, canvas_height=context.canvas_height, metadata={"vlm_layout_json": path, "processed_count": 0, "updated_count": 0, "skipped_reason": artifact["skipped_reason"]})

    def _get_vlm_client(self, context: ProcessingContext):
        shared = context.shared_models.get("vlm_client")
        if shared is not None:
            return shared
        client = OpenAICompatibleVLMClient(self.vlm_config)
        return client if client.available else None

    def _collect_elements(self, context: ProcessingContext) -> List[Dict[str, Any]]:
        ocr_blocks = self._load_ocr_blocks(context)
        elements = []
        for elem in context.elements[: self.max_elements]:
            elements.append({
                "id": elem.id,
                "element_type": elem.element_type,
                "bbox": elem.bbox.to_list(),
                "source_prompt": elem.source_prompt,
                "ocr_label": self._match_ocr_label(elem, ocr_blocks),
            })
        return elements

    def _load_ocr_blocks(self, context: ProcessingContext) -> List[Dict[str, Any]]:
        path = context.intermediate_results.get("ocr_result_json")
        if not path or not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("processed_text_blocks") or payload.get("raw_ocr_blocks") or []
        except Exception:
            return []

    def _match_ocr_label(self, elem: ElementInfo, blocks: List[Dict[str, Any]]) -> Optional[str]:
        best_text = None
        best_score = 0.0
        ex1, ey1, ex2, ey2 = elem.bbox.to_list()
        for block in blocks:
            text = str(block.get("text", "")).strip()
            poly = block.get("polygon") or []
            if not text or not poly:
                continue
            xs = [p[0] for p in poly if len(p) >= 2]
            ys = [p[1] for p in poly if len(p) >= 2]
            if not xs or not ys:
                continue
            bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
            ix1, iy1, ix2, iy2 = max(ex1, bx1), max(ey1, by1), min(ex2, bx2), min(ey2, by2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            block_area = max(1, (bx2 - bx1) * (by2 - by1))
            score = ((ix2 - ix1) * (iy2 - iy1)) / block_area
            if score > best_score:
                best_score = score
                best_text = text
        return best_text if best_score >= 0.25 else None

    def _build_prompt(self, elements: List[Dict[str, Any]]) -> str:
        return (
            "You are refining a diagram graph before draw.io XML generation. Return STRICT JSON only. "
            "Use the full image and the element inventory to infer which shapes each existing arrow/connector/line connects. "
            "Do not invent element IDs. Schema: {\"relationships\":[{\"edge_id\":int,\"source_id\":int|null,\"target_id\":int|null,\"line_style\":\"solid|dashed|dotted|null\",\"arrow_style\":\"none|classic|open|block|null\",\"confidence\":number,\"reason\":string}]}. "
            "For arrows, source_id is the tail element and target_id is the arrowhead element. For plain lines use visual left/top as source if direction is unclear and arrow_style none. "
            f"Element inventory JSON: {json.dumps(elements, ensure_ascii=False)}"
        )

    def _image_as_data_url(self, image_path: str) -> str:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            buf = io.BytesIO()
            image.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    def _apply_response(self, elements: List[ElementInfo], response: Dict[str, Any]) -> List[Dict[str, Any]]:
        by_id = {int(e.id): e for e in elements}
        relations = response.get("relationships") if isinstance(response, dict) else None
        if relations is None and isinstance(response, dict):
            relations = response.get("edges") or response.get("connections")
        if not isinstance(relations, list):
            return []
        applied = []
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            edge_id = self._to_int(rel.get("edge_id") or rel.get("id"))
            edge = by_id.get(edge_id) if edge_id is not None else None
            if edge is None or edge.element_type.lower() not in EDGE_TYPES:
                continue
            source_id = self._to_int(rel.get("source_id"))
            target_id = self._to_int(rel.get("target_id"))
            if source_id == edge_id or source_id not in by_id:
                source_id = None
            if target_id == edge_id or target_id not in by_id:
                target_id = None
            line_style = self._clean_choice(rel.get("line_style"), LINE_STYLES)
            arrow_style = self._clean_choice(rel.get("arrow_style"), ARROW_STYLES)
            if source_id is None and target_id is None and line_style is None and arrow_style is None:
                continue
            edge.source_id = source_id
            edge.target_id = target_id
            if line_style:
                edge.line_style = line_style
            if arrow_style:
                edge.arrow_style = arrow_style
            edge.processing_notes.append(f"VLM layout relation: source={source_id}, target={target_id}, line_style={line_style}, arrow_style={arrow_style}")
            item = {"edge_id": edge.id, "source_id": source_id, "target_id": target_id, "line_style": line_style, "arrow_style": arrow_style}
            applied.append(item)
        return applied

    def _edge_elements(self, elements: List[ElementInfo]) -> List[ElementInfo]:
        return [e for e in elements if e.element_type.lower() in EDGE_TYPES]

    def _save_artifact(self, context: ProcessingContext, artifact: Dict[str, Any]) -> str:
        os.makedirs(context.output_dir, exist_ok=True)
        path = os.path.join(context.output_dir, "vlm_layout.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False, indent=2)
        context.intermediate_results["vlm_layout_json"] = path
        return path

    def _to_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            match = re.search(r"\d+", str(value))
            return int(match.group(0)) if match else None

    def _clean_choice(self, value: Any, allowed: set) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {"dash": "dashed", "dot": "dotted", "arrow": "classic", "classic_arrow": "classic", "no_arrow": "none"}
        cleaned = aliases.get(cleaned, cleaned)
        return cleaned if cleaned in allowed else None
