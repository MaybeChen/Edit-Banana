"""Step 1: full-page Layout VLM recognition artifacts.

This module intentionally stops after Layout VLM.  It creates two artifacts:
- a structured JSON file with regions in original-image pixel coordinates;
- an overlay image drawn on the original image with pixel bboxes mapped from
  the exact image sent to VLM.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from PIL import Image, ImageDraw, ImageFont

from prompts.vlm_structure import build_vlm_page_regions_prompt
from .vlm_client import create_vlm_client_from_config

MAX_VLM_LONG_EDGE = 2048
COORDINATE_SYSTEM = "pixel_original"
LAYOUT_REGION_TYPES = {
    "background",
    "header",
    "footer",
    "sidebar",
    "main_content",
    "container_group",
    "card_group",
    "image_region",
    "icon_logo_region",
    "table_region",
    "chart_region",
    "diagram_region",
    "complex_visual_region",
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML config; missing config falls back to an empty dict."""
    path = Path(config_path or "config/config.yaml")
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_layout_step(
    image_path: str,
    output_dir: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Step 1 exactly once and return artifact metadata."""
    image = Path(image_path)
    if not image.exists():
        raise FileNotFoundError(f"image not found: {image_path}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    preprocess = preprocess_image_for_vlm(image, output)
    result_path = output / "layout_vlm_result.json"
    overlay_path = output / "layout_vlm_overlay.png"

    artifact: Dict[str, Any] = {
        "step": "step_1_layout_vlm",
        "status": "started",
        "coordinate_system": COORDINATE_SYSTEM,
        "prompt": "VLM_PAGE_REGIONS_PROMPT",
        **preprocess,
    }

    try:
        client = create_vlm_client_from_config(config or {})
        prompt = build_vlm_page_regions_prompt(preprocess["vlm_image_size"]["width"], preprocess["vlm_image_size"]["height"])
        response = client.analyze_image(preprocess["vlm_image_path"], prompt)
        data = parse_json_response(response)
        regions = normalize_regions(data.get("regions", []), preprocess["original_size"], preprocess["vlm_image_size"])
        artifact.update(
            {
                "status": "completed",
                "page_aspect_ratio_estimate": data.get("page_aspect_ratio_estimate"),
                "layout_pattern": data.get("layout_pattern"),
                "page_structure": data.get("page_structure"),
                "regions": regions,
                "reading_order": normalize_reading_order(data.get("reading_order"), regions),
                "raw_region_count": len(data.get("regions", [])) if isinstance(data.get("regions"), list) else 0,
            }
        )
    except Exception as exc:
        artifact.update({"status": "failed", "error": str(exc), "regions": [], "reading_order": []})

    write_json(result_path, artifact)
    draw_layout_overlay(Path(preprocess["saved_original_image_path"]), overlay_path, artifact.get("regions", []))
    artifact["json_path"] = str(result_path)
    artifact["overlay_path"] = str(overlay_path)
    write_json(result_path, artifact)
    return artifact


def preprocess_image_for_vlm(image_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Save original image and create a <=2048 long-edge VLM image."""
    suffix = image_path.suffix or ".png"
    saved_original = output_dir / f"original{suffix}"
    if image_path.resolve() != saved_original.resolve():
        shutil.copy2(image_path, saved_original)

    with Image.open(image_path) as img:
        width, height = img.size
        long_edge = max(width, height)
        if long_edge <= MAX_VLM_LONG_EDGE:
            vlm_image_path = str(saved_original)
            vlm_size = {"width": width, "height": height}
            scale = 1.0
        else:
            scale = MAX_VLM_LONG_EDGE / float(long_edge)
            resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
            thumbnail = output_dir / "layout_vlm_thumbnail.png"
            img.convert("RGB").resize(resized_size, Image.LANCZOS).save(thumbnail)
            vlm_image_path = str(thumbnail)
            vlm_size = {"width": resized_size[0], "height": resized_size[1]}

    return {
        "original_image_path": str(image_path),
        "saved_original_image_path": str(saved_original),
        "original_size": {"width": width, "height": height},
        "vlm_image_path": vlm_image_path,
        "vlm_image_size": vlm_size,
        "vlm_long_edge_limit": MAX_VLM_LONG_EDGE,
        "vlm_scale_from_original": scale,
    }


def parse_json_response(response: Any) -> Dict[str, Any]:
    """Parse common VLM JSON response shapes and fenced JSON text."""
    if isinstance(response, dict) and isinstance(response.get("regions"), list):
        return response
    text = extract_response_text(response).strip()
    if not text:
        return {}
    if "\\n" in text and "\n" not in text:
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        return json.loads(match.group(0))


def extract_response_text(response: Any) -> str:
    """Extract assistant text from direct strings or chat-completion responses."""
    if isinstance(response, str):
        return response
    if not isinstance(response, dict):
        return ""
    if isinstance(response.get("content"), str):
        return response["content"]
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else {}
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]
        if isinstance(first.get("text"), str):
            return first["text"]
    return ""


def normalize_regions(raw_regions: Any, original_size: Dict[str, int], vlm_size: Dict[str, int]) -> List[Dict[str, Any]]:
    """Validate VLM-image pixel bboxes and map them to original-image pixels."""
    if not isinstance(raw_regions, list):
        return []
    regions: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_regions):
        if not isinstance(raw, dict):
            continue
        bbox = normalize_bbox(raw.get("bbox"), vlm_size, original_size)
        if bbox is None:
            continue
        region_type = str(raw.get("type") or "").strip()
        if region_type not in LAYOUT_REGION_TYPES:
            region_type = "complex_visual_region"
        confidence = clamp_float(raw.get("confidence"), 0.0, 1.0)
        region = {
            "id": str(raw.get("id") or f"region_{index + 1:03d}"),
            "type": region_type,
            "bbox": bbox,
            "pixel_bbox": bbox,
            "coordinate_system": COORDINATE_SYSTEM,
            "confidence": confidence,
        }
        regions.append(region)
    return regions


def normalize_bbox(raw_bbox: Any, vlm_size: Dict[str, int], original_size: Dict[str, int]) -> Optional[Dict[str, int]]:
    """Clamp VLM-image pixel bbox and map it to original-image pixels."""
    if not isinstance(raw_bbox, dict):
        return None
    vlm_width = max(1, int(vlm_size.get("width") or original_size["width"]))
    vlm_height = max(1, int(vlm_size.get("height") or original_size["height"]))
    x = clamp_int(raw_bbox.get("x"), 0, vlm_width)
    y = clamp_int(raw_bbox.get("y"), 0, vlm_height)
    width = clamp_int(raw_bbox.get("width"), 0, vlm_width - x)
    height = clamp_int(raw_bbox.get("height"), 0, vlm_height - y)
    if width <= 0 or height <= 0:
        return None
    scale_x = original_size["width"] / vlm_width
    scale_y = original_size["height"] / vlm_height
    x1 = round(x * scale_x)
    y1 = round(y * scale_y)
    x2 = round((x + width) * scale_x)
    y2 = round((y + height) * scale_y)
    return {"x": x1, "y": y1, "width": max(0, x2 - x1), "height": max(0, y2 - y1)}


def normalize_reading_order(raw_order: Any, regions: List[Dict[str, Any]]) -> List[str]:
    """Keep valid reading-order ids; fallback to top-left visual order."""
    valid_ids = {region["id"] for region in regions}
    if isinstance(raw_order, list):
        order = [str(item) for item in raw_order if str(item) in valid_ids]
        if order:
            return order
    sorted_regions = sorted(regions, key=lambda r: (r["pixel_bbox"]["y"], r["pixel_bbox"]["x"]))
    return [region["id"] for region in sorted_regions]


def draw_layout_overlay(image_path: Path, output_path: Path, regions: List[Dict[str, Any]]) -> None:
    """Draw original-pixel layout regions."""
    with Image.open(image_path) as img:
        canvas = img.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    palette = ["#ff3b30", "#007aff", "#34c759", "#ff9500", "#af52de", "#00c7be"]
    for index, region in enumerate(regions):
        pixel_bbox = region.get("pixel_bbox") or region["bbox"]
        x1 = int(pixel_bbox["x"])
        y1 = int(pixel_bbox["y"])
        x2 = x1 + int(pixel_bbox["width"])
        y2 = y1 + int(pixel_bbox["height"])
        color = palette[index % len(palette)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{region['id']} {region['type']} {region['confidence']:.2f}"
        draw.text((max(0, x1), max(0, y1 - 12)), label, fill=color, font=font)
    canvas.save(output_path)


def clamp_int(value: Any, min_value: int, max_value: int) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        number = min_value
    return max(min_value, min(max_value, number))


def clamp_float(value: Any, min_value: float, max_value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = min_value
    return max(min_value, min(max_value, number))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Step 1: full-page Layout VLM only.")
    parser.add_argument("-i", "--input", required=True, help="Input image path")
    parser.add_argument("-o", "--output", default="output/layout_step", help="Output directory")
    parser.add_argument("-c", "--config", default="config/config.yaml", help="YAML config path")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    artifact = run_layout_step(args.input, args.output, config)
    print(f"Layout VLM JSON: {artifact['json_path']}")
    print(f"Layout VLM overlay: {artifact['overlay_path']}")
    return 0 if artifact.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
