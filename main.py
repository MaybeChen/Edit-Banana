#!/usr/bin/env python3
"""
Edit Banana — CLI entry: image to editable PowerPoint.

Pipeline: input image -> PaddleOCR -> VLM text styles -> coarse SAM3 -> VLM segmentation/attribute enrichment -> PPTX quality loop -> output .pptx.
Requires: config/config.yaml (sam3.checkpoint_path, sam3.bpe_path), SAM3 library and weights, Tesseract or PaddleOCR.
See README and docs/SETUP_SAM3.md.

Usage:
    python main.py -i input/test.png
    python main.py
    python main.py -i input/test.png -o output/custom/
    python main.py -i input/test.png --refine
    python main.py -i input/test.png --no-text
"""

import os
import sys
import argparse
import warnings
import yaml
import json
from pathlib import Path
from typing import Optional, List

# Skip PaddleX model host connectivity check to avoid startup delay
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
# Suppress requests urllib3/chardet version warning
warnings.filterwarnings("ignore", message=".*doesn't match a supported version.*")

# Project root on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from modules import (
    # Core processors
    Sam3InfoExtractor,
    IconPictureProcessor,
    BasicShapeProcessor,
    MetricEvaluator,
    RefinementProcessor,
    VLMElementRefiner,
    VLMLayoutRefiner,
    VLMExportValidator,
    
    # Text (modules/text/)
    TextRestorer,

    # Context and data types
    ProcessingContext,
    ProcessingResult,
    ElementInfo,
    BoundingBox,
    LayerLevel,
    get_layer_level,
)

# Prompt groups enum
from modules.sam3_info_extractor import PromptGroup
from modules.vlm_enhancer import VLMEnhancer

# Text module available (depends on ocr/coord_processor etc.)
TEXT_MODULE_AVAILABLE = TextRestorer is not None


class _VLMProcessorAdapter:
    """Compatibility adapter for older pipeline properties like vlm_element_refiner."""

    def __init__(self, enhancer: VLMEnhancer, method_name: str):
        self.enhancer = enhancer
        self.method_name = method_name

    def process(self, context: ProcessingContext) -> ProcessingResult:
        result = getattr(self.enhancer, self.method_name)(context)
        return ProcessingResult(
            success=True,
            elements=context.elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata=result if isinstance(result, dict) else {"result": result},
        )


# ======================== config ========================
def load_config() -> dict:
    """Load config/config.yaml."""
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    
    if not os.path.exists(config_path):
        print(f"Warning: config file not found at {config_path}, using defaults")
        return {
            'paths': {
                'input_dir': './input',
                'output_dir': './output',
            },
            'recognition': {
                'mode': 'sam3_first',
            },
            'multimodal': {
                'mode': 'api',
                'api_key': '',
                'base_url': '',
                'model': '',
                'x_hw_id': '',
                'x_hw_appkey': '',
                'max_tokens': 4000,
                'timeout': 60,
                'enabled': False,
                'use_for': {
                    'prompt_planning': True,
                    'text_style': True,
                    'segmentation_refine': True,
                    'element_refine': False,
                    'region_refine': False,
                    'layout_refine': False,
                    'element_attributes': True,
                    'export_validate': True,
                },
                'thresholds': {
                    'element_refine_confidence': 0.75,
                    'region_refine_confidence': 0.70,
                    'layout_refine_confidence': 0.70,
                    'text_style_confidence': 0.65,
                    'segmentation_refine_confidence': 0.65,
                    'element_attribute_confidence': 0.65,
                    'vlm_structure_confidence': 0.60,
                },
            },
            'ocr': {
                'engine': 'paddleocr',
                'paddleocr': {
                    'lang': 'ch',
                    'use_angle_cls': False,
                    'allow_download': True,
                    'allow_fallback_to_tesseract': False,
                    'allow_legacy_fallback': False,
                    'ocr_version': 'PP-OCRv6',
                    'text_detection_model_name': 'PP-OCRv6_medium_det',
                    'text_recognition_model_name': 'PP-OCRv6_medium_rec',
                    'textline_orientation_model_name': 'PP-LCNet_x1_0_textline_ori',
                    'text_det_limit_side_len': 64,
                    'text_det_limit_type': 'min',
                    'text_det_thresh': 0.3,
                    'text_det_box_thresh': 0.6,
                    'text_det_unclip_ratio': 1.5,
                    'text_rec_score_thresh': 0.0,
                    'scale': 1.0,
                    'min_confidence': 0.30,
                },
            }
        }
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ======================== pipeline ========================
class Pipeline:
    """Runs segmentation, text extraction, and direct PPTX export."""

    def __init__(self, config: dict = None):
        self.config = config or load_config()
        self._text_restorer = None
        self._sam3_extractor = None
        self._icon_processor = None
        self._shape_processor = None
        self._metric_evaluator = None
        self._refinement_processor = None
        self._vlm_enhancer = None
        self._vlm_element_refiner = None
        self._vlm_region_refiner = None
        self._vlm_layout_refiner = None
        self._coarse_sam_prompts_applied = False
    
    @property
    def text_restorer(self):
        """OCR/text step; None if deps missing."""
        if self._text_restorer is None and TextRestorer is not None:
            ocr_config = self.config.get("ocr") or {}
            ocr_engine = os.environ.get("OCR_ENGINE", ocr_config.get("engine", "paddleocr"))
            self._text_restorer = TextRestorer(
                formula_engine="none",
                ocr_engine=ocr_engine,
                ocr_config=ocr_config,
            )
        return self._text_restorer
    
    @property
    def sam3_extractor(self) -> Sam3InfoExtractor:
        if self._sam3_extractor is None:
            self._sam3_extractor = Sam3InfoExtractor()
            self._apply_coarse_sam_prompts_if_enabled()
        return self._sam3_extractor

    def _apply_coarse_sam_prompts_if_enabled(self) -> None:
        multimodal_cfg = self.config.get("multimodal") or {}
        if self._coarse_sam_prompts_applied or multimodal_cfg.get("coarse_sam_prompts", True) is False:
            return
        if self._sam3_extractor is None:
            return
        coarse_prompts = {
            PromptGroup.IMAGE: ["icon", "picture"],
            PromptGroup.BASIC_SHAPE: ["shape", "rectangle", "rounded rectangle", "circle", "ellipse", "diamond", "triangle", "hexagon", "cylinder"],
            PromptGroup.ARROW: ["arrow", "connector", "line"],
            PromptGroup.BACKGROUND: ["background", "container", "panel"],
        }
        for group, prompts in coarse_prompts.items():
            cfg = self._sam3_extractor.prompt_groups.get(group)
            if cfg is not None:
                cfg.prompts = prompts.copy()
        self._coarse_sam_prompts_applied = True
    
    @property
    def icon_processor(self) -> IconPictureProcessor:
        if self._icon_processor is None:
            rmbg_cfg = self.config.get("rmbg") or {}
            rmbg_path = rmbg_cfg.get("model_path")
            self._icon_processor = IconPictureProcessor(rmbg_model_path=rmbg_path)
        return self._icon_processor
    
    @property
    def shape_processor(self) -> BasicShapeProcessor:
        if self._shape_processor is None:
            self._shape_processor = BasicShapeProcessor()
        return self._shape_processor
    
    @property
    def metric_evaluator(self) -> MetricEvaluator:
        if self._metric_evaluator is None:
            self._metric_evaluator = MetricEvaluator()
        return self._metric_evaluator
    
    @property
    def refinement_processor(self) -> RefinementProcessor:
        if self._refinement_processor is None:
            self._refinement_processor = RefinementProcessor()
        return self._refinement_processor

    @property
    def vlm_enhancer(self) -> VLMEnhancer:
        if self._vlm_enhancer is None:
            self._vlm_enhancer = VLMEnhancer(self.config)
        return self._vlm_enhancer

    @property
    def vlm_element_refiner(self):
        if self._vlm_element_refiner is None:
            self._vlm_element_refiner = _VLMProcessorAdapter(self.vlm_enhancer, "refine_elements")
        return self._vlm_element_refiner

    @property
    def vlm_region_refiner(self):
        if self._vlm_region_refiner is None:
            self._vlm_region_refiner = _VLMProcessorAdapter(self.vlm_enhancer, "refine_regions")
        return self._vlm_region_refiner

    @property
    def vlm_layout_refiner(self):
        if self._vlm_layout_refiner is None:
            self._vlm_layout_refiner = _VLMProcessorAdapter(self.vlm_enhancer, "refine_layout")
        return self._vlm_layout_refiner
    
    @property
    def vlm_export_validator(self) -> VLMExportValidator:
        if self._vlm_export_validator is None:
            vlm_config = self.config.get("multimodal") or {}
            self._vlm_export_validator = VLMExportValidator(vlm_config)
        return self._vlm_export_validator

    def process_image(self,
                      image_path: str,
                      output_dir: str = None,
                      with_refinement: bool = False,
                      with_text: bool = True,
                      groups: List[PromptGroup] = None) -> Optional[str]:
        """Run the VLM-first pipeline on one image. Returns final PPTX path or None."""
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}")

        if output_dir is None:
            output_dir = self.config.get('paths', {}).get('output_dir', './output')

        img_stem = Path(image_path).stem
        img_output_dir = os.path.join(output_dir, img_stem)
        os.makedirs(img_output_dir, exist_ok=True)

        print("\n[0] Preprocess...")
        context = ProcessingContext(image_path=image_path, output_dir=img_output_dir)
        context.intermediate_results['original_image_path'] = image_path
        context.intermediate_results['was_upscaled'] = False
        context.intermediate_results['upscale_factor'] = 1.0

        try:
            text_blocks = []
            if with_text and self.text_restorer is not None:
                print("\n[1] PaddleOCR text extraction...")
                try:
                    text_blocks = self.text_restorer.process_image(image_path)
                    for idx, block in enumerate(text_blocks):
                        block.setdefault("id", idx)
                    context.intermediate_results['ocr_text_blocks'] = text_blocks
                    if hasattr(self.text_restorer, "save_ocr_artifacts"):
                        ocr_artifact_path = self.text_restorer.save_ocr_artifacts(img_output_dir, image_path)
                        context.intermediate_results['ocr_result_json'] = ocr_artifact_path
                        print(f"   OCR result: {ocr_artifact_path}")
                    else:
                        self._save_json(os.path.join(img_output_dir, "ocr_result.json"), {"text_blocks": text_blocks})
                    if hasattr(self.text_restorer, "save_ocr_overlay"):
                        ocr_overlay_path = self.text_restorer.save_ocr_overlay(img_output_dir, image_path)
                        context.intermediate_results['ocr_overlay'] = ocr_overlay_path
                        print(f"   OCR overlay: {ocr_overlay_path}")
                    print(f"   Text blocks: {len(text_blocks)}")
                except Exception as e:
                    print(f"   Text step failed: {e}")
                    print("   Continuing without text...")
            elif with_text:
                print("\n[1] PaddleOCR text extraction (skipped - deps)")
            else:
                print("\n[1] PaddleOCR text extraction (skipped)")

            if text_blocks:
                print("\n[2] VLM text style enrichment...")
                styled = self.vlm_enhancer.enrich_text_styles(context, text_blocks)
                text_blocks = styled.get("text_blocks", text_blocks)
                context.intermediate_results['ocr_text_blocks'] = text_blocks
                text_style_path = os.path.join(img_output_dir, "vlm_text_result.json")
                self._save_json(text_style_path, {"text_blocks": text_blocks, "changes": styled.get("changes", [])})
                context.intermediate_results['vlm_text_result_json'] = text_style_path
                print(f"   VLM text styles: {styled.get('updated', 0)} updated")
                print(f"   VLM text result: {text_style_path}")

            if self._recognition_mode() == "vlm_only":
                print("\n[3] VLM-only structure recognition...")
                self._initialize_canvas_from_image(context)
                structure_result = self.vlm_enhancer.recognize_structure(context)
                structure_path = os.path.join(img_output_dir, "vlm_structure_result.json")
                self._save_json(structure_path, structure_result)
                context.intermediate_results['vlm_structure_result_json'] = structure_path
                if not structure_result.get("recognized"):
                    raise Exception(f"VLM-only recognition failed: {structure_result.get('error', 'no structured result')}")
                overlay_path = self._save_vlm_structure_overlay(context, img_output_dir)
                context.intermediate_results['vlm_structure_overlay'] = overlay_path
                print(f"   VLM structure overlay: {overlay_path}")
                if structure_result.get("raw_count", 0) and not context.elements:
                    print("   Warning: VLM returned raw items but none passed schema/bbox validation")
                print(f"   VLM-only elements: {len(context.elements)}")

                print("\n[4] VLM element attribute enrichment...")
                attr_result = self.vlm_enhancer.enrich_element_attributes(context)
                attr_path = os.path.join(img_output_dir, "vlm_element_attributes.json")
                self._save_json(attr_path, {"elements": [e.to_dict() for e in context.elements], "changes": attr_result.get("changes", [])})
                context.intermediate_results['vlm_element_attributes_json'] = attr_path
                print(f"   VLM element attributes: {attr_result.get('updated', 0)} updated")

                print("\n[5] Direct PPTX generation + VLM quality loop...")
                pptx_path = self._run_pptx_quality_loop(context, text_blocks, img_output_dir, img_stem)
                print(f"\n{'='*60}\nDone.\n{'='*60}")
                return pptx_path

            print("\n[3] SAM3 segmentation (coarse prompts)...")
            if groups:
                all_elements = []
                last_result = None
                for group in groups:
                    last_result = self.sam3_extractor.extract_by_group(context, group)
                    all_elements.extend(last_result.elements)
                for i, elem in enumerate(all_elements):
                    elem.id = i
                context.elements = all_elements
                if last_result:
                    context.canvas_width = last_result.canvas_width
                    context.canvas_height = last_result.canvas_height
            else:
                result = self.sam3_extractor.process(context)
                if not result.success:
                    raise Exception(f"SAM3 extraction failed: {result.error_message}")
                context.elements = result.elements
                context.canvas_width = result.canvas_width
                context.canvas_height = result.canvas_height
            print(f"   SAM3 elements: {len(context.elements)}")
            sam3_vis_path = os.path.join(img_output_dir, "sam3_extraction.png")
            self.sam3_extractor.save_visualization(context, sam3_vis_path)
            sam3_meta_path = os.path.join(img_output_dir, "sam3_metadata.json")
            self.sam3_extractor.save_metadata(context, sam3_meta_path)
            context.intermediate_results['sam3_visualization'] = sam3_vis_path
            context.intermediate_results['sam3_metadata_json'] = sam3_meta_path

            print("\n[4] VLM segmentation refinement...")
            seg_result = self.vlm_enhancer.refine_segmentation(context)
            vlm_sam3_vis_path = os.path.join(img_output_dir, "sam3_vlm_refined.png")
            self.sam3_extractor.save_visualization(context, vlm_sam3_vis_path)
            vlm_sam3_json_path = os.path.join(img_output_dir, "sam3_vlm_refined.json")
            self._save_json(vlm_sam3_json_path, {"elements": [e.to_dict() for e in context.elements], "changes": seg_result})
            context.intermediate_results['sam3_vlm_refined_visualization'] = vlm_sam3_vis_path
            context.intermediate_results['sam3_vlm_refined_json'] = vlm_sam3_json_path
            print(f"   VLM segmentation: {seg_result.get('updated', 0)} updated, {seg_result.get('added', 0)} added")

            print("\n[5] Shape/icon CV processing...")
            result = self.icon_processor.process(context)
            print(f"   Icons: {result.metadata.get('processed_count', 0)}")
            result = self.shape_processor.process(context)
            print(f"   Shapes: {result.metadata.get('processed_count', 0)}")
            vlm_layout_result = self.vlm_enhancer.refine_layout(context)
            if vlm_layout_result.get("updated"):
                print(f"   VLM layout refinements: {vlm_layout_result.get('updated')}")

            print("\n[6] VLM element attribute enrichment...")
            attr_result = self.vlm_enhancer.enrich_element_attributes(context)
            attr_path = os.path.join(img_output_dir, "vlm_element_attributes.json")
            self._save_json(attr_path, {"elements": [e.to_dict() for e in context.elements], "changes": attr_result.get("changes", [])})
            context.intermediate_results['vlm_element_attributes_json'] = attr_path
            print(f"   VLM element attributes: {attr_result.get('updated', 0)} updated")

            print("\n[7] Direct PPTX generation + VLM quality loop...")
            pptx_path = self._run_pptx_quality_loop(context, text_blocks, img_output_dir, img_stem)

            print(f"\n{'='*60}\nDone.\n{'='*60}")
            return pptx_path

        except Exception as e:
            print(f"\nFailed: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def _save_json(path: str, data: dict) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return path

    def _recognition_mode(self) -> str:
        recognition_cfg = self.config.get("recognition") or {}
        mode = recognition_cfg.get("mode") or (self.config.get("multimodal") or {}).get("recognition_mode") or "sam3_first"
        return str(mode).strip().lower().replace("-", "_")

    @staticmethod
    def _initialize_canvas_from_image(context: ProcessingContext) -> None:
        from PIL import Image
        with Image.open(context.image_path) as img:
            context.canvas_width, context.canvas_height = img.size

    @staticmethod
    def _save_vlm_structure_overlay(context: ProcessingContext, output_dir: str) -> str:
        """Save a debug image with VLM-only recognized elements annotated."""
        from PIL import Image, ImageDraw, ImageFont
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "vlm_structure_overlay.png")
        with Image.open(context.image_path) as img:
            canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        palette = {
            "container": "#00a6ff",
            "rounded rectangle": "#ff5c8a",
            "rounded_rectangle": "#ff5c8a",
            "rectangle": "#ff5c8a",
            "cylinder": "#a855f7",
            "arrow": "#2563eb",
            "connector": "#2563eb",
            "line": "#2563eb",
            "icon": "#22c55e",
            "picture": "#22c55e",
            "logo": "#22c55e",
            "chart": "#22c55e",
        }
        for elem in context.elements:
            elem_type = str(elem.element_type or "").lower()
            color = palette.get(elem_type, "#f97316")
            bbox = elem.bbox
            if elem_type in {"arrow", "connector", "line"}:
                start = elem.arrow_start or (bbox.x1, (bbox.y1 + bbox.y2) / 2)
                end = elem.arrow_end or (bbox.x2, (bbox.y1 + bbox.y2) / 2)
                draw.line([start, end], fill=color, width=max(2, int(elem.stroke_width or 1)))
                draw.rectangle(bbox.to_list(), outline=color, width=1)
            else:
                draw.rectangle(bbox.to_list(), outline=color, width=2)
            label = f"{elem.id}:{elem.element_type}"
            text_xy = (max(0, bbox.x1), max(0, bbox.y1 - 12))
            draw.text(text_xy, label, fill=color, font=font)
        canvas.save(output_path)
        return output_path

    def _run_pptx_quality_loop(self, context: ProcessingContext, text_blocks: List[dict], output_dir: str, img_stem: str) -> Optional[str]:
        pptx_path = None
        max_rounds = int((self.config.get("multimodal") or {}).get("max_quality_rounds", 3) or 3)
        quality_threshold = float((self.config.get("multimodal") or {}).get("quality_threshold", 90) or 90)
        for round_idx in range(1, max_rounds + 1):
            pptx_elements = self._prepare_pptx_elements(context)
            pptx_path = self._export_pptx_direct(context, pptx_elements, text_blocks, output_dir, img_stem)
            context.intermediate_results['pptx_output'] = pptx_path
            print(f"   Round {round_idx} PPTX: {pptx_path}")
            validation = self.vlm_enhancer.validate_pptx_export(context, pptx_path, round_idx)
            score = float(validation.get("score", 100) or 0)
            print(f"   Round {round_idx} quality score: {score:.1f}/100")
            if score >= quality_threshold or round_idx >= max_rounds:
                break
            repairs = self.vlm_enhancer.apply_export_repairs(context, validation)
            if repairs.get("updated", 0) == 0 and repairs.get("added", 0) == 0:
                print("   No structured repairs returned; stopping quality loop")
                break
        return pptx_path

    def _prepare_pptx_elements(self, context: ProcessingContext) -> List[ElementInfo]:
        """Prepare recognized elements for direct PPTX export."""
        prepared = []
        for elem in context.elements:
            elem_type = elem.element_type.lower()

            if elem_type in {'arrow', 'line', 'connector'}:
                if self._is_border_like_connector(elem, context):
                    elem.element_type = "container"
                    elem.fill_color = "none"
                    elem.stroke_color = elem.stroke_color or "#000000"
                    elem.stroke_width = max(1, int(elem.stroke_width or 1))
                    elem.layer_level = LayerLevel.BACKGROUND.value
                else:
                    if self._is_duplicate_line_fragment(elem, context.elements):
                        elem.processing_notes.append("Skipped duplicate line fragment contained by an arrow")
                        continue
                    if not elem.arrow_start or not elem.arrow_end:
                        elem.arrow_start, elem.arrow_end = self._infer_edge_points(elem, context.elements)
                    inferred_heads = self._infer_arrow_heads_from_polygon(elem)
                    if inferred_heads and elem.arrow_heads in {None, "none"}:
                        elem.arrow_heads = inferred_heads
                    if (
                        self._recognition_mode() != "vlm_only"
                        and elem.line_style not in {"dashed", "dotted"}
                        and self._edge_looks_dashed(elem, context)
                    ):
                        elem.line_style = "dashed"
                    if self._edge_looks_curved(elem):
                        elem.arrow_style = "curved"
                    elem.layer_level = LayerLevel.ARROW.value

            elif self._is_background_like_element(elem, context):
                # Large panels/frames are often detected by SAM3 as generic image or
                # shape prompts. Export them as transparent background containers so
                # their borders remain visible without covering inner content.
                elem.element_type = "container"
                elem.fill_color = "none"
                elem.stroke_color = elem.stroke_color or "#000000"
                elem.stroke_width = max(1, int(elem.stroke_width or 1))
                elem.layer_level = LayerLevel.BACKGROUND.value

            elif elem_type in {'icon', 'picture', 'logo', 'chart', 'function_graph'}:
                if elem.base64:
                    elem.layer_level = LayerLevel.IMAGE.value
                else:
                    # VLM-only mode has no SAM mask crop for icons/pictures; export a
                    # visible placeholder box instead of silently dropping the element.
                    elem.element_type = "rectangle"
                    elem.fill_color = "none"
                    elem.stroke_color = elem.stroke_color or "#000000"
                    elem.stroke_width = max(1, int(elem.stroke_width or 1))
                    elem.layer_level = LayerLevel.BASIC_SHAPE.value

            else:
                if elem_type in {"rectangle", "rounded rectangle", "rounded_rectangle", "container", "cylinder"}:
                    elem.fill_color = "none"
                else:
                    elem.fill_color = elem.fill_color or "#ffffff"
                elem.stroke_color = elem.stroke_color or "#000000"
                elem.stroke_width = max(1, int(elem.stroke_width or 1))
                elem.layer_level = LayerLevel.BASIC_SHAPE.value

            prepared.append(elem)

        if self._recognition_mode() != "vlm_only":
            prepared.extend(self._detect_missing_divider_lines(context, prepared))
        return prepared



    @staticmethod
    def _is_border_like_connector(elem: ElementInfo, context: ProcessingContext) -> bool:
        """Detect connector candidates that are actually container/panel borders."""
        elem_type = elem.element_type.lower()
        if elem_type not in {"connector", "line"}:
            return False
        if elem.arrow_heads not in {None, "none"}:
            return False
        if not elem.bbox:
            return False
        canvas_area = max(1, int((context.canvas_width or 0) * (context.canvas_height or 0)))
        area_ratio = elem.bbox.area / canvas_area
        # A true connector is usually thin. A large two-dimensional connector box is
        # typically a misclassified frame/container boundary, such as a grouped panel.
        return bool(elem.bbox.width >= 40 and elem.bbox.height >= 40 and area_ratio >= 0.025)

    @staticmethod
    def _edge_looks_curved(elem: ElementInfo) -> bool:
        if elem.element_type.lower() not in {"arrow", "connector", "line"} or not elem.polygon:
            return False
        if not elem.bbox or min(elem.bbox.width, elem.bbox.height) < 20:
            return False
        return len(elem.polygon) >= 5

    def _infer_arrow_heads_from_polygon(self, elem: ElementInfo) -> Optional[str]:
        """Infer start/end/both arrowheads from sharp polygon tips near endpoints."""
        if elem.element_type.lower() not in {"arrow", "connector", "line"} or not elem.polygon:
            return None
        if not elem.arrow_start or not elem.arrow_end:
            return None
        points = [(float(p[0]), float(p[1])) for p in elem.polygon if len(p) >= 2]
        if len(points) < 4:
            return None

        sharp_points = []
        for point in points:
            angle = self._polygon_angle_at_point(points, point)
            if angle is not None and angle <= 75:
                sharp_points.append(point)
        if not sharp_points:
            return None

        def near(candidate, endpoint):
            import math
            tolerance = max(10.0, min(elem.bbox.width, elem.bbox.height) * 0.35)
            return math.hypot(candidate[0] - endpoint[0], candidate[1] - endpoint[1]) <= tolerance

        start_has = any(near(point, elem.arrow_start) for point in sharp_points)
        end_has = any(near(point, elem.arrow_end) for point in sharp_points)
        if start_has and end_has:
            return "both"
        if start_has:
            return "start"
        if end_has:
            return "end"
        return None

    @staticmethod
    def _polygon_angle_at_point(points, point) -> Optional[float]:
        import math
        try:
            idx = points.index(point)
        except ValueError:
            return None
        prev_p = points[(idx - 1) % len(points)]
        next_p = points[(idx + 1) % len(points)]
        v1 = (prev_p[0] - point[0], prev_p[1] - point[1])
        v2 = (next_p[0] - point[0], next_p[1] - point[1])
        len1 = math.hypot(*v1)
        len2 = math.hypot(*v2)
        if len1 <= 0 or len2 <= 0:
            return None
        cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)))
        return math.degrees(math.acos(cos_a))

    def _detect_missing_divider_lines(self, context: ProcessingContext, existing_elements: List[ElementInfo]) -> List[ElementInfo]:
        """Use lightweight image analysis to recover long horizontal divider lines."""
        if not context.image_path:
            return []
        from PIL import Image
        import numpy as np
        with Image.open(context.image_path) as img:
            gray = np.array(img.convert("L"))

        dark = gray < 110
        height, width = dark.shape
        candidates = []
        for y in range(height):
            if y < 8 or y > height - 8:
                continue
            xs = np.flatnonzero(dark[y])
            if len(xs) < width * 0.35:
                continue
            runs = self._continuous_runs(xs)
            for x1, x2 in runs:
                if x2 - x1 >= width * 0.35:
                    candidates.append((x1, y, x2, y + 1))

        merged = self._merge_horizontal_line_candidates(candidates)
        new_elements = []
        next_id = max([e.id for e in existing_elements], default=-1) + 1
        for x1, y1, x2, y2 in merged:
            if self._overlaps_existing_line([x1, y1, x2, y2], existing_elements):
                continue
            elem = ElementInfo(
                id=next_id + len(new_elements),
                element_type="line",
                bbox=BoundingBox(int(x1), int(y1), int(x2), int(max(y2, y1 + 2))),
                score=0.8,
                source_prompt="cv_horizontal_divider",
                stroke_color="#000000",
                stroke_width=1,
                line_style="solid",
                layer_level=LayerLevel.ARROW.value,
            )
            cy = (elem.bbox.y1 + elem.bbox.y2) / 2
            elem.arrow_start = (elem.bbox.x1, cy)
            elem.arrow_end = (elem.bbox.x2, cy)
            elem.arrow_heads = "none"
            elem.processing_notes.append("cv_horizontal_divider:add")
            new_elements.append(elem)
        return new_elements

    @staticmethod
    def _continuous_runs(indices):
        runs = []
        start = int(indices[0]) if len(indices) else 0
        prev = start
        for value in indices[1:]:
            value = int(value)
            if value > prev + 1:
                runs.append((start, prev + 1))
                start = value
            prev = value
        if len(indices):
            runs.append((start, prev + 1))
        return runs

    @staticmethod
    def _merge_horizontal_line_candidates(candidates):
        if not candidates:
            return []
        candidates = sorted(candidates, key=lambda box: (box[1], box[0]))
        merged = []
        for box in candidates:
            x1, y1, x2, y2 = box
            if not merged or y1 - merged[-1][3] > 2 or x1 > merged[-1][2] + 8:
                merged.append([x1, y1, x2, y2])
            else:
                merged[-1][0] = min(merged[-1][0], x1)
                merged[-1][1] = min(merged[-1][1], y1)
                merged[-1][2] = max(merged[-1][2], x2)
                merged[-1][3] = max(merged[-1][3], y2)
        return merged

    @staticmethod
    def _overlaps_existing_line(box, elements) -> bool:
        x1, y1, x2, y2 = box
        for elem in elements:
            if elem.element_type.lower() not in {"line", "connector", "arrow"}:
                continue
            bx1, by1, bx2, by2 = elem.bbox.to_list()
            if x2 <= bx1 or bx2 <= x1 or y2 <= by1 or by2 <= y1:
                continue
            inter = (min(x2, bx2) - max(x1, bx1)) * (min(y2, by2) - max(y1, by1))
            if inter / max(1, (x2 - x1) * max(1, y2 - y1)) > 0.5:
                return True
        return False

    @staticmethod
    def _is_background_like_element(elem: ElementInfo, context: ProcessingContext) -> bool:
        elem_type = elem.element_type.lower()
        if elem_type in {'section_panel', 'title_bar', 'container', 'background', 'panel'}:
            return True
        canvas_area = max(1, int((context.canvas_width or 0) * (context.canvas_height or 0)))
        return bool(elem.bbox and elem.bbox.area / canvas_area >= 0.12)

    @staticmethod
    def _export_pptx_direct(context: ProcessingContext, elements: List[ElementInfo], text_blocks: List[dict], output_dir: str, img_stem: str) -> Optional[str]:
        from pptx_exporter import (
            export_elements_to_pptx,
            is_pptx_export_available,
            missing_pptx_dependency_message,
        )
        if not is_pptx_export_available():
            print(f"   PPTX skipped: {missing_pptx_dependency_message()}")
            return None
        pptx_path = os.path.join(output_dir, f"{img_stem}.pptx")
        return export_elements_to_pptx(
            elements=elements,
            text_blocks=text_blocks,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            pptx_path=pptx_path,
        )

    def _is_duplicate_line_fragment(self, elem, elements) -> bool:
        """Skip short line detections that are already part of an arrow mask."""
        if elem.element_type.lower() != "line":
            return False
        line_box = elem.bbox.to_list()
        line_area = max(1, elem.bbox.area)
        for other in elements:
            if other is elem or other.element_type.lower() not in {"arrow", "connector"}:
                continue
            other_box = other.bbox.to_list()
            x1 = max(line_box[0], other_box[0])
            y1 = max(line_box[1], other_box[1])
            x2 = min(line_box[2], other_box[2])
            y2 = min(line_box[3], other_box[3])
            if x2 <= x1 or y2 <= y1:
                continue
            if ((x2 - x1) * (y2 - y1)) / line_area >= 0.65:
                return True
        return False

    def _edge_looks_dashed(self, elem, context: ProcessingContext = None) -> bool:
        """Heuristically detect dashed/dotted connectors from the source crop.

        SAM3 only returns element type and geometry, so dashed source lines need a
        lightweight pixel check before PPTX export. We sample the long axis
        of thin line/connector detections and mark it dashed when ink occupancy is
        broken into several separated runs.
        """
        if elem.element_type.lower() not in {"arrow", "line", "connector"}:
            return False
        if context is None or not getattr(context, "image_path", None):
            return False
        bbox = elem.bbox
        if max(bbox.width, bbox.height) < 40:
            return False
        try:
            from PIL import Image
            import numpy as np
            with Image.open(context.image_path) as img:
                gray = img.convert("L")
                x1, y1, x2, y2 = map(int, bbox.to_list())
                pad = 3
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(gray.width, x2 + pad)
                y2 = min(gray.height, y2 + pad)
                crop = np.array(gray.crop((x1, y1, x2, y2)))
            if crop.size == 0:
                return False
            ink = crop < 180
            profile = ink.any(axis=0) if bbox.width >= bbox.height else ink.any(axis=1)
            runs = []
            run = 0
            for value in profile:
                if bool(value):
                    run += 1
                elif run:
                    runs.append(run)
                    run = 0
            if run:
                runs.append(run)
            if len(runs) < 3:
                return False
            coverage = sum(runs) / max(1, len(profile))
            return coverage < 0.72
        except Exception:
            return False

    def _infer_edge_points(self, elem, elements=None) -> tuple:
        """Infer connector endpoints from the arrow/line geometry.

        Arrows are intentionally exported as coordinate-based edges instead of
        binding to source/target elements. This avoids incorrect attachments when
        relationship inference is noisy and preserves the source image geometry.
        """
        bbox = elem.bbox
        cx = (bbox.x1 + bbox.x2) / 2
        cy = (bbox.y1 + bbox.y2) / 2
        elem_type = elem.element_type.lower()

        if elem_type in {"line", "connector"} or not elem.polygon:
            if bbox.width >= bbox.height:
                return (bbox.x1, cy), (bbox.x2, cy)
            return (cx, bbox.y1), (cx, bbox.y2)

        points = [(float(p[0]), float(p[1])) for p in elem.polygon if len(p) >= 2]
        if len(points) < 2:
            if bbox.width >= bbox.height:
                return (bbox.x1, cy), (bbox.x2, cy)
            return (cx, bbox.y1), (cx, bbox.y2)

        tip = self._find_sharpest_polygon_point(points)
        start = max(points, key=lambda p: (p[0] - tip[0]) ** 2 + (p[1] - tip[1]) ** 2)
        return start, tip

    def _find_sharpest_polygon_point(self, points):
        """Find the most likely arrow head tip as the sharpest polygon vertex."""
        if len(points) < 3:
            return points[-1]
        import math
        best_point = points[0]
        best_angle = 360.0
        n = len(points)
        for i, point in enumerate(points):
            prev_p = points[(i - 1) % n]
            next_p = points[(i + 1) % n]
            v1 = (prev_p[0] - point[0], prev_p[1] - point[1])
            v2 = (next_p[0] - point[0], next_p[1] - point[1])
            len1 = math.hypot(*v1)
            len2 = math.hypot(*v2)
            if len1 <= 0 or len2 <= 0:
                continue
            cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)))
            angle = math.degrees(math.acos(cos_a))
            if angle < best_angle:
                best_angle = angle
                best_point = point
        return best_point


# ======================== CLI ========================
def main():
    parser = argparse.ArgumentParser(
        description="Edit Banana — image to editable PPTX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py -i input/test.png
  python main.py
  python main.py -i test.png --refine
  python main.py -i test.png --groups image arrow
        """
    )
    
    parser.add_argument("-i", "--input", type=str, 
                        help="Input image path (omit to process all images in input/)")
    parser.add_argument("-o", "--output", type=str, 
                        help="Output directory (default: ./output)")
    parser.add_argument("--refine", action="store_true",
                        help="Enable quality evaluation and refinement")
    parser.add_argument("--no-text", action="store_true",
                        help="Skip text step (no OCR)")
    parser.add_argument("--groups", nargs='+', 
                        choices=['image', 'arrow', 'shape', 'background'],
                        help="Prompt groups to process (default: all)")
    parser.add_argument("--vlm-only", action="store_true",
                        help="Skip SAM3 and use VLM-only page structure recognition")
    parser.add_argument("--show-prompts", action="store_true",
                        help="Show prompt config")
    
    args = parser.parse_args()
    
    # Show prompt config
    if args.show_prompts:
        extractor = Sam3InfoExtractor()
        extractor.print_prompt_groups()
        return
    
    # Load config
    config = load_config()
    
    # Create pipeline
    if args.vlm_only:
        config.setdefault("recognition", {})["mode"] = "vlm_only"
    pipeline = Pipeline(config)
    
    # Parse group args
    groups = None
    if args.groups:
        group_map = {
            'image': PromptGroup.IMAGE,
            'arrow': PromptGroup.ARROW,
            'shape': PromptGroup.BASIC_SHAPE,
            'background': PromptGroup.BACKGROUND,
        }
        groups = [group_map[g] for g in args.groups]
    
    # Output dir
    output_dir = args.output or config.get('paths', {}).get('output_dir', './output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect images
    image_paths = []
    supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    if args.input:
        # Single image
        if not os.path.exists(args.input):
            print(f"Error: file not found {args.input}")
            sys.exit(1)
        image_paths.append(args.input)
    else:
        # Batch from input/
        input_dir = config.get('paths', {}).get('input_dir', './input')
        
        if not os.path.exists(input_dir):
            print(f"Error: input directory does not exist: {input_dir}")
            print(f"   Create it and add images, or use -i to specify an image path")
            sys.exit(1)
        
        for file in os.listdir(input_dir):
            ext = Path(file).suffix.lower()
            if ext in supported_formats:
                image_paths.append(os.path.join(input_dir, file))
        
        if not image_paths:
            print(f"Error: no supported image files in {input_dir}")
            print(f"   Supported formats: {', '.join(supported_formats)}")
            sys.exit(1)
    
    # Process
    print(f"\nProcessing {len(image_paths)} image(s)...")
    
    success_count = 0
    for img_path in image_paths:
        result = pipeline.process_image(
            img_path,
            output_dir=output_dir,
            with_refinement=args.refine,
            with_text=not args.no_text,
            groups=groups
        )
        if result:
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Done: {success_count}/{len(image_paths)} succeeded")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
