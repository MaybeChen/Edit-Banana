"""Main image-to-PPTX pipeline orchestration."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import List, Optional

from .base import ProcessingContext
from .data_types import (
    ProcessingResult,
    ElementInfo,
    BoundingBox,
    LayerLevel,
    get_layer_level,
)
from .pipeline_config import load_config
from .pipeline_mixins import PipelineVLMAndExportMixin
from .sam3_config import PromptGroup
from .vlm_enhancer import VLMEnhancer


def _optional_class(module_name: str, class_name: str):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except Exception as exc:
        print(f"[Pipeline] {class_name} unavailable: {exc}", flush=True)
        return None


TEXT_MODULE_AVAILABLE = True


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



class Pipeline(PipelineVLMAndExportMixin):
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
        if self._text_restorer is None:
            TextRestorer = _optional_class("modules.text.restorer", "TextRestorer")
            if TextRestorer is None:
                return None
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
            Sam3InfoExtractor = _optional_class("modules.sam3_info_extractor", "Sam3InfoExtractor")
            if Sam3InfoExtractor is None:
                raise RuntimeError("Sam3InfoExtractor unavailable")
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
            IconPictureProcessor = _optional_class("modules.icon_picture_processor", "IconPictureProcessor")
            if IconPictureProcessor is None:
                raise RuntimeError("IconPictureProcessor unavailable")
            self._icon_processor = IconPictureProcessor(rmbg_model_path=rmbg_path)
        return self._icon_processor
    
    @property
    def shape_processor(self) -> BasicShapeProcessor:
        if self._shape_processor is None:
            BasicShapeProcessor = _optional_class("modules.basic_shape_processor", "BasicShapeProcessor")
            if BasicShapeProcessor is None:
                raise RuntimeError("BasicShapeProcessor unavailable")
            self._shape_processor = BasicShapeProcessor()
        return self._shape_processor
    
    @property
    def metric_evaluator(self) -> MetricEvaluator:
        if self._metric_evaluator is None:
            MetricEvaluator = _optional_class("modules.metric_evaluator", "MetricEvaluator")
            if MetricEvaluator is None:
                raise RuntimeError("MetricEvaluator unavailable")
            self._metric_evaluator = MetricEvaluator()
        return self._metric_evaluator
    
    @property
    def refinement_processor(self) -> RefinementProcessor:
        if self._refinement_processor is None:
            RefinementProcessor = _optional_class("modules.refinement_processor", "RefinementProcessor")
            if RefinementProcessor is None:
                raise RuntimeError("RefinementProcessor unavailable")
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
            VLMExportValidator = _optional_class("modules.vlm_export_validator", "VLMExportValidator")
            if VLMExportValidator is None:
                raise RuntimeError("VLMExportValidator unavailable")
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
            if self._recognition_mode() == "vlm_only":
                return self._process_image_vlm_only(context, img_output_dir, img_stem)

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
