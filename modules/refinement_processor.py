import base64
import io
import os
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from .base import BaseProcessor, ProcessingContext
from .data_types import ElementInfo, BoundingBox, ProcessingResult, LayerLevel, get_layer_level
from .vlm.client import OpenAICompatibleVLMClient
from .vlm.schemas import REGION_ANALYSIS_SCHEMA
class RefinementProcessor(BaseProcessor):
    DEFAULT_CONFIG = {'min_region_area': 100, 'min_region_ratio': 0.0005, 'default_confidence': 0.5, 'expand_margin': 5, 'skip_if_mostly_white': True, 'white_threshold': 0.95, 'use_vlm': True, 'vlm_confidence_threshold': 0.7}
    VLM_CONFIG_KEYS = {'base_url', 'api_key', 'model', 'mode', 'local_base_url', 'local_api_key', 'local_model', 'timeout', 'max_tokens', 'proxy', 'ca_cert_path'}
    def __init__(self, config=None):
        super().__init__(config)
        raw_config = config or {}
        refinement_config = raw_config.get('refinement', {}) if isinstance(raw_config, dict) else {}
        flat_config = {k: v for k, v in raw_config.items() if k != 'refinement'} if isinstance(raw_config, dict) else {}
        self.refine_config = {**self.DEFAULT_CONFIG, **flat_config, **refinement_config}
        self.vlm_config = {k: v for k, v in flat_config.items() if k in self.VLM_CONFIG_KEYS}
        self.vlm_config.update(refinement_config.get('vlm', {}) if isinstance(refinement_config, dict) else {})
        for key in self.VLM_CONFIG_KEYS:
            if key in self.refine_config:
                self.vlm_config[key] = self.refine_config[key]
    def process(self, context: ProcessingContext) -> ProcessingResult:
        self._log('开始二次处理（Fallback补救）')
        bad_regions = context.intermediate_results.get('bad_regions', [])
        if not bad_regions:
            self._log('没有问题区域需要处理')
            return ProcessingResult(success=True, elements=context.elements, canvas_width=context.canvas_width, canvas_height=context.canvas_height, metadata={'new_elements_count': 0, 'regions_processed': 0, 'regions_skipped': 0})
        if not context.image_path or not os.path.exists(context.image_path):
            return ProcessingResult(success=False, error_message='图片路径无效')
        original_image = Image.open(context.image_path).convert('RGB')
        img_width, img_height = original_image.size
        img_area = img_width * img_height
        cv2_image = None
        if self.refine_config.get('skip_if_mostly_white', True):
            cv2_image = cv2.imread(context.image_path)
        new_elements = []
        skipped_count = 0
        start_id = max([elem.id for elem in context.elements], default=-1) + 1
        min_area = self.refine_config.get('min_region_area', 100)
        min_ratio = self.refine_config.get('min_region_ratio', 0.0005)
        for i, region in enumerate(bad_regions):
            try:
                bbox = region.get('bbox', [])
                if len(bbox) != 4:
                    skipped_count += 1
                    continue
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                if area < min_area or area < img_area * min_ratio:
                    self._log(f'  区域{i}面积太小({area}px)，跳过')
                    skipped_count += 1
                    continue
                if cv2_image is not None and self._is_mostly_white(cv2_image, bbox):
                    self._log(f'  区域{i}大部分为白色，跳过')
                    skipped_count += 1
                    continue
                elems = self._process_region_with_optional_vlm(region, original_image, start_id + len(new_elements), img_width, img_height, context)
                if elems:
                    new_elements.extend(elems)
                else:
                    skipped_count += 1
            except Exception as e:
                self._log(f'区域{i}处理失败: {e}')
                skipped_count += 1
        context.elements.extend(new_elements)
        self._log(f'二次处理完成: 新增{len(new_elements)}个元素，跳过{skipped_count}个')
        return ProcessingResult(success=True, elements=context.elements, canvas_width=context.canvas_width, canvas_height=context.canvas_height, metadata={'new_elements_count': len(new_elements), 'regions_processed': len(bad_regions), 'regions_skipped': skipped_count})
    def _is_mostly_white(self, cv2_image: np.ndarray, bbox: List[int]) -> bool:
        x1, y1, x2, y2 = bbox
        h, w = cv2_image.shape[:2]
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return True
        region = cv2_image[y1:y2, x1:x2]
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        white_pixels = np.sum(gray > 245)
        total_pixels = gray.size
        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 1.0
        threshold = self.refine_config.get('white_threshold', 0.85)
        return white_ratio > threshold
    def _process_region_with_optional_vlm(self, region: Dict[str, Any], original_image: Image.Image, element_id: int, img_width: int, img_height: int, context: ProcessingContext) -> List[ElementInfo]:
        vlm_attempted = bool(self.refine_config.get('use_vlm', False))
        if vlm_attempted:
            try:
                crop, expanded_bbox = self._crop_region(region, original_image, img_width, img_height)
                analyzer = self._get_vlm_region_analyzer(context)
                if analyzer is not None:
                    elements = analyzer.analyze(crop, expanded_bbox, element_id)
                    if elements:
                        for elem in elements:
                            elem.processing_notes.append('vlm_refined=true')
                        return elements
                self._log('  VLM区域分析不可用或未返回可信结构化元素，回退picture')
            except Exception as exc:
                self._log(f'  VLM区域分析失败，回退picture: {exc}')
        elem = self._process_region(region, original_image, element_id, img_width, img_height)
        if elem:
            if vlm_attempted:
                elem.processing_notes.append('vlm_fallback_picture=true')
            return [elem]
        return []
    def _get_vlm_region_analyzer(self, context: ProcessingContext) -> Optional['VLMRegionAnalyzer']:
        shared = context.shared_models.get('vlm_region_analyzer')
        if shared is not None:
            return shared
        client = context.shared_models.get('vlm_client')
        if client is None:
            client = OpenAICompatibleVLMClient(self.vlm_config)
            if not client.available:
                return None
            context.shared_models['vlm_client'] = client
        analyzer = VLMRegionAnalyzer(client, self.refine_config)
        context.shared_models['vlm_region_analyzer'] = analyzer
        return analyzer
    def _crop_region(self, region: Dict[str, Any], original_image: Image.Image, img_width: int, img_height: int) -> Tuple[Image.Image, BoundingBox]:
        bbox = region.get('bbox', [])
        if len(bbox) != 4:
            raise ValueError('bad region bbox must contain 4 values')
        x1, y1, x2, y2 = [int(v) for v in bbox]
        margin = self.refine_config.get('expand_margin', 2)
        if margin > 0:
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(img_width, x2 + margin)
            y2 = min(img_height, y2 + margin)
        if x2 <= x1 or y2 <= y1:
            raise ValueError('bad region bbox is empty after clipping')
        expanded_bbox = BoundingBox(x1, y1, x2, y2)
        return (original_image.crop((x1, y1, x2, y2)), expanded_bbox)
    def _process_region(self, region: Dict[str, Any], original_image: Image.Image, element_id: int, img_width: int, img_height: int) -> Optional[ElementInfo]:
        bbox = region.get('bbox', [])
        if len(bbox) != 4:
            return None
        x1, y1, x2, y2 = bbox
        margin = self.refine_config.get('expand_margin', 2)
        if margin > 0:
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(img_width, x2 + margin)
            y2 = min(img_height, y2 + margin)
        cropped = original_image.crop((x1, y1, x2, y2))
        base64_str = self._image_to_base64(cropped)
        confidence = self.refine_config.get('default_confidence', 0.5)
        channel = region.get('channel', 'unknown')
        area_ratio = region.get('area_ratio', 0) * 100
        missing_pixels = region.get('missing_pixels', 0)
        notes = [f'Fallback补救: 检测通道={channel}', f'区域占比={area_ratio:.2f}%, 漏检像素={missing_pixels}', region.get('description', '')]
        element = ElementInfo(id=element_id, element_type='picture', bbox=BoundingBox(x1, y1, x2, y2), score=confidence, base64=base64_str, layer_level=LayerLevel.IMAGE.value, source_prompt='refinement_fallback', processing_notes=[n for n in notes if n])
        self._generate_xml_fragment(element)
        return element
    def _generate_xml_fragment(self, element: ElementInfo):
        x1 = element.bbox.x1
        y1 = element.bbox.y1
        width = element.bbox.x2 - element.bbox.x1
        height = element.bbox.y2 - element.bbox.y1
        style = f'shape=image;verticalLabelPosition=bottom;verticalAlign=top;imageAspect=0;aspect=fixed;image=data:image/png,{element.base64};'
        cell_id = element.id + 2
        element.xml_fragment = f'<mxCell id="{cell_id}" parent="1" vertex="1" value="" style="{style}">\n  <mxGeometry x="{x1}" y="{y1}" width="{width}" height="{height}" as="geometry"/>\n</mxCell>'
    def _image_to_base64(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    def save_visualization(self, context: ProcessingContext, new_elements: List[ElementInfo], output_path: str):
        if not context.image_path or not os.path.exists(context.image_path):
            return
        img = cv2.imread(context.image_path)
        if img is None:
            return
        h, w = img.shape[:2]
        for elem in context.elements:
            if elem not in new_elements:
                x1 = max(0, min(w, elem.bbox.x1))
                y1 = max(0, min(h, elem.bbox.y1))
                x2 = max(0, min(w, elem.bbox.x2))
                y2 = max(0, min(h, elem.bbox.y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 1)
        for i, elem in enumerate(new_elements):
            x1 = max(0, min(w, elem.bbox.x1))
            y1 = max(0, min(h, elem.bbox.y1))
            x2 = max(0, min(w, elem.bbox.x2))
            y2 = max(0, min(h, elem.bbox.y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            text = f'NEW-{i}'
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(img, f'Original: {len(context.elements) - len(new_elements)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        cv2.putText(img, f'New (Fallback): {len(new_elements)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(output_path, img)
        self._log(f'保存refinement可视化结果: {output_path}')
class VLMRegionAnalyzer:
    STRUCTURED_TYPES = {'rectangle', 'rounded_rectangle', 'rounded rectangle', 'circle', 'ellipse', 'cylinder', 'arrow', 'connector', 'container', 'section_panel', 'title_bar', 'diamond', 'triangle', 'hexagon', 'parallelogram', 'cloud', 'actor', 'line', 'text', 'icon', 'chart', 'logo', 'function_graph'}
    TYPE_ALIASES = {'rounded rectangle': 'rounded_rectangle', 'round rectangle': 'rounded_rectangle', 'container': 'section_panel', 'image': 'picture', 'photo': 'picture'}
    def __init__(self, client: Any, config: Dict[str, Any]):
        self.client = client
        self.confidence_threshold = float(config.get('vlm_confidence_threshold', 0.7))
    def analyze(self, crop: Image.Image, region_bbox: BoundingBox, start_id: int) -> List[ElementInfo]:
        output = self.client.classify(self._image_to_data_url(crop), self._build_prompt(crop, region_bbox), REGION_ANALYSIS_SCHEMA)
        elements_data = self._extract_elements(output)
        elements: List[ElementInfo] = []
        for idx, item in enumerate(elements_data):
            elem = self._convert_element(item, region_bbox, start_id + idx)
            if elem:
                elements.append(elem)
        return elements
    def _extract_elements(self, output: Any) -> List[Dict[str, Any]]:
        if not isinstance(output, dict):
            raise ValueError('VLM output is not a JSON object')
        top_conf = output.get('confidence', 1.0)
        if top_conf is not None and float(top_conf) < self.confidence_threshold:
            return []
        elements = output.get('elements', [])
        if not isinstance(elements, list) or not elements:
            return []
        return [item for item in elements if isinstance(item, dict)]
    def _convert_element(self, item: Dict[str, Any], region_bbox: BoundingBox, element_id: int) -> Optional[ElementInfo]:
        confidence = float(item.get('confidence', 0.0))
        if confidence < self.confidence_threshold:
            return None
        element_type = self._normalize_type(str(item.get('element_type', 'unknown')))
        if element_type not in self.STRUCTURED_TYPES:
            return None
        bbox = self._absolute_bbox(item.get('bbox'), region_bbox)
        if bbox is None or bbox.area <= 0:
            return None
        notes = ['VLM bad-region structured element']
        if item.get('reason'):
            notes.append(str(item['reason'])[:160])
        elem = ElementInfo(id=element_id, element_type=element_type, bbox=bbox, score=confidence, layer_level=get_layer_level(element_type), source_prompt='refinement_vlm_region', processing_notes=notes)
        if item.get('line_style'):
            elem.line_style = str(item['line_style']).strip().lower()
        return elem
    def _absolute_bbox(self, bbox: Any, region_bbox: BoundingBox) -> Optional[BoundingBox]:
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        x1 += region_bbox.x1
        x2 += region_bbox.x1
        y1 += region_bbox.y1
        y2 += region_bbox.y1
        x1 = max(region_bbox.x1, min(region_bbox.x2, x1))
        x2 = max(region_bbox.x1, min(region_bbox.x2, x2))
        y1 = max(region_bbox.y1, min(region_bbox.y2, y1))
        y2 = max(region_bbox.y1, min(region_bbox.y2, y2))
        if x2 <= x1 or y2 <= y1:
            return None
        return BoundingBox(x1, y1, x2, y2)
    def _normalize_type(self, value: str) -> str:
        normalized = value.strip().lower().replace('-', '_').replace(' ', '_')
        return self.TYPE_ALIASES.get(normalized.replace('_', ' '), normalized)
    def _image_to_data_url(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    def _build_prompt(self, crop: Image.Image, region_bbox: BoundingBox) -> str:
        return f'Analyze this cropped missing/bad diagram region and return STRICT JSON only. If it contains clear structured diagram/UI elements, return {{"confidence": number, "elements": [{{"element_type": string, "bbox": [x1,y1,x2,y2], "confidence": number, "line_style": string|null, "reason": string}}]}}. bbox coordinates must be relative to this crop in pixels. Only include elements that are visually clear and structural; do not include photo-like/background content. Allowed element_type values: rectangle, rounded_rectangle, circle, ellipse, cylinder, arrow, connector, section_panel, diamond, triangle, hexagon, parallelogram, cloud, actor, line, text, icon, chart, logo, function_graph. If no reliable structured element exists, return {{"confidence": 0, "elements": []}}. Crop size={crop.width}x{crop.height}; original bbox={region_bbox.to_list()}.'
def refine_bad_regions(elements: List[ElementInfo], bad_regions: List[Dict], image_path: str, config: Dict=None) -> List[ElementInfo]:
    processor = RefinementProcessor(config)
    context = ProcessingContext(image_path=image_path, elements=elements.copy())
    context.intermediate_results['bad_regions'] = bad_regions
    result = processor.process(context)
    return result.elements
def evaluate_and_refine(elements: List[ElementInfo], image_path: str, eval_config: Dict=None, refine_config: Dict=None) -> Dict[str, Any]:
    from .metric_evaluator import MetricEvaluator
    evaluator = MetricEvaluator(eval_config)
    context = ProcessingContext(image_path=image_path, elements=elements.copy())
    eval_result = evaluator.process(context)
    result = {'evaluation': eval_result.metadata, 'refinement': None, 'elements': context.elements}
    bad_regions = eval_result.metadata.get('bad_regions', [])
    if eval_result.metadata.get('needs_refinement', False) and bad_regions:
        context.intermediate_results['bad_regions'] = bad_regions
        processor = RefinementProcessor(refine_config)
        refine_result = processor.process(context)
        result['refinement'] = refine_result.metadata
        result['elements'] = context.elements
    return result
def refine_from_rendered_comparison(elements: List[ElementInfo], original_path: str, rendered_path: str, config: Dict=None) -> Dict[str, Any]:
    from .metric_evaluator import compare_with_rendered
    import io
    import base64
    default_config = {'diff_threshold': 30, 'min_region_area': 300, 'expand_margin': 5, 'default_confidence': 0.4}
    cfg = {**default_config, **(config or {})}
    comparison = compare_with_rendered(original_path, rendered_path, {'diff_threshold': cfg['diff_threshold'], 'min_region_area': cfg['min_region_area'], 'merge_distance': 15})
    missing_regions = comparison.get('missing_regions', [])
    if not missing_regions:
        return {'elements': elements, 'comparison': comparison, 'new_count': 0}
    original = Image.open(original_path).convert('RGB')
    img_w, img_h = original.size
    new_elements = []
    start_id = max([e.id for e in elements], default=0) + 1
    margin = cfg['expand_margin']
    for i, region in enumerate(missing_regions):
        x1, y1, x2, y2 = region['bbox']
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(img_w, x2 + margin)
        y2 = min(img_h, y2 + margin)
        cropped = original.crop((x1, y1, x2, y2))
        buffer = io.BytesIO()
        cropped.save(buffer, format='PNG')
        b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        element = ElementInfo(id=start_id + i, element_type='picture', bbox=BoundingBox(x1, y1, x2, y2), score=cfg['default_confidence'], base64=b64_data, layer_level=LayerLevel.IMAGE.value, source_prompt='rendered_comparison_fallback', processing_notes=[f'渲染对比补救', f"差异强度: {region.get('diff_intensity', 0):.1f}", region.get('description', '')])
        new_elements.append(element)
    all_elements = elements + new_elements
    return {'elements': all_elements, 'comparison': comparison, 'new_count': len(new_elements)}
