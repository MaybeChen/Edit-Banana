import os
import sys
import cv2
import copy
import json
import numpy as np
import torch
import yaml
import warnings
import time
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import threading
from prompts.arrow import ARROW_PROMPT
from prompts.background import BACKGROUND_PROMPT
from prompts.shape import SHAPE_PROMPT
from prompts.image import IMAGE_PROMPT
warnings.filterwarnings('ignore', message='pkg_resources is deprecated as an API.*', category=UserWarning)
warnings.filterwarnings('ignore', message="User provided device_type of 'cuda', but CUDA is not available. Disabling", category=UserWarning)
warnings.filterwarnings('ignore', message='Importing from timm.models.layers is deprecated.*', category=FutureWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .base import BaseProcessor, ProcessingContext, ModelWrapper
from .data_types import ElementInfo, BoundingBox, ProcessingResult
from .vlm.prompt_planner import VLMPromptPlanner
class PromptGroup(Enum):
    IMAGE = 'image'
    ARROW = 'arrow'
    BASIC_SHAPE = 'shape'
    BACKGROUND = 'background'
@dataclass
class PromptGroupConfig:
    name: str
    prompts: List[str] = field(default_factory=list)
    score_threshold: float = 0.5
    min_area: int = 100
    priority: int = 1
    description: str = ''
    def add_prompt(self, prompt: str):
        if prompt not in self.prompts:
            self.prompts.append(prompt)
    def remove_prompt(self, prompt: str):
        if prompt in self.prompts:
            self.prompts.remove(prompt)
class ConfigLoader:
    _instance = None
    _config = None
    _config_path = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    @classmethod
    def get_config_path(cls) -> str:
        if cls._config_path is None:
            cls._config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'config.yaml')
        return cls._config_path
    @classmethod
    def load_config(cls, force_reload: bool=False) -> dict:
        if cls._config is None or force_reload:
            config_path = cls.get_config_path()
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    cls._config = yaml.safe_load(f)
            else:
                print(f'[ConfigLoader] Config not found: {config_path}, using defaults')
                cls._config = cls._get_default_config()
        return cls._config
    @classmethod
    def _get_default_config(cls) -> dict:
        return {'sam3': {'checkpoint_path': '', 'bpe_path': '', 'use_vlm_prompts': True, 'vlm_prompt_max_per_group': 6}, 'prompt_groups': {'image': {'name': '图片类', 'prompts': ['icon', 'picture', 'logo', 'chart'], 'score_threshold': 0.5, 'min_area': 100, 'priority': 2}, 'arrow': {'name': '箭头类', 'prompts': ['arrow', 'line', 'connector'], 'score_threshold': 0.4, 'min_area': 40, 'priority': 4}, 'shape': {'name': '基本图形', 'prompts': ['rectangle', 'rounded rectangle', 'diamond', 'ellipse'], 'score_threshold': 0.5, 'min_area': 150, 'priority': 3}, 'background': {'name': '背景容器', 'prompts': ['section_panel', 'title bar', 'container'], 'score_threshold': 0.25, 'min_area': 400, 'priority': 1}}, 'text_filter': {'blacklist': ['text', 'word', 'label'], 'keywords': ['text', 'word']}, 'deduplication': {'iou_threshold': 0.7, 'arrow_iou_threshold': 0.85}}
    @staticmethod
    def _merge_prompts(*prompt_lists: List[str]) -> List[str]:
        merged = []
        seen = set()
        for prompts in prompt_lists:
            for prompt in prompts or []:
                prompt = str(prompt).strip()
                if not prompt or prompt in seen:
                    continue
                seen.add(prompt)
                merged.append(prompt)
        return merged
    @classmethod
    def get_prompt_groups(cls) -> Dict[PromptGroup, PromptGroupConfig]:
        config = cls.load_config()
        prompt_groups_config = config.get('prompt_groups', {})
        result = {}
        key_to_enum = {'image': PromptGroup.IMAGE, 'arrow': PromptGroup.ARROW, 'shape': PromptGroup.BASIC_SHAPE, 'background': PromptGroup.BACKGROUND}
        prompt_mapping = {'image': IMAGE_PROMPT, 'arrow': ARROW_PROMPT, 'shape': SHAPE_PROMPT, 'background': BACKGROUND_PROMPT}
        for key, enum_val in key_to_enum.items():
            if key in prompt_groups_config:
                group_cfg = prompt_groups_config.get(key, {})
                default_prompts = list(prompt_mapping.get(key, []))
                configured_prompts = list(group_cfg.get('prompts') or [])
                extra_prompts = list(group_cfg.get('extra_prompts') or [])
                if group_cfg.get('replace_default_prompts'):
                    prompts = configured_prompts or default_prompts
                else:
                    prompts = cls._merge_prompts(default_prompts, configured_prompts, extra_prompts)
                result[enum_val] = PromptGroupConfig(name=group_cfg.get('name', key), prompts=prompts, score_threshold=group_cfg.get('score_threshold', 0.5), min_area=group_cfg.get('min_area', 100), priority=group_cfg.get('priority', 1), description=group_cfg.get('description', ''))
        return result
    @classmethod
    def get_text_filter(cls) -> dict:
        config = cls.load_config()
        return config.get('text_filter', {'blacklist': [], 'keywords': []})
    @classmethod
    def get_deduplication_config(cls) -> dict:
        config = cls.load_config()
        return config.get('deduplication', {'iou_threshold': 0.7, 'arrow_iou_threshold': 0.85})
    @classmethod
    def get_drawio_styles(cls) -> dict:
        config = cls.load_config()
        return config.get('drawio_styles', {})
    @classmethod
    def get_sam3_config(cls) -> dict:
        config = cls.load_config()
        return config.get('sam3', {})
    @classmethod
    def get_multimodal_config(cls) -> dict:
        config = cls.load_config()
        return config.get('multimodal', {})
class SAM3Model(ModelWrapper):
    def __init__(self, checkpoint_path: str, bpe_path: str, device: str=None):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        requested_device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if str(requested_device).startswith('cuda') and (not torch.cuda.is_available()):
            print('[SAM3Model] Warning: CUDA requested but PyTorch has no CUDA support; falling back to CPU')
            requested_device = 'cpu'
        self.device = requested_device
        self._processor = None
        self._state_cache = OrderedDict()
        self._max_cache_size = 3
        self._cache_lock = threading.Lock()
    def load(self):
        if self._is_loaded:
            return
        print(f'[SAM3Model] 加载模型中... (设备: {self.device})')
        from sam3_imports import import_sam3_image_components
        build_sam3_image_model, Sam3Processor = import_sam3_image_components()
        with self._redirect_cuda_allocations_when_cpu_only():
            self._model = build_sam3_image_model(bpe_path=self.bpe_path, checkpoint_path=self.checkpoint_path, load_from_HF=False, device=self.device)
            self._install_cpu_dtype_compatibility_hooks()
            self._processor = Sam3Processor(self._model, device=self.device)
        self._is_loaded = True
        print('[SAM3Model] 模型加载完成！')
    def _install_cpu_dtype_compatibility_hooks(self):
        if self.device != 'cpu' or self._model is None:
            return
        def match_linear_input_dtype(module, inputs):
            if not inputs:
                return inputs
            first_arg = inputs[0]
            if isinstance(first_arg, torch.Tensor) and first_arg.is_floating_point() and (first_arg.dtype != module.weight.dtype):
                return (first_arg.to(dtype=module.weight.dtype),) + inputs[1:]
            return inputs
        self._cpu_dtype_hook_handles = []
        for module in self._model.modules():
            if isinstance(module, torch.nn.Linear):
                self._cpu_dtype_hook_handles.append(module.register_forward_pre_hook(match_linear_input_dtype))
    @contextmanager
    def _redirect_cuda_allocations_when_cpu_only(self):
        should_redirect = self.device == 'cpu' and (not torch.cuda.is_available())
        if not should_redirect:
            yield
            return
        factory_names = ('arange', 'empty', 'full', 'linspace', 'ones', 'rand', 'randn', 'tensor', 'zeros')
        originals = {name: getattr(torch, name) for name in factory_names}
        original_pin_memory = torch.Tensor.pin_memory
        def make_cpu_fallback(original_func):
            def cpu_fallback(*args, **kwargs):
                device = kwargs.get('device')
                if device is not None and str(device).startswith('cuda'):
                    kwargs['device'] = 'cpu'
                return original_func(*args, **kwargs)
            return cpu_fallback
        for name, original in originals.items():
            setattr(torch, name, make_cpu_fallback(original))
        torch.Tensor.pin_memory = lambda tensor, *args, **kwargs: tensor
        try:
            yield
        finally:
            for name, original in originals.items():
                setattr(torch, name, original)
            torch.Tensor.pin_memory = original_pin_memory
    def predict(self, image_path: str, prompts: List[str], score_threshold: float=0.5, min_area: int=100) -> List[Dict[str, Any]]:
        if not self._is_loaded:
            self.load()
        predict_start = time.time()
        print(f'[SAM3Model] 准备图像状态: {image_path} (prompts={len(prompts)}, device={self.device})', flush=True)
        with self._redirect_cuda_allocations_when_cpu_only():
            state, pil_image = self._get_image_state(image_path)
        print(f'[SAM3Model] 图像状态完成: size={pil_image.size}, elapsed={time.time() - predict_start:.2f}s', flush=True)
        results = []
        for prompt_idx, prompt in enumerate(prompts, start=1):
            prompt_start = time.time()
            print(f'[SAM3Model]   prompt {prompt_idx}/{len(prompts)}: {prompt!r} 开始', flush=True)
            with self._redirect_cuda_allocations_when_cpu_only():
                self._processor.reset_all_prompts(state)
                result_state = self._processor.set_text_prompt(prompt=prompt, state=state)
            masks = result_state.get('masks', [])
            boxes = result_state.get('boxes', [])
            scores = result_state.get('scores', [])
            num_masks = masks.shape[0] if isinstance(masks, torch.Tensor) and masks.dim() > 0 else len(masks)
            kept_before = len(results)
            for i in range(num_masks):
                score = scores[i]
                score_val = score.item() if hasattr(score, 'item') else float(score)
                if score_val < score_threshold:
                    continue
                box = boxes[i]
                bbox = box.cpu().numpy().tolist() if isinstance(box, torch.Tensor) else box
                bbox = [int(coord) for coord in bbox]
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area < min_area:
                    continue
                mask = masks[i]
                binary_mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
                if binary_mask.ndim > 2:
                    binary_mask = binary_mask.squeeze()
                binary_mask = (binary_mask > 0.5).astype(np.uint8) * 255
                polygon = self._extract_polygon(binary_mask, min_area)
                if polygon:
                    results.append({'prompt': prompt, 'bbox': bbox, 'score': score_val, 'mask': binary_mask, 'polygon': polygon, 'area': area})
            print(f'[SAM3Model]   prompt {prompt_idx}/{len(prompts)}: {prompt!r} 完成 masks={num_masks}, kept={len(results) - kept_before}, elapsed={time.time() - prompt_start:.2f}s', flush=True)
        return results
    def _get_image_state(self, image_path: str):
        with self._cache_lock:
            if image_path in self._state_cache:
                self._state_cache.move_to_end(image_path)
                cache_item = self._state_cache[image_path]
                return (cache_item['state'], cache_item['pil_image'])
        pil_image = Image.open(image_path).convert('RGB')
        state = self._processor.set_image(pil_image)
        cache_item = {'state': state, 'pil_image': pil_image}
        with self._cache_lock:
            if image_path in self._state_cache:
                self._state_cache.move_to_end(image_path)
                return (self._state_cache[image_path]['state'], self._state_cache[image_path]['pil_image'])
            self._state_cache[image_path] = cache_item
            if len(self._state_cache) > self._max_cache_size:
                self._state_cache.popitem(last=False)
        return (state, pil_image)
    def _extract_polygon(self, binary_mask: np.ndarray, min_area: int=100, epsilon_factor: float=0.02) -> List[List[int]]:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            epsilon = epsilon_factor * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            return approx.reshape(-1, 2).tolist()
        return []
    def clear_cache(self):
        with self._cache_lock:
            self._state_cache.clear()
class Sam3InfoExtractor(BaseProcessor):
    def __init__(self, config=None, checkpoint_path: str=None, bpe_path: str=None):
        super().__init__(config)
        self.prompt_groups = ConfigLoader.get_prompt_groups()
        self.text_filter = ConfigLoader.get_text_filter()
        self.dedup_config = ConfigLoader.get_deduplication_config()
        sam3_config = ConfigLoader.get_sam3_config()
        self._checkpoint_path = checkpoint_path or sam3_config.get('checkpoint_path', '')
        self._bpe_path = bpe_path or sam3_config.get('bpe_path', '')
        self._sam3_model: Optional[SAM3Model] = None
        self._current_image_path: Optional[str] = None
    def reload_config(self):
        ConfigLoader.load_config(force_reload=True)
        self.prompt_groups = ConfigLoader.get_prompt_groups()
        self.text_filter = ConfigLoader.get_text_filter()
        self.dedup_config = ConfigLoader.get_deduplication_config()
        self._log('Config reloaded')
    def load_model(self):
        if self._sam3_model is None:
            sam3_config = ConfigLoader.get_sam3_config()
            device = sam3_config.get('device')
            self._sam3_model = SAM3Model(checkpoint_path=self._checkpoint_path, bpe_path=self._bpe_path, device=device)
        if not self._sam3_model.is_loaded:
            self._sam3_model.load()
    def process(self, context: ProcessingContext) -> ProcessingResult:
        self._log(f'开始处理: {context.image_path}')
        self._current_image_path = context.image_path
        self.prompt_groups = self._build_prompt_groups_for_image(context)
        self.load_model()
        pil_image = Image.open(context.image_path)
        context.canvas_width, context.canvas_height = pil_image.size
        all_elements = []
        group_stats = {}
        process_order = [PromptGroup.BACKGROUND, PromptGroup.BASIC_SHAPE, PromptGroup.IMAGE, PromptGroup.ARROW]
        for group_type in process_order:
            if group_type not in self.prompt_groups:
                continue
            group_config = self.prompt_groups[group_type]
            if not group_config.prompts:
                continue
            self._log(f'  处理组 [{group_config.name}]: {len(group_config.prompts)}个提示词')
            raw_results = self._sam3_model.predict(context.image_path, group_config.prompts, score_threshold=group_config.score_threshold, min_area=group_config.min_area)
            raw_results = self._filter_text_elements(raw_results)
            elements = self._convert_to_elements(raw_results, start_id=len(all_elements), source_group=group_type.value, group_priority=group_config.priority)
            all_elements.extend(elements)
            group_stats[group_config.name] = len(elements)
            self._log(f'    提取到 {len(elements)} 个元素')
        all_elements = self._deduplicate_cross_groups(all_elements)
        all_elements = self._filter_contained_elements(all_elements)
        context.elements = all_elements
        result = ProcessingResult(success=True, elements=all_elements, canvas_width=context.canvas_width, canvas_height=context.canvas_height, metadata={'group_stats': group_stats, 'total_before_dedup': sum(group_stats.values()), 'total_after_dedup': len(all_elements), 'groups_processed': list(group_stats.keys()), 'vlm_prompts': context.intermediate_results.get('vlm_prompts', {})})
        self._log(f'Done: {len(all_elements)} elements (before dedup: {sum(group_stats.values())})')
        return result
    def _build_prompt_groups_for_image(self, context: ProcessingContext) -> Dict[PromptGroup, PromptGroupConfig]:
        prompt_groups = copy.deepcopy(ConfigLoader.get_prompt_groups())
        sam3_config = ConfigLoader.get_sam3_config()
        multimodal_config = ConfigLoader.get_multimodal_config()
        use_vlm_prompts = bool(sam3_config.get('use_vlm_prompts', True))
        prompt_planning_enabled = bool(use_vlm_prompts and multimodal_config.get('enabled', False) and (multimodal_config.get('use_for') or {}).get('prompt_planning', False))
        max_per_group = int(sam3_config.get('vlm_prompt_max_per_group', 6) or 6)
        vlm_record = {'enabled': prompt_planning_enabled, 'configured': use_vlm_prompts, 'image_path': context.image_path, 'max_per_group': max_per_group, 'dynamic_prompts': {key.value: [] for key in PromptGroup}, 'merged_prompts': {}, 'error': None}
        if use_vlm_prompts and (not prompt_planning_enabled):
            vlm_record['error'] = 'VLM prompt planning disabled because multimodal.enabled/use_for.prompt_planning is not enabled'
        if prompt_planning_enabled:
            try:
                planner = VLMPromptPlanner(multimodal_config)
                dynamic_prompts = planner.plan(context.image_path, max_per_group=max_per_group)
                mapping = {'image': PromptGroup.IMAGE, 'shape': PromptGroup.BASIC_SHAPE, 'arrow': PromptGroup.ARROW, 'background': PromptGroup.BACKGROUND}
                for key, group_type in mapping.items():
                    prompts = dynamic_prompts.get(key, [])[:max_per_group]
                    vlm_record['dynamic_prompts'][group_type.value] = prompts
                    if group_type in prompt_groups:
                        prompt_groups[group_type].prompts = ConfigLoader._merge_prompts(prompt_groups[group_type].prompts, prompts)
                self._log(f'VLM动态提示词已注入: {sum((len(v) for v in dynamic_prompts.values()))}个')
            except Exception as exc:
                vlm_record['error'] = str(exc)
                self._log(f'VLM动态提示词跳过: {exc}')
        vlm_record['merged_prompts'] = {group_type.value: group_config.prompts for group_type, group_config in prompt_groups.items()}
        context.intermediate_results['vlm_prompts'] = vlm_record
        self._save_vlm_prompts(context, vlm_record)
        return prompt_groups
    def _save_vlm_prompts(self, context: ProcessingContext, vlm_record: Dict[str, Any]):
        output_dir = context.output_dir or './output'
        self._ensure_output_dir(output_dir)
        output_path = os.path.join(output_dir, 'vlm_prompts.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vlm_record, f, indent=2, ensure_ascii=False)
        self._log(f'VLM提示词已保存: {output_path}')
    def extract_by_group(self, context: ProcessingContext, group_type: PromptGroup) -> ProcessingResult:
        self._log(f'Extract group [{group_type.value}]: {context.image_path}')
        self.load_model()
        pil_image = Image.open(context.image_path)
        context.canvas_width, context.canvas_height = pil_image.size
        if group_type not in self.prompt_groups:
            return ProcessingResult(success=False, error_message=f'未知的组类型: {group_type}')
        group_config = self.prompt_groups[group_type]
        raw_results = self._sam3_model.predict(context.image_path, group_config.prompts, score_threshold=group_config.score_threshold, min_area=group_config.min_area)
        raw_results = self._filter_text_elements(raw_results)
        elements = self._convert_to_elements(raw_results, start_id=0, source_group=group_type.value, group_priority=group_config.priority)
        elements = self._deduplicate_within_group(elements)
        context.elements = elements
        return ProcessingResult(success=True, elements=elements, canvas_width=context.canvas_width, canvas_height=context.canvas_height, metadata={'group': group_type.value, 'prompts_used': group_config.prompts, 'element_count': len(elements)})
    def extract_with_custom_prompts(self, context: ProcessingContext, prompts: List[str], score_threshold: float=0.5, min_area: int=100) -> ProcessingResult:
        self._log(f'自定义提取: {prompts}')
        self.load_model()
        pil_image = Image.open(context.image_path)
        context.canvas_width, context.canvas_height = pil_image.size
        raw_results = self._sam3_model.predict(context.image_path, prompts, score_threshold=score_threshold, min_area=min_area)
        elements = self._convert_to_elements(raw_results, start_id=0, source_group='custom', group_priority=2)
        elements = self._deduplicate_within_group(elements)
        context.elements = elements
        return ProcessingResult(success=True, elements=elements, canvas_width=context.canvas_width, canvas_height=context.canvas_height, metadata={'prompts_used': prompts, 'element_count': len(elements)})
    def _filter_text_elements(self, raw_results: List[Dict]) -> List[Dict]:
        blacklist = set(self.text_filter.get('blacklist', []))
        keywords = self.text_filter.get('keywords', [])
        filtered = []
        for item in raw_results:
            prompt = item['prompt'].lower()
            if prompt in blacklist:
                continue
            is_text = False
            for kw in keywords:
                if kw in prompt:
                    is_text = True
                    break
            if not is_text:
                filtered.append(item)
        return filtered
    @staticmethod
    def _normalize_prompt_element_type(prompt: str) -> tuple[str, Optional[str]]:
        text = (prompt or '').lower().strip()
        line_style = 'dashed' if any((token in text for token in ('dashed', 'dotted', 'dot'))) else None
        if 'arrow' in text:
            return ('arrow', line_style)
        if 'connector' in text or 'line' in text:
            return ('connector', line_style)
        if 'cylinder' in text or 'database' in text:
            return ('cylinder', None)
        if 'circle' in text:
            return ('circle', None)
        if 'ellipse' in text:
            return ('ellipse', None)
        if 'rounded rectangle' in text or 'card' in text or 'box' in text:
            return ('rounded rectangle', None)
        if 'rectangle' in text:
            return ('rectangle', None)
        if 'diamond' in text:
            return ('diamond', None)
        if 'triangle' in text:
            return ('triangle', None)
        if 'hexagon' in text:
            return ('hexagon', None)
        if 'panel' in text or 'container' in text or 'background' in text or ('frame' in text) or ('boundary' in text):
            return ('container', None)
        if 'icon' in text or 'symbol' in text or 'logo' in text or ('chart' in text) or ('picture' in text):
            return ('icon', None)
        return (text, line_style)
    def _convert_to_elements(self, raw_results: List[Dict], start_id: int=0, source_group: str='', group_priority: int=1) -> List[ElementInfo]:
        elements = []
        for i, item in enumerate(raw_results):
            bbox = BoundingBox.from_list(item['bbox'])
            element_type, line_style = self._normalize_prompt_element_type(item['prompt'])
            element = ElementInfo(id=start_id + i, element_type=element_type, bbox=bbox, score=item['score'], polygon=item['polygon'], mask=item['mask'], source_prompt=item['prompt'])
            if line_style:
                element.line_style = line_style
            element.processing_notes.append(f"source_prompt={item['prompt']}")
            element.processing_notes.append(f'source_group={source_group}')
            element.processing_notes.append(f"area={item.get('area', bbox.area)}")
            element._group_priority = group_priority
            element._source_group = source_group
            elements.append(element)
        return elements
    def _deduplicate_within_group(self, elements: List[ElementInfo], iou_threshold: float=None) -> List[ElementInfo]:
        if not elements:
            return elements
        if iou_threshold is None:
            iou_threshold = self.dedup_config.get('iou_threshold', 0.7) + 0.15
        sorted_elements = sorted(elements, key=lambda x: x.score, reverse=True)
        keep = []
        dropped = set()
        for i, elem_i in enumerate(sorted_elements):
            if i in dropped:
                continue
            keep.append(elem_i)
            for j in range(i + 1, len(sorted_elements)):
                if j in dropped:
                    continue
                iou = self._calculate_iou(elem_i.bbox.to_list(), sorted_elements[j].bbox.to_list())
                if iou > iou_threshold:
                    dropped.add(j)
        for i, elem in enumerate(keep):
            elem.id = i
        return keep
    def _analyze_region_complexity(self, image_path: str, bbox: List[int]) -> dict:
        try:
            cv2_image = cv2.imread(image_path)
            x1, y1, x2, y2 = bbox
            roi = cv2_image[y1:y2, x1:x2]
            if roi.size == 0:
                return {'classification': 'unknown', 'is_complex': False}
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            std_dev = np.std(gray)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.count_nonzero(edges) / edges.size
            h, w = roi.shape[:2]
            border_size = max(3, min(10, w // 20, h // 20))
            top_edge = gray[:border_size, :].flatten()
            bottom_edge = gray[-border_size:, :].flatten()
            left_edge = gray[:, :border_size].flatten()
            right_edge = gray[:, -border_size:].flatten()
            border_pixels = np.concatenate([top_edge, bottom_edge, left_edge, right_edge])
            inner_margin = border_size + 2
            if w > 2 * inner_margin and h > 2 * inner_margin:
                inner = gray[inner_margin:-inner_margin, inner_margin:-inner_margin].flatten()
            else:
                inner = gray.flatten()
            border_mean = np.mean(border_pixels)
            inner_mean = np.mean(inner)
            border_contrast = abs(border_mean - inner_mean)
            has_clear_border = border_contrast > 25 and edge_ratio > 0.03
            is_complex = laplacian_var > 800 or std_dev > 55
            if is_complex and (not has_clear_border):
                classification = 'image_only'
            elif has_clear_border and (not is_complex):
                classification = 'shape_only'
            elif has_clear_border and is_complex:
                classification = 'shape_with_content'
            else:
                classification = 'image_fallback'
            return {'laplacian_var': laplacian_var, 'std_dev': std_dev, 'edge_ratio': edge_ratio, 'is_complex': is_complex, 'has_clear_border': has_clear_border, 'border_contrast': border_contrast, 'classification': classification}
        except Exception as e:
            return {'classification': 'unknown', 'is_complex': False, 'error': str(e)}
    def _deduplicate_cross_groups(self, elements: List[ElementInfo]) -> List[ElementInfo]:
        if not elements:
            return elements
        iou_threshold = self.dedup_config.get('iou_threshold', 0.7)
        arrow_iou_threshold = self.dedup_config.get('arrow_iou_threshold', 0.85)
        shape_image_iou_threshold = self.dedup_config.get('shape_image_iou_threshold', 0.6)
        sorted_elements = sorted(elements, key=lambda x: (getattr(x, '_group_priority', 1), x.score), reverse=True)
        keep = []
        dropped = set()
        for i, elem_i in enumerate(sorted_elements):
            if i in dropped:
                continue
            keep.append(elem_i)
            for j in range(i + 1, len(sorted_elements)):
                if j in dropped:
                    continue
                elem_j = sorted_elements[j]
                group_i = getattr(elem_i, '_source_group', '')
                group_j = getattr(elem_j, '_source_group', '')
                effective_threshold = iou_threshold
                if group_i == 'arrow' or group_j == 'arrow':
                    effective_threshold = arrow_iou_threshold
                iou = self._calculate_iou(elem_i.bbox.to_list(), elem_j.bbox.to_list())
                if iou < 0.1:
                    continue
                if iou > shape_image_iou_threshold:
                    is_shape_image_overlap = group_i == 'shape' and group_j == 'image' or (group_i == 'image' and group_j == 'shape')
                    if is_shape_image_overlap:
                        if group_i == 'shape':
                            if elem_i in keep:
                                keep.remove(elem_i)
                            if elem_j not in keep:
                                keep.append(elem_j)
                            dropped.add(j)
                            break
                        else:
                            dropped.add(j)
                        continue
                if iou > effective_threshold:
                    dropped.add(j)
        for i, elem in enumerate(keep):
            elem.id = i
        return keep
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    def _filter_contained_elements(self, elements: List[ElementInfo]) -> List[ElementInfo]:
        IMAGE_TYPES = {'icon', 'picture', 'logo', 'chart', 'function_graph', 'image'}
        if not elements:
            return elements
        to_remove = set()
        for i, elem_i in enumerate(elements):
            if i in to_remove:
                continue
            bbox_i = elem_i.bbox.to_list()
            area_i = (bbox_i[2] - bbox_i[0]) * (bbox_i[3] - bbox_i[1])
            type_i = elem_i.element_type.lower()
            for j, elem_j in enumerate(elements):
                if i == j or j in to_remove:
                    continue
                bbox_j = elem_j.bbox.to_list()
                area_j = (bbox_j[2] - bbox_j[0]) * (bbox_j[3] - bbox_j[1])
                type_j = elem_j.element_type.lower()
                if area_i > area_j:
                    containment = self._calculate_containment(bbox_i, bbox_j)
                    if containment > 0.85 and type_i in IMAGE_TYPES:
                        to_remove.add(j)
                        self._log(f'Filter {elem_j.id}({type_j}): contained by {elem_i.id}({type_i}) {containment:.0%}')
                elif area_j > area_i:
                    containment = self._calculate_containment(bbox_j, bbox_i)
                    if containment > 0.85 and type_j in IMAGE_TYPES:
                        to_remove.add(i)
                        self._log(f'Filter {elem_i.id}({type_i}): contained by {elem_j.id}({type_j}) {containment:.0%}')
                        break
        result = [e for i, e in enumerate(elements) if i not in to_remove]
        for i, elem in enumerate(result):
            elem.id = i
        if to_remove:
            self._log(f'完全包含过滤: 移除了 {len(to_remove)} 个被大图包含的小元素')
        return result
    def _calculate_containment(self, box_outer: List[int], box_inner: List[int]) -> float:
        x1 = max(box_outer[0], box_inner[0])
        y1 = max(box_outer[1], box_inner[1])
        x2 = min(box_outer[2], box_inner[2])
        y2 = min(box_outer[3], box_inner[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter_area = (x2 - x1) * (y2 - y1)
        inner_area = (box_inner[2] - box_inner[0]) * (box_inner[3] - box_inner[1])
        return inter_area / inner_area if inner_area > 0 else 0.0
    def save_visualization(self, context: ProcessingContext, output_path: str):
        cv2_image = cv2.imread(context.image_path)
        GROUP_COLORS = {'image': (0, 255, 0), 'arrow': (255, 0, 0), 'shape': (0, 0, 255), 'background': (255, 255, 0), 'custom': (128, 0, 128)}
        DEFAULT_COLOR = (128, 128, 128)
        image = cv2_image.copy()
        overlay = cv2_image.copy()
        for elem in context.elements:
            group = getattr(elem, '_source_group', '')
            color = GROUP_COLORS.get(group, DEFAULT_COLOR)
            points = np.array(elem.polygon, dtype=np.int32)
            if points.size > 0:
                cv2.fillPoly(overlay, [points], color)
            x1, y1, x2, y2 = elem.bbox.to_list()
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f'{elem.id}:{elem.element_type}'
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(output_path, result)
        self._log(f'可视化已保存: {output_path}')
    def save_metadata(self, context: ProcessingContext, output_path: str):
        import json
        metadata = {'image_path': context.image_path, 'image_size': {'width': context.canvas_width, 'height': context.canvas_height}, 'total_elements': len(context.elements), 'by_group': {}, 'by_type': {}, 'elements': []}
        for elem in context.elements:
            group = getattr(elem, '_source_group', 'unknown')
            if group not in metadata['by_group']:
                metadata['by_group'][group] = []
            elem_type = elem.element_type
            if elem_type not in metadata['by_type']:
                metadata['by_type'][elem_type] = []
            elem_data = elem.to_dict()
            elem_data['source_group'] = group
            metadata['by_group'][group].append(elem_data)
            metadata['by_type'][elem_type].append(elem_data)
            metadata['elements'].append(elem_data)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        self._log(f'Metadata saved: {output_path}')
    def get_all_prompts(self) -> Dict[str, List[str]]:
        return {group.value: config.prompts.copy() for group, config in self.prompt_groups.items()}
    def get_group_config(self, group_type: PromptGroup) -> Optional[PromptGroupConfig]:
        return self.prompt_groups.get(group_type)
    def add_prompts_to_group(self, group_type: PromptGroup, prompts: List[str]):
        if group_type in self.prompt_groups:
            for p in prompts:
                self.prompt_groups[group_type].add_prompt(p)
    def remove_prompts_from_group(self, group_type: PromptGroup, prompts: List[str]):
        if group_type in self.prompt_groups:
            for p in prompts:
                self.prompt_groups[group_type].remove_prompt(p)
    def set_group_threshold(self, group_type: PromptGroup, score_threshold: float=None, min_area: int=None):
        if group_type in self.prompt_groups:
            if score_threshold is not None:
                self.prompt_groups[group_type].score_threshold = score_threshold
            if min_area is not None:
                self.prompt_groups[group_type].min_area = min_area
    def print_prompt_groups(self):
        print('\n' + '=' * 60)
        print('当前SAM3提示词词库配置 (从 config.yaml 加载)')
        print('=' * 60)
        for group_type, config in self.prompt_groups.items():
            print(f'\n[{config.name}] ({group_type.value})')
            print(f'  置信度阈值: {config.score_threshold}')
            print(f'  最小面积: {config.min_area}')
            print(f'  优先级: {config.priority}')
            print(f'  提示词 ({len(config.prompts)}个):')
            for p in config.prompts:
                print(f'    - {p}')
        print('\n' + '=' * 60)
        print(f'配置文件路径: {ConfigLoader.get_config_path()}')
        print('=' * 60)
def extract_elements(image_path: str, groups: List[PromptGroup]=None) -> ProcessingResult:
    extractor = Sam3InfoExtractor()
    context = ProcessingContext(image_path=image_path)
    if groups:
        all_elements = []
        for group in groups:
            result = extractor.extract_by_group(context, group)
            all_elements.extend(result.elements)
        for i, elem in enumerate(all_elements):
            elem.id = i
        return ProcessingResult(success=True, elements=all_elements, canvas_width=context.canvas_width, canvas_height=context.canvas_height)
    return extractor.process(context)
def extract_with_prompts(image_path: str, prompts: List[str], score_threshold: float=0.5) -> ProcessingResult:
    extractor = Sam3InfoExtractor()
    context = ProcessingContext(image_path=image_path)
    return extractor.extract_with_custom_prompts(context, prompts, score_threshold=score_threshold)
