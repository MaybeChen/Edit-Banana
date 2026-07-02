"""
SAM3 info extractor: extract diagram elements (shapes, arrows, icons, background) from images.
Prompt groups and thresholds are loaded from config.yaml.
"""

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
from .sam3_config import PromptGroup, PromptGroupConfig, ConfigLoader

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*", category=UserWarning)
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling", category=UserWarning)
warnings.filterwarnings("ignore", message="Importing from timm.models.layers is deprecated.*", category=FutureWarning)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import BaseProcessor, ProcessingContext, ModelWrapper
from .data_types import ElementInfo, BoundingBox, ProcessingResult
from .vlm.prompt_planner import VLMPromptPlanner
from .sam3_extractor_mixin import Sam3ExtractorMixin




# ======================== SAM3模型封装 ========================
class SAM3Model(ModelWrapper):
    """SAM3模型封装"""
    
    def __init__(self, checkpoint_path: str, bpe_path: str, device: str = None):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.bpe_path = bpe_path
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
            print("[SAM3Model] Warning: CUDA requested but PyTorch has no CUDA support; falling back to CPU")
            requested_device = "cpu"
        self.device = requested_device
        self._processor = None
        
        # 图像状态缓存
        self._state_cache = OrderedDict()
        self._max_cache_size = 3
        self._cache_lock = threading.Lock()
    
    def load(self):
        """加载SAM3模型"""
        if self._is_loaded:
            return
            
        print(f"[SAM3Model] 加载模型中... (设备: {self.device})")
        
        from sam3_imports import import_sam3_image_components

        build_sam3_image_model, Sam3Processor = import_sam3_image_components()

        with self._redirect_cuda_allocations_when_cpu_only():
            self._model = build_sam3_image_model(
                bpe_path=self.bpe_path,
                checkpoint_path=self.checkpoint_path,
                load_from_HF=False,
                device=self.device
            )
            self._install_cpu_dtype_compatibility_hooks()
            self._processor = Sam3Processor(self._model, device=self.device)
        self._is_loaded = True
        
        print("[SAM3Model] 模型加载完成！")

    def _install_cpu_dtype_compatibility_hooks(self):
        """Keep CPU linear inputs aligned with layer weights when SAM3 emits bf16 tensors."""
        if self.device != "cpu" or self._model is None:
            return

        def match_linear_input_dtype(module, inputs):
            if not inputs:
                return inputs
            first_arg = inputs[0]
            if (
                isinstance(first_arg, torch.Tensor)
                and first_arg.is_floating_point()
                and first_arg.dtype != module.weight.dtype
            ):
                return (first_arg.to(dtype=module.weight.dtype),) + inputs[1:]
            return inputs

        self._cpu_dtype_hook_handles = []
        for module in self._model.modules():
            if isinstance(module, torch.nn.Linear):
                self._cpu_dtype_hook_handles.append(
                    module.register_forward_pre_hook(match_linear_input_dtype)
                )

    @contextmanager
    def _redirect_cuda_allocations_when_cpu_only(self):
        """Redirect SAM3 hard-coded CUDA tensor allocations to CPU for CPU-only PyTorch."""
        should_redirect = self.device == "cpu" and not torch.cuda.is_available()
        if not should_redirect:
            yield
            return

        factory_names = ("arange", "empty", "full", "linspace", "ones", "rand", "randn", "tensor", "zeros")
        originals = {name: getattr(torch, name) for name in factory_names}
        original_pin_memory = torch.Tensor.pin_memory

        def make_cpu_fallback(original_func):
            def cpu_fallback(*args, **kwargs):
                device = kwargs.get("device")
                if device is not None and str(device).startswith("cuda"):
                    kwargs["device"] = "cpu"
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
    
    def predict(self, image_path: str, prompts: List[str], 
                score_threshold: float = 0.5,
                min_area: int = 100) -> List[Dict[str, Any]]:
        """
        SAM3推理
        
        Args:
            image_path: 图片路径
            prompts: 提示词列表
            score_threshold: 置信度阈值
            min_area: 最小面积阈值
            
        Returns:
            元素列表
        """
        if not self._is_loaded:
            self.load()

        predict_start = time.time()
        print(
            f"[SAM3Model] 准备图像状态: {image_path} "
            f"(prompts={len(prompts)}, device={self.device})",
            flush=True,
        )
        with self._redirect_cuda_allocations_when_cpu_only():
            state, pil_image = self._get_image_state(image_path)
        print(
            f"[SAM3Model] 图像状态完成: size={pil_image.size}, elapsed={time.time() - predict_start:.2f}s",
            flush=True,
        )

        results = []
        for prompt_idx, prompt in enumerate(prompts, start=1):
            prompt_start = time.time()
            print(
                f"[SAM3Model]   prompt {prompt_idx}/{len(prompts)}: {prompt!r} 开始",
                flush=True,
            )
            with self._redirect_cuda_allocations_when_cpu_only():
                self._processor.reset_all_prompts(state)
                result_state = self._processor.set_text_prompt(prompt=prompt, state=state)
            
            masks = result_state.get("masks", [])
            boxes = result_state.get("boxes", [])
            scores = result_state.get("scores", [])
            
            num_masks = masks.shape[0] if (isinstance(masks, torch.Tensor) and masks.dim() > 0) else len(masks)
            kept_before = len(results)
            
            for i in range(num_masks):
                score = scores[i]
                score_val = score.item() if hasattr(score, 'item') else float(score)
                
                if score_val < score_threshold:
                    continue
                
                # 提取bbox
                box = boxes[i]
                bbox = box.cpu().numpy().tolist() if isinstance(box, torch.Tensor) else box
                bbox = [int(coord) for coord in bbox]
                
                # 检查面积
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area < min_area:
                    continue
                
                # 提取mask
                mask = masks[i]
                binary_mask = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
                if binary_mask.ndim > 2:
                    binary_mask = binary_mask.squeeze()
                binary_mask = (binary_mask > 0.5).astype(np.uint8) * 255
                
                # 提取polygon
                polygon = self._extract_polygon(binary_mask, min_area)
                
                if polygon:
                    results.append({
                        'prompt': prompt,
                        'bbox': bbox,
                        'score': score_val,
                        'mask': binary_mask,
                        'polygon': polygon,
                        'area': area
                    })
            print(
                f"[SAM3Model]   prompt {prompt_idx}/{len(prompts)}: {prompt!r} "
                f"完成 masks={num_masks}, kept={len(results) - kept_before}, "
                f"elapsed={time.time() - prompt_start:.2f}s",
                flush=True,
            )
        
        return results
    
    def _get_image_state(self, image_path: str):
        """获取或创建图像状态（LRU缓存）"""
        with self._cache_lock:
            if image_path in self._state_cache:
                self._state_cache.move_to_end(image_path)
                cache_item = self._state_cache[image_path]
                return cache_item["state"], cache_item["pil_image"]
        
        pil_image = Image.open(image_path).convert("RGB")
        state = self._processor.set_image(pil_image)
        
        cache_item = {"state": state, "pil_image": pil_image}
        
        with self._cache_lock:
            if image_path in self._state_cache:
                self._state_cache.move_to_end(image_path)
                return self._state_cache[image_path]["state"], self._state_cache[image_path]["pil_image"]
            
            self._state_cache[image_path] = cache_item
            
            if len(self._state_cache) > self._max_cache_size:
                self._state_cache.popitem(last=False)
        
        return state, pil_image
    
    def _extract_polygon(self, binary_mask: np.ndarray, 
                         min_area: int = 100, 
                         epsilon_factor: float = 0.02) -> List[List[int]]:
        """从mask提取多边形轮廓"""
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
        """清空图像缓存"""
        with self._cache_lock:
            self._state_cache.clear()


# ======================== SAM3信息提取器 ========================
class Sam3InfoExtractor(Sam3ExtractorMixin, BaseProcessor):
    """Extract diagram elements via SAM3; prompt groups and thresholds from config."""

    def __init__(self, config=None, checkpoint_path: str = None, bpe_path: str = None):
        super().__init__(config)
        
        # 从配置文件加载词组（不再硬编码）
        self.prompt_groups = ConfigLoader.get_prompt_groups()
        self.text_filter = ConfigLoader.get_text_filter()
        self.dedup_config = ConfigLoader.get_deduplication_config()
        
        # 加载SAM3模型配置
        sam3_config = ConfigLoader.get_sam3_config()
        self._checkpoint_path = checkpoint_path or sam3_config.get('checkpoint_path', '')
        self._bpe_path = bpe_path or sam3_config.get('bpe_path', '')
        
        self._sam3_model: Optional[SAM3Model] = None
        self._current_image_path: Optional[str] = None

    def reload_config(self):
        """Reload config from disk."""
        ConfigLoader.load_config(force_reload=True)
        self.prompt_groups = ConfigLoader.get_prompt_groups()
        self.text_filter = ConfigLoader.get_text_filter()
        self.dedup_config = ConfigLoader.get_deduplication_config()
        self._log("Config reloaded")

    def load_model(self):
        """Load SAM3 model."""
        if self._sam3_model is None:
            sam3_config = ConfigLoader.get_sam3_config()
            device = sam3_config.get("device")  # e.g. "cpu" or "cuda", None = auto
            self._sam3_model = SAM3Model(
                checkpoint_path=self._checkpoint_path,
                bpe_path=self._bpe_path,
                device=device
            )
        if not self._sam3_model.is_loaded:
            self._sam3_model.load()
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """
        处理入口 - 分组提取图片中的所有元素
        
        Args:
            context: 处理上下文，需要包含 image_path
            
        Returns:
            ProcessingResult: 包含所有提取的ElementInfo
        """
        self._log(f"开始处理: {context.image_path}")
        
        # 保存当前图像路径（供去重分析使用）
        self._current_image_path = context.image_path
        self.prompt_groups = self._build_prompt_groups_for_image(context)
        
        self.load_model()
        
        pil_image = Image.open(context.image_path)
        context.canvas_width, context.canvas_height = pil_image.size
        
        all_elements = []
        group_stats = {}
        process_order = [
            PromptGroup.BACKGROUND,
            PromptGroup.BASIC_SHAPE, 
            PromptGroup.IMAGE,
            PromptGroup.ARROW
        ]
        
        for group_type in process_order:
            if group_type not in self.prompt_groups:
                continue
                
            group_config = self.prompt_groups[group_type]
            
            if not group_config.prompts:
                continue
                
            self._log(f"  处理组 [{group_config.name}]: {len(group_config.prompts)}个提示词")
            
            raw_results = self._sam3_model.predict(
                context.image_path,
                group_config.prompts,
                score_threshold=group_config.score_threshold,
                min_area=group_config.min_area
            )
            raw_results = self._filter_text_elements(raw_results)
            
            elements = self._convert_to_elements(
                raw_results, 
                start_id=len(all_elements),
                source_group=group_type.value,
                group_priority=group_config.priority
            )
            
            all_elements.extend(elements)
            group_stats[group_config.name] = len(elements)
            
            self._log(f"    提取到 {len(elements)} 个元素")
        
        # 组间去重
        all_elements = self._deduplicate_cross_groups(all_elements)
        
        # 过滤被大图完全包含的小元素
        all_elements = self._filter_contained_elements(all_elements)
        
        context.elements = all_elements
        
        result = ProcessingResult(
            success=True,
            elements=all_elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'group_stats': group_stats,
                'total_before_dedup': sum(group_stats.values()),
                'total_after_dedup': len(all_elements),
                'groups_processed': list(group_stats.keys()),
                'vlm_prompts': context.intermediate_results.get('vlm_prompts', {}),
            }
        )
        
        self._log(f"Done: {len(all_elements)} elements (before dedup: {sum(group_stats.values())})")
        return result

    def _build_prompt_groups_for_image(self, context: ProcessingContext) -> Dict[PromptGroup, PromptGroupConfig]:
        """Load default prompt groups and optionally inject image-specific VLM prompts."""
        prompt_groups = copy.deepcopy(ConfigLoader.get_prompt_groups())
        sam3_config = ConfigLoader.get_sam3_config()
        multimodal_config = ConfigLoader.get_multimodal_config()
        use_vlm_prompts = bool(sam3_config.get('use_vlm_prompts', True))
        prompt_planning_enabled = bool(
            use_vlm_prompts
            and multimodal_config.get('enabled', False)
            and (multimodal_config.get('use_for') or {}).get('prompt_planning', False)
        )
        max_per_group = int(sam3_config.get('vlm_prompt_max_per_group', 6) or 6)
        vlm_record = {
            'enabled': prompt_planning_enabled,
            'configured': use_vlm_prompts,
            'image_path': context.image_path,
            'max_per_group': max_per_group,
            'dynamic_prompts': {key.value: [] for key in PromptGroup},
            'merged_prompts': {},
            'error': None,
        }
        if use_vlm_prompts and not prompt_planning_enabled:
            vlm_record['error'] = 'VLM prompt planning disabled because multimodal.enabled/use_for.prompt_planning is not enabled'

        if prompt_planning_enabled:
            try:
                planner = VLMPromptPlanner(multimodal_config)
                dynamic_prompts = planner.plan(context.image_path, max_per_group=max_per_group)
                mapping = {
                    'image': PromptGroup.IMAGE,
                    'shape': PromptGroup.BASIC_SHAPE,
                    'arrow': PromptGroup.ARROW,
                    'background': PromptGroup.BACKGROUND,
                }
                for key, group_type in mapping.items():
                    prompts = dynamic_prompts.get(key, [])[:max_per_group]
                    vlm_record['dynamic_prompts'][group_type.value] = prompts
                    if group_type in prompt_groups:
                        prompt_groups[group_type].prompts = ConfigLoader._merge_prompts(
                            prompt_groups[group_type].prompts,
                            prompts,
                        )
                self._log(f"VLM动态提示词已注入: {sum(len(v) for v in dynamic_prompts.values())}个")
            except Exception as exc:
                vlm_record['error'] = str(exc)
                self._log(f"VLM动态提示词跳过: {exc}")

        vlm_record['merged_prompts'] = {
            group_type.value: group_config.prompts
            for group_type, group_config in prompt_groups.items()
        }
        context.intermediate_results['vlm_prompts'] = vlm_record
        self._save_vlm_prompts(context, vlm_record)
        return prompt_groups

    def _save_vlm_prompts(self, context: ProcessingContext, vlm_record: Dict[str, Any]):
        """Save prompts used for this image to output_dir/vlm_prompts.json for debugging."""
        output_dir = context.output_dir or "./output"
        self._ensure_output_dir(output_dir)
        output_path = os.path.join(output_dir, "vlm_prompts.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vlm_record, f, indent=2, ensure_ascii=False)
        self._log(f"VLM提示词已保存: {output_path}")

    def extract_by_group(self, context: ProcessingContext, 
                         group_type: PromptGroup) -> ProcessingResult:
        """Extract only the given prompt group."""
        self._log(f"Extract group [{group_type.value}]: {context.image_path}")
        
        self.load_model()
        
        pil_image = Image.open(context.image_path)
        context.canvas_width, context.canvas_height = pil_image.size
        
        if group_type not in self.prompt_groups:
            return ProcessingResult(
                success=False,
                error_message=f"未知的组类型: {group_type}"
            )
        
        group_config = self.prompt_groups[group_type]
        
        raw_results = self._sam3_model.predict(
            context.image_path,
            group_config.prompts,
            score_threshold=group_config.score_threshold,
            min_area=group_config.min_area
        )
        raw_results = self._filter_text_elements(raw_results)
        elements = self._convert_to_elements(
            raw_results,
            start_id=0,
            source_group=group_type.value,
            group_priority=group_config.priority
        )
        
        elements = self._deduplicate_within_group(elements)
        
        context.elements = elements
        
        return ProcessingResult(
            success=True,
            elements=elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'group': group_type.value,
                'prompts_used': group_config.prompts,
                'element_count': len(elements)
            }
        )
    
    def extract_with_custom_prompts(self, context: ProcessingContext,
                                    prompts: List[str],
                                    score_threshold: float = 0.5,
                                    min_area: int = 100) -> ProcessingResult:
        """
        使用自定义提示词提取（不使用分组）
        
        Args:
            context: 处理上下文
            prompts: 自定义提示词列表
            score_threshold: 置信度阈值
            min_area: 最小面积
        """
        self._log(f"自定义提取: {prompts}")
        
        self.load_model()
        
        pil_image = Image.open(context.image_path)
        context.canvas_width, context.canvas_height = pil_image.size
        
        raw_results = self._sam3_model.predict(
            context.image_path,
            prompts,
            score_threshold=score_threshold,
            min_area=min_area
        )
        
        elements = self._convert_to_elements(
            raw_results,
            start_id=0,
            source_group="custom",
            group_priority=2
        )
        
        elements = self._deduplicate_within_group(elements)
        
        context.elements = elements
        
        return ProcessingResult(
            success=True,
            elements=elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'prompts_used': prompts,
                'element_count': len(elements)
            }
        )
    


# ======================== 快捷函数 ========================
def extract_elements(image_path: str, 
                     groups: List[PromptGroup] = None) -> ProcessingResult:
    """
    快捷函数 - 一行代码提取元素
    
    Args:
        image_path: 图片路径
        groups: 要处理的组列表（默认全部）
        
    Returns:
        ProcessingResult
        
    使用示例:
        # 提取所有元素
        result = extract_elements("test.png")
        
        # 只提取图片和箭头
        result = extract_elements("test.png", groups=[PromptGroup.IMAGE, PromptGroup.ARROW])
    """
    extractor = Sam3InfoExtractor()
    context = ProcessingContext(image_path=image_path)
    
    if groups:
        all_elements = []
        for group in groups:
            result = extractor.extract_by_group(context, group)
            all_elements.extend(result.elements)
        
        for i, elem in enumerate(all_elements):
            elem.id = i
        
        return ProcessingResult(
            success=True,
            elements=all_elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height
        )
    
    return extractor.process(context)

def extract_with_prompts(image_path: str, 
                         prompts: List[str],
                         score_threshold: float = 0.5) -> ProcessingResult:
    """Convenience: extract with custom prompts only."""
    extractor = Sam3InfoExtractor()
    context = ProcessingContext(image_path=image_path)
    
    return extractor.extract_with_custom_prompts(
        context, 
        prompts,
        score_threshold=score_threshold
    )
