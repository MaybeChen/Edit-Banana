"""Helper methods for Sam3InfoExtractor."""

import copy
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from .base import ProcessingContext
from .data_types import ElementInfo, BoundingBox, ProcessingResult
from .sam3_config import ConfigLoader, PromptGroup, PromptGroupConfig

class Sam3ExtractorMixin:
    def _filter_text_elements(self, raw_results: List[Dict]) -> List[Dict]:
        """Filter out text-type elements by blacklist/keywords."""
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
        """Map descriptive SAM3 prompts back to canonical element types.

        Descriptive prompts such as "dotted connector line" improve SAM3 recall,
        but downstream processors expect canonical types like connector/cylinder.
        Return (element_type, line_style).
        """
        text = (prompt or "").lower().strip()
        line_style = "dashed" if any(token in text for token in ("dashed", "dotted", "dot")) else None

        if "arrow" in text:
            return "arrow", line_style
        if "connector" in text or "line" in text:
            return "connector", line_style
        if "cylinder" in text or "database" in text:
            return "cylinder", None
        if "circle" in text:
            return "circle", None
        if "ellipse" in text:
            return "ellipse", None
        if "rounded rectangle" in text or "card" in text or "box" in text:
            return "rounded rectangle", None
        if "rectangle" in text:
            return "rectangle", None
        if "diamond" in text:
            return "diamond", None
        if "triangle" in text:
            return "triangle", None
        if "hexagon" in text:
            return "hexagon", None
        if "panel" in text or "container" in text or "background" in text or "frame" in text or "boundary" in text:
            return "container", None
        if "icon" in text or "symbol" in text or "logo" in text or "chart" in text or "picture" in text:
            return "icon", None
        return text, line_style

    def _convert_to_elements(self, raw_results: List[Dict], 
                             start_id: int = 0,
                             source_group: str = "",
                             group_priority: int = 1) -> List[ElementInfo]:
        """将原始结果转换为ElementInfo列表"""
        elements = []
        
        for i, item in enumerate(raw_results):
            bbox = BoundingBox.from_list(item['bbox'])
            
            element_type, line_style = self._normalize_prompt_element_type(item['prompt'])
            element = ElementInfo(
                id=start_id + i,
                element_type=element_type,
                bbox=bbox,
                score=item['score'],
                polygon=item['polygon'],
                mask=item['mask'],
                source_prompt=item['prompt']
            )
            if line_style:
                element.line_style = line_style
            
            element.processing_notes.append(f"source_prompt={item['prompt']}")
            element.processing_notes.append(f"source_group={source_group}")
            element.processing_notes.append(f"area={item.get('area', bbox.area)}")
            element._group_priority = group_priority
            element._source_group = source_group
            
            elements.append(element)
        
        return elements
    
    def _deduplicate_within_group(self, elements: List[ElementInfo], 
                                  iou_threshold: float = None) -> List[ElementInfo]:
        """组内去重"""
        if not elements:
            return elements
        
        if iou_threshold is None:
            iou_threshold = self.dedup_config.get('iou_threshold', 0.7) + 0.15  # 组内阈值稍高
        
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
                
                iou = self._calculate_iou(
                    elem_i.bbox.to_list(),
                    sorted_elements[j].bbox.to_list()
                )
                
                if iou > iou_threshold:
                    dropped.add(j)
        
        for i, elem in enumerate(keep):
            elem.id = i
        
        return keep
    
    def _analyze_region_complexity(self, image_path: str, bbox: List[int]) -> dict:
        """Analyze region complexity (texture, border) for shape vs image classification."""
        try:
            cv2_image = cv2.imread(image_path)
            x1, y1, x2, y2 = bbox
            roi = cv2_image[y1:y2, x1:x2]
            
            if roi.size == 0:
                return {'classification': 'unknown', 'is_complex': False}
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 计算拉普拉斯方差（纹理/边缘丰富度）
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 计算标准差（颜色变化）
            std_dev = np.std(gray)
            
            # 检测边缘
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.count_nonzero(edges) / edges.size
            
            # 检测是否有清晰的矩形边框
            h, w = roi.shape[:2]
            border_size = max(3, min(10, w // 20, h // 20))
            
            # 采样边框区域
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
            
            # 边框和内部的对比度
            border_mean = np.mean(border_pixels)
            inner_mean = np.mean(inner)
            border_contrast = abs(border_mean - inner_mean)
            
            has_clear_border = border_contrast > 25 and edge_ratio > 0.03
            
            # 分类判断
            is_complex = laplacian_var > 800 or std_dev > 55
            
            if is_complex and not has_clear_border:
                classification = 'image_only'  # 真实图片（照片、图表）
            elif has_clear_border and not is_complex:
                classification = 'shape_only'  # 基础图形
            elif has_clear_border and is_complex:
                classification = 'shape_with_content'  # 图形容器+内容
            else:
                classification = 'image_fallback'  # 兜底当图片
            
            return {
                'laplacian_var': laplacian_var,
                'std_dev': std_dev,
                'edge_ratio': edge_ratio,
                'is_complex': is_complex,
                'has_clear_border': has_clear_border,
                'border_contrast': border_contrast,
                'classification': classification
            }
            
        except Exception as e:
            return {'classification': 'unknown', 'is_complex': False, 'error': str(e)}
    
    def _deduplicate_cross_groups(self, elements: List[ElementInfo]) -> List[ElementInfo]:
        """
        跨组去重（智能版）
        
        规则：
        1. 优先保留 priority 高的组
        2. 同优先级时，保留 score 高的
        3. 箭头与其他元素重叠时特殊处理
        4. 【新增】基础图形和图片类重叠时，分析图像复杂度决定保留策略
        """
        if not elements:
            return elements
        
        iou_threshold = self.dedup_config.get('iou_threshold', 0.7)
        arrow_iou_threshold = self.dedup_config.get('arrow_iou_threshold', 0.85)
        shape_image_iou_threshold = self.dedup_config.get('shape_image_iou_threshold', 0.6)
        
        sorted_elements = sorted(
            elements,
            key=lambda x: (getattr(x, '_group_priority', 1), x.score),
            reverse=True
        )
        
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
                
                iou = self._calculate_iou(
                    elem_i.bbox.to_list(),
                    elem_j.bbox.to_list()
                )
                
                if iou < 0.1:
                    continue  # 无重叠
                
                # 【新增】基础图形和图片类重叠的智能判断
                if iou > shape_image_iou_threshold:
                    is_shape_image_overlap = (
                        (group_i == 'shape' and group_j == 'image') or
                        (group_i == 'image' and group_j == 'shape')
                    )
                    # Optional: use _analyze_region_complexity for shape vs image
                    # if is_shape_image_overlap:
                    #     analysis = self._analyze_region_complexity(
                    #         self._current_image_path,
                    #         elem_i.bbox.to_list()
                    #     )
                        
                    #     classification = analysis.get('classification', 'unknown')
                        
                    #     if classification == 'image_only':
                    #         # 真实图片：保留图片类，丢弃图形类
                    #         if group_i == 'shape':
                    #             # elem_i是图形，应该丢弃它，保留elem_j（图片）
                    #             keep.remove(elem_i)
                    #             keep.append(elem_j)
                    #             dropped.add(j)
                    #         else:
                    #             # elem_i是图片，保留
                    #             dropped.add(j)
                    #     elif classification == 'shape_only':
                    #         # 基础图形：保留图形类，丢弃图片类
                    #         if group_i == 'image':
                    #             # elem_i是图片，应该丢弃它，保留elem_j（图形）
                    #             keep.remove(elem_i)
                    #             keep.append(elem_j)
                    #             dropped.add(j)
                    #         else:
                    #             # elem_i是图形，保留
                    #             dropped.add(j)
                    #     elif classification == 'shape_with_content':
                    #         # 图形容器+内容：两者都保留（不去重）
                    #         # 标记为层叠关系
                    #         elem_i.processing_notes.append(f"与{elem_j.id}层叠")
                    #         elem_j.processing_notes.append(f"与{elem_i.id}层叠")
                    #         continue
                    #     else:
                    #         # 兜底：当图片处理，保留图片类
                    #         if group_i == 'shape':
                    #             keep.remove(elem_i)
                    #             keep.append(elem_j)
                    #         dropped.add(j)
                    #     continue
                    if is_shape_image_overlap:
                        # Prefer image over shape when overlapping
                        if group_i == 'shape':
                            if elem_i in keep:
                                keep.remove(elem_i)
                            if elem_j not in keep:
                                keep.append(elem_j)
                            dropped.add(j)
                            break  # elem_i 已移除，退出内层循环
                        else:
                            # elem_i 是 image，保留它，丢弃 shape (elem_j)
                            dropped.add(j)
                        continue
                
                # 标准去重逻辑
                if iou > effective_threshold:
                    dropped.add(j)
        
        for i, elem in enumerate(keep):
            elem.id = i
        
        return keep
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Compute IoU of two boxes."""
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
        """
        过滤被大图完全包含的小元素
        
        规则：
        1. 如果小元素被图片类大元素包含 > 85%，只保留大元素
        2. 图片类：icon, picture, logo, chart, function_graph
        3. 这样可以避免大图里的小箭头/小图形被单独提取
        """
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
                        self._log(f"Filter {elem_j.id}({type_j}): contained by {elem_i.id}({type_i}) {containment:.0%}")
                elif area_j > area_i:
                    containment = self._calculate_containment(bbox_j, bbox_i)
                    if containment > 0.85 and type_j in IMAGE_TYPES:
                        to_remove.add(i)
                        self._log(f"Filter {elem_i.id}({type_i}): contained by {elem_j.id}({type_j}) {containment:.0%}")
                        break
        
        result = [e for i, e in enumerate(elements) if i not in to_remove]
        
        # 重新编号
        for i, elem in enumerate(result):
            elem.id = i
        
        if to_remove:
            self._log(f"完全包含过滤: 移除了 {len(to_remove)} 个被大图包含的小元素")
        
        return result
    
    def _calculate_containment(self, box_outer: List[int], box_inner: List[int]) -> float:
        """
        计算 box_inner 被 box_outer 包含的比例
        
        返回值范围 [0, 1]：
        - 1.0 表示完全包含
        - 0.0 表示无重叠
        """
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
        """Save visualization (one color per group)."""
        cv2_image = cv2.imread(context.image_path)
        GROUP_COLORS = {
            'image': (0, 255, 0),
            'arrow': (255, 0, 0),
            'shape': (0, 0, 255),
            'background': (255, 255, 0),
            'custom': (128, 0, 128),
        }
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
            
            label = f"{elem.id}:{elem.element_type}"
            cv2.putText(image, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(output_path, result)
        
        self._log(f"可视化已保存: {output_path}")
    
    def save_metadata(self, context: ProcessingContext, output_path: str):
        """保存元数据JSON"""
        import json
        
        metadata = {
            'image_path': context.image_path,
            'image_size': {
                'width': context.canvas_width,
                'height': context.canvas_height
            },
            'total_elements': len(context.elements),
            'by_group': {},
            'by_type': {},
            'elements': []
        }
        
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
        self._log(f"Metadata saved: {output_path}")

    def get_all_prompts(self) -> Dict[str, List[str]]:
        """Return all prompt groups."""
        return {
            group.value: config.prompts.copy()
            for group, config in self.prompt_groups.items()
        }
    
    def get_group_config(self, group_type: PromptGroup) -> Optional[PromptGroupConfig]:
        """获取指定组的配置"""
        return self.prompt_groups.get(group_type)
    
    def add_prompts_to_group(self, group_type: PromptGroup, prompts: List[str]):
        """向指定组添加提示词（运行时）"""
        if group_type in self.prompt_groups:
            for p in prompts:
                self.prompt_groups[group_type].add_prompt(p)
    
    def remove_prompts_from_group(self, group_type: PromptGroup, prompts: List[str]):
        """Remove prompts from a group at runtime."""
        if group_type in self.prompt_groups:
            for p in prompts:
                self.prompt_groups[group_type].remove_prompt(p)
    
    def set_group_threshold(self, group_type: PromptGroup,
                            score_threshold: float = None,
                            min_area: int = None):
        """Set group thresholds at runtime."""
        if group_type in self.prompt_groups:
            if score_threshold is not None:
                self.prompt_groups[group_type].score_threshold = score_threshold
            if min_area is not None:
                self.prompt_groups[group_type].min_area = min_area
    
    def print_prompt_groups(self):
        """打印当前词库配置"""
        print("\n" + "="*60)
        print("当前SAM3提示词词库配置 (从 config.yaml 加载)")
        print("="*60)
        
        for group_type, config in self.prompt_groups.items():
            print(f"\n[{config.name}] ({group_type.value})")
            print(f"  置信度阈值: {config.score_threshold}")
            print(f"  最小面积: {config.min_area}")
            print(f"  优先级: {config.priority}")
            print(f"  提示词 ({len(config.prompts)}个):")
            for p in config.prompts:
                print(f"    - {p}")
        
        print("\n" + "="*60)
        print(f"配置文件路径: {ConfigLoader.get_config_path()}")
        print("="*60)
