"""
任务7：质量评估模块（MetricEvaluator）

================================================================================
设计思路
================================================================================

【核心目标】
计算SAM3和其他检测器"检测到了多少"，以及"漏掉了多少"，用于决定是否需要
进行fallback补救（二次处理）。

这个模块回答的核心问题是：
    "把检测到的元素框掩掉后，剩下还有多少内容没覆盖？这些内容在哪里？"

【评分系统设计 - 满分100分】

核心指标：内容覆盖率（Content Coverage Score）

    计算公式：
        score = (covered_content_pixels / total_content_pixels) × 100
    
    关键概念 - 什么是"内容"（前景）vs "背景"：
        ❌ 背景：纯白、浅灰、浅色的大面积连续区域，这些不需要被检测
        ✅ 内容（前景）：图形、图标、文字、箭头、图片等需要被检测的元素
        
    内容识别策略（改进后）：
        1. 边缘检测（主要）：有边缘的地方才是真正的内容边界
        2. 灰度阈值（辅助）：灰度 < 240 的区域（更严格，排除浅色背景）
        3. 形态学去噪：去除小噪点
        4. 连通域过滤：去除面积太小的区域（可能是噪点）
        
    其中：
        - total_content_pixels: 原图中的前景内容像素（经过背景过滤）
        - covered_content_pixels: 已检测元素的bbox覆盖的前景内容像素
        
    含义解读：
        - 100分：所有前景内容都被检测到了（完美覆盖）
        - 90分：90%的内容被检测到，10%漏检（很好）
        - 70分：70%的内容被检测到，30%漏检（需要refinement）
        - <70分：大量内容漏检，refinement必要
        
    漏检率 = 100 - score（即有多少前景内容没被检测到）
    
    注意：背景不参与计算，所以浅色/白色背景不会影响分数

【问题区域检测 - 双通道策略】
使用双通道检测策略，目标是：
    - 电商长图：鞋子、人脸等商品主体，给出稳定的小框
    - 学术海报：左上小图、热力图、3D块等，每个版块给出清晰矩形

1. 细粒度通道（Fine Channel）：
   - 不做形态学操作，直接在未覆盖内容上做连通域分析
   - 用于检测：小图、图标、人脸、小子图等小目标
   - 参数：面积0.1%~15%，填充率>=20%，宽高比<=6

2. 粗粒度通道（Coarse Channel）：
   - 使用中等核(7×7)的闭操作合并相邻内容
   - 用于检测：版块、大图、分散的图形组
   - 参数：面积0.3%~25%，填充率>=30%，宽高比<=6

3. 小框优先NMS：
   - 关键创新：按面积从小到大排序处理
   - 保留小框，抑制被小框高度覆盖的大框
   - 避免把多个小目标误合并成一个大框

4. 去重过滤：
   - 与已检测元素IoU > 30% 的丢弃（避免重复）
   - 框内部被已覆盖区域占比 > 50% 的丢弃（说明大部分已识别）
   - 框内实际漏检内容 < 10% 的丢弃（说明内容太少）

【可选增强：边缘检测】
对于线条、虚线等稀疏内容，灰度阈值可能漏检。
开启 use_edge_detection 后会同时使用Canny边缘检测，并与灰度阈值取并集。

【与IMG2XML的对比和改进】
- 整合了 detect_missed_images（结构化分块）和 detect_missing_by_coverage（宽松过滤）的优点
- 简化配置参数，提供合理默认值（可通过config覆盖）
- 评分系统更直观：分数=覆盖率，一目了然
- 增加详细的 metrics 字典，便于调试和分析
- 提供可视化保存函数

================================================================================
接口说明
================================================================================

输入：
    - context.image_path: 原始图片路径
    - context.elements: 已检测的元素列表（包含bbox信息）
    - context.canvas_width/height: 画布尺寸

输出（ProcessingResult.metadata）：
    - overall_score: 总体评分（0-100，即覆盖率，越高越好）
    - missing_rate: 漏检率（0-100%，越低越好）
    - bad_regions: 问题区域列表，每个包含：
        - bbox: [x1, y1, x2, y2] - 问题区域的边界框
        - area: 面积（像素）
        - area_ratio: 占图片面积的比例
        - missing_pixels: 该区域内的漏检内容像素数
        - reason: 'uncovered_content'
        - channel: 'fine'（细粒度）或 'coarse'（粗粒度）
        - description: 可读的描述文本
    - metrics: 详细指标字典（用于调试）
    - needs_refinement: 是否建议进行二次处理

================================================================================
使用示例
================================================================================

    from modules import MetricEvaluator, ProcessingContext
    
    evaluator = MetricEvaluator()
    context = ProcessingContext(image_path="test.png", elements=[...])
    
    result = evaluator.process(context)
    
    print(f"覆盖率评分: {result.metadata['overall_score']}/100")
    print(f"漏检率: {result.metadata['missing_rate']}%")
    print(f"问题区域: {len(result.metadata['bad_regions'])}个")
    print(f"是否需要refinement: {result.metadata['needs_refinement']}")
    
    for region in result.metadata['bad_regions']:
        print(f"  位置: {region['bbox']}, 占比: {region['area_ratio']*100:.2f}%")
        print(f"    检测通道: {region['channel']}, 漏检像素: {region['missing_pixels']}")
    
    # 保存可视化结果
    evaluator.save_visualization(context, result.metadata['bad_regions'], "eval_result.png")
    evaluator.save_uncovered_mask(context, "uncovered_mask.png")

================================================================================
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

from .base import BaseProcessor, ProcessingContext
from .data_types import ElementInfo, BoundingBox, ProcessingResult


def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """计算两个bbox的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    inter_area = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0




class MetricRegionMixin:
    def _detect_complex_image_regions(self, 
                                       cv2_image: np.ndarray, 
                                       elements: List[ElementInfo],
                                       img_area: int,
                                       context: ProcessingContext) -> List[List[int]]:
        """
        检测复杂图像区域（热力图、照片、图表等）
        
        三重策略：
        1. 检查"图片类但没有 base64"的元素
        2. 检测"完全没有被任何元素覆盖"的高复杂度区域（SAM3 漏检）
        3. 基于图像分析的补充检测
        
        Returns:
            问题区域 bbox 列表
        """
        h, w = cv2_image.shape[:2]
        complex_regions = []
        
        min_region_ratio = self.eval_config.get('complex_min_area_ratio', 0.002)  # 0.2%（更小）
        max_region_ratio = self.eval_config.get('complex_max_area_ratio', 0.30)   # 30%
        min_area = img_area * min_region_ratio
        max_area = img_area * max_region_ratio
        
        # ===== 策略1: 检测"图片类但没有 base64"的元素 =====
        # 注意：跳过面积超过 50% 的大元素（通常是 SAM3 把整图作为 diagram 检测的结果）
        # 这种情况下，图中的其他小组件（箭头、形状、文字）通常已经被正确处理了
        large_element_threshold = 0.50  # 50%
        if elements:
            for elem in elements:
                elem_type = elem.element_type.lower()
                if elem_type in self.IMAGE_CONTENT_TYPES and elem.base64 is None:
                    x1, y1 = max(0, elem.bbox.x1), max(0, elem.bbox.y1)
                    x2, y2 = min(w, elem.bbox.x2), min(h, elem.bbox.y2)
                    area = (x2 - x1) * (y2 - y1)
                    area_ratio = area / img_area
                    
                    # 跳过覆盖整图的大元素（如整图被检测为 diagram）
                    if area_ratio > large_element_threshold:
                        self._log(f"跳过大面积元素: {elem.id}({elem_type}), 面积={area_ratio*100:.1f}% > {large_element_threshold*100}%（可能是整图背景）")
                        continue
                    
                    if area >= min_area * 0.5:  # 正常大小的图片类元素
                        complex_regions.append([x1, y1, x2, y2])
                        self._log(f"检测到未处理的图片类元素: {elem.id}({elem_type}), 面积={area_ratio*100:.1f}%")
        
        # ===== 策略2: 检测没有被"实质内容"覆盖的高复杂度区域 =====
        # 需要排除的区域：
        # 1. 有 base64 图片的元素（真正的图片内容）
        # 2. 面积足够大的矢量图形（如大矩形，占比 > 1%）
        # 3. OCR 文字区域
        # 
        # 注意：小箭头、小圆等不算"覆盖"，因为它们不能代表复杂图像内容
        
        # 创建"已处理"掩码
        processed_mask = np.zeros((h, w), dtype=np.uint8)
        min_element_ratio = 0.01  # 至少 1% 面积的元素才算"覆盖"
        
        # 1. 有 base64 的元素（图片内容，无论大小都算覆盖）
        if elements:
            for elem in elements:
                if elem.base64 is not None:
                    x1, y1 = max(0, elem.bbox.x1), max(0, elem.bbox.y1)
                    x2, y2 = min(w, elem.bbox.x2), min(h, elem.bbox.y2)
                    if x2 > x1 and y2 > y1:
                        processed_mask[y1:y2, x1:x2] = 255
        
        # 2. 面积足够大的矢量图形（排除小箭头、container等）
        # Container 是布局容器，不代表实际内容，不应该算作"覆盖"
        SKIP_TYPES_FOR_COVERAGE = {'container', 'group', 'frame', 'background'}
        if elements:
            for elem in elements:
                if elem.has_xml() and elem.base64 is None:
                    # 跳过布局容器类型
                    if elem.element_type.lower() in SKIP_TYPES_FOR_COVERAGE:
                        continue
                    x1, y1 = max(0, elem.bbox.x1), max(0, elem.bbox.y1)
                    x2, y2 = min(w, elem.bbox.x2), min(h, elem.bbox.y2)
                    elem_area = (x2 - x1) * (y2 - y1)
                    elem_ratio = elem_area / img_area
                    # 只有面积足够大的矢量图形才算"覆盖"
                    if elem_ratio >= min_element_ratio and x2 > x1 and y2 > y1:
                        processed_mask[y1:y2, x1:x2] = 255
        
        # 3. OCR 文字区域
        # 这部分很重要：OCR 已经处理的文字区域不应该被 fallback 重复处理
        text_xml = context.intermediate_results.get('text_xml', '') if hasattr(context, 'intermediate_results') else ''
        if not text_xml and hasattr(context, 'output_dir') and context.output_dir:
            text_xml_path = os.path.join(context.output_dir, 'text_only.drawio')
            if os.path.exists(text_xml_path):
                with open(text_xml_path, 'r', encoding='utf-8') as f:
                    text_xml = f.read()
        
        if text_xml:
            text_bboxes = self._extract_text_bboxes_from_xml(text_xml, w, h)
            for bbox in text_bboxes:
                x1, y1, x2, y2 = bbox
                if x2 > x1 and y2 > y1:
                    # 稍微扩展一点 OCR 区域，避免边缘被检测
                    pad = 5
                    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                    processed_mask[y1:y2, x1:x2] = 255
        
        all_elements_mask = processed_mask
        
        
        # 计算图像复杂度（局部方差）
        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        kernel_size = max(21, min(h, w) // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        local_sq_mean = cv2.blur((gray.astype(np.float32) ** 2), (kernel_size, kernel_size))
        local_variance = np.maximum(local_sq_mean - local_mean ** 2, 0)
        
        # 边缘检测补充（边缘密度高的区域更可能是图像内容）
        edges = cv2.Canny(gray, 30, 100)
        edge_density = cv2.blur(edges.astype(np.float32), (kernel_size, kernel_size))
        
        # 组合复杂度指标
        variance_norm = local_variance / (np.max(local_variance) + 1e-6)
        edge_norm = edge_density / (np.max(edge_density) + 1e-6)
        complexity = variance_norm * 0.6 + edge_norm * 0.4
        
        # 阈值化找高复杂度区域
        complexity_threshold = np.percentile(complexity, 75)  # 前 25% 复杂度
        high_complexity_mask = (complexity > complexity_threshold).astype(np.uint8) * 255
        
        # 形态学处理：适度闭操作，不要过度合并（保留独立区域如 4 个热力图）
        kernel_close = np.ones((15, 15), np.uint8)  # 更小的核，避免合并独立区域
        high_complexity_mask = cv2.morphologyEx(high_complexity_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 找出"高复杂度但没有元素覆盖"的区域
        uncovered_complex = cv2.bitwise_and(high_complexity_mask, cv2.bitwise_not(all_elements_mask))
        
        # 进一步形态学处理：先开操作去噪，再闭操作合并相邻区域
        kernel_open = np.ones((7, 7), np.uint8)  # 小核去噪
        uncovered_complex = cv2.morphologyEx(uncovered_complex, cv2.MORPH_OPEN, kernel_open)
        # 使用较大的核合并相邻区域（如同一个图表的多个部分）
        kernel_close2 = np.ones((51, 51), np.uint8)  # 较大核，合并相邻区域
        uncovered_complex = cv2.morphologyEx(uncovered_complex, cv2.MORPH_CLOSE, kernel_close2)
        
        # 连通域分析
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(uncovered_complex, connectivity=8)
        
        self._log(f"未覆盖复杂区域连通域: {num_labels - 1} 个")
        
        for i in range(1, num_labels):
            x, y, rw, rh, pixel_area = stats[i]
            bbox_area = rw * rh
            
            # 面积过滤
            if bbox_area < min_area or bbox_area > max_area:
                continue
            
            # 填充率检查
            fill_ratio = pixel_area / bbox_area if bbox_area > 0 else 0
            if fill_ratio < 0.15:  # 更宽松
                continue
            
            # 宽高比检查
            aspect = max(rw, rh) / max(1, min(rw, rh))
            if aspect > 8:
                continue
            
            new_bbox = [x, y, x + rw, y + rh]
            
            # 检查是否与已有区域重叠
            is_duplicate = False
            for existing in complex_regions:
                iou = calculate_iou(new_bbox, existing)
                if iou > 0.3:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                complex_regions.append(new_bbox)
                self._log(f"检测到未覆盖的复杂区域: ({x},{y})-({x+rw},{y+rh}), 面积={bbox_area/img_area*100:.1f}%")
        
        # ===== 策略3: 检测中等面积未覆盖内容区域（不依赖复杂度） =====
        # 热力图等渐变色图像边缘不多，复杂度检测可能漏掉
        # 直接检测"有内容但未被覆盖"的中等面积区域
        
        # 创建内容掩码（非白色/接近白色的区域）
        _, content_mask_simple = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        
        # 减去已处理区域（SAM3 元素 + OCR）
        uncovered_content = cv2.bitwise_and(content_mask_simple, cv2.bitwise_not(all_elements_mask))
        
        # 形态学处理：使用较小的核，避免连接独立区域
        # 开操作去噪
        kernel_open3 = np.ones((5, 5), np.uint8)
        uncovered_content = cv2.morphologyEx(uncovered_content, cv2.MORPH_OPEN, kernel_open3)
        # 小核闭操作，只连接非常近的碎片
        kernel_close3 = np.ones((11, 11), np.uint8)
        uncovered_content = cv2.morphologyEx(uncovered_content, cv2.MORPH_CLOSE, kernel_close3)
        
        # 连通域分析
        num_labels3, labels3, stats3, _ = cv2.connectedComponentsWithStats(uncovered_content, connectivity=8)
        
        # 面积阈值：检测 1% - 15% 的中等面积区域
        min_uncovered_threshold = 0.01  # 最小 1%
        max_uncovered_threshold = 0.15  # 最大 15%（避免检测整个边缘区域）
        
        for i in range(1, num_labels3):
            x, y, rw, rh, pixel_area = stats3[i]
            bbox_area = rw * rh
            area_ratio = bbox_area / img_area
            
            # 面积范围过滤
            if area_ratio < min_uncovered_threshold or area_ratio > max_uncovered_threshold:
                continue
            
            # 填充率检查（避免狭长的边缘区域）
            fill_ratio = pixel_area / bbox_area if bbox_area > 0 else 0
            if fill_ratio < 0.25:  # 更严格的填充率
                continue
            
            # 宽高比检查
            aspect = max(rw, rh) / max(1, min(rw, rh))
            if aspect > 4:  # 更严格的宽高比
                continue
            
            new_bbox = [x, y, x + rw, y + rh]
            
            # 检查是否与已有区域重叠
            is_duplicate = False
            for existing in complex_regions:
                iou = calculate_iou(new_bbox, existing)
                if iou > 0.3:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                complex_regions.append(new_bbox)
                self._log(f"检测到未覆盖区域: ({x},{y})-({x+rw},{y+rh}), 面积={area_ratio*100:.1f}%")
        
        return complex_regions
    
    def _merge_nearby_regions(self, 
                               regions: List[Dict], 
                               merge_distance: float,
                               img_area: int) -> List[Dict]:
        """
        合并相邻的小问题区域
        
        只对面积 < 3% 的小区域进行合并，大区域保持独立
        """
        if len(regions) <= 1:
            return regions
        
        # 分离大区域和小区域
        small_threshold = 0.03  # 3%
        large_regions = [r for r in regions if r['area_ratio'] >= small_threshold]
        small_regions = [r for r in regions if r['area_ratio'] < small_threshold]
        
        if len(small_regions) <= 1:
            return regions  # 没有足够的小区域需要合并
        
        def box_distance(box1, box2):
            """计算两个框之间的最小距离"""
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            if x2_1 < x1_2:
                dx = x1_2 - x2_1
            elif x2_2 < x1_1:
                dx = x1_1 - x2_2
            else:
                dx = 0
            
            if y2_1 < y1_2:
                dy = y1_2 - y2_1
            elif y2_2 < y1_1:
                dy = y1_1 - y2_2
            else:
                dy = 0
            
            return max(dx, dy)
        
        # 只对小区域使用并查集合并
        n = len(small_regions)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = box_distance(small_regions[i]['bbox'], small_regions[j]['bbox'])
                if dist < merge_distance:
                    union(i, j)
        
        # 按组合并小区域
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        merged_small = []
        for indices in groups.values():
            if len(indices) == 1:
                merged_small.append(small_regions[indices[0]])
            else:
                boxes = [small_regions[i]['bbox'] for i in indices]
                merged_box = [
                    min(b[0] for b in boxes),
                    min(b[1] for b in boxes),
                    max(b[2] for b in boxes),
                    max(b[3] for b in boxes)
                ]
                merged_area = (merged_box[2] - merged_box[0]) * (merged_box[3] - merged_box[1])
                
                merged_small.append({
                    'bbox': merged_box,
                    'area': merged_area,
                    'area_ratio': round(merged_area / img_area, 4),
                    'missing_pixels': sum(small_regions[i]['missing_pixels'] for i in indices),
                    'reason': 'merged_regions',
                    'channel': 'merged',
                    'description': f'合并了{len(indices)}个相邻区域',
                })
        
        # 返回大区域 + 合并后的小区域
        return large_regions + merged_small
    
    def _merge_overlapping_boxes(self, boxes: List[List[int]]) -> List[List[int]]:
        """合并重叠的边界框"""
        if not boxes:
            return []
        
        # 转换为 numpy 数组方便处理
        boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
        merged = [boxes[0]]
        
        for box in boxes[1:]:
            last = merged[-1]
            # 检查是否重叠
            if (box[0] <= last[2] and box[1] <= last[3] and
                box[2] >= last[0] and box[3] >= last[1]):
                # 合并
                merged[-1] = [
                    min(last[0], box[0]),
                    min(last[1], box[1]),
                    max(last[2], box[2]),
                    max(last[3], box[3])
                ]
            else:
                merged.append(box)
        
        return merged
    
    def _detect_fine_channel(self, uncovered_content: np.ndarray, img_area: int) -> List[List[int]]:
        """
        细粒度通道：不做形态学操作，直接连通域分析
        用于检测小图、图标、人脸等小目标
        """
        min_area = img_area * self.eval_config['fine_min_area_ratio']
        max_area = img_area * self.eval_config['fine_max_area_ratio']
        min_fill = self.eval_config['fine_min_fill_ratio']
        max_aspect = self.eval_config['fine_max_aspect_ratio']
        
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(uncovered_content, connectivity=8)
        
        boxes = []
        for i in range(1, num_labels):
            x, y, rw, rh, cc_area = stats[i]
            if rw <= 0 or rh <= 0:
                continue
            
            bbox_area = rw * rh
            
            # 面积过滤
            if bbox_area < min_area or bbox_area > max_area:
                continue
            
            # 宽高比过滤
            aspect = max(rw, rh) / max(1, min(rw, rh))
            if aspect > max_aspect:
                continue
            
            # 填充率过滤
            fill = cc_area / bbox_area if bbox_area > 0 else 0.0
            if fill < min_fill:
                continue
            
            boxes.append([x, y, x + rw, y + rh])
        
        return boxes
    
    def _detect_coarse_channel(self, uncovered_content: np.ndarray, img_area: int) -> List[List[int]]:
        """
        粗粒度通道：使用闭操作合并相邻内容
        用于检测版块、大图、分散的图形组
        """
        min_area = img_area * self.eval_config['coarse_min_area_ratio']
        max_area = img_area * self.eval_config['coarse_max_area_ratio']
        min_fill = self.eval_config['coarse_min_fill_ratio']
        max_aspect = self.eval_config['coarse_max_aspect_ratio']
        kernel_size = self.eval_config['coarse_kernel_size']
        
        # 闭操作合并相邻内容
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(uncovered_content, cv2.MORPH_CLOSE, kernel)
        
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        
        boxes = []
        for i in range(1, num_labels):
            x, y, rw, rh, cc_area = stats[i]
            if rw <= 0 or rh <= 0:
                continue
            
            bbox_area = rw * rh
            
            # 面积过滤
            if bbox_area < min_area or bbox_area > max_area:
                continue
            
            # 宽高比过滤
            aspect = max(rw, rh) / max(1, min(rw, rh))
            if aspect > max_aspect:
                continue
            
            # 填充率过滤
            fill = cc_area / bbox_area if bbox_area > 0 else 0.0
            if fill < min_fill:
                continue
            
            boxes.append([x, y, x + rw, y + rh])
        
        return boxes
    
    def _nms_smallest_first(self, 
                            candidates: List[Tuple[List[int], str]], 
                            iou_threshold: float) -> List[Tuple[List[int], str]]:
        """
        小框优先NMS：保留小框，抑制被小框高度覆盖的大框
        
        逻辑：
        1. 按面积从小到大排序
        2. 依次处理每个框，保留最小的
        3. 用保留的小框去抑制与之高度重叠的大框
        
        这样可以避免把多个小目标误合并成一个大框
        """
        if not candidates:
            return []
        
        # 计算面积并排序
        boxes_with_area = [(box, channel, (box[2]-box[0])*(box[3]-box[1])) 
                           for box, channel in candidates]
        boxes_with_area.sort(key=lambda x: x[2])  # 按面积升序
        
        keep = []
        suppressed = [False] * len(boxes_with_area)
        
        for i, (box_i, channel_i, area_i) in enumerate(boxes_with_area):
            if suppressed[i]:
                continue
            
            # 保留当前最小的未被抑制的框
            keep.append((box_i, channel_i))
            
            # 用这个小框去抑制后面所有与之高度重叠的框（大的会被抑制）
            for j in range(i + 1, len(boxes_with_area)):
                if suppressed[j]:
                    continue
                
                box_j = boxes_with_area[j][0]
                if calculate_iou(box_i, box_j) > iou_threshold:
                    suppressed[j] = True
        
        return keep
    
    def _filter_candidates(self,
                           candidates: List[Tuple[List[int], str]],
                           covered_mask: np.ndarray,
                           existing_bboxes: List[List[int]],
                           uncovered_content: np.ndarray,
                           img_area: int) -> List[Dict[str, Any]]:
        """
        最终过滤：与已有元素去重 + 覆盖比例过滤
        
        对于 complex 通道（复杂图像检测），使用更宽松的过滤条件
        """
        iou_threshold = self.eval_config['existing_iou_threshold']
        max_covered_ratio = self.eval_config['max_covered_ratio']
        
        bad_regions = []
        
        for box, channel in candidates:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            
            # 对于复杂图像通道，使用更宽松的过滤条件
            is_complex_channel = (channel == 'complex')
            
            # 1. 与已有元素IoU过高则丢弃（复杂图像通道使用更高阈值）
            current_iou_threshold = 0.8 if is_complex_channel else iou_threshold
            if any(calculate_iou(box, eb) > current_iou_threshold for eb in existing_bboxes):
                self._log(f"过滤候选({channel}): IoU过高") if is_complex_channel else None
                continue
            
            # 2. 框内部若大比例像素已被覆盖（SAM/OCR），则丢弃
            # 对于复杂图像通道，跳过这个检查（因为可能被文字覆盖但图像内容没处理）
            if not is_complex_channel and x2 > x1 and y2 > y1:
                cover_ratio = float(np.mean(covered_mask[y1:y2, x1:x2] > 0))
                if cover_ratio > max_covered_ratio:
                    continue
            
            # 3. 计算该区域内的实际漏检内容像素数
            region_uncovered = uncovered_content[y1:y2, x1:x2]
            missing_pixels = int(np.sum(region_uncovered > 0))
            
            # 对于复杂图像通道，使用区域面积作为 missing_pixels（因为整个区域都是复杂图像）
            if is_complex_channel:
                missing_pixels = area
            
            # 如果漏检内容太少，跳过（复杂图像通道不检查）
            min_missing_ratio = self.eval_config.get('min_missing_content_ratio', 0.10)
            if not is_complex_channel and missing_pixels < area * min_missing_ratio:
                continue
            
            bad_regions.append({
                'bbox': [x1, y1, x2, y2],
                'area': area,
                'area_ratio': round(area / img_area, 4),
                'missing_pixels': missing_pixels,
                'reason': 'uncovered_content' if not is_complex_channel else 'complex_image_no_base64',
                'channel': channel,
                'description': f'区域({x1},{y1})-({x2},{y2})存在未识别的{"复杂图像内容" if is_complex_channel else "内容"} [{channel}通道]',
            })
        
        return bad_regions
    
