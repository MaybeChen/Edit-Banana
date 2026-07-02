import os
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
from .base import BaseProcessor, ProcessingContext
from .data_types import ElementInfo, BoundingBox, ProcessingResult
def calculate_iou(box1: List[int], box2: List[int]) -> float:
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
class MetricEvaluator(BaseProcessor):
    DEFAULT_CONFIG = {'content_threshold': 245, 'use_edge_detection': True, 'edge_low_threshold': 30, 'edge_high_threshold': 100, 'filter_background': True, 'background_denoise_kernel': 2, 'min_content_area': 30, 'fine_min_area_ratio': 0.0005, 'fine_max_area_ratio': 0.2, 'fine_min_fill_ratio': 0.15, 'fine_max_aspect_ratio': 8.0, 'coarse_min_area_ratio': 0.002, 'coarse_max_area_ratio': 0.3, 'coarse_min_fill_ratio': 0.2, 'coarse_max_aspect_ratio': 8.0, 'coarse_kernel_size': 5, 'nms_iou_threshold': 0.3, 'existing_iou_threshold': 0.5, 'max_covered_ratio': 0.7, 'good_coverage_threshold': 95, 'acceptable_threshold': 80, 'min_missing_content_ratio': 0.05}
    def __init__(self, config=None):
        super().__init__(config)
        self.eval_config = {**self.DEFAULT_CONFIG, **(config or {})}
    def process(self, context: ProcessingContext) -> ProcessingResult:
        self._log('开始质量评估')
        if not context.image_path or not os.path.exists(context.image_path):
            return ProcessingResult(success=False, error_message='图片路径无效')
        cv2_image = cv2.imread(context.image_path)
        if cv2_image is None:
            return ProcessingResult(success=False, error_message='无法读取图片')
        h, w = cv2_image.shape[:2]
        img_area = h * w
        content_mask = self._create_content_mask(cv2_image)
        total_content_pixels = int(np.sum(content_mask > 0))
        text_xml = context.intermediate_results.get('text_xml', None)
        covered_mask, existing_bboxes = self._create_covered_mask(context.elements, h, w, text_xml)
        covered_content = cv2.bitwise_and(content_mask, covered_mask)
        covered_content_pixels = int(np.sum(covered_content > 0))
        if total_content_pixels > 0:
            content_coverage = covered_content_pixels / total_content_pixels * 100
        else:
            content_coverage = 100.0
        missing_rate = 100.0 - content_coverage
        uncovered_content = cv2.bitwise_and(content_mask, cv2.bitwise_not(covered_mask))
        bad_regions = self._detect_bad_regions(cv2_image, content_mask, covered_mask, existing_bboxes, img_area, context.elements, context)
        bad_region_mask = np.zeros((h, w), dtype=np.uint8)
        for region in bad_regions:
            x1, y1, x2, y2 = region['bbox']
            x1, y1 = (max(0, x1), max(0, y1))
            x2, y2 = (min(w, x2), min(h, y2))
            if x2 > x1 and y2 > y1:
                bad_region_mask[y1:y2, x1:x2] = 255
        actual_bad_region_pixels = int(np.sum(bad_region_mask > 0))
        total_bad_region_ratio = actual_bad_region_pixels / img_area * 100 if img_area > 0 else 0
        overall_score = max(0, 100.0 - total_bad_region_ratio)
        has_complex_image_regions = any((r.get('channel') == 'complex' for r in bad_regions))
        needs_refinement = len(bad_regions) > 0 or has_complex_image_regions
        total_bad_region_area = actual_bad_region_pixels
        metrics = {'overall_score': round(overall_score, 2), 'pixel_coverage': round(content_coverage, 2), 'missing_rate': round(missing_rate, 2), 'total_content_pixels': total_content_pixels, 'covered_content_pixels': covered_content_pixels, 'missing_content_pixels': total_content_pixels - covered_content_pixels, 'image_area': img_area, 'content_ratio': round(total_content_pixels / img_area * 100, 2), 'element_count': len(context.elements), 'bad_region_count': len(bad_regions), 'total_bad_region_area': total_bad_region_area, 'total_bad_region_ratio': round(total_bad_region_ratio, 2)}
        self._log(f'评估完成: 评分={overall_score:.1f}, 问题区域={len(bad_regions)}个, 问题面积={total_bad_region_ratio:.1f}%')
        if context.output_dir and os.path.exists(context.output_dir):
            uncovered_vis_path = os.path.join(context.output_dir, 'metric_uncovered.png')
            self._save_uncovered_visualization(cv2_image, uncovered_content, covered_mask, bad_regions, uncovered_vis_path)
            eval_json_path = os.path.join(context.output_dir, 'metric_evaluation.json')
            self._save_evaluation_json(metrics, bad_regions, needs_refinement, overall_score, eval_json_path)
        return ProcessingResult(success=True, elements=context.elements, canvas_width=context.canvas_width or w, canvas_height=context.canvas_height or h, metadata={'overall_score': round(overall_score, 2), 'pixel_coverage': round(content_coverage, 2), 'bad_region_ratio': round(total_bad_region_ratio, 2), 'metrics': metrics, 'bad_regions': bad_regions, 'needs_refinement': needs_refinement})
    def _create_content_mask(self, cv2_image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if self.eval_config.get('use_edge_detection', True):
            edges = cv2.Canny(gray, self.eval_config['edge_low_threshold'], self.eval_config['edge_high_threshold'])
            kernel = np.ones((5, 5), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)
            edge_mask = edges_dilated
        else:
            edge_mask = np.zeros((h, w), dtype=np.uint8)
        threshold = self.eval_config['content_threshold']
        content_by_gray = (gray < threshold).astype(np.uint8) * 255
        if self.eval_config.get('use_edge_detection', True):
            content_mask = cv2.bitwise_or(content_by_gray, edge_mask)
        else:
            content_mask = content_by_gray
        if self.eval_config.get('filter_background', True):
            denoise_size = self.eval_config.get('background_denoise_kernel', 3)
            if denoise_size > 0:
                denoise_kernel = np.ones((denoise_size, denoise_size), np.uint8)
                content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, denoise_kernel)
            min_area = self.eval_config.get('min_content_area', 50)
            if min_area > 0:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(content_mask, connectivity=8)
                filtered_mask = np.zeros_like(content_mask)
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= min_area:
                        filtered_mask[labels == i] = 255
                content_mask = filtered_mask
        content_pixels = int(np.sum(content_mask > 0))
        content_ratio = content_pixels / (h * w) * 100
        self._log(f'内容检测: {content_pixels}px ({content_ratio:.1f}% of image)')
        return content_mask
    IMAGE_CONTENT_TYPES = {'icon', 'picture', 'photo', 'chart', 'function_graph', 'screenshot', 'image', 'diagram', 'logo', 'heatmap', 'graph', 'line graph', 'bar graph', 'pie chart', 'scatter plot', 'histogram'}
    VECTOR_SHAPE_TYPES = {'rectangle', 'rounded rectangle', 'circle', 'ellipse', 'diamond', 'triangle', 'cloud', 'arrow', 'line', 'connector', 'polygon', 'section_panel', 'title_bar', 'background'}
    def _create_covered_mask(self, elements: List[ElementInfo], height: int, width: int, text_xml: str=None) -> Tuple[np.ndarray, List[List[int]]]:
        covered_mask = np.zeros((height, width), dtype=np.uint8)
        existing_bboxes = []
        img_area = height * width
        valid_count = 0
        skipped_image_no_base64 = 0
        skipped_no_output = 0
        for elem in elements:
            elem_type = elem.element_type.lower()
            elem_area = (elem.bbox.x2 - elem.bbox.x1) * (elem.bbox.y2 - elem.bbox.y1)
            area_ratio = elem_area / img_area if img_area > 0 else 0
            is_image_content = elem_type in self.IMAGE_CONTENT_TYPES
            if is_image_content:
                has_valid_output = elem.base64 is not None
                if not has_valid_output:
                    skipped_image_no_base64 += 1
                    continue
            else:
                has_valid_output = elem.has_xml() or elem.base64 is not None
                if not has_valid_output:
                    skipped_no_output += 1
                    continue
            x1 = max(0, min(width, elem.bbox.x1))
            y1 = max(0, min(height, elem.bbox.y1))
            x2 = max(0, min(width, elem.bbox.x2))
            y2 = max(0, min(height, elem.bbox.y2))
            if x2 > x1 and y2 > y1:
                covered_mask[y1:y2, x1:x2] = 255
                existing_bboxes.append([x1, y1, x2, y2])
                valid_count += 1
        text_count = 0
        if text_xml:
            text_bboxes = self._extract_text_bboxes_from_xml(text_xml, width, height)
            for bbox in text_bboxes:
                x1, y1, x2, y2 = bbox
                if x2 > x1 and y2 > y1:
                    covered_mask[y1:y2, x1:x2] = 255
                    existing_bboxes.append([x1, y1, x2, y2])
                    text_count += 1
        self._log(f'覆盖区域统计: 有效元素={valid_count}, 图片类无base64={skipped_image_no_base64}, 无输出={skipped_no_output}, OCR文字={text_count}')
        return (covered_mask, existing_bboxes)
    def _extract_text_bboxes_from_xml(self, text_xml: str, img_width: int, img_height: int) -> List[List[int]]:
        import re
        bboxes = []
        pattern = '<mxGeometry\\s+x="([^"]+)"\\s+y="([^"]+)"\\s+width="([^"]+)"\\s+height="([^"]+)"'
        for match in re.finditer(pattern, text_xml):
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                w = float(match.group(3))
                h = float(match.group(4))
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(img_width, int(x + w))
                y2 = min(img_height, int(y + h))
                if x2 > x1 and y2 > y1:
                    bboxes.append([x1, y1, x2, y2])
            except (ValueError, IndexError):
                continue
        return bboxes
    def _detect_bad_regions(self, cv2_image: np.ndarray, content_mask: np.ndarray, covered_mask: np.ndarray, existing_bboxes: List[List[int]], img_area: int, elements: List[ElementInfo]=None, context: ProcessingContext=None) -> List[Dict[str, Any]]:
        h, w = cv2_image.shape[:2]
        uncovered_content = cv2.bitwise_and(content_mask, cv2.bitwise_not(covered_mask))
        candidates = []
        fine_candidates = self._detect_fine_channel(uncovered_content, img_area)
        candidates.extend([(box, 'fine') for box in fine_candidates])
        coarse_candidates = self._detect_coarse_channel(uncovered_content, img_area)
        candidates.extend([(box, 'coarse') for box in coarse_candidates])
        complex_candidates = self._detect_complex_image_regions(cv2_image, elements, img_area, context)
        candidates.extend([(box, 'complex') for box in complex_candidates])
        self._log(f'三重检测: 细粒度={len(fine_candidates)}, 粗粒度={len(coarse_candidates)}, 复杂图像={len(complex_candidates)}')
        nms_threshold = self.eval_config['nms_iou_threshold']
        candidates = self._nms_smallest_first(candidates, nms_threshold)
        self._log(f'NMS后: {len(candidates)}个候选')
        bad_regions = self._filter_candidates(candidates, covered_mask, existing_bboxes, uncovered_content, img_area)
        h, w = covered_mask.shape[:2]
        merge_distance = min(h, w) * 0.1
        bad_regions = self._merge_nearby_regions(bad_regions, merge_distance, img_area)
        bad_regions.sort(key=lambda r: r['area'], reverse=True)
        return bad_regions
    def _detect_complex_image_regions(self, cv2_image: np.ndarray, elements: List[ElementInfo], img_area: int, context: ProcessingContext) -> List[List[int]]:
        h, w = cv2_image.shape[:2]
        complex_regions = []
        min_region_ratio = self.eval_config.get('complex_min_area_ratio', 0.002)
        max_region_ratio = self.eval_config.get('complex_max_area_ratio', 0.3)
        min_area = img_area * min_region_ratio
        max_area = img_area * max_region_ratio
        large_element_threshold = 0.5
        if elements:
            for elem in elements:
                elem_type = elem.element_type.lower()
                if elem_type in self.IMAGE_CONTENT_TYPES and elem.base64 is None:
                    x1, y1 = (max(0, elem.bbox.x1), max(0, elem.bbox.y1))
                    x2, y2 = (min(w, elem.bbox.x2), min(h, elem.bbox.y2))
                    area = (x2 - x1) * (y2 - y1)
                    area_ratio = area / img_area
                    if area_ratio > large_element_threshold:
                        self._log(f'跳过大面积元素: {elem.id}({elem_type}), 面积={area_ratio * 100:.1f}% > {large_element_threshold * 100}%（可能是整图背景）')
                        continue
                    if area >= min_area * 0.5:
                        complex_regions.append([x1, y1, x2, y2])
                        self._log(f'检测到未处理的图片类元素: {elem.id}({elem_type}), 面积={area_ratio * 100:.1f}%')
        processed_mask = np.zeros((h, w), dtype=np.uint8)
        min_element_ratio = 0.01
        if elements:
            for elem in elements:
                if elem.base64 is not None:
                    x1, y1 = (max(0, elem.bbox.x1), max(0, elem.bbox.y1))
                    x2, y2 = (min(w, elem.bbox.x2), min(h, elem.bbox.y2))
                    if x2 > x1 and y2 > y1:
                        processed_mask[y1:y2, x1:x2] = 255
        SKIP_TYPES_FOR_COVERAGE = {'container', 'group', 'frame', 'background'}
        if elements:
            for elem in elements:
                if elem.has_xml() and elem.base64 is None:
                    if elem.element_type.lower() in SKIP_TYPES_FOR_COVERAGE:
                        continue
                    x1, y1 = (max(0, elem.bbox.x1), max(0, elem.bbox.y1))
                    x2, y2 = (min(w, elem.bbox.x2), min(h, elem.bbox.y2))
                    elem_area = (x2 - x1) * (y2 - y1)
                    elem_ratio = elem_area / img_area
                    if elem_ratio >= min_element_ratio and x2 > x1 and (y2 > y1):
                        processed_mask[y1:y2, x1:x2] = 255
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
                    pad = 5
                    x1, y1 = (max(0, x1 - pad), max(0, y1 - pad))
                    x2, y2 = (min(w, x2 + pad), min(h, y2 + pad))
                    processed_mask[y1:y2, x1:x2] = 255
        all_elements_mask = processed_mask
        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        kernel_size = max(21, min(h, w) // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(gray.astype(np.float32) ** 2, (kernel_size, kernel_size))
        local_variance = np.maximum(local_sq_mean - local_mean ** 2, 0)
        edges = cv2.Canny(gray, 30, 100)
        edge_density = cv2.blur(edges.astype(np.float32), (kernel_size, kernel_size))
        variance_norm = local_variance / (np.max(local_variance) + 1e-06)
        edge_norm = edge_density / (np.max(edge_density) + 1e-06)
        complexity = variance_norm * 0.6 + edge_norm * 0.4
        complexity_threshold = np.percentile(complexity, 75)
        high_complexity_mask = (complexity > complexity_threshold).astype(np.uint8) * 255
        kernel_close = np.ones((15, 15), np.uint8)
        high_complexity_mask = cv2.morphologyEx(high_complexity_mask, cv2.MORPH_CLOSE, kernel_close)
        uncovered_complex = cv2.bitwise_and(high_complexity_mask, cv2.bitwise_not(all_elements_mask))
        kernel_open = np.ones((7, 7), np.uint8)
        uncovered_complex = cv2.morphologyEx(uncovered_complex, cv2.MORPH_OPEN, kernel_open)
        kernel_close2 = np.ones((51, 51), np.uint8)
        uncovered_complex = cv2.morphologyEx(uncovered_complex, cv2.MORPH_CLOSE, kernel_close2)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(uncovered_complex, connectivity=8)
        self._log(f'未覆盖复杂区域连通域: {num_labels - 1} 个')
        for i in range(1, num_labels):
            x, y, rw, rh, pixel_area = stats[i]
            bbox_area = rw * rh
            if bbox_area < min_area or bbox_area > max_area:
                continue
            fill_ratio = pixel_area / bbox_area if bbox_area > 0 else 0
            if fill_ratio < 0.15:
                continue
            aspect = max(rw, rh) / max(1, min(rw, rh))
            if aspect > 8:
                continue
            new_bbox = [x, y, x + rw, y + rh]
            is_duplicate = False
            for existing in complex_regions:
                iou = calculate_iou(new_bbox, existing)
                if iou > 0.3:
                    is_duplicate = True
                    break
            if not is_duplicate:
                complex_regions.append(new_bbox)
                self._log(f'检测到未覆盖的复杂区域: ({x},{y})-({x + rw},{y + rh}), 面积={bbox_area / img_area * 100:.1f}%')
        _, content_mask_simple = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        uncovered_content = cv2.bitwise_and(content_mask_simple, cv2.bitwise_not(all_elements_mask))
        kernel_open3 = np.ones((5, 5), np.uint8)
        uncovered_content = cv2.morphologyEx(uncovered_content, cv2.MORPH_OPEN, kernel_open3)
        kernel_close3 = np.ones((11, 11), np.uint8)
        uncovered_content = cv2.morphologyEx(uncovered_content, cv2.MORPH_CLOSE, kernel_close3)
        num_labels3, labels3, stats3, _ = cv2.connectedComponentsWithStats(uncovered_content, connectivity=8)
        min_uncovered_threshold = 0.01
        max_uncovered_threshold = 0.15
        for i in range(1, num_labels3):
            x, y, rw, rh, pixel_area = stats3[i]
            bbox_area = rw * rh
            area_ratio = bbox_area / img_area
            if area_ratio < min_uncovered_threshold or area_ratio > max_uncovered_threshold:
                continue
            fill_ratio = pixel_area / bbox_area if bbox_area > 0 else 0
            if fill_ratio < 0.25:
                continue
            aspect = max(rw, rh) / max(1, min(rw, rh))
            if aspect > 4:
                continue
            new_bbox = [x, y, x + rw, y + rh]
            is_duplicate = False
            for existing in complex_regions:
                iou = calculate_iou(new_bbox, existing)
                if iou > 0.3:
                    is_duplicate = True
                    break
            if not is_duplicate:
                complex_regions.append(new_bbox)
                self._log(f'检测到未覆盖区域: ({x},{y})-({x + rw},{y + rh}), 面积={area_ratio * 100:.1f}%')
        return complex_regions
    def _merge_nearby_regions(self, regions: List[Dict], merge_distance: float, img_area: int) -> List[Dict]:
        if len(regions) <= 1:
            return regions
        small_threshold = 0.03
        large_regions = [r for r in regions if r['area_ratio'] >= small_threshold]
        small_regions = [r for r in regions if r['area_ratio'] < small_threshold]
        if len(small_regions) <= 1:
            return regions
        def box_distance(box1, box2):
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
        n = len(small_regions)
        parent = list(range(n))
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            px, py = (find(x), find(y))
            if px != py:
                parent[px] = py
        for i in range(n):
            for j in range(i + 1, n):
                dist = box_distance(small_regions[i]['bbox'], small_regions[j]['bbox'])
                if dist < merge_distance:
                    union(i, j)
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
                merged_box = [min((b[0] for b in boxes)), min((b[1] for b in boxes)), max((b[2] for b in boxes)), max((b[3] for b in boxes))]
                merged_area = (merged_box[2] - merged_box[0]) * (merged_box[3] - merged_box[1])
                merged_small.append({'bbox': merged_box, 'area': merged_area, 'area_ratio': round(merged_area / img_area, 4), 'missing_pixels': sum((small_regions[i]['missing_pixels'] for i in indices)), 'reason': 'merged_regions', 'channel': 'merged', 'description': f'合并了{len(indices)}个相邻区域'})
        return large_regions + merged_small
    def _merge_overlapping_boxes(self, boxes: List[List[int]]) -> List[List[int]]:
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: (b[0], b[1]))
        merged = [boxes[0]]
        for box in boxes[1:]:
            last = merged[-1]
            if box[0] <= last[2] and box[1] <= last[3] and (box[2] >= last[0]) and (box[3] >= last[1]):
                merged[-1] = [min(last[0], box[0]), min(last[1], box[1]), max(last[2], box[2]), max(last[3], box[3])]
            else:
                merged.append(box)
        return merged
    def _detect_fine_channel(self, uncovered_content: np.ndarray, img_area: int) -> List[List[int]]:
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
            if bbox_area < min_area or bbox_area > max_area:
                continue
            aspect = max(rw, rh) / max(1, min(rw, rh))
            if aspect > max_aspect:
                continue
            fill = cc_area / bbox_area if bbox_area > 0 else 0.0
            if fill < min_fill:
                continue
            boxes.append([x, y, x + rw, y + rh])
        return boxes
    def _detect_coarse_channel(self, uncovered_content: np.ndarray, img_area: int) -> List[List[int]]:
        min_area = img_area * self.eval_config['coarse_min_area_ratio']
        max_area = img_area * self.eval_config['coarse_max_area_ratio']
        min_fill = self.eval_config['coarse_min_fill_ratio']
        max_aspect = self.eval_config['coarse_max_aspect_ratio']
        kernel_size = self.eval_config['coarse_kernel_size']
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(uncovered_content, cv2.MORPH_CLOSE, kernel)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        boxes = []
        for i in range(1, num_labels):
            x, y, rw, rh, cc_area = stats[i]
            if rw <= 0 or rh <= 0:
                continue
            bbox_area = rw * rh
            if bbox_area < min_area or bbox_area > max_area:
                continue
            aspect = max(rw, rh) / max(1, min(rw, rh))
            if aspect > max_aspect:
                continue
            fill = cc_area / bbox_area if bbox_area > 0 else 0.0
            if fill < min_fill:
                continue
            boxes.append([x, y, x + rw, y + rh])
        return boxes
    def _nms_smallest_first(self, candidates: List[Tuple[List[int], str]], iou_threshold: float) -> List[Tuple[List[int], str]]:
        if not candidates:
            return []
        boxes_with_area = [(box, channel, (box[2] - box[0]) * (box[3] - box[1])) for box, channel in candidates]
        boxes_with_area.sort(key=lambda x: x[2])
        keep = []
        suppressed = [False] * len(boxes_with_area)
        for i, (box_i, channel_i, area_i) in enumerate(boxes_with_area):
            if suppressed[i]:
                continue
            keep.append((box_i, channel_i))
            for j in range(i + 1, len(boxes_with_area)):
                if suppressed[j]:
                    continue
                box_j = boxes_with_area[j][0]
                if calculate_iou(box_i, box_j) > iou_threshold:
                    suppressed[j] = True
        return keep
    def _filter_candidates(self, candidates: List[Tuple[List[int], str]], covered_mask: np.ndarray, existing_bboxes: List[List[int]], uncovered_content: np.ndarray, img_area: int) -> List[Dict[str, Any]]:
        iou_threshold = self.eval_config['existing_iou_threshold']
        max_covered_ratio = self.eval_config['max_covered_ratio']
        bad_regions = []
        for box, channel in candidates:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            is_complex_channel = channel == 'complex'
            current_iou_threshold = 0.8 if is_complex_channel else iou_threshold
            if any((calculate_iou(box, eb) > current_iou_threshold for eb in existing_bboxes)):
                self._log(f'过滤候选({channel}): IoU过高') if is_complex_channel else None
                continue
            if not is_complex_channel and x2 > x1 and (y2 > y1):
                cover_ratio = float(np.mean(covered_mask[y1:y2, x1:x2] > 0))
                if cover_ratio > max_covered_ratio:
                    continue
            region_uncovered = uncovered_content[y1:y2, x1:x2]
            missing_pixels = int(np.sum(region_uncovered > 0))
            if is_complex_channel:
                missing_pixels = area
            min_missing_ratio = self.eval_config.get('min_missing_content_ratio', 0.1)
            if not is_complex_channel and missing_pixels < area * min_missing_ratio:
                continue
            bad_regions.append({'bbox': [x1, y1, x2, y2], 'area': area, 'area_ratio': round(area / img_area, 4), 'missing_pixels': missing_pixels, 'reason': 'uncovered_content' if not is_complex_channel else 'complex_image_no_base64', 'channel': channel, 'description': f"区域({x1},{y1})-({x2},{y2})存在未识别的{('复杂图像内容' if is_complex_channel else '内容')} [{channel}通道]"})
        return bad_regions
    def _save_uncovered_visualization(self, cv2_image: np.ndarray, uncovered_content: np.ndarray, covered_mask: np.ndarray, bad_regions: List[Dict], output_path: str):
        h, w = cv2_image.shape[:2]
        result = cv2_image.copy()
        overlay = cv2_image.copy()
        for i, region in enumerate(bad_regions):
            x1, y1, x2, y2 = region['bbox']
            x1, y1 = (max(0, x1), max(0, y1))
            x2, y2 = (min(w, x2), min(h, y2))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 4)
            area_pct = region.get('area_ratio', 0) * 100
            channel = region.get('channel', 'unknown')
            reason = region.get('reason', 'unknown')
            if 'complex' in channel:
                reason_short = 'IMAGE_NO_BASE64'
            elif channel == 'fine':
                reason_short = 'UNCOVERED_FINE'
            elif channel == 'coarse':
                reason_short = 'UNCOVERED_COARSE'
            else:
                reason_short = channel.upper()
            label = f'#{i + 1} {reason_short} ({area_pct:.1f}%)'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(result, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 0, 255), -1)
            cv2.putText(result, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        alpha = 0.3
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
        legend_bg = np.zeros((120, w, 3), dtype=np.uint8)
        legend_bg[:] = (40, 40, 40)
        cv2.putText(legend_bg, f'METRIC EVALUATION - Problem Regions for Fallback', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(legend_bg, f'Total Bad Regions: {len(bad_regions)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(legend_bg, f'RED = regions that need fallback (image content without base64 processing)', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        result = np.vstack([legend_bg, result])
        cv2.imwrite(output_path, result)
        self._log(f'保存问题区域可视化: {output_path}')
    def _save_evaluation_json(self, metrics: Dict, bad_regions: List[Dict], needs_refinement: bool, overall_score: float, output_path: str):
        import json
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        evaluation_result = {'overall_score': round(float(overall_score), 2), 'needs_refinement': bool(needs_refinement), 'metrics': convert_to_native(metrics), 'bad_regions': convert_to_native(bad_regions), 'summary': {'score': f'{overall_score:.1f}/100', 'bad_region_ratio': f"{metrics['total_bad_region_ratio']:.1f}%", 'bad_region_count': int(metrics['bad_region_count']), 'pixel_coverage': f"{metrics['pixel_coverage']:.1f}%", 'element_count': int(metrics['element_count'])}}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        self._log(f'保存评估结果: {output_path}')
    def save_visualization(self, context: ProcessingContext, bad_regions: List[Dict], output_path: str):
        if not context.image_path or not os.path.exists(context.image_path):
            return
        img = cv2.imread(context.image_path)
        if img is None:
            return
        h, w = img.shape[:2]
        for elem in context.elements:
            x1 = max(0, min(w, elem.bbox.x1))
            y1 = max(0, min(h, elem.bbox.y1))
            x2 = max(0, min(w, elem.bbox.x2))
            y2 = max(0, min(h, elem.bbox.y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 2)
        for region in bad_regions:
            x1, y1, x2, y2 = region['bbox']
            color = (0, 0, 255) if region.get('channel') == 'coarse' else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            text = f"{region['area_ratio'] * 100:.1f}%"
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(img, 'Blue: Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
        cv2.putText(img, 'Red: Missing (coarse)', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, 'Green: Missing (fine)', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(output_path, img)
        self._log(f'保存可视化结果: {output_path}')
    def save_uncovered_mask(self, context: ProcessingContext, output_path: str, bad_regions: List[Dict[str, Any]]=None):
        if not context.image_path or not os.path.exists(context.image_path):
            return
        img = cv2.imread(context.image_path)
        if img is None:
            return
        h, w = img.shape[:2]
        result = img.copy()
        overlay = img.copy()
        for elem in context.elements:
            x1 = max(0, min(w, elem.bbox.x1))
            y1 = max(0, min(h, elem.bbox.y1))
            x2 = max(0, min(w, elem.bbox.x2))
            y2 = max(0, min(h, elem.bbox.y2))
            elem_type = elem.element_type.lower()
            is_image_type = elem_type in self.IMAGE_CONTENT_TYPES
            if elem.base64 is not None:
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 200, 0), 2)
            elif is_image_type:
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 200, 255), 2)
        if bad_regions:
            for i, region in enumerate(bad_regions):
                bbox = region['bbox']
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(w, x1))
                y1 = max(0, min(h, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 4)
                area_pct = region.get('area_ratio', 0) * 100
                label = f'BAD #{i + 1} ({area_pct:.1f}%)'
                cv2.putText(result, label, (x1 + 5, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        alpha = 0.25
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
        legend_y = 40
        cv2.putText(result, f'GREEN: elements with base64 (OK)', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
        cv2.putText(result, f'YELLOW: image-type without base64 (PROBLEM)', (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        cv2.putText(result, f'RED: detected bad regions for fallback ({len(bad_regions or [])})', (10, legend_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.imwrite(output_path, result)
        self._log(f'保存问题区域可视化: {output_path}')
def evaluate_result(elements: List[ElementInfo], image_path: str, canvas_width: int=0, canvas_height: int=0, config: Dict=None) -> Dict[str, Any]:
    evaluator = MetricEvaluator(config)
    context = ProcessingContext(image_path=image_path, elements=elements, canvas_width=canvas_width, canvas_height=canvas_height)
    result = evaluator.process(context)
    return result.metadata
def compute_content_coverage(image_path: str, bboxes: List[List[int]], content_threshold: int=245) -> Dict[str, float]:
    img = cv2.imread(image_path)
    if img is None:
        return {'coverage': 0, 'missing': 100}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    content_mask = (gray < content_threshold).astype(np.uint8)
    total_content = np.sum(content_mask > 0)
    if total_content == 0:
        return {'coverage': 100, 'missing': 0}
    covered_mask = np.zeros((h, w), dtype=np.uint8)
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, x2 = (max(0, x1), min(w, x2))
        y1, y2 = (max(0, y1), min(h, y2))
        if x2 > x1 and y2 > y1:
            covered_mask[y1:y2, x1:x2] = 1
    covered_content = np.sum(np.logical_and(content_mask, covered_mask))
    coverage = covered_content / total_content * 100
    return {'coverage': round(coverage, 2), 'missing': round(100 - coverage, 2)}
def compare_with_rendered(original_path: str, rendered_path: str, config: Dict=None) -> Dict[str, Any]:
    default_config = {'diff_threshold': 30, 'min_region_area': 500, 'merge_distance': 20, 'output_path': None}
    cfg = {**default_config, **(config or {})}
    original = cv2.imread(original_path)
    rendered = cv2.imread(rendered_path)
    if original is None or rendered is None:
        return {'overall_similarity': 0, 'missing_regions': [], 'error': '无法读取图像'}
    if original.shape != rendered.shape:
        rendered = cv2.resize(rendered, (original.shape[1], original.shape[0]))
    h, w = original.shape[:2]
    total_area = h * w
    diff = cv2.absdiff(original, rendered)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, diff_mask = cv2.threshold(diff_gray, cfg['diff_threshold'], 255, cv2.THRESH_BINARY)
    kernel_size = cfg['merge_distance']
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    missing_regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg['min_region_area']:
            continue
        x, y, rw, rh = cv2.boundingRect(cnt)
        region_diff = diff_gray[y:y + rh, x:x + rw]
        diff_intensity = np.mean(region_diff)
        missing_regions.append({'bbox': [x, y, x + rw, y + rh], 'area': int(rw * rh), 'area_ratio': rw * rh / total_area, 'diff_intensity': float(diff_intensity), 'description': f'渲染差异区域 ({rw}x{rh})'})
    diff_pixels = np.count_nonzero(diff_mask)
    similarity = max(0, 100 - diff_pixels / total_area * 100)
    output_path = cfg.get('output_path')
    if output_path:
        vis = original.copy()
        for region in missing_regions:
            x1, y1, x2, y2 = region['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(output_path, vis)
    return {'overall_similarity': round(similarity, 2), 'missing_regions': missing_regions, 'diff_pixels': int(diff_pixels)}
def detect_missing_from_rendered_diff(original_path: str, rendered_path: str, output_dir: str=None) -> List[Dict]:
    import base64
    from io import BytesIO
    result = compare_with_rendered(original_path, rendered_path, {'diff_threshold': 25, 'min_region_area': 300, 'merge_distance': 15})
    if not result.get('missing_regions'):
        return []
    original_pil = Image.open(original_path).convert('RGB')
    missing_elements = []
    for i, region in enumerate(result['missing_regions']):
        x1, y1, x2, y2 = region['bbox']
        cropped = original_pil.crop((x1, y1, x2, y2))
        buffer = BytesIO()
        cropped.save(buffer, format='PNG')
        b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        elem = {'bbox': region['bbox'], 'area': region['area'], 'area_ratio': region['area_ratio'], 'diff_intensity': region['diff_intensity'], 'cropped_image': cropped, 'base64': b64_data, 'description': region['description']}
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            crop_path = os.path.join(output_dir, f'missing_region_{i}.png')
            cropped.save(crop_path)
            elem['saved_path'] = crop_path
        missing_elements.append(elem)
    return missing_elements
