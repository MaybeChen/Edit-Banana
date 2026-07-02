"""
任务3：基本图形处理模块

功能：
    - 处理rectangle、ellipse、diamond等基本图形
    - 从图片中提取填充色和描边色
    - 检测边框宽度
    - 用XML描述这些图形
    - 支持CV补充检测（检测SAM3遗漏的矩形/容器）
    - 输出XML片段

负责人：[已实现]
负责任务：任务3 - 基本图形类（取色，用XML描述）

使用示例：
    from modules import BasicShapeProcessor, ProcessingContext
    
    processor = BasicShapeProcessor()
    context = ProcessingContext(image_path="test.png")
    context.elements = [...]  # 从SAM3获取的元素
    
    result = processor.process(context)
    # 处理后的元素会包含 fill_color, stroke_color, xml_fragment 字段

接口说明：
    输入：
        - context.elements: ElementInfo列表，筛选出基本图形
        - context.image_path: 原始图片路径，用于取色
        
    输出：
        - 更新 element.fill_color: 填充颜色（十六进制）
        - 更新 element.stroke_color: 描边颜色（十六进制）
        - 更新 element.stroke_width: 描边宽度
        - 更新 element.xml_fragment: 该元素的XML片段
"""

import os
import cv2
import numpy as np
import math
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image

from .base import BaseProcessor, ProcessingContext
from .data_types import ElementInfo, BoundingBox, ProcessingResult, LayerLevel, get_layer_level


# ======================== DrawIO样式配置 ========================
DRAWIO_STYLES = {
    "rectangle": "rounded=0;whiteSpace=wrap;html=1;",
    "rounded rectangle": "rounded=1;whiteSpace=wrap;html=1;",
    "title_bar": "rounded=0;whiteSpace=wrap;html=1;fillColor=#E6E6E6;",
    "section_panel": "rounded=0;whiteSpace=wrap;html=1;dashed=1;dashPattern=1 1;",
    "container": "rounded=1;whiteSpace=wrap;html=1;",
    "diamond": "rhombus;whiteSpace=wrap;html=1;",
    "ellipse": "ellipse;whiteSpace=wrap;html=1;",
    "circle": "ellipse;whiteSpace=wrap;html=1;",
    "cylinder": "shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=15;",
    "cloud": "ellipse;shape=cloud;whiteSpace=wrap;html=1;",
    "actor": "shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;html=1;outlineConnect=0;",
    "hexagon": "shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;fixedSize=1;",
    "triangle": "triangle;whiteSpace=wrap;html=1;",
    "parallelogram": "shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;fixedSize=1;",
    "trapezoid": "shape=trapezoid;perimeter=trapezoidPerimeter;whiteSpace=wrap;html=1;fixedSize=1;",
    "square": "rounded=0;whiteSpace=wrap;html=1;aspect=fixed;",
}

# 支持矢量化的图形类型
VECTOR_TYPES = {
    "rectangle", "rounded_rectangle", "rounded rectangle",
    "diamond", "ellipse", "circle",
    "cylinder", "cloud", "actor",
    "hexagon", "triangle", "parallelogram",
    "title_bar", "section_panel", "container",
    "trapezoid", "square"
}


# ======================== 几何参数提取 ========================
def _merge_nearby_lines(lines, threshold=10):
    """
    合并相近的平行线段，减少冗余
    
    Args:
        lines: 线段列表，格式为 [(y, x1, x2), ...] 或 [(x, y1, y2), ...]
        threshold: 合并阈值，位置差小于此值的线段会被合并
        
    Returns:
        合并后的线段列表
    """
    if not lines:
        return []
    
    merged = []
    used = set()
    
    for i, line in enumerate(lines):
        if i in used:
            continue
        
        pos, start, end = line  # y/x, x1/y1, x2/y2
        # 找到所有相近的线段
        group_pos = [pos]
        group_start = [start]
        group_end = [end]
        
        for j, other in enumerate(lines[i+1:], i+1):
            if j in used:
                continue
            o_pos, o_start, o_end = other
            if abs(o_pos - pos) < threshold:
                group_pos.append(o_pos)
                group_start.append(o_start)
                group_end.append(o_end)
                used.add(j)
        
        # 合并为一条线
        merged.append((
            int(np.mean(group_pos)),
            min(group_start),
            max(group_end)
        ))
        used.add(i)
    
    return merged


# ======================== CV结果验证 ========================
def _validate_cv_rectangle(cv2_image: np.ndarray, bbox: list, min_std: float = 8) -> bool:
    """
    验证CV检测到的矩形是否有效
    
    检查内容：
    1. 内部颜色是否有足够变化（排除纯色背景误检）
    2. 边框与内部是否有明显区别
    
    :param cv2_image: BGR图像
    :param bbox: [x1, y1, x2, y2]
    :param min_std: 最小颜色标准差
    :return: True=有效, False=可能是误检
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = cv2_image.shape[:2]
    
    # 边界检查
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    if x2 - x1 < 20 or y2 - y1 < 20:
        return False
    
    roi = cv2_image[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 检查1：内部是否有足够的颜色变化
    roi_h, roi_w = gray_roi.shape
    margin = max(3, min(roi_w, roi_h) // 10)
    
    if roi_h > 2 * margin and roi_w > 2 * margin:
        inner = gray_roi[margin:-margin, margin:-margin]
        inner_std = np.std(inner)
        
        # 如果内部颜色太均匀，可能是误检的背景区域
        if inner_std < min_std:
            return False
    
    # 检查2：边框与内部是否有对比度
    border_size = max(2, min(roi_w, roi_h) // 20)
    
    if roi_h > 2 * border_size and roi_w > 2 * border_size:
        border_top = gray_roi[:border_size, :].mean()
        border_bottom = gray_roi[-border_size:, :].mean()
        border_left = gray_roi[:, :border_size].mean()
        border_right = gray_roi[:, -border_size:].mean()
        border_mean = np.mean([border_top, border_bottom, border_left, border_right])
        
        inner_region = gray_roi[border_size:-border_size, border_size:-border_size]
        inner_mean = inner_region.mean()
        
        contrast = abs(border_mean - inner_mean)
        
        # 边框和内部需要有一定对比度
        if contrast < 5:
            return False
    
    return True


# ======================== CV矩形检测 ========================
def detect_rectangles_robust(cv2_image: np.ndarray, existing_elements: dict, config: dict = None) -> dict:
    """
    精准矩形检测（补充SAM3遗漏的矩形）
    
    采用保守策略：
    - 默认只启用可靠的检测方法（contour, nested_contour）
    - 提高检测门槛减少误检
    - 对检测结果进行内容验证
    
    :param cv2_image: BGR格式的OpenCV图像
    :param existing_elements: SAM3已识别的元素字典
    :param config: 配置参数字典
    :return: {"rectangles": [...], "containers": [...]}
    """
    default_config = {
        # 面积限制（提高门槛减少误检）
        "min_area": 5000,            # 提高最小面积（原3000）
        "min_area_ratio": 0.005,     # 最小面积占比
        "max_area_ratio": 0.5,
        
        # 去重阈值（更积极去重）
        "iou_threshold": 0.2,        # 降低IoU阈值（原0.3）
        "nms_threshold": 0.25,       # 降低NMS阈值（原0.3）
        
        # 形状验证（提高要求）
        "min_rectangularity": 0.7,   # 提高矩形度（原0.6）
        "border_contrast": 15,       # 提高边框对比度（原10）
        
        # 容器检测
        "container_threshold": 0.8,
        "min_contained": 3,
        
        # 启用的检测方法（保守模式：只启用可靠的方法）
        "enabled_methods": ["contour", "nested_contour"],
        # 完整模式可用: ["contour", "region", "low_contrast", "hough_lines", "nested_contour"]
        
        # 内容验证（CV结果需要通过验证）
        "validate_content": True,
        "min_content_std": 8,        # 内部颜色标准差阈值
    }
    cfg = {**default_config, **(config or {})}
    
    enabled_methods = set(cfg.get("enabled_methods", ["contour", "nested_contour"]))
    
    h, w = cv2_image.shape[:2]
    total_area = h * w
    max_area = total_area * cfg["max_area_ratio"]
    min_area = max(cfg["min_area"], int(total_area * cfg.get("min_area_ratio", 0)))
    
    # 收集SAM3已检测的bbox
    sam3_bboxes = []
    for elem_type, items in existing_elements.items():
        for item in items:
            sam3_bboxes.append({"bbox": item["bbox"], "type": elem_type})
    
    all_candidates = []
    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    
    # 方法1：边缘轮廓检测（最可靠的方法）
    if "contour" in enabled_methods:
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            if peri < 100:
                continue
            
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if not (4 <= len(approx) <= 8):
                continue
            
            x, y, rw, rh = cv2.boundingRect(approx)
            bbox = [x, y, x+rw, y+rh]
            area = rw * rh
            
            if area < min_area or area > max_area:
                continue
            
            aspect = max(rw, rh) / max(1, min(rw, rh))
            if aspect > 4:
                continue
            
            cnt_area = cv2.contourArea(approx)
            rectangularity = cnt_area / area if area > 0 else 0
            if rectangularity < cfg["min_rectangularity"]:
                continue
            
            # 验证边框线
            border_w = max(3, min(8, rw // 15, rh // 15))
            
            if rw > 2 * border_w and rh > 2 * border_w:
                roi = gray[y:y+rh, x:x+rw]
                
                border_top = roi[:border_w, :].flatten()
                border_bottom = roi[-border_w:, :].flatten()
                border_left = roi[:, :border_w].flatten()
                border_right = roi[:, -border_w:].flatten()
                border_pixels = np.concatenate([border_top, border_bottom, border_left, border_right])
                
                inner = roi[border_w:-border_w, border_w:-border_w].flatten()
                
                if len(inner) > 0 and len(border_pixels) > 0:
                    border_mean = np.mean(border_pixels)
                    inner_mean = np.mean(inner)
                    contrast = abs(border_mean - inner_mean)
                    
                    if contrast < cfg["border_contrast"]:
                        continue
            
            is_rounded = rectangularity < 0.98
            
            all_candidates.append({
                "bbox": bbox,
                "area": area,
                "method": "contour",
                "score": rectangularity,
                "rectangularity": rectangularity,
                "is_rounded": is_rounded
            })
    
    # 方法2：区域颜色检测（容易误检，默认禁用）
    if "region" in enabled_methods:
        hsv = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2HSV)
        lower_gray = np.array([0, 0, 180])
        upper_gray = np.array([180, 50, 252])
        
        mask_region = cv2.inRange(hsv, lower_gray, upper_gray)
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close_region = np.ones((7, 7), np.uint8)
        
        mask_region = cv2.morphologyEx(mask_region, cv2.MORPH_OPEN, kernel_open)
        mask_region = cv2.morphologyEx(mask_region, cv2.MORPH_CLOSE, kernel_close_region)
        
        contours_region, _ = cv2.findContours(mask_region, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours_region:
            peri = cv2.arcLength(cnt, True)
            if peri < 100:
                continue
            
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if not (4 <= len(approx) <= 12):
                continue
                
            x, y, rw, rh = cv2.boundingRect(approx)
            bbox = [x, y, x+rw, y+rh]
            area = rw * rh
            
            if area < min_area or area > max_area:
                continue
                
            if max(rw, rh) / max(1, min(rw, rh)) > 5:
                continue
                
            cnt_area = cv2.contourArea(approx)
            if area > 0 and cnt_area / area < 0.6:
                continue
                
            rect_ratio = cnt_area / area if area > 0 else 0
            is_rounded_region = False
            if 0.85 <= rect_ratio < 0.96:
                is_rounded_region = True
            elif len(approx) > 4 and rect_ratio < 0.96:
                is_rounded_region = True
                
            all_candidates.append({
                "bbox": bbox,
                "area": area,
                "method": "region",
                "score": rect_ratio,
                "rectangularity": rect_ratio,
                "is_rounded": is_rounded_region
            })

    # 方法3：低对比度框检测（容易误检，默认禁用，通过 enabled_methods 过滤）
    edges_low = cv2.Canny(gray, 10, 50)
    edges_low = cv2.dilate(edges_low, np.ones((3, 3), np.uint8), iterations=2)
    
    kernel_close = np.ones((5, 5), np.uint8)
    edges_closed = cv2.morphologyEx(edges_low, cv2.MORPH_CLOSE, kernel_close)
    
    contours_low, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours_low:
        peri = cv2.arcLength(cnt, True)
        if peri < 100:
            continue
        
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        
        if not (4 <= len(approx) <= 8):
            continue
        
        x, y, rw, rh = cv2.boundingRect(approx)
        bbox = [x, y, x+rw, y+rh]
        area = rw * rh
        
        max_area_expanded = total_area * 0.8
        if area < min_area or area > max_area_expanded:
            continue
        
        aspect = max(rw, rh) / max(1, min(rw, rh))
        if aspect > 5:
            continue
        
        cnt_area = cv2.contourArea(approx)
        rectangularity = cnt_area / area if area > 0 else 0
        if rectangularity < 0.55:
            continue
        
        # 浅色背景检查
        color_check_passed = False
        
        if rw > 15 and rh > 15:
            roi_bgr = cv2_image[y:y+rh, x:x+rw]
            roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            
            margin = max(5, min(rw // 12, rh // 12))
            if rw > 2 * margin and rh > 2 * margin:
                center_hsv = roi_hsv[margin:-margin, margin:-margin]
                s_channel = center_hsv[:, :, 1]
                median_saturation = np.median(s_channel)
                
                v_channel = center_hsv[:, :, 2]
                median_value = np.median(v_channel)
                
                if median_saturation < 75 and median_value > 150:
                    color_check_passed = True
        
        if not color_check_passed:
            continue
        
        is_rounded = rectangularity < 0.92
        
        all_candidates.append({
            "bbox": bbox,
            "area": area,
            "method": "low_contrast_gray",
            "score": rectangularity,
            "rectangularity": rectangularity,
            "is_rounded": is_rounded
        })
    
    # 方法4：霍夫线检测（检测虚线框、表格线等）
    edges_hough = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges_hough, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    if lines is not None:
        # 分类水平线和垂直线
        h_lines = []
        v_lines = []
        
        for line in lines:
            x1_l, y1_l, x2_l, y2_l = line[0]
            angle = np.arctan2(y2_l - y1_l, x2_l - x1_l) * 180 / np.pi
            length = np.sqrt((x2_l - x1_l)**2 + (y2_l - y1_l)**2)
            
            if length < 30:
                continue
            
            if abs(angle) < 15 or abs(angle) > 165:
                h_lines.append((min(y1_l, y2_l), min(x1_l, x2_l), max(x1_l, x2_l)))
            elif 75 < abs(angle) < 105:
                v_lines.append((min(x1_l, x2_l), min(y1_l, y2_l), max(y1_l, y2_l)))
        
        # 尝试从线段重建矩形
        if len(h_lines) >= 2 and len(v_lines) >= 2:
            # 优化：合并相近线段，减少冗余
            h_lines = _merge_nearby_lines(h_lines, threshold=15)
            v_lines = _merge_nearby_lines(v_lines, threshold=15)
            
            # 优化：限制线段数量，控制复杂度上限
            MAX_LINES = 30
            h_lines = h_lines[:MAX_LINES]
            v_lines = v_lines[:MAX_LINES]
            
            h_lines.sort(key=lambda x: x[0])
            v_lines.sort(key=lambda x: x[0])
            
            tolerance = 15
            
            # 优化：提前终止，找到足够候选后停止
            MAX_HOUGH_CANDIDATES = 50
            hough_found = 0
            
            for i, h_top in enumerate(h_lines):
                if hough_found >= MAX_HOUGH_CANDIDATES:
                    break
                for h_bottom in h_lines[i+1:]:
                    if hough_found >= MAX_HOUGH_CANDIDATES:
                        break
                    rect_height = h_bottom[0] - h_top[0]
                    if rect_height < 30:
                        continue
                    
                    for j, v_left in enumerate(v_lines):
                        if hough_found >= MAX_HOUGH_CANDIDATES:
                            break
                        for v_right in v_lines[j+1:]:
                            if hough_found >= MAX_HOUGH_CANDIDATES:
                                break
                            rect_width = v_right[0] - v_left[0]
                            if rect_width < 30:
                                continue
                            
                            # 检查四条边是否能形成矩形
                            h_top_valid = (h_top[1] <= v_left[0] + tolerance and 
                                          h_top[2] >= v_right[0] - tolerance)
                            h_bottom_valid = (h_bottom[1] <= v_left[0] + tolerance and 
                                             h_bottom[2] >= v_right[0] - tolerance)
                            v_left_valid = (v_left[1] <= h_top[0] + tolerance and 
                                           v_left[2] >= h_bottom[0] - tolerance)
                            v_right_valid = (v_right[1] <= h_top[0] + tolerance and 
                                            v_right[2] >= h_bottom[0] - tolerance)
                            
                            if h_top_valid and h_bottom_valid and v_left_valid and v_right_valid:
                                bbox_h = [v_left[0], h_top[0], v_right[0], h_bottom[0]]
                                area_h = rect_width * rect_height
                                
                                if min_area <= area_h <= max_area:
                                    all_candidates.append({
                                        "bbox": bbox_h,
                                        "area": area_h,
                                        "method": "hough_lines",
                                        "score": 0.8,
                                        "rectangularity": 0.9,
                                        "is_rounded": False
                                    })
                                    hough_found += 1
    
    # 方法5：嵌套轮廓检测（检测容器框）
    _, binary_nest = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_nest, hierarchy_nest = cv2.findContours(binary_nest, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy_nest is not None:
        hierarchy_nest = hierarchy_nest[0]
        
        for idx, cnt in enumerate(contours_nest):
            # 检查是否有子轮廓
            if hierarchy_nest[idx][2] == -1:  # 没有子轮廓
                continue
            
            # 计算子轮廓数量
            child_count = 0
            child_idx = hierarchy_nest[idx][2]
            while child_idx != -1:
                child_count += 1
                child_idx = hierarchy_nest[child_idx][0]
            
            if child_count < 2:  # 至少包含2个子元素
                continue
            
            peri = cv2.arcLength(cnt, True)
            if peri < 200:
                continue
            
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
            if not (4 <= len(approx) <= 10):
                continue
            
            x, y, rw, rh = cv2.boundingRect(approx)
            bbox_n = [x, y, x+rw, y+rh]
            area_n = rw * rh
            
            if area_n < min_area * 2 or area_n > max_area:  # 容器通常较大
                continue
            
            aspect_n = max(rw, rh) / max(1, min(rw, rh))
            if aspect_n > 4:
                continue
            
            cnt_area_n = cv2.contourArea(approx)
            rectangularity_n = cnt_area_n / area_n if area_n > 0 else 0
            
            if rectangularity_n < 0.5:
                continue
            
            all_candidates.append({
                "bbox": bbox_n,
                "area": area_n,
                "method": "nested_contour",
                "score": rectangularity_n,
                "rectangularity": rectangularity_n,
                "is_rounded": rectangularity_n < 0.9,
                "child_count": child_count
            })
    
    # 按方法过滤（只保留启用的方法的结果）
    method_mapping = {
        "contour": "contour",
        "region": "region", 
        "low_contrast_gray": "low_contrast",
        "hough_lines": "hough_lines",
        "nested_contour": "nested_contour"
    }
    
    all_candidates = [
        cand for cand in all_candidates 
        if method_mapping.get(cand["method"], cand["method"]) in enabled_methods
    ]
    
    # NMS去重
    all_candidates.sort(key=lambda x: x["area"], reverse=True)
    
    filtered_candidates = []
    validate_content = cfg.get("validate_content", True)
    min_content_std = cfg.get("min_content_std", 8)
    
    for cand in all_candidates:
        bbox = cand["bbox"]
        
        # 内容验证（过滤误检的背景区域）
        if validate_content:
            if not _validate_cv_rectangle(cv2_image, bbox, min_std=min_content_std):
                continue
        
        # 与SAM3结果对比
        is_dup_sam3 = False
        for sam3_item in sam3_bboxes:
            iou = calculate_iou(bbox, sam3_item["bbox"])
            if iou > cfg["iou_threshold"]:
                is_dup_sam3 = True
                break
        if is_dup_sam3:
            continue
        
        # NMS
        is_dup_nms = False
        for existing in filtered_candidates:
            iou = calculate_iou(bbox, existing["bbox"])
            if iou > cfg["nms_threshold"]:
                is_dup_nms = True
                break
        if is_dup_nms:
            continue
        
        filtered_candidates.append(cand)
    
    # 自动分层（判断谁是容器）
    all_bboxes_for_contain = [item["bbox"] for item in sam3_bboxes] + [c["bbox"] for c in filtered_candidates]
    
    for cand in filtered_candidates:
        x1, y1, x2, y2 = cand["bbox"]
        contained_count = 0
        
        for other_bbox in all_bboxes_for_contain:
            if other_bbox == cand["bbox"]:
                continue
            ox1, oy1, ox2, oy2 = other_bbox
            if x1 <= ox1 and y1 <= oy1 and x2 >= ox2 and y2 >= oy2:
                contained_count += 1
            elif calculate_iou(cand["bbox"], other_bbox) > 0:
                inter_x1 = max(x1, ox1)
                inter_y1 = max(y1, oy1)
                inter_x2 = min(x2, ox2)
                inter_y2 = min(y2, oy2)
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    other_area = (ox2 - ox1) * (oy2 - oy1)
                    if other_area > 0 and inter_area / other_area > cfg["container_threshold"]:
                        contained_count += 1
        
        cand["contained_count"] = contained_count
        cand["is_container"] = contained_count >= cfg["min_contained"]
    
    # 颜色提取
    rectangles = []
    containers = []
    
    for cand in filtered_candidates:
        x1, y1, x2, y2 = cand["bbox"]
        rw, rh = x2 - x1, y2 - y1
        
        # 填充色提取
        margin_x = max(3, int(rw * 0.25))
        margin_y = max(3, int(rh * 0.25))
        inner_x1, inner_y1 = x1 + margin_x, y1 + margin_y
        inner_x2, inner_y2 = x2 - margin_x, y2 - margin_y
        
        if inner_x2 > inner_x1 and inner_y2 > inner_y1:
            inner_roi = cv2_image[inner_y1:inner_y2, inner_x1:inner_x2]
            inner_rgb = cv2.cvtColor(inner_roi, cv2.COLOR_BGR2RGB)
            pixels = inner_rgb.reshape(-1, 3).astype(np.float32)
            
            if len(pixels) > 10:
                try:
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                    _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
                    counts = np.bincount(labels.flatten())
                    dominant_idx = np.argmax(counts)
                    fill_rgb = centers[dominant_idx].astype(int)
                except:
                    fill_rgb = np.median(pixels, axis=0).astype(int)
            else:
                fill_rgb = np.median(pixels, axis=0).astype(int) if len(pixels) > 0 else np.array([255, 255, 255])
            
            fill_color = "#{:02x}{:02x}{:02x}".format(*np.clip(fill_rgb, 0, 255))
        else:
            fill_color = "#ffffff"
        
        # CV检测结果强制使用黑色细边框（风格统一）
        stroke_color = "#000000"
        
        result_item = {
            "bbox": cand["bbox"],
            "area": cand["area"],
            "fill_color": fill_color,
            "stroke_color": stroke_color,
            "score": cand["score"],
            "method": cand["method"],
            "contained_count": cand.get("contained_count", 0),
            "is_rounded": cand.get("is_rounded", False)
        }
        
        if cand["is_container"]:
            containers.append(result_item)
        else:
            rectangles.append(result_item)
    
    return {
        "rectangles": rectangles,
        "containers": containers
    }
