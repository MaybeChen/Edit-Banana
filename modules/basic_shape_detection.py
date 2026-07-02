"""
任务3：基本图形处理模块

功能：
    - 处理rectangle、ellipse、diamond等基本图形
    - 从图片中提取填充色和描边色
    - 检测边框宽度
    - 用XML描述这些图形
    - 支持CV补充检测（检测segmentation遗漏的矩形/容器）
    - 输出XML片段

负责人：[已实现]
负责任务：任务3 - 基本图形类（取色，用XML描述）

使用示例：
    from modules import BasicShapeProcessor, ProcessingContext
    
    processor = BasicShapeProcessor()
    context = ProcessingContext(image_path="test.png")
    context.elements = [...]  # 从segmentation获取的元素
    
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
def extract_geometric_params(image: np.ndarray, bbox: list, shape_type: str) -> dict:
    """
    针对特定形状提取几何参数（如平行四边形的倾斜度、圆柱体的顶部高度等）。
    返回参数字典，例如 {"size": 0.2, "direction": "south"}
    """
    params = {}
    x1, y1, x2, y2 = map(int, bbox)
    w_box, h_box = x2 - x1, y2 - y1
    
    if w_box <= 0 or h_box <= 0:
        return params

    # 提取 ROI 用于分析
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 通用预处理：获取轮廓
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    main_cnt = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            main_cnt = cnt
            
    if main_cnt is None:
        return params

    # 针对不同形状的分析
    if shape_type == "parallelogram":
        # 计算倾斜比例 size (0~1)
        epsilon = 0.02 * cv2.arcLength(main_cnt, True)
        approx = cv2.approxPolyDP(main_cnt, epsilon, True)
        
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            pts = pts[np.argsort(pts[:, 1])]
            top_pts = pts[:2]
            bottom_pts = pts[2:]
            
            top_pts = top_pts[np.argsort(top_pts[:, 0])]
            bottom_pts = bottom_pts[np.argsort(bottom_pts[:, 0])]
            
            tl, tr = top_pts[0], top_pts[1]
            bl, br = bottom_pts[0], bottom_pts[1]
            
            dx = abs(tl[0] - bl[0])
            size_val = dx / w_box if w_box > 0 else 0.2
            params["size"] = max(0.05, min(0.5, size_val))
            
    elif shape_type == "cylinder":
        params["size"] = max(10, int(w_box * 0.15))
        
    elif shape_type == "triangle":
        epsilon = 0.04 * cv2.arcLength(main_cnt, True)
        approx = cv2.approxPolyDP(main_cnt, epsilon, True)
        
        if len(approx) == 3:
            M = cv2.moments(main_cnt)
            if M["m00"] != 0:
                cy = int(M["m01"] / M["m00"])
                rel_cy = cy / h_box
                if rel_cy > 0.55:
                    params["direction"] = "north"
                elif rel_cy < 0.45:
                    params["direction"] = "south"
                else:
                    cx = int(M["m10"] / M["m00"])
                    rel_cx = cx / w_box
                    if rel_cx > 0.55:
                        params["direction"] = "west"
                    elif rel_cx < 0.45:
                        params["direction"] = "east"

    return params


# ======================== IoU计算 ========================
def calculate_iou(box1: list, box2: list) -> float:
    """计算两个矩形框的 IoU"""
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area


# ======================== 边框宽度检测 ========================
def calculate_stroke_width(image: np.ndarray, bbox: list, max_width: int = 8) -> int:
    """
    计算边框粗细 (Stroke Width)
    逻辑：沿四边向内扫描，寻找颜色突变点，多个采样点综合取中位数。
    
    优化：
    - 提高突变阈值（35），减少误检
    - 限制最大宽度（8像素），避免过粗
    - 大多数边框在 1-5 像素
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return 1
    
    roi_h, roi_w = roi.shape[:2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    scan_limit = min(max_width, roi_w // 2 - 1, roi_h // 2 - 1)
    if scan_limit < 1:
        return 1
    
    detected_widths = []
    
    def scan_line(pixels, limit):
        if len(pixels) < limit + 1:
            return None
        diffs = np.abs(np.diff(pixels[:limit+2].astype(int)))
        threshold = 35  # 提高阈值，减少误检（原值20太敏感）
        candidates = np.where(diffs > threshold)[0]
        if len(candidates) > 0:
            return candidates[0] + 1
        return None

    num_samples = 5
    
    # Top Edge
    for i in range(1, num_samples + 1):
        x = int(roi_w * i / (num_samples + 1))
        col = roi_gray[:, x]
        w_val = scan_line(col, scan_limit)
        if w_val:
            detected_widths.append(w_val)
        
    # Bottom Edge
    for i in range(1, num_samples + 1):
        x = int(roi_w * i / (num_samples + 1))
        col = roi_gray[::-1, x]
        w_val = scan_line(col, scan_limit)
        if w_val:
            detected_widths.append(w_val)
        
    # Left Edge
    for i in range(1, num_samples + 1):
        y = int(roi_h * i / (num_samples + 1))
        row = roi_gray[y, :]
        w_val = scan_line(row, scan_limit)
        if w_val:
            detected_widths.append(w_val)

    # Right Edge
    for i in range(1, num_samples + 1):
        y = int(roi_h * i / (num_samples + 1))
        row = roi_gray[y, ::-1]
        w_val = scan_line(row, scan_limit)
        if w_val:
            detected_widths.append(w_val)
        
    if not detected_widths:
        return 1
        
    final_width = int(np.median(detected_widths))
    # 限制合理范围：大多数边框在 1-2 像素（降低上限以匹配原图）
    return max(1, min(final_width, 2))


# ======================== 颜色提取 ========================
def extract_style_colors(image: np.ndarray, bbox: list) -> tuple:
    """
    精细化取色逻辑：区分 边框区域(Stroke) 和 内部区域(Fill)
    
    优化策略：
    1. Fill: 采样边框内侧的"回"字形区域（避开中心可能的文字），使用K-Means聚类找主色
    2. Stroke: 提取边界框外围10%区域，取最暗的25%像素的均值作为边框色
    3. 同时返回检测到的边框宽度
    
    :param image: BGR格式的OpenCV图像
    :param bbox: [x1, y1, x2, y2]
    :return: (fill_color_hex, stroke_color_hex, stroke_width)
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_box, w_box = y2 - y1, x2 - x1
    
    # 截取ROI
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return "#ffffff", "#000000", 1
    
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    
    # --- 0. 检测边框宽度 ---
    max_w = min(15, w_box // 5, h_box // 5)
    stroke_width = calculate_stroke_width(image, bbox, max_width=max(1, max_w))
    
    # --- 1. 提取填充色 (Fill Color) ---
    # 优化：采样边框内侧的"回"字形区域，避开中心可能存在的文字
    s_w = int(stroke_width)
    border_padding = max(2, s_w + 2)
    sample_depth = max(5, min(20, w_box // 10, h_box // 10))
    
    fill_samples = []
    
    if w_box > 2 * (border_padding + sample_depth) and h_box > 2 * (border_padding + sample_depth):
        # Top strip (边框内侧上方)
        fill_samples.append(roi_rgb[border_padding:border_padding+sample_depth, border_padding:w_box-border_padding])
        # Bottom strip (边框内侧下方)
        fill_samples.append(roi_rgb[h_box-border_padding-sample_depth:h_box-border_padding, border_padding:w_box-border_padding])
        # Left strip (边框内侧左侧)
        fill_samples.append(roi_rgb[border_padding:h_box-border_padding, border_padding:border_padding+sample_depth])
        # Right strip (边框内侧右侧)
        fill_samples.append(roi_rgb[border_padding:h_box-border_padding, w_box-border_padding-sample_depth:w_box-border_padding])
    else:
        # Fallback: 区域太小，取中心区域
        margin_x = min(int(stroke_width + 2), w_box // 2 - 1)
        margin_y = min(int(stroke_width + 2), h_box // 2 - 1)
        if margin_x > 0 and margin_y > 0:
            fill_samples.append(roi_rgb[margin_y:h_box-margin_y, margin_x:w_box-margin_x])
        else:
            fill_samples.append(roi_rgb)

    # 合并所有采样像素
    if fill_samples:
        valid_samples = [s.reshape(-1, 3) for s in fill_samples if s.size > 0]
        if valid_samples:
            inner_pixels = np.concatenate(valid_samples)
        else:
            inner_pixels = roi_rgb.reshape(-1, 3)
    else:
        inner_pixels = roi_rgb.reshape(-1, 3)

    if inner_pixels.size == 0:
        inner_pixels = roi_rgb.reshape(-1, 3)

    # 使用K-Means聚类找主色（比中位数更准确）
    fill_rgb = np.median(inner_pixels, axis=0).astype(int)  # 默认用中位数
    
    if len(inner_pixels) > 200:
        try:
            # 降采样以提高速度
            if len(inner_pixels) > 2000:
                indices = np.random.choice(len(inner_pixels), 2000, replace=False)
                pixels_for_kmeans = inner_pixels[indices].astype(np.float32)
            else:
                pixels_for_kmeans = inner_pixels.astype(np.float32)
                
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = 2  # 假设背景+前景杂噪
            _, labels, centers = cv2.kmeans(pixels_for_kmeans, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            counts = np.bincount(labels.flatten())
            dominant_idx = np.argmax(counts)
            fill_rgb = centers[dominant_idx].astype(int)
        except:
            pass  # 保持中位数结果
    
    # --- 2. 提取描边色 (Stroke Color) ---
    border_w = max(2, stroke_width)
    
    top = roi_rgb[:border_w, :]
    bottom = roi_rgb[h_box-border_w:, :]
    left = roi_rgb[:, :border_w]
    right = roi_rgb[:, w_box-border_w:]
    
    border_pixels = np.concatenate([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3)
    ], axis=0)
    
    if border_pixels.size > 0:
        # 计算亮度 (Luminance): L = 0.299*R + 0.587*G + 0.114*B
        luminance = np.dot(border_pixels, [0.299, 0.587, 0.114])
        # 提取最暗的 25% 像素 (假设边框比背景深)
        dark_threshold = np.percentile(luminance, 25)
        darker_pixels = border_pixels[luminance <= dark_threshold]
        
        if len(darker_pixels) > 0:
            stroke_rgb = np.mean(darker_pixels, axis=0).astype(int)
        else:
            stroke_rgb = np.mean(border_pixels, axis=0).astype(int)
    else:
        stroke_rgb = np.array([0, 0, 0])

    # RGB -> Hex
    def rgb2hex(rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    # 限制 stroke_width 在合理范围内（1-3），避免过粗的边框
    stroke_width = min(3, max(1, stroke_width))
    
    return rgb2hex(fill_rgb), rgb2hex(stroke_rgb), stroke_width


def extract_style_specific(image: np.ndarray, bbox: list, shape_type: str) -> dict:
    """
    针对不同基础形状的特定取色和边框算法。
    
    - 对于矩形类形状，使用动态边框宽度检测
    - 对于非矩形形状（椭圆、菱形等），使用Mask提取更准确的填充色
    """
    fill_hex, stroke_hex, stroke_w = extract_style_colors(image, bbox)
    
    # 针对非矩形形状，使用 Mask 提取更准确的填充色
    if shape_type in ["ellipse", "cloud", "circle", "diamond", "triangle", "hexagon"]:
        x1, y1, x2, y2 = map(int, bbox)
        h_img, w_img = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        roi = image[y1:y2, x1:x2]
        if roi.size > 0:
            h, w = roi.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            if shape_type in ["ellipse", "cloud", "circle"]:
                cv2.ellipse(mask, (w//2, h//2), (w//2, h//2), 0, 0, 360, 255, -1)
            elif shape_type == "diamond":
                pts = np.array([[w//2, 0], [w, h//2], [w//2, h], [0, h//2]], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            elif shape_type == "triangle":
                pts = np.array([[w//2, 0], [w, h], [0, h]], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            elif shape_type == "hexagon":
                pts = np.array([
                    [w//4, 0], [w*3//4, 0], 
                    [w, h//2], 
                    [w*3//4, h], [w//4, h], 
                    [0, h//2]
                ], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            
            # 腐蚀掉边缘区域
            kernel_size = max(3, stroke_w * 2 + 1)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.erode(mask, kernel)
            
            if cv2.countNonZero(mask) > 0:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                masked_pixels = roi_rgb[mask > 0]
                masked_pixels = masked_pixels.reshape(-1, 3)
                
                if masked_pixels.size > 0:
                    fill_rgb = np.median(masked_pixels, axis=0).astype(int)
                    fill_hex = "#{:02x}{:02x}{:02x}".format(*map(int, fill_rgb))

    geo_params = extract_geometric_params(image, bbox, shape_type)

    return {
        "fill_color": fill_hex,
        "stroke_color": stroke_hex,
        "stroke_width": stroke_w,
        "geo_params": geo_params
    }


# ======================== Mask精确取色 ========================
def extract_color_with_mask(image: np.ndarray, bbox: list, mask: np.ndarray,
                            shape_type: str = "unknown") -> dict:
    """
    使用segmentation提供的Mask进行精确取色
    
    Args:
        image: BGR格式的OpenCV图像
        bbox: [x1, y1, x2, y2] 边界框
        mask: segmentation提供的二值掩码 (full size or cropped)
        shape_type: 形状类型
        
    Returns:
        {
            'fill_color': '#xxxxxx',
            'stroke_color': '#xxxxxx', 
            'stroke_width': int,
            'geo_params': dict,
            'has_gradient': bool,
            'gradient_info': dict or None
        }
    """
    x1, y1, x2, y2 = map(int, bbox)
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    
    roi = image[y1:y2, x1:x2]
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    h_roi, w_roi = roi.shape[:2]
    
    if h_roi == 0 or w_roi == 0:
        return {
            'fill_color': '#ffffff',
            'stroke_color': '#000000',
            'stroke_width': 1,
            'geo_params': {},
            'has_gradient': False,
            'gradient_info': None
        }
    
    # 处理Mask：确保和ROI尺寸一致
    if mask is not None and mask.size > 0:
        # 如果mask是全图尺寸，裁剪到ROI
        if mask.shape[0] == h_img and mask.shape[1] == w_img:
            mask_crop = mask[y1:y2, x1:x2]
        elif mask.shape[0] == h_roi and mask.shape[1] == w_roi:
            mask_crop = mask
        else:
            # 尺寸不匹配，resize
            mask_crop = cv2.resize(mask.astype(np.uint8), (w_roi, h_roi))
        
        # 二值化
        if mask_crop.max() > 1:
            mask_crop = (mask_crop > 127).astype(np.uint8)
        else:
            mask_crop = mask_crop.astype(np.uint8)
    else:
        # 没有mask，创建全1掩码
        mask_crop = np.ones((h_roi, w_roi), dtype=np.uint8)
    
    # =========== 1. 使用Mask精确提取填充色 ===========
    # 腐蚀Mask去除边框区域
    kernel_erode = np.ones((5, 5), np.uint8)
    inner_mask = cv2.erode(mask_crop, kernel_erode, iterations=2)
    
    # 提取内部像素
    if cv2.countNonZero(inner_mask) > 10:
        fill_pixels = roi_rgb[inner_mask > 0]
    else:
        fill_pixels = roi_rgb[mask_crop > 0] if cv2.countNonZero(mask_crop) > 0 else roi_rgb.reshape(-1, 3)
    
    if len(fill_pixels) > 0:
        fill_pixels = fill_pixels.reshape(-1, 3)
        
        # K-Means找主色（比中位数更准确）
        if len(fill_pixels) > 50:
            try:
                pixels_f32 = fill_pixels.astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                k = min(3, len(fill_pixels) // 20)
                k = max(2, k)
                _, labels, centers = cv2.kmeans(pixels_f32, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
                
                # 选择占比最大的颜色
                counts = np.bincount(labels.flatten())
                dominant_idx = np.argmax(counts)
                fill_rgb = centers[dominant_idx].astype(int)
            except:
                fill_rgb = np.median(fill_pixels, axis=0).astype(int)
        else:
            fill_rgb = np.median(fill_pixels, axis=0).astype(int)
    else:
        fill_rgb = np.array([255, 255, 255])
    
    # =========== 2. 使用Mask边缘提取描边色 ===========
    # 获取Mask边缘
    edge_mask = mask_crop - inner_mask
    edge_mask = np.maximum(edge_mask, 0).astype(np.uint8)
    
    # 如果边缘太薄，膨胀一下
    if cv2.countNonZero(edge_mask) < 50:
        kernel_edge = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask_crop, kernel_edge, iterations=1)
        edge_mask = dilated_mask - cv2.erode(mask_crop, kernel_edge, iterations=1)
        edge_mask = np.maximum(edge_mask, 0).astype(np.uint8)
    
    if cv2.countNonZero(edge_mask) > 5:
        stroke_pixels = roi_rgb[edge_mask > 0].reshape(-1, 3)
        
        # 取最暗的像素作为描边色
        if len(stroke_pixels) > 0:
            luminance = np.dot(stroke_pixels, [0.299, 0.587, 0.114])
            dark_threshold = np.percentile(luminance, 30)
            dark_pixels = stroke_pixels[luminance <= dark_threshold]
            
            if len(dark_pixels) > 0:
                stroke_rgb = np.mean(dark_pixels, axis=0).astype(int)
            else:
                stroke_rgb = np.mean(stroke_pixels, axis=0).astype(int)
        else:
            stroke_rgb = np.array([0, 0, 0])
    else:
        stroke_rgb = np.array([0, 0, 0])
    
    # =========== 3. 估算描边宽度 ===========
    # 基于Mask边缘厚度，限制在1-3范围内
    if cv2.countNonZero(edge_mask) > 0:
        # 计算边缘区域的平均厚度
        dist_transform = cv2.distanceTransform(mask_crop, cv2.DIST_L2, 5)
        max_dist = dist_transform.max()
        stroke_width = max(1, min(3, int(max_dist * 0.15)))  # 限制最大为3
    else:
        stroke_width = 1
    
    # =========== 4. 检测渐变 ===========
    has_gradient = False
    gradient_info = None
    
    if len(fill_pixels) > 100:
        # 将填充区域分为上下/左右两半，比较颜色差异
        coords = np.argwhere(inner_mask > 0 if cv2.countNonZero(inner_mask) > 10 else mask_crop > 0)
        if len(coords) > 20:
            mid_y = (coords[:, 0].min() + coords[:, 0].max()) // 2
            mid_x = (coords[:, 1].min() + coords[:, 1].max()) // 2
            
            # 上下分区
            top_coords = coords[coords[:, 0] < mid_y]
            bottom_coords = coords[coords[:, 0] >= mid_y]
            
            if len(top_coords) > 10 and len(bottom_coords) > 10:
                top_colors = roi_rgb[top_coords[:, 0], top_coords[:, 1]]
                bottom_colors = roi_rgb[bottom_coords[:, 0], bottom_coords[:, 1]]
                
                top_mean = np.mean(top_colors, axis=0)
                bottom_mean = np.mean(bottom_colors, axis=0)
                v_diff = np.linalg.norm(top_mean - bottom_mean)
                
                if v_diff > 35:
                    has_gradient = True
                    gradient_info = {
                        'direction': 'vertical',
                        'start_color': "#{:02x}{:02x}{:02x}".format(*top_mean.astype(int).clip(0, 255)),
                        'end_color': "#{:02x}{:02x}{:02x}".format(*bottom_mean.astype(int).clip(0, 255))
                    }
            
            # 左右分区（如果垂直没有渐变）
            if not has_gradient:
                left_coords = coords[coords[:, 1] < mid_x]
                right_coords = coords[coords[:, 1] >= mid_x]
                
                if len(left_coords) > 10 and len(right_coords) > 10:
                    left_colors = roi_rgb[left_coords[:, 0], left_coords[:, 1]]
                    right_colors = roi_rgb[right_coords[:, 0], right_coords[:, 1]]
                    
                    left_mean = np.mean(left_colors, axis=0)
                    right_mean = np.mean(right_colors, axis=0)
                    h_diff = np.linalg.norm(left_mean - right_mean)
                    
                    if h_diff > 35:
                        has_gradient = True
                        gradient_info = {
                            'direction': 'horizontal',
                            'start_color': "#{:02x}{:02x}{:02x}".format(*left_mean.astype(int).clip(0, 255)),
                            'end_color': "#{:02x}{:02x}{:02x}".format(*right_mean.astype(int).clip(0, 255))
                        }
    
    # =========== 5. 提取几何参数 ===========
    geo_params = extract_geometric_params(image, bbox, shape_type)
    
    # 格式化输出
    fill_color = "#{:02x}{:02x}{:02x}".format(*fill_rgb.clip(0, 255))
    stroke_color = "#{:02x}{:02x}{:02x}".format(*stroke_rgb.clip(0, 255))
    
    return {
        'fill_color': fill_color,
        'stroke_color': stroke_color,
        'stroke_width': stroke_width,
        'geo_params': geo_params,
        'has_gradient': has_gradient,
        'gradient_info': gradient_info
    }


# ======================== 样式统一 ========================
def unify_element_styles(elements: list) -> list:
    """
    统一相似大小和类型的基本图形的边框厚度。
    
    注意：参考 segmentation_extractor.py 的简化逻辑，默认边框宽度为1，
    这里主要用于确保同类元素风格一致。
    """
    if not elements:
        return elements

    groups = {}
    
    for i, elem in enumerate(elements):
        shape_type = elem.get("_type", "rectangle")
        bbox = elem["bbox"]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        diag = math.sqrt(w**2 + h**2)
        size_key = int(round(diag / 20))
        
        key = (shape_type, size_key)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)
        
    for key, indices in groups.items():
        if len(indices) < 2:
            continue
        
        # 获取边框宽度，如果不存在则默认为1
        widths = []
        for i in indices:
            style = elements[i].get("_style", {})
            widths.append(style.get("stroke_width", 1))
        
        if not widths:
            continue
        median_width = int(np.median(widths))
        
        for i in indices:
            if "_style" not in elements[i]:
                elements[i]["_style"] = {}
            elements[i]["_style"]["stroke_width"] = median_width
            
    return elements


# ======================== CV矩形检测优化辅助函数 ========================

from .basic_shape_rectangles import detect_rectangles_robust
