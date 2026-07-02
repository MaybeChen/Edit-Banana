"""Convenience APIs built on MetricEvaluator."""

import os
from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image

from .base import ProcessingContext
from .data_types import ElementInfo
from .metric_evaluator import MetricEvaluator

def evaluate_result(elements: List[ElementInfo],
                    image_path: str,
                    canvas_width: int = 0,
                    canvas_height: int = 0,
                    config: Dict = None) -> Dict[str, Any]:
    """
    快捷函数 - 评估转换结果
    
    Args:
        elements: 元素列表
        image_path: 原始图片路径
        canvas_width: 画布宽度（可选，会自动从图片获取）
        canvas_height: 画布高度（可选）
        config: 评估配置
        
    Returns:
        评估结果字典，包含：
        - overall_score: 总体分数（0-100，即覆盖率）
        - content_coverage: 内容覆盖率
        - missing_rate: 漏检率
        - bad_regions: 问题区域列表
        - metrics: 详细指标
        
    使用示例:
        result = evaluate_result(elements, "test.png")
        print(f"评分: {result['overall_score']}/100")
        print(f"覆盖率: {result['content_coverage']}%")
        print(f"漏检率: {result['missing_rate']}%")
        print(f"问题区域: {len(result['bad_regions'])}个")
        
        for region in result['bad_regions']:
            print(f"  - {region['bbox']}: {region['description']}")
    """
    evaluator = MetricEvaluator(config)
    context = ProcessingContext(
        image_path=image_path,
        elements=elements,
        canvas_width=canvas_width,
        canvas_height=canvas_height
    )
    
    result = evaluator.process(context)
    return result.metadata

def compute_content_coverage(image_path: str, 
                              bboxes: List[List[int]],
                              content_threshold: int = 245) -> Dict[str, float]:
    """
    计算内容覆盖率的简化函数
    
    Args:
        image_path: 图片路径
        bboxes: bbox列表，格式 [[x1,y1,x2,y2], ...]
        content_threshold: 内容检测阈值
        
    Returns:
        {'coverage': 覆盖率, 'missing': 漏检率}
    """
    img = cv2.imread(image_path)
    if img is None:
        return {'coverage': 0, 'missing': 100}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 内容掩码
    content_mask = (gray < content_threshold).astype(np.uint8)
    total_content = np.sum(content_mask > 0)
    
    if total_content == 0:
        return {'coverage': 100, 'missing': 0}
    
    # 覆盖掩码
    covered_mask = np.zeros((h, w), dtype=np.uint8)
    for bbox in bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 > x1 and y2 > y1:
            covered_mask[y1:y2, x1:x2] = 1
    
    # 覆盖的内容
    covered_content = np.sum(np.logical_and(content_mask, covered_mask))
    coverage = (covered_content / total_content) * 100
    
    return {
        'coverage': round(coverage, 2),
        'missing': round(100 - coverage, 2)
    }


# ======================== 渲染对比功能 ========================

def compare_with_rendered(original_path: str, 
                          rendered_path: str,
                          config: Dict = None) -> Dict[str, Any]:
    """
    对比原图和渲染后的图像，找出差异区域（遗漏的内容）
    
    Args:
        original_path: 原始图片路径
        rendered_path: DrawIO渲染后的图片路径
        config: 配置参数
            - diff_threshold: 差异阈值（默认30）
            - min_region_area: 最小区域面积（默认500）
            - merge_distance: 相邻区域合并距离（默认20）
    
    Returns:
        {
            'overall_similarity': 整体相似度 (0-100),
            'missing_regions': 差异区域列表 [{'bbox': [x1,y1,x2,y2], 'area': int, ...}],
            'diff_image_path': 差异可视化图路径（如果指定了output_path）
        }
    
    使用示例:
        result = compare_with_rendered("original.png", "rendered.png")
        print(f"相似度: {result['overall_similarity']}%")
        for region in result['missing_regions']:
            print(f"遗漏区域: {region['bbox']}")
    """
    default_config = {
        'diff_threshold': 30,
        'min_region_area': 500,
        'merge_distance': 20,
        'output_path': None,  # 差异可视化输出路径
    }
    cfg = {**default_config, **(config or {})}
    
    # 读取图像
    original = cv2.imread(original_path)
    rendered = cv2.imread(rendered_path)
    
    if original is None or rendered is None:
        return {
            'overall_similarity': 0,
            'missing_regions': [],
            'error': '无法读取图像'
        }
    
    # 确保尺寸一致
    if original.shape != rendered.shape:
        rendered = cv2.resize(rendered, (original.shape[1], original.shape[0]))
    
    h, w = original.shape[:2]
    total_area = h * w
    
    # 计算差异
    diff = cv2.absdiff(original, rendered)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # 阈值化找出显著差异区域
    _, diff_mask = cv2.threshold(diff_gray, cfg['diff_threshold'], 255, cv2.THRESH_BINARY)
    
    # 形态学处理：合并相邻区域
    kernel_size = cfg['merge_distance']
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    
    # 找连通域
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    missing_regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg['min_region_area']:
            continue
        
        x, y, rw, rh = cv2.boundingRect(cnt)
        
        # 计算该区域的差异强度
        region_diff = diff_gray[y:y+rh, x:x+rw]
        diff_intensity = np.mean(region_diff)
        
        missing_regions.append({
            'bbox': [x, y, x+rw, y+rh],
            'area': int(rw * rh),
            'area_ratio': (rw * rh) / total_area,
            'diff_intensity': float(diff_intensity),
            'description': f'渲染差异区域 ({rw}x{rh})'
        })
    
    # 计算整体相似度
    diff_pixels = np.count_nonzero(diff_mask)
    similarity = max(0, 100 - (diff_pixels / total_area) * 100)
    
    # 可视化输出
    output_path = cfg.get('output_path')
    if output_path:
        vis = original.copy()
        for region in missing_regions:
            x1, y1, x2, y2 = region['bbox']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(output_path, vis)
    
    return {
        'overall_similarity': round(similarity, 2),
        'missing_regions': missing_regions,
        'diff_pixels': int(diff_pixels)
    }


def detect_missing_from_rendered_diff(original_path: str,
                                       rendered_path: str,
                                       output_dir: str = None) -> List[Dict]:
    """
    从渲染对比中检测遗漏区域，并裁剪保存
    
    这个函数会：
    1. 对比原图和渲染图
    2. 找出遗漏的区域
    3. 从原图裁剪这些区域
    4. 可选保存为单独的图片
    
    Args:
        original_path: 原始图片路径
        rendered_path: 渲染后的图片路径
        output_dir: 裁剪区域保存目录（可选）
    
    Returns:
        遗漏区域列表，每个包含:
        - bbox: 边界框
        - cropped_image: PIL Image对象
        - base64: base64编码的图像
    
    使用示例:
        missing = detect_missing_from_rendered_diff("original.png", "rendered.png")
        for i, region in enumerate(missing):
            # 可以直接用于生成XML
            base64_data = region['base64']
            bbox = region['bbox']
    """
    import base64
    from io import BytesIO
    
    # 检测差异区域
    result = compare_with_rendered(original_path, rendered_path, {
        'diff_threshold': 25,
        'min_region_area': 300,
        'merge_distance': 15
    })
    
    if not result.get('missing_regions'):
        return []
    
    # 读取原图
    original_pil = Image.open(original_path).convert("RGB")
    
    missing_elements = []
    
    for i, region in enumerate(result['missing_regions']):
        x1, y1, x2, y2 = region['bbox']
        
        # 裁剪
        cropped = original_pil.crop((x1, y1, x2, y2))
        
        # 转base64
        buffer = BytesIO()
        cropped.save(buffer, format='PNG')
        b64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        elem = {
            'bbox': region['bbox'],
            'area': region['area'],
            'area_ratio': region['area_ratio'],
            'diff_intensity': region['diff_intensity'],
            'cropped_image': cropped,
            'base64': b64_data,
            'description': region['description']
        }
        
        # 可选保存
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            crop_path = os.path.join(output_dir, f"missing_region_{i}.png")
            cropped.save(crop_path)
            elem['saved_path'] = crop_path
        
        missing_elements.append(elem)
    
    return missing_elements
