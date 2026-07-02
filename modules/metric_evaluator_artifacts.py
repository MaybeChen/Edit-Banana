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




class MetricArtifactMixin:
    def _save_uncovered_visualization(self,
                                      cv2_image: np.ndarray,
                                      uncovered_content: np.ndarray,
                                      covered_mask: np.ndarray,
                                      bad_regions: List[Dict],
                                      output_path: str):
        """
        保存问题区域可视化图 - 重点突出需要 fallback 补救的区域
        
        显示：
        - 原图作为背景
        - 红色半透明填充 + 粗红框标记问题区域（需要 fallback）
        - 问题区域的详细标注
        
        Args:
            cv2_image: 原始图像
            uncovered_content: 未覆盖内容掩码（不再显示，因为大部分是噪点）
            covered_mask: 已覆盖区域掩码（不再显示）
            bad_regions: 问题区域列表
            output_path: 输出路径
        """
        h, w = cv2_image.shape[:2]
        
        # 创建输出图像（原图副本）
        result = cv2_image.copy()
        overlay = cv2_image.copy()
        
        # 1. 画问题区域（红色半透明填充 + 粗边框）
        for i, region in enumerate(bad_regions):
            x1, y1, x2, y2 = region['bbox']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # 半透明红色填充
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
            
            # 粗红色边框
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # 标注：序号 + 面积 + 原因
            area_pct = region.get('area_ratio', 0) * 100
            channel = region.get('channel', 'unknown')
            reason = region.get('reason', 'unknown')
            
            # 显示简化的原因
            if 'complex' in channel:
                reason_short = "IMAGE_NO_BASE64"
            elif channel == 'fine':
                reason_short = "UNCOVERED_FINE"
            elif channel == 'coarse':
                reason_short = "UNCOVERED_COARSE"
            else:
                reason_short = channel.upper()
            
            label = f"#{i+1} {reason_short} ({area_pct:.1f}%)"
            
            # 背景框让文字更清晰
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(result, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 0, 255), -1)
            cv2.putText(result, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 混合半透明层
        alpha = 0.3
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
        
        # 2. 添加图例（顶部）
        legend_bg = np.zeros((120, w, 3), dtype=np.uint8)
        legend_bg[:] = (40, 40, 40)  # 深灰色背景
        
        cv2.putText(legend_bg, f"METRIC EVALUATION - Problem Regions for Fallback", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(legend_bg, f"Total Bad Regions: {len(bad_regions)}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(legend_bg, f"RED = regions that need fallback (image content without base64 processing)", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        
        # 拼接图例和结果图
        result = np.vstack([legend_bg, result])
        
        cv2.imwrite(output_path, result)
        self._log(f"保存问题区域可视化: {output_path}")
    
    def _save_evaluation_json(self,
                              metrics: Dict,
                              bad_regions: List[Dict],
                              needs_refinement: bool,
                              overall_score: float,
                              output_path: str):
        """
        保存评估结果到 JSON 文件
        
        Args:
            metrics: 详细指标
            bad_regions: 问题区域列表
            needs_refinement: 是否需要二次处理
            overall_score: 总体评分
            output_path: 输出路径
        """
        import json
        
        # 转换 numpy 类型为 Python 原生类型
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
        
        evaluation_result = {
            'overall_score': round(float(overall_score), 2),
            'needs_refinement': bool(needs_refinement),
            'metrics': convert_to_native(metrics),
            'bad_regions': convert_to_native(bad_regions),
            'summary': {
                'score': f"{overall_score:.1f}/100",
                'bad_region_ratio': f"{metrics['total_bad_region_ratio']:.1f}%",
                'bad_region_count': int(metrics['bad_region_count']),
                'pixel_coverage': f"{metrics['pixel_coverage']:.1f}%",
                'element_count': int(metrics['element_count']),
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, ensure_ascii=False, indent=2)
        
        self._log(f"保存评估结果: {output_path}")
    
    def save_visualization(self, 
                           context: ProcessingContext,
                           bad_regions: List[Dict],
                           output_path: str):
        """
        保存评估结果可视化图
        
        Args:
            context: 处理上下文
            bad_regions: 问题区域列表
            output_path: 输出路径
        """
        if not context.image_path or not os.path.exists(context.image_path):
            return
        
        img = cv2.imread(context.image_path)
        if img is None:
            return
        
        h, w = img.shape[:2]
        
        # 1. 画已检测元素（蓝色）
        for elem in context.elements:
            x1 = max(0, min(w, elem.bbox.x1))
            y1 = max(0, min(h, elem.bbox.y1))
            x2 = max(0, min(w, elem.bbox.x2))
            y2 = max(0, min(h, elem.bbox.y2))
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 100, 0), 2)
        
        # 2. 画问题区域（红色=粗粒度，绿色=细粒度）
        for region in bad_regions:
            x1, y1, x2, y2 = region['bbox']
            color = (0, 0, 255) if region.get('channel') == 'coarse' else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            # 标注
            text = f"{region['area_ratio']*100:.1f}%"
            cv2.putText(img, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # 3. 图例
        cv2.putText(img, "Blue: Detected", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)
        cv2.putText(img, "Red: Missing (coarse)", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, "Green: Missing (fine)", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, img)
        self._log(f"保存可视化结果: {output_path}")
    
    def save_uncovered_mask(self,
                            context: ProcessingContext,
                            output_path: str,
                            bad_regions: List[Dict[str, Any]] = None):
        """
        保存问题区域可视化图
        
        显示：
        1. 检测到的问题区域（红色边框 + 半透明填充）
        2. 有 base64 的图片元素（绿色边框）
        3. 无 base64 的图片类元素（黄色边框）
        """
        if not context.image_path or not os.path.exists(context.image_path):
            return
        
        img = cv2.imread(context.image_path)
        if img is None:
            return
        
        h, w = img.shape[:2]
        result = img.copy()
        overlay = img.copy()
        
        # 1. 画有 base64 的元素（绿色）和无 base64 的图片类元素（黄色）
        for elem in context.elements:
            x1 = max(0, min(w, elem.bbox.x1))
            y1 = max(0, min(h, elem.bbox.y1))
            x2 = max(0, min(w, elem.bbox.x2))
            y2 = max(0, min(h, elem.bbox.y2))
            
            elem_type = elem.element_type.lower()
            is_image_type = elem_type in self.IMAGE_CONTENT_TYPES
            
            if elem.base64 is not None:
                # 有 base64 的元素：绿色
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 200, 0), 2)
            elif is_image_type:
                # 图片类但无 base64：黄色（这是问题！）
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 200, 255), 2)
        
        # 2. 画检测到的问题区域（红色粗框 + 半透明填充）
        if bad_regions:
            for i, region in enumerate(bad_regions):
                bbox = region['bbox']
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(w, x1))
                y1 = max(0, min(h, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                
                # 半透明红色填充
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        
                # 红色粗边框
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 4)
                
                # 标注序号和面积
                area_pct = region.get('area_ratio', 0) * 100
                label = f"BAD #{i+1} ({area_pct:.1f}%)"
                cv2.putText(result, label, (x1 + 5, y1 + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # 混合半透明层
        alpha = 0.25
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
        
        # 3. 图例
        legend_y = 40
        cv2.putText(result, f"GREEN: elements with base64 (OK)", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)
        cv2.putText(result, f"YELLOW: image-type without base64 (PROBLEM)", (10, legend_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        cv2.putText(result, f"RED: detected bad regions for fallback ({len(bad_regions or [])})", (10, legend_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        cv2.imwrite(output_path, result)
        self._log(f"保存问题区域可视化: {output_path}")
