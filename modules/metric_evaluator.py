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
from .metric_evaluator_regions import MetricRegionMixin
from .metric_evaluator_artifacts import MetricArtifactMixin


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


class MetricEvaluator(MetricRegionMixin, MetricArtifactMixin, BaseProcessor):
    """
    质量评估模块
    
    评估SAM3+其他检测器的覆盖效果，识别漏检区域。
    
    核心功能：
        1. 计算内容覆盖率（检测到的内容占总内容的比例）= 评分
        2. 识别问题区域（有内容但未被检测覆盖）
        3. 输出总体评分和问题区域列表供RefinementProcessor使用
    """

    DEFAULT_CONFIG = {
        # ===== 内容检测参数 =====
        # 注意：这里的"内容"指的是需要被检测的前景元素，不包括背景
        'content_threshold': 245,      # 灰度阈值，低于此值认为有内容
        'use_edge_detection': True,    # 是否使用边缘检测增强
        'edge_low_threshold': 30,      # Canny边缘检测低阈值（更敏感）
        'edge_high_threshold': 100,    # Canny边缘检测高阈值（更敏感）
        
        # ===== 背景过滤参数 =====
        'filter_background': True,     # 是否过滤背景区域
        'background_denoise_kernel': 2, # 去噪形态学核大小（更小，保留更多细节）
        'min_content_area': 30,        # 最小内容连通域面积（更小，保留更多细节）
        
        # ===== 细粒度通道参数（检测小目标：图标/人脸/小图）=====
        'fine_min_area_ratio': 0.0005,     # 最小面积比例 0.05%（更敏感）
        'fine_max_area_ratio': 0.20,       # 最大面积比例 20%
        'fine_min_fill_ratio': 0.15,       # 最小填充率（更宽松，检测稀疏内容）
        'fine_max_aspect_ratio': 8.0,      # 最大宽高比
        
        # ===== 粗粒度通道参数（检测版块/大图）=====
        'coarse_min_area_ratio': 0.002,    # 最小面积比例 0.2%（更敏感）
        'coarse_max_area_ratio': 0.30,     # 最大面积比例 30%
        'coarse_min_fill_ratio': 0.20,     # 最小填充率（更宽松）
        'coarse_max_aspect_ratio': 8.0,    # 最大宽高比
        'coarse_kernel_size': 5,           # 闭操作核大小（更小，避免过度合并）
        
        # ===== NMS和去重参数 =====
        'nms_iou_threshold': 0.3,          # 小框优先NMS的IoU阈值（更严格）
        'existing_iou_threshold': 0.5,     # 与已有元素去重的IoU阈值（更宽松，保留更多候选）
        'max_covered_ratio': 0.7,          # 候选框内最大已覆盖比例（更宽松）
        
        # ===== 评分阈值 =====
        'good_coverage_threshold': 95,     # 覆盖率>=95%才认为很好（更严格）
        'acceptable_threshold': 80,        # 覆盖率>=80%认为可接受
        
        # ===== 漏检内容最小比例 =====
        'min_missing_content_ratio': 0.05, # 候选框内至少5%是漏检内容才保留（更敏感）
    }
    
    def __init__(self, config=None):
        super().__init__(config)
        # 合并用户配置和默认配置
        self.eval_config = {**self.DEFAULT_CONFIG, **(config or {})}
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """
        处理入口 - 评估质量
        
        Args:
            context: 处理上下文
            
        Returns:
            ProcessingResult，metadata中包含评估结果
        """
        self._log("开始质量评估")
        
        # 验证输入
        if not context.image_path or not os.path.exists(context.image_path):
            return ProcessingResult(
                success=False,
                error_message="图片路径无效"
            )
        
        # 加载原图
        cv2_image = cv2.imread(context.image_path)
        if cv2_image is None:
            return ProcessingResult(
                success=False,
                error_message="无法读取图片"
            )
        
        h, w = cv2_image.shape[:2]
        img_area = h * w
        
        # ========== 1. 创建内容掩码（识别有内容的区域） ==========
        content_mask = self._create_content_mask(cv2_image)
        total_content_pixels = int(np.sum(content_mask > 0))
        
        # ========== 2. 创建覆盖掩码（已检测元素覆盖的区域） ==========
        # 获取 OCR 文字 XML（如果有的话）
        text_xml = context.intermediate_results.get('text_xml', None)
        covered_mask, existing_bboxes = self._create_covered_mask(context.elements, h, w, text_xml)
        
        # ========== 3. 计算覆盖率评分 ==========
        # 内容区域中被覆盖的像素
        covered_content = cv2.bitwise_and(content_mask, covered_mask)
        covered_content_pixels = int(np.sum(covered_content > 0))
        
        # 像素级覆盖率（辅助指标）
        if total_content_pixels > 0:
            content_coverage = (covered_content_pixels / total_content_pixels) * 100
        else:
            content_coverage = 100.0  # 没有内容，认为完全覆盖
        
        missing_rate = 100.0 - content_coverage
        
        # ========== 4. 计算未覆盖内容掩码 ==========
        uncovered_content = cv2.bitwise_and(
            content_mask,
            cv2.bitwise_not(covered_mask)
        )
        
        # ========== 5. 识别问题区域（三重策略） ==========
        bad_regions = self._detect_bad_regions(
            cv2_image, content_mask, covered_mask, existing_bboxes, img_area, context.elements, context
        )
        
        # ========== 6. 计算真实评分（基于问题区域面积，去重） ==========
        # 创建问题区域掩码，去除重叠部分
        bad_region_mask = np.zeros((h, w), dtype=np.uint8)
        for region in bad_regions:
            x1, y1, x2, y2 = region['bbox']
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                bad_region_mask[y1:y2, x1:x2] = 255
        
        # 计算去重后的实际问题区域面积
        actual_bad_region_pixels = int(np.sum(bad_region_mask > 0))
        total_bad_region_ratio = (actual_bad_region_pixels / img_area) * 100 if img_area > 0 else 0
        
        # 评分 = 100 - 问题区域总面积比例（去重后）
        overall_score = max(0, 100.0 - total_bad_region_ratio)
        
        # 是否需要refinement
        # 条件: 有问题区域就需要补救
        has_complex_image_regions = any(r.get('channel') == 'complex' for r in bad_regions)
        needs_refinement = len(bad_regions) > 0 or has_complex_image_regions
        
        # 构建详细指标（使用去重后的面积）
        total_bad_region_area = actual_bad_region_pixels
        metrics = {
            'overall_score': round(overall_score, 2),  # 最终评分（100 - 问题区域面积比例）
            'pixel_coverage': round(content_coverage, 2),  # 像素级覆盖率（辅助指标）
            'missing_rate': round(missing_rate, 2),
            'total_content_pixels': total_content_pixels,
            'covered_content_pixels': covered_content_pixels,
            'missing_content_pixels': total_content_pixels - covered_content_pixels,
            'image_area': img_area,
            'content_ratio': round(total_content_pixels / img_area * 100, 2),
            'element_count': len(context.elements),
            'bad_region_count': len(bad_regions),
            'total_bad_region_area': total_bad_region_area,
            'total_bad_region_ratio': round(total_bad_region_ratio, 2),  # 问题区域占图片面积的比例
        }
        
        self._log(f"评估完成: 评分={overall_score:.1f}, 问题区域={len(bad_regions)}个, 问题面积={total_bad_region_ratio:.1f}%")
        
        # ========== 6. 自动保存可视化和评估结果到 output_dir ==========
        if context.output_dir and os.path.exists(context.output_dir):
            # 保存未覆盖内容可视化
            uncovered_vis_path = os.path.join(context.output_dir, "metric_uncovered.png")
            self._save_uncovered_visualization(cv2_image, uncovered_content, covered_mask, bad_regions, uncovered_vis_path)
            
            # 保存评估分数到 JSON
            eval_json_path = os.path.join(context.output_dir, "metric_evaluation.json")
            self._save_evaluation_json(metrics, bad_regions, needs_refinement, overall_score, eval_json_path)
        
        return ProcessingResult(
            success=True,
            elements=context.elements,
            canvas_width=context.canvas_width or w,
            canvas_height=context.canvas_height or h,
            metadata={
                'overall_score': round(overall_score, 2),
                'pixel_coverage': round(content_coverage, 2),
                'bad_region_ratio': round(total_bad_region_ratio, 2),
                'metrics': metrics,
                'bad_regions': bad_regions,
                'needs_refinement': needs_refinement,
            }
        )
    
    def _create_content_mask(self, cv2_image: np.ndarray) -> np.ndarray:
        """
        创建内容掩码 - 识别图片中真正需要被检测的前景内容
        
        关键改进：区分"前景内容"和"背景"
        - 背景：大面积连续的单色区域（白色、浅灰色、浅蓝色等）
        - 前景内容：图形、图标、文字、箭头等需要被检测的元素
        
        策略：
        1. 边缘检测（推荐）：有边缘的地方才是真正的内容边界
        2. 灰度阈值：低于阈值的非背景区域
        3. 形态学去噪：去除小噪点
        4. 连通域过滤：去除太小的连通域
        
        Returns:
            二值掩码，255表示有前景内容，0表示背景/无内容
        """
        gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # ========== 方法1：边缘检测（推荐开启，能更好地识别内容边界）==========
        # 边缘才是真正区分前景和背景的标志
        if self.eval_config.get('use_edge_detection', True):
            edges = cv2.Canny(
                gray,
                self.eval_config['edge_low_threshold'],
                self.eval_config['edge_high_threshold']
            )
            # 膨胀边缘，让边缘区域扩展成有效内容区域
            kernel = np.ones((5, 5), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel, iterations=2)
            edge_mask = edges_dilated
        else:
            edge_mask = np.zeros((h, w), dtype=np.uint8)
        
        # ========== 方法2：灰度阈值（作为补充）==========
        threshold = self.eval_config['content_threshold']
        content_by_gray = (gray < threshold).astype(np.uint8) * 255
        
        # ========== 合并两种方法 ==========
        if self.eval_config.get('use_edge_detection', True):
            # 边缘检测开启时：取交集或并集（推荐取并集，但边缘为主）
            # 这里使用并集，但主要依赖边缘检测的结果
            content_mask = cv2.bitwise_or(content_by_gray, edge_mask)
        else:
            content_mask = content_by_gray
        
        # ========== 背景过滤：去除噪点和小区域 ==========
        if self.eval_config.get('filter_background', True):
            # 1. 形态学去噪（开操作：先腐蚀后膨胀，去除小噪点）
            denoise_size = self.eval_config.get('background_denoise_kernel', 3)
            if denoise_size > 0:
                denoise_kernel = np.ones((denoise_size, denoise_size), np.uint8)
                content_mask = cv2.morphologyEx(content_mask, cv2.MORPH_OPEN, denoise_kernel)
            
            # 2. 连通域过滤：去除太小的连通域（可能是噪点）
            min_area = self.eval_config.get('min_content_area', 50)
            if min_area > 0:
                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(content_mask, connectivity=8)
                filtered_mask = np.zeros_like(content_mask)
                for i in range(1, num_labels):  # 跳过背景（标签0）
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= min_area:
                        filtered_mask[labels == i] = 255
                content_mask = filtered_mask
        
        # 日志输出
        content_pixels = int(np.sum(content_mask > 0))
        content_ratio = content_pixels / (h * w) * 100
        self._log(f"内容检测: {content_pixels}px ({content_ratio:.1f}% of image)")
        
        return content_mask
    
    # 需要有 base64 图片才算真正覆盖的元素类型（复杂图片内容）
    IMAGE_CONTENT_TYPES = {
        'icon', 'picture', 'photo', 'chart', 'function_graph', 'screenshot', 
        'image', 'diagram', 'logo', 'heatmap', 'graph', 'line graph', 'bar graph',
        'pie chart', 'scatter plot', 'histogram'
    }
    
    # 基本矢量图形类型（有 XML 就算覆盖）
    VECTOR_SHAPE_TYPES = {
        'rectangle', 'rounded rectangle', 'circle', 'ellipse', 'diamond', 
        'triangle', 'cloud', 'arrow', 'line', 'connector', 'polygon',
        'section_panel', 'title_bar', 'background'
    }
    
    def _create_covered_mask(self, 
                              elements: List[ElementInfo],
                              height: int,
                              width: int,
                              text_xml: str = None) -> Tuple[np.ndarray, List[List[int]]]:
        """
        创建覆盖掩码 - 严格判断有效输出
        
        有效输出的定义：
        1. 对于图片类内容（热力图、chart等）：必须有 base64 才算覆盖
        2. 对于基本矢量图形：有 xml_fragment 就算覆盖
        3. 加上 OCR 识别的文字区域
        
        Returns:
            (covered_mask, existing_bboxes)
            - covered_mask: 二值掩码，255表示已覆盖，0表示未覆盖
            - existing_bboxes: 已有元素的bbox列表
        """
        covered_mask = np.zeros((height, width), dtype=np.uint8)
        existing_bboxes = []
        
        img_area = height * width
        
        valid_count = 0
        skipped_image_no_base64 = 0
        skipped_no_output = 0
        
        for elem in elements:
            elem_type = elem.element_type.lower()
            
            # 计算元素面积比例
            elem_area = (elem.bbox.x2 - elem.bbox.x1) * (elem.bbox.y2 - elem.bbox.y1)
            area_ratio = elem_area / img_area if img_area > 0 else 0
            
            # 判断是否为图片类内容
            is_image_content = elem_type in self.IMAGE_CONTENT_TYPES
            
            # 判断是否有有效输出
            if is_image_content:
                # 图片类内容：必须有 base64 才算覆盖
                has_valid_output = elem.base64 is not None
                if not has_valid_output:
                    skipped_image_no_base64 += 1
                    continue
            else:
                # 基本矢量图形：有 XML 或 base64 就算覆盖
                # 矩形、圆等都是有效的流程图元素，不再跳过"大面积基本图形"
                # 真正漏检的复杂图像内容由策略2检测
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
        
        # 从 text_xml 中提取文字区域
        text_count = 0
        if text_xml:
            text_bboxes = self._extract_text_bboxes_from_xml(text_xml, width, height)
            for bbox in text_bboxes:
                x1, y1, x2, y2 = bbox
                if x2 > x1 and y2 > y1:
                    covered_mask[y1:y2, x1:x2] = 255
                    existing_bboxes.append([x1, y1, x2, y2])
                    text_count += 1
        
        self._log(f"覆盖区域统计: 有效元素={valid_count}, 图片类无base64={skipped_image_no_base64}, 无输出={skipped_no_output}, OCR文字={text_count}")
        
        return covered_mask, existing_bboxes
    
    def _extract_text_bboxes_from_xml(self, text_xml: str, img_width: int, img_height: int) -> List[List[int]]:
        """
        从文字 XML 中提取所有文字元素的 bbox
        
        Args:
            text_xml: 文字处理生成的 XML 内容
            img_width, img_height: 图片尺寸
            
        Returns:
            文字 bbox 列表 [[x1, y1, x2, y2], ...]
        """
        import re
        
        bboxes = []
        
        # 匹配 mxGeometry 标签中的坐标
        # <mxGeometry x="100" y="200" width="50" height="20" as="geometry"/>
        pattern = r'<mxGeometry\s+x="([^"]+)"\s+y="([^"]+)"\s+width="([^"]+)"\s+height="([^"]+)"'
        
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
    
    def _detect_bad_regions(self,
                            cv2_image: np.ndarray,
                            content_mask: np.ndarray,
                            covered_mask: np.ndarray,
                            existing_bboxes: List[List[int]],
                            img_area: int,
                            elements: List[ElementInfo] = None,
                            context: ProcessingContext = None) -> List[Dict[str, Any]]:
        """
        识别问题区域 - 三重策略检测漏检区域
        
        策略：
        1. 双通道检测：基于内容掩码的细粒度/粗粒度连通域检测
        2. 复杂图像区域检测：检测高方差区域（热力图、照片等）是否有 base64 覆盖
        3. 小框优先NMS + 去重过滤
        
        Returns:
            问题区域列表
        """
        h, w = cv2_image.shape[:2]
        
        # 1. 有内容但未覆盖的区域
        uncovered_content = cv2.bitwise_and(
            content_mask,
            cv2.bitwise_not(covered_mask)
        )
        
        candidates = []
        
        # ===== 策略1: 细粒度通道 =====
        fine_candidates = self._detect_fine_channel(uncovered_content, img_area)
        candidates.extend([(box, 'fine') for box in fine_candidates])
        
        # ===== 策略2: 粗粒度通道 =====
        coarse_candidates = self._detect_coarse_channel(uncovered_content, img_area)
        candidates.extend([(box, 'coarse') for box in coarse_candidates])
        
        # ===== 策略3: 复杂图像区域检测 =====
        # 检测高方差区域（热力图、照片等），如果没有 base64 覆盖则标记为问题区域
        complex_candidates = self._detect_complex_image_regions(cv2_image, elements, img_area, context)
        candidates.extend([(box, 'complex') for box in complex_candidates])
        
        self._log(f"三重检测: 细粒度={len(fine_candidates)}, 粗粒度={len(coarse_candidates)}, 复杂图像={len(complex_candidates)}")
        
        # ===== 小框优先NMS =====
        nms_threshold = self.eval_config['nms_iou_threshold']
        candidates = self._nms_smallest_first(candidates, nms_threshold)
        
        self._log(f"NMS后: {len(candidates)}个候选")
        
        # ===== 与已有元素去重 + 覆盖比例过滤 =====
        bad_regions = self._filter_candidates(
            candidates, covered_mask, existing_bboxes, uncovered_content, img_area
        )
        
        # ===== 合并相邻的区域（距离小于图片短边的10%则合并）=====
        h, w = covered_mask.shape[:2]
        merge_distance = min(h, w) * 0.10  # 合并距离阈值
        bad_regions = self._merge_nearby_regions(bad_regions, merge_distance, img_area)
        
        # 按面积从大到小排序
        bad_regions.sort(key=lambda r: r['area'], reverse=True)
        
        return bad_regions
    


# ======================== 快捷函数 ========================




def evaluate_result(image_path: str, elements: List[ElementInfo], output_dir: str = "./output") -> Dict[str, Any]:
    from .metric_evaluator_api import evaluate_result as _evaluate_result
    return _evaluate_result(image_path, elements, output_dir)


def compute_content_coverage(image_path: str, elements: List[ElementInfo]) -> float:
    from .metric_evaluator_api import compute_content_coverage as _compute_content_coverage
    return _compute_content_coverage(image_path, elements)


def compare_with_rendered(original_path: str, rendered_path: str, output_dir: str = "./output") -> Dict[str, Any]:
    from .metric_evaluator_api import compare_with_rendered as _compare_with_rendered
    return _compare_with_rendered(original_path, rendered_path, output_dir)


def detect_missing_from_rendered_diff(original_path: str, rendered_path: str, output_dir: str = "./output") -> Dict[str, Any]:
    from .metric_evaluator_api import detect_missing_from_rendered_diff as _detect_missing_from_rendered_diff
    return _detect_missing_from_rendered_diff(original_path, rendered_path, output_dir)
