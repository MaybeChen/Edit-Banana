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

from .basic_shape_detection import (
    extract_geometric_params, calculate_iou, calculate_stroke_width,
    extract_style_colors, extract_style_specific, extract_color_with_mask,
    unify_element_styles, detect_rectangles_robust,
)


# ======================== 基本图形处理器 ========================
class BasicShapeProcessor(BaseProcessor):
    """
    基本图形处理模块
    
    处理流程：
        1. 从context.elements中筛选基本图形
        2. 对每个图形提取填充色和描边色
        3. 生成XML片段
        4. 可选：运行CV补充检测遗漏的矩形
    """
    
    def __init__(self, config=None, enable_cv_detection: bool = True):
        """
        Args:
            config: 处理配置
            enable_cv_detection: 是否启用CV补充检测（检测SAM3遗漏的矩形）
        """
        super().__init__(config)
        self.enable_cv_detection = enable_cv_detection
    
    def process(self, context: ProcessingContext) -> ProcessingResult:
        """
        处理入口
        
        Args:
            context: 处理上下文
            
        Returns:
            ProcessingResult
        """
        self._log("开始处理基本图形")
        
        if not context.image_path or not os.path.exists(context.image_path):
            return ProcessingResult(
                success=False,
                error_message="图片路径无效"
            )
        
        cv2_image = cv2.imread(context.image_path)
        if cv2_image is None:
            return ProcessingResult(
                success=False,
                error_message="无法读取图片"
            )
        
        # 筛选基本图形
        elements_to_process = self._get_elements_to_process(context.elements)
        
        # 计算画布面积，用于判断大面积元素
        canvas_area = context.canvas_width * context.canvas_height
        
        processed_count = 0
        for elem in elements_to_process:
            try:
                self._process_element(elem, cv2_image, canvas_area)
                processed_count += 1
            except Exception as e:
                elem.processing_notes.append(f"处理失败: {str(e)}")
                self._log(f"元素{elem.id}处理失败: {e}")
        
        # CV补充检测
        cv_added_count = 0
        if self.enable_cv_detection:
            cv_added_count = self._run_cv_detection(context, cv2_image)
        
        self._log(f"处理完成: {processed_count}个SAM3图形, {cv_added_count}个CV补充")
        
        return ProcessingResult(
            success=True,
            elements=context.elements,
            canvas_width=context.canvas_width,
            canvas_height=context.canvas_height,
            metadata={
                'processed_count': processed_count,
                'cv_added_count': cv_added_count,
                'total_to_process': len(elements_to_process)
            }
        )
    
    def _get_elements_to_process(self, elements: List[ElementInfo]) -> List[ElementInfo]:
        """筛选需要处理的基本图形"""
        return [
            e for e in elements
            if e.element_type.lower() in VECTOR_TYPES and e.fill_color is None
        ]
    
    def _process_element(self, elem: ElementInfo, cv2_image: np.ndarray, canvas_area: int = 0):
        """
        处理单个元素：提取颜色并生成XML
        
        优先使用SAM3提供的Mask进行精确取色
        
        Args:
            elem: 元素信息
            cv2_image: OpenCV格式的图像
            canvas_area: 画布总面积，用于判断大面积元素
        """
        elem_type = elem.element_type.lower()
        
        # 提取样式 - 优先使用Mask
        if elem.mask is not None and hasattr(elem.mask, 'shape') and elem.mask.size > 0:
            # 使用SAM3提供的Mask进行精确取色
            style_data = extract_color_with_mask(
            cv2_image,
                elem.bbox.to_list(), 
                elem.mask,
                elem_type
            )
            elem.processing_notes.append("使用Mask精确取色")
        else:
            # 降级：使用传统的bbox取色
            style_data = extract_style_specific(cv2_image, elem.bbox.to_list(), elem_type)
            elem.processing_notes.append("使用bbox取色(无Mask)")
        
        elem.fill_color = style_data["fill_color"]
        elem.stroke_color = style_data["stroke_color"]
        elem.stroke_width = style_data["stroke_width"]
        
        # 记录渐变信息（如果有）
        if style_data.get('has_gradient'):
            elem.processing_notes.append(f"检测到渐变: {style_data.get('gradient_info')}")
        
        # 设置层级 - 根据类型和面积判断
        elem_area = elem.bbox.area if elem.bbox else 0
        area_ratio = elem_area / canvas_area if canvas_area > 0 else 0
        
        # 大面积元素（>15%画布面积）或特定类型放到背景层
        if elem_type in {'section_panel', 'title_bar', 'container'} or area_ratio > 0.15:
            elem.layer_level = LayerLevel.BACKGROUND.value
            if area_ratio > 0.15:
                elem.processing_notes.append(f"大面积元素({area_ratio:.1%})，放入背景层")
        else:
            elem.layer_level = LayerLevel.BASIC_SHAPE.value
        
        # 生成XML片段
        elem.xml_fragment = self._generate_xml(elem, style_data)
        elem.processing_notes.append("BasicShapeProcessor处理完成")
    
    def _generate_xml(self, elem: ElementInfo, style_data: dict) -> str:
        """生成mxCell XML"""
        elem_type = elem.element_type.lower()
        
        # 获取基础样式
        base_style = DRAWIO_STYLES.get(elem_type, "rounded=0;whiteSpace=wrap;html=1;")
        
        # 动态应用几何参数
        geo_params = style_data.get("geo_params", {})
        if elem_type == "parallelogram" and "size" in geo_params:
            base_style += f"size={geo_params['size']:.2f};"
        elif elem_type == "cylinder" and "size" in geo_params:
            base_style += f"size={geo_params['size']};"
        elif elem_type == "triangle" and "direction" in geo_params:
            base_style += f"direction={geo_params['direction']};"
        
        # 构建完整样式
        fill_color = style_data["fill_color"]
        stroke_color = style_data["stroke_color"]
        stroke_width = style_data["stroke_width"]
        
        # Use transparent fills for vectorized diagram shapes so nested raster/OCR
        # details are not covered when SAM3 detects an inner module as a rectangle.
        if elem_type in {"rectangle", "rounded rectangle", "rounded_rectangle", "container", "cylinder"}:
            fill_color = "none"
        style = f"{base_style}fillColor={fill_color};strokeColor={stroke_color};strokeWidth={stroke_width};"
        
        # DrawIO的id必须从2开始（0和1是保留的根元素）
        cell_id = elem.id + 2
        
        return f'''<mxCell id="{cell_id}" parent="1" vertex="1" value="" style="{style}">
  <mxGeometry x="{elem.bbox.x1}" y="{elem.bbox.y1}" width="{elem.bbox.width}" height="{elem.bbox.height}" as="geometry"/>
</mxCell>'''
    
    def _run_cv_detection(self, context: ProcessingContext, cv2_image: np.ndarray) -> int:
        """运行CV补充检测"""
        # 构建SAM3元素字典格式
        sam3_elements = {}
        for elem in context.elements:
            elem_type = elem.element_type.lower()
            if elem_type not in sam3_elements:
                sam3_elements[elem_type] = []
            sam3_elements[elem_type].append({
                "bbox": elem.bbox.to_list(),
                "score": elem.score
            })
        
        # 运行检测
        h, w = cv2_image.shape[:2]
        cv_results = detect_rectangles_robust(cv2_image, sam3_elements, {
            # Diagram cards are small relative to the full canvas (~2-4% in common exports).
            "min_area_ratio": 0.015,
            "max_area_ratio": 0.95
        })
        
        added_count = 0
        start_id = max([e.id for e in context.elements], default=0) + 1
        
        # 添加检测到的矩形
        for item in cv_results["rectangles"]:
            new_elem = self._create_element_from_cv(item, start_id + added_count, "rectangle", cv2_image)
            context.elements.append(new_elem)
            added_count += 1
        
        # 添加检测到的容器
        for item in cv_results["containers"]:
            new_elem = self._create_element_from_cv(item, start_id + added_count, "container", cv2_image)
            context.elements.append(new_elem)
            added_count += 1

        added_count += self._add_container_shapes_from_card_crops(
            context, cv2_image, start_id + added_count
        )
        
        return added_count
    
    def _add_container_shapes_from_card_crops(
        self, context: ProcessingContext, cv2_image: np.ndarray, start_id: int
    ) -> int:
        """Use large SAM3 icon/card crops as container border hints when CV misses them."""
        h, w = cv2_image.shape[:2]
        canvas_area = max(1, h * w)
        existing_shape_boxes = [
            e.bbox.to_list()
            for e in context.elements
            if e.element_type.lower() in VECTOR_TYPES
        ]
        added_count = 0

        for elem in list(context.elements):
            elem_type = elem.element_type.lower()
            if elem_type not in {"icon", "image", "picture", "logo"}:
                continue

            bbox = elem.bbox.to_list()
            bw = max(0, bbox[2] - bbox[0])
            bh = max(0, bbox[3] - bbox[1])
            if bw <= 0 or bh <= 0:
                continue

            area_ratio = (bw * bh) / canvas_area
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if area_ratio < 0.015 or aspect > 2.2:
                continue

            if any(calculate_iou(bbox, existing_box) > 0.80 for existing_box in existing_shape_boxes):
                continue

            item = {
                "bbox": bbox,
                "score": max(elem.score, 0.72),
                "method": "sam3_card_bbox",
                "is_rounded": True,
                "fill_color": "none",
                "stroke_color": "#000000",
                "overlay_border": True,
            }
            new_elem = self._create_element_from_cv(
                item, start_id + added_count, "container", cv2_image
            )
            new_elem.processing_notes.append(
                f"由大图块bbox补充容器边框(area_ratio={area_ratio:.3f})"
            )
            context.elements.append(new_elem)
            existing_shape_boxes.append(bbox)
            added_count += 1

        if added_count:
            self._log(f"从大图块bbox补充了 {added_count} 个容器边框")
        return added_count

    def _create_element_from_cv(self, item: dict, elem_id: int, elem_type: str, cv2_image: np.ndarray) -> ElementInfo:
        """从CV检测结果创建ElementInfo"""
        bbox = BoundingBox.from_list(item["bbox"])
        
        # 判断是圆角还是直角
        actual_type = elem_type
        if item.get("is_rounded", False) and elem_type == "rectangle":
            actual_type = "rounded rectangle"
        
        elem = ElementInfo(
            id=elem_id,
            element_type=actual_type,
            bbox=bbox,
            score=item.get("score", 0.8),
            fill_color=item.get("fill_color"),
            stroke_color=item.get("stroke_color"),
            source_prompt="cv_detection"
        )
        
        # 设置层级. Card overlay borders must sit above raster crops so the crisp
        # vector stroke is visible; normal containers remain behind content.
        if item.get("overlay_border"):
            elem.layer_level = LayerLevel.ARROW.value
        elif elem_type == "container":
            elem.layer_level = LayerLevel.BACKGROUND.value
        else:
            elem.layer_level = LayerLevel.BASIC_SHAPE.value
        
        # 提取填充色（使用优化的取色逻辑）
        style_data = extract_style_specific(cv2_image, item["bbox"], actual_type)
        elem.fill_color = item.get("fill_color") or style_data["fill_color"]
        
        # CV检测结果强制使用黑色细边框（风格统一、更清晰）
        elem.stroke_color = "#000000"
        elem.stroke_width = 1
        
        # 生成XML（使用强制的边框样式）
        style_data_for_xml = {
            "fill_color": elem.fill_color,
            "stroke_color": elem.stroke_color,
            "stroke_width": elem.stroke_width,
            "geo_params": style_data.get("geo_params", {})
        }
        elem.xml_fragment = self._generate_xml(elem, style_data_for_xml)
        elem.processing_notes.append(f"CV检测补充 (method={item.get('method', 'unknown')})")
        
        return elem


# ======================== 独立处理函数 ========================
def process_basic_shapes(image: np.ndarray, sam3_elements: dict) -> str:
    """
    处理所有基本图形（SAM3结果 + CV补充检测），生成DrawIO XML。
    
    :param image: 原始图像 (BGR)
    :param sam3_elements: SAM3提取的元素字典
    :return: 格式化的XML字符串
    """
    h, w = image.shape[:2]
    
    # 运行CV补充检测
    cv_results = detect_rectangles_robust(image, sam3_elements, {
        "min_area_ratio": 0.07,
        "max_area_ratio": 0.95
    })
    
    # 收集所有需要绘制的元素
    containers_list = []
    shapes_list = []
    
    # 来自 SAM3 的 container
    if "container" in sam3_elements:
        for item in sam3_elements["container"]:
            item_copy = item.copy()
            item_copy["_type"] = "container"
            item_copy["_source"] = "sam3"
            containers_list.append(item_copy)
            
    # 来自 CV 检测的 containers
    for item in cv_results["containers"]:
        item_copy = item.copy()
        item_copy["_type"] = "container"
        item_copy["_source"] = "cv"
        containers_list.append(item_copy)
        
    # 来自 SAM3 的其他形状
    for key, items in sam3_elements.items():
        if key in VECTOR_TYPES and key != "container":
            for item in items:
                item_copy = item.copy()
                item_copy["_type"] = key
                item_copy["_source"] = "sam3"
                shapes_list.append(item_copy)
                
    # 来自 CV 检测的 rectangles
    for item in cv_results["rectangles"]:
        item_copy = item.copy()
        item_copy["_type"] = "rectangle"
        item_copy["_source"] = "cv"
        shapes_list.append(item_copy)
    
    # 排序：面积大的在底层
    def calculate_element_area(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    
    containers_list.sort(key=lambda x: calculate_element_area(x["bbox"]), reverse=True)
    shapes_list.sort(key=lambda x: calculate_element_area(x["bbox"]), reverse=True)
    
    # 提取样式
    all_elements_ref = []
    
    def get_style_for_item(item):
        """获取元素样式，CV检测的强制使用黑色细边框"""
        style = extract_style_specific(image, item["bbox"], item["_type"])
        if item.get("_source") == "cv":
            style["stroke_width"] = 1
            style["stroke_color"] = "#000000"
            # 如果detect_rectangles_robust已经提取了填充色，优先使用
            if item.get("fill_color"):
                style["fill_color"] = item["fill_color"]
        return style
    
    for item in containers_list:
        item["_style"] = get_style_for_item(item)
        all_elements_ref.append(item)
        
    for item in shapes_list:
        item["_style"] = get_style_for_item(item)
        all_elements_ref.append(item)
        
    # 统一边框厚度
    unify_element_styles(all_elements_ref)
    
    # 构建XML结构
    mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "type": "device"})
    diagram = ET.SubElement(mxfile, "diagram", {"id": "BasicShapes", "name": "Page-1"})
    mx_graph_model = ET.SubElement(diagram, "mxGraphModel", {
        "dx": str(w), "dy": str(h), "grid": "1", "gridSize": "10",
        "guides": "1", "tooltips": "1", "connect": "1", "arrows": "1",
        "fold": "1", "page": "1", "pageScale": "1",
        "pageWidth": str(w), "pageHeight": str(h),
        "background": "#ffffff"
    })
    root = ET.SubElement(mx_graph_model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    
    cell_id = 2
    
    def add_cell(item_list):
        nonlocal cell_id
        for item in item_list:
            x1, y1, x2, y2 = map(int, item["bbox"])
            width, height = x2 - x1, y2 - y1
            elem_type = item["_type"]
            
            style_data = item.get("_style")
            if not style_data:
                f_color, s_color, s_width = extract_style_colors(image, item["bbox"])
                style_data = {"fill_color": f_color, "stroke_color": s_color, "stroke_width": s_width}
                
            fill_color = style_data["fill_color"]
            stroke_color = style_data["stroke_color"]
            stroke_width = style_data["stroke_width"]
            
            if elem_type in ("rectangle", "rounded rectangle", "container"):
                is_rounded = item.get("is_rounded", elem_type == "rounded rectangle")
                rounded_val = "1" if is_rounded else "0"
                base_style = f"rounded={rounded_val};whiteSpace=wrap;html=1;"
            else:
                base_style = DRAWIO_STYLES.get(elem_type, "rounded=0;whiteSpace=wrap;html=1;")
                
                geo_params = style_data.get("geo_params", {})
                if elem_type == "parallelogram" and "size" in geo_params:
                    base_style += f"size={geo_params['size']:.2f};"
                elif elem_type == "cylinder" and "size" in geo_params:
                    base_style += f"size={geo_params['size']};"
                elif elem_type == "triangle" and "direction" in geo_params:
                    base_style += f"direction={geo_params['direction']};"
            
            # Use transparent fills for vectorized diagram shapes so nested raster/OCR
            # details are not covered when SAM3 detects an inner module as a rectangle.
            if elem_type in {"rectangle", "rounded rectangle", "rounded_rectangle", "container", "cylinder"}:
                fill_color = "none"
            style = f"{base_style}fillColor={fill_color};strokeColor={stroke_color};strokeWidth={stroke_width};"
            
            cell = ET.SubElement(root, "mxCell", {
                "id": str(cell_id),
                "parent": "1",
                "vertex": "1",
                "value": "",
                "style": style
            })
            ET.SubElement(cell, "mxGeometry", {
                "x": str(x1), "y": str(y1),
                "width": str(width), "height": str(height),
                "as": "geometry"
            })
            
            cell_id += 1

    add_cell(containers_list)
    add_cell(shapes_list)
    
    # 格式化XML
    rough_string = ET.tostring(mxfile, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return '\n'.join([
        line for line in reparsed.toprettyxml(indent="  ").split('\n')
        if line.strip() and not line.strip().startswith("<?xml")
    ])


# ======================== 快捷函数 ========================
def extract_shape_colors(elements: List[ElementInfo], 
                         image_path: str) -> List[ElementInfo]:
    """
    快捷函数 - 提取所有基本图形的颜色
    
    Args:
        elements: 元素列表
        image_path: 原始图片路径
        
    Returns:
        处理后的元素列表
    """
    processor = BasicShapeProcessor()
    context = ProcessingContext(
        image_path=image_path,
        elements=elements
    )
    
    result = processor.process(context)
    return result.elements
