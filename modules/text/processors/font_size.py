"""
Font size processor: map OCR block height to pt; optionally unify sizes by proximity.
"""

import copy
import statistics
from typing import List, Dict, Any


class FontSizeProcessor:
    """Compute font size from block height; optional clustering to unify nearby blocks."""

    def __init__(
        self,
        formula_ratio: float = 0.6,
        text_offset: float = 1.0,
        text_height_ratio: float = 0.58,
        max_body_font_size: float = 18.0,
    ):
        self.formula_ratio = formula_ratio
        self.text_offset = text_offset
        self.text_height_ratio = text_height_ratio
        self.max_body_font_size = max_body_font_size
    
    def process(
        self, 
        text_blocks: List[Dict[str, Any]],
        unify: bool = True,
        vertical_threshold_ratio: float = 0.5,
        font_diff_threshold: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        处理字号（主入口）
        
        Args:
            text_blocks: 文字块列表
            unify: 是否执行聚类统一
            vertical_threshold_ratio: 垂直距离阈值比例
            font_diff_threshold: 字号差异阈值
            
        Returns:
            处理后的文字块列表
        """
        # 步骤 1: 计算初始字号
        blocks = self.calculate_font_sizes(text_blocks)
        blocks = self.promote_title_text_sizes(blocks)
        
        # 步骤 2: 聚类统一
        if unify and len(blocks) > 1:
            blocks = self.unify_by_clustering(
                blocks, 
                vertical_threshold_ratio, 
                font_diff_threshold
            )
            blocks = self.unify_body_text_size(blocks)
        
        return blocks
    
    def calculate_font_sizes(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Set draw.io fontSize from OCR bbox height.

        Draw.io uses point-like font sizes while OCR geometry is in source-image
        pixels. Using the raw bbox height makes exported text larger than the
        original glyphs; the default ratio keeps rendered text inside the OCR box.
        """
        result = []
        for block in text_blocks:
            block = copy.copy(block)
            geometry = block.get("geometry", {})
            height = geometry.get("height", 12)
            is_latex = block.get("is_latex", False)
            
            if is_latex:
                font_size = height * self.formula_ratio
            else:
                # OCR boxes are source-image pixels; draw.io fontSize is point-like
                # and line-height adds extra visual height. Keep text inside its OCR
                # geometry to preserve relative size/layout in the original image.
                font_size = min(height * self.text_height_ratio, height - self.text_offset)
                font_size = min(font_size, self.max_body_font_size)
            
            block["font_size"] = max(font_size, 6)
            result.append(block)
        return result


    def promote_title_text_sizes(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preserve large/top-level title text instead of capping it to body size.

        OCR bbox heights are used for sizing, but diagram titles are often much
        taller than body labels and are located near the top of the canvas. The
        body cap keeps labels stable; this pass restores title-like blocks so
        the relative hierarchy is closer to the source image and PPTX export.
        """
        if not text_blocks:
            return text_blocks

        heights = [
            b.get("geometry", {}).get("height", 0)
            for b in text_blocks
            if not b.get("is_latex") and b.get("geometry", {}).get("height", 0) > 0
        ]
        if not heights:
            return text_blocks

        median_height = statistics.median(heights)
        max_right = max(
            b.get("geometry", {}).get("x", 0) + b.get("geometry", {}).get("width", 0)
            for b in text_blocks
        )
        max_bottom = max(
            b.get("geometry", {}).get("y", 0) + b.get("geometry", {}).get("height", 0)
            for b in text_blocks
        )
        canvas_w = max(1, max_right)
        canvas_h = max(1, max_bottom)

        result = copy.deepcopy(text_blocks)
        promoted_count = 0
        for block in result:
            if block.get("is_latex") or block.get("text_role") == "title":
                continue
            geo = block.get("geometry", {})
            height = geo.get("height", 0)
            width = geo.get("width", 0)
            y = geo.get("y", 0)
            text = str(block.get("text", "")).strip()
            if not text or height <= 0:
                continue

            top_band = y <= canvas_h * 0.16
            wide_heading = width >= canvas_w * 0.22
            large_heading = height >= median_height * 1.55
            short_label = len(text) <= 36
            if (large_heading and short_label) or (top_band and wide_heading and height >= median_height * 1.15):
                title_size = min(max(height * 0.72, block.get("font_size", 0)), 44.0)
                if title_size > block.get("font_size", 0) + 0.1:
                    block["font_size"] = round(title_size, 1)
                    block["font_weight"] = block.get("font_weight") or "bold"
                    block["text_role"] = "title"
                    promoted_count += 1

        if promoted_count:
            print(f"     Font size: preserved {promoted_count} title/header blocks")
        return result

    def unify_by_clustering(
        self,
        text_blocks: List[Dict[str, Any]],
        vertical_threshold_ratio: float = 0.5,
        font_diff_threshold: float = 5.0
    ) -> List[Dict[str, Any]]:
        """Unify font sizes for spatially close blocks (union-find + median)."""
        if not text_blocks:
            return text_blocks
        
        n = len(text_blocks)
        parent = list(range(n))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 聚类
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_group(
                    text_blocks[i], text_blocks[j],
                    vertical_threshold_ratio, font_diff_threshold
                ):
                    union(i, j)
        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        result = copy.deepcopy(text_blocks)
        adjusted_count = 0
        for group_indices in groups.values():
            if len(group_indices) < 2:
                continue
            font_sizes = [result[i].get("font_size", 12) for i in group_indices]
            median_size = statistics.median(font_sizes)
            for idx in group_indices:
                old_size = result[idx].get("font_size", 12)
                if abs(old_size - median_size) > 0.1:
                    adjusted_count += 1
                result[idx]["font_size"] = round(median_size, 1)
        multi_groups = [g for g in groups.values() if len(g) > 1]
        if multi_groups and adjusted_count > 0:
            print(f"     Font size: unified {adjusted_count} blocks in {len(multi_groups)} groups")
        return result

    def unify_body_text_size(self, text_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize similarly sized non-formula labels to one body size across a diagram."""
        body_sizes = [
            b.get("font_size", 12)
            for b in text_blocks
            if not b.get("is_latex") and b.get("font_size", 0) > 0
        ]
        if len(body_sizes) < 4:
            return text_blocks

        median_size = statistics.median(body_sizes)
        if median_size <= 0:
            return text_blocks

        result = copy.deepcopy(text_blocks)
        adjusted_count = 0
        for block in result:
            if block.get("is_latex") or block.get("text_role") == "title":
                continue
            size = block.get("font_size", median_size)
            # Preserve real titles/annotations; normalize ordinary labels only.
            if median_size * 0.55 <= size <= median_size * 1.65:
                normalized = round(min(median_size, self.max_body_font_size), 1)
                if abs(size - normalized) > 0.1:
                    adjusted_count += 1
                block["font_size"] = normalized

        if adjusted_count:
            print(
                f"     Font size: globally normalized {adjusted_count} body labels "
                f"to {min(median_size, self.max_body_font_size):.1f}pt"
            )
        return result

    def _should_group(
        self, 
        block_a: Dict, 
        block_b: Dict,
        vertical_threshold_ratio: float,
        font_diff_threshold: float
    ) -> bool:
        """判断两个文字块是否应该分到同一组"""
        if block_a.get("text_role") == "title" or block_b.get("text_role") == "title":
            return False
        geo_a = block_a.get("geometry", {})
        geo_b = block_b.get("geometry", {})
        
        x1, y1 = geo_a.get("x", 0), geo_a.get("y", 0)
        w1, h1 = geo_a.get("width", 0), geo_a.get("height", 0)
        x2, y2 = geo_b.get("x", 0), geo_b.get("y", 0)
        w2, h2 = geo_b.get("width", 0), geo_b.get("height", 0)
        
        font_a = block_a.get("font_size", 12)
        font_b = block_b.get("font_size", 12)
        bottom_a, bottom_b = y1 + h1, y2 + h2
        gap_a_above_b = y2 - bottom_a
        gap_b_above_a = y1 - bottom_b
        
        if gap_a_above_b < 0 and gap_b_above_a < 0:
            vertical_distance = 0
        else:
            vertical_distance = min(abs(gap_a_above_b), abs(gap_b_above_a))
        
        min_height = min(h1, h2) if min(h1, h2) > 0 else 1
        vertical_close = vertical_distance < min_height * vertical_threshold_ratio
        right_a, left_b = x1 + w1, x2
        right_b, left_a = x2 + w2, x1
        horizontal_overlap = not (right_a < left_b or right_b < left_a)
        abs_diff = abs(font_a - font_b)
        avg_font = (font_a + font_b) / 2 if (font_a + font_b) > 0 else 1
        rel_diff = abs_diff / avg_font
        font_close = abs_diff < font_diff_threshold or rel_diff < 0.30
        
        return vertical_close and horizontal_overlap and font_close


if __name__ == "__main__":
    # 测试代码
    processor = FontSizeProcessor()
    
    test_blocks = [
        {"geometry": {"x": 100, "y": 100, "width": 200, "height": 25}},
        {"geometry": {"x": 100, "y": 130, "width": 180, "height": 24}},
        {"geometry": {"x": 100, "y": 160, "width": 190, "height": 26}},
    ]
    
    result = processor.process(test_blocks)
    for i, block in enumerate(result):
        print(f"Block {i+1}: font_size = {block['font_size']}pt")
