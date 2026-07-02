# 图片 → 可编辑 PPTX：VLM 识别流程与 Prompt
## 一、总流程
```text
原始图片
  │
  ├─ 1. 图片预处理：保存原图、生成 VLM 缩略图
  ├─ 2. OCR：提取文字、文字框、表格文字
  ├─ 3. Layout VLM：全图识别页面区域与主布局
  ├─ 4. ROI 裁剪：根据 Layout 结果从原图裁剪局部区域
  ├─ 5. Shape VLM：识别卡片、图形、线条、箭头
  ├─ 6. Asset VLM：识别图片、图标、Logo、复杂视觉区域
  ├─ 7. Structured VLM：按需识别表格、图表、流程图
  ├─ 8. Rule Engine：去重、分组、层级、连接关系、重建策略
  └─ 9. PPTX Builder：生成可编辑元素与图片兜底元素
```
---
# 二、统一约定
## 坐标体系
所有 VLM Prompt 都使用归一化坐标：
```text
左上角：(0, 0)
右下角：(1000, 1000)
```
全图识别时，坐标相对于整页图片。
ROI 专项识别时，坐标相对于当前 ROI 图片；后端负责将 ROI 坐标换算回全图坐标。
## 统一元素策略
```text
native_text           可编辑文本框
native_shape          可编辑 Shape
native_line           可编辑线条或箭头
native_table          可编辑表格
native_chart          可编辑图表
vector_asset          SVG 或矢量图标
cropped_image         原图局部裁剪插入
group_reconstruction  多元素组合重建
hybrid                部分可编辑、部分图片
image_fallback        整块保留为图片
```
---
# 三、步骤 0：图片预处理
这一阶段不调用 VLM。
```text
输入：原始图片
处理：
1. 保存原图。
2. 获取真实像素宽高，仅供后端使用。
3. 生成 VLM 识别缩略图：
   - 长边 <= 2048：直接使用
   - 长边 > 2048：等比例缩放到 2048
4. 后续 ROI 必须从原图裁剪，而不是从缩略图裁剪。
5. 所有 VLM 坐标统一使用 normalized_0_1000。
```
---
# 四、步骤 1：全图 Layout VLM
## 目标
只识别页面骨架和大区域，不要识别逐字文本、小图标、细线、表格单元格或图表数值。

## 调用条件

每页图片固定调用一次。

## Prompt

```text
你是 PPT 页面布局分析器。

请分析整张页面图片，只识别页面级结构和主要视觉区域。
不要逐字识别文本，不要识别表格单元格，不要提取图表数据，不要识别小图标细节。

坐标使用 normalized_0_1000：
左上角为 (0,0)，右下角为 (1000,1000)。

请识别以下区域类型：
- background
- header
- footer
- sidebar
- main_content
- container_group
- card_group
- image_region
- icon_logo_region
- table_region
- chart_region
- diagram_region
- complex_visual_region

要求：
1. 每个区域必须包含唯一 id、type、bbox、confidence。
2. bbox 尽量贴合实际区域，不要包含过多无关空白。
3. 识别页面主布局，例如 single_column、two_column、three_column、dashboard、grid、timeline。
4. 输出阅读顺序。
5. 不要输出页面内的具体文字内容。
6. 只输出合法 JSON，不要输出解释或 Markdown。

输出格式：

{
  "page_aspect_ratio_estimate": "16:9",
  "layout_pattern": "two_column",
  "page_structure": "header + main_content + footer",
  "regions": [
    {
      "id": "region_001",
      "type": "header",
      "bbox": {"x": 0, "y": 0, "width": 1000, "height": 140},
      "confidence": 0.95
    }
  ],
  "reading_order": ["region_001"]
}
```

---

# 五、步骤 2：OCR + 文本样式 VLM

## OCR 职责

OCR 负责：

```text
文字内容
文字行 / 段落
文字 bbox
文字置信度
表格中的文字
```

不要让 VLM 重新做主 OCR。

## Text VLM 调用条件

仅对以下情况调用：

```text
标题层级判断
字体大小估计
字体粗细
文本颜色
居中 / 左对齐 / 右对齐
文字角色识别
文字是否位于图片或卡片上
```

建议按 `header`、`card_group`、`chart_region`、`table_region` 等 ROI 调用。

## Prompt

```text
你是 PPT 文本样式与语义分析器。

当前输入是一张 ROI 图片，并附带 OCR 识别结果。
不要修改 OCR 文本内容；OCR 的 text 字段是文字内容的唯一可信来源。
你的任务是为每个 OCR 文本块补充 PPT 重建所需的语义和样式信息。

当前 ROI 坐标使用 normalized_0_1000。
只分析 ROI 内的 OCR 文本，不要识别 ROI 外内容。

OCR 输入：
{{OCR_BLOCKS_JSON}}

请为每个 OCR 文本块输出：
- ocr_id
- subtype：title、subtitle、section_title、body、label、number、percentage、date、caption、footnote、button_text、table_header、table_cell、chart_label、chart_axis_label、legend_label
- font_size_estimate
- font_weight：normal 或 bold
- font_style：normal 或 italic
- font_color：#RRGGBB
- text_align：left、center、right
- vertical_align：top、middle、bottom
- is_text_on_image：true 或 false
- is_text_on_shape：true 或 false
- confidence

规则：
1. 不要输出 OCR 中不存在的新文本。
2. 标题、正文、数字、标签应根据视觉层级区分 subtype。
3. 字体无法判断时使用 Microsoft YaHei。
4. 不确定的样式可以降低 confidence，不要编造复杂字体效果。
5. 只输出合法 JSON。

输出格式：

{
  "texts": [
    {
      "ocr_id": "ocr_001",
      "subtype": "title",
      "font_family": "Microsoft YaHei",
      "font_size_estimate": 28,
      "font_weight": "bold",
      "font_style": "normal",
      "font_color": "#1A1A1A",
      "text_align": "left",
      "vertical_align": "middle",
      "is_text_on_image": false,
      "is_text_on_shape": false,
      "confidence": 0.88
    }
  ]
}
```

---

# 六、步骤 3：Shape、Container、Line、Arrow 专项识别

## 调用条件

仅对以下 ROI 调用：

```text
header
footer
container_group
card_group
diagram_region
button 区域
tag / badge 区域
```

不要对全图单独跑一次。

## Prompt

```text
你是 PPT 原生图形识别器。

当前输入是一张 ROI 图片。
请只识别当前 ROI 内可用 PPTX 原生元素重建的图形，不要识别文字内容、照片、复杂插画或 Logo。

坐标使用 ROI 内 normalized_0_1000。
只输出当前 ROI 内元素。

需要识别的类型：
- container
- shape
- line
- arrow

container subtype 可选：
- card
- panel
- section
- button
- tag
- badge
- tab
- dialog
- metric_card

shape subtype 可选：
- rectangle
- rounded_rectangle
- circle
- ellipse
- triangle
- diamond
- pill
- chevron
- hexagon
- freeform_shape

line subtype 可选：
- horizontal_line
- vertical_line
- diagonal_line
- divider
- dashed_line
- dotted_line

arrow subtype 可选：
- right_arrow
- left_arrow
- up_arrow
- down_arrow
- double_arrow
- curved_arrow
- connector_arrow
- process_arrow

要求：
1. 卡片、色块、按钮背景、标签底板优先识别为 container 或 shape。
2. 普通分割线识别为 line。
3. 有方向或流程含义的连接线识别为 arrow。
4. 每个元素输出 bbox、fill、stroke、corner_radius_estimate、shadow、rotation、confidence。
5. 箭头额外输出 start_point、end_point、arrow_head、arrow_tail。
6. 不要把复杂图片或插画错误识别为 shape。
7. 只输出合法 JSON。

输出格式：

{
  "elements": [
    {
      "id": "shape_001",
      "type": "container",
      "subtype": "card",
      "bbox": {"x": 50, "y": 120, "width": 400, "height": 300},
      "fill": {"type": "solid", "color": "#FFFFFF", "transparency": 0},
      "stroke": {"color": "#E5E7EB", "width": 1, "dash": "solid"},
      "corner_radius_estimate": 16,
      "shadow": {
        "enabled": true,
        "blur": 8,
        "offset_x": 0,
        "offset_y": 4,
        "opacity": 0.12
      },
      "rotation": 0,
      "editable_strategy": "native_shape",
      "confidence": 0.91
    }
  ]
}
```

---

# 七、步骤 4：Image、Icon、Logo、Complex Visual 专项识别

## 调用条件

仅对以下 ROI 调用：

```text
image_region
icon_logo_region
complex_visual_region
```

## Prompt

```text
你是 PPT 视觉资产识别器。

当前输入是一张 ROI 图片。
请识别当前 ROI 内的视觉资产，并判断它们在 PPTX 中最合适的重建方式。

坐标使用 ROI 内 normalized_0_1000。

可选 type：
- image
- icon
- logo
- decoration
- complex_visual
- unknown

image subtype：
- photo
- illustration
- screenshot
- product_image
- portrait
- scene_image
- map
- ui_mockup
- embedded_visual

icon subtype：
- line_icon
- filled_icon
- glyph_icon
- ui_icon
- social_icon
- status_icon
- illustrative_icon

logo subtype：
- brand_logo
- product_logo
- partner_logo
- app_logo
- text_logo
- combined_logo

complex_visual subtype：
- 3d_illustration
- complex_infographic
- complex_ui
- complex_map
- complex_artwork
- mixed_visual
- text_image_fusion

要求：
1. 品牌标识优先识别为 logo，不要误判为 icon。
2. 简单小型功能符号识别为 icon。
3. 照片、截图、地图、复杂插画识别为 image。
4. 无法稳定拆分的复杂区域识别为 complex_visual。
5. 输出 vectorizable、has_transparent_background、preserve_as_image、editable_strategy。
6. 简单图标优先 editable_strategy=vector_asset。
7. Logo、照片、截图、复杂插画优先 cropped_image 或 image_fallback。
8. 不要输出文字内容。
9. 只输出合法 JSON。

输出格式：

{
  "elements": [
    {
      "id": "asset_001",
      "type": "logo",
      "subtype": "brand_logo",
      "bbox": {"x": 60, "y": 80, "width": 180, "height": 120},
      "vectorizable": false,
      "has_transparent_background": true,
      "preserve_as_image": true,
      "editable_strategy": "cropped_image",
      "confidence": 0.87
    }
  ]
}
```

---

# 八、步骤 5A：Table 专项识别

## 调用条件

只对 `table_region` 调用。

OCR 或文档布局服务已经能正确识别表格时，可以跳过 VLM Table Prompt。

## Prompt

```text
你是 PPT 表格结构识别器。

当前输入是一张表格 ROI 图片，并附带 OCR 或文档布局结果。
请识别表格的结构，不要分析 ROI 外内容。

坐标使用 ROI 内 normalized_0_1000。

输入的 OCR / Layout 结果：
{{TABLE_OCR_JSON}}

任务：
1. 判断该区域是否为结构清晰、可重建的表格。
2. 若可以重建，识别 rows、columns、header_row、header_column、merged_cells。
3. 提取每个单元格的 row、col、rowspan、colspan、text、fill_color、font_color、font_weight、text_align。
4. 若表格结构不可靠，返回 image_fallback 和原因。
5. OCR 文本内容优先，不要编造单元格文字。
6. 只输出合法 JSON。

输出格式：

{
  "table_reconstructable": true,
  "editable_strategy": "native_table",
  "rows": 4,
  "columns": 3,
  "has_header_row": true,
  "has_header_column": false,
  "merged_cells": [],
  "cells": [
    {
      "row": 0,
      "col": 0,
      "rowspan": 1,
      "colspan": 1,
      "text": "指标",
      "fill_color": "#F3F4F6",
      "font_color": "#111827",
      "font_weight": "bold",
      "text_align": "center"
    }
  ],
  "confidence": 0.85,
  "reason": null
}
```

---

# 九、步骤 5B：Chart 专项识别

## 调用条件

只对 `chart_region` 调用。

## Prompt

```text
你是 PPT 图表结构识别器。

当前输入是一张图表 ROI 图片。
请只识别当前 ROI 内图表，不要识别 ROI 外内容。

坐标使用 ROI 内 normalized_0_1000。

任务：
1. 判断图表类型。
2. 识别标题、图例、类别、系列、坐标轴、数据标签。
3. 判断数据是否足够可靠，可以重建为原生 PPTX Chart。
4. 如果数据无法可靠读取，不要编造数值。
5. 如果可视觉重绘但不能恢复真实数据，使用 group_reconstruction。
6. 如果图表太复杂或数据不可读，使用 image_fallback。

chart_type 可选：
- bar_chart
- column_chart
- line_chart
- area_chart
- pie_chart
- donut_chart
- scatter_chart
- radar_chart
- funnel_chart
- waterfall_chart
- gauge_chart
- combo_chart
- kpi_card
- unknown_chart

只输出合法 JSON。

输出格式：

{
  "chart_type": "column_chart",
  "title": "季度收入",
  "legend": ["2025", "2026"],
  "categories": ["Q1", "Q2", "Q3", "Q4"],
  "series": [
    {
      "name": "2025",
      "values": [10, 20, 18, 30],
      "color": "#4F81BD"
    }
  ],
  "axis": {
    "x_label": "季度",
    "y_label": "收入",
    "y_min": 0,
    "y_max": 40
  },
  "editable_strategy": "native_chart",
  "data_reconstruction_confidence": 0.78,
  "confidence": 0.86,
  "reason": null
}
```

---

# 十、步骤 5C：Diagram 专项识别

## 调用条件

只对 `diagram_region` 调用。

## Prompt

```text
你是 PPT 流程图与关系图识别器。

当前输入是一张 Diagram ROI 图片。
请识别当前 ROI 中的节点、连接线、箭头、标签和整体关系结构。

坐标使用 ROI 内 normalized_0_1000。

diagram_type 可选：
- flowchart
- process_flow
- timeline
- organization_chart
- mind_map
- relationship_graph
- cycle_diagram
- funnel_diagram
- pyramid_diagram
- roadmap
- stepper
- matrix_diagram

要求：
1. 节点必须作为独立元素输出。
2. 节点可使用 container、shape、group 或 text。
3. 连线必须作为 line 或 arrow 单独输出。
4. 箭头必须尽量关联 from_element_id 和 to_element_id。
5. 输出 flow_direction 和 layout_pattern。
6. 文本内容优先使用 OCR 结果；没有 OCR 时只输出可清晰识别的文字。
7. 如果关系无法可靠判断，保留节点，但将 arrow 的连接关系标记为 uncertain。
8. 只输出合法 JSON。

输出格式：

{
  "diagram_type": "process_flow",
  "flow_direction": "left_to_right",
  "layout_pattern": "horizontal",
  "nodes": [
    {
      "id": "node_001",
      "type": "container",
      "subtype": "process_node",
      "bbox": {"x": 50, "y": 300, "width": 180, "height": 180},
      "editable_strategy": "group_reconstruction",
      "confidence": 0.88
    }
  ],
  "connectors": [
    {
      "id": "arrow_001",
      "type": "arrow",
      "subtype": "right_arrow",
      "start_point": {"x": 240, "y": 390},
      "end_point": {"x": 320, "y": 390},
      "from_element_id": "node_001",
      "to_element_id": "node_002",
      "relationship_type": "flow",
      "confidence": 0.76
    }
  ],
  "confidence": 0.82
}
```

---

# 十一、步骤 6：代码规则合并

这一阶段优先使用代码，不建议让 VLM 再识别一次整页。

```text
输入：
- OCR 结果
- Layout VLM 结果
- Shape VLM 结果
- Asset VLM 结果
- Table / Chart / Diagram 结果

处理：
1. 统一 ROI 坐标到全图 normalized_0_1000。
2. 去重。
3. 建立 parent_id。
4. 建立 group_id。
5. 推断 z_index。
6. 建立箭头连接关系。
7. 确定 editable_strategy。
8. 识别重复组件。
```

## 推荐规则

```text
1. OCR 文本与 VLM 文本重叠：
   OCR 内容优先，VLM 样式优先。

2. text 完整落在 card / container 内：
   text.parent_id = container.id。

3. icon 与 text 水平相邻、间距小：
   组成 icon_text_group。

4. logo 与 icon 重叠：
   logo 优先。

5. complex_visual 与 image 重叠：
   complex_visual 优先。

6. 同类型元素 IoU > 0.85：
   保留 confidence 更高的元素。

7. 箭头端点接近两个节点：
   自动建立 from_element_id / to_element_id。

8. 默认图层顺序：
   background
   → decoration
   → container / shape
   → image / complex_visual
   → table / chart / diagram
   → line / arrow
   → icon / logo
   → text
```

---

# 十二、步骤 7：可选 Relationship Repair VLM

只有在规则无法解决关系时才调用，例如：

```text
箭头连接到哪个节点不明确
两个相邻元素是独立卡片还是一个分组
一个复杂区域是图表还是流程图
```

## Prompt

```text
你是 PPT 元素关系校正器。

当前输入包含：
1. ROI 图片；
2. 已识别元素列表；
3. 当前规则引擎无法确定的关系问题。

不要重新识别整张图片，不要新增无关元素。
只判断给定元素之间的关系。

输入元素：
{{ELEMENTS_JSON}}

待解决问题：
{{QUESTIONS_JSON}}

请输出：
- confirmed_parent_child
- confirmed_groups
- confirmed_connectors
- rejected_relationships
- confidence

只输出合法 JSON。
```

---

# 十三、最终建议的调用顺序

```text
每页固定调用：
1. OCR
2. Layout VLM

按需调用：
3. Text Style VLM
4. Shape VLM
5. Asset VLM
6. Table VLM
7. Chart VLM
8. Diagram VLM

通常不调用：
9. Relationship Repair VLM
```

## 推荐路由规则

```text
存在 chart_region → 调用 Chart VLM
存在 table_region 且 OCR 表格不可靠 → 调用 Table VLM
存在 diagram_region → 调用 Diagram VLM
存在 image_region / icon_logo_region → 调用 Asset VLM
存在 card_group / container_group → 调用 Shape VLM
存在标题、正文、数字样式要求 → 调用 Text Style VLM
```

---

# 十四、最小可用版本

第一版只实现：

```text
OCR
+ Layout VLM
+ Text Style VLM
+ Shape VLM
+ Asset VLM
+ Rule Engine
+ cropped_image / image_fallback
```

第二版再加入：

```text
Table VLM
Chart VLM
Diagram VLM
Relationship Repair VLM
```

这样可以优先保证：

```text
文本可编辑
卡片和色块可编辑
图片和 Logo 不丢失
复杂区域有图片兜底
整页结构稳定
```
