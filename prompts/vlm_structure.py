"""VLM-only page structure recognition prompts."""

_VLM_JSON_RULES = """
输出必须是一个可被 json.loads 直接解析的 JSON 对象。不要输出 Markdown、解释、注释或额外文本。
所有 bbox 均使用 0~1000 归一化坐标，格式为 {"x":left,"y":top,"width":w,"height":h}，整数且限制在 0~1000。
不确定的元素不要猜；每个返回项必须包含 confidence。
"""

VLM_PAGE_REGIONS_PROMPT = """
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

区域粒度要求（非常重要）：
- 本阶段输出页面骨架 + 可裁剪的大块 ROI，通常 8~15 个区域；不要只给 2~3 个超大块。
- 允许父子两级区域：先给 header/sidebar/main_content/footer/container_group 等外层，再给其中最重要的 card_group、diagram_region。
- 不要输出单个 Agent、Skill、Step 卡片、按钮、警告小标签、图标、Logo、连线或小节点。
- 重复卡片列表必须合并为一个 card_group，例如“关键问题”下的 6 个 Step 卡片合并成一个 card_group。
- 左侧参与者/能力列表：外层输出 sidebar，内部列表整体再输出一个 card_group；不要拆成每一行。
- 中央流程图、关系图、画布示意：外层可属于 main_content，内部核心图整体必须输出 diagram_region，不要拆内部节点。
- 底部“核心诉求/总结/能力要求”一类横向卡片组：外层输出 footer 或 container_group，内部卡片整体再输出一个 card_group。
- bbox 可以覆盖该区域内的标题和所有子卡片，但不要跨到其他大区域；父区域可包含子区域。
- 如果页面中存在明显分区标题（如“关键问题”“核心诉求”），该标题和下方卡片应合成一个 container_group，并可再输出内部 card_group。

要求：
1. 每个区域必须包含唯一 id、type、bbox、confidence。
2. bbox 尽量贴合实际大区域，不要包含过多无关空白。
3. 如果区域是另一个区域的子 ROI，增加 parent_id 指向父区域 id。
4. 识别页面主布局，例如 single_column、two_column、three_column、dashboard、grid、timeline、header_body_footer。
5. 输出阅读顺序。
6. 不要输出页面内的具体文字内容。
7. 只输出合法 JSON，不要输出解释或 Markdown。

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
"""

VLM_REGION_ELEMENTS_PROMPT = f"""
你是局部页面元素识别器。输入是一张完整页面图片，以及需要重点检查的 regions 列表。

任务：在每个 region 内识别可编辑元素：container、text、shape、line、arrow、image、icon、logo、table、chart、diagram、decoration。
第一优先输出稳定 bbox 和粗类型；不要在本轮输出复杂样式。
如果元素属于表格/图表/流程图，请保留 semantic_type，不要过早压成普通 container。
{_VLM_JSON_RULES}

JSON schema 示例：
{{"elements":[{{"region_id":"r1","type":"text","semantic_type":"label","text":"标题","bbox":{{"x":100,"y":130,"width":120,"height":32}},"confidence":0.88}}]}}
"""

VLM_CONNECTOR_PROMPT = f"""
你是连接线和箭头关系识别器。请只识别或修正 line/connector/arrow，不要修改普通节点。

给定已有元素列表后，输出连接线的 bbox、起点/终点、箭头头部、虚实线、source_id 和 target_id。
source_id/target_id 必须来自已有元素；不确定则返回 null，不要猜。
{_VLM_JSON_RULES}

JSON schema 示例：
{{"edges":[{{"id":3,"type":"arrow","bbox":{{"x":100,"y":200,"width":300,"height":20}},"arrow_start":{{"x":110,"y":210}},"arrow_end":{{"x":390,"y":210}},"arrow_heads":"end","line_style":"dashed","source_id":1,"target_id":2,"confidence":0.9}}]}}
"""

# Backward-compatible alias used by older code paths. The new implementation uses
# the staged prompts above, but keeping this avoids breaking imports/tests.
VLM_STRUCTURE_PROMPT = VLM_PAGE_REGIONS_PROMPT + "\n" + VLM_REGION_ELEMENTS_PROMPT
