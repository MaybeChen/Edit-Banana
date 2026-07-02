"""VLM-only page structure recognition prompts."""

_VLM_JSON_RULES = """
输出必须是一个可被 json.loads 直接解析的 JSON 对象。不要输出 Markdown、解释、注释或额外文本。
所有 bbox 均使用当前输入图片的实际像素坐标，格式为 {"x":left,"y":top,"width":w,"height":h}，整数且不得超过当前输入图片宽高。不要输出归一化坐标。
不确定的元素不要猜；每个返回项必须包含 confidence。
"""

VLM_PAGE_REGIONS_PROMPT_TEMPLATE = """
你是 PPT 页面布局分析器。

请分析整张页面图片，只识别页面级结构和主要视觉区域。
不要逐字识别文本，不要识别表格单元格，不要提取图表数据，不要识别小图标细节。

坐标必须使用当前输入图片的实际像素坐标，不要使用归一化坐标。
当前输入的原始图片宽度：{image_width}px，高度：{image_height}px。
左上角为 (0,0)，右下角为 ({image_width},{image_height})。
所有 bbox 的 x、y、width、height 都必须是基于这个宽高的像素整数。

定位方法（非常重要）：
- 先观察真实视觉边界：标题文字、浅灰分隔线、圆角容器边框、卡片组外框、左右栏分割线。
- bbox 必须贴合这些真实边界，不要按页面高度平均切成几段，不要套用示例值或固定模板。
- 除 header/footer/background 外，不要轻易给整页宽度；例如 main_content 不能覆盖左侧 sidebar。
- 如果一个区域右侧/下方有明显空白或其他分区，bbox 必须停在可见边界处。

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
- 父子区域不能使用完全相同或几乎相同的 bbox；如果子区域边界无法比父区域更精确，就不要输出这个子区域。
- 不要同时输出语义重复且 bbox 相同的区域，例如 sidebar 和 card_group_sidebar 完全同框、main_content 和 diagram_region 完全同框。
- 如果页面中存在明显分区标题（如“关键问题”“核心诉求”），该标题和下方卡片应合成一个 container_group，并可再输出内部 card_group。

要求：
1. 每个区域必须包含唯一 id、type、bbox、confidence。
2. bbox 尽量贴合实际大区域，不要包含过多无关空白。
3. 如果区域是另一个区域的子 ROI，增加 parent_id 指向父区域 id。
4. 识别页面主布局，例如 single_column、two_column、three_column、dashboard、grid、timeline、header_body_footer。
5. 输出 reading_order，顺序必须按真实阅读顺序排列。
6. 不要输出页面内的具体文字内容。
7. 输出前自检：如果 bbox 的 y/height 看起来像 120/400/520/920 等模板分段，而不是贴合图片边界，必须重新定位。
8. 输出前自检：如果两个区域 bbox 几乎完全重合，只保留更有用的一个，不要重复输出。
9. 只输出合法 JSON，不要输出解释或 Markdown。

输出 JSON 字段结构：
- page_aspect_ratio_estimate: 字符串
- layout_pattern: 字符串
- page_structure: 字符串
- regions: 数组；每项包含 id、type、bbox、confidence，可选 parent_id
- bbox: 对象，包含 x、y、width、height，全部是当前输入图片像素整数
- reading_order: region id 字符串数组
"""


def build_vlm_page_regions_prompt(image_width: int, image_height: int) -> str:
    """Build a Layout VLM prompt using actual input-image pixel dimensions."""
    width = max(1, int(image_width or 1))
    height = max(1, int(image_height or 1))
    return (
        VLM_PAGE_REGIONS_PROMPT_TEMPLATE
        .replace("{image_width}", str(width))
        .replace("{image_height}", str(height))
    )


# Backward-compatible constant for older imports; runtime code should call
# build_vlm_page_regions_prompt with the actual VLM image dimensions.
VLM_PAGE_REGIONS_PROMPT = build_vlm_page_regions_prompt(1000, 1000)

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


ROI_PIXEL_RULES_TEMPLATE = """
当前 ROI 图片宽度：{roi_width}px，高度：{roi_height}px。
所有 bbox、start_point、end_point 都必须使用当前 ROI 图片的实际像素坐标：左上角 (0,0)，右下角 ({roi_width},{roi_height})。
不要使用 0~1000 归一化坐标，不要输出百分比。
输出必须是可被 json.loads 直接解析的 JSON 对象，不要输出 Markdown、解释、注释或额外文本。
"""

SHAPE_ROI_PROMPT_TEMPLATE = """
你是 PPT 原生图形识别器。

当前输入是一张 ROI 图片，ROI 类型：{region_type}，region_id：{region_id}。
请只识别当前 ROI 内可用 draw.io/PPT 原生元素重建的图形，不要识别文字内容、照片、复杂插画或 Logo。
{pixel_rules}

需要识别的 type：container、shape、line、arrow。
container subtype 可选：card、panel、section、button、tag、badge、tab、dialog、metric_card。
shape subtype 可选：rectangle、rounded_rectangle、circle、ellipse、triangle、diamond、pill、chevron、hexagon、freeform_shape。
line subtype 可选：horizontal_line、vertical_line、diagonal_line、divider、dashed_line、dotted_line。
arrow subtype 可选：right_arrow、left_arrow、up_arrow、down_arrow、double_arrow、curved_arrow、connector_arrow、process_arrow。

要求：
1. 卡片、色块、按钮背景、标签底板优先识别为 container 或 shape。
2. 普通分割线识别为 line；有方向或流程含义的连接线识别为 arrow。
3. 每个元素输出 id、type、subtype、bbox、fill、stroke、corner_radius_estimate、shadow、rotation、editable_strategy、confidence。
4. 箭头额外输出 start_point、end_point、arrow_head、arrow_tail。
5. 不要把复杂图片或插画错误识别为 shape。

输出格式：{"elements":[{"id":"shape_001","type":"container","subtype":"card","bbox":{"x":10,"y":20,"width":300,"height":120},"fill":{"type":"solid","color":"#FFFFFF","transparency":0},"stroke":{"color":"#E5E7EB","width":1,"dash":"solid"},"corner_radius_estimate":16,"shadow":{"enabled":false},"rotation":0,"editable_strategy":"native_shape","confidence":0.91}]}
"""

ASSET_ROI_PROMPT_TEMPLATE = """
你是 PPT 视觉资产识别器。

当前输入是一张 ROI 图片，ROI 类型：{region_type}，region_id：{region_id}。
请识别当前 ROI 内的视觉资产，并判断它们在 draw.io/PPT 中最合适的重建方式。
{pixel_rules}

可选 type：image、icon、logo、decoration、complex_visual、unknown。
要求：品牌标识优先识别为 logo；简单小型功能符号识别为 icon；照片、截图、地图、复杂插画识别为 image；无法稳定拆分的复杂区域识别为 complex_visual。输出 vectorizable、has_transparent_background、preserve_as_image、editable_strategy、confidence。不要输出文字内容。

输出格式：{"elements":[{"id":"asset_001","type":"logo","subtype":"brand_logo","bbox":{"x":10,"y":20,"width":180,"height":80},"vectorizable":false,"has_transparent_background":true,"preserve_as_image":true,"editable_strategy":"cropped_image","confidence":0.87}]}
"""

TABLE_ROI_PROMPT_TEMPLATE = """
你是 PPT 表格结构识别器。

当前输入是一张表格 ROI 图片，region_id：{region_id}，并附带 OCR 结果。请识别表格结构，不要分析 ROI 外内容。
{pixel_rules}
OCR 输入：{ocr_json}

任务：判断是否可重建为原生表格；若可以，识别 rows、columns、header_row、header_column、merged_cells、cells；OCR 文本内容优先，不要编造单元格文字。
输出格式：{"table_reconstructable":true,"editable_strategy":"native_table","rows":4,"columns":3,"has_header_row":true,"has_header_column":false,"merged_cells":[],"cells":[],"confidence":0.85,"reason":null}
"""

CHART_ROI_PROMPT_TEMPLATE = """
你是 PPT 图表结构识别器。

当前输入是一张图表 ROI 图片，region_id：{region_id}。请只识别当前 ROI 内图表，不要识别 ROI 外内容。
{pixel_rules}
任务：判断图表类型、标题、图例、类别、系列、坐标轴、数据标签，以及 editable_strategy。数据无法可靠读取时不要编造数值。
输出格式：{"chart_type":"bar_chart","title":null,"legend":[],"categories":[],"series":[],"axis":{},"editable_strategy":"image_fallback","data_reconstruction_confidence":0.0,"confidence":0.8,"reason":null}
"""

DIAGRAM_ROI_PROMPT_TEMPLATE = """
你是 PPT 流程图与关系图识别器。

当前输入是一张 Diagram ROI 图片，region_id：{region_id}。请识别当前 ROI 中的节点、连接线、箭头、标签和整体关系结构。
{pixel_rules}
要求：节点必须作为独立元素输出；连线必须作为 line 或 arrow 单独输出；箭头尽量关联 from_element_id 和 to_element_id；文本优先使用 OCR 结果。
输出格式：{"diagram_type":"process_flow","flow_direction":"left_to_right","layout_pattern":"horizontal","nodes":[],"connectors":[],"confidence":0.82}
"""

def build_roi_prompt(template: str, region: dict, roi_width: int, roi_height: int, ocr_json: str = "[]") -> str:
    """Fill ROI prompt placeholders without treating JSON examples as format fields."""
    width = max(1, int(roi_width or 1))
    height = max(1, int(roi_height or 1))
    pixel_rules = (
        ROI_PIXEL_RULES_TEMPLATE
        .replace("{roi_width}", str(width))
        .replace("{roi_height}", str(height))
    )
    return (
        template
        .replace("{region_id}", str(region.get("id", "region")))
        .replace("{region_type}", str(region.get("type", "unknown")))
        .replace("{roi_width}", str(width))
        .replace("{roi_height}", str(height))
        .replace("{pixel_rules}", pixel_rules)
        .replace("{ocr_json}", ocr_json)
    )
