"""VLM-only page structure recognition prompts."""

_VLM_JSON_RULES = """
输出必须是一个可被 json.loads 直接解析的 JSON 对象。不要输出 Markdown、解释、注释或额外文本。
所有 bbox 均使用 0~1000 归一化坐标，格式为 {"x":left,"y":top,"width":w,"height":h}，整数且限制在 0~1000。
不确定的元素不要猜；每个返回项必须包含 confidence。
"""

VLM_PAGE_REGIONS_PROMPT = f"""
你是 PPT/网页/信息图的页面粗结构识别器。只识别大块区域，不识别区域内部的小元素。

任务：识别 background、header、footer、sidebar、section、card、panel、table、chart、diagram、group、image 区域。
不要输出文字内容、箭头端点、详细样式或内部结构。
{_VLM_JSON_RULES}

JSON schema 示例：
{{"regions":[{{"id":"r1","type":"card","semantic_type":"metric_card","bbox":{{"x":80,"y":120,"width":260,"height":140}},"confidence":0.9}}]}}
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
