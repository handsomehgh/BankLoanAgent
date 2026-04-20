# author hgh
# version 1.0
import json
import logging
import re

logger = logging.getLogger(__name__)

def safe_parse_extraction_output(raw_output: str) -> list:
    """安全解析 LLM 提取输出，处理常见格式变异"""
    # 1. 尝试直接解析
    try:
        return json.loads(raw_output.strip())
    except json.JSONDecodeError:
        pass

    # 2. 尝试提取被 Markdown 代码块包裹的 JSON
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_output)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. 尝试找到第一个 [ 和最后一个 ] 之间的内容
    start = raw_output.find("[")
    end = raw_output.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_output[start:end + 1])
        except json.JSONDecodeError:
            pass

    # 4. 解析失败，返回空列表
    logger.warning(f"无法解析提取输出，已返回空列表。原始输出: {raw_output[:200]}")
    return []
