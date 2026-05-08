# author hgh
# version 1.0
import hashlib
import json
import logging
import re
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from modules.agent.constants import MessageCommonFields

logger = logging.getLogger(__name__)


def format_message(msg: BaseMessage) -> str:
    if isinstance(msg, HumanMessage):
        return f"用户: {msg.content}"
    elif isinstance(msg, AIMessage):
        prefix = "助手"
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            prefix += " [调用工具]"
        return f"{prefix}: {msg.content}"
    elif isinstance(msg, ToolMessage):
        return f"工具结果({msg.name}): {msg.content}"
    else:
        return f"系统: {msg.content}"


def get_message_index(msg: BaseMessage) -> Optional[int]:
    """安全获取消息的全局序号"""
    if hasattr(msg, MessageCommonFields.ADDITIONAL_KWARGS.value) and isinstance(msg.additional_kwargs, dict):
        return msg.additional_kwargs.get(MessageCommonFields.MESSAGE_INDEX.value)
    return None


def safe_parse_extraction_output(raw_output: str) -> list:
    """安全解析 LLM 提取输出，处理常见格式变异"""

    if not raw_output or len(raw_output) == 0:
        return []

    # 1. parse directly
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

