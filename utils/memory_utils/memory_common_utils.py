# author hgh
# version 1.0
from typing import Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from config.constants import MessageCommonFields


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