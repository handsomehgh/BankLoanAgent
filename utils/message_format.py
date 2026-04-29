# author hgh
# version 1.0
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, ToolMessage


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
