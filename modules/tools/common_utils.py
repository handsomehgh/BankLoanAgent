# author hgh
# version 1.0
import logging
from typing import List, Dict

from config.models.agent_config import AgentConfig
from modules.agent.multi_agent_state import AgentContext
from modules.tools.tool_selector import ToolSelector
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)

def get_agent_tools(agent_cfg: AgentConfig,tool_selector: ToolSelector,caller_agent: str):
    exposure_mode = getattr(agent_cfg, 'tool_exposure', 'skills')
    return tool_selector.get_tools(caller_agent, exposure_mode)

def get_tools_metadata(
    agent_cfg: AgentConfig,
    caller_agent: str,
    tool_selector: ToolSelector
) -> List[Dict[str, str]]:
    tools = get_agent_tools(agent_cfg, tool_selector,caller_agent)
    metadata = []
    for tool in tools:
        metadata.append({
            "name": tool.name,
            "description": tool.description
        })
    return metadata

def assign_message_index(msg, user_id: str, session_id: str, seq_generator: SequenceGenerator):
    """assign global message index"""
    if not hasattr(msg, "additional_kwargs") or msg.additional_kwargs is None:
        msg.additional_kwargs = {}
    if "message_index" not in msg.additional_kwargs:
        msg.additional_kwargs["message_index"] = seq_generator.next_seq(user_id, session_id)

def build_text_a_for_bert(context: AgentContext) -> str:
    """为 BERT 分类器构造 text_a"""
    parts = []
    if context.user_profile_summary and context.user_profile_summary != "暂无相关信息":
        parts.append(f"用户画像：{context.user_profile_summary[:150]}")
    if context.conversation_summary and context.conversation_summary != "暂无相关信息":
        parts.append(f"对话摘要：{context.conversation_summary[:150]}")
    if context.sub_conversation and context.sub_conversation != "暂无相关信息":
        parts.append(f"近期工具操作：{context.sub_conversation[:100]}")
    if context.recent_conversation and context.recent_conversation != "暂无相关信息":
        parts.append(f"最近对话：{context.recent_conversation[-200:]}")
    return "；".join(parts) if parts else ""