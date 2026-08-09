import logging
from typing import Dict, Any

from modules.agent.multi_agent_state import AgentContext

logger = logging.getLogger(__name__)


def build_context_vars(context: AgentContext) -> Dict[str, Any]:
    """context变量派生自agent_context,decision节点和reply节点各自调用一次即可,不需要跨节点传递"""
    return {
        "user_profile": context.user_profile_summary or "暂无相关信息",
        "compliance_rule": context.compliance_warnings or "暂无相关信息",
        "interaction_log": context.conversation_summary or "暂无相关信息",
        "business_knowledge": context.retrieved_knowledge or "暂无相关信息",
        "tool_conversation": context.sub_conversation or "暂无相关信息",
        "recent_conversation": context.recent_conversation or "暂无相关信息",
    }
