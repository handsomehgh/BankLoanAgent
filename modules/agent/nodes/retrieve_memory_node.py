# author hgh
# version 1.0
import logging

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from config.global_constant.common_fields import CommonFields
from config.global_constant.constants import MemoryType
from modules.agent.agent_constant.constants import StateFields, MessageCommonFields
from modules.agent.state import AgentState
from modules.memory.base import BaseRetriever

logger = logging.getLogger(__name__)


def retrieve_memory_node(state: AgentState, config: RunnableConfig, retrieval: BaseRetriever) -> dict:
    """
    retrieve memory and assign a globally incrementing sequence to all unnumbered user/assistant mesasges
    """
    print(f"=======================\n{state}\n==========================")
    user_id = state.get(StateFields.USER_ID, "unknown")
    messages = state.get(StateFields.MESSAGES, [])

    # assign global message sequence
    next_index = state.get(StateFields.NEXT_MESSAGE_INDEX, 0)
    updated = False
    for msg in messages:
        if not hasattr(msg, MessageCommonFields.ADDITIONAL_KWARGS.value) or msg.additional_kwargs is None:
            msg.additional_kwargs = {}
        if MessageCommonFields.MESSAGE_INDEX.value not in msg.additional_kwargs:
            msg.additional_kwargs[MessageCommonFields.MESSAGE_INDEX.value] = next_index
            next_index += 1
            updated = True

    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    memory_types = [
        MemoryType.USER_PROFILE,
        MemoryType.COMPLIANCE_RULE,
        MemoryType.INTERACTION_LOG
    ]
    try:
        context = retrieval.retrieve(query=user_query, user_id=user_id, memory_types=memory_types)
    except Exception as e:
        logger.error(f"Retrieval failed, using empty context: {e}")
        context = {MemoryType.USER_PROFILE.value: [], MemoryType.COMPLIANCE_RULE.value: [],
                   MemoryType.INTERACTION_LOG.value: []}
        return {
            StateFields.RETRIEVED_CONTEXT.value: context,
            StateFields.FORMATTED_CONTEXT.value: {
                MemoryType.USER_PROFILE.value: "暂无相关信息",
                MemoryType.COMPLIANCE_RULE.value: "暂无相关信息",
                MemoryType.INTERACTION_LOG.value: "暂无相关信息"
            },
            StateFields.ERROR.value: f"Retrieval error: {e}",
            StateFields.NEXT_MESSAGE_INDEX.value: next_index if updated else state.get(StateFields.NEXT_MESSAGE_INDEX)
        }

    def fmt(mems):
        return "\n".join(f"- {m[CommonFields.TEXT]}" for m in mems) if mems else "暂无相关信息"

    formatted = {
        MemoryType.USER_PROFILE.value: fmt(context.get(MemoryType.USER_PROFILE.value, [])),
        MemoryType.COMPLIANCE_RULE.value: fmt(context.get(MemoryType.COMPLIANCE_RULE.value, [])),
        MemoryType.INTERACTION_LOG.value: fmt(context.get(MemoryType.INTERACTION_LOG.value, []))
    }
    return {
        StateFields.RETRIEVED_CONTEXT.value: context,
        StateFields.FORMATTED_CONTEXT.value: formatted,
        StateFields.NEXT_MESSAGE_INDEX.value: next_index,
        StateFields.ERROR.value: None
    }
