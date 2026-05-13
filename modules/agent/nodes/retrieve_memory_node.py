# author hgh
# version 1.0
import logging

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from config.global_constant.fields import CommonFields
from config.global_constant.constants import MemoryType, ConfigFields
from modules.agent.constants import StateFields, MessageCommonFields
from modules.agent.state import AgentState
from modules.memory.base import BaseRetriever
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)


def retrieve_memory_node(state: AgentState, config: RunnableConfig, retrieval: BaseRetriever,seq_generator: SequenceGenerator) -> dict:
    """
    retrieve memory and assign a globally incrementing sequence to all unnumbered user/assistant mesasges
    """
    logger.debug("Entering retrieve_memory_node with state : %s", state)

    user_id = state.get(StateFields.USER_ID, "unknown")
    logger.info("Retrieving memory for user_id=%s", user_id)
    messages = state.get(StateFields.MESSAGES, [])

    # assign global message sequence
    next_index = state.get(StateFields.NEXT_MESSAGE_INDEX, 0)
    updated = False
    for msg in messages:
        if not hasattr(msg, MessageCommonFields.ADDITIONAL_KWARGS.value) or msg.additional_kwargs is None:
            msg.additional_kwargs = {}
        if MessageCommonFields.MESSAGE_INDEX.value not in msg.additional_kwargs:
            configurable = config.get(ConfigFields.CONFIGURABLE, {})
            session_id = configurable.get(ConfigFields.THREAD_ID.value, "unknown")
            idx = seq_generator.next_seq(user_id,session_id)
            msg.additional_kwargs[MessageCommonFields.MESSAGE_INDEX.value] = idx
            updated = True

    if updated:
        logger.debug("Assigned new message indexes, next_index=%d", next_index)

    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    if user_query:
        logger.info("Memory retrieval query: '%.60s...'", user_query)
    else:
        logger.warning("No user query found in messages, user_id=%s", user_id)

    memory_types = [
        MemoryType.USER_PROFILE,
        MemoryType.COMPLIANCE_RULE,
        MemoryType.INTERACTION_LOG
    ]
    try:
        context = retrieval.retrieve(query=user_query, user_id=user_id, memory_types=memory_types)
        profile_count = len(context.get(MemoryType.USER_PROFILE.value, []))
        rule_count = len(context.get(MemoryType.COMPLIANCE_RULE.value, []))
        interaction_count = len(context.get(MemoryType.INTERACTION_LOG.value, []))
        logger.info(
            "Memory retrieval succeeded: user_profile=%d, compliance_rules=%d, interaction_logs=%d",
            profile_count, rule_count, interaction_count
        )
    except Exception as e:
        logger.error("Retrieval failed, using empty context: %s", e, exc_info=True)
        empty_formatted = {
            MemoryType.USER_PROFILE.value: "暂无相关记录",
            MemoryType.COMPLIANCE_RULE.value: "暂无相关记录",
            MemoryType.INTERACTION_LOG.value: "暂无相关记录"
        }
        return {
            StateFields.RETRIEVED_CONTEXT.value: {
                MemoryType.USER_PROFILE.value: [],
                MemoryType.COMPLIANCE_RULE.value: [],
                MemoryType.INTERACTION_LOG.value: []
            },
            StateFields.FORMATTED_CONTEXT.value: empty_formatted,
            StateFields.ERROR.value: f"Retrieval error: {e}",
            StateFields.NEXT_MESSAGE_INDEX.value: next_index if updated else state.get(StateFields.NEXT_MESSAGE_INDEX)
        }

    def fmt(mems):
        return "\n".join(f"- {m[CommonFields.TEXT]}" for m in mems) if mems else "暂无相关记录"

    formatted = {
        MemoryType.USER_PROFILE.value: fmt(context.get(MemoryType.USER_PROFILE.value, [])),
        MemoryType.COMPLIANCE_RULE.value: fmt(context.get(MemoryType.COMPLIANCE_RULE.value, [])),
        MemoryType.INTERACTION_LOG.value: fmt(context.get(MemoryType.INTERACTION_LOG.value, []))
    }
    logger.debug("Formatted memory contexts successfully")
    return {
        StateFields.RETRIEVED_CONTEXT.value: context,
        StateFields.FORMATTED_CONTEXT.value: formatted,
        StateFields.NEXT_MESSAGE_INDEX.value: next_index,
        StateFields.ERROR.value: None
    }
