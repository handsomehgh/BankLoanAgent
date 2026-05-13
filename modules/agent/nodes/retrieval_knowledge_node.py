# author hgh
# version 1.0
import logging

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from config.global_constant.constants import MemoryType
from config.models.retrieval_config import RetrievalConfig
from modules.agent.constants import StateFields
from modules.agent.state import AgentState
from modules.retrieval.knowledge_utils.knowledge_formatter import format_context
from modules.retrieval.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


def retrieval_knowledge_node(
        state: AgentState,
        config: RunnableConfig,
        retrieval_service: RetrievalService,
        retrieval_config: RetrievalConfig):
    """
    search the knowledge base and write the results back to the state

    Args:
        state: current state of agent
        config: LangGraph runnable configuration
        retrieval_service: retrieves business services instance
        retrieval_config: retrieval config

    Returns:
        state update dictionary
    """
    logger.debug("Entering retrieval_knowledge_node with state : %s", state)

    messages = state.get(StateFields.MESSAGES,[])
    if not messages:
        logger.warning("No messages in state, skip retrieval")
        return _empty_result_for_knowledge(state)

    last_msg = messages[-1]
    if not isinstance(last_msg, HumanMessage):
        logger.warning("Last message is not HumanMessage, skip retrieval")
        return _empty_result_for_knowledge(state)

    user_query = last_msg.content
    if not user_query or not user_query.strip():
        logger.warning("Empty user query, skip retrieval")
        return _empty_result_for_knowledge(state)

    formatted_context = state.get(StateFields.FORMATTED_CONTEXT, {})
    last_summary = formatted_context.get(MemoryType.INTERACTION_LOG.value, "")
    placeholder_summaries = ("暂无相关记录", "暂无信息")
    has_context = last_summary and last_summary not in placeholder_summaries
    context = {"last_summary": last_summary} if has_context else None

    logger.info(
        "Knowledge retrieval starting: query='%.80s...', context=%s",
        user_query, "available" if context else "absent"
    )
    try:
        knowledge_list: list = retrieval_service.retrieve(user_query,context)
        logger.info("Retrieval success: %d documents", len(knowledge_list))
    except Exception as e:
        logger.error("Retrieval failed: %s", e, exc_info=True)
        return _empty_result_for_knowledge(state)

    formatted = format_context(knowledge_list,retrieval_config.rag_max_context_length)
    logger.debug("Formatted knowledge context, length=%d chars", len(formatted))

    existing_context = state.get(StateFields.RETRIEVED_CONTEXT, {})
    existing_formatted = state.get(StateFields.FORMATTED_CONTEXT, {})

    existing_context[MemoryType.BUSINESS_KNOWLEDGE.value] = knowledge_list
    existing_formatted[MemoryType.BUSINESS_KNOWLEDGE.value] = formatted

    return {
        StateFields.RETRIEVED_CONTEXT.value: existing_context,
        StateFields.FORMATTED_CONTEXT.value: existing_formatted
    }


def _empty_result_for_knowledge(state: AgentState) -> dict:
    existing_context = state.get(StateFields.RETRIEVED_CONTEXT, {})
    existing_formatted = state.get(StateFields.FORMATTED_CONTEXT, {})
    existing_context[MemoryType.BUSINESS_KNOWLEDGE.value] = []
    existing_formatted[MemoryType.BUSINESS_KNOWLEDGE.value] = "暂无相关记录"
    logger.debug("Returning empty knowledge context")
    return {
        StateFields.RETRIEVED_CONTEXT.value: existing_context,
        StateFields.FORMATTED_CONTEXT.value: existing_formatted
    }

