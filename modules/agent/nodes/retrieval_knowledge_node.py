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

    try:
        knowledge_list: list = retrieval_service.retrieve(user_query)
        logger.info(f"Retrieval success: {len(knowledge_list)} documents")
    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        return _empty_result_for_knowledge(state)

    formatted = format_context(knowledge_list,retrieval_config.rag_max_context_length)

    existing_context = state.get(StateFields.RETRIEVED_CONTEXT, {})
    existing_formatted = state.get(StateFields.FORMATTED_CONTEXT, {})

    existing_context[MemoryType.BUSINESS_KNOWLEDGE] = knowledge_list
    existing_formatted[MemoryType.BUSINESS_KNOWLEDGE] = formatted

    return {
        StateFields.RETRIEVED_CONTEXT.value: {MemoryType.BUSINESS_KNOWLEDGE.value,existing_context},
        StateFields.FORMATTED_CONTEXT.value: {MemoryType.BUSINESS_KNOWLEDGE.value,existing_formatted},
    }


def _empty_result_for_knowledge(state: AgentState) -> dict:
    existing_context = state.get(StateFields.RETRIEVED_CONTEXT, {})
    existing_formatted = state.get(StateFields.FORMATTED_CONTEXT, {})
    existing_context[MemoryType.BUSINESS_KNOWLEDGE.value] = []
    existing_formatted[MemoryType.BUSINESS_KNOWLEDGE.value] = "暂无相关知识"
    return {
        StateFields.RETRIEVED_CONTEXT.value: existing_context,
        StateFields.FORMATTED_CONTEXT.value: existing_formatted
    }

