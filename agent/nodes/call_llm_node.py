# author hgh
# version 1.0
import logging

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from agent.state import AgentState
from config.constants import StateFields, MemoryType, MessageCommonFields
from config.settings import agentConfig
from llm.chat_models import get_llm
from prompt.system_prompt import SYSTEM_TEMPLATE

logger = logging.getLogger(__name__)
llm = get_llm()

def call_model_node(state: AgentState, config: RunnableConfig) -> dict:
    """call llm"""
    formatted = state.get(StateFields.FORMATTED_CONTEXT.value, {})
    system = SYSTEM_TEMPLATE.format(
        user_profile=formatted.get(MemoryType.USER_PROFILE.value, "暂无"),
        compliance_rule=formatted.get(MemoryType.COMPLIANCE_RULE.value, "暂无"),
        interaction_log=formatted.get(MemoryType.INTERACTION_LOG.value, "暂无")
    )

    messages = state.get(StateFields.MESSAGES.value, [])
    recent = messages
    if len(messages) > agentConfig.max_context_messages:
        recent = messages[-agentConfig.max_context_messages:]
    full_messages = [SystemMessage(content=system)] + recent
    response = llm.invoke_with_fallback(full_messages,
                                        fallback_response="Sorry, I am temporarily unable to handle your request. Please try again later.")

    #assign global incremental message sequence number for AiMessage
    next_index = state.get(StateFields.NEXT_MESSAGE_INDEX.value,0)
    if not response.additional_kwargs:
        response.additional_kwargs = {}
    response.additional_kwargs[MessageCommonFields.MESSAGE_INDEX.value] = next_index
    next_index += 1

    return {StateFields.MESSAGES.value: [response],StateFields.NEXT_MESSAGE_INDEX.value: next_index}

