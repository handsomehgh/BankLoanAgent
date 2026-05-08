# author hgh
# version 1.0
import logging

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from config.global_constant.constants import MemoryType
from config.models.memory_config import MemorySystemConfig
from modules.agent.constants import StateFields, MessageCommonFields
from modules.agent.state import AgentState
from modules.module_services.chat_models import RobustLLM
from config.prompts.system_prompt import SYSTEM_TEMPLATE

logger = logging.getLogger(__name__)

def call_model_node(state: AgentState, config: RunnableConfig, memory_config: MemorySystemConfig,llm_client: RobustLLM) -> dict:
    """call llm"""
    print(f"=======================\n{state}\n==========================")
    formatted = state.get(StateFields.FORMATTED_CONTEXT, {})
    system = SYSTEM_TEMPLATE.format(
        user_profile=formatted.get(MemoryType.USER_PROFILE.value, "暂无"),
        compliance_rule=formatted.get(MemoryType.COMPLIANCE_RULE.value, "暂无"),
        interaction_log=formatted.get(MemoryType.INTERACTION_LOG.value, "暂无"),
        business_knowledge=formatted.get(MemoryType.BUSINESS_KNOWLEDGE.value,"暂无")
    )

    messages = state.get(StateFields.MESSAGES, [])
    recent = messages
    if len(messages) > memory_config.max_context_messages:
        recent = messages[-memory_config.max_context_messages:]
    full_messages = [SystemMessage(content=system)] + recent
    response = llm_client.invoke_with_fallback(full_messages,
                                        fallback_response="Sorry, I am temporarily unable to handle your request. Please try again later.")

    # assign global incremental message sequence number for AiMessage
    next_index = state.get(StateFields.NEXT_MESSAGE_INDEX, 0)
    if not response.additional_kwargs:
        response.additional_kwargs = {}
    response.additional_kwargs[MessageCommonFields.MESSAGE_INDEX.value] = next_index
    next_index += 1

    return {StateFields.MESSAGES.value: [response], StateFields.NEXT_MESSAGE_INDEX.value: next_index}
