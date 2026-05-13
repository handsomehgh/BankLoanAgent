# author hgh
# version 1.0
import logging

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from config.global_constant.constants import MemoryType, ConfigFields
from config.models.memory_config import MemorySystemConfig
from modules.agent.constants import StateFields, MessageCommonFields
from modules.agent.state import AgentState
from modules.module_services.chat_models import RobustLLM
from config.prompts.system_prompt import SYSTEM_TEMPLATE
from utils.monitor_utils.metrics import record_llm_metrics
from utils.serialize_utils.seq_generator import SequenceGenerator

logger = logging.getLogger(__name__)

def call_model_node(
        state: AgentState,
        config: RunnableConfig,
        memory_config: MemorySystemConfig,
        llm_client: RobustLLM,
        seq_generator: SequenceGenerator
) -> dict:
    """call llm"""
    logger.debug("Entering call_model_node with state : %s", state)

    #extract and format context
    formatted = state.get(StateFields.FORMATTED_CONTEXT, {})
    system = SYSTEM_TEMPLATE.format(
        user_profile=formatted.get(MemoryType.USER_PROFILE.value, "暂无"),
        compliance_rule=formatted.get(MemoryType.COMPLIANCE_RULE.value, "暂无"),
        interaction_log=formatted.get(MemoryType.INTERACTION_LOG.value, "暂无"),
        business_knowledge=formatted.get(MemoryType.BUSINESS_KNOWLEDGE.value,"暂无")
    )


    messages = state.get(StateFields.MESSAGES, [])
    total_messages = len(messages)
    if total_messages > memory_config.max_context_messages:
        recent = messages[-memory_config.max_context_messages:]
        logger.info(
            "Truncated message history from %d to %d (max_context_messages=%d)",
            total_messages, len(recent), memory_config.max_context_messages
        )
    else:
        recent = messages

    full_messages = [SystemMessage(content=system)] + recent
    response = llm_client.invoke_with_fallback(
        full_messages,
        fallback_response="Sorry, I am temporarily unable to handle your request. Please try again later."
    )
    logger.debug("LLM response received, content length=%d", len(response.content) if response.content else 0)

    #output metrics
    token_usage = 0
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        token_usage = response.usage_metadata.get("total_tokens", 0)
    provider = getattr(llm_client, 'provider', 'deepseek')

    record_llm_metrics(provider=provider, total_tokens=token_usage)

    # assign global incremental message sequence number for AiMessage
    if not response.additional_kwargs:
        response.additional_kwargs = {}
    user_id = state.get(StateFields.USER_ID.value)
    configurable = config.get(ConfigFields.CONFIGURABLE, {})
    session_id = configurable.get(ConfigFields.THREAD_ID.value, "unknown")
    idx = seq_generator.next_seq(user_id,session_id)
    response.additional_kwargs[MessageCommonFields.MESSAGE_INDEX.value] = idx
    logger.debug("Assigned message_index=%d to AI response",  idx)

    return {StateFields.MESSAGES.value: [response]}
