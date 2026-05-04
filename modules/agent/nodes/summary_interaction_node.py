# author hgh
# version 1.0
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from config.global_constant.fields import CommonFields
from config.global_constant.constants import ConfigFields, MemoryType
from config.models.memory_config import MemorySystemConfig
from modules.agent.constants import StateFields
from modules.agent.state import AgentState
from modules.memory.memory_business_store.base_memory_store import BaseMemoryStore
from modules.memory.memory_constant.constants import InteractionEventType, MemorySource, MemoryStatus
from modules.memory.memory_utils.base_memory_utils import get_message_index, format_message, detect_sentiment
from modules.module_services.chat_models import get_llm

logger = logging.getLogger(__name__)
llm = get_llm()


def log_interaction_node(state: AgentState, config: RunnableConfig, memory_store: BaseMemoryStore,memory_config: MemorySystemConfig):
    """generate a conversation summary and store it in the interaction memory"""
    # obtain session_id
    configurable = config.get(ConfigFields.CONFIGURABLE, {})
    session_id = configurable.get(ConfigFields.THREAD_ID.value, "unknown")

    # No message returned, interaction_logged flag is false
    messages = state.get(StateFields.MESSAGES, [])
    if not messages:
        return {"interaction_logged": False}

    # cursor
    last_logged_index = state.get(StateFields.LAST_LOGGED_MESSAGE_INDEX)

    # find cursor index
    start_pos = 0
    if last_logged_index is not None:
        for i, m in enumerate(messages):
            idx = get_message_index(m)
            if idx is None:
                continue
            if idx > last_logged_index:
                start_pos = i
                break
        else:
            start_pos = len(messages)

    # No messages to extract, return false
    new_context = messages[start_pos:]
    if not new_context:
        return {"interaction_logged": False}

    # Returning false if the new user message is less than the minimum withdrawable amount
    new_user_count = sum(1 for m in new_context if isinstance(m, HumanMessage))
    if new_user_count < memory_config.interaction_log_min_new_msgs:
        logger.debug(f"Only {new_user_count} new user msgs, threshold {memory_config.interaction_log_min_new_msgs}")
        return {"interaction_logged": False}

    # If the number of messages to be extracted is greater than the maximum extractable number,only extract the maximum extractable number
    if len(new_context) > memory_config.interaction_log_max_context:
        logger.warning(
            f"New context length {len(new_context)} exceeds max {memory_config.interaction_log_max_context}, truncating to recent")
        new_context = new_context[-memory_config.interaction_log_max_context:]

    # extract log info
    conversation = "\n".join(
        format_message(m)
        for m in new_context
    )
    try:
        summary_prompt = f"请用一句话总结以下对话核心内容，不要包含冗余信息:\n{conversation}\n\n摘要:"
        summary = llm.invoke([HumanMessage(content=summary_prompt)],
                             fallback_response="Failed to generate conversation summary").content
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        user_parts = [m.content for m in new_context if isinstance(m, HumanMessage)]
        summary = f"用户询问：{'；'.join(user_parts[:2])}" if user_parts else "对话摘要生成失败"

    # detect sentiment
    sentiment = detect_sentiment(summary,memory_config.sentiment_rules.strong_keywords)

    # build log memory data
    metadata = {
        CommonFields.SOURCE: MemorySource.AUTO_SUMMARY.value,
        CommonFields.STATUS: MemoryStatus.ACTIVE.value,
        CommonFields.CONFIDENCE: 1.0,
        CommonFields.EVENT_TYPE: InteractionEventType.INQUIRY.value,
        CommonFields.SESSION_ID: session_id,
        CommonFields.SENTIMENT: sentiment,
        CommonFields.KEY_ENTITIES: [],
        CommonFields.TIMESTAMP: datetime.now().isoformat(),
    }

    #add to memory
    try:
        memory_store.add_memory(
            user_id=state.get(StateFields.USER_ID),
            content=summary,
            memory_type=MemoryType.INTERACTION_LOG,
            metadata=metadata
        )
        logger.info(f"Logged interaction for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to write interaction log: {'interaction_logged': True}")

    # update last_logged_message_index
    last_index = get_message_index(new_context[-1])
    return {
        StateFields.INTERACTION_LOGGED.value: True,
        StateFields.LAST_LOGGED_MESSAGE_INDEX.value: last_index
    }
