# author hgh
# version 1.0
import logging
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agent.state import AgentState
from config.constants import ConfigFields, StateFields, GeneralFieldNames, MemorySource, MemoryStatus, \
    InteractionEventType, MemoryType
from config.settings import agentConfig
from llm.chat_models import get_llm
from memory.base_memory_store import BaseMemoryStore
from memory.classifiers import detect_sentiment
from utils.message_format import format_message

logger = logging.getLogger(__name__)
llm = get_llm()


def log_interaction_node(state: AgentState, config: RunnableConfig, memory_store: BaseMemoryStore):
    """generate a conversation summary and store it in the interaction memory"""
    # obtain session_id
    configurable = config.get(ConfigFields.CONFIGURABLE.value, {})
    session_id = configurable.get(ConfigFields.THREAD_ID.value, "unknown")

    recent = state.get(StateFields.MESSAGES.value, [])
    if len(recent) > agentConfig.interaction_recent_num:
        recent = recent[-agentConfig.interaction_recent_num:]
    conversation = "\n".join(
        format_message(m)
        for m in recent
    )

    try:
        summary_prompt = f"请用一句话总结以下对话核心内容，不要包含冗余信息:\n{conversation}\n\n摘要:"
        summary = llm.invoke([HumanMessage(content=summary_prompt)],
                             fallback_response="Failed to generate conversation summary").content
    except Exception as e:
        logger.error(f"Failed to generate conversation summary: {e}")
        user_parts = [m.content for m in recent if isinstance(m, HumanMessage)]
        summary = f"用户询问：{'；'.join(user_parts[:2])}" if user_parts else "对话摘要生成失败"

    # dynamic detect sentiment
    sentiment = detect_sentiment(summary)
    metadata = {
        GeneralFieldNames.SOURCE: MemorySource.AUTO_SUMMARY.value,
        GeneralFieldNames.STATUS: MemoryStatus.ACTIVE.value,
        GeneralFieldNames.CONFIDENCE: 1.0,
        GeneralFieldNames.EVENT_TYPE: InteractionEventType.INQUIRY.value,
        GeneralFieldNames.SESSION_ID: session_id,
        GeneralFieldNames.SENTIMENT: sentiment,
        GeneralFieldNames.KEY_ENTITIES: [],
        GeneralFieldNames.TIMESTAMP: datetime.now().isoformat(),
    }

    try:
        memory_store.add_memory(
            user_id=state.get(StateFields.USER_ID.value),
            content=summary,
            memory_type=MemoryType.INTERACTION_LOG,
            metadata=metadata
        )
        logger.info(f"Logged interaction for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to write interaction log: {e}")
    return {"interaction_logged": True}
