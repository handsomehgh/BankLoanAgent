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
from modules.memory.memory_constant.constants import InteractionEventType, MemorySource, MemoryStatus, \
    InteractionSentiment
from modules.memory.memory_utils.base_memory_utils import get_message_index, format_message
from modules.module_services.SummaryGenerator import SummaryGenerator
from modules.module_services.sentiment_analyser import SentimentAnalyzer

logger = logging.getLogger(__name__)

def log_interaction_node(
        state: AgentState,
        config: RunnableConfig,
        memory_store: BaseMemoryStore,
        memory_config: MemorySystemConfig,
        summary_generator: SummaryGenerator,
        sentiment_analyzer: SentimentAnalyzer):
    """generate a conversation summary and store it in the interaction memory"""
    # obtain session_id
    logger.debug("Entering log_interaction_node with state : %s", state)

    configurable = config.get(ConfigFields.CONFIGURABLE, {})
    session_id = configurable.get(ConfigFields.THREAD_ID.value, "unknown")
    logger.debug("Entering log_interaction_node for session_id=%s", session_id)

    # No message returned, interaction_logged flag is false
    messages = state.get(StateFields.MESSAGES, [])
    if not messages:
        logger.debug("No messages in state, skipping interaction log")
        return {StateFields.INTERACTION_LOGGED.value: False}

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
        logger.debug("Cursor set to message_index=%d, start_pos=%d", last_logged_index, start_pos)

        # No messages to extract, return false
    new_context = messages[start_pos:]
    if not new_context:
        logger.debug("No new messages to log")
        return {StateFields.INTERACTION_LOGGED.value: False}

    # Returning false if the new user message is less than the minimum withdrawable amount
    new_user_count = sum(1 for m in new_context if isinstance(m, HumanMessage))
    if new_user_count < memory_config.interaction_log_min_new_msgs:
        logger.debug(
            "Only %d new user msgs, threshold %d, skipping log",
            new_user_count, memory_config.interaction_log_min_new_msgs
        )
        return {StateFields.INTERACTION_LOGGED.value: False}

    # If the number of messages to be extracted is greater than the maximum extractable number,only extract the maximum extractable number
    if len(new_context) > memory_config.interaction_log_max_context:
        logger.warning(
            "New context length %d exceeds max %d, truncating to recent",
            len(new_context), memory_config.interaction_log_max_context
        )
        new_context = new_context[-memory_config.interaction_log_max_context:]

    # extract log info
    conversation = "\n".join(
        format_message(m)
        for m in new_context
    )
    logger.debug("Prepared conversation for summary, length=%d chars", len(conversation))

    #summary interactions
    logger.info("Generating interaction summary for session_id=%s", session_id)
    summary = summary_generator.generate(conversation,new_context)
    logger.info("Interaction summary generated: '%.60s...'", summary)

    # detect sentiment
    logger.debug("Analyzing sentiment for summary")
    sentiment = sentiment_analyzer.analyze(summary)
    logger.info("Detected sentiment: %s", sentiment)

    # build log memory data
    metadata = {
        CommonFields.SOURCE: MemorySource.AUTO_SUMMARY,
        CommonFields.STATUS: MemoryStatus.ACTIVE,
        CommonFields.CONFIDENCE: 1.0,
        CommonFields.EVENT_TYPE: InteractionEventType.INQUIRY,
        CommonFields.SESSION_ID: session_id,
        CommonFields.SENTIMENT: InteractionSentiment(sentiment),
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
        logger.info("Logged interaction for session %s", session_id)
    except Exception as e:
        logger.error(
            "Failed to write interaction log for session %s: %s",
            session_id, e, exc_info=True
        )

        # update last_logged_message_index
    last_index = get_message_index(new_context[-1])
    logger.debug("Updated last_logged_message_index to %d", last_index)
    return {
        StateFields.INTERACTION_LOGGED.value: True,
        StateFields.LAST_LOGGED_MESSAGE_INDEX.value: last_index
    }
