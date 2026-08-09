# author hgh
# version 1.0
import asyncio
import logging
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import HumanMessage, messages_to_dict
from langchain_core.runnables import RunnableConfig

from config.global_constant.fields import CommonFields
from config.global_constant.constants import ConfigFields, MemoryType, CursorType
from config.models.memory_config import MemorySystemConfig
from infra.message_queue import MessageProducer
from modules.agent.constants import StateFields, StreamName, AgentContextFields, AgentName
from modules.agent.multi_agent_state import SupervisorState
from modules.memory.memory_business_store.base_memory_store import BaseMemoryStore
from modules.memory.memory_constant.constants import InteractionEventType, MemorySource, MemoryStatus, \
    InteractionSentiment
from modules.memory.memory_utils.base_memory_utils import get_message_index, format_messages
from modules.memory.memory_utils.cursor_manager import CursorManager
from modules.module_services.SummaryGenerator import SummaryGenerator
from modules.module_services.sentiment_analyser import SentimentAnalyzer
from utils.serialize_utils.write_to_dlq import write_to_local_dlq

logger = logging.getLogger(__name__)


class SummaryInteractionNode:
    """generate a conversation summary and store it in the interaction memory"""

    def __init__(
            self,
            memory_store: BaseMemoryStore,
            memory_config: MemorySystemConfig,
            summary_generator: SummaryGenerator,
            sentiment_analyzer: SentimentAnalyzer,
            message_producer: MessageProducer,
            cursor_manager: CursorManager
    ):
        self.memory_store = memory_store
        self.memory_config = memory_config
        self.summary_generator = summary_generator
        self.sentiment_analyzer = sentiment_analyzer
        self.message_producer = message_producer
        self.cursor_manager = cursor_manager

    async def __call__(self, state: SupervisorState, config: RunnableConfig):
        user_id = state.get(StateFields.USER_ID.value)
        configurable = config.get(ConfigFields.CONFIGURABLE, {})
        session_id = configurable.get(ConfigFields.THREAD_ID.value, "unknown")

        # process sub graph messages
        sub_messages = state.get(StateFields.SUB_MESSAGES.value, {})
        # if the subgraph does not extract a summary, it is cleared each round.
        if not self.memory_config.sub_summary_enabled:
            sub_messages.pop(AgentName.LOAN_ADVISOR.value, None)
            sub_messages.pop(AgentName.AFTER_LOAN.value, None)
            sub_messages.pop(AgentName.RISK_ASSESSMENT.value, None)
        # the subgraph extraction summary clears agents messages that reach the threshold.
        if self.memory_config.sub_summary_enabled and sub_messages:
            agents_over_threshold = [
                (agent, msgs) for agent, msgs in sub_messages.items()
                if len(msgs) >= self.memory_config.sub_summary_threshold
            ]
            if agents_over_threshold:
                for agent, msgs in agents_over_threshold:
                    logger.info("[SummaryInteractionNode] start processing sub interaction")
                    conversation = format_messages(msgs)
                    try:
                        payload = {
                            CommonFields.USER_ID: user_id,
                            CommonFields.SESSION_ID: session_id,
                            CommonFields.CONTENT: conversation,
                            CommonFields.AGENT_NAME: agent,
                            StateFields.MESSAGES.value: messages_to_dict(msgs)
                        }
                        msg_id = await asyncio.to_thread(
                            self.message_producer.publish,
                            stream_name=StreamName.SUB_INTERACTION.value,
                            event_type=StreamName.SUB_INTERACTION.value,
                            payload=payload,
                            trace_id=state.get(AgentContextFields.TRACE_ID.value, "")
                        )
                        logger.info("[SummaryInteractionNode] sub interaction log has send to redis: user=%s",
                                    user_id)
                    except Exception as e:
                        logger.error(
                            "[SummaryInteractionNode] sub interaction log failed send to redis: %s, write to local DLQ",
                            e)
                        write_to_local_dlq(payload, agent)
                    finally:
                        sub_messages.pop(agent, None)

        # process main graph messages
        messages = state.get(StateFields.MESSAGES.value, [])
        if not messages:
            logger.debug("[SummaryInteractionNode] No messages in state, skipping interaction log")
            return {
                StateFields.INTERACTION_LOGGED.value: False,
                StateFields.SUB_MESSAGES.value: sub_messages
            }

        logger.debug("[SummaryInteractionNode] entering log_interaction_node with messages : %s", messages)

        # cursor
        last_logged_index = state.get(StateFields.LAST_LOGGED_MESSAGE_INDEX.value)

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
            logger.debug("[SummaryInteractionNode] cursor set to message_index=%d, start_pos=%d", last_logged_index,
                         start_pos)

        # No messages to extract, return false
        new_context = messages[start_pos:]
        if not new_context:
            logger.debug("[SummaryInteractionNode] no new messages to log")
            return {StateFields.INTERACTION_LOGGED.value: False, StateFields.SUB_MESSAGES.value: sub_messages}

        # idempotency guard: drop messages already marked processed in cursor_manager,
        # this protects against the numeric cursor failing to advance on a previous failed/retried run
        new_context = self._filter_unprocessed(new_context, user_id)
        if not new_context:
            logger.debug("[SummaryInteractionNode] all candidate messages already marked processed, skipping log")
            return {StateFields.INTERACTION_LOGGED.value: False, StateFields.SUB_MESSAGES.value: sub_messages}

        # Returning false if the new user message is less than the minimum withdrawable amount
        new_user_count = sum(1 for m in new_context if isinstance(m, HumanMessage))
        if new_user_count < self.memory_config.interaction_log_min_new_msgs:
            logger.debug(
                "[SummaryInteractionNode] only %d new user msgs, threshold %d, skipping log",
                new_user_count, self.memory_config.interaction_log_min_new_msgs
            )
            return {StateFields.INTERACTION_LOGGED.value: False, StateFields.SUB_MESSAGES.value: sub_messages}

        # If the number of messages to be extracted is greater than the maximum extractable number,only extract the maximum extractable number
        if len(new_context) > self.memory_config.interaction_log_max_context:
            logger.warning(
                "[SummaryInteractionNode] new context length %d exceeds max %d, truncating to recent",
                len(new_context), self.memory_config.interaction_log_max_context
            )
            new_context = new_context[-self.memory_config.interaction_log_max_context:]

        # extract log info
        conversation = format_messages(new_context)
        logger.debug("[SummaryInteractionNode] prepared conversation for summary, length=%d chars", len(conversation))

        # summary interactions
        if self.memory_config.async_log_enabled and self.message_producer:
            logger.info("[SummaryInteractionNode] start asynchronous write of interaction log")
            try:
                payload = {
                    CommonFields.USER_ID: user_id,
                    CommonFields.SESSION_ID: session_id,
                    CommonFields.CONTENT: conversation,
                    StateFields.MESSAGES.value: messages_to_dict(messages)
                }
                msg_id = await asyncio.to_thread(
                    self.message_producer.publish,
                    stream_name=StreamName.INTERACTION_LOG.value,
                    event_type=StreamName.INTERACTION_LOG.value,
                    payload=payload,
                    trace_id=state.get(AgentContextFields.TRACE_ID.value, "")
                )
                logger.info("[SummaryInteractionNode] interaction log has send to redis: user=%s", user_id)
            except Exception as e:
                logger.error("[SummaryInteractionNode] interaction log failed send to redis: %s, write to local DLQ", e)
                write_to_local_dlq(payload, StreamName.INTERACTION_LOG.value)
                # not committed anywhere,keep cursor and processed set untouched so it retries next turn
                return {StateFields.INTERACTION_LOGGED.value: False, StateFields.SUB_MESSAGES.value: sub_messages}
        else:
            logger.info("[SummaryInteractionNode] start synchronous write of interaction log")

            logger.info("[SummaryInteractionNode] generating interaction summary for session_id=%s", session_id)
            summary = await asyncio.to_thread(self.summary_generator.generate, conversation, new_context)
            logger.info("[SummaryInteractionNode] interaction summary generated: '%.60s...'", summary)

            # detect sentiment
            logger.info("[SummaryInteractionNode] analyzing sentiment for summary")
            sentiment = await asyncio.to_thread(self.sentiment_analyzer.analyze, summary)
            logger.info("[SummaryInteractionNode] detected sentiment: %s", sentiment)

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

            # add to memory
            try:
                await asyncio.to_thread(
                    self.memory_store.add_memory,
                    user_id=state.get(StateFields.USER_ID.value),
                    content=summary,
                    memory_type=MemoryType.INTERACTION_LOG,
                    metadata=metadata
                )
                logger.info("[SummaryInteractionNode] logged interaction for session %s", session_id)
            except Exception as e:
                logger.error(
                    "[SummaryInteractionNode] failed to write interaction log for session %s: %s",
                    session_id, e, exc_info=True
                )
                # write failed,keep cursor and processed set untouched so it retries next turn
                return {StateFields.INTERACTION_LOGGED.value: False, StateFields.SUB_MESSAGES.value: sub_messages}

        # commit succeeded: mark processed first,then advance the cursor
        self._mark_processed(new_context, user_id, CursorType.LOGGING)
        last_index = get_message_index(new_context[-1])
        if last_index is None:
            logger.warning(
                "[SummaryInteractionNode] cannot update last_logged_message_index: no message_index for last message")
            return {StateFields.INTERACTION_LOGGED.value: True, StateFields.SUB_MESSAGES.value: sub_messages}
        logger.debug("[SummaryInteractionNode] updated last_logged_message_index to %d", last_index)
        return {
            StateFields.LAST_LOGGED_MESSAGE_INDEX.value: last_index,
            StateFields.HANDOFF_SUMMARY.value: "",
            StateFields.SUB_MESSAGES.value: sub_messages
        }

    def _filter_unprocessed(self, context: List, user_id: Optional[str]) -> List:
        """
        idempotency guard based on CursorManager's processed set:
        drop messages whose global index has already been marked processed,
        this covers the case where the numeric cursor failed to advance on a previous run
        """
        if not user_id:
            return context
        try:
            processed = self.cursor_manager.get_process_at(user_id, CursorType.LOGGING.value)
        except Exception as e:
            logger.warning("[SummaryInteractionNode] failed to read processed set for user=%s: %s", user_id, e)
            return context
        if not processed:
            return context
        return [m for m in context if get_message_index(m) not in processed]

    def _mark_processed(self, context: List, user_id: Optional[str], cursor_type: CursorType) -> None:
        """mark the given messages' global indexes as processed,and trim the processed set"""
        if not user_id:
            return
        seqs = {get_message_index(m) for m in context if get_message_index(m) is not None}
        if not seqs:
            return
        try:
            self.cursor_manager.add_batch_to_processed_set(user_id, cursor_type.value, seqs)
            self.cursor_manager.remove_old_entries(user_id, cursor_type.value, keep_last_n=2000)
        except Exception as e:
            logger.warning("[SummaryInteractionNode] failed to mark processed set for user=%s: %s", user_id, e)
