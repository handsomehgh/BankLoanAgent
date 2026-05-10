# author hgh
# version 1.0
import logging
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig

from config.global_constant.fields import CommonFields
from config.global_constant.constants import MemoryType
from config.models.memory_config import MemorySystemConfig
from modules.agent.constants import MessageCommonFields, StateFields
from modules.agent.state import AgentState
from exceptions.exception import MemoryWriteFailedError
from modules.memory.memory_business_store.base_memory_store import BaseMemoryStore
from modules.memory.memory_constant.constants import ProfileEntityKey, MemorySource, MemoryStatus, EvidenceType
from modules.memory.memory_utils.base_memory_utils import get_message_index, format_message, \
    safe_parse_extraction_output
from modules.memory.memory_utils.profile_gate_util import ProfileGate
from modules.module_services.evidence_infer import EvidenceTypeInfer
from modules.module_services.profile_extractor import ProfileExtractor

logger = logging.getLogger(__name__)


def _get_new_user_messages(
        messages: List[BaseMessage],
        last_extracted_index: Optional[int],
        fallback_window: int = 10
) -> List[BaseMessage]:
    """
    retrieve unprocessed messages after last_extracted_index,prefer using global message_index,
    if unavailable,fall back to the message ID
    """
    user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    if not user_msgs:
        return []

    if last_extracted_index is None:
        recent = messages[-fallback_window:] if len(messages) > fallback_window else messages
        return [m for m in recent if isinstance(m, HumanMessage)]

    all_have_index = all(
        hasattr(m, MessageCommonFields.ADDITIONAL_KWARGS.value) and isinstance(m.additional_kwargs, dict)
        and MessageCommonFields.MESSAGE_INDEX.value in m.additional_kwargs
        for m in user_msgs
    )

    if all_have_index:
        return [
            m for m in user_msgs if
            m.additional_kwargs.get(MessageCommonFields.MESSAGE_INDEX.value) > last_extracted_index
        ]
    else:
        # if there are messages without a global index,return the fallback window size
        logger.warning(
            "some messages lack message_index despite cursor being set."
            f"Falling back to recent{fallback_window} messages"
        )
        recent = messages[-fallback_window:] if len(messages) > fallback_window else messages
        return [m for m in recent if isinstance(m, HumanMessage)]


def extract_profile_node(
        state: AgentState,
        config: RunnableConfig,
        memory_store: BaseMemoryStore,
        profile_gate: ProfileGate,
        memory_config: MemorySystemConfig,
        evidence_infer: EvidenceTypeInfer,
        profile_extractor: ProfileExtractor
) -> dict:
    """extract user profile and save to store"""
    # get user id
    logger.debug("Entering extract_profile_node with state : %s", state)

    user_id = state.get(StateFields.USER_ID)
    if not user_id:
        logger.warning("No user_id found in state, skipping profile extraction")
        return {StateFields.PROFILE_UPDATED.value: False}

    # no messages,return false
    messages = state.get(StateFields.MESSAGES, [])
    if not messages:
        logger.debug("No messages to extract profile from")
        return {StateFields.PROFILE_UPDATED.value: False}

    logger.debug("Starting profile extraction for user_id=%s, total_messages=%d", user_id, len(messages))

    # cursor: prefer cross-session(reserved),otherwise state
    cursor = memory_store.get_extraction_cursor(user_id)
    if cursor is None:
        cursor = state.get(StateFields.LAST_EXTRACTED_MESSAGE_INDEX)
    logger.debug("Extraction cursor: %s", cursor)

    # no new messages,return false
    new_user_messages = _get_new_user_messages(messages, cursor, memory_config.profile_extraction_fallback_window)
    if not new_user_messages:
        logger.info("No new user messages for user_id=%s, skipping profile extraction", user_id)
        return {StateFields.PROFILE_UPDATED.value: False}

    logger.info("Found %d new user messages for user_id=%s", len(new_user_messages), user_id)

    # lightweight filtering
    if not profile_gate.should_extract(new_user_messages):
        logger.info("Profile gate filtered out all messages for user_id=%s (msg_count=%d)", user_id, len(new_user_messages))
        last_msg = messages[-1]
        last_index = get_message_index(last_msg)
        if last_index is None:
            logger.warning("Cannot update cursor after filter skip: no message_index for user_id=%s", user_id)
            return {StateFields.PROFILE_UPDATED.value: False}
        return {
            StateFields.PROFILE_UPDATED.value: True,
            StateFields.LAST_EXTRACTED_MESSAGE_INDEX.value: last_index
        }

    # build context window
    first_idx = messages.index(new_user_messages[0])
    last_idx = messages.index(new_user_messages[-1])
    context_start = max(0, first_idx - 2)
    context_end = min(len(messages), last_idx + 3)
    context_messages = messages[context_start:context_end]
    conversations = "\n".join(
        format_message(m)
        for m in context_messages
    )
    logger.debug("Built context window for extraction: %d messages", len(context_messages))

    # obtain a desensitized profile summary
    known_profile = "暂无已知用户画像"
    try:
        summary = memory_store.get_profile_summary(user_id)
        if summary:
            known_profile = summary
            logger.debug("Using existing profile summary for user_id=%s", user_id)
    except Exception as e:
        logger.warning("Failed to get profile summary for user_id=%s, using default: %s", user_id, e)

    # llm extract
    logger.info("Calling LLM for profile extraction: user_id=%s, msg_count=%d", user_id, len(new_user_messages))
    extract_str = profile_extractor.extract(conversations, known_profile)
    logger.debug("LLM extraction response (first 200 chars): %.200s", extract_str)

    # parsing,verification
    items = safe_parse_extraction_output(extract_str)
    allowed_entity_keys = {e.value for e in ProfileEntityKey}
    updated = False

    logger.info("Extracted %d potential profile items for user_id=%s", len(items), user_id)
    for item in items:
        content = item.get(CommonFields.CONTENT)
        entity_key_raw = item.get(CommonFields.ENTITY_KEY)
        if not content or not entity_key_raw:
            continue
        if entity_key_raw not in allowed_entity_keys:
            logger.warning("Ignored invalid entity_key '%s' for user_id=%s", entity_key_raw, user_id)
            continue

        #infer evidence type
        evidence_type = evidence_infer.infer(content,[m.content for m in new_user_messages])
        logger.debug("Inferred evidence type '%s' for entity '%s'", evidence_type, entity_key_raw)

        metadata = {
            CommonFields.SOURCE: MemorySource.CHAT_EXTRACTION,
            CommonFields.CONFIDENCE: item.get(CommonFields.CONFIDENCE, 0.7),
            CommonFields.STATUS: MemoryStatus.ACTIVE,
            CommonFields.EVIDENCE_TYPE: EvidenceType(evidence_type),
            CommonFields.EFFECTIVE_DATE: datetime.now().isoformat(),
            CommonFields.EXPIRES_AT: None,
        }

        # insert the extract profile
        try:
            memory_store.add_memory(
                user_id=user_id,
                content=content,
                memory_type=MemoryType.USER_PROFILE,
                entity_key=ProfileEntityKey(entity_key_raw),
                metadata=metadata
            )
            updated = True
            logger.info("Added profile memory for user_id=%s, entity=%s", user_id, entity_key_raw)
        except MemoryWriteFailedError as e:
            logger.error("Memory write failed (DLQ) for user_id=%s: %s", user_id, e, exc_info=True)
        except Exception as e:
            logger.error("Unexpected error during profile extraction for user_id=%s: %s", user_id, e, exc_info=True)

        if updated:
            logger.info("Profile updated for user_id=%s: %d new items", user_id, len(items))
        else:
            logger.info("No new profile information for user_id=%s", user_id)

    # update cursor
    last_msg = messages[-1]
    last_index = get_message_index(last_msg)
    if last_index is None:
        logger.warning("Cannot update cursor after extraction: no message_index for user_id=%s", user_id)
        return {StateFields.PROFILE_UPDATED.value: updated}

    # update cross-session cursor
    try:
        memory_store.set_extraction_cursor(user_id, last_index)
        logger.debug("Updated extraction cursor for user_id=%s to %d", user_id, last_index)
    except Exception as e:
        logger.warning("Failed to set extraction cursor for user_id=%s: %s", user_id, e)
        pass

    return {
        StateFields.PROFILE_UPDATED.value: updated,
        StateFields.LAST_EXTRACTED_MESSAGE_INDEX.value: last_index
    }
