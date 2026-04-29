# author hgh
# version 1.0
import logging
import re
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from agent.state import AgentState
from config.constants import StateFields, PromptKeys, ProfileEntityKey, GeneralFieldNames, MemorySource, MemoryStatus, \
    MemoryType, MessageCommonFields
from config.settings import agentConfig
from exceptions.exception import MemoryWriteFailedError
from llm.chat_models import get_llm
from memory.base_memory_store import BaseMemoryStore
from memory.classifiers import infer_evidence_type
from prompt.extract_prompt import EXTRACT_PROMPT
from utils.message_format import format_message
from utils.parser import safe_parse_extraction_output

logger = logging.getLogger(__name__)
llm = get_llm()

_PROFILE_SIGNALS = re.compile(
    r"(我是|我叫|我姓|我在|我今年|我住|我电话|我邮箱|我职业|我工作|我爱好|"
    r"年龄|生日|地址|手机号|月入|年薪|收入|贷款|负债|资产|"
    r"公司|岗位|职位|偏好|想要|需要|计划|职业|行业|"
    r"利率|期限|信用|征信|变更|更新|更改|修改|改成)",
    re.IGNORECASE
)


def _likely_contains_profile(user_messages: List[BaseMessage]) -> bool:
    for m in user_messages:
        if _PROFILE_SIGNALS.search(m.content):
            return True
        return False


def _get_new_user_messages(
        messages: List[BaseMessage],
        last_extracted_index: Optional[int],
        fallback_window: int = None
) -> List[BaseMessage]:
    """
    retrieve unprocessed messages after last_extracted_index,prefer using global message_index,
    if unavailable,fall back to the message ID
    """
    user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
    if not user_msgs:
        return []

    if fallback_window is None:
        fallback_window = agentConfig.profile_extraction_fallback_window

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


def extract_profile_node(state: AgentState, config: RunnableConfig, memory_store: BaseMemoryStore) -> dict:
    """extract user profile and save to store"""
    # get user id
    user_id = state.get(StateFields.USER_ID.value)
    if not user_id:
        logger.warning("No user_id found in state, skipping profile extraction")
        return {StateFields.PROFILE_UPDATED.value: False}

    # no messages,return false
    messages = state.get(StateFields.MESSAGES.value, [])
    if not messages:
        logger.debug("No messages to extract profile from")
        return {StateFields.PROFILE_UPDATED.value: False}

    # cursor: prefer cross-session(reserved),otherwise state
    cursor = memory_store.get_extraction_cursor(user_id)
    if cursor is None:
        cursor = state.get(StateFields.LAST_EXTRACTED_MESSAGE_INDEX.value)

    # no new messages,return false
    new_user_messages = _get_new_user_messages(messages, cursor)
    if not new_user_messages:
        logger.info(f"extraction_skipped_no_new_user_messages user_id{user_id}")
        return {StateFields.PROFILE_UPDATED.value: False}

    # lightweight filtering
    if not _likely_contains_profile(new_user_messages):
        logger.info(f"extraction_skipped_filter user_id{user_id} msg_count{len(new_user_messages)}")
        last_msg = messages[-1]
        last_index = _get_message_index(last_msg)
        if last_index is None:
            logger.warning(f"cannot update cursor after filter skip: no message_index user_id{user_id}")
            return {StateFields.PROFILE_UPDATED.value: False}
        return {
            StateFields.PROFILE_UPDATED.value: True,
            StateFields.LAST_EXTRACTED_MESSAGE_INDEX.value: last_index
        }

    # build context window
    first_idx = messages.index(new_user_messages[0])
    last_idx = messages.index[new_user_messages[-1]]
    context_start = max(0, first_idx - 2)
    context_end = min(len(messages), last_idx + 3)
    context_messages = messages[context_start:context_end]
    conversations = "\n".join(
        format_message(m)
        for m in context_messages
    )

    # obtain a desensitized profile summary
    try:
        know_profile = memory_store.get_profile_summary(user_id)
    except Exception as e:
        logger.warning(f"Failed to get profile summary: {e}")
        known_profile = "暂无已知用户画像"

    # llm extract
    logger.info(f"extraction_llm_call user_id={user_id} msg_count={len(new_user_messages)}")
    try:
        chain = EXTRACT_PROMPT | llm | StrOutputParser()
        extract_str = chain.invoke({
            PromptKeys.CONVERSATION.value: conversations,
            PromptKeys.KNOWN_PROFILE.value: known_profile
        })
    except Exception as e:
        logger.error(f"extraction_llm_error user_id={user_id} error={e}")

    # parsing,verification
    items = safe_parse_extraction_output(extract_str)
    allowed_entity_keys = {e.value for e in ProfileEntityKey}
    updated = False

    for item in items:
        content = item.get(GeneralFieldNames.CONTENT)
        entity_key_raw = item.get(GeneralFieldNames.ENTITY_KEY)
        if not content or not entity_key_raw:
            continue
        if entity_key_raw not in allowed_entity_keys:
            logger.warning(f"Ignored invalid entity_key '{entity_key_raw}'")
            continue

        evidence_type = infer_evidence_type(content, [m.content for m in new_user_messages])

        metadata = {
            GeneralFieldNames.SOURCE: MemorySource.CHAT_EXTRACTION.value,
            GeneralFieldNames.CONFIDENCE: item.get(GeneralFieldNames.CONFIDENCE, 0.7),
            GeneralFieldNames.STATUS: MemoryStatus.ACTIVE.value,
            GeneralFieldNames.EVIDENCE_TYPE: evidence_type,
            GeneralFieldNames.EFFECTIVE_DATE: datetime.now().isoformat(),
            GeneralFieldNames.EXPIRES_AT: None,
        }

        # insert the extract profile
        try:
            memory_store.add_memory(
                user_id=user_id,
                content=content,
                memory_type=MemoryType.USER_PROFILE,
                entity_key=entity_key_raw,
                metadata=metadata
            )
            updated = True
        except MemoryWriteFailedError as e:
            logger.error(f"Memory write failed (DLQ): {e}")
        except Exception as e:
            logger.error(f"Unexpected error during profile extraction: {e}")

        if updated:
            logger.info(f"extraction_updated user_id={user_id} item_count={len(items)}")
        else:
            logger.info(f"extraction_no_new_info user_id={user_id}")

        # update cursor
        last_msg = messages[-1]
        last_index = _get_message_index(last_msg)
        if last_index is None:
            logger.warning(f"Cannot update cursor after extraction: no message_index user_id={user_id}")
            return {StateFields.PROFILE_UPDATED.value: updated}

        # update cross-session cursor
        try:
            memory_store.set_extraction_cursor(user_id, last_index)
        except Exception as e:
            pass

        return {
            StateFields.PROFILE_UPDATED.value: updated,
            StateFields.LAST_EXTRACTED_MESSAGE_INDEX.value: last_index
        }


def _get_message_index(msg: BaseMessage) -> Optional[int]:
    """安全获取消息的全局序号"""
    if hasattr(msg, 'additional_kwargs') and isinstance(msg.additional_kwargs, dict):
        return msg.additional_kwargs.get('message_index')
    return None
