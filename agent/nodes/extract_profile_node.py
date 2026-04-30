# author hgh
# version 1.0
import logging
import re
from datetime import datetime
from typing import List, Optional, Dict

import yaml
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

from agent.state import AgentState
from config.constants import StateFields, PromptKeys, ProfileEntityKey, GeneralFieldNames, MemorySource, MemoryStatus, \
    MemoryType, MessageCommonFields, ProfileGateConfigFields
from config.settings import agentConfig
from exceptions.exception import MemoryWriteFailedError
from llm.chat_models import get_llm
from memory.base_memory_store import BaseMemoryStore
from memory.classifiers import infer_evidence_type
from prompt.extract_prompt import EXTRACT_PROMPT
from utils.memory_utils.memory_common_utils import format_message, get_message_index
from utils.parser import safe_parse_extraction_output

logger = logging.getLogger(__name__)
llm = get_llm()


def _load_gate_config() -> Dict:
    """load the gate configuration file,and use hard_coded defaults value if it fails"""
    path = agentConfig.profile_gate_config_path
    default_config = {
        "strong_patterns": [
            r'\d+\s*(?:万|元|岁|年|月|天|k|w|块|毛)',
            r'1[3-9]\d{9}',
            r'不对|错了|其实是|应该是|实际上是|更正一下|更新一下|确切说|准确来说|我记错了'
        ],
        "explicit_triggers": [
            r'(?:我(?:想|要|需要|要求|准备)|请(?:帮我|给我))(?:更改|修改|更新|变更|改成|改为)',
            r'现在(?:是|在)|已经(?:是|在)|我刚换|我最近换|我(?:刚|才)换'
        ],
        "weak_signals": [
            {"words": ["公司", "工作", "收入", "工资", "贷款", "资产", "负债"], "score": 2},
            {"words": ["结婚", "孩子", "父母", "学历"], "score": 1}
        ],
        "threshold": 4
    }
    try:
        with open(path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
            if not config_data:
                raise ValueError("Empty config file")
            for key in [ProfileGateConfigFields.STRONG_PATTERNS, ProfileGateConfigFields.EXPLICIT_TRIGGERS,
                        ProfileGateConfigFields.WEAK_SIGNALS, ProfileGateConfigFields.MATCH_THRESHOLD]:
                if key not in config_data:
                    raise ValueError(f"Missing key '{key}' in gate config")
            logger.info(f"Profile gate config loaded from {path}")
            return config_data
    except Exception as e:
        logger.warning(f"Failed to load gate config from {path}: {e}. Using built-in defaults.")
        return default_config


_gate_config = _load_gate_config()

# strong signal regular merging
_STRONG_SIGNALS_RULES = _gate_config[ProfileGateConfigFields.STRONG_PATTERNS]
_STRONG_SIGNALS = re.compile('|'.join(f'(?:{p})' for p in _STRONG_SIGNALS_RULES), re.IGNORECASE)

# explicit instruction regular merging
_EXPLICIT_TRIGGERS_RULES = _gate_config[ProfileGateConfigFields.EXPLICIT_TRIGGERS]
_EXPLICIT_TRIGGERS = re.compile('|'.join(f'(?:{p})' for p in _EXPLICIT_TRIGGERS_RULES), re.IGNORECASE)

# weak signal lexicon: {words: score}
_WEAK_SIGNALS_DICT: Dict[str, int] = {}
for group in _gate_config[ProfileGateConfigFields.WEAK_SIGNALS]:
    score = int(group[ProfileGateConfigFields.SCORE])
    for word in group[ProfileGateConfigFields.WORDS]:
        _WEAK_SIGNALS_DICT[word.lower()] = score

_WEAK_SIGNAL_THRESHOLD = int(_gate_config[ProfileGateConfigFields.MATCH_THRESHOLD])


def _likely_contains_profile(user_messages: List[BaseMessage]) -> bool:
    """
    Production-level gating (profile-based):
        - Explicit update command → trigger directly
        - Strong signal regular match → trigger directly
        - Weak signal accumulation ≥ threshold → trigger
        - Otherwise return False
    """
    for msg in user_messages:
        content = msg.content if hasattr(msg, 'content') else ""
        if not content:
            continue

        # 显式更新指令
        if _EXPLICIT_TRIGGERS.search(content):
            return True

        # 强信号快速通道
        if _STRONG_SIGNALS.search(content):
            return True

        # 弱信号累积
        score = 0
        for word, point in _WEAK_SIGNALS_DICT.items():
            if re.search(r'\b' + re.escape(word) + r'\b', content, re.IGNORECASE):
                score += point
                if score >= _WEAK_SIGNAL_THRESHOLD:
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
        last_index = get_message_index(last_msg)
        if last_index is None:
            logger.warning(f"cannot update cursor after filter skip: no message_index user_id{user_id}")
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

    # obtain a desensitized profile summary
    known_profile = "暂无已知用户画像"
    try:
        summary = memory_store.get_profile_summary(user_id)
        if summary:
            known_profile = summary
    except Exception as e:
        logger.warning(f"Failed to get profile summary, using default: {e}")

    # llm extract
    logger.info(f"extraction_llm_call user_id={user_id} msg_count={len(new_user_messages)}")
    try:
        chain = EXTRACT_PROMPT | llm.llm | StrOutputParser()
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
        last_index = get_message_index(last_msg)
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
