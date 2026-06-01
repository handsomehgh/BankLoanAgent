# author hgh
# version 1.0
import logging
import time
from datetime import datetime
from typing import List, Optional, Dict, Any

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from config.global_constant.fields import CommonFields
from config.global_constant.constants import MemoryType
from config.models.memory_config import MemorySystemConfig
from infra.message_queue import MessageProducer
from modules.agent.constants import MessageCommonFields, StateFields
from modules.agent.multi_agent_state import SupervisorState
from exceptions.exception import MemoryWriteFailedError
from modules.memory.memory_business_store.base_memory_store import BaseMemoryStore
from modules.memory.memory_constant.constants import ProfileEntityKey, MemorySource, MemoryStatus, EvidenceType
from modules.memory.memory_utils.base_memory_utils import get_message_index, \
    safe_parse_extraction_output, format_messages
from modules.memory.memory_utils.profile_gate_util import ProfileGate
from modules.module_services.evidence_infer import EvidenceTypeInfer
from modules.module_services.profile_extractor import ProfileExtractor

logger = logging.getLogger(__name__)


class ExtractProfileNode:
    """extract user profile and save to store"""

    def __init__(
            self,
            memory_store: BaseMemoryStore,
            profile_gate: ProfileGate,
            memory_config: MemorySystemConfig,
            evidence_infer: EvidenceTypeInfer,
            profile_extractor: ProfileExtractor,
            message_producer: MessageProducer
    ):
        self.memory_store = memory_store
        self.profile_gate = profile_gate
        self.memory_config = memory_config
        self.evidence_infer = evidence_infer
        self.profile_extractor = profile_extractor
        self.message_producer = message_producer

    def __call__(self, state: SupervisorState, config: RunnableConfig) -> Dict[str, Any]:
        logger.info("[ExtractProfileNode] entering extract_profile_node")

        # 1. no messages,return false
        user_id = state.get(StateFields.USER_ID.value)
        messages = state.get(StateFields.MESSAGES.value, [])
        if not messages:
            logger.debug("[ExtractProfileNode] no messages to extract profile")
            return {StateFields.PROFILE_UPDATED.value: False}
        logger.info("[ExtractProfileNode] starting profile extraction for user_id=%s, messages=%s", user_id, messages)

        # 2. obtain extract profile cursor
        cursor = state.get(StateFields.LAST_EXTRACTED_MESSAGE_INDEX.value)
        logger.debug("[ExtractProfileNode] extraction profile cursor: %s", cursor)

        # 3. no new messages,return false
        new_user_messages = self._get_new_user_messages(messages, cursor)
        if not new_user_messages:
            logger.info("[ExtractProfileNode] no new user messages, skipping profile extraction")
            return {StateFields.PROFILE_UPDATED.value: False}
        logger.info("[ExtractProfileNode] found %d new user messages", len(new_user_messages))

        # 4. lightweight filtering
        if not self.profile_gate.should_extract(new_user_messages):
            logger.info("[ExtractProfileNode] profile gate filtered out all messages for user_id=%s (msg_count=%d)",
                        user_id, len(new_user_messages))
            last_msg = messages[-1]
            last_index = get_message_index(last_msg)
            if last_index is None:
                logger.warning(
                    "[ExtractProfileNode] cannot update cursor after filter skip: no message_index for user_id=%s",
                    user_id)
                return {StateFields.PROFILE_UPDATED.value: False}
            return {
                StateFields.PROFILE_UPDATED.value: True,
                StateFields.LAST_EXTRACTED_MESSAGE_INDEX.value: last_index
            }

        # 5. format human message
        formatted = state.get(StateFields.FORMATTED_CONTEXT.value, {})
        summary = formatted.get(MemoryType.INTERACTION_LOG.value, "")
        conversations = summary + "\n" + format_messages(new_user_messages)

        # 6. obtain a desensitized profile summary
        known_profile = "暂无已知用户画像"
        try:
            summary = self.memory_store.get_profile_summary(user_id)
            if summary:
                known_profile = summary
                logger.debug("[ExtractProfileNode] using existing profile summary %s", summary)
        except Exception as e:
            logger.warning("[ExtractProfileNode] failed to get profile summary for user_id=%s, exception is: %s",
                           user_id, e)

        # 7. llm extract
        logger.info("[ExtractProfileNode] calling LLM for profile extraction")
        extract_str = self.profile_extractor.extract(conversations, known_profile)
        logger.info("[ExtractProfileNode] LLM extraction response (first 200 chars): %.200s", extract_str)

        # 8. parsing,verification
        items = safe_parse_extraction_output(extract_str)
        allowed_entity_keys = {e.value for e in ProfileEntityKey}
        updated = False

        # 9. insert to store
        logger.info("[ExtractProfileNode] extracted %d potential profile items", len(items))
        for item in items:
            confidence = item.get(CommonFields.CONFIDENCE, 0.0)
            content = item.get(CommonFields.CONTENT)
            entity_key_raw = item.get(CommonFields.ENTITY_KEY)
            if not content or not entity_key_raw:
                continue
            if confidence < 0.5:
                logger.warning("[ExtractProfileNode] ignored low confidence entity_key '%s' for user_id=%s", entity_key_raw,user_id)
                continue
            if entity_key_raw not in allowed_entity_keys:
                logger.warning("[ExtractProfileNode] ignored invalid entity_key '%s' for user_id=%s", entity_key_raw,user_id)
                continue

            # infer evidence type
            evidence_type = self.evidence_infer.infer(content, [m.content for m in new_user_messages])
            logger.debug("[ExtractProfileNode] inferred evidence type '%s' for entity '%s'", evidence_type,
                         entity_key_raw)

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
                self.memory_store.add_memory(
                    user_id=user_id,
                    content=content,
                    memory_type=MemoryType.USER_PROFILE,
                    entity_key=ProfileEntityKey(entity_key_raw),
                    metadata=metadata
                )
                updated = True
                logger.info("[ExtractProfileNode] added profile memory for user_id=%s, entity=%s", user_id,
                            entity_key_raw)
            except MemoryWriteFailedError as e:
                logger.error("[ExtractProfileNode] memory write failed (DLQ): %s", e, exc_info=True)
            except Exception as e:
                logger.error("[ExtractProfileNode] unexpected error during profile extraction: %s", e, exc_info=True)

            if updated:
                logger.info("[ExtractProfileNode] profile updated for user_id=%s: %d new items", user_id, len(items))
            else:
                logger.info("[ExtractProfileNode] no new profile information for user_id=%s", user_id)

        # 10. update cursor
        last_msg = messages[-1]
        last_index = get_message_index(last_msg)
        if last_index is None:
            logger.warning(
                "[ExtractProfileNode] cannot update cursor after extraction: no message_index for user_id=%s", user_id)
            return {StateFields.PROFILE_UPDATED.value: updated}

        return {
            StateFields.PROFILE_UPDATED.value: updated,
            StateFields.LAST_EXTRACTED_MESSAGE_INDEX.value: last_index
        }

    def _get_new_user_messages(
            self,
            messages: List[BaseMessage],
            last_extracted_index: Optional[int]
    ) -> List[BaseMessage]:
        """
        retrieve unprocessed messages after last_extracted_index,prefer using global message_index,
        if unavailable,fall back to the message ID
        """
        fallback_window = self.memory_config.profile_extraction_fallback_window
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
                "[ExtractProfileNode] some messages lack message_index despite cursor being set."
                f"Falling back to recent{fallback_window} messages"
            )
            recent = messages[-fallback_window:] if len(messages) > fallback_window else messages
            return [m for m in recent if isinstance(m, HumanMessage)]