# author hgh
# version 1.0
import logging
from datetime import datetime

from config.global_constant.constants import MemoryType
from config.global_constant.fields import CommonFields
from infra.database.redis_manager import RedisManager
from infra.message_queue import Message
from modules.consumer.base_message_consumer import BaseMessageConsumer
from modules.memory.memory_business_store.base_memory_store import BaseMemoryStore
from modules.memory.memory_constant.constants import ProfileEntityKey, MemorySource, MemoryStatus, EvidenceType
from modules.memory.memory_utils.base_memory_utils import safe_parse_extraction_output
from modules.module_services.evidence_infer import EvidenceTypeInfer
from modules.module_services.profile_extractor import ProfileExtractor

logger = logging.getLogger(__name__)


class UserProfileConsumer(BaseMessageConsumer):
    def __init__(self, stream_name: str, group_name: str, redis_manager: RedisManager, memory_store: BaseMemoryStore,
                 evidence_infer: EvidenceTypeInfer, profile_extractor: ProfileExtractor):
        super().__init__(redis_manager, stream_name, group_name, "worker-1")
        self.memory_store = memory_store
        self.evidence_infer = evidence_infer
        self.profile_extractor = profile_extractor

    def handle(self, data: Message):
        trace_id = data.trace_id
        user_id = data.payload.get(CommonFields.USER_ID)
        conversations = data.payload.get(CommonFields.CONTENT)
        user_text = data.payload.get(CommonFields.TEXT)

        known_profile = "暂无已知用户画像"
        try:
            summary = self.memory_store.get_profile_summary(user_id)
            if summary:
                known_profile = summary
                logger.debug("TraceId--%s [ExtractProfileConsumer] using existing profile summary %s", trace_id,
                             summary)
        except Exception as e:
            logger.warning(
                "TraceId--%s [ExtractProfileConsumer] failed to get profile summary for user_id=%s, exception is: %s",
                trace_id, user_id, e)
            raise

            # 7. llm extract
        logger.info("TraceId--%s [ExtractProfileConsumer] calling LLM for profile extraction")
        extract_str = self.profile_extractor.extract(conversations, known_profile)
        logger.info("TraceId--%s [ExtractProfileConsumer] LLM extraction response (first 200 chars): %.200s", trace_id,
                    extract_str)

        # 8. parsing,verification
        items = safe_parse_extraction_output(extract_str)
        allowed_entity_keys = {e.value for e in ProfileEntityKey}
        updated = False

        # 9. insert to store
        logger.info("TraceId--%s [ExtractProfileConsumer] extracted %d potential profile items", trace_id, len(items))
        for item in items:
            confidence = item.get(CommonFields.CONFIDENCE, 0.0)
            content = item.get(CommonFields.CONTENT)
            entity_key_raw = item.get(CommonFields.ENTITY_KEY)
            if not content or not entity_key_raw:
                continue
            if confidence < 0.5:
                logger.warning(
                    "TraceId--%s [ExtractProfileConsumer] ignored low confidence entity_key '%s' for user_id=%s",
                    trace_id, entity_key_raw, user_id)
                continue
            if entity_key_raw not in allowed_entity_keys:
                logger.warning("TraceId--%s [ExtractProfileConsumer] ignored invalid entity_key '%s' for user_id=%s",
                               trace_id, entity_key_raw,
                               user_id)
                continue

            # infer evidence type
            evidence_type = self.evidence_infer.infer(content, user_text)
            logger.debug("TraceId--%s [ExtractProfileConsumer] inferred evidence type '%s' for entity '%s'", trace_id,
                         evidence_type,
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
                logger.info("TraceId--%s [ExtractProfileConsumer] added profile memory for user_id=%s, entity=%s",
                            trace_id, user_id,
                            entity_key_raw)
            except Exception as e:
                logger.error("TraceId--%s [ExtractProfileConsumer] unexpected error during profile extraction: %s",
                             trace_id, e, exc_info=True)
                raise
