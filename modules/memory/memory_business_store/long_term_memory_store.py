# author hgh
# version 1.0
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from chromadb.errors import ChromaError

from config.global_constant.constants import VectorQueryFields, MemoryType, ComplianceSeverity, CacheNamespace
from config.global_constant.fields import CommonFields
from config.models.memory_config import MemorySystemConfig
from exceptions.exception import MemoryWriteFailedError, MemoryRetrievalError, MemoryUpdateError
from modules.memory.memory_constant.constants import MemoryStatus, EvidenceType, InteractionEventType, \
    InteractionSentiment, ProfileEntityKey, MemorySource
from modules.memory.memory_constant.fields import MemoryFields
from modules.memory.memory_business_store.base_memory_store import BaseMemoryStore
from modules.memory.memory_vector_store.base_vector_store import BaseVectorStore
from modules.memory.models.memory_base import MemoryBase
from modules.memory.models.memory_schema import UserProfileMemory, InteractionLogMemory, \
    ComplianceRuleMemory
from utils.cache_utils.cache_decorator import custom_cached
from utils.model_mapper.model_to_storage import MemoryToStorageMapper
from utils.model_mapper.storage_to_model import StorageToMemoryMapper
from utils.query_utils.query_model import Query, Condition
from utils.retry import retry_on_failure

logger = logging.getLogger(__name__)


class LongTermMemoryStore(BaseMemoryStore):
    def __init__(self, vector_store: BaseVectorStore, config: MemorySystemConfig):
        """
        initial chroma client and collection 
        
        Args:
            vector_store: vector databases
            config: memory system config
        """
        self.vector_store = vector_store
        self.config = config

        # dead letter queue
        self.dlq_path = Path(self.config.memory_dlq_path)
        self.dlq_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("LongTermMemoryStore initialized with vector_store")

    def _write_to_dlq(
            self, user_id: str, content: str, model: MemoryBase, memory_type: MemoryType
    ):
        entry = {
            MemoryFields.USER_ID: user_id,
            MemoryFields.CONTENT: content,
            MemoryFields.MEMORY_TYPE: memory_type,
            MemoryFields.METADATA: model.model_dump(mode='json'),
            MemoryFields.TIMESTAMP: datetime.now().isoformat(),
        }
        with open(self.dlq_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info("DLQ entry written for user=%s, memory_type=%s", user_id, memory_type.value)

    def add_memory(
            self,
            user_id: str,
            content: str,
            memory_type: MemoryType,
            entity_key: Optional[ProfileEntityKey] = None,
            metadata: Optional[Dict[str, Any]] = None,
            permanent: bool = False
    ) -> str:
        logger.info("Adding memory for user=%s, type=%s, entity_key=%s", user_id, memory_type.value,
                    entity_key.value if entity_key else 'N/A')
        now = datetime.now()
        meta_input = metadata or {}
        try:
            if memory_type == MemoryType.USER_PROFILE:
                model = UserProfileMemory(
                    user_id=user_id,
                    confidence=meta_input.get(MemoryFields.CONFIDENCE, 0.8),
                    status=MemoryStatus.ACTIVE,
                    permanent=permanent,
                    created_at=now,
                    last_accessed_at=meta_input.get(MemoryFields.LAST_ACCESSED_AT, now),
                    source=meta_input.get(MemoryFields.SOURCE, MemorySource.CHAT_EXTRACTION),
                    entity_key=entity_key,
                    evidence_type=meta_input.get(MemoryFields.EVIDENCE_TYPE, EvidenceType.EXPLICIT_STATEMENT),
                    effective_date=meta_input.get(MemoryFields.EFFECTIVE_DATE, now),
                    expires_at=meta_input.get(MemoryFields.EXPIRES_AT),
                    superseded_by=meta_input.get(MemoryFields.SUPERSEDED_BY),
                    extra=meta_input.get(MemoryFields.EXTRA) or {}
                )
            elif memory_type == MemoryType.INTERACTION_LOG:
                model = InteractionLogMemory(
                    user_id=user_id,
                    confidence=meta_input.get(MemoryFields.CONFIDENCE, 1.0),
                    status=MemoryStatus.ACTIVE,
                    permanent=permanent,
                    created_at=now,
                    last_accessed_at=now,
                    source=meta_input.get(MemoryFields.SOURCE, MemorySource.AUTO_SUMMARY),
                    event_type=meta_input.get(MemoryFields.EVENT_TYPE, InteractionEventType.INQUIRY),
                    session_id=meta_input.get(MemoryFields.SESSION_ID, "unknown"),
                    sentiment=meta_input.get(MemoryFields.SENTIMENT, InteractionSentiment.NEUTRAL),
                    key_entities=meta_input.get(MemoryFields.KEY_ENTITIES),
                    timestamp=meta_input.get(MemoryFields.TIMESTAMP, now),
                    extra=meta_input.get(MemoryFields.EXTRA) or {}
                )
            elif memory_type == MemoryType.COMPLIANCE_RULE:
                model = ComplianceRuleMemory(
                    user_id=user_id,
                    confidence=meta_input.get(MemoryFields.CONFIDENCE, 1.0),
                    status=MemoryStatus.ACTIVE,
                    permanent=permanent,
                    created_at=now,
                    last_accessed_at=now,
                    source=meta_input.get(MemoryFields.SOURCE, MemorySource.ADMIN_IMPORT),
                    rule_id=meta_input[MemoryFields.RULE_ID],
                    rule_name=meta_input[MemoryFields.RULE_NAME],
                    rule_type=meta_input[MemoryFields.RULE_TYPE],
                    pattern=meta_input.get(MemoryFields.PATTERN, ""),
                    action=meta_input[MemoryFields.ACTION],
                    severity=meta_input.get(MemoryFields.SEVERITY, ComplianceSeverity.MEDIUM),
                    priority=meta_input.get(MemoryFields.PRIORITY, 100),
                    version=meta_input.get(MemoryFields.VERSION, now.strftime("%Y-%m-%d")),
                    effective_from=meta_input.get(MemoryFields.EFFECTIVE_FROM, now),
                    effective_to=meta_input.get(MemoryFields.EFFECTIVE_TO),
                    template=meta_input.get(MemoryFields.TEMPLATE),
                    superseded_by=meta_input.get(MemoryFields.SUPERSEDED_BY),
                    description=meta_input.get(MemoryFields.DESCRIPTION),
                    extra=meta_input.get(MemoryFields.EXTRA) or {}
                )
            else:
                raise ValueError(f"Unsupported memory type: {memory_type}")
        except Exception as e:
            logger.error("Failed to validate metadata with Pydantic model: %s", e, exc_info=True)
            fallback = MemoryBase(user_id=user_id, confidence=1.0)
            self._write_to_dlq(user_id, content, fallback, memory_type)
            raise MemoryWriteFailedError(f"Memory validation failed, queued: {e}") from e

        # ---------- Invoke conflict detection (for user profiles only) ----------
        superseded_ids = []
        if memory_type == MemoryType.USER_PROFILE and entity_key and not permanent:
            skip_result, superseded_from_conflict = self._resolve_profile_conflicts(
                user_id=user_id,
                entity_key=entity_key,
                content=content,
                new_model=model,
                evidence_weights=self.config.evidence_rules.evidence_weights,
            )
            superseded_ids.extend(superseded_from_conflict)

            if skip_result is not None:
                logger.info(
                    "Skipping memory insertion for user=%s, entity=%s, returning old_id=%s",
                    user_id, entity_key.value, skip_result
                )
                return skip_result

        # add memory
        memory_id = str(uuid.uuid4())
        try:
            self.vector_store.add(memory_type=memory_type, ids=[memory_id], texts=[content], models=[model])
            logger.debug("Vector store write successful for memory_id=%s", memory_id)
        except Exception as e:
            # write to dlq
            logger.error("Write failed for user %s: %s", user_id, e, exc_info=True)
            self._write_to_dlq(user_id, content, model, memory_type)
            raise MemoryWriteFailedError(f"Memory write failed, queued: {e}") from e

        # update the superseded_by of the memory to the new memory ID
        for old_id in superseded_ids:
            try:
                self.update_memory_status(old_id, memory_type, MemoryStatus.SUPERSEDED,
                                          {MemoryFields.SUPERSEDED_BY: memory_id})
            except Exception as e:
                logger.error("Failed to update superseded_by for %s: %s", old_id, e)

        logger.info("Memory added successfully, id=%s, type=%s, user=%s", memory_id, memory_type.value, user_id)
        return memory_id

    @retry_on_failure(max_retries=3, initial_delay=0.2, exceptions=(Exception, MemoryRetrievalError))
    def search_memory(
            self,
            user_id: str,
            query: str,
            memory_type: MemoryType,
            limit: int = 3,
            min_confidence: Optional[float] = None,
            apply_decay: bool = False
    ) -> List[Dict[str, Any]]:
        """search memory by query_utils"""
        logger.info("Searching memory for user=%s, type=%s, query='%s...', limit=%d", user_id, memory_type.value,
                    query[:50], limit)
        # build query_utils conditions
        conditions = [
            Condition(field=MemoryFields.USER_ID, op="==", value=user_id),
            Condition(field=MemoryFields.STATUS, op="==", value=MemoryStatus.ACTIVE.value)
        ]
        if min_confidence is not None:
            conditions.append(Condition(field=MemoryFields.CONFIDENCE, op=">=", value=min_confidence))
        where = Query(conditions=conditions, logic="AND")

        # the number of result
        fetch_limit = limit * 2 if apply_decay else limit

        # execute query_utils
        try:
            results = self.vector_store.search(
                memory_type=memory_type,
                query=query,
                where=where,
                limit=fetch_limit,
            )
        except Exception as e:
            logger.error("Search failed for user=%s, type=%s: %s", user_id, memory_type.value, e, exc_info=True)
            raise MemoryRetrievalError(f"Search failed: {e}") from e

        # organize output and apply time decay
        memories = []
        for hit in results:
            try:
                model = StorageToMemoryMapper.from_db_dict(hit, memory_type)
            except Exception as e:
                logger.warning("Failed to deserialize memory id=%s: %s", hit.get(CommonFields.ID), e)
                continue

            similarity = hit.get(VectorQueryFields.SCORE)
            mem = {
                MemoryFields.ID: hit.get(MemoryFields.ID),
                MemoryFields.TEXT: hit.get(MemoryFields.TEXT),
                MemoryFields.METADATA: {k: v for k, v in
                                        MemoryToStorageMapper.to_db_meta(model, target_db="milvus").items()},
                VectorQueryFields.DISTANCE: hit.get(VectorQueryFields.DISTANCE, 0.0),
                MemoryFields.SIMILARITY: similarity
            }

            if apply_decay and not model.permanent:
                decay_factor = self._calculate_decay_factor(model)
                mem[MemoryFields.DECAYED_SIMILARITY] = similarity * decay_factor
            else:
                mem[MemoryFields.DECAYED_SIMILARITY] = similarity
            memories.append(mem)

        if apply_decay:
            memories.sort(key=lambda x: x[MemoryFields.DECAYED_SIMILARITY], reverse=True)

        # update last accessed
        for m in memories[:limit]:
            self._update_last_accessed(memory_type, m[MemoryFields.ID])

        logger.info("Search memory returned %d results (limited to %d)", len(memories), limit)
        return memories[:limit]

    @retry_on_failure(max_retries=3, exceptions=(ChromaError,))
    def get_memory_by_entity(
            self,
            user_id: str,
            entity_key: ProfileEntityKey,
            status: MemoryStatus = MemoryStatus.ACTIVE
    ) -> List[Dict[str, Any]]:
        """get memory by entity key"""
        logger.info("Getting memory by entity for user=%s, entity=%s, status=%s", user_id, entity_key.value,
                    status.value)

        # build query_utils conditions
        where = Query(conditions=[
            Condition(field=MemoryFields.USER_ID, op="==", value=user_id),
            Condition(field=MemoryFields.ENTITY_KEY, op="==", value=entity_key.value),
            Condition(field=MemoryFields.STATUS, op="==", value=status.value)
        ])

        try:
            results = self.vector_store.get(
                memory_type=MemoryType.USER_PROFILE,
                where=where,
            )
        except Exception as e:
            logger.error("get by entity failed for user=%s, entity=%s: %s", user_id, entity_key.value, e, exc_info=True)
            return []

        memories = self._assemble_memories(results, MemoryType.USER_PROFILE)
        logger.info("Found %d memories for entity=%s", len(memories), entity_key.value)
        return memories

    @retry_on_failure(max_retries=3, exceptions=(ChromaError,))
    def update_memory_status(
            self,
            memory_id: str,
            memory_type: MemoryType,
            new_status: MemoryStatus,
            metadata_updates: Optional[Dict[str, Any]] = None) -> bool:
        """update memory status"""
        logger.info("Updating memory status for id=%s, type=%s, new_status=%s", memory_id, memory_type.value,
                    new_status.value)
        updates = {MemoryFields.STATUS: new_status.value}
        if metadata_updates:
            updates.update(metadata_updates)
        try:
            self.vector_store.update(
                memory_type=memory_type,
                ids=[memory_id],
                metadatas=[updates],
            )
            return True
        except Exception as e:
            logger.error("Update failed for id=%s: %s", memory_id, e, exc_info=True)
            raise MemoryUpdateError(f"Update failed: {e}") from e

    def apply_forgetting(
            self,
            memory_type: MemoryType,
            user_id: Optional[str] = None,
            threshold: Optional[float] = None
    ) -> int:
        """apply forgetting"""
        logger.info("Applying forgetting for type=%s, user=%s, threshold=%s", memory_type.value, user_id or 'ALL',
                    threshold or self.config.decay_threshold)
        # build conditions
        conditions = [
            Condition(field=MemoryFields.STATUS, op="==", value=MemoryStatus.ACTIVE.value),
            Condition(field=MemoryFields.PERMANENT, op="==", value=False)
        ]
        if user_id:
            conditions.append(Condition(field=MemoryFields.USER_ID, op="==", value=user_id))
        where = Query(conditions=conditions, logic="AND")

        # query_utils the memories that need to be forgotten
        try:
            res = self.vector_store.get(
                memory_type=MemoryType.USER_PROFILE,
                where=where
            )
        except Exception as e:
            logger.error("Forgetting scan failed: %s", e, exc_info=True)
            return 0

        count = 0
        for hit in res:
            try:
                model = StorageToMemoryMapper.from_db_dict(hit, memory_type)
            except Exception as e:
                continue

            threshold = threshold or self.config.decay_threshold
            decay_factor = self._calculate_decay_factor(model)
            if decay_factor < threshold:
                try:
                    self.update_memory_status(
                        hit[MemoryFields.ID],
                        memory_type,
                        MemoryStatus.FORGOTTEN
                    )
                    count += 1
                except Exception:
                    pass
        logger.info("Forgotten %d memories", count)
        return count

    def delete_user_memories(
            self,
            user_id: str,
            memory_type: Optional[MemoryType] = None
    ) -> bool:
        logger.info("Deleting memories for user=%s, type=%s", user_id, memory_type.value if memory_type else 'ALL')
        """delete user memory"""
        types = [memory_type] if memory_type else list(MemoryType)
        for mem_type in types:
            where = Query(conditions=[Condition(field=MemoryFields.USER_ID, op="==", value=user_id)])
            try:
                self.vector_store.delete(memory_type=mem_type, where=where)
            except Exception as e:
                logger.error("Delete failed for type=%s, user=%s: %s", mem_type.value, user_id, e, exc_info=True)
                return False
        return True

    def _calculate_decay_factor(self, model: MemoryBase) -> float:
        """apply decay(original similarity * e ** (-decay_factor * (now()-last_accessed)))"""
        last = model.last_accessed_at
        if not last:
            return 1.0
        try:
            days = (datetime.now() - last).days
        except:
            days = 0
        return float(np.exp(-self.config.decay_factor * days))

    @retry_on_failure(max_retries=3, exceptions=(ChromaError,))
    def _update_last_accessed(self, memory_type: MemoryType, memory_id: str):
        try:
            self.vector_store.update(
                memory_type=memory_type,
                ids=[memory_id],
                metadatas=[{MemoryFields.LAST_ACCESSED_AT: datetime.now().isoformat()}]
            )
        except Exception as e:
            logger.warning("Failed to update access time for %s: %s", memory_id, e)

    @custom_cached(
        namespace=CacheNamespace.RECENT_INTERACTION,
        ttl=120,
        null_ttl=30,
        empty_result_factory=list,
        ignore_args=[0, 2]
    )
    def get_recent_interactions(self, user_id: str, limit: int = 5) -> Optional[List[Dict[str, Any]]]:
        logger.info("Fetching recent interactions for user=%s, limit=%d", user_id, limit)
        # build query_utils condition
        where = Query(conditions=[
            Condition(field=MemoryFields.USER_ID, op="==", value=user_id),
            Condition(field=MemoryFields.STATUS, op="==", value=MemoryStatus.ACTIVE.value)
        ])

        # execute query_utils
        try:
            results = self.vector_store.get(
                memory_type=MemoryType.INTERACTION_LOG,
                where=where,
                limit=limit * 3
            )
            if not results:
                return []
        except Exception as e:
            logger.error("Failed to retrieve interactions for user=%s: %s", user_id, e, exc_info=True)
            return []

        # organize result data
        memories = self._assemble_memories(results, MemoryType.INTERACTION_LOG)

        # sort by time
        def get_timestamp(mem):
            ts = mem[MemoryFields.METADATA].get(MemoryFields.TIMESTAMP)
            if ts:
                try:
                    return datetime.fromisoformat(ts)
                except:
                    pass
            return datetime.min

        memories.sort(key=get_timestamp, reverse=True)
        return memories[:limit]

    @custom_cached(
        namespace=CacheNamespace.COMPLIANCE,
        ttl=600,
        null_ttl=60,
        empty_result_factory=list,
        ignore_args=[0, 1]
    )
    def get_active_compliance_rules(self, limit: int = 10) -> List[Dict[str, Any]]:
        logger.info("Fetching active compliance rules, limit=%d", limit)
        where = Query(conditions=[Condition(field=MemoryFields.STATUS, op="==", value=MemoryStatus.ACTIVE.value)])

        try:
            results = self.vector_store.get(
                memory_type=MemoryType.COMPLIANCE_RULE,
                where=where
            )
        except Exception as e:
            logger.error("Failed to retrieve compliance rules: %s", e, exc_info=True)
            return []

        severity_order = {
            ComplianceSeverity.CRITICAL: 0,
            ComplianceSeverity.HIGH: 1,
            ComplianceSeverity.MEDIUM: 2,
            ComplianceSeverity.LOW: 3,
            ComplianceSeverity.MANDATORY: 0
        }

        rules = self._assemble_memories(results, MemoryType.COMPLIANCE_RULE)

        rules.sort(key=lambda r: (
            severity_order.get(r[MemoryFields.METADATA].get(MemoryFields.SEVERITY), 4),
            r[MemoryFields.METADATA].get(MemoryFields.PRIORITY, 100)
        ))
        logger.info("Loaded %d active compliance rules", len(rules))
        return rules[:limit]

    def get_all_user_profile_memories(self, user_id: str, status: Optional[MemoryStatus] = None) -> List[
        Dict[str, Any]]:
        # build condition
        logger.info("Fetching all user profile memories for user=%s, status=%s", user_id,
                    status.value if status else 'ALL')
        conditions = [Condition(field=MemoryFields.USER_ID, op="==", value=user_id)]
        if status:
            conditions.append(Condition(field=MemoryFields.STATUS, op="==", value=status.value))
        where = Query(conditions=conditions, logic="AND")

        # execute query_utils
        try:
            results = self.vector_store.get(
                memory_type=MemoryType.USER_PROFILE,
                where=where
            )
        except Exception as e:
            logger.error("Failed to get user profile memories for user=%s: %s", user_id, e, exc_info=True)
            return []

        memories = self._assemble_memories(results, MemoryType.USER_PROFILE)
        logger.info("Retrieved %d user profile memories for user=%s", len(memories), user_id)
        return memories

    def _assemble_memories(self, results: List[Dict[str, Any]], memory_type: MemoryType) -> List[Dict[str, Any]]:
        memories = []
        for i, hit in enumerate(results):
            try:
                model = StorageToMemoryMapper.from_db_dict(hit, memory_type)
            except Exception as e:
                logger.warning("Deserialization failed for hit %d: %s", i, e)
                continue
            memories.append({
                MemoryFields.ID: hit.get(MemoryFields.ID),
                MemoryFields.TEXT: hit.get(MemoryFields.TEXT),
                MemoryFields.METADATA: {k: v for k, v in
                                        MemoryToStorageMapper.to_db_meta(model, target_db="milvus").items()},
            })
        return memories

    @custom_cached(
        namespace=CacheNamespace.PROFILE_SUMMARY,
        ttl=60,
        null_ttl=60,
        ignore_args=[0, 2]
    )
    def get_profile_summary(self, user_id: str, max_chars: int = 500) -> Optional[str]:
        """
        Get a brief summary of the user's current active profile for extracting node injection prompts

        Args:
            user_id: unique user id
            max_chars: maximum number of characters to display

        Returns:
            each line "- entity_key: content (confidence X.XX)"
            If there is no profile, return "No known profile."
        """
        logger.info("Generating profile summary for user=%s, max_chars=%d", user_id, max_chars)
        try:
            memories = self.get_all_user_profile_memories(user_id, MemoryStatus.ACTIVE)
            if not memories:
                return None

            # messages are sorted by last accessed time or creation time
            def get_ts(mem):
                meta = mem.get(MemoryFields.METADATA, {})
                ts_str = meta.get(MemoryFields.LAST_ACCESSED_AT) or meta.get(MemoryFields.CREATED_AT)
                if ts_str:
                    try:
                        return datetime.fromisoformat(ts_str)
                    except:
                        pass
                    return datetime.min

            memories.sort(key=get_ts, reverse=True)

            # sensitive field desensitization regex(example: phone/email)
            SENSITIVE_KEYS = {'contact', 'phone', 'email'}

            def mask_content(key, content):
                if not isinstance(content, str):
                    return content
                if key in SENSITIVE_KEYS:
                    if len(content) > 4:
                        return content[:2] + "*" * (len(content) - 4) + content[-2:]
                    else:
                        return "***"

            lines = []
            total_chars = 0
            for mem in memories:
                metadata = mem.get(MemoryFields.METADATA, {})
                entity_key = metadata.get(MemoryFields.ENTITY_KEY, 'unknown')
                content = mem.get(MemoryFields.TEXT, '')
                confidence = metadata.get(MemoryFields.CONFIDENCE, 0.0)
                # mask sensitive content
                content_masked = mask_content(entity_key, content)
                line = f"- {entity_key}: {content_masked} (置信度 {confidence:.2f})"

                if total_chars + len(line) > max_chars:
                    break
                lines.append(line)
                total_chars += len(line) + 1

            if not lines:
                return None
            result = "\n".join(lines)
            logger.info("Profile summary generated for user=%s, length=%d chars", user_id, len(result))
            return result
        except Exception as e:
            logger.error("Failed to get user profile summary for user=%s: %s", user_id, e, exc_info=True)
            return None

    def _resolve_profile_conflicts(
            self,
            user_id: str,
            entity_key: ProfileEntityKey,
            content: str,
            new_model: UserProfileMemory,
            evidence_weights: Dict[str, int],
    ) -> tuple[Optional[str], List[str]]:
        """
       Handle conflicts between new and old memories under the same entity.

        Returns:
        (skip_old_id, superseded_ids)
            - skip_old_id: The old record ID to return if the insertion should be skipped, otherwise None
            - superseded_ids: List of old record IDs that need to have superseded_by set after the new record is inserted
        """
        superseded = []
        try:
            existing = self.get_memory_by_entity(user_id, entity_key, MemoryStatus.ACTIVE)
        except Exception as e:
            logger.warning(
                "Conflict check failed for user=%s, entity=%s, error: %s. Proceeding without conflict resolution.",
                user_id, entity_key.value, e
            )
            return None, superseded

        for old in existing:
            old_id = old[MemoryFields.ID]
            old_content = old.get(MemoryFields.TEXT, "").strip()
            old_conf = float(old.get(MemoryFields.CONFIDENCE, 0.0))
            old_evidence = old.get(MemoryFields.EVIDENCE_TYPE, EvidenceType.EXPLICIT_STATEMENT)
            old_weight = evidence_weights.get(old_evidence, 50)

            new_conf = new_model.confidence
            new_evidence = new_model.evidence_type if hasattr(new_model,
                                                              MemoryFields.EVIDENCE_TYPE) else EvidenceType.EXPLICIT_STATEMENT
            new_weight = evidence_weights.get(new_evidence, 50)

            if content.strip() == old_content:
                if new_weight > old_weight:
                    logger.info(
                        "IDENTICAL_CONTENT_EVIDENCE_UPGRADE | user=%s | entity=%s | "
                        "old_id=%s (content='%s', conf=%.2f, evidence=%s, weight=%d) | "
                        "new_id=<will be created> (content='%s', conf=%.2f, evidence=%s, weight=%d). "
                        "Action: supersede old, insert new.",
                        user_id, entity_key.value,
                        old_id, old_content, old_conf, old_evidence, old_weight,
                        content, new_conf, new_evidence, new_weight
                    )
                    try:
                        self.update_memory_status(old_id, MemoryType.USER_PROFILE,
                                                  MemoryStatus.SUPERSEDED,
                                                  {MemoryFields.SUPERSEDED_BY: None})
                        superseded.append(old_id)
                    except Exception as e:
                        logger.warning(
                            "Failed to supersede old_id=%s during evidence upgrade: %s",
                            old_id, e
                        )
                    continue
                else:
                    logger.info(
                        "IDENTICAL_CONTENT_NO_UPGRADE | user=%s | entity=%s | "
                        "old_id=%s (content='%s', conf=%.2f, evidence=%s, weight=%d) | "
                        "new content identical, evidence=%s (weight=%d) not stronger. "
                        "Action: skip insertion, keep old record.",
                        user_id, entity_key.value,
                        old_id, old_content, old_conf, old_evidence, old_weight,
                        new_evidence, new_weight
                    )
                    self._update_last_accessed(MemoryType.USER_PROFILE, old_id)
                    return old_id, superseded

            logger.info(
                "CONFLICT_DETECTED | user=%s | entity=%s | "
                "old_id=%s (content='%s', conf=%.2f, evidence=%s, weight=%d) | "
                "new_id=<will be created> (content='%s', conf=%.2f, evidence=%s, weight=%d).",
                user_id, entity_key.value,
                old_id, old_content, old_conf, old_evidence, old_weight,
                content, new_conf, new_evidence, new_weight
            )

            if new_conf > old_conf + 0.1 or (abs(new_conf - old_conf) <= 0.1 and new_weight > old_weight):
                logger.info(
                    "CONFLICT_RESOLVED_SUPERSEDE | user=%s | entity=%s | "
                    "old_id=%s will be superseded by new record. "
                    "Reason: %s",
                    user_id, entity_key.value,
                    old_id,
                    "higher confidence" if new_conf > old_conf + 0.1 else "stronger evidence"
                )
                try:
                    self.update_memory_status(old_id, MemoryType.USER_PROFILE,
                                              MemoryStatus.SUPERSEDED,
                                              {MemoryFields.SUPERSEDED_BY: None})
                    superseded.append(old_id)
                except Exception as e:
                    logger.warning(
                        "Failed to supersede old_id=%s during standard conflict resolution: %s",
                        old_id, e
                    )
            else:
                logger.info(
                    "CONFLICT_RESOLVED_KEEP_OLD | user=%s | entity=%s | "
                    "old_id=%s remains active. New record discarded. "
                    "Reason: confidence and evidence not sufficient to replace.",
                    user_id, entity_key.value, old_id
                )
                pass

        return None, superseded

    def get_extraction_cursor(self, user_id: str) -> Optional[int]:
        logger.info("get_extraction_cursor called for user=%s (not implemented)", user_id)
        return None

    def set_extraction_cursor(self, user_id: str, message_index: int) -> None:
        logger.info("set_extraction_cursor called for user=%s to %d (not implemented)", user_id, message_index)
