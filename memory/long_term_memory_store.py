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

from config.settings import config
from exceptions.exception import MemoryWriteFailedError, MemoryRetrievalError, MemoryUpdateError
from memory.classifiers.rules.rules_loader import get_evidence_loader
from query.query_model import Condition, Query
from memory.models.memory_mappers.mappers import StorageToMemoryMapper
from memory.base_memory_store import BaseMemoryStore
from config.constants import MemoryType, MemoryStatus, ComplianceSeverity, EvidenceType, InteractionSentiment, \
    GeneralFieldNames, InteractionEventType
from memory.memory_vector_store.base_vector_store import BaseVectorStore
from memory.models.memory_data.memory_base import MemoryBase
from memory.models.memory_data.memory_schema import UserProfileMemory, InteractionLogMemory, ComplianceRuleMemory
from llm.retry import retry_on_failure

logger = logging.Logger(__name__)


class LongTermMemoryStore(BaseMemoryStore):
    def __init__(self, vector_store: BaseVectorStore):
        """
        initial chroma client and collection 
        
        Args:
            vector_store: vector databases
        """
        self.vector_store = vector_store

        # dead letter queue
        self.dlq_path = Path(config.memory_dlq_path)
        self.dlq_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("ChromaMemoryStore initialized with vector_store")

    def _write_to_dlq(
            self, user_id: str, content: str, model: MemoryBase, memory_type: MemoryType
    ):
        entry = {
            GeneralFieldNames.USER_ID: user_id,
            GeneralFieldNames.CONTENT: content,
            GeneralFieldNames.MEMORY_TYPE: memory_type.value,
            GeneralFieldNames.METADATA: model.model_dump(mode='json'),
            GeneralFieldNames.TIMESTAMP: datetime.now().isoformat(),
        }
        with open(self.dlq_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def add_memory(
            self,
            user_id: str,
            content: str,
            memory_type: MemoryType,
            entity_key: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            permanent: bool = False
    ) -> str:
        now = datetime.now()
        meta_input = metadata or {}
        model = None
        try:
            if memory_type == MemoryType.USER_PROFILE:
                model = UserProfileMemory(
                    user_id=user_id,
                    confidence=meta_input.get(GeneralFieldNames.CONFIDENCE, 0.8),
                    status=MemoryStatus.ACTIVE,
                    permanent=permanent,
                    created_at=now,
                    last_accessed_at=meta_input.get(GeneralFieldNames.LAST_ACCESSED_AT, now),
                    source=meta_input.get(GeneralFieldNames.SOURCE, "chat_extraction"),
                    entity_key=entity_key,
                    evidence_type=meta_input.get(GeneralFieldNames.EVIDENCE_TYPE, EvidenceType.EXPLICIT_STATEMENT),
                    effective_date=meta_input.get(GeneralFieldNames.EFFECTIVE_DATE, now),
                    expires_at=meta_input.get(GeneralFieldNames.EXPIRES_AT),
                    superseded_by=meta_input.get(GeneralFieldNames.SUPERSEDED_BY),
                    extra=meta_input.get(GeneralFieldNames.EXTRA) or {}
                )
            elif memory_type == MemoryType.INTERACTION_LOG:
                model = InteractionLogMemory(
                    user_id=user_id,
                    confidence=meta_input.get(GeneralFieldNames.CONFIDENCE, 1.0),
                    status=MemoryStatus.ACTIVE,
                    permanent=permanent,
                    created_at=now,
                    last_accessed_at=now,
                    source=meta_input.get(GeneralFieldNames.SOURCE, "auto_summary"),
                    event_type=meta_input.get(GeneralFieldNames.EVENT_TYPE, InteractionEventType.INQUIRY),
                    session_id=meta_input.get(GeneralFieldNames.SESSION_ID, "unknown"),
                    sentiment=meta_input.get(GeneralFieldNames.SENTIMENT, InteractionSentiment.NEUTRAL),
                    key_entities=meta_input.get(GeneralFieldNames.KEY_ENTITIES),
                    timestamp=meta_input.get(GeneralFieldNames.TIMESTAMP, now),
                    extra=meta_input.get(GeneralFieldNames.EXTRA) or {}
                )
            elif memory_type == MemoryType.COMPLIANCE_RULE:
                model = ComplianceRuleMemory(
                    user_id=user_id,
                    confidence=meta_input.get(GeneralFieldNames.CONFIDENCE, 1.0),
                    status=MemoryStatus.ACTIVE,
                    permanent=permanent,
                    created_at=now,
                    last_accessed_at=now,
                    source=meta_input.get(GeneralFieldNames.SOURCE, "admin_import"),
                    rule_id=meta_input[GeneralFieldNames.RULE_ID],
                    rule_name=meta_input[GeneralFieldNames.RULE_NAME],
                    rule_type=meta_input[GeneralFieldNames.RULE_TYPE],
                    pattern=meta_input.get(GeneralFieldNames.PATTERN, ""),
                    action=meta_input[GeneralFieldNames.ACTION],
                    severity=meta_input.get(GeneralFieldNames.SEVERITY, ComplianceSeverity.MEDIUM),
                    priority=meta_input.get(GeneralFieldNames.PRIORITY, 100),
                    version=meta_input.get(GeneralFieldNames.VERSION, now.strftime("%Y-%m-%d")),
                    effective_from=meta_input.get(GeneralFieldNames.EFFECTIVE_FROM, now),
                    effective_to=meta_input.get(GeneralFieldNames.EFFECTIVE_TO),
                    template=meta_input.get(GeneralFieldNames.TEMPLATE),
                    superseded_by=meta_input.get(GeneralFieldNames.SUPERSEDED_BY),
                    description=meta_input.get(GeneralFieldNames.DESCRIPTION),
                    extra=meta_input.get(GeneralFieldNames.EXTRA) or {}
                )
            else:
                raise ValueError(f"Unsupported memory type: {memory_type}")
        except Exception as e:
            logger.error(f"Failed to validate metadata with Pydantic model: {e}")
            fallback = MemoryBase(user_id=user_id, confidence=1.0)
            self._write_to_dlq(user_id, content, fallback, memory_type)
            raise MemoryWriteFailedError(f"Memory validation failed, queued: {e}") from e

        # conflict detection(only required for user profile)
        superseded_ids = []
        if memory_type == MemoryType.USER_PROFILE and entity_key and not permanent:
            try:
                existing = self.get_memory_by_entity(user_id, entity_key, MemoryStatus.ACTIVE.value)
            except Exception as e:
                logger.warning(f"Conflict check failed,proceeding: {e}")
                existing = []

            try:
                loader = get_evidence_loader()
                evidence_weights = loader.get_evidence_weights()
            except Exception as e:
                logger.warning(f"Failed to load evidence weights,user default: {e}")
                evidence_weights = {}

            # update the memory status to superseded
            for old in existing:
                old_conf = float(old.get(GeneralFieldNames.CONFIDENCE, 0.0))
                old_evidence = old.get(GeneralFieldNames.EVIDENCE_TYPE, EvidenceType.EXPLICIT_STATEMENT.value)
                old_weight = evidence_weights.get(old_evidence, 50)

                new_conf = model.confidence
                new_evidence = model.evidence_type if hasattr(model,
                                                              GeneralFieldNames.EVIDENCE_TYPE) else EvidenceType.EXPLICIT_STATEMENT.value
                new_weight = evidence_weights.get(new_evidence, 50)

                # Overriding Conditions:
                # 1. The new confidence is significantly higher (> old_conf + 0.1)
                # 2. Confidence difference is not large (≤ 0.1) but the new evidence is more authoritative
                if new_conf > old_conf + 0.1 or ((abs(new_conf - old_conf) <= 0.1) and new_weight > old_weight):
                    try:
                        self.update_memory_status(old[GeneralFieldNames.ID], memory_type, MemoryStatus.SUPERSEDED.value,
                                                  {GeneralFieldNames.SUPERSEDED_BY: None})
                        superseded_ids.append(old[GeneralFieldNames.ID])
                        logger.info(f"Superseded old memory {old[GeneralFieldNames.ID]} (entity: {entity_key})")
                    except Exception as e:
                        logger.warning(f"Failed to superseded {old[GeneralFieldNames.ID]: {e}}")

        # add memory
        memory_id = str(uuid.uuid4())
        try:
            self.vector_store.add(memory_type=memory_type, ids=[memory_id], texts=[content], models=[model])
        except Exception as e:
            # write to dlq
            logger.error(f"Write failed for user {user_id}: {e}")
            self._write_to_dlq(user_id, content, model, memory_type)
            raise MemoryWriteFailedError(f"Memory write failed, queued: {e}") from e

        # update the superseded_by of the memory to the new memory ID
        for old_id in superseded_ids:
            try:
                self.update_memory_status(old_id, memory_type, MemoryStatus.SUPERSEDED.value,
                                          {GeneralFieldNames.SUPERSEDED_BY: memory_id})
            except Exception as e:
                logger.error(f"Failed to update superseded_by for {old_id}: {e}")

        logger.debug(f"Added memory {memory_id}")
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
        """search memory by query"""

        # build query conditions
        conditions = [
            Condition(field=GeneralFieldNames.USER_ID, op="==", value=user_id),
            Condition(field=GeneralFieldNames.STATUS, op="==", value=MemoryStatus.ACTIVE.value)
        ]
        if min_confidence is not None:
            conditions.append(Condition(field=GeneralFieldNames.CONFIDENCE, op=">=", value=min_confidence))
        where = Query(conditions=conditions, logic="AND")

        # the number of result
        fetch_limit = limit * 2 if apply_decay else limit

        # execute query
        try:
            results = self.vector_store.search(
                memory_type=memory_type,
                query=query,
                where=where,
                limit=fetch_limit,
            )
        except Exception as e:
            raise MemoryRetrievalError(f"Search failed: {e}") from e

        # organize output and apply time decay
        memories = []
        for hit in results:
            try:
                model = StorageToMemoryMapper.from_db_dict(hit, memory_type)
            except Exception as e:
                logger.warning(f"Failed to deserialize memory {hit.get('id')}: {e}")
                continue
            similarity = hit.get(GeneralFieldNames.SCORE)
            mem = {
                GeneralFieldNames.ID: hit.get(GeneralFieldNames.ID),
                GeneralFieldNames.TEXT: hit.get(GeneralFieldNames.TEXT),
                GeneralFieldNames.METADATA: {k: v for k, v in model.model_dump().items()},
                GeneralFieldNames.DISTANCE: hit.get(GeneralFieldNames.DISTANCE, 0.0),
                GeneralFieldNames.SIMILARITY: similarity
            }

            if apply_decay and not model.permanent:
                decay_factor = self._calculate_decay_factor(model)
                mem[GeneralFieldNames.DECAYED_SIMILARITY] = similarity * decay_factor
            else:
                mem[GeneralFieldNames.DECAYED_SIMILARITY] = similarity
            memories.append(mem)

        if apply_decay:
            memories.sort(key=lambda x: x[GeneralFieldNames.DECAYED_SIMILARITY], reverse=True)

        # update last accessed
        for m in memories[:limit]:
            self._update_last_accessed(memory_type, m[GeneralFieldNames.ID])
        return memories[:limit]

    @retry_on_failure(max_retries=3, exceptions=(ChromaError,))
    def get_memory_by_entity(
            self,
            user_id: str,
            entity_key: str,
            status: str = MemoryStatus.ACTIVE.value
    ) -> List[Dict[str, Any]]:
        """get memory by entity key"""

        # build query conditions
        where = Query(conditions=[
            Condition(field=GeneralFieldNames.USER_ID, op="==", value=user_id),
            Condition(field=GeneralFieldNames.ENTITY_KEY, op="==", value=entity_key),
            Condition(field=GeneralFieldNames.STATUS, op="==", value=status)
        ])

        try:
            results = self.vector_store.get(
                memory_type=MemoryType.USER_PROFILE,
                where=where,
            )
        except Exception as e:
            logger.error(f"get by entity failed:{e}")
            return []

        return self._assemble_memories(results,MemoryType.USER_PROFILE)

    @retry_on_failure(max_retries=3, exceptions=(ChromaError,))
    def update_memory_status(
            self,
            memory_id: str,
            memory_type: MemoryType,
            new_status: str,
            metadata_updates: Optional[Dict[str, Any]] = None) -> bool:
        """update memory status"""
        updates = {GeneralFieldNames.STATUS: new_status}
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
            raise MemoryUpdateError(f"Update failed: {e}") from e

    def apply_forgetting(
            self,
            memory_type: MemoryType,
            user_id: Optional[str] = None,
            threshold: Optional[float] = None
    ) -> int:
        """apply forgetting"""
        #build conditions
        conditions = [
            Condition(field=GeneralFieldNames.STATUS, op="==", value=MemoryStatus.ACTIVE.value),
            Condition(field=GeneralFieldNames.PERMANENT,op="==",value=False)
        ]
        if user_id:
            conditions.append(Condition(field=GeneralFieldNames.USER_ID, op="==", value=user_id))
        where = Query(conditions=conditions, logic="AND")

        # query the memories that need to be forgotten
        try:
            res = self.vector_store.get(
                memory_type=MemoryType.USER_PROFILE,
                where=where
            )
        except Exception as e:
            logger.error(f"Forgetting scan failed: {e}")
            return 0

        count = 0
        for hit in res:
            try:
                model = StorageToMemoryMapper.from_db_dict(hit, memory_type)
            except Exception as e:
                continue

            threshold = threshold or config.decay_threshold
            decay_factor = self._calculate_decay_factor(model)
            if decay_factor < threshold:
                try:
                    self.update_memory_status(
                        hit[GeneralFieldNames.ID],
                        memory_type,
                        MemoryStatus.FORGOTTEN.value
                    )
                    count += 1
                except Exception:
                    pass
        logger.info(f"Forgotten {count} memories")
        return count

    def delete_user_memories(
            self,
            user_id: str,
            memory_type: Optional[MemoryType] = None
    ) -> bool:
        """delete user memory"""
        types = [memory_type] if memory_type else list(MemoryType)
        for mem_type in types:
            where = Query(conditions=[Condition(field=GeneralFieldNames.USER_ID, op="==", value=user_id)])
            try:
                self.vector_store.delete(memory_type=mem_type, where=where)
            except Exception as e:
                logger.error(f"Delete failed: {e}")
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
        return np.exp(-config.decay_factor * days)

    @retry_on_failure(max_retries=3, exceptions=(ChromaError,))
    def _update_last_accessed(self, memory_type: MemoryType, memory_id: str):
        try:
            self.vector_store.update(
                memory_type=memory_type,
                ids=[memory_id],
                metadatas=[{GeneralFieldNames.LAST_ACCESSED_AT: datetime.now().isoformat()}]
            )
        except Exception as e:
            logger.warning(f"Failed to update access time for {memory_id}: {e}")

    def get_recent_interactions(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:

        # build query condition
        where = Query(conditions=[
            Condition(field=GeneralFieldNames.USER_ID, op="==", value=user_id),
            Condition(field=GeneralFieldNames.STATUS, op="==", value=MemoryStatus.ACTIVE.value)
        ])

        # execute query
        try:
            results = self.vector_store.get(
                memory_type=MemoryType.INTERACTION_LOG,
                where=where,
                limit=limit * 3
            )
        except Exception as e:
            logger.error(f"Failed to retrieve interactions: {e}")
            return []

        # organize result data
        memories = self._assemble_memories(results,MemoryType.INTERACTION_LOG)

        # sort by time
        def get_timestamp(mem):
            ts = mem[GeneralFieldNames.METADATA].get(GeneralFieldNames.TIMESTAMP)
            if ts:
                try:
                    return datetime.fromisoformat(ts)
                except:
                    pass
            return datetime.min

        memories.sort(key=get_timestamp, reverse=True)
        return memories[:limit]

    def get_active_compliance_rules(self, limit: int = 10) -> List[Dict[str, Any]]:
        where = Query(conditions=[Condition(field=GeneralFieldNames.STATUS, op="==", value=MemoryStatus.ACTIVE.value)])

        try:
            results = self.vector_store.get(
                memory_type=MemoryType.COMPLIANCE_RULE,
                where=where
            )
        except Exception as e:
            logger.error(f"Failed to retrieve compliance rules: {e}")
            return []

        severity_order = {
            ComplianceSeverity.CRITICAL.value: 0,
            ComplianceSeverity.HIGH.value: 1,
            ComplianceSeverity.MEDIUM.value: 2,
            ComplianceSeverity.LOW.value: 3,
            ComplianceSeverity.MANDATORY.value: 0
        }

        rules = self._assemble_memories(results,MemoryType.COMPLIANCE_RULE)

        rules.sort(key=lambda r: (
            severity_order.get(r[GeneralFieldNames.METADATA].get(GeneralFieldNames.SEVERITY), 4),
            r[GeneralFieldNames.METADATA].get(GeneralFieldNames.PRIORITY, 100)
        ))
        return rules[:limit]

    def get_all_user_profile_memories(self, user_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
        # build condition
        conditions = [Condition(field=GeneralFieldNames.USER_ID, op="==", value=user_id)]
        if status:
            conditions.append(Condition(field=GeneralFieldNames.STATUS, op="==", value=status))
        where = Query(conditions=conditions,logic="AND")

        # execute query
        try:
            results = self.vector_store.get(
                memory_type=MemoryType.USER_PROFILE,
                where=where
            )
        except Exception as e:
            logger.error(f"Failed to get user profile memories: {e}")
            return []

        return self._assemble_memories(results,MemoryType.USER_PROFILE)

    def _assemble_memories(self,results: List[Dict[str, Any]],memory_type: MemoryType) -> List[Dict[str, Any]]:
        memories = []
        for i, hit in enumerate(results):
            try:
                model = StorageToMemoryMapper.from_db_dict(hit, memory_type)
            except Exception as e:
                print(e)
                continue
            memories.append({
                GeneralFieldNames.ID: hit.get(GeneralFieldNames.ID),
                GeneralFieldNames.TEXT: hit.get(GeneralFieldNames.TEXT),
                GeneralFieldNames.METADATA: {k: v for k, v in model.model_dump().items()},
            })
        return memories

    def get_profile_summary(self,user_id: str,max_chars: int = 500) -> str:
        """
        Get a brief summary of the user's current active profile for extracting node injection prompts

        Args:
            user_id: unique user id
            max_chars: maximum number of characters to display

        Returns:
            each line "- entity_key: content (confidence X.XX)"
            If there is no profile, return "No known profile."
        """
        try:
            memories = self.get_all_user_profile_memories(user_id,MemoryStatus.ACTIVE.value)
            if not memories:
                return "暂无已知用户画像"

            #messages are sorted by last accessed time or creation time
            def get_ts(mem):
                meta = mem.get(GeneralFieldNames.METADATA,{})
                ts_str = meta.get(GeneralFieldNames.LAST_ACCESSED_AT) or meta.get(GeneralFieldNames.CREATED_AT)
                if ts_str:
                    try:
                        return datetime.fromisoformat(ts_str)
                    except:
                        pass
                    return datetime.min
            memories.sort(key=get_ts, reverse=True)

            #sensitive field desensitization regex(example: phone/email)
            SENSITIVE_KEYS = {'contact', 'phone', 'email'}
            def mask_content(key,content):
                if not isinstance(content,str):
                    return content
                if key in SENSITIVE_KEYS:
                    if len(content) > 4:
                        return content[:2] + "*" * (len(content)-4) + content[-2:]
                    else:
                        return "***"

            lines = []
            total_chars = 0
            for mem in memories:
                metadata = mem.get(GeneralFieldNames.METADATA, {})
                entity_key = metadata.get(GeneralFieldNames.ENTITY_KEY, 'unknown')
                content = mem.get(GeneralFieldNames.TEXT, '')
                confidence = metadata.get(GeneralFieldNames.CONFIDENCE, 0.0)
                #mask sensitive content
                content_masked = mask_content(entity_key, content)
                line = f"- {entity_key}: {content_masked} (置信度 {confidence:.2f})"

                if total_chars + len(line) > max_chars:
                    break
                lines.append(line)
                total_chars += len(line) + 1

            if not lines:
                return "暂无已知用户画像"
            result = "\n".join(lines)
            return result
        except Exception as e:
            logger.error(f"Failed to get user profile summary: {e}")
            return "暂无已知用户画像"

    def get_extraction_cursor(self, user_id: str) -> Optional[int]:
        logger.info(f"get_extraction_cursor called for {user_id} (not implemented)")
        return None

    def set_extraction_cursor(self, user_id: str, message_index: int) -> None:
        logger.info(f"set_extraction_cursor called for {user_id} to {message_index} (not implemented)")
