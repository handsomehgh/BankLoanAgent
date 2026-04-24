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

from config import config
from exception import MemoryWriteFailedError, MemoryRetrievalError, MemoryUpdateError
from memory.classifiers.rules.rules_loader import get_evidence_loader
from memory.db_adpter.adpter_model.query_model import Condition, Query
from memory.db_adpter.query_builder import QueryBuilder
from memory.memory_store.memory_base import BaseMemoryStore
from memory.constant.constants import MetadataFields, MemoryType, MemorySource, MemoryStatus, MemoryModelFields, \
    ChromaOperator, ChromaResFields, ComplianceSeverity, EvidenceType, InteractionSentiment, ComplianceRuleFields
from memory.models.schema import ComplianceRuleMetadata, UserProfileMetadata, InteractionLogMetadata
from memory.memory_vector_store.vector_store import BaseVectorStore
from utils.retry import retry_on_failure

logger = logging.Logger(__name__)


class LongTermMemoryStore(BaseMemoryStore):
    COLLECTION_NAMES = {
        MemoryType.USER_PROFILE: "user_profile_memories",
        MemoryType.INTERACTION_LOG: "interaction_logs",
        MemoryType.COMPLIANCE_RULE: "compliance_rules",
    }

    def __init__(self, vector_store: BaseVectorStore,query_builder: QueryBuilder):
        """
        initial chroma client and collection 
        
        Args:
            vector_store: vector databases
            query_builder: builder query
        """
        self.vector_store = vector_store
        self.query_builder = query_builder

        # dead letter queue
        self.dlq_path = Path(config.memory_dlq_path)
        self.dlq_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("ChromaMemoryStore initialized with vector_store")

    def _get_collection_name(self, memory_type: MemoryType) -> str:
        if memory_type not in self.COLLECTION_NAMES:
            raise ValueError(f"Unsupported memory type: {memory_type}")
        return self.COLLECTION_NAMES[memory_type]

    def _write_to_dlq(self, user_id: str, content: str, meta: Dict, memory_type: MemoryType, permanent: bool):
        entry = {
            MetadataFields.USER_ID.value: user_id,
            MemoryModelFields.CONTENT.value: content,
            MetadataFields.MEMORY_TYPE.value: memory_type.value,
            MemoryModelFields.METADATA.value: meta,
            MetadataFields.PERMANENT.value: permanent,
            MetadataFields.TIMESTAMP.value: datetime.now().isoformat(),
            MetadataFields.RETRY_COUNT.value: 0
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
        now = datetime.now().isoformat()

        # conver dict to chroma dict
        meta_input = metadata.copy() if metadata else {}
        try:
            if memory_type == MemoryType.USER_PROFILE:
                model = UserProfileMetadata(
                    user_id=user_id,
                    memory_type=memory_type,
                    source=meta_input.get(MetadataFields.SOURCE.value, MemorySource.CHAT_EXTRACTION.value),
                    confidence=meta_input.get(MetadataFields.CONFIDENCE.value, 0.8),
                    status=meta_input.get(MetadataFields.STATUS.value, MemoryStatus.ACTIVE.value),
                    permanent=permanent,
                    created_at=meta_input.get(MetadataFields.CREATE_AT.value, now),
                    last_accessed_at=meta_input.get(MetadataFields.LAST_ACCESS_AT.value, now),
                    superseded_by=meta_input.get(MetadataFields.SUPERSEDED_BY.value),
                    entity_key=entity_key,
                    evidence_type=meta_input.get(MetadataFields.EVIDENCE_TYPE.value,
                                                 EvidenceType.EXPLICIT_STATEMENT.value),
                    effective_date=meta_input.get(MetadataFields.EFFECTIVE_DATE.value, now),
                    expires_at=meta_input.get(MetadataFields.EXPIRES_AT.value),
                    extra=meta_input.get(MetadataFields.EXTRA.value, {})
                )
            elif memory_type == MemoryType.INTERACTION_LOG:
                model = InteractionLogMetadata(
                    user_id=user_id,
                    memory_type=memory_type,
                    source=meta_input.get(MetadataFields.SOURCE.value, MemorySource.AUTO_SUMMARY.value),
                    confidence=meta_input.get(MetadataFields.CONFIDENCE.value, 1.0),
                    status=meta_input.get(MetadataFields.STATUS.value, MemoryStatus.ACTIVE.value),
                    permanent=permanent,
                    created_at=meta_input.get(MetadataFields.CREATE_AT.value, now),
                    last_accessed_at=meta_input.get(MetadataFields.LAST_ACCESS_AT.value, now),
                    superseded_by=meta_input.get(MetadataFields.SUPERSEDED_BY.value),
                    event_type=meta_input.get(MetadataFields.EVENT_TYPE.value, InteractionLogMetadata.INQUIRY.value),
                    session_id=meta_input.get(MetadataFields.SESSION_ID.value, "unknown"),
                    sentiment=meta_input.get(MetadataFields.SENTIMENT.value, InteractionSentiment.NEUTRAL.value),
                    key_entities=meta_input.get(MetadataFields.KEY_ENTITIES.value, []),
                    timestamp=meta_input.get(MetadataFields.TIMESTAMP.value, now),
                    extra=meta_input.get(MetadataFields.EXTRA.value, {})
                )
            elif memory_type == MemoryType.COMPLIANCE_RULE:
                model = ComplianceRuleMetadata(
                    user_id=user_id,
                    memory_type=memory_type,
                    source=meta_input.get(MetadataFields.SOURCE.value, MemorySource.ADMIN_IMPORT.value),
                    confidence=meta_input.get(MetadataFields.CONFIDENCE.value, 1.0),
                    status=meta_input.get(MetadataFields.STATUS.value, MemoryStatus.ACTIVE.value),
                    permanent=permanent,
                    created_at=meta_input.get(MetadataFields.CREATE_AT.value, now),
                    last_accessed_at=meta_input.get(MetadataFields.LAST_ACCESS_AT.value, now),
                    superseded_by=meta_input.get(MetadataFields.SUPERSEDED_BY.value),
                    rule_id=meta_input[ComplianceRuleFields.RULE_ID.value],
                    rule_name=meta_input[ComplianceRuleFields.RULE_NAME.value],
                    rule_type=meta_input[ComplianceRuleFields.RULE_TYPE.value],
                    pattern=meta_input.get(ComplianceRuleFields.PATTERN.value, ""),
                    action=meta_input[ComplianceRuleFields.ACTION.value],
                    severity=meta_input.get(ComplianceRuleFields.SEVERITY.value, ComplianceSeverity.MEDIUM.value),
                    priority=meta_input.get(ComplianceRuleFields.PRIORITY.value, 100),
                    version=meta_input.get(ComplianceRuleFields.VERSION.value, now.strftime("%Y-%m-%d")),
                    effective_from=meta_input.get(ComplianceRuleFields.EFFECTIVE_FROM.value, now),
                    effective_to=meta_input.get(ComplianceRuleFields.EFFECTIVE_TO.value),
                    template=meta_input.get(ComplianceRuleFields.TEMPLATE.value),
                    extra=meta_input.get(MetadataFields.EXTRA.value, {})
                )
            else:
                raise ValueError(f"Unsupported memory type: {memory_type}")
        except Exception as e:
            logger.error(f"Failed to validate metadata with Pydantic model: {e}")
            self._write_to_dlq(user_id, content, memory_type, meta_input, permanent)
            raise MemoryWriteFailedError(f"Memory validation failed, queued: {e}") from e

        # user profile specific field
        if memory_type == MemoryType.USER_PROFILE:
            meta_input[MetadataFields.ENTITY_KEY.value] = entity_key

        # conflict detection(only required for user profile)
        superseded_memories = []
        if memory_type == MemoryType.USER_PROFILE and entity_key and not permanent:
            existing = []
            try:
                existing = self.get_memory_by_entity(user_id, entity_key, MemoryStatus.ACTIVE.value)
            except Exception as e:
                logger.warning(f"Conflict check failed,proceeding: {e}")

            try:
                loader = get_evidence_loader()
                evidence_weights = loader.get_evidence_weights()
            except Exception as e:
                logger.warning(f"Failed to load evidence weights,user default: {e}")
                evidence_weights = {}

            # update the memory status to superseded
            for old in existing:
                old_meta = old[MemoryModelFields.METADATA.value]
                old_conf = float(old.get(MetadataFields.CONFIDENCE.value, 0.0))
                old_evidence = old.get(MetadataFields.EVIDENCE_TYPE.value, EvidenceType.EXPLICIT_STATEMENT.value)
                old_weight = evidence_weights.get(old_evidence,50)

                new_conf = model.confidence
                new_evidence = model.evidence_type if hasattr(model, MetadataFields.EVIDENCE_TYPE.value) else EvidenceType.EXPLICIT_STATEMENT.value
                new_weight = evidence_weights.get(new_evidence,50)

                # Overriding Conditions:
                # 1. The new confidence is significantly higher (> old_conf + 0.1)
                # 2. Confidence difference is not large (≤ 0.1) but the new evidence is more authoritative
                if new_conf > old_conf + 0.1 or (abs(new_conf - old_conf) <= 0,1) and new_weight > old_weight:
                    try:
                        self.update_memory_status(old[MemoryModelFields.ID.value], memory_type,
                                                  MemoryStatus.SUPERSEDED.value,
                                                  {MetadataFields.SUPERSEDED_BY.value: None})
                        superseded_memories.append(old["id"])
                        logger.info(f"Superseded old memory {old['id']} (entity: {entity_key})")
                    except Exception as e:
                        logger.warning(f"Failed to superseded {old[MemoryModelFields.ID.value]: {e}}")

        # add memory
        memory_id = str(uuid.uuid4())
        try:
            collection_name = self._get_collection_name(memory_type)
            self.vector_store.add(collection_name=collection_name,ids=[memory_id],texts=[content],metadatas=model.model_dump())
        except Exception as e:
            # write to dlq
            logger.error(f"Write failed for user {user_id}: {e}")
            self._write_to_dlq(user_id, content, model.to_chroma_dict(), memory_type, permanent)
            raise MemoryWriteFailedError(f"Memory write failed, queued: {e}") from e

        # update the superseded_by of the memory to the new memory ID
        for old in superseded_memories:
            if old[MemoryModelFields.METADATA.value].get(MetadataFields.STATUS.value) == MemoryStatus.SUPERSEDED.value:
                try:
                    self.update_memory_status(old[MemoryModelFields.ID.value], memory_type,
                                              MemoryStatus.SUPERSEDED.value,
                                              {MetadataFields.SUPERSEDED_BY.value: memory_id})
                except Exception as e:
                    logger.error(f"Failed to update superseded_by for {old[MemoryModelFields.ID.value]}: {e}")

        logger.debug(f"Added memory {memory_id}")
        return memory_id

    @retry_on_failure(max_retries=3, initial_delay=0.2, exceptions=(ChromaError, MemoryRetrievalError))
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

        collection_name = self._get_collection_name(memory_type)

        # build query conditions
        conditions = [
            Condition(field=MetadataFields.USER_ID.value,op="==",value=user_id),
            Condition(field=MetadataFields.STATUS.value,op="==",value=MemoryStatus.ACTIVE.value)
        ]
        if min_confidence:
            conditions.append(Condition(field=MetadataFields.CONFIDENCE.value,op=">=",value=min_confidence))
        query_obj = Query(conditions=conditions,logic="AND")
        where = self.query_builder.build(query_obj)

        # the number of result
        fetch_limit = limit * 2 if apply_decay else limit

        # execute query
        try:
            results = self.vector_store.search(
                collection_name=collection_name,
                query=query,
                where=where,
                limit=fetch_limit,
                include=[ChromaResFields.DOCUMENTS.value, ChromaResFields.METADATAS.value,
                         ChromaResFields.DISTANCES.value]
            )
        except Exception as e:
            raise MemoryRetrievalError(f"Search failed: {e}") from e

        # organize output and apply time decay
        memories = []
        if results[ChromaResFields.IDS.value][0]:
            for i, doc in enumerate(results[ChromaResFields.DOCUMENTS.value][0]):
                try:
                    if memory_type == MemoryType.USER_PROFILE:
                        model = UserProfileMetadata.from_chroma_dict(results[ChromaResFields.METADATAS.value][0][i])
                    elif memory_type == MemoryType.INTERACTION_LOG:
                        model = InteractionLogMetadata.from_chroma_dict(results[ChromaResFields.METADATAS.value][0][i])
                    elif memory_type == MemoryType.COMPLIANCE_RULE:
                        model = ComplianceRuleMetadata.from_chroma_dict(results[ChromaResFields.METADATAS.value][0][i])
                    else:
                        continue
                except Exception as e:
                    logger.warning(f"Failed to deserialize memory {results['ids'][0][i]}: {e}")
                    continue

                mem = {
                    MemoryModelFields.ID.value: results[ChromaResFields.IDS.value][0][i],
                    MemoryModelFields.CONTENT.value: doc,
                    MemoryModelFields.METADATA.value: model.model_dump(exclude={"extra"}),
                    MemoryModelFields.DISTANCE.value: results[ChromaResFields.DISTANCES.value][0][i],
                    MemoryModelFields.SIMILARITY.value: 1 - results[ChromaResFields.DISTANCES.value][0][i],
                }

                if apply_decay and not mem[MemoryModelFields.METADATA.value].get(MetadataFields.PERMANENT.value):
                    decay_factor = self._calculate_decay_factor(model)
                    mem[MemoryModelFields.DECAYED_SIMILARITY.value] = mem[MemoryModelFields.SIMILARITY.value] * decay_factor
                else:
                    mem[MemoryModelFields.DECAYED_SIMILARITY.value] = mem[MemoryModelFields.SIMILARITY.value]
                memories.append(mem)

        # reordering the memories after time decay
        if apply_decay:
            memories.sort(key=lambda x: x[MemoryModelFields.DECAYED_SIMILARITY.value], reverse=True)

        # update last accessed
        for m in memories[:limit]:
            self._update_last_accessed(memory_type, m[MemoryModelFields.ID.value])
        return memories[:limit]

    @retry_on_failure(max_retries=3, exceptions=(ChromaError,))
    def get_memory_by_entity(
            self,
            user_id: str,
            entity_key: str,
            status: str = MemoryStatus.ACTIVE.value
    ) -> List[Dict[str, Any]]:
        """get memory by entity key"""
        collection_name = self._get_collection_name(MemoryType.USER_PROFILE)

        #build query conditions
        query_obj = Query(conditions=[
            Condition(field=MetadataFields.USER_ID.value,op="==",value=user_id),
            Condition(field=MetadataFields.ENTITY_KEY.value,op="==",value=entity_key),
            Condition(field=MetadataFields.STATUS.value,op="==",value=status)
        ])
        where = self.query_builder.build(query_obj)

        try:
            results = self.vector_store.get(
                collection_name=collection_name,
                where=where,
                include=[ChromaResFields.DOCUMENTS.value, ChromaResFields.METADATAS.value]
            )
        except Exception as e:
            logger.error(f"get by entity failed:{e}")
            return []

        memories = []
        if results[ChromaResFields.IDS.value]:
            for i, doc in enumerate(results[ChromaResFields.DOCUMENTS.value]):
                memories.append({
                    MemoryModelFields.ID.value: results[ChromaResFields.IDS.value][i],
                    MemoryModelFields.CONTENT.value: doc,
                    MemoryModelFields.METADATA.value: results[ChromaResFields.METADATAS.value][i]
                })
        return memories

    @retry_on_failure(max_retries=3, exceptions=(ChromaError,))
    def update_memory_status(
            self,
            memory_id: str,
            memory_type: MemoryType,
            new_status: str,
            metadata_updates: Optional[Dict[str, Any]] = None) -> bool:
        """update memory status"""
        collection_name = self._get_collection_name(memory_type)
        try:
            current = self.vector_store.get(collection_name=collection_name,ids=[memory_id], include=[ChromaResFields.METADATAS.value])
            if not current[ChromaResFields.METADATAS.value]:
                return False

            new_meta = current[ChromaResFields.METADATAS.value][0].copy()
            new_meta[MetadataFields.STATUS.value] = new_status
            if metadata_updates:
                new_meta.update(metadata_updates)
            self.vector_store.update(collection_name=collection_name,ids=[memory_id], metadatas=[new_meta])
            return True
        except Exception as e:
            raise MemoryUpdateError(f"Update failed: {e}") from e

    def apply_forgetting(
            self,
            memory_type: Optional[MemoryType] = None,
            user_id: Optional[str] = None,
            threshold: Optional[float] = None
    ) -> int:
        """apply forgetting"""
        if memory_type and memory_type != MemoryType.USER_PROFILE:
            logger.info(f"Skipping forgetting for {memory_type.value} (not supported)")
            return 0

        conditions = [
            Condition(field=MetadataFields.STATUS.value,op="==",value=MemoryStatus.ACTIVE.value),
        ]
        if user_id:
            conditions.append(Condition(field=MetadataFields.USER_ID.value,op="==",value=user_id))
        query_obj = Query(conditions=conditions,logic="AND")
        where = self.query_builder.build(query_obj)

        # query the memories that need to be forgotten
        collection_name = self._get_collection_name(MemoryType.USER_PROFILE)
        try:
            res = self.vector_store.get(
                collection_name=collection_name,
                where=where,
                include=[ChromaResFields.METADATAS.value]
            )
        except Exception as e:
            logger.error(f"Forgetting scan failed: {e}")
            return 0

        count = 0
        for i, mem_id in enumerate(res[ChromaResFields.IDS.value]):
            meta = res[ChromaResFields.METADATAS.value][i]
            if meta.get(MetadataFields.PERMANENT.value):
                continue
            try:
                model = UserProfileMetadata.from_chroma_dict(meta)
            except Exception:
                continue

            threshold = threshold if threshold else config.decay_threshold
            decay_factor = self._calculate_decay_factor(model)
            if decay_factor < threshold:
                try:
                    self.update_memory_status(mem_id, MemoryType.USER_PROFILE, MemoryStatus.FORGOTTEN.value)
                    count += 1
                except:
                    pass
        logger.info(f"Forgotten {count} memories")
        return count

    def delete_user_memories(self, user_id: str, memory_type: Optional[MemoryType] = None) -> bool:
        """delete user memory"""
        types = [memory_type] if memory_type else [v.value for v in MemoryType]
        for mem_type in types:
            collection_name = self._get_collection_name(mem_type)
            query_obj = Query(conditions=[Condition(field=MetadataFields.USER_ID.value,op="==",value=user_id)])
            where = self.query_builder.build(query_obj)
            try:
                self.vector_store.delete(collection_name=collection_name,where=where)
                return True
            except Exception as e:
                logger.error(f"Delete failed: {e}")
                return False
        return True

    def _calculate_decay_factor(self, model) -> float:
        """apply decay(original similarity * e ** (-decay_factor * (now()-last_accessed)))"""
        last = model.lass_accessed_at
        if not last:
            return model.similarity
        try:
            days = (datetime.now() - datetime.fromisoformat(last)).days
        except:
            days = 0
        return np.exp(-config.decay_lambda * days)


    @retry_on_failure(max_retries=3, exceptions=(ChromaError,))
    def _update_last_accessed(self, memory_type: MemoryType, memory_id: str):
        collection_name = self._get_collection_name(memory_type)
        try:
            self.vector_store.update(
                collection_name=collection_name,
                ids=[memory_id],
                metadatas=[{MetadataFields.LAST_ACCESS_AT.value: datetime.now().isoformat()}]
            )
        except Exception as e:
            logger.warning(f"Failed to update access time for {memory_id}: {e}")

    def get_recent_interactions(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        collection_name = self._get_collection_name(MemoryType.INTERACTION_LOG)

        # build query condition
        where = self.query_builder.build(Query(conditions=[
            Condition(field=MetadataFields.USER_ID.value,op="==",value=user_id),
            Condition(field=MetadataFields.STATUS.value,op="==",value=MemoryStatus.ACTIVE.value)
        ]))

        # execute query
        try:
            results = self.vector_store.get(
                collection_name=collection_name,
                where=where,
                limit=limit * 3,
                include=[ChromaResFields.METADATAS.value, ChromaResFields.DOCUMENTS.value]
            )
        except Exception as e:
            logger.error(f"Failed to retrieve interactions: {e}")
            return []

        # organize result data
        memories = []
        if results[ChromaResFields.IDS.value]:
            for i, doc in enumerate(results[ChromaResFields.DOCUMENTS.value]):
                try:
                    model = InteractionLogMetadata.from_chroma_dict(results[ChromaResFields.METADATAS.value][i])
                except Exception:
                    continue
                memories.append({
                    MemoryModelFields.ID.value: results[ChromaResFields.IDS.value][i],
                    MemoryModelFields.CONTENT.value: doc,
                    MemoryModelFields.METADATA.value: model.model_dump(exclude={MetadataFields.EXTRA.value}),
                    MemoryModelFields.SIMILARITY.value: 1.0,
                    MemoryModelFields.DECAYED_SIMILARITY.value: 1.0
                })

        # sort by time
        def get_timestamp(mem):
            ts = mem[MemoryModelFields.METADATA.value].get(MetadataFields.TIMESTAMP.value)
            if ts:
                try:
                    return datetime.fromisoformat(ts)
                except:
                    pass
            return datetime.min

        memories.sort(key=get_timestamp, reverse=True)
        return memories[:limit]

    def get_active_compliance_rules(self, limit: int = 10) -> List[Dict[str, Any]]:
        collection_name = self._get_collection_name(MemoryType.COMPLIANCE_RULE)
        where = self.query_builder.build(Query(conditions=[Condition(field=MetadataFields.STATUS.value,op="==",value=MemoryStatus.ACTIVE.value)]))

        try:
            results = self.vector_store.get(
                collection_name=collection_name,
                where=where,
                include=[ChromaResFields.METADATAS.value, ChromaResFields.DOCUMENTS.value]
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

        rules = []
        if results[ChromaResFields.IDS.value]:
            for i, doc in enumerate(results[ChromaResFields.DOCUMENTS.value]):
                try:
                    model = ComplianceRuleMetadata.from_chroma_dict(results[ChromaResFields.METADATAS.value][i])
                except Exception:
                    continue
                rules.append({
                    MemoryModelFields.ID.value: results[ChromaResFields.IDS.value][i],
                    MemoryModelFields.CONTENT.value: doc,
                    MemoryModelFields.METADATA.value: model.model_dump(exclude={MetadataFields.EXTRA.value}),
                })

        rules.sort(key=lambda r: (
            severity_order.get(r["metadata"].get("severity"), 4),
            r["metadata"].get("priority", 100)
        ))
        return rules[:limit]

    def get_all_user_profile_memories(self, user_id: str, status: str = "active") -> List[Dict[str, Any]]:
        collection_name = self._get_collection_name(MemoryType.USER_PROFILE)

        # build condition
        where = self.query_builder.build(Query(conditions=[
            Condition(field=MetadataFields.USER_ID.value,op="==",value=user_id),
            Condition(field=MetadataFields.STATUS.value,op="==",value=MemoryStatus.ACTIVE.value),
        ]))

        # execute query
        try:
            results = self.vector_store.get(
                collection_name=collection_name,
                where=where,
                include=[ChromaResFields.METADATAS.value, ChromaResFields.DOCUMENTS.value]
            )
        except Exception as e:
            logger.error(f"Failed to get user profile memories: {e}")
            return []

        memories = []
        if results[ChromaResFields.IDS.value]:
            for i, doc in enumerate(results[ChromaResFields.DOCUMENTS.value]):
                try:
                    model = UserProfileMetadata.from_chroma_dict(results[ChromaResFields.METADATAS.value][i])
                except Exception:
                    continue
                memories.append({
                    MemoryModelFields.ID.value: results[ChromaResFields.IDS.value][i],
                    MemoryModelFields.CONTENT.value: doc,
                    MemoryModelFields.METADATA.value: model.model_dump(exclude={MetadataFields.EXTRA.value}),
                })
        return memories


if __name__ == '__main__':
    store = LongTermMemoryStore("../../test")
    # store.add_memory(user_id="hgh001",content="这是测试文件2",memory_type=MemoryType.USER_PROFILE,entity_key="test",metadata={"type": "user_profile","source": "test","confidence": 0.6},permanent=False)

    # result = store.search_memory("hgh001","测试",MemoryType.USER_PROFILE,2)
    # store.apply_forgetting(MemoryType.USER_PROFILE, "hgh001",2)
    result = store.get_memory_by_entity("hgh001", "test", MemoryStatus.FORGOTTEN.value)
    # store.delete_user_memories("hgh001")
    print(result)
