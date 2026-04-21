# author hgh
# version 1.0
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import chromadb
import numpy as np
from chromadb import Settings
from chromadb.errors import ChromaError
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config import config
from exception import MemoryWriteFailedError, MemoryRetrievalError, MemoryUpdateError
from memory.base import BaseMemoryStore
from memory.constant.constants import MetadataFields, MemoryType, MemorySource, MemoryStatus, MemoryModelFields, \
    ChromaOperator, ChromaResFields, ComplianceSeverity
from utils.retry import retry_on_failure

logger = logging.Logger(__name__)


class ChromaMemoryStore(BaseMemoryStore):

    COLLECTION_NAMES = {
        MemoryType.USER_PROFILE: "user_profile_memories",
        MemoryType.INTERACTION_LOG: "interaction_logs",
        MemoryType.COMPLIANCE_RULE: "compliance_rules",
    }

    def __init__(self, persist_dir: str):
        """
        initial chroma client and collection 
        
        Args:
            persist_dir: data persistence folder
        """
        self.persist_dir = persist_dir

        # dead letter queue
        self.dlq_path = Path(persist_dir) / "memory_dlq.jsonl"
        self.dlq_path.parent.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        #get or create collections
        self.collections = {}
        try:
            for mem_type, coll_name in self.COLLECTION_NAMES.items():
                self.collections[mem_type] = self._get_or_create_collection(coll_name)
                logger.info(f"Collection '{coll_name}' initialized for {mem_type.value}")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma collection: {e}")
            raise

        logger.info(f"chroma initial at {persist_dir}")

    def _get_collection(self, memory_type: MemoryType):
        """get collection by memory type"""
        if memory_type not in self.collections:
            raise ValueError(f"Unsupported memory type: {memory_type}")
        return self.collections[memory_type]

    @retry_on_failure(max_retries=3, initial_delay=0.3, exceptions=(ChromaError,))
    def _get_or_create_collection(self,collection_name: str):
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=OpenAIEmbeddingFunction(api_key=config.alibaba_api_key,
                                                       model_name=config.qwen_emb_name,
                                                       api_base=config.alibaba_base_url,
                                                       dimensions=1024)
        )

    def _write_to_dlq(self, user_id: str, content: str, meta: Dict,memory_type: MemoryType):
        entry = {
            MetadataFields.USER_ID.value: user_id,
            MemoryModelFields.CONTENT.value: content,
            MetadataFields.TYPE.value: memory_type.value,
            MemoryModelFields.METADATA.value: meta,
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
        meta = metadata.copy() if metadata else {}
        meta = {
            MetadataFields.USER_ID.value: user_id,
            MetadataFields.SOURCE.value: metadata.get(MetadataFields.SOURCE.value, MemorySource.CHAT_EXTRACTION.value),
            MetadataFields.CONFIDENCE.value: metadata.get(MetadataFields.CONFIDENCE.value, 0.8),
            MetadataFields.STATUS.value: metadata.get(MetadataFields.STATUS.value, MemoryStatus.ACTIVE.value),
            MetadataFields.PERMANENT.value: permanent,
            MetadataFields.CREATE_AT.value: now,
            MetadataFields.LAST_ACCESS_AT.value: now,
        }

        #user profile specific field
        if memory_type == MemoryType.USER_PROFILE:
            meta[MetadataFields.ENTITY_KEY.value]  = entity_key

        #conflict detection(only required for user profile)
        existing = []
        if memory_type == MemoryType.USER_PROFILE and entity_key and not permanent:
            try:
                existing = self.get_memory_by_entity(user_id, entity_key, MemoryStatus.ACTIVE.value)
            except Exception as e:
                logger.warning(f"Conflict check failed,proceeding: {e}")

            # update the memory status to superseded
            for old in existing:
                if float(meta[MetadataFields.CONFIDENCE.value]) > float(old[MemoryModelFields.METADATA.value].get(MetadataFields.CONFIDENCE.value, 0)) + 0.1:
                    try:
                        self.update_memory_status(old[MemoryModelFields.ID.value], memory_type,MemoryStatus.SUPERSEDED.value,{MetadataFields.SUPERSEDED_BY.value:None})
                    except Exception as e:
                        logger.warning(f"Failed to superseded {old[MemoryModelFields.ID.value]: {e}}")

        # add memory
        memory_id = str(uuid.uuid4())
        try:
            collection = self._get_collection(memory_type)
            collection.add(ids=[memory_id],documents=[content],metadatas=meta)
        except Exception as e:
            # write to dlq
            logger.error(f"Write failed for user {user_id}: {e}")
            self._write_to_dlq(user_id, content, meta, memory_type)
            raise MemoryWriteFailedError(f"Memory write failed, queued: {e}") from e

        # update the superseded_by of the memory to the new memory ID
        for old in existing:
            if old[MemoryModelFields.METADATA.value].get(MetadataFields.STATUS.value) == MemoryStatus.SUPERSEDED.value:
                try:
                    self.update_memory_status(old[MemoryModelFields.ID.value],memory_type,MemoryStatus.SUPERSEDED.value,{MetadataFields.SUPERSEDED_BY.value: memory_id})
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
            apply_decay: bool = True
    ) -> List[Dict[str, Any]]:
        """search memory by query"""

        collection = self._get_collection(memory_type)

        # build query conditions
        conditions = [{MetadataFields.USER_ID.value: {ChromaOperator.EQ.value: user_id}}]
        if memory_type != MemoryType.COMPLIANCE_RULE:
            conditions.append({MetadataFields.STATUS.value: {ChromaOperator.EQ.value: MemoryStatus.ACTIVE.value}})
        if min_confidence:
            conditions.append({MetadataFields.CONFIDENCE.value: {ChromaOperator.GTE.value: min_confidence}})
        where = conditions[0] if len(conditions) == 1 else {ChromaOperator.AND.value: conditions}

        # the number of result
        fetch_limit = limit * 2 if apply_decay else limit

        # execute query
        try:
            results = collection.query(
                query_texts=[query],
                where=where,
                n_results=fetch_limit,
                include=[ChromaResFields.DOCUMENTS.value, ChromaResFields.METADATAS.value,ChromaResFields.DISTANCES.value]
            )
        except Exception as e:
            raise MemoryRetrievalError(f"Search failed: {e}") from e

        # organize output and apply time decay
        memories = []
        if results[ChromaResFields.IDS.value][0]:
            for i, doc in enumerate(results[ChromaResFields.DOCUMENTS.value][0]):
                mem = {
                    MemoryModelFields.ID.value: results[ChromaResFields.IDS.value][0][i],
                    MemoryModelFields.CONTENT.value: doc,
                    MemoryModelFields.METADATA.value: results[ChromaResFields.METADATAS.value][0][i],
                    MemoryModelFields.DISTANCE.value: results[ChromaResFields.DISTANCES.value][0][i],
                    MemoryModelFields.SIMILARITY.value: 1 - results[ChromaResFields.DISTANCES.value][0][i],
                }

                if apply_decay and not mem[MemoryModelFields.METADATA.value].get(MetadataFields.PERMANENT.value):
                    mem[MemoryModelFields.DECAYED_SIMILARITY.value] = self._apply_decay(mem)
                else:
                    mem[MemoryModelFields.DECAYED_SIMILARITY.value] = mem[MemoryModelFields.SIMILARITY.value]
                memories.append(mem)

        # reordering the memories after time decay
        if apply_decay:
            memories.sort(key=lambda x: x[MemoryModelFields.DECAYED_SIMILARITY.value], reverse=True)

        # update last accessed
        for m in memories[:limit]:
            self._update_last_accessed(memory_type,m[MemoryModelFields.ID.value])
        return memories[:limit]

    @retry_on_failure(max_retries=3, exceptions=(ChromaError,))
    def get_memory_by_entity(
            self,
            user_id: str,
            entity_key: str,
            status: str = MemoryStatus.ACTIVE.value
    ) -> List[Dict[str, Any]]:
        """get memory by entity key"""
        collection = self._get_collection(MemoryType.USER_PROFILE)

        where_filter = {ChromaOperator.AND.value:
            [
                {MetadataFields.USER_ID.value: {ChromaOperator.EQ.value: user_id}},
                {MetadataFields.ENTITY_KEY.value: {ChromaOperator.EQ.value: entity_key}},
                {MetadataFields.STATUS.value: {ChromaOperator.EQ.value: status}}
            ]
        }

        try:
            results = collection.get(
                where=where_filter,
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
        collection = self._get_collection(memory_type)
        try:
            current = collection.get(ids=[memory_id], include=[ChromaResFields.METADATAS.value])
            if not current[ChromaResFields.METADATAS.value]:
                return False

            new_meta = current[ChromaResFields.METADATAS.value][0].copy()
            new_meta[MetadataFields.STATUS.value] = new_status
            if metadata_updates:
                new_meta.update(metadata_updates)
            collection.update(ids=[memory_id], metadatas=[new_meta])
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

        collection = self._get_collection(MemoryType.USER_PROFILE)

        threshold = threshold if threshold else config.decay_threshold
        conditions = [{MetadataFields.STATUS.value: {ChromaOperator.EQ.value: MemoryStatus.ACTIVE.value}}]
        if user_id:
            conditions.append({MetadataFields.USER_ID.value: {ChromaOperator.EQ.value: user_id}})
        where = conditions if len(conditions) == 1 else {ChromaOperator.AND.value: conditions}

        # query the memories that need to be forgotten
        try:
            res = collection.get(
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
            mem = {MemoryModelFields.METADATA.value: meta, MemoryModelFields.SIMILARITY.value: 1.0}
            if self._apply_decay(mem) < float(threshold):
                try:
                    self.update_memory_status(mem_id, MemoryType.USER_PROFILE,MemoryStatus.FORGOTTEN.value)
                    count += 1
                except:
                    pass
        logger.info(f"Forgotten {count} memories")
        return count

    def delete_user_memories(self, user_id: str,memory_type: Optional[MemoryType] = None) -> bool:
        """delete user memory"""
        types = memory_type if memory_type else list(self.collections.keys())
        for mem_type in types:
            collection = self._get_collection(mem_type)
            try:
                collection.delete(where={MetadataFields.USER_ID.value: {ChromaOperator.EQ.value: user_id}})
                return True
            except Exception as e:
                logger.error(f"Delete failed: {e}")
                return False
        return True

    def _apply_decay(self, mem: Dict) -> float:
        """apply decay(original similarity * e ** (-decay_factor * (now()-last_accessed)))"""
        last = mem[MemoryModelFields.METADATA.value].get(MetadataFields.LAST_ACCESS_AT.value)
        if not last:
            return mem[MemoryModelFields.SIMILARITY.value]
        try:
            days = (datetime.now() - datetime.fromisoformat(last)).days
        except:
            days = 0
        return mem[MemoryModelFields.SIMILARITY.value] * np.exp(-config.decay_factor * days)

    @retry_on_failure(max_retries=3, exceptions=(ChromaError,))
    def _update_last_accessed(self,memory_type: MemoryType, memory_id: str):
        collection = self._get_collection(memory_type)
        try:
            collection.update(
                ids=[memory_id],
                metadatas=[{MetadataFields.LAST_ACCESS_AT.value: datetime.now().isoformat()}]
            )
        except Exception as e:
            logger.warning(f"Failed to update access time for {memory_id}: {e}")

    def get_recent_interactions(self,user_id: str,limit: int = 5) -> List[Dict[str,Any]]:
        collection = self._get_collection(MemoryType.COMPLIANCE_RULE)

        #build query condition
        where = {
            ChromaOperator.AND.value:
                [
                    {MetadataFields.USER_ID.value: {ChromaOperator.EQ.value: user_id}},
                    {MetadataFields.STATUS.value: {ChromaOperator.EQ.value,MemoryStatus.ACTIVE.value}}
                ]
            }

        #execute query
        try:
            results = collection.get(
                where=where,
                limit = limit * 3,
                include=[ChromaResFields.METADATAS.value,ChromaResFields.DOCUMENTS.value]
            )
        except Exception as e:
            logger.error(f"Failed to retrieve interactions: {e}")
            return []

        #organize result data
        memories = []
        if results[ChromaResFields.IDS.value]:
            for i,doc in enumerate(results[ChromaResFields.DOCUMENTS.value]):
                memories.append({
                    MemoryModelFields.ID.value: results[ChromaResFields.IDS.value][i],
                    MemoryModelFields.CONTENT.value: doc,
                    MemoryModelFields.METADATA.value: results[ChromaResFields.METADATAS.value][i],
                    MemoryModelFields.SIMILARITY.value: 1.0,
                    MemoryModelFields.DECAYED_SIMILARITY.value: 1.0
                })

        #sort by time
        def get_timestamp(mem):
            ts = mem[MemoryModelFields.METADATA.value].get(MetadataFields.TIMESTAMP.value)
            if ts:
                try:
                    return datetime.fromisoformat(ts)
                except:
                    pass
            return datetime.min

        memories.sort(key=get_timestamp,reverse=True)
        return memories[:limit]

    def get_active_compliance_rules(self,limit: int = 10) -> List[Dict[str,Any]]:
        collection = self._get_collection(MemoryType.COMPLIANCE_RULE)
        where = {MetadataFields.STATUS.value: {ChromaOperator.EQ.value: MemoryStatus.ACTIVE.value}}

        try:
            results = collection.get(
                where=where,
                include=[ChromaResFields.METADATAS.value,ChromaResFields.DOCUMENTS.value]
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
            for i,doc in enumerate(results[ChromaResFields.DOCUMENTS.value]):
                rules.append({
                    MemoryModelFields.ID.value: results[ChromaResFields.IDS.value][i],
                    MemoryModelFields.CONTENT.value: doc,
                    MemoryModelFields.METADATA.value: results[ChromaResFields.METADATAS.value][i]
                })

        rules.sort(key = lambda r: severity_order.get(r[MemoryModelFields.METADATA.value].get(MetadataFields.SEVERITY.value,ComplianceSeverity.LOW.value),4))
        return rules[:limit]

    def get_all_user_profile_memories(self, user_id: str, status: str = "active") -> List[Dict[str, Any]]:
        collection = self._get_collection(MemoryType.USER_PROFILE)

        #build condition
        where = {ChromaOperator.AND.value:
                     [
                         {MetadataFields.USER_ID.value: {ChromaOperator.EQ.value: user_id}},
                         {MetadataFields.STATUS.value: {ChromaOperator.EQ.value,MemoryStatus.ACTIVE.value}}
                     ]
                 }

        #execute query
        try:
            results = collection.get(
                where=where,
                include=[ChromaResFields.METADATAS.value,ChromaResFields.DOCUMENTS.value]
            )
        except Exception as e:
            logger.error(f"Failed to get user profile memories: {e}")
            return []

        memories = []
        if results[ChromaResFields.IDS.value]:
            for i,doc in enumerate(results[ChromaResFields.DOCUMENTS.value]):
                memories.append({
                    MemoryModelFields.ID.value: results[ChromaResFields.IDS.value][i],
                    MemoryModelFields.CONTENT.value: doc,
                    MemoryModelFields.METADATA.value: results[ChromaResFields.METADATAS.value][i]
                })
        return memories


if __name__ == '__main__':
    store = ChromaMemoryStore("../test")
    #store.add_memory(user_id="hgh001",content="这是测试文件2",memory_type=MemoryType.USER_PROFILE,entity_key="test",metadata={"type": "user_profile","source": "test","confidence": 0.6},permanent=False)

    #result = store.search_memory("hgh001","测试",MemoryType.USER_PROFILE,2)
    #store.apply_forgetting(MemoryType.USER_PROFILE, "hgh001",2)
    result = store.get_memory_by_entity("hgh001", "test", MemoryStatus.FORGOTTEN.value)
    #store.delete_user_memories("hgh001")
    print(result)
