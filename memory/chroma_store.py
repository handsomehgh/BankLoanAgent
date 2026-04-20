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
from models.constant.constants import MetadataFields, MemoryType, MemorySource, MemoryStatus, MemoryModelFields, \
    ChromaOperator, ChromaResFields
from utils.retry import retry_on_failure

logger = logging.Logger(__name__)


class ChromaMemoryStore(BaseMemoryStore):
    def __init__(self, persist_dir: str, collection_name: str = MemoryType.USER_PROFILE.value):
        """
        initial chroma client and collection 
        
        Args:
            persist_dir: data persistence folder
            collection_name: the name of collection
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        # dead letter queue
        self.dlq_path = Path(persist_dir) / "memory_dlq.jsonl"
        self.dlq_path.parent.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        try:
            self._get_or_create_collection()
        except Exception as e:
            logger.error(f"Failed to initialize Chroma collection: {e}")
            raise
        logger.info(f"chroma initial at {persist_dir}")

    @retry_on_failure(max_retries=3, initial_delay=0.3, exceptions=(ChromaError,))
    def _get_or_create_collection(self):
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=OpenAIEmbeddingFunction(api_key=config.alibaba_api_key,
                                                       model_name=config.qwen_emb_name,
                                                       api_base=config.alibaba_base_url,
                                                       dimensions=1024)
        )

    def _write_to_dlq(self, user_id: str, content: str, meta: Dict, permanent: bool, entity_key: Optional[str] = None):
        entry = {
            MetadataFields.USER_ID.value: user_id,
            MemoryModelFields.CONTENT.value: content,
            MetadataFields.ENTITY_KEY.value: entity_key,
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
            entity_key: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            permanent: bool = False
    ) -> str:
        now = datetime.now().isoformat()
        meta = metadata.copy() if metadata else {}
        meta = {
            MetadataFields.USER_ID.value: user_id,
            MetadataFields.TYPE.value: metadata.get(MetadataFields.TYPE.value, MemoryType.USER_PROFILE.value),
            MetadataFields.SOURCE.value: metadata.get(MetadataFields.SOURCE.value, MemorySource.CHAT_EXTRACTION.value),
            MetadataFields.CONFIDENCE.value: metadata.get(MetadataFields.CONFIDENCE.value, 0.8),
            MetadataFields.STATUS.value: metadata.get(MetadataFields.STATUS.value, MemoryStatus.ACTIVE.value),
            MetadataFields.PERMANENT.value: permanent,
            MetadataFields.CREATE_AT.value: now,
            MetadataFields.LAST_ACCESS_AT.value: now,
        }
        if entity_key:
            meta[MetadataFields.ENTITY_KEY.value] = entity_key

        existing = []
        if entity_key and not permanent:
            try:
                existing = self.get_memory_by_entity(user_id, entity_key, MemoryStatus.ACTIVE.value)
            except Exception as e:
                logger.warning(f"Conflict check failed,proceeding: {e}")

        # update the memory status to superseded
        for old in existing:
            if float(meta[MetadataFields.CONFIDENCE.value]) > float(
                    old[MemoryModelFields.METADATA.value].get(MetadataFields.CONFIDENCE.value, 0)) + 0.1:
                try:
                    self.update_memory_status(old[MemoryModelFields.ID.value], MemoryStatus.SUPERSEDED.value,
                                              {MemoryStatus.SUPERSEDED.value: None})
                except Exception as e:
                    logger.warning(f"Failed to superseded {old[MemoryModelFields.ID.value]: {e}}")

        # add memory
        memory_id = str(uuid.uuid4())
        try:
            self.collection.add(ids=[memory_id], documents=[content], metadatas=[meta])
        except Exception as e:
            # write to dlq
            logger.error(f"Write failed for user {user_id}: {e}")
            self._write_to_dlq(user_id, content, meta, permanent, entity_key)
            raise MemoryWriteFailedError(f"Memory write failed, queued: {e}") from e

        # update the superseded_by of the memory to the new memory ID
        if entity_key:
            for old in existing:
                if old[MemoryModelFields.METADATA.value].get(
                        MetadataFields.STATUS.value) == MemoryStatus.SUPERSEDED.value:
                    try:
                        self.update_memory_status(old[MemoryModelFields.ID.value], MemoryStatus.SUPERSEDED.value,
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
            limit: int = 3,
            memory_type: Optional[str] = None,
            min_confidence: Optional[float] = None,
            apply_decay: bool = True
    ) -> List[Dict[str, Any]]:
        """search memory by query"""
        # build query conditions
        where = {
            ChromaOperator.AND.value: [
                {MetadataFields.USER_ID.value: {ChromaOperator.EQ.value: user_id}},
                {MetadataFields.STATUS.value: {ChromaOperator.EQ.value: MemoryStatus.ACTIVE.value}}
            ]
        }
        if memory_type:
            where[ChromaOperator.AND.value].append({MetadataFields.TYPE.value: {ChromaOperator.EQ.value: memory_type}})
        if min_confidence:
            where[ChromaOperator.AND.value].append(
                {MetadataFields.CONFIDENCE.value: {ChromaOperator.GTE.value: min_confidence}})

        # the number of result
        fetch_limit = limit * 2 if apply_decay else limit

        # execute query
        try:
            results = self.collection.query(
                query_texts=query,
                where=where,
                n_results=fetch_limit,
                include=[ChromaResFields.DOCUMENTS.value, ChromaResFields.METADATAS.value,
                         ChromaResFields.DISTANCES.value]
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
            memories.sort(key=lambda x: x[MemoryModelFields.DECAYED_SIMILARITY], reverse=True)

        # update last accessed
        for m in memories[:limit]:
            self._update_last_accessed(m[MemoryModelFields.ID.value])
        return memories[:limit]

    @retry_on_failure(max_retries=3, exceptions=(ChromaError,))
    def get_memory_by_entity(
            self,
            user_id: str,
            entity_key: str,
            status: str = MemoryStatus.ACTIVE.value
    ) -> List[Dict[str, Any]]:
        """get memory by entity key"""
        where_filter = {
            ChromaOperator.AND.value: [
                {MetadataFields.USER_ID.value: {ChromaOperator.EQ.value: user_id}},
                {MetadataFields.ENTITY_KEY.value: {ChromaOperator.EQ.value: entity_key}},
                {MetadataFields.STATUS.value: {ChromaOperator.EQ.value: status}}
            ]
        }
        try:
            results = self.collection.get(
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
            new_status: str,
            metadata_updates: Optional[Dict[str, Any]] = None) -> bool:
        """update memory status"""
        try:
            current = self.collection.get(ids=[memory_id], include=[ChromaResFields.METADATAS.value])
            if not current[ChromaResFields.METADATAS.value]:
                return False
            new_meta = current[ChromaResFields.METADATAS.value][0].copy()
            new_meta[MetadataFields.STATUS.value] = new_status
            if metadata_updates:
                new_meta.update(metadata_updates)
            self.collection.update(ids=[memory_id], metadatas=[new_meta])
            return True
        except Exception as e:
            raise MemoryUpdateError(f"Update failed: {e}") from e

    def apply_forgetting(
            self,
            user_id: Optional[str] = None,
            threshold: Optional[float] = None
    ) -> int:
        """apply forgetting"""
        threshold = threshold if threshold else config.decay_threshold
        where = {ChromaOperator.AND.value: [
            {MetadataFields.STATUS.value: {ChromaOperator.EQ.value: MemoryStatus.ACTIVE.value}}]}
        if user_id:
            where[ChromaOperator.AND.value].append({MetadataFields.USER_ID.value: {ChromaOperator.EQ.value: user_id}})

        # query the memories that need to be forgotten
        try:
            res = self.collection.get(
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
                    self.update_memory_status(mem_id, MemoryStatus.FORGOTTEN.value)
                    count += 1
                except:
                    pass
        logger.info(f"Forgotten {count} memories")
        return count

    def delete_user_memories(self, user_id: str) -> bool:
        """delete user memory"""
        try:
            self.collection.delete(where={MetadataFields.USER_ID.value: {ChromaOperator.EQ.value: user_id}})
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

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
    def _update_last_accessed(self, memory_id: str):
        try:
            self.collection.update(
                ids=[memory_id],
                metadatas=[{MetadataFields.LAST_ACCESS_AT.value: datetime.now().isoformat()}]
            )
        except Exception as e:
            logger.warning(f"Failed to update access time for {memory_id}: {e}")


if __name__ == '__main__':
    store = ChromaMemoryStore("./test", "test")
    # store.add_memory("hgh001","这是测试文件替换文件","preference",{"type": "user_profile","source": "test","confidence": "0.5"},False)
    # result = store.search_memory("hgh001","测试",2)
    store.apply_forgetting("hgh001", "1.2")
    result = store.get_memory_by_entity("hgh001", "preference", "forgotten")
    # store.delete_user_memories("hgh001")
    print(result)
