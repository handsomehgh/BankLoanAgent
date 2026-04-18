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

from config import config
from exception import MemoryWriteFailedError, MemoryRetrievalError, MemoryUpdateError
from memory.base import BaseMemoryStore
from utils.retry import retry_on_failure

logger = logging.Logger(__name__)


class ChromaMemoryStore(BaseMemoryStore):
    def __init__(self, persist_dir: str, collection_name: str = "user_profile"):
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

        self._get_or_create_collection()
        logger.info(f"chroma initial at {persist_dir}")

    @retry_on_failure(max_retries=3, initial_delay=0.3, exceptions=(ChromaError,))
    def _get_or_create_collection(self):
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "consine"}
        )

    def _write_to_dlq(self, user_id: str, content: str, meta: Dict, permanent: bool, entity_key: Optional[str] = None):
        entry = {
            "user_id": user_id,
            "content": content,
            "entity_key": entity_key,
            "metadata": meta,
            "permanent": permanent,
            "timestamp": datetime.now().isoformat(),
            "retry_count": 0
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
            "user_id": user_id,
            "type": metadata.get("type", "user_profile"),
            "source": metadata.get("source", "extracted"),
            "confidence": metadata.get("confidence", 0.8),
            "status": metadata.get("status", "active"),
            "permanent": permanent,
            "created_at": now,
            "lass_access_at": now,
        }
        if entity_key:
            meta["entity_key"] = entity_key

        existing = []
        if entity_key and not permanent:
            try:
                existing = self.get_memory_by_entity(user_id, entity_key, "active")
            except Exception as e:
                logger.warning(f"Conflict check failed,proceeding: {e}")

        # update the memory status to superseded
        for old in existing:
            if meta["confidence"] > old["metadata"].get("confidence", 0) + 0.1:
                try:
                    self.update_memory_status(user_id, "superseded", {"superseded": None})
                except Exception as e:
                    logger.warning(f"Failed to superseded {old['id']: {e}}")

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
                if old["metadata"].get("status") == "superseded":
                    try:
                        self.update_memory_status(old["id"], "superseded", {"superseded_by": memory_id})
                    except Exception as e:
                        logger.error(f"Failed to update superseded_by for {old['id']}: {e}")

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
            "and$": [
                {"user_id": {"$eq": user_id}},
                {"status": {"$eq": "active"}}
            ]
        }
        if memory_type:
            where["and$"].append({"type": {"$eq": memory_type}})
        if min_confidence:
            where["$and"].append({"confidence": {"$gte": min_confidence}})

        # the number of result
        fetch_limit = limit * 2 if apply_decay else limit

        # execute query
        try:
            results = self.collection.query(
                query_texts=query,
                where=where,
                n_results=fetch_limit,
                include=["documents", "metadatas", "distances"]
            )
        except Exception as e:
            raise MemoryRetrievalError(f"Search failed: {e}") from e

        # organize output and apply time decay
        memories = []
        if results["ids"][0]:
            for i, doc in enumerate(results["documents"][0]):
                mem = {
                    "id": results["ids"][0][i],
                    "content": doc,
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1 - results["distances"][0][i],
                }

                if apply_decay and not mem["metadata"].get["permanent"]:
                    mem["decayed_similarity"] = self._apply_decay(mem)
                else:
                    mem["decayed_similarity"] = mem["similarity"]
                memories.append(mem)

        # reordering the memories after time decay
        if apply_decay:
            memories.sort(key=lambda x: x["decayed_similarity"], reverse=True)

        # update last accessed
        for m in memories[:limit]:
            self._update_last_accessed(m["id"])
        return memories[:limit]

    @retry_on_failure(max_retries=3, exceptions=(ChromaError,))
    def get_memory_by_entity(
            self,
            user_id: str,
            entity_key: str,
            status: str = "active"
    ) -> List[Dict[str, Any]]:
        """get memory by entity key"""
        where_filter = {
            "$and": [
                {"user_id": {"$eq": user_id}},
                {"entity_key": {"$eq": entity_key}},
                {"status": {"$eq": status}}
            ]
        }
        try:
            results = self.collection.get(
                where=where_filter,
                include=["documents", "metadatas"]
            )
        except Exception as e:
            logger.error(f"get by entity failed:{e}")
            return []

        memories = []
        if results["ids"]:
            for i, doc in enumerate(results["documents"]):
                memories.append({
                    "id": results["ids"][i],
                    "content": doc,
                    "metadata": results["metadata"][i]
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
            current = self.collection.get(ids=[memory_id], include=["metadatas"])
            if not current["metadatas"]:
                return False
            new_meta = current["metadatas"][0].copy()
            new_meta["status"] = new_status
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
        threshold = threshold or config.decay_threshold
        where = {"status": {"$eq": "active"}}
        if user_id:
            where["user_id"] = {"$eq": user_id}

        # query the memories that need to be forgotten
        try:
            res = self.collection.query(
                where=where,
                include=["metadatas"]
            )
        except Exception as e:
            logger.error(f"Forgetting scan failed: {e}")
            return 0

        count = 0
        for i, mem_id in enumerate(res["ids"]):
            meta = res["metadatas"][i]
            if meta.get("permanent"):
                continue
            mem = {"metadata": meta, "similarity": 1.0}
            if self._apply_decay(mem) < threshold:
                try:
                    self.update_memory_status(mem["id"], "forgotten")
                    count += 1
                except:
                    pass
        logger.info(f"Forgotten {count} memories")
        return count

    def delete_user_memories(self, user_id: str) -> bool:
        """delete user memory"""
        try:
            self.collection.delete(where={"user_id": {"$eq": user_id}})
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def _apply_decay(self, mem: Dict) -> float:
        """apply decay(original similarity * e ** (-decay_factor * (now()-last_accessed)))"""
        last = mem["metadate"].get("last_accessed_at")
        if not last:
            return mem["similarity"]
        try:
            days = (datetime.now() - datetime.fromisoformat(last)).days
        except:
            days = 0
        return mem["similarity"] * np.exp(-config.decay_factor * days)

    @retry_on_failure(max_retries=3, exceptions=(ChromaError,))
    def _update_last_accessed(self, memory_id: str):
        try:
            self.collection.update(
                ids=[memory_id],
                metadatas=[{"last_accessed_at": datetime.now().isoformat()}]
            )
        except Exception as e:
            logger.warning(f"Failed to update access time for {memory_id}: {e}")


if __name__ == '__main__':
    now = datetime.now().isoformat()
    print(now)
    print(type(now))
