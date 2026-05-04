# author hgh
# version 1.0
import logging
from typing import Dict, Any, List, Optional

import chromadb
from chromadb import Settings
from chromadb.errors import ChromaError

from config.global_constant.constants import VectorQueryFields, MemoryType
from infra.collections import CollectionNames
from modules.memory.memory_constant.fields import MemoryFields
from utils.query.chroma_query_builder import ChromaQueryBuilder
from utils.query.query_model import Query
from modules.memory.models.memory_mappers.mappers import MemoryToStorageMapper
from modules.memory.memory_vector_store.base_vector_store import BaseVectorStore
from utils.retry import retry_on_failure

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):

    def __init__(self, persist_dir: str):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self._collections: Dict[str, Any] = {}
        self._query_builder = ChromaQueryBuilder()
        logger.info(f"ChromaVectorStore initialized at {persist_dir}")

    def _get_collection(self, collection_name: str):
        if collection_name not in self._collections:
            self._collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collections[collection_name]

    @staticmethod
    def _flatten_chroma_result(results: Dict[str, Any]) -> List[Dict[str, Any]]:
        ids = results.get(VectorQueryFields.IDS, [[]])[0] if results.get(VectorQueryFields.IDS) else []
        documents = results.get(VectorQueryFields.DOCUMENTS, [[]])[0] if results.get(
            VectorQueryFields.DOCUMENTS) else []
        metadatas = results.get(VectorQueryFields.METADATAS, [[]])[0] if results.get(
            VectorQueryFields.METADATAS) else []
        distances = results.get(VectorQueryFields.DISTANCES, [[]])[0] if results.get(
            VectorQueryFields.DISTANCES) else []

        flat = []
        for i, doc_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 0.0
            item = {
                MemoryFields.ID: doc_id,
                MemoryFields.TEXT: documents[i] if i < len(documents) else None,
                VectorQueryFields.DISTANCE: distance,
                VectorQueryFields.SCORE: 1.0 - distance
            }

            # expand all metadata fields to the top level
            for key, value in meta.items():
                if key not in (MemoryFields.ID, MemoryFields.TEXT, VectorQueryFields.DISTANCE):
                    item[key] = value

            flat.append(item)
        return flat

    @retry_on_failure(max_retries=3, initial_delay=0.3, exceptions=(ChromaError,))
    def add(
            self,
            memory_type: MemoryType,
            ids: List[str],
            texts: List[str],
            models: List[Any]
    ) -> None:
        collection_name = CollectionNames.for_type(memory_type)
        collection = self._get_collection(collection_name)
        metadatas = [MemoryToStorageMapper.to_db_meta(m, target_db="chroma") for m in models]
        collection.add(ids=ids, documents=texts, metadatas=metadatas)

    @retry_on_failure(max_retries=3, initial_delay=0.2, exceptions=(ChromaError,))
    def search(
            self,
            memory_type: MemoryType,
            query: str,
            where: Optional[Query] = None,
            limit: int = 5
    ) -> List[Dict[str, Any]]:
        collection_name = CollectionNames.for_type(memory_type)
        collection = self._get_collection(collection_name)

        where = self._query_builder.build(where) if where else None

        raw_results = collection.query(
            query_texts=[query],
            where=where,
            n_results=limit,
            include=[
                VectorQueryFields.DOCUMENTS.value,
                VectorQueryFields.METADATAS.value,
                VectorQueryFields.DISTANCES.value,
            ]
        )

        return self._flatten_chroma_result(raw_results)

    @retry_on_failure(max_retries=3, initial_delay=0.2, exceptions=(ChromaError,))
    def get(
            self,
            memory_type: MemoryType,
            where: Optional[Query] = None,
            ids: Optional[List[str]] = None,
            limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        collection_name = CollectionNames.for_type(memory_type)
        collection = self._get_collection(collection_name)

        # build conditions
        kwargs = {}
        if where is not None:
            kwargs["where"] = self._query_builder.build(where)
        if ids is not None:
            kwargs["ids"] = ids
        if limit is not None:
            kwargs["limit"] = limit

        kwargs["include"] = [
            VectorQueryFields.DOCUMENTS.value,
            VectorQueryFields.METADATAS.value,
        ]

        # execute query
        raw_results = collection.get(**kwargs)

        adapted_results = {
            VectorQueryFields.IDS.value: [raw_results.get(VectorQueryFields.IDS.value, [])],
            VectorQueryFields.DOCUMENTS.value: [raw_results.get(VectorQueryFields.DOCUMENTS.value, [])],
            VectorQueryFields.METADATAS.value: [raw_results.get(VectorQueryFields.METADATAS.value, [])],
            VectorQueryFields.DISTANCES.value: [[0.0] * len(raw_results.get(VectorQueryFields.IDS.value, []))],
        }

        return self._flatten_chroma_result(adapted_results)

    @retry_on_failure(max_retries=3, initial_delay=0.2, exceptions=(ChromaError,))
    def update(
            self,
            memory_type: MemoryType,
            ids: List[str],
            metadatas: List[Dict[str, Any]]
    ) -> None:
        collection_name = CollectionNames.for_type(memory_type)
        collection = self._get_collection(collection_name)

        try:
            existing = collection.get(
                ids=ids,
                include=[VectorQueryFields.METADATAS.value],
            )
            existing_metas = existing.get(VectorQueryFields.METADATAS, [])
        except Exception as e:
            logger.warning(f"Failed to fetch existing metadata for update: {e}")
            existing_metas = []

        existing_map = {}
        for i, mem_id in enumerate(ids):
            if i < len(existing_metas):
                existing_map[mem_id] = existing_metas[i]
            else:
                existing_map[mem_id] = {}

        for i, mem_id in enumerate(ids):
            merged = existing_map[mem_id].copy()
            merged.update(metadatas[i])
            metadatas[i] = merged

        collection.update(ids=ids, metadatas=metadatas)

    @retry_on_failure(max_retries=3, initial_delay=0.2, exceptions=(ChromaError,))
    def delete(
            self,
            memory_type: MemoryType,
            where: Optional[Query]
    ) -> None:
        collection_name = CollectionNames.for_type(memory_type)
        collection = self._get_collection(collection_name)
        chroma_where = self._query_builder.build(where) if where else None
        collection.delete(where=chroma_where)


if __name__ == '__main__':
    store = ChromaVectorStore("./test")
    store.add(MemoryType.USER_PROFILE)
