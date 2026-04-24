# author hgh
# version 1.0
import logging
from typing import Dict, Any, List, Optional

import chromadb
from chromadb import Settings
from chromadb.errors import ChromaError

from memory.constant.constants import ChromaResFields
from memory.memory_vector_store.vector_store import BaseVectorStore
from utils.retry import retry_on_failure

logger = logging.getLogger(__name__)

class ChromaVectorStore(BaseVectorStore):

    def __init__(self,persist_dir: str):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        self._collections: Dict[str,Any] = {}
        logger.info(f"ChromaVectorStore initialized at {persist_dir}")

    def _get_collection(self,collection_name: str):
        if collection_name not in self._collections:
            self._collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collections[collection_name]

    @retry_on_failure(max_retries=3, initial_delay=0.3, exceptions=(ChromaError,))
    def add(
            self,
            collection_name: str,
            texts: List[str],
            metadatas: List[Dict[str, Any]],
            ids: List[str]
    ) -> None:
        collection = self._get_collection(collection_name)
        collection.add(ids=ids, documents=texts, metadatas=metadatas)

    @retry_on_failure(max_retries=3,initial_delay=0.2,exceptions=(ChromaError,))
    def search(
            self,
            collection_name: str,
            query: str,
            where: Optional[Any] = None,
            limit: int = 5,
            include: List[str] = None,
    ) -> Dict[str, Any]:
        collection = self._get_collection(collection_name)
        return collection.query(query=[query], where=where, limit=limit, include=include)

    @retry_on_failure(max_retries=3,initial_delay=0.2,exceptions=(ChromaError,))
    def get(
            self,
            collection_name: str,
            where: Optional[Any] = None,
            ids: Optional[List[str]] = None,
            limit: Optional[int] = None,
            include: List[str] = None
    ) -> Dict[str, Any]:
        collection = self._get_collection(collection_name)
        kwargs = {}
        if where is not None:
            kwargs["where"] = where
        if ids is not None:
            kwargs["ids"] = ids
        if limit is not None:
            kwargs["limit"] = limit
        if include is not None:
            kwargs["include"] = include
        else:
            kwargs["include"] = [ChromaResFields.DOCUMENTS.value, ChromaResFields.METADATAS.value]
        return collection.get(**kwargs)

    @retry_on_failure(max_retries=3,initial_delay=0.2,exceptions=(ChromaError,))
    def update(
            self,
            collection_name: str,
            ids: List[str],
            metadatas: List[Dict[str, Any]]
    ) -> None:
        collection = self._get_collection(collection_name)
        collection.update(ids=ids, metadatas=metadatas)

    @retry_on_failure(max_retries=3,initial_delay=0.2,exceptions=(ChromaError,))
    def delete(
            self,
            collection_name: str,
            where: Optional[Any]
    ) -> None:
        collection = self._get_collection(collection_name)
        collection.delete(where=where)

