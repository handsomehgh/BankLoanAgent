# author hgh
# version 1.0
from typing import List, Any, Dict, Optional

from memory.memory_vector_store.vector_store import BaseVectorStore


class MilvusVectorStore(BaseVectorStore):
    def __init__(self, uri: str):
        pass

    def add(self, collection_name: str, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        pass

    def search(self, collection_name: str, query: str, where: Optional[Any], limit: int,
               include: Optional[List[str]] = None) -> Dict[str, Any]:
        pass

    def get(self, collection_name: str, where: Optional[Any] = None, ids: Optional[List[str]] = None,
            limit: Optional[int] = None, include: List[str] = None) -> Dict[str, Any]:
        pass

    def update(self, collection_name: str, ids: List[str], metadatas: List[Dict[str, Any]]) -> None:
        pass

    def delete(self, collection_name: str, where: List[Any]) -> None:
        pass