# author hgh
# version 1.0
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from config.constants import SearchStrategy, MemoryType
from query.query_model import Query


class BaseVectorStore(ABC):
    @abstractmethod
    def add(
            self,
            memory_type: MemoryType,
            ids: List[str],
            texts: List[str],
            models: List[Any],
            search_strategy: SearchStrategy = SearchStrategy.AUTO
    ) -> None:
        """
        batch add vector record

        Args:
            memory_type: the type of collection
            ids: list of ids
            texts: list of texts to be added
            models: list of data to be added
            search_strategy: search strategy
        """
        pass

    @abstractmethod
    def search(
            self,
            memory_type: MemoryType,
            query: str,
            where: Optional[Query],
            limit: int,
            search_strategy: SearchStrategy = SearchStrategy.AUTO,
    ) -> List[Dict[str, Any]]:
        """
        semantic retrieve

        Args:
            memory_type: the type of collection
            query: query text
            where: query Conditions
            limit: the number of return records
            search_strategy: the search strategy
        """
        pass

    @abstractmethod
    def get(
            self,
            memory_type: MemoryType,
            where: Optional[Query] = None,
            ids: Optional[List[str]] = None,
            limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        get records by conditions (without calculating vector similarity)

        Args:
            memory_type: the type of collection
            ids: list of ids
            where: query conditions
            limit: the number of return records
            include: the content included in the return result

        """
        pass

    @abstractmethod
    def update(
            self,
            memory_type: MemoryType,
            ids: List[str],
            metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        update metadatas of specified record

        Args:
            memory_type: the type of collection
            ids: unique ids of records
            metadatas: the data to be updated
        """
        pass

    @abstractmethod
    def delete(
            self,
            memory_type: MemoryType,
            where: Optional[Query] = None
    ) -> None:
        """
        delete records based on condition

        Args:
            memory_type: the type of collection
            where: conditions
        """
        pass
