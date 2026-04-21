# author hgh
# version 1.0
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseVectorStore(ABC):
    @abstractmethod
    def add(
            self,
            collection_name: str,
            texts: List[str],
            metadatas: List[Dict[str, Any]],
            ids: List[str]
    ) -> None:
        """
        batch add vector record

        Args:
            collection_name: the name of collection
            texts: list of texts to be added
            metadatas: list of metadata to be added
            ids: list of ids
        """
        pass

    @abstractmethod
    def search(
            self,
            collection_name: str,
            query: str,
            where: Dict[str, Any],
            limit: int,
            include: List[str]
    ) -> Dict[str, Any]:
        """
        semantic retrieve

        Args:
            collection_name: the name of collection
            query: query text
            where: query Conditions
            limit: the number of return records
            include: the content included in the return result
        """
        pass

    @abstractmethod
    def get(
            self,
            collection_name: str,
            where: Optional[Dict[str, Any]] = None,
            ids: Optional[List[str]] = None,
            limit: Optional[int] = None,
            include: List[str] = None
    ) -> Dict[str, Any]:
        """
        get records by conditions (without calculating vector similarity)

        Args:
            collection_name: the name of collection
            ids: list of ids
            where: query conditions
            limit: the number of return records
            include: the content included in the return result

        """
        pass

    @abstractmethod
    def update(
            self,
            collection_name: str,
            ids: List[str],
            metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        update metadatas of specified record

        Args:
            collection_name: the name of collection
            ids: unique ids of records
            metadatas: the metadatas to be updated
        """
        pass

    @abstractmethod
    def delete(
            self,
            collection_name: str,
            where: List[Dict[str, Any]]
    ) -> None:
        """
        delete records based on condition

        Args:
            collection_name: the name of collection
            where: conditions
        """
        pass
