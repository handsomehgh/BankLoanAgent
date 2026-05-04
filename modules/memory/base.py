# author hgh
# version 1.0
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

from config.global_constant.constants import MemoryType


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
            self,
            query: str,
            user_id: str,
            top_k: int = 5,
            memory_types:  Optional[List[MemoryType]] = None,
            **kwargs
    ) -> Dict[str,List[Dict[str,Any]]]:
        """
        retrieve multi-source memory

        Args:
            query: query text
            user_id: unique id of user
            top_k: the maximum number returned for each type of memory
            memory_types: list of memory types to be retrieved(such as "user_profile","business_knowledge")
            **kwargs: additional arguments

        Returns:
            a dictionary with keys representing memory types and values as lists of retrieval results
            such as:{"user_profile": [...], "business_knowledge": [...]}
        """
        pass

