# author hgh
# version 1.0
from abc import ABC, abstractmethod
from typing import Any

from memory.db_adpter.adpter_model.query_model import Query


class QueryBuilder(ABC):
    @abstractmethod
    def build(self,query: Query) -> Any:
        """convert Query to database query format(chroma,milvus)"""
        pass
