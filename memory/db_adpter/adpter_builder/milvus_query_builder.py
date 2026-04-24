# author hgh
# version 1.0
from typing import Any

from memory.db_adpter.adpter_model.query_model import Query
from memory.db_adpter.query_builder import QueryBuilder


class MilvusQueryBuilder(QueryBuilder):
    def build(self,query: Query) -> Any:
        pass