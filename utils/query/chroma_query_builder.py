# author hgh
# version 1.0
from typing import Dict, Any

from utils.query.query_model import Query
from utils.query.query_builder import QueryBuilder


class ChromaQueryBuilder(QueryBuilder):
    """convert general query to chroma query"""
    def build(self,query:Query) -> Dict[str,Any]:
        if not query.conditions:
            return {}

        cond_dicts = []
        for cond in query.conditions:
            op_map = {
                "==": "$eq",
                "!=": "$ne",
                ">=": "$gte",
                "<=": "$lte",
                "in": "$in",
            }
            chroma_op = op_map.get(cond.op,"$eq")
            cond_dicts.append({cond.field: {chroma_op: cond.value}})

        if len(cond_dicts) == 1:
            return cond_dicts[0]

        if query.logic.upper() == "AND":
            return {"$and": cond_dicts}
        else:
            return {"$or": cond_dicts}