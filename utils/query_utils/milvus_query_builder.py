# author hgh
# version 1.0
from typing import Any

from utils.query_utils.query_builder import QueryBuilder
from utils.query_utils.query_model import Query


class MilvusQueryBuilder(QueryBuilder):
    def build(self, query: Query) -> Any:
        if not query or not query.conditions:
            return ""

        parts = []
        for cond in query.conditions:
            if cond.op.upper() == "IN":
                if isinstance(cond.value, list):
                    values = ", ".join(
                        f'"{v}"' if isinstance(v, str) else str(v) for v in cond.value
                    )
                    parts.append(f'{cond.field} in [{values}]')
            if cond.op.upper() == "ARRAY_CONTAINS":
                parts.append(f'ARRAY_CONTAINS({cond.field},"{cond.value}")')
            else:
                val = f'"{cond.value}"' if isinstance(cond.value, str) else str(cond.value)
                parts.append(f'{cond.field} {cond.op} {val}')

        connector = " AND " if query.logic.upper() == "AND" else " OR "
        return connector.join(parts)
