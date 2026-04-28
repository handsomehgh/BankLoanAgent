# author hgh
# version 1.0
from typing import Any

from query.query_model import Query
from query.query_builder import QueryBuilder


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
            else:
                val = f'"{cond.value}"' if isinstance(cond.value, str) else str(cond.value)
                parts.append(f'{cond.field} {cond.op} {val}')

        connector = " AND " if query.logic.upper() == "AND" else " OR "
        return connector.join(parts)
