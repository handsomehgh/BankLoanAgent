# author hgh
# version 1.0
from dataclasses import dataclass, field
from typing import Literal, Any, List

from sqlalchemy.orm import defer


@dataclass
class Condition:
    """single filter condition"""
    field: str
    op: Literal["==", "!=", ">=", "<=", "in"] = "=="
    value: Any = None

@dataclass
class Query:
    """combination of multiple conditions,supporting AND/OR logic"""
    conditions: List[Condition] = field(default_factory=list)
    logic: Literal["AND","OR"] = "AND"


