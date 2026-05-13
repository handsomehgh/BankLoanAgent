# author hgh
# version 1.0
from enum import Enum


class KnowledgeStatus(str, Enum):
    ACTIVE = "active"          # 有效，可被检索和引用
    ARCHIVED = "archived"      # 归档，长期保留但不参与日常检索
    DEPRECATED = "deprecated"  # 内容已过时（法规变更、产品下架），可检索但需标注
    DELETED = "deleted"        # 软删除，逻辑不可见但数据保留

class RewritingStrategy(str, Enum):
    HYDE = "hyde"
    STEP_BACK = "step_back"
    MULTI_QUERY = "multi_query"