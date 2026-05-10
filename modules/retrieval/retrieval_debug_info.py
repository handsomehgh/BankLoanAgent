# author hgh
# version 1.0
"""
retrieval debug info model
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class DocSummary:
    """检索结果文档摘要"""
    id: str
    text: str
    score: float


@dataclass
class RouteDebugInfo:
    """单路检索调试信息"""
    route_name: str = ""                # dense / sparse / term
    top_k: int = 0
    hits_count: int = 0                 # 实际命中数
    first_hit_score: Optional[float] = None   # 最高分
    elapsed_ms: float = 0.0             # 耗时(毫秒)
    top_docs: List[DocSummary] = field(default_factory=list)  # 前3条文档摘要
    is_error: bool = False              # 是否发生异常


@dataclass
class RetrievalDebugInfo:
    """检索全链路调试信息"""
    query_original: str = ""
    rewritten_queries: List[str] = field(default_factory=list)
    filter_expr: Optional[str] = None
    cache_hit: bool = False               # 是否命中缓存
    dense: Optional[RouteDebugInfo] = None
    sparse: Optional[RouteDebugInfo] = None
    term: Optional[RouteDebugInfo] = None
    fused_count: int = 0                # RRF融合后去重数量
    fused_top_scores: List[float] = field(default_factory=list)
    rerank_candidates: int = 0
    rerank_top_scores: List[float] = field(default_factory=list)
    compressed_info: List[Dict[str, Any]] = field(default_factory=list)  # 压缩详情
    total_elapsed_ms: float = 0.0
