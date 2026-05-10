"""
RAG 检索与索引配置模型
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class IndexParam(BaseModel):
    metric_type: str = "COSINE"
    index_type: str = "HNSW"
    params: Optional[Dict[str, Any]] = None


class SummaryConfig(BaseModel):
    enable_source_filter: bool = True
    enabled_sources: List[str] = Field(default_factory=list)
    min_chunk_length: int = 200
    temperature: float = 0.1
    max_output_tokens: int = 80
    fallback_to_original: bool = True


class FaqSimilarConfig(BaseModel):
    num_variants: int = 3
    temperature: float = 0.3
    max_output_tokens: int = 150
    fallback_to_original: bool = True


class MultiVectorCfg(BaseModel):
    enable: bool = False
    term_vector: bool = False
    summary_vector: bool = False
    faq_similar_vector: bool = False
    graph_embedding_dim: int = 0
    summary_config: SummaryConfig = Field(default_factory=SummaryConfig)
    faq_similar_config: Optional[FaqSimilarConfig] = Field(default_factory=FaqSimilarConfig)


class SearchConfig(BaseModel):
    dense_top_k: int = 20
    text_match_top_k: int = 20
    sparse_top_k: int = 20
    term_top_k: int = 10
    dense_ef: int = 64
    term_ef: int = 64
    dense_metric_type: str = "COSINE"
    bm25_metric_type: str = "BM25"


class RewriterConfig(BaseModel):
    enable_dynamic: bool = True
    override_strategy: Optional[str] = None
    num_variants: int = 3
    temperature: float = 0.3
    max_tokens: int = 200
    fallback_to_original: bool = True


class FilterConfig(BaseModel):
    enabled: bool = True
    confidence_threshold: float = 0.3
    temperature: float = 0.0


class FusionConfig(BaseModel):
    k: int = 60


class RerankerConfig(BaseModel):
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_k: int = 5
    batch_size: int = 16


class CompressorConfig(BaseModel):
    enabled: bool = True
    max_context_tokens: int = 1500
    sentences_to_keep: int = 3
    fallback_to_full: bool = True
    model_name: str = "BAAI/bge-reranker-v2-m3"

class WeakSignalItem(BaseModel):
    words: List[str] = Field(...)
    score: int = Field(...)

class RuleBasedRouterConfig(BaseModel):
    strong_keywords: List[str] = Field(
        default_factory=list,
        description="strong business keywords,triggering search if any one is hit"
    )
    weak_signals: List[WeakSignalItem] = Field(
        default_factory=list,
        description="weak business keywords,triggering search if any one is hit"
    )
    stop_patterns: List[str] = Field(
        default_factory=list,
        description="a list of regular expressions for non-business sentence patterns; if any one is matched, the search is skipped."
    )
    ambiguous_threshold: int = Field(
        default=6,
        description="when the query length is less than or equal to this value and only contains weak keywords, retrieval is not triggered"
    )
    weak_signal_threshold: int = Field(
        default=4,
        description="weak word score"
    )

class RetrievalRoutingConfig(BaseModel):
    """retrieve the overall routing configuration,supporting multiple strategy"""
    enabled: bool = Field(default=True, description="whether to enable routing,if disabled,all queries will be covered")
    strategy: str = Field(default="rule_based", description="route strategy: rule_based, ml_based")
    rule_based: RuleBasedRouterConfig = Field(default_factory=RuleBasedRouterConfig)


class RetrievalConfig(BaseModel):
    milvus_uri: str = "http://localhost:19530"
    sqlite_db_path: str = "./checkpoints.db"
    insert_batch_size: int = 50

    # search strategy
    retrieval_top_k: int = 5
    retrieval_fetch_k: int = 10
    retrieval_min_similarity: float = 0.3

    # index and multivector
    index_params: Dict[str, IndexParam] = Field(default_factory=dict)
    multi_vector: MultiVectorCfg = Field(default_factory=MultiVectorCfg)

    # RAG internal context limitation (unrelated to the memory system window)
    rag_max_context_length: int = 3000

    # search param
    search: SearchConfig = Field(default_factory=SearchConfig)

    # query rewriter
    rewriter: RewriterConfig = Field(default_factory=RewriterConfig)

    # structure query
    filter: FilterConfig = Field(default_factory=FilterConfig)

    # rrf fusion
    fusion: FusionConfig = Field(default_factory=FusionConfig)

    # reranker
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)

    # context compressor
    compressor: CompressorConfig = Field(default_factory=CompressorConfig)

    # retrieve routing config
    retrieval_routing: RetrievalRoutingConfig = Field(default_factory=RetrievalRoutingConfig)
