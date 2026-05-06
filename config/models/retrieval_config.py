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

class EmbeddingCfg(BaseModel):
    model_name: str = "BAAI/bge-large-zh-v1.5"
    device: str = "cpu"
    normalize: bool = True
    max_seq_length: int = 512
    batch_size: int = 32

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
    embedding: EmbeddingCfg = Field(default_factory=EmbeddingCfg)

    # RAG internal context limitation (unrelated to the memory system window)
    rag_max_context_length: int = 3000