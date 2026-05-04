"""
RAG 检索与索引配置模型
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class IndexParam(BaseModel):
    metric_type: str = "IP"
    index_type: str = "IVF_FLAT"
    nlist: Optional[int] = 1024
    drop_ratio_build: Optional[float] = None

class MultiVectorCfg(BaseModel):
    enable: bool = False
    term_vector: bool = False
    summary_vector: bool = False
    faq_similar_vector: bool = False
    graph_embedding_dim: int = 0

class EmbeddingCfg(BaseModel):
    model_name: str = "BAAI/bge-large-zh-v1.5"
    device: str = "cpu"
    normalize: bool = True
    max_seq_length: int = 512
    batch_size: int = 32

class ChunkingRule(BaseModel):
    method: str = "recursive"
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    min_chunk_length: Optional[int] = None
    structure_delimiter: Optional[str] = None

class RetrievalConfig(BaseModel):
    milvus_uri: str = "http://localhost:19530"
    sqlite_db_path: str = "./checkpoints.db"
    # 检索策略
    retrieval_top_k: int = 5
    retrieval_fetch_k: int = 10
    retrieval_min_similarity: float = 0.3
    # 索引与多向量
    index_params: Dict[str, IndexParam] = Field(default_factory=dict)
    multi_vector: MultiVectorCfg = Field(default_factory=MultiVectorCfg)
    embedding: EmbeddingCfg = Field(default_factory=EmbeddingCfg)

    # 分块策略
    chunking: Dict[str, ChunkingRule] = Field(default_factory=dict)

    # RAG 内部上下文限制（与记忆系统的窗口无关）
    rag_max_context_length: int = 3000   # 一次 Prompt 中知识块总字符数上限