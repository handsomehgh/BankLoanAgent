# author hgh
# version 2.0
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

from config.global_constant.constants import KnowledgeFileSourceType
from modules.retrieval.knowledge_constant import KnowledgeStatus


class BusinessKnowledge(BaseModel):
    """business knowledge model（对齐 Milvus business_knowledge schema v2，作为检索 output_fields 来源）"""
    id: str = Field(..., description="deterministic chunk id: {source_file}:{doc_seq}:{chunk_index}")
    text: str = Field(..., description="chunk content")
    status: KnowledgeStatus = Field(default=KnowledgeStatus.ACTIVE, description="text status")
    topics: List[str] = Field(default=list, description="topic list")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="confidence score")
    source_type: KnowledgeFileSourceType = Field(..., description="source type")
    source_file: str = Field(..., description="source file name")
    product_type: Optional[str] = Field(default=None, description="product type")
    parent_doc_id: Optional[str] = Field(default=None, description="stable parent doc id: {source_file}:{doc_seq}")
    chunk_index: Optional[int] = Field(default=None, description="chunk index within parent doc")
    question: Optional[str] = Field(default=None, description="original FAQ question, faq source only")
    regulation_names: List[str] = Field(default=list, description="regulation names")
    doc_version: str = Field(default="", description="source doc version from md header, e.g. v4.0")
    ingest_batch_id: str = Field(default="", description="ingest batch id for cleanup and rollback")
    created_at: int = Field(default=0, description="create time, unix seconds")
    updated_at: int = Field(default=0, description="update time, unix seconds")
    extra: Dict[str, Any] = Field(default_factory=dict, description="extra data")
