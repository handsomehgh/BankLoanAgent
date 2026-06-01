# author hgh
# version 1.0
from datetime import datetime
from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field

from config.global_constant.constants import KnowledgeFileSourceType
from modules.retrieval.knowledge_constant import KnowledgeStatus


class BusinessKnowledge(BaseModel):
    """business knowledge model"""
    id: str = Field(..., description="unique chunk id")
    text: str = Field(..., description="chunk content")
    status: KnowledgeStatus = Field(default=KnowledgeStatus.ACTIVE, description="text status")
    topics: List[str] = Field(default=list, description="topic list")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="confidence score")
    entity_id: Optional[str] = Field(default=None, description="related graph entity id")
    entity_type: Optional[str] = Field(default=None, description="related graph entity id")
    source_type: KnowledgeFileSourceType = Field(..., description="source type")
    source_file: str = Field(..., description="source file path")
    product_type: Optional[str] = Field(default=None, description="product type")
    parent_doc_id: Optional[str] = Field(default=None, description="parent doc id")
    chunk_index: Optional[int] = Field(default=None, description="chunk index")
    relation_predicate: Optional[str] = Field(default=None, description="relation predicate")
    regulation_names: List[str] = Field(default=list, description="regulation names")
    created_at: datetime = Field(default_factory=datetime.now, description="chunk creation time")
    updated_at: datetime = Field(default_factory=datetime.now, description="chunk update time")
    extra: Dict[str, Any] = Field(default_factory=dict, description="extra data")
