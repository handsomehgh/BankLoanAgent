# author hgh
# version 1.0
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from config.global_constant.constants import KnowledgeFileSourceType


class PreprocessingConfig(BaseModel):
    min_content_length: int = 10
    enable_advanced_cleaning: bool = True

class FileSourceItem(BaseModel):
    loader_type: str
    source_type: KnowledgeFileSourceType

class MetadataExtractionConfig(BaseModel):
    product_keywords: Dict[str, List[str]] = Field(default_factory=dict)
    topic_keywords: Dict[str, List[str]] = Field(default_factory=dict)
    default_confidence_by_source: Dict[str, float] = Field(default_factory=dict)
    fallback_confidence: float = 0.7

class ChunkingRule(BaseModel):
    method: str
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    min_chunk_length: Optional[int] = None

class ChunkingConfig(BaseModel):
    min_chunk_length: int = 30
    separators: List[str] = Field(default_factory=lambda: ["\n\n", "\n", "。", "？", "！", "，", "；", "：", " ", ""])
    strategies: Dict[str, ChunkingRule] = Field(default_factory=dict)
    default: ChunkingRule = Field(default_factory=lambda: ChunkingRule(method="recursive", chunk_size=600, chunk_overlap=60, min_chunk_length=20))

class FileProcessConfig(BaseModel):
    data_dir: Path = Path("./data/raw")
    stage1_output_dir: Path = Path("./data/stage1_output")
    stage2_output_dir: Path = Path("./data/stage2_output")
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    file_source_mapping: Dict[str, FileSourceItem] = Field(default_factory=dict)
    metadata_extraction: MetadataExtractionConfig = Field(default_factory=MetadataExtractionConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)