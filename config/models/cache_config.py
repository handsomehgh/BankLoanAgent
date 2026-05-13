# author hgh
# version 1.0
from typing import Dict, Optional
from pydantic import BaseModel, Field

class L1Config(BaseModel):
    maxsize: int
    ttl: int


class L2Config(BaseModel):
    ttl: int
    null_ttl: int
    ttl_jitter: float = 0.1
    compression_threshold: int = 4096
    max_value_size: int = 1_048_576


class NamespaceCacheConfig(BaseModel):
    enable_l1: bool = True
    enable_l2: bool = False
    l1: Optional[L1Config] = None
    l2: Optional[L2Config] = None


class CacheConfig(BaseModel):
    knowledge_base_version: int = 1
    namespaces: Dict[str, NamespaceCacheConfig] = Field(default_factory=dict)