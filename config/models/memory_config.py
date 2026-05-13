"""
Memory System Complete Configuration Model
Includes: gating rules, evidence rules, emotional rules, compliance grading, context window, dead letter queue, etc.
"""
from pydantic import BaseModel, Field
from typing import List, Dict

# ---- sub model ----
class ComplianceSeverity(BaseModel):
    critical: int = 0
    high: int = 1
    medium: int = 2
    low: int = 3
    mandatory: int = 0

class EvidenceRules(BaseModel):
    version: str = "1.0"
    strong_keywords: Dict[str, List[str]] = Field(default_factory=dict)
    evidence_weights: Dict[str, int] = Field(default_factory=dict)

class SentimentRules(BaseModel):
    version: str = "1.0"
    strong_keywords: Dict[str, List[str]] = Field(default_factory=dict)

class WeakSignalItem(BaseModel):
    words: List[str] = Field(...)
    score: int = Field(...)

class MemoryGateRules(BaseModel):
    version: str = "1.1"
    strong_patterns: List[str] = Field(default_factory=list)
    explicit_triggers: List[str] = Field(default_factory=list)
    weak_signals: List[WeakSignalItem] = Field(default_factory=list)
    match_threshold: int = 4

# ---- Overall Configuration Model----
class MemorySystemConfig(BaseModel):
    version: str = "1.5"

    #default vector db
    vector_backend: str = "milvus"

    #chroma address
    chroma_persist_dir: str = "./chroma_data"

    #vector search strategy
    default_search_strategy: str = "semantic"

    # ------ retrieval base configurable -----------
    memory_top_k: int =  5
    memory_fetch_k: int = 10
    memory_min_similarity: float = 0.3

    # context window & log interaction max context
    max_context_messages: int = 10
    max_summary_length: int = 500
    interaction_log_max_context: int = 50
    interaction_log_max_length: int = 1500
    interaction_log_min_new_msgs: int = 3
    profile_extraction_fallback_window: int = 10

    # compliance & cache
    compliance_cache_ttl: int = 300
    strict_enum_validation: bool = True

    # decay and clean
    decay_factor: float = 0.0462
    decay_threshold: float = 0.3
    cleanup_interval_hours: int = 24

    # dead letter quene
    memory_dlq_path: str = "memory_dlq.jsonl"

    # sub module
    compliance_severity: ComplianceSeverity = Field(default_factory=ComplianceSeverity)
    evidence_rules: EvidenceRules = Field(default_factory=EvidenceRules)
    sentiment_rules: SentimentRules = Field(default_factory=SentimentRules)
    memory_gate: MemoryGateRules = Field(default_factory=MemoryGateRules)