# author hgh
# version 1.0
from enum import Enum

class RegistryModules(str, Enum):
    MEMORY_SYSTEM = "memory_system"
    RETRIEVAL = "retrieval"
    LLM = "llm"
    FILE_PROCESS = "file_process"

class SpecialUserID(str, Enum):
    GLOBAL = "global"

class ConfidenceThreshold(float, Enum):
    HIGH_CONFIDENCE = 0.8
    MEDIUM_CONFIDENCE = 0.5
    OVERRIDE_MARGIN = 0.1

class ConfigFields(str, Enum):
    CONFIGURABLE = "configurable"
    THREAD_ID = "thread_id"

class SearchStrategy(str, Enum):
    AUTO = "auto"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    MMR = "mmr"

class VectorQueryFields(str, Enum):
    IDS = "ids"
    SCORE = "score"
    DENSE_VECTOR = "dense_vector"
    SPARSE_VECTOR = "sparse_vector"
    TERM_VECTOR = "term_vector"
    DISTANCE = "distance"
    DISTANCES = "distances"
    DOCUMENTS = "documents"
    EMBEDDINGS = "embeddings"
    METADATAS = "metadatas"

class VectorIndexType(str, Enum):
    HNSW = "HNSW"
    IVF_FLAT = "IVF_FLAT"
    IVF_SQ8 = "IVF_SQ8"
    FLAT = "FLAT"
    DISKANN = "DISKANN"
    AUTOINDEX = "AUTOINDEX"
    SPARSE_INVERTED_INDEX = "SPARSE_INVERTED_INDEX"

class MemoryType(str, Enum):
    USER_PROFILE = "user_profile"
    COMPLIANCE_RULE = "compliance_rule"
    INTERACTION_LOG = "interaction_log"
    BUSINESS_KNOWLEDGE = "business_knowledge"

class ComplianceSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MANDATORY = "mandatory"

class ComplianceAction(str, Enum):
    BLOCK = "block"
    WARN = "warn"
    APPEND = "append"
    REMIND = "remind"
    MASK = "mask"

class KnowledgeFileSourceType(str,Enum):
    FAQ = "faq"
    PRODUCT_MANUAL = "product_manual"
    PROCESS_GUIDE = "process_guide"
    REGULATION = "regulation"
    GLOSSARY = "glossary"


