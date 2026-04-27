# author hgh
# version 1.0
from enum import Enum, StrEnum


# ========================= memory type ================================
class MemoryType(str, Enum):
    """Long-term memory type enumeration,used for the metadata[type] filed"""
    USER_PROFILE = "user_profile"  # the user info
    COMPLIANCE_RULE = "compliance_rule"  # compliance and safety memory
    INTERACTION_LOG = "interaction_log"  # interactive trajectory memory


# ========================== entity type of user profile ===========================
class ProfileEntityKey(str, Enum):
    """Profile entity key enumeration,used for extract prompt and metadata['entity_key'] field"""
    OCCUPATION = "occupation"
    INCOME = "income"
    LOAN_PURPOSE = "loan_purpose"
    LOAN_AMOUNT = "loan_amount"
    LOAN_TERM = "loan_term"
    ASSERT = "assert"
    LIABILITY = "liability"
    CREDIT_SCORE = "credit_score"
    PREFERENCE = "preference"
    CONTACT = "contact"

    @classmethod
    def to_list(cls) -> list:
        return [e.value for e in cls]


# ======================= evidence type(user_profile) =================================
class EvidenceType(str, Enum):
    EXPLICIT_STATEMENT = "explicit_statement"  # custom clearly described
    BANK_STATEMENT = "bank_statement"  # bank transaction statement
    CREDIT_REPORT = "credit_report"
    TAX_DOCUMENT = "tax_document"  # tax bill
    INFERRED = "inferred"  # infer from context


# =========================== memory status ====================================
class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"  # overwritte by higher-confidence memory
    FORGOTTEN = "forgotten"  # forgotten due to decay
    DELETE = "delete"  # soft delete


# =========================== the source of memory =============================
class MemorySource(str, Enum):
    CHAT_EXTRACTION = "chat_extraction"
    EXPLICIT_CORRECTION = "explicit_correction"
    BANK_STATEMENT = "bank_statement"
    CREDIT_REPORT = "credit_report"
    TAX_DOCUMENT = "tax_document"
    FORM_SUBMISSION = "form_submission"
    AUTO_SUMMARY = "auto_summary"  # interactive log automatic summary
    ADMIN_IMPORT = "admin_import"  # offline import

# ============================ memory metadata fields =============================
class GeneralFieldNames:
    ID = "id"
    USER_ID = "user_id"
    MEMORY_TYPE = "memory_type"
    TEXT = "text"
    CONTENT = "content"
    STATUS = "status"
    CONFIDENCE = "confidence"
    PERMANENT = "permanent"
    SOURCE = "source"
    METADATA = "metadata"
    SUPERSEDED_BY = "superseded_by"
    LAST_ACCESSED_AT = "last_accessed_at"
    CREATED_AT = "created_at"
    EXTRA = "extra"

    ENTITY_KEY = "entity_key"
    EVIDENCE_TYPE = "evidence_type"
    EFFECTIVE_DATE = "effective_date"
    EXPIRES_AT = "expires_at"

    SESSION_ID = "session_id"
    TIMESTAMP = "timestamp"
    EVENT_TYPE = "event_type"
    SENTIMENT = "sentiment"
    KEY_ENTITIES = "key_entities"

    TYPE = "type"
    RULE_ID = "rule_id"
    RULE_NAME = "rule_name"
    RULE_TYPE = "rule_type"
    PATTERN = "pattern"
    ACTION = "action"
    SEVERITY = "severity"
    PRIORITY = "priority"
    VERSION = "version"
    EFFECTIVE_FROM = "effective_from"
    EFFECTIVE_TO = "effective_to"
    TEMPLATE = "template"
    DESCRIPTION = "description"

    DECAYED_SIMILARITY = "decayed_similarity"
    SIMILARITY = "similarity"

    SCORE = "score"
    DISTANCES = "distances"
    DISTANCE = "distance"
    DOCUMENTS = "documents"
    EMBEDDINGS = "embeddings"
    METADATAS = "metadatas"

    DENSE_VECTOR = "dense_vector"
    SPARSE_VECTOR = "sparse_vector"
    COLLECTION_NAME = "collection_name"

# ========================== compliance rule action type ==========================
class ComplianceAction(str, Enum):
    """executions actions after compliance rules are triggered"""
    BLOCK = "block"  # directly intercept,don't invoke llm
    WARN = "warn"  # log warning,but continue execution
    APPEND = "append"  # append a disclaimer at the end of the answer
    REMIND = "remind"  # inject system prompt reminder
    MASK = "mask"  # desensitization


# ========================= severity of compliance rules =========================
class ComplianceSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MANDATORY = "mandatory"  # mandatory enforcement,cannot be bypassed


# ========================== special user id =====================================
class SpecialUserID(str, Enum):
    """special identifier for non-real user"""
    GLOBAL = "GLOBAL"  # used for global shared data(business knowledge,compliance rules)


# ============================ agent node name =====================================
class AgentNodeName(str, Enum):
    RETRIEVE = "retrieve"
    COMPLIANCE_GUARD = "compliance_guard"
    CALL_MODEL = "call_model"
    EXTRACT_PROFILE = "extract_profile"
    LOG_INTERACTION = "log_interaction"

# ============================ confidence threshold =================================
class ConfidenceThreshold(float, Enum):
    """Confidence boundary for memory retrieval and conflict resolution"""
    HIGH_CONFIDENCE = 0.8  # values above this are considered highly reliable
    MEDIUM_CONFIDENCE = 0.5
    OVERRIDE_MARGIN = 0.1  # confidence advantage difference required to overwrite old memories


# ============================= agent state fields =================================
class StateFields(str,Enum):
    USER_ID = "user_id"
    MESSAGES = "messages"
    RETRIEVED_CONTEXT = "retrieved_context"
    FORMATTED_CONTEXT = "formatted_context"
    PROFILE_UPDATED = "profile_updated"
    INTERACTION_LOGGED = "interaction_logged"
    COMPLIANCE_BLOCKED = "compliance_blocked"
    COMPLIANCE_WARNINGS = "compliance_warnings"
    BLOCK_REASON = "block_reason"
    MANDATORY_APPENDS = "mandatory_appends"
    SHOULD_SKIP_LLM = "should_skip_llm"
    ERROR = "error"


# ==================== interaction trajectory event type ====================
class InteractionEventType(str, Enum):
    INQUIRY = "inquiry"
    COMPLAINT = "complaint"
    APPLICATION_STARTED = "application_started"
    APPLICATION_COMPLETED = "application_completed"
    HANDOFF_REQUESTED = "handoff_requested"
    FEEDBACK = "feedback"


# ==================== interactive trajectory emotion label ====================
class InteractionSentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    ANXIOUS = "anxious"
    FRUSTRATED = "frustrated"


# ======================= chroma result field ===============================
class ChromaResFields(str, Enum):
    IDS = "ids"
    METADATAS = "metadatas"
    DOCUMENTS = "documents"
    DISTANCES = "distances"
    EMBEDDINGS = "embeddings"


# ===================== chroma operator ===================================
class ChromaOperator(str, Enum):
    AND = "$and"
    EQ = "$eq"
    GTE = "$gte"


# ======================= prompt key =======================================
class PromptKeys(str, Enum):
    CONVERSATION = "conversation"


# ======================= index type =====================================
class VectorIndexType(str, Enum):
    HNSW = "HNSW"
    IVF = "IVF"
    FLAT = "FLAT"


# ======================= config fields ==================================
class ConfigFields(str, Enum):
    CONFIGURABLE = "configurable"
    THREAD_ID = "thread_id"

#======================== search strategies ==============================
class SearchStrategy(str, Enum):
    AUTO = "auto"  # dynamic inference
    HYBRID = "hybrid"  # dense and sparse Hybrid retrieval (RRF)
    SEMANTIC = "semantic"  # pure dense vector retrieval
    KEYWORD = "keyword"  # full-text search by keywords
    MMR = "mmr"  # hybrid retrieval and MMR rerank

#========================= collection name ===============================
class CollectionNames:
    USER_PROFILE = "user_profile_memories"
    INTERACTION_LOG = "interaction_logs"
    COMPLIANCE_RULE = "compliance_rules"

    @classmethod
    def for_type(cls, memory_type: MemoryType) -> str:
        mapping = {
            MemoryType.USER_PROFILE: cls.USER_PROFILE,
            MemoryType.INTERACTION_LOG: cls.INTERACTION_LOG,
            MemoryType.COMPLIANCE_RULE: cls.COMPLIANCE_RULE,
        }
        return mapping[memory_type]

if __name__ == '__main__':
    print(MemoryType.__members__.values())
