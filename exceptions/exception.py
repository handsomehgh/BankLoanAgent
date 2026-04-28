# author hgh
# version 1.0
"""
unified exception definition
"""


class BankLoanException(Exception):
    """application base exception"""


class ConfigurationError(BankLoanException):
    """configuration error"""


class LLMError(BankLoanException):
    """LLM invoke error"""


class LLMTimeoutError(LLMError):
    """LLM timeout error"""


class LLMRateLimitError(LLMError):
    """LLM rate limit error"""


# ====================memory=======================
class MemoryStoreError(BankLoanException):
    """memory base error"""


class MemoryWriteFailedError(MemoryStoreError):
    """memory write failed error"""


class MemoryRetrievalError(MemoryStoreError):
    """memory retrieval error"""


class MemoryUpdateError(MemoryStoreError):
    """memory update error"""


# ====================retrieval=====================
class RetrievalError(BankLoanException):
    """retrieval error"""


# ====================agent=======================
class AgentWorkFlowError(BankLoanException):
    """agent work flow error"""


class EvaluationError(AgentWorkFlowError):
    """evaluation error"""


class ExtractionError(AgentWorkFlowError):
    """extraction error"""


class MappingError(BankLoanException):
    """映射层异常，用于序列化/反序列化过程中的不可恢复错误"""
    pass

#=======================embedding====================
class EmbeddingError(LLMError):
    """嵌入模型异常"""
    pass

class EmbeddingTimeoutError(LLMTimeoutError):
    """嵌入超时"""
    pass

class EmbeddingRateLimitError(LLMRateLimitError):
    """嵌入限流"""
    pass