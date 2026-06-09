# author hgh
# version 1.0
"""
unified exception definition
"""
from enum import Enum


class ToolErrorType(str, Enum):
    PARAMETER_ERROR = "parameter_error"  # 用户输入问题：金额为负、期限为0等
    BUSINESS_ERROR = "business_error"  # 配置/数据/业务逻辑问题：政策缺失、规则未定义
    TEMPORARY_ERROR = "temporary_error"  # 临时性故障：超时、网络断开、服务暂时不可用
    EXTERNAL_ERROR = "external_error"  # 其他未知异常：兜底类型
    CIRCUIT_OPEN = "circuit_open"


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

#====================skill=======================
class SkillExecutionError(BankLoanException):
    """Skill 执行异常"""
    pass

#=================tool===========================
class ToolExecutionError(BankLoanException):
    pass

class CircuitBreakerOpenError(BankLoanException):
    pass

class ToolExecutionException(BankLoanException):
    def __init__(self, message: str, error_type: ToolErrorType = ToolErrorType.EXTERNAL_ERROR):
        super().__init__(message)
        self.error_type = error_type