# author hgh
# version 1.0
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class PrepaymentMethod(str, Enum):
    """Early Repayment method"""
    SHORTEN_TERM = "shorten_term"
    REDUCE_PAYMENT = "reduce_payment"
    COMPARE = "compare"

class LoanProductType(str, Enum):
    HOUSING_LOAN = "住房贷款"
    CONSUMER_LOAN = "消费贷款"
    BUSINESS_LOAN = "经营贷款"

class RepaymentMethod(str,Enum):
    EQUAL_INSTALLMENT = "等额本息"
    EQUAL_PRINCIPAL = "等额本金"

class CollateralType(str, Enum):
    """type of collateral"""
    REAL_ESTATE = "房产"
    VEHICLE = "车辆"
    NONE = "无抵押"

class FeeBaseType(str, Enum):
    LOAN_AMOUNT = "loan_amount"
    COLLATERAL_VALUE = "collateral_value"
    COMBINED = "combined"
    FIXED = "fixed"

class ConditionType(str, Enum):
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"
    NEQ = "neq"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"

class ToolErrorType(str, Enum):
    PARAMETER_ERROR = "parameter_error"  # 用户输入问题：金额为负、期限为0等
    BUSINESS_ERROR = "business_error"  # 配置/数据/业务逻辑问题：政策缺失、规则未定义
    TEMPORARY_ERROR = "temporary_error"  # 临时性故障：超时、网络断开、服务暂时不可用
    EXTERNAL_ERROR = "external_error"  # 其他未知异常：兜底类型
    CIRCUIT_OPEN = "circuit_open"


class ToolResult(BaseModel):
    success: bool = Field(..., description="whether success or not")
    data: Any = Field(None, description="raw data returned by the tool")
    summary: str = Field("", description="result summary (for audit logs and display)")
    error: Optional[str] = Field(None, description="error message")
    error_type: ToolErrorType = ToolErrorType.EXTERNAL_ERROR

    def to_message_content(self) -> str:
        if not self.success:
            return f"工具调用失败: {self.error}"
        if isinstance(self.data, (dict, list)):
            import json
            return json.dumps(self.data, ensure_ascii=False)
        return str(self.data) if self.data else self.summary


