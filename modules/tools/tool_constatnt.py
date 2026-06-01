# author hgh
# version 1.0
from enum import Enum

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


