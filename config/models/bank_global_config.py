# author hgh
# version 1.0
from pydantic import BaseModel, Field
from typing import List, Any, Optional

from modules.tools.tool_constatnt import FeeBaseType, ConditionType


class LprConfig(BaseModel):
    lpr_1y: float
    lpr_5y: float
    date: str


class ProductPointConfig(BaseModel):
    product_type: str
    min_diff: float
    max_diff: float
    notes: str


class ProductDtiThresholds(BaseModel):
    """DTI 审批阈值"""
    product_type: str = Field(..., description="产品类型")
    safe: float = Field(..., ge=0, le=1, description="安全线（DTI ≤ 此值视为安全）")
    warn: float = Field(..., ge=0, le=1, description="警戒线（DTI 在此值以下但高于 safe 为关注）")
    max: float = Field(..., ge=0, le=1, description="上限（DTI 超过此值为超标）")


class PrepaymentConfig(BaseModel):
    """提前还款违约金配置"""
    penalty_rate: float = Field(0.01, ge=0, le=0.05, description="违约金比例（如 0.01 表示 1%）")
    free_after_months: int = Field(12, ge=0, description="正常还款多少个月后免收违约金")
    max_penalty_rate: float = Field(0.03, ge=0, le=0.05, description="最高违约金比例")


class ExtensionRules(BaseModel):
    """展期业务规则配置"""
    min_paid_months: int = Field(6, ge=0, description="申请展期前至少正常还款的月数")
    max_extension_ratio: float = Field(0.5, gt=0, le=1,
                                       description="展期后总期限不超过原期限的倍数，如0.5表示最多延长原期限的一半")
    allow_with_overdue: bool = Field(False, description="有逾期记录是否允许申请展期")
    extension_rate_adjustment: float = Field(0.10, ge=0, description="展期利率上浮基点，如0.10表示上浮10BP")
    supported_loan_types: List[str] = Field(
        default_factory=lambda: ["住房贷款", "消费贷款", "经营贷款"],
        description="支持展期的贷款类型"
    )


class FeeItem(BaseModel):
    name: str
    description: str = ""
    rate: float = 0.0
    fixed: float = 0.0
    calc_base: FeeBaseType = FeeBaseType.LOAN_AMOUNT
    applicable_loan_types: List[str] = Field(default_factory=list)
    applicable_collateral_types: List[str] = Field(default_factory=list)
    is_required: bool = True  # 是否强制费用


class RuleCondition(BaseModel):
    field: str
    op: ConditionType
    value: Any = None
    values: Optional[List[Any]] = None


class EligibilityRule(BaseModel):
    name: str
    description: str = ""
    level: str = "MUST"
    conditions: List[RuleCondition] = Field(default_factory=list)
    applicable_loan_types: List[str] = Field(default_factory=list)
    fail_message: str = ""
    suggestion: str = ""


class EligibilityConfig(BaseModel):
    rules: List[EligibilityRule] = Field(default_factory=list)


class CreditScoreRange(BaseModel):
    """信用评分区间扣分项"""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    deduction: float = 0.0


class CreditScoreRuleItem(BaseModel):
    """单条信用评分规则"""
    field: str
    description: str = ""
    condition: str = "gte"
    threshold: Any = None
    deduction_type: str = "fixed"
    deduction_value: float = 0.0
    ranges: List[CreditScoreRange] = Field(default_factory=list)


class CreditScoreRatingThreshold(BaseModel):
    """评级阈值"""
    label: str
    min_score: int


class CreditScoreRules(BaseModel):
    """信用评分规则集"""
    base_score: int = 750
    min_score: int = 300
    max_score: int = 900
    rules: List[CreditScoreRuleItem] = Field(default_factory=list)
    rating_thresholds: List[CreditScoreRatingThreshold] = Field(default_factory=list)


class LtvRule(BaseModel):
    """LTV 上限规则"""
    max_ltv: float = Field(..., gt=0, le=1, description="最高贷款价值比（如 0.8 表示 80%）")
    applicable_loan_types: List[str] = Field(default_factory=list, description="适用贷款类型，空表示全部")
    first_house_only: Optional[bool] = Field(None, description="是否仅适用于首套房（True=首套，False=二套，None=不限）")
    description: str = ""


class BankGlobalConfig(BaseModel):
    lpr: LprConfig
    method_switch_fee: float = Field(default=200.0, ge=0, description="还款方式变更手续费（元）")
    product_point: List[ProductPointConfig]
    product_dti: List[ProductDtiThresholds] = Field(
        default_factory=ProductDtiThresholds,
        description="按贷款类型区分的 DTI 审批标准，键为贷款类型（如 住房贷款、消费贷款）"
    )
    prepayment: PrepaymentConfig = Field(default_factory=PrepaymentConfig, description="提前还款相关配置")
    extension: ExtensionRules = Field(default_factory=ExtensionRules, description="展期相关配置")
    loan_fees: List[FeeItem] = Field(default_factory=list, description="贷款附加费用配置")
    eligibility_rules: List[EligibilityRule] = Field(default_factory=list, description="贷款资格准入规则")
    credit_score_rules: CreditScoreRules = Field(default_factory=CreditScoreRules, description="信用评分规则")
    ltv_rules: List[LtvRule] = Field(default_factory=list, description="LTV 上限规则")
    overdue_penalty_multiplier: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
        description="逾期罚息倍数，央行基准为1.5倍"
    )
