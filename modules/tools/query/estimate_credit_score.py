# author hgh
# version 1.0
"""
credit score tool
"""
import logging
from typing import Annotated, Optional

from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field

from config.models.bank_global_config import BankGlobalConfig
from modules.agent.constants import AgentName
from modules.tools.error_handler import with_tool_error_handling

logger = logging.getLogger(__name__)


class EstimateCreditScoreInput(BaseModel):
    """信用评分估算输入"""
    overdue_count: Optional[int] = Field(None, ge=0, description="逾期次数")
    has_severe_overdue: Optional[bool] = Field(None, description="是否有严重逾期（如连三累六）")
    overdue_days_max: Optional[int] = Field(None, ge=0, description="最长逾期天数")
    recent_inquiries: Optional[int] = Field(None, ge=0, description="最近3个月征信硬查询次数")
    is_white_account: Optional[bool] = Field(None, description="是否征信白户（无任何信用记录）")


@tool(
    "estimate_credit_score",
    description="根据提供的征信信息（逾期次数、严重逾期情况、查询次数等）估算个人信用评分区间和评级。"
                "返回基础分、各项扣分、最终评分区间及评级（差/一般/良好/优秀）。",
    args_schema=EstimateCreditScoreInput,
    extras={"version": "1.0.0", "tags": [AgentName.RISK_ASSESSMENT.value]}
)
@with_tool_error_handling
def estimate_credit_score(
    input: EstimateCreditScoreInput,
    bank_config: Annotated[BankGlobalConfig, InjectedToolArg],
) -> dict:
    """主函数"""
    rules = bank_config.credit_score_rules
    base_score = rules.base_score
    deductions = []
    current_score = base_score

    # 遍历所有规则，根据输入匹配执行
    for rule in rules.rules:
        deduction = _apply_rule(rule, input)
        if deduction > 0:
            deductions.append({
                "reason": rule.description,
                "points": -deduction
            })
            current_score -= deduction

    # 确保分数在合理范围内
    min_score = getattr(rules, 'min_score', 300)
    max_score = getattr(rules, 'max_score', 900)
    current_score = max(min_score, min(max_score, current_score))

    # 评级
    rating = _get_rating(rules.rating_thresholds, current_score)

    return {
        "base_score": base_score,
        "deductions": deductions,
        "total_deduction": sum(d["points"] for d in deductions),
        "final_score": current_score,
        "rating": rating,
        "disclaimer": "本评分为模拟估算，仅供参考，实际以银行审批和征信中心官方评分为准。"
    }


def _apply_rule(rule, input: EstimateCreditScoreInput) -> float:
    """应用单条评分规则，返回应扣分数"""
    # 若规则需要某字段，但未提供，则跳过
    field = rule.field
    if field is None:
        return 0.0
    value = getattr(input, field, None)
    if value is None:
        return 0.0

    # 检查条件
    if not _check_condition(rule, value):
        return 0.0

    # 根据扣分方式计算
    if rule.deduction_type == "fixed":
        return rule.deduction_value
    elif rule.deduction_type == "per_unit":
        # 按单位扣分：例如逾期次数，每逾期一次扣 rule.deduction_value 分
        if isinstance(value, (int, float)):
            return rule.deduction_value * float(value)
    elif rule.deduction_type == "range":
        # 根据数值区间扣分，需配置 ranges
        for r in rule.ranges:
            if _value_in_range(value, r):
                return r.deduction
    return 0.0


def _check_condition(rule, value: any) -> bool:
    """判断字段值是否满足规则的条件"""
    if rule.condition == "gte":
        return float(value) >= rule.threshold
    elif rule.condition == "gt":
        return float(value) > rule.threshold
    elif rule.condition == "lte":
        return float(value) <= rule.threshold
    elif rule.condition == "lt":
        return float(value) < rule.threshold
    elif rule.condition == "eq":
        return value == rule.threshold
    elif rule.condition == "neq":
        return value != rule.threshold
    elif rule.condition == "bool_true":
        return bool(value) == True
    elif rule.condition == "bool_false":
        return bool(value) == False
    return False


def _value_in_range(value: float, r) -> bool:
    """检查值是否在某个区间内（闭区间）"""
    return (r.min_value is None or value >= r.min_value) and (r.max_value is None or value <= r.max_value)


def _get_rating(thresholds, score: int) -> str:
    """根据分数和评级阈值获取评级"""
    for t in sorted(thresholds, key=lambda x: x.min_score, reverse=True):
        if score >= t.min_score:
            return t.label
    return "未知"
