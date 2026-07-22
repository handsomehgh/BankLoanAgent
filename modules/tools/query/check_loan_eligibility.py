# author hgh
# version 1.0
"""
loan eligibility pre-approval tool
"""
import logging
from typing import List, Optional, Any, Annotated

from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field, field_validator

from config.models.bank_global_config import EligibilityRule, BankGlobalConfig
from modules.agent.constants import AgentName
from modules.tools.error_handler import with_tool_error_handling

logger = logging.getLogger(__name__)


class CheckLoanEligibilityInput(BaseModel):
    """贷款资格预审输入"""
    loan_type: str = Field(..., description="贷款类型：住房贷款、消费贷款、经营贷款")
    desired_amount: float = Field(None, description="期望贷款金额（元），不提供则跳过额度相关检查")
    term_years: int = Field(None, description="期望贷款期限（年），不提供则跳过期限相关检查")
    age: int = Field(None, description="年龄（岁）")
    monthly_income: float = Field(None, description="月收入（元）")
    occupation: Optional[str] = Field(None, description="职业")
    credit_history: Optional[str] = Field(None, description="征信简况")
    existing_monthly_debt: float = Field(0, description="现有月债务（元）")
    loan_purpose: Optional[str] = Field(None, description="贷款用途描述")
    has_real_estate: Optional[bool] = Field(None, description="名下是否有房产")
    work_years: int = Field(None, description="当前工作年限（年）")

    @field_validator("loan_type")
    @classmethod
    def validate_loan_type(cls, v: str) -> str:
        allowed = ["住房贷款", "消费贷款", "经营贷款"]
        if v not in allowed:
            raise ValueError(f"贷款类型必须是 {', '.join(allowed)} 之一")
        return v


@tool(
    "check_loan_eligibility",
    description="根据客户画像和贷款产品要求，快速判断是否满足基本准入条件。"
                "返回是否通过初步筛选、不满足的规则列表、满足的规则列表以及改进建议。"
                "支持年龄、收入、征信、工作年限、负债率等多维度检查。",
    args_schema=CheckLoanEligibilityInput,
    extras={"version": "1.0.0", "tags": [AgentName.LOAN_ADVISOR.value]}
)
@with_tool_error_handling
def check_loan_eligibility(
    input: CheckLoanEligibilityInput,
    bank_config: Annotated[BankGlobalConfig, InjectedToolArg],
) -> dict:
    """主函数：执行多维度资格检查"""
    rules = bank_config.eligibility_rules
    # 筛选适用于当前贷款类型的规则
    applicable_rules = [
        r for r in rules
        if not r.applicable_loan_types or input.loan_type in r.applicable_loan_types
    ]

    passed = []
    failed = []
    skipped = []

    # 构建上下文数据，用于规则条件评估
    context = {
        "age": input.age,
        "monthly_income": input.monthly_income,
        "occupation": input.occupation,
        "credit_history": input.credit_history,
        "existing_monthly_debt": input.existing_monthly_debt,
        "desired_amount": input.desired_amount,
        "term_years": input.term_years,
        "loan_purpose": input.loan_purpose,
        "has_real_estate": input.has_real_estate,
        "work_years": input.work_years,
    }

    for rule in applicable_rules:
        # 如果规则需要的字段不存在，且不是必选规则，则跳过
        if not _has_required_fields(rule, context) and rule.level != "MUST":
            skipped.append({"rule_name": rule.name, "reason": "缺少必要信息，无法判断"})
            continue

        result = _evaluate_rule(rule, context)
        if result is True:
            passed.append({
                "rule_name": rule.name,
                "description": rule.description,
                "level": rule.level,
            })
        elif result is False:
            failed.append({
                "rule_name": rule.name,
                "description": rule.description,
                "level": rule.level,
                "message": rule.fail_message or f"不满足 {rule.name}",
                "suggestion": rule.suggestion or "",
            })
        else:
            # 跳过的情况
            skipped.append({"rule_name": rule.name, "reason": "缺少必要信息，无法判断"})

    # 生成建议
    suggestions = _generate_suggestions(failed, passed)

    eligible = len(failed) == 0 or all(r["level"] != "MUST" for r in failed)

    return {
        "eligible": eligible,
        "passed_rules": passed,
        "failed_rules": failed,
        "skipped_rules": skipped,
        "suggestions": suggestions,
        "disclaimer": "以上为初步预审结果，最终以银行正式审批为准。"
    }


def _has_required_fields(rule: EligibilityRule, context: dict) -> bool:
    """检查是否提供了规则所需的必要字段"""
    required = _get_rule_required_fields(rule)
    for field in required:
        if context.get(field) is None:
            return False
    return True


def _get_rule_required_fields(rule: EligibilityRule) -> List[str]:
    """获取规则条件中涉及的所有字段"""
    fields = set()
    for condition in rule.conditions:
        fields.add(condition.field)
    return list(fields)


def _evaluate_rule(rule: EligibilityRule, context: dict) -> Optional[bool]:
    """评估单条规则：所有条件都满足时返回 True，任一条件不满足返回 False，无法判断返回 None"""
    if not rule.conditions:
        return True
    for condition in rule.conditions:
        value = context.get(condition.field)
        if value is None:
            return None  # 无法判断
        if not _evaluate_condition(condition, value):
            return False
    return True


def _evaluate_condition(condition, value: Any) -> bool:
    """评估单个条件"""
    from config.models.bank_global_config import ConditionType
    if condition.op == ConditionType.GT:
        return value > condition.value
    elif condition.op == ConditionType.GTE:
        return value >= condition.value
    elif condition.op == ConditionType.LT:
        return value < condition.value
    elif condition.op == ConditionType.LTE:
        return value <= condition.value
    elif condition.op == ConditionType.EQ:
        return value == condition.value
    elif condition.op == ConditionType.NEQ:
        return value != condition.value
    elif condition.op == ConditionType.IN:
        return value in condition.values if condition.values else False
    elif condition.op == ConditionType.NOT_IN:
        return value not in condition.values if condition.values else True
    elif condition.op == ConditionType.CONTAINS:
        if isinstance(value, str) and condition.value:
            return condition.value in value
        return False
    elif condition.op == ConditionType.NOT_CONTAINS:
        if isinstance(value, str) and condition.value:
            return condition.value not in value
        return True
    return False


def _generate_suggestions(failed: list, passed: list) -> List[str]:
    """根据失败和通过的规则生成改进建议"""
    suggestions = []
    must_fail = [r for r in failed if r["level"] == "MUST"]
    suggest_fail = [r for r in failed if r["level"] != "MUST"]

    if must_fail:
        suggestions.append("您当前不满足以下必要准入条件，建议改善后再申请：")
        for r in must_fail:
            suggestions.append(f"- {r['message']}。建议：{r['suggestion']}" if r['suggestion'] else f"- {r['message']}")

    if suggest_fail:
        suggestions.append("以下条件为非必要条件，但可能影响审批结果：")
        for r in suggest_fail:
            suggestions.append(f"- {r['message']}。建议：{r['suggestion']}" if r['suggestion'] else f"- {r['message']}")

    if not failed:
        suggestions.append("您当前满足所有已检查的准入条件，可以提交正式申请。")

    return suggestions
