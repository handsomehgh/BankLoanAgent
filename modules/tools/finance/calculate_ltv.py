# author hgh
# version 1.0
"""
loan to value calculate tool
"""
import logging
from typing import Annotated, Optional

from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field, field_validator

from config.models.bank_global_config import BankGlobalConfig, LtvRule
from modules.agent.constants import AgentName
from modules.tools.error_handler import with_tool_error_handling

logger = logging.getLogger(__name__)


class CalculateLtvInput(BaseModel):
    """抵押率计算输入"""
    loan_amount: float = Field(..., gt=0, description="贷款金额（元）")
    collateral_value: float = Field(..., gt=0, description="抵押物评估价值（元）")
    loan_type: str = Field(..., description="贷款类型：住房贷款、消费贷款、经营贷款")
    is_first_house: Optional[bool] = Field(None, description="是否首套房（仅住房贷款需要）")

    @field_validator("loan_type")
    @classmethod
    def validate_loan_type(cls, v: str) -> str:
        allowed = ["住房贷款", "消费贷款", "经营贷款"]
        if v not in allowed:
            raise ValueError(f"贷款类型必须是 {', '.join(allowed)} 之一")
        return v


@tool(
    "calculate_ltv",
    description="计算抵押率 (LTV) = 贷款金额 / 抵押物评估价值。根据贷款类型和是否首套房判断是否超过银行规定的 LTV 上限。"
                "返回 LTV 百分比、是否超标、允许的最高 LTV 及风险提示。",
    args_schema=CalculateLtvInput,
    extras={"version": "1.0.0", "tags": [AgentName.RISK_ASSESSMENT.value]}
)
@with_tool_error_handling
def calculate_ltv(
    input: CalculateLtvInput,
    bank_config: Annotated[BankGlobalConfig, InjectedToolArg],
) -> dict:
    """主函数：计算 LTV 并判断合规性"""
    ltv = input.loan_amount / input.collateral_value

    # 查找适用的 LTV 规则
    rule = _find_applicable_ltv_rule(
        bank_config.ltv_rules,
        input.loan_type,
        input.is_first_house
    )

    max_ltv = rule.max_ltv if rule else 0.8  # 默认上限 80%
    compliant = ltv <= max_ltv
    risk_level = "安全" if compliant else "超标"
    message = (
        f"当前 LTV 为 {ltv*100:.2f}%，未超过该贷款类型允许的最高 LTV {max_ltv*100:.0f}%。"
        if compliant
        else f"当前 LTV 为 {ltv*100:.2f}%，已超过该贷款类型允许的最高 LTV {max_ltv*100:.0f}%。"
    )

    return {
        "ltv": round(ltv, 4),
        "ltv_percent": f"{ltv*100:.2f}%",
        "loan_amount": input.loan_amount,
        "collateral_value": input.collateral_value,
        "max_ltv": max_ltv,
        "max_ltv_percent": f"{max_ltv*100:.0f}%",
        "compliant": compliant,
        "risk_level": risk_level,
        "message": message,
        "disclaimer": "以上计算仅供参考，最终以银行审批为准。"
    }


def _find_applicable_ltv_rule(rules: list, loan_type: str, is_first_house: Optional[bool]) -> Optional[LtvRule]:
    """根据贷款类型和首套房标志匹配 LTV 规则"""
    # 优先匹配最具体的规则
    for rule in sorted(rules, key=lambda r: (0 if r.applicable_loan_types else 1)):  # 有适用类型优先
        # 检查贷款类型
        if rule.applicable_loan_types and loan_type not in rule.applicable_loan_types:
            continue
        # 检查首套房标志
        if rule.first_house_only is not None:
            if is_first_house is None:
                continue  # 需要首套房信息但未提供，跳过此规则
            if rule.first_house_only != is_first_house:
                continue
        # 匹配成功
        return rule
    # 没有匹配的规则，返回 None（调用方采用默认值）
    return None
