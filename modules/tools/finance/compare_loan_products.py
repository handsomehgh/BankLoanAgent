# author hgh
# version 1.0
"""
loan plan comparison tool
supports comparison of monthly payments,total interest,and total cost for multiple loan plan,optionally including additional fees
"""
import logging
from typing import List, Optional, Annotated

from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field, field_validator

from config.models.bank_global_config import BankGlobalConfig
from modules.agent.constants import AgentName
from modules.module_services.lpr_data_service import LPRDataService

logger = logging.getLogger(__name__)


class LoanScenario(BaseModel):
    """单个贷款方案"""
    principal: float = Field(..., gt=0, description="贷款本金（元）")
    annual_rate: Optional[float] = Field(None, gt=0, le=50, description="年利率，不填则自动使用当前LPR")
    term_years: int = Field(..., gt=0, le=50, description="贷款期限（年）")
    method: str = Field("等额本息", description="还款方式：等额本息或等额本金")
    loan_type: Optional[str] = Field(None, description="贷款类型，用于费用计算（住房贷款/消费贷款/经营贷款）")
    collateral_type: Optional[str] = Field(None, description="抵押物类型（房产/车辆/无抵押），用于费用计算")
    collateral_value: Optional[float] = Field(None, description="抵押物评估价值（元），用于费用计算")

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        if v not in ("等额本息", "等额本金"):
            raise ValueError("还款方式必须为 '等额本息' 或 '等额本金'")
        return v


class CompareLoanProductsInput(BaseModel):
    """贷款方案对比输入"""
    scenarios: List[LoanScenario] = Field(..., min_length=2, max_length=5, description="需要对比的贷款方案列表")
    include_total_cost: bool = Field(False, description="是否包含附加费用计算（需要提供贷款类型和抵押物信息）")


@tool(
    "compare_loan_products",
    description="对比多个贷款方案的月供、总利息和总成本。返回每个方案的详细结果及简要对比。可选包含附加费用（评估费、登记费等）。",
    args_schema=CompareLoanProductsInput,
    extras={"version": "1.0.0", "tags": [AgentName.LOAN_ADVISOR.value]}
)
def compare_loan_products(
    input: CompareLoanProductsInput,
    bank_config: Annotated[BankGlobalConfig, InjectedToolArg],
    lpr_service: Annotated[LPRDataService,InjectedToolArg]
) -> dict:
    """主函数"""
    if input.annual_rate is None:
        lpr_data = lpr_service.get_latest_lpr()
        input.annual_rate = lpr_data["lpr_5y"]

    results = []
    for i, scenario in enumerate(input.scenarios):
        # 计算月供和利息
        monthly_payment, total_interest = _calc_monthly_payment(
            scenario.principal, scenario.annual_rate, scenario.term_years, scenario.method
        )
        # 如果包含费用计算，计算总成本
        total_cost = None
        fees = []
        fee_total = 0.0
        if input.include_total_cost and scenario.loan_type:
            fees = _get_applicable_fees(
                bank_config.loan_fees,
                scenario.loan_type,
                scenario.collateral_type or "无抵押",
                scenario.principal,
                scenario.collateral_value
            )
            fee_total = sum(f["amount"] for f in fees)
            total_cost = scenario.principal + total_interest + fee_total

        result = {
            "scenario_id": i + 1,
            "principal": scenario.principal,
            "annual_rate": scenario.annual_rate,
            "term_years": scenario.term_years,
            "method": scenario.method,
            "monthly_payment": round(monthly_payment, 2),
            "total_interest": round(total_interest, 2),
        }
        if input.include_total_cost and scenario.loan_type:
            result["fees"] = fees
            result["total_fees"] = round(fee_total, 2)
            result["total_cost"] = round(total_cost, 2)
        results.append(result)

    # 生成简要对比摘要（可由 LLM 进一步解读）
    summary = _generate_comparison_summary(results, input.include_total_cost)

    return {
        "scenarios": results,
        "summary": summary,
        "disclaimer": "计算结果仅供参考，实际以银行合同为准。"
    }


# ---------- 公共计算函数 ----------
def _calc_monthly_payment(principal: float, annual_rate: float, term_years: int, method: str) -> tuple:
    months = term_years * 12
    monthly_rate = annual_rate / 100 / 12

    if method == "等额本息":
        if monthly_rate == 0:
            monthly_payment = principal / months
            total_interest = 0.0
        else:
            factor = (1 + monthly_rate) ** months
            monthly_payment = principal * monthly_rate * factor / (factor - 1)
            total_interest = monthly_payment * months - principal
    elif method == "等额本金":
        monthly_principal = principal / months
        total_interest = 0.0
        remaining = principal
        for _ in range(months):
            interest = remaining * monthly_rate
            total_interest += interest
            remaining -= monthly_principal
        monthly_payment = monthly_principal + principal * monthly_rate
    else:
        raise ValueError(f"不支持的还款方式: {method}")
    return monthly_payment, total_interest


def _get_applicable_fees(fee_config, loan_type, collateral_type, principal, collateral_value) -> List[dict]:
    """获取适用的费用项，逻辑与 calculate_loan_total_cost 一致"""
    from config.models.bank_global_config import FeeBaseType
    result = []
    for item in fee_config:
        if item.applicable_loan_types and loan_type not in item.applicable_loan_types:
            continue
        if item.applicable_collateral_types and collateral_type not in item.applicable_collateral_types:
            continue

        amount = 0.0
        if item.calc_base == FeeBaseType.LOAN_AMOUNT:
            amount = principal * item.rate + item.fixed
        elif item.calc_base == FeeBaseType.COLLATERAL_VALUE:
            val = collateral_value or 0.0
            amount = val * item.rate + item.fixed
        elif item.calc_base == FeeBaseType.COMBINED:
            val = (principal + (collateral_value or 0.0))
            amount = val * item.rate + item.fixed
        else:
            amount = item.fixed

        result.append({
            "name": item.name,
            "amount": round(amount, 2),
            "description": item.description,
            "is_required": item.is_required
        })
    return result


def _generate_comparison_summary(results: List[dict], include_cost: bool) -> str:
    """生成简单的文本对比摘要"""
    if not results:
        return ""
    parts = []
    for r in results:
        s = f"方案{r['scenario_id']}：月供 {r['monthly_payment']} 元，总利息 {r['total_interest']} 元"
        if include_cost and 'total_cost' in r:
            s += f"，总成本（含费用）{r['total_cost']} 元"
        parts.append(s)
    # 找出最低月供和最低总利息
    min_monthly = min(results, key=lambda x: x['monthly_payment'])
    min_interest = min(results, key=lambda x: x['total_interest'])
    summary = "\n".join(parts)
    summary += f"\n最低月供方案：方案{min_monthly['scenario_id']} ({min_monthly['monthly_payment']} 元)"
    summary += f"\n最低总利息方案：方案{min_interest['scenario_id']} ({min_interest['total_interest']} 元)"
    return summary