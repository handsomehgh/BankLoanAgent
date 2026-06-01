# author hgh
# version 1.0
"""
loan comprehensive cost trial calculation tool
calculate the total cost of the loan: interest and various fees(appraisal fee, registration fee, notarization fee, insurance fee, etc.)
"""
from typing import Optional, List, Annotated

from langchain_core.tools import tool, InjectedToolArg
from pydantic import Field, BaseModel

from config.models.bank_global_config import FeeItem, BankGlobalConfig
from modules.agent.constants import AgentName
from modules.module_services.lpr_data_service import LPRDataService
from modules.tools.tool_constatnt import LoanProductType, CollateralType, FeeBaseType


class CalculateLoanTotalCostInput(BaseModel):
    """贷款综合成本试算输入"""
    principal: float = Field(..., gt=0, description="贷款本金（元）")
    annual_rate: Optional[float] = Field(None, gt=0, le=50, description="年利率，不填则自动使用当前LPR")
    term_years: int = Field(..., gt=0, le=50, description="贷款期限（年）")
    loan_type: LoanProductType = Field(..., description="贷款类型:住房贷款或消费贷款或经营贷款)")
    collateral_type: CollateralType = Field(CollateralType.NONE, description="抵押物类型，无抵押则为 NONE")
    collateral_value: Optional[float] = Field(None, description="抵押物评估价值（元），如有抵押物需提供")
    method: str = Field("等额本息", description="还款方式：等额本息或等额本金")


@tool(
    "calculate_loan_total_cost",
    description="计算贷款的综合总成本：利息 + 各项附加费用（评估费、登记费、公证费、保险费等）。"
                "返回利息总额、各项费用明细、总成本及含费用摊销的月供参考。",
    args_schema=CalculateLoanTotalCostInput,
    extras={"version": "1.0.0", "tags": [AgentName.LOAN_ADVISOR.value]}
)
def calculate_loan_total_cost(
        input: CalculateLoanTotalCostInput,
        bank_config: Annotated[BankGlobalConfig, InjectedToolArg],
        lpr_service: Annotated[LPRDataService, InjectedToolArg]
) -> dict:
    """主函数"""
    if input.annual_rate is None:
        lpr_data = lpr_service.get_latest_lpr()
        input.annual_rate = lpr_data["lpr_5y"]

    # 1. 计算利息和月供（复用与 calculate_monthly_payment 一致的计算逻辑）
    monthly_payment, total_interest = _calc_monthly_payment(
        input.principal, input.annual_rate, input.term_years, input.method
    )

    # 2. 根据贷款类型和抵押物获取适用的费用项
    fees = _get_applicable_fees(
        bank_config.loan_fees,
        input.loan_type.value,
        input.collateral_type.value,
        input.principal,
        input.collateral_value
    )

    total_fees = sum(f["amount"] for f in fees)
    total_cost = input.principal + total_interest + total_fees
    # 月供费用摊销：总费用分摊到每期
    fee_per_month = total_fees / (input.term_years * 12)
    total_monthly = monthly_payment + fee_per_month

    return {
        "principal": input.principal,
        "annual_rate": input.annual_rate,
        "term_years": input.term_years,
        "method": input.method,
        "monthly_payment": round(monthly_payment, 2),
        "total_interest": round(total_interest, 2),
        "fees": fees,  # 每项费用的明细
        "total_fees": round(total_fees, 2),
        "monthly_fee_amortized": round(fee_per_month, 2),
        "total_monthly_with_fees": round(total_monthly, 2),
        "total_cost": round(total_cost, 2),
        "disclaimer": "附加费用为预估值，实际以银行合同为准。"
    }


# ---------- 辅助函数 ----------
def _calc_monthly_payment(principal: float, annual_rate: float, term_years: int, method: str) -> tuple:
    """计算月供和总利息，与 calculate_monthly_payment 内部逻辑一致"""
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
        monthly_payment = monthly_principal + principal * monthly_rate  # 首月
    else:
        raise ValueError(f"不支持的还款方式: {method}")

    return monthly_payment, total_interest


def _get_applicable_fees(
        fee_config: List[FeeItem],
        loan_type: str,
        collateral_type: str,
        principal: float,
        collateral_value: Optional[float]
) -> List[dict]:
    """根据贷款类型和抵押物匹配费用项并计算金额"""
    result = []
    for item in fee_config:
        # 检查适用条件：贷款类型匹配，且抵押物类型在 item 的支持列表中（若支持列表为空则适用于所有）
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
