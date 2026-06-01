# author hgh
# version 1.0
"""
extension plan simulation tool
"""
from typing import Annotated, Optional

from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field

from config.models.bank_global_config import BankGlobalConfig
from modules.agent.constants import AgentName
from modules.module_services.lpr_data_service import LPRDataService
from modules.tools.tool_constatnt import RepaymentMethod


class CalculateExtensionPlanInput(BaseModel):
    remaining_principal: float = Field(..., gt=0, description="剩余本金（元）")
    annual_rate: Optional[float] = Field(None, gt=0, le=50, description="年利率，不填则自动使用当前LPR")
    remaining_months: int = Field(..., gt=0, le=600, description="剩余期数（月）")
    extension_months: int = Field(..., gt=0, le=300, description="申请延长月数")
    method: RepaymentMethod = Field(RepaymentMethod.EQUAL_INSTALLMENT, description="还款方式：等额本息或等额本金")


@tool(
    "calculate_extension_plan",
    description="试算展期方案：输入剩余本金、利率、剩余期限、延长月数，计算展期后的新月供、总利息变化。",
    args_schema=CalculateExtensionPlanInput,
    extras={"version": "1.0.0", "tags": [AgentName.AFTER_LOAN.value]}
)
def calculate_extension_plan(
    input: CalculateExtensionPlanInput,
    bank_global_config: Annotated[BankGlobalConfig, InjectedToolArg],
    lpr_service: Annotated[LPRDataService, InjectedToolArg]
) -> dict:
    if input.annual_rate is None:
        lpr_data = lpr_service.get_latest_lpr()
        input.annual_rate = lpr_data["lpr_5y"]

    rules = bank_global_config.extension
    new_rate = input.annual_rate + rules.extension_rate_adjustment / 100.0
    new_months = input.remaining_months + input.extension_months
    monthly_rate_current = input.annual_rate / 100 / 12
    monthly_rate_new = new_rate / 100 / 12

    # 计算当前月供与剩余总利息
    if input.method == RepaymentMethod.EQUAL_INSTALLMENT:
        if monthly_rate_current == 0:
            current_monthly = input.remaining_principal / input.remaining_months
            current_interest = 0
        else:
            factor = (1 + monthly_rate_current) ** input.remaining_months
            current_monthly = input.remaining_principal * monthly_rate_current * factor / (factor - 1)
            current_interest = current_monthly * input.remaining_months - input.remaining_principal
    elif input.method == RepaymentMethod.EQUAL_PRINCIPAL:
        current_monthly = input.remaining_principal / input.remaining_months + input.remaining_principal * monthly_rate_current
        current_interest = 0
        rem = input.remaining_principal
        m_principal = input.remaining_principal / input.remaining_months
        for _ in range(input.remaining_months):
            interest = rem * monthly_rate_current
            current_interest += interest
            rem -= m_principal

    # 计算新方案月供与总利息
    if input.method == RepaymentMethod.EQUAL_INSTALLMENT:
        if monthly_rate_new == 0:
            new_monthly = input.remaining_principal / new_months
            new_interest = 0
        else:
            factor = (1 + monthly_rate_new) ** new_months
            new_monthly = input.remaining_principal * monthly_rate_new * factor / (factor - 1)
            new_interest = new_monthly * new_months - input.remaining_principal
    elif input.method == RepaymentMethod.EQUAL_PRINCIPAL:
        new_monthly_first = input.remaining_principal / new_months + input.remaining_principal * monthly_rate_new
        new_monthly = new_monthly_first  # 首月
        new_interest = 0
        rem = input.remaining_principal
        m_principal = input.remaining_principal / new_months
        for _ in range(new_months):
            interest = rem * monthly_rate_new
            new_interest += interest
            rem -= m_principal

    monthly_reduction = round(current_monthly - new_monthly, 2)
    additional_interest = round(new_interest - current_interest, 2)

    return {
        "method": input.method,
        "current_monthly_payment": round(current_monthly, 2),
        "new_monthly_payment": round(new_monthly, 2),
        "note": "等额本金方式下，月供逐月递减，此处展示的是首月月供。" if input.method == "等额本金" else "等额本息每月还款额固定。",
        "monthly_reduction": monthly_reduction,
        "current_remaining_interest": round(current_interest, 2),
        "new_total_interest": round(new_interest, 2),
        "additional_interest": additional_interest,
        "new_rate_used": round(new_rate, 4),
        "extension_months": input.extension_months,
        "new_term_months": new_months,
        "disclaimer": "展期利率可能上浮，具体以银行审批为准。展期后总利息将增加，请谨慎决定。"
    }
