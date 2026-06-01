# author hgh
# version 1.0
"""
最高可贷额度计算工具
根据月收入、现有负债、贷款期限和利率，基于 DTI ≤ 55% 反推最高可贷额度
"""
import logging
from typing import Optional, Annotated

from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field

from modules.agent.constants import AgentName
from modules.module_services.lpr_data_service import LPRDataService
from modules.tools.tool_constatnt import RepaymentMethod

logger = logging.getLogger(__name__)


class CalculateMaxLoanAmountInput(BaseModel):
    monthly_income: float = Field(..., gt=0, description="月收入（元）")
    existing_monthly_debt: float = Field(default=0.0, ge=0, description="现有月供（元）")
    term_years: int = Field(..., gt=0, le=50, description="贷款期限（年）")
    annual_rate: Optional[float] = Field(None, gt=0, le=50, description="年利率，不填则自动使用当前LPR")
    method: RepaymentMethod = Field(default=RepaymentMethod.EQUAL_INSTALLMENT, description="还款方式")


@tool(
    "calculate_max_loan_amount",
    description="根据月收入、现有月供、年利率和期限，基于银行 DTI ≤ 55% 的监管要求反推最高可贷额度。"
                "返回 max_amount（最高可贷额度，元）、max_monthly_payment（允许的最大月供）、dti_limit（负债率上限）",
    args_schema=CalculateMaxLoanAmountInput,
    extras={"version": "1.0.0", "tags": [AgentName.LOAN_ADVISOR.value]}
)
def calculate_max_loan_amount(input: CalculateMaxLoanAmountInput,lpr_service: Annotated[LPRDataService,InjectedToolArg]) -> dict:
    if input.annual_rate is None:
        lpr_data = lpr_service.get_latest_lpr()
        input.annual_rate = lpr_data["lpr_5y"]
    # 1. 计算允许的最大月供
    max_monthly_payment = input.monthly_income * 0.55 - input.existing_monthly_debt
    if max_monthly_payment <= 0:
        return {
            "max_amount": 0,
            "max_monthly_payment": 0,
            "dti_limit": 0.55,
            "message": "当前负债率已超过银行上限，建议先降低现有债务",
            "disclaimer": "仅供参考，最终额度以银行审批为准"
        }

    # 2. 反推本金
    months = input.term_years * 12
    monthly_rate = input.annual_rate / 100 / 12

    if input.method == RepaymentMethod.EQUAL_INSTALLMENT:
        if monthly_rate == 0:
            max_amount = max_monthly_payment * months
        else:
            factor = (1 + monthly_rate) ** months
            max_amount = max_monthly_payment * (factor - 1) / (monthly_rate * factor)
    elif input.method == RepaymentMethod.EQUAL_PRINCIPAL:
        # 等额本金：首月月供 = 本金/期限 + 本金×月利率
        # max_monthly_payment = P/n + P×r → P = max_monthly_payment / (1/n + r)
        max_amount = max_monthly_payment / (1 / months + monthly_rate)
    else:
        return {"error": f"不支持的还款方式: {input.method.value}"}

    return {
        "max_amount": round(max_amount, 2),
        "max_monthly_payment": round(max_monthly_payment, 2),
        "dti_limit": 0.55,
        "formula": "等额本息反推" if input.method == RepaymentMethod.EQUAL_INSTALLMENT else "等额本金反推",
        "disclaimer": "仅供参考，最终额度以银行审批为准"
    }