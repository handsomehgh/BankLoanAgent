# author hgh
# version 1.0
"""
repayment schedule generator
"""
from datetime import timedelta, datetime
from typing import Optional, Annotated

from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field

from modules.agent.constants import AgentName
from modules.module_services.lpr_data_service import LPRDataService
from modules.tools.error_handler import with_tool_error_handling
from modules.tools.tool_constatnt import RepaymentMethod


class GenerateRepaymentScheduleInput(BaseModel):
    principal: float = Field(..., gt=0, description="贷款本金（元）")
    annual_rate: Optional[float] = Field(None, gt=0, le=50, description="年利率，不填则自动使用当前LPR")
    term_years: int = Field(..., gt=0, le=50, description="贷款期限（年）")
    method: RepaymentMethod = Field(RepaymentMethod.EQUAL_INSTALLMENT, description="还款方式：等额本息或等额本金")
    start_date: str = Field(None, description="放款日期，格式 YYYY-MM-DD，默认当前日期")

@tool(
    "generate_repayment_schedule",
    description="生成完整还款计划表，包含每期还款日、月供、本金、利息、剩余本金。返回 summary（摘要）和 schedule（前3期+末3期）。",
    args_schema=GenerateRepaymentScheduleInput,
    extras={"version": "1.0.0","tags": [AgentName.AFTER_LOAN.value,AgentName.LOAN_ADVISOR.value]}
)
@with_tool_error_handling
def generate_repayment_schedule(input: GenerateRepaymentScheduleInput,lpr_service: Annotated[LPRDataService,InjectedToolArg]) -> dict:
    if input.annual_rate is None:
        lpr_data = lpr_service.get_latest_lpr()
        input.annual_rate = lpr_data["lpr_5y"]

    months = input.term_years * 12
    monthly_rate = input.annual_rate / 100 / 12
    start = datetime.strptime(input.start_date, "%Y-%m-%d") if input.start_date else datetime.now()

    schedule = []
    remaining = input.principal
    total_interest = 0.0
    total_payment = 0.0

    if input.method == RepaymentMethod.EQUAL_INSTALLMENT:
        if monthly_rate == 0:
            monthly_payment = input.principal / months
        else:
            factor = (1 + monthly_rate) ** months
            monthly_payment = input.principal * monthly_rate * factor / (factor - 1)

        for i in range(1, months + 1):
            interest = remaining * monthly_rate
            principal_paid = monthly_payment - interest
            remaining -= principal_paid
            total_interest += interest
            total_payment += monthly_payment
            date = start + timedelta(days=30 * (i - 1))
            record = {
                "period": i,
                "date": date.strftime("%Y-%m-%d"),
                "payment": round(monthly_payment, 2),
                "principal": round(principal_paid, 2),
                "interest": round(interest, 2),
                "remaining": max(0, round(remaining, 2)),
            }
            if i <= 3 or i > months - 3:
                schedule.append(record)

    elif input.method == RepaymentMethod.EQUAL_PRINCIPAL:
        monthly_principal = input.principal / months
        for i in range(1, months + 1):
            interest = remaining * monthly_rate
            payment = monthly_principal + interest
            remaining -= monthly_principal
            total_interest += interest
            total_payment += payment
            date = start + timedelta(days=30 * (i - 1))
            record = {
                "period": i,
                "date": date.strftime("%Y-%m-%d"),
                "payment": round(payment, 2),
                "principal": round(monthly_principal, 2),
                "interest": round(interest, 2),
                "remaining": max(0, round(remaining, 2)),
            }
            if i <= 3 or i > months - 3:
                schedule.append(record)

    return {
        "method": input.method,
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_interest, 2),
        "monthly_payment_first": schedule[0]["payment"] if schedule else 0,
        "schedule_summary": schedule,
        "full_schedule_available": True,
        "disclaimer": "计算结果仅供参考，实际还款计划以银行合同为准"
    }
