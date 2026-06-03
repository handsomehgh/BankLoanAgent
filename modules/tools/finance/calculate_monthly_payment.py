# author hgh
# version 1.0
"""
Equal Principal and Interest / Equal Principal Monthly Payment Calculator
"""
from typing import Dict, Annotated, Optional

from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field, field_validator

from exceptions.exception import ToolExecutionException
from modules.agent.constants import AgentName
from modules.module_services.lpr_data_service import LPRDataService
from modules.tools.base_tool import ToolErrorType
from modules.tools.error_handler import with_tool_error_handling
from modules.tools.tool_constatnt import RepaymentMethod


# ====================== strategy implementation ===================
class EqualInstallmentCalculator:
    @staticmethod
    def calculate(principal: float, annual_rate: float, term_years: int) -> dict:
        months = term_years * 12
        monthly_rate = annual_rate / 100 / 12
        if monthly_rate == 0:
            monthly_payment = principal / months
            total_interest = 0.0
        else:
            factor = (1 + monthly_rate) ** months
            monthly_payment = principal * monthly_rate * factor / (factor - 1)
            total_interest = monthly_payment * months - principal
        schedule = EqualInstallmentCalculator._build_schedule(principal, monthly_rate, months, monthly_payment)
        total_payment = principal + total_interest
        return {
            "method": "等额本息",
            "monthly_payment": round(monthly_payment, 2),
            "total_interest": round(total_interest, 2),
            "total_payment": round(total_payment, 2),
            "first_monthly_payment": round(monthly_payment, 2),
            "last_monthly_payment": round(monthly_payment, 2),
            "schedule": schedule,
        }

    @staticmethod
    def _build_schedule(principal, rate, months, monthly):
        schedule = []
        remaining = principal
        for i in range(1, months + 1):
            interest = remaining * rate
            principal_paid = monthly - interest
            remaining -= principal_paid
            if i <= 3 or i > months - 3:
                schedule.append({
                    "period": i,
                    "principal": round(principal_paid, 2),
                    "interest": round(interest, 2),
                    "payment": round(monthly, 2),
                })
        return schedule


class EqualPrincipalCalculator:
    @staticmethod
    def calculate(principal: float, annual_rate: float, term_years: int) -> dict:
        months = term_years * 12
        monthly_principal = principal / months
        monthly_rate = annual_rate / 100 / 12
        total_interest = 0.0
        schedule = []
        remaining = principal
        first_monthly = None
        last_monthly = None

        for i in range(1, months + 1):
            interest = remaining * monthly_rate
            # 最后一期：本金取剩余金额，避免尾差
            if i == months:
                principal_paid = remaining
                payment = remaining + interest
            else:
                principal_paid = monthly_principal
                payment = monthly_principal + interest

            total_interest += interest
            remaining -= principal_paid

            if i == 1:
                first_monthly = payment
            last_monthly = payment

            if i <= 3 or i > months - 3:
                schedule.append({
                    "period": i,
                    "principal": round(principal_paid, 2),
                    "interest": round(interest, 2),
                    "payment": round(payment, 2),
                })

        total_payment = principal + total_interest
        return {
            "method": "等额本金",
            "monthly_payment": round(first_monthly, 2) if first_monthly else 0,  # 首月月供
            "total_interest": round(total_interest, 2),
            "total_payment": round(total_payment, 2),
            "first_monthly_payment": round(first_monthly, 2) if first_monthly else 0,
            "last_monthly_payment": round(last_monthly, 2) if last_monthly else 0,
            "schedule": schedule,
        }


# ============== strategy register ==========================
CALCULATOR_REGISTRY: Dict[RepaymentMethod, object] = {
    RepaymentMethod.EQUAL_INSTALLMENT: EqualInstallmentCalculator(),
    RepaymentMethod.EQUAL_PRINCIPAL: EqualPrincipalCalculator(),
}


# ================= parameter model ====================
class CalculateMonthlyPaymentInput(BaseModel):
    principal: float = Field(..., gt=0, description="贷款本金（元）")
    annual_rate: Optional[float] = Field(None, gt=0, le=50, description="年利率，不填则自动使用当前LPR")
    term_years: int = Field(..., gt=0, le=50, description="贷款期限（年）")
    method: RepaymentMethod = Field(default=RepaymentMethod.EQUAL_INSTALLMENT,
                                    description="还款方式：等额本息 / 等额本金")

    @field_validator("method", mode="before")
    @classmethod
    def parse_method(cls, v):
        if isinstance(v, str):
            for m in RepaymentMethod:
                if m.value == v:
                    return m
            raise ValueError(f"不支持的还款方式: {v}")
        return v

@tool(
    "calculate_monthly_payment",
    description="计算等额本息或等额本金的月供、总利息、还款总额及还款计划摘要。"
                "返回字段：method（还款方式）, monthly_payment（首月月供，等额本息每月相同，等额本金为最高月供），"
                "total_interest（总利息）, total_payment（还款总额）, first_monthly_payment（首月月供）, "
                "last_monthly_payment（末月月供）, schedule（前3期和末3期详情）。所有金额单位为元，四舍五入到分。",
    args_schema=CalculateMonthlyPaymentInput,
    extras={"version": "1.0.0", "tags": [AgentName.LOAN_ADVISOR.value, AgentName.AFTER_LOAN.value]})
@with_tool_error_handling
def calculate_monthly_payment(input: CalculateMonthlyPaymentInput,lpr_service: Annotated[LPRDataService,InjectedToolArg]) -> dict:
    """计算等额本息或等额本金的月供和总利息。

     Args:
        input: 包含贷款本金、年利率、期限和还款方式的输入对象
        lpr_service: 利率提供服务

    Returns:
        dict: 包含 monthly_payment (首月月供), total_interest (总利息), schedule (前3期/末3期摘要)
    """
    if input.annual_rate is None:
        lpr_data = lpr_service.get_latest_lpr()
        input.annual_rate = lpr_data["lpr_5y"]

    calculator = CALCULATOR_REGISTRY.get(input.method)
    if calculator is None:
        raise ToolExecutionException(f"不支持的还款方式",ToolErrorType.PARAMETER_ERROR)
    result = calculator.calculate(
        principal=input.principal,
        annual_rate=input.annual_rate,
        term_years=input.term_years,
    )
    result["disclaimer"] = "计算结果仅供参考，实际还款计划以银行合同为准。"
    return result
