# author hgh
# version 1.0
"""
repayment method change trial tool
"""
import logging
from typing import Annotated, Optional

from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field, field_validator

from config.models.bank_global_config import BankGlobalConfig
from modules.agent.constants import AgentName
from modules.module_services.lpr_data_service import LPRDataService

logger = logging.getLogger(__name__)


class CalculateRepaymentMethodSwitchInput(BaseModel):
    """还款方式变更试算输入"""
    original_principal: float = Field(..., gt=0, description="原始贷款金额（元）")
    annual_rate: Optional[float] = Field(None, gt=0, le=50, description="年利率，不填则自动使用当前LPR")
    total_months: int = Field(..., gt=0, le=600, description="贷款总期数（月）")
    paid_months: int = Field(..., ge=0, le=600, description="已还期数（月）")
    current_method: str = Field(..., description="当前还款方式：等额本息 或 等额本金")
    target_method: str = Field(..., description="目标还款方式：等额本息 或 等额本金")

    @field_validator("current_method", "target_method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        if v not in ("等额本息", "等额本金"):
            raise ValueError("还款方式必须为 '等额本息' 或 '等额本金'")
        return v


@tool(
    "calculate_repayment_method_switch",
    description="试算还款方式变更：从等额本息改为等额本金，或反之。计算剩余本金、新月供（首月）、利息变化及手续费。"
                "返回对比结果、节省利息、手续费及是否值得变更的建议。",
    args_schema=CalculateRepaymentMethodSwitchInput,
    extras={"version": "1.0.0", "tags": [AgentName.AFTER_LOAN.value]}
)
def calculate_repayment_method_switch(
    input: CalculateRepaymentMethodSwitchInput,
    bank_config: Annotated[BankGlobalConfig, InjectedToolArg],
    lpr_service: Annotated[LPRDataService, InjectedToolArg]
) -> dict:
    if input.paid_months >= input.total_months:
        return {"error": "已还期数不能大于等于总期数"}
    if input.current_method == input.target_method:
        return {"error": "当前还款方式和目标还款方式相同，无需变更"}

    if input.annual_rate is None:
        lpr_data = lpr_service.get_latest_lpr()
        input.annual_rate = lpr_data["lpr_5y"]

    remaining_months = input.total_months - input.paid_months
    monthly_rate = input.annual_rate / 100 / 12

    # 1. 计算剩余本金
    remaining_principal = _calc_remaining_principal(
        input.original_principal, monthly_rate, input.total_months,
        input.paid_months, input.current_method
    )

    # 2. 原方式剩余总利息
    original_interest = _calc_remaining_interest(
        remaining_principal, monthly_rate, remaining_months, input.current_method
    )

    # 3. 新方式首月月供和剩余总利息
    new_first_monthly, new_interest = _calc_new_method(
        remaining_principal, monthly_rate, remaining_months, input.target_method
    )

    saved_interest = original_interest - new_interest

    # 4. 手续费
    switch_fee = getattr(bank_config, 'method_switch_fee', 200.0)
    net_saving = saved_interest - switch_fee

    return {
        "remaining_principal": round(remaining_principal, 2),
        "remaining_months": remaining_months,
        "current_method": input.current_method,
        "current_remaining_interest": round(original_interest, 2),
        "target_method": input.target_method,
        "new_first_monthly": round(new_first_monthly, 2),
        "new_total_interest": round(new_interest, 2),
        "saved_interest": round(saved_interest, 2),
        "switch_fee": switch_fee,
        "net_saving": round(net_saving, 2),
        "worth_switching": net_saving > 0,
        "note": (
            f"变更后首月月供 {round(new_first_monthly, 2)} 元，"
            f"剩余总利息 {round(original_interest, 2)} → {round(new_interest, 2)} 元，"
            f"{'节省' if saved_interest > 0 else '增加'} {round(abs(saved_interest), 2)} 元。"
            f"{f'手续费 {switch_fee} 元，净{"节省" if net_saving > 0 else "增加"} {round(abs(net_saving), 2)} 元。' if switch_fee > 0 else ''}"
        ),
        "disclaimer": "试算结果仅供参考，实际以银行审批为准。"
    }


# ---------- 辅助函数 ----------
def _calc_remaining_principal(original: float, rate: float, total: int, paid: int, method: str) -> float:
    if method == "等额本息":
        if rate == 0:
            return original * (1 - paid / total)
        factor = (1 + rate) ** total
        return original * (factor - (1 + rate) ** paid) / (factor - 1)
    else:
        return original * (1 - paid / total)


def _calc_remaining_interest(principal: float, rate: float, months: int, method: str) -> float:
    if method == "等额本息":
        if rate == 0:
            return 0.0
        factor = (1 + rate) ** months
        monthly = principal * rate * factor / (factor - 1)
        return monthly * months - principal
    else:
        interest = 0.0
        rem = principal
        mp = principal / months
        for _ in range(months):
            interest += rem * rate
            rem -= mp
        return interest


def _calc_new_method(principal: float, rate: float, months: int, method: str) -> tuple:
    if method == "等额本息":
        if rate == 0:
            return principal / months, 0.0
        factor = (1 + rate) ** months
        monthly = principal * rate * factor / (factor - 1)
        return monthly, monthly * months - principal
    else:
        mp = principal / months
        interest = 0.0
        rem = principal
        for _ in range(months):
            interest += rem * rate
            rem -= mp
        return mp + principal * rate, interest
