# author hgh
# version 1.0
"""
overdue interest calculation
"""
import logging
from typing import Annotated, Optional

from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field

from config.models.bank_global_config import BankGlobalConfig
from modules.agent.constants import AgentName
from modules.module_services.lpr_data_service import LPRDataService

logger = logging.getLogger(__name__)


class CalculateOverduePenaltyInput(BaseModel):
    """逾期罚息计算输入"""
    overdue_principal: float = Field(..., gt=0, description="逾期本金（元）")
    annual_rate: Optional[float] = Field(None, gt=0, le=50, description="年利率，不填则自动使用当前LPR")
    overdue_days: int = Field(..., gt=0, description="逾期天数")
    penalty_multiplier: Optional[float] = Field(
        None,
        ge=1.0, le=3.0,
        description="罚息倍数，不填则使用银行默认值（央行规定为1.5倍）"
    )


@tool(
    "calculate_overdue_penalty",
    description="计算贷款逾期罚息。罚息 = 逾期本金 × (合同年利率 × 罚息倍数 / 360) × 逾期天数。"
                "返回罚息金额、应还总额（本金+罚息）及日利率。",
    args_schema=CalculateOverduePenaltyInput,
    extras={"version": "1.0.0", "tags": [AgentName.AFTER_LOAN.value]}
)
def calculate_overdue_penalty(
    input: CalculateOverduePenaltyInput,
    bank_config: Annotated[BankGlobalConfig, InjectedToolArg],
    lpr_service: Annotated[LPRDataService, InjectedToolArg]
) -> dict:
    if input.annual_rate is None:
        lpr_data = lpr_service.get_latest_lpr()
        input.annual_rate = lpr_data["lpr_5y"]
    # 确定罚息倍数：优先手动指定，否则从配置获取
    multiplier = input.penalty_multiplier if input.penalty_multiplier is not None else bank_config.overdue_penalty_multiplier
    # 罚息日利率 = 年利率 / 360 × 倍数 （银行常用360天计息）
    daily_rate = (input.annual_rate / 100.0) / 360.0 * multiplier
    penalty = input.overdue_principal * daily_rate * input.overdue_days
    total_due = input.overdue_principal + penalty

    return {
        "overdue_principal": input.overdue_principal,
        "annual_rate": input.annual_rate,
        "penalty_multiplier": multiplier,
        "overdue_days": input.overdue_days,
        "daily_rate_percent": round(daily_rate * 100, 6),  # 日利率（%）
        "penalty": round(penalty, 2),
        "total_due": round(total_due, 2),
        "formula": f"罚息 = {input.overdue_principal} × ({input.annual_rate}% × {multiplier} / 360) × {input.overdue_days} = {round(penalty, 2)} 元",
        "disclaimer": "罚息计算仅供参考，实际以银行合同为准。逾期将影响征信，请尽快还款。"
    }
