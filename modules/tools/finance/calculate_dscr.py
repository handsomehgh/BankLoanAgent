# author hgh
# version 1.0
"""
debt service coverage ration calculate tool
dscr = annual operating net income / annual principal and interest payments.
"""
import logging
from typing import Annotated

from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field

from config.models.bank_global_config import BankGlobalConfig
from modules.agent.constants import AgentName

logger = logging.getLogger(__name__)


class CalculateDscrInput(BaseModel):
    """偿债覆盖率计算输入"""
    annual_net_income: float = Field(..., gt=0, description="年经营净收入（元）")
    annual_debt_service: float = Field(..., gt=0, description="年还本付息额（元）")


@tool(
    "calculate_dscr",
    description="计算偿债覆盖率 (DSCR) = 年经营净收入 / 年还本付息额。用于评估经营贷款的还款能力。"
                "返回 DSCR 值、是否达标及建议。",
    args_schema=CalculateDscrInput,
    extras={"version": "1.0.0", "tags": [AgentName.RISK_ASSESSMENT.value]}
)
def calculate_dscr(
    input: CalculateDscrInput,
    bank_config: Annotated[BankGlobalConfig, InjectedToolArg],
) -> dict:
    """主函数"""
    dscr = input.annual_net_income / input.annual_debt_service
    threshold = bank_config.dscr_threshold
    compliant = dscr >= threshold

    if compliant:
        status = "达标"
        suggestion = f"DSCR 为 {dscr:.2f}，不低于最低要求 {threshold}，偿债能力良好。"
    else:
        status = "不达标"
        suggestion = f"DSCR 为 {dscr:.2f}，低于最低要求 {threshold}，建议提高收入或降低贷款额度。"

    return {
        "dscr": round(dscr, 4),
        "threshold": threshold,
        "compliant": compliant,
        "status": status,
        "suggestion": suggestion,
        "disclaimer": "计算结果仅供参考，最终以银行审批为准。"
    }
