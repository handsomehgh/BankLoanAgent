# author hgh
# version 1.0
"""
Debt_to_Income(DTI) calculation tool
calculate the debt_to_income ratio based on monthly income and various monthly debts
return the evaluation result according to bank approval standards
"""
import logging
from typing import List, Annotated, Optional

from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field

from config.models.bank_global_config import BankGlobalConfig
from exceptions.exception import ToolExecutionException
from modules.agent.constants import AgentName
from modules.tools.base_tool import ToolErrorType
from modules.tools.error_handler import with_tool_error_handling
from modules.tools.tool_constatnt import LoanProductType

logger = logging.getLogger(__name__)


class CalculateDTIInput(BaseModel):
    monthly_income: float = Field(..., gt=0, description="月收入(元)")
    monthly_debts: List[float] = Field(default_factory=list,ge=0,
                                       description="各项月债务（元），如车贷、信用卡最低还款、其他贷款月供等")
    loan_type: Optional[LoanProductType] = Field(default=LoanProductType.CONSUMER_LOAN, description="贷款类型：住房贷款、消费贷款、经营贷款")


@tool(
    "calculate_dti",
    args_schema=CalculateDTIInput,
    description="计算负债率（DTI = 月债务总额 / 月收入），并根据贷款类型返回银行审批标准下的风险评估（安全/关注/超标）。"
                "返回字段：dti（负债率），total_debt（总月债务），monthly_income（月收入），loan_type（贷款类型），"
                "status（评估状态），message（说明），threshold_used（适用标准）",
    extras={"version": "1.0.0", "tags": [AgentName.RISK_ASSESSMENT.value]}
)
@with_tool_error_handling
def calculate_dti(input: CalculateDTIInput, bank_global_config: Annotated[BankGlobalConfig, InjectedToolArg]) -> dict:
    # parameter validate
    total_debt = sum(input.monthly_debts)

    # calculate dit
    dti = total_debt / input.monthly_income

    loan_type = input.loan_type.value if input.loan_type else LoanProductType.CONSUMER_LOAN.value
    product_dti = None
    for dti in bank_global_config.product_dti:
        if dti.product_type == loan_type:
            product_dti = dti
            break
    thresholds = product_dti

    safe = thresholds.safe
    warn = thresholds.warn
    max_limit = thresholds.max

    if dti <= safe:
        status = "安全"
        message = f"当前负债率符合{loan_type}审批要求"
    elif dti <= warn:
        status = "关注"
        message = f"负债率略高，接近{loan_type}上限（{max_limit * 100:.0f}%），可能影响部分贷款审批"
    else:
        status = "超标"
        message = f"负债率超过{loan_type}上限（{max_limit * 100:.0f}%），建议优先偿还部分债务以降低负债率"

    return {
        "dti": round(dti, 4),
        "dti_percent": f"{dti * 100:.2f}%",
        "total_debt": round(total_debt, 2),
        "monthly_income": input.monthly_income,
        "loan_type": loan_type,
        "status": status,
        "message": message,
        "threshold_used": f"安全≤{safe * 100:.0f}%，关注≤{warn * 100:.0f}%，上限{max_limit * 100:.0f}%",
        "disclaimer": "仅供参考，最终审批以银行内部政策为准"
    }
