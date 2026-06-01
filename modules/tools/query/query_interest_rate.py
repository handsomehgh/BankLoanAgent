# author hgh
# version 1.0
import logging
from typing import Optional, Annotated

from pydantic import BaseModel, Field

from config.models.bank_global_config import BankGlobalConfig
from modules.agent.constants import AgentName
from modules.module_services.lpr_data_service import LPRDataService
from langchain_core.tools import tool, InjectedToolArg

from modules.tools.tool_constatnt import LoanProductType

logger = logging.getLogger(__name__)


class QueryInterestRateInput(BaseModel):
    product_type: LoanProductType = Field(..., description="贷款类型：住房贷款、消费贷款、经营贷款")
    term_years: int = Field(..., ge=1, le=50, description="贷款期限（年）")
    is_first_house: Optional[bool] = Field(False, description="是否首套房（仅住房贷款类型需要）")


@tool(
    "query_interest_rate",
    description="查询指定贷款产品和期限的最新参考利率区间（基于实时LPR和银行政策）。住房贷款需提供is_first_house参数。",
    args_schema=QueryInterestRateInput,
    extras={"version": "1.0.0", "tags": [AgentName.LOAN_ADVISOR.value]}
)
def query_interest_rate(
    input: QueryInterestRateInput,
    lpr_service: Annotated[LPRDataService, InjectedToolArg],
    bank_global_config: Annotated[BankGlobalConfig, InjectedToolArg],
) -> dict:
    # 1. 获取实时LPR
    lpr_data = lpr_service.get_latest_lpr()
    if not lpr_data or "lpr_1y" not in lpr_data or "lpr_5y" not in lpr_data:
        logger.error("LPR数据不可用")
        return {"error": "利率数据暂时不可用，请稍后再试"}

    lpr = lpr_data["lpr_5y"] if input.term_years > 5 else lpr_data["lpr_1y"]

    # 2. 匹配产品政策
    product_policy = None
    for item in bank_global_config.product_point:
        if item.product_type == input.product_type.value:
            product_policy = item
            break
    if not product_policy:
        return {"error": f"未找到产品政策: {input.product_type.value}"}

    # 3. 调整加点区间（二套房）
    min_diff = product_policy.min_diff
    max_diff = product_policy.max_diff
    notes = product_policy.notes
    if input.product_type == LoanProductType.HOUSING_LOAN and input.is_first_house is not None:
        if not input.is_first_house:
            min_diff = max(min_diff, 0.60)
            notes += "。二套房利率不低于LPR+60BP"

    # 4. 计算利率区间
    min_rate = lpr + min_diff
    max_rate = lpr + max_diff

    return {
        "lpr": f"{lpr}%",
        "lpr_value": lpr,
        "min_rate": f"{min_rate:.2f}%",
        "min_rate_value": round(min_rate, 4),
        "max_rate": f"{max_rate:.2f}%",
        "max_rate_value": round(max_rate, 4),
        "notes": notes,
        "source": lpr_data.get("source", ""),
        "disclaimer": "以上利率仅供参考，具体以银行实际审批为准。"
    }
