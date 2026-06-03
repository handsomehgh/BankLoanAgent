# author hgh
# version 1.0
"""
early repayment trial calculation tool
"""
from math import log
from typing import Annotated, Optional

from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field

from config.models.bank_global_config import BankGlobalConfig
from exceptions.exception import ToolExecutionException
from modules.agent.constants import AgentName
from modules.module_services.lpr_data_service import LPRDataService
from modules.tools.base_tool import ToolErrorType
from modules.tools.error_handler import with_tool_error_handling
from modules.tools.tool_constatnt import RepaymentMethod, PrepaymentMethod


class CalculatePrepaymentInput(BaseModel):
    remaining_principle: float = Field(..., description="剩余本金(元)")
    annual_rate: Optional[float] = Field(None, gt=0, le=50, description="年利率，不填则自动使用当前LPR")
    total_months: int = Field(..., gt=0, le=120, description="贷款总期数(月)")
    paid_month: int = Field(..., gt=0, le=120, description="已还期数(月)")
    prepay_amount: float = Field(..., ge=0, description="提前还款金额（元），0 或默认等于剩余本金（全部结清）")
    method: RepaymentMethod = Field(default=RepaymentMethod.EQUAL_INSTALLMENT,
                                    description="还款方式：等额本息或等额本金")
    penalty_rate_override: float = Field(None, ge=0, le=1,
                                         description="手动指定违约金比例（如 0.01），None 则按规则自动计算")
    option: PrepaymentMethod = Field(PrepaymentMethod.COMPARE,
                                     description="提前还款方案：shorten_term(缩短期限)、reduce_payment(减少月供)、compare(对比)")


@tool(
    "calculate_prepayment",
    description="提前还款试算：根据剩余本金、利率、已还期数和提前还款金额，计算缩短期限或减少月供两种方案的节省利息和违约金。"
                "返回各方案的节省利息、违约金、净节省及新月供/新期限。支持全部结清或部分还款。",
    args_schema=CalculatePrepaymentInput,
    extras={"version": "1.0.0", "tags": [AgentName.AFTER_LOAN.value]}
)
@with_tool_error_handling
def calculate_prepayment(
        input: CalculatePrepaymentInput,
        bank_config: Annotated[BankGlobalConfig, InjectedToolArg],
        lpr_service: Annotated[LPRDataService, InjectedToolArg]
) -> dict:
    """提前还款试算主函数"""
    # 1. 参数预处理
    if input.paid_months >= input.total_months:
        raise ToolExecutionException(f"已还期数不能大于等于总期数",ToolErrorType.PARAMETER_ERROR)

    if input.prepay_amount > input.remaining_principal:
        raise ToolExecutionException(f"提前还款金额不能超过剩余本金", ToolErrorType.PARAMETER_ERROR)

    if input.prepay_amount == 0:
        input.prepay_amount = input.remaining_principal

    if input.annual_rate is None:
        lpr_data = lpr_service.get_latest_lpr()
        input.annual_rate = lpr_data["lpr_5y"]

    monthly_rate = input.annual_rate / 100 / 12
    prepay_cfg = bank_config.prepayment

    # 2. 计算违约金
    penalty_rate = _calc_penalty_rate(
        input.paid_months, prepay_cfg, input.penalty_rate_override
    )
    penalty = round(input.prepay_amount * penalty_rate, 2)

    # 3. 根据选项计算
    if input.option == PrepaymentMethod.COMPARE:
        shorten_result = _calc_shorten_term(
            input.remaining_principal, monthly_rate, input.total_months,
            input.paid_months, input.prepay_amount, input.method.value
        )
        reduce_result = _calc_reduce_payment(
            input.remaining_principal, monthly_rate, input.total_months,
            input.paid_months, input.prepay_amount, input.method.value
        )
        return {
            "penalty": penalty,
            "penalty_rate_used": penalty_rate,
            "compare": {
                "shorten_term": {
                    **shorten_result,
                    "net_saving": round(shorten_result["saved_interest"] - penalty, 2)
                },
                "reduce_payment": {
                    **reduce_result,
                    "net_saving": round(reduce_result["saved_interest"] - penalty, 2)
                }
            },
            "disclaimer": "计算结果仅供参考，实际以银行合同为准。"
        }
    elif input.option == PrepaymentMethod.SHORTEN_TERM:
        result = _calc_shorten_term(
            input.remaining_principal, monthly_rate, input.total_months,
            input.paid_months, input.prepay_amount, input.method.value
        )
    else:  # REDUCE_PAYMENT
        result = _calc_reduce_payment(
            input.remaining_principal, monthly_rate, input.total_months,
            input.paid_months, input.prepay_amount, input.method.value
        )

    return {
        "option": input.option.value,
        **result,
        "penalty": penalty,
        "penalty_rate_used": penalty_rate,
        "net_saving": round(result["saved_interest"] - penalty, 2),
        "disclaimer": "计算结果仅供参考，实际以银行合同为准。"
    }


# ---------- 辅助计算函数 ----------
def _calc_penalty_rate(
        paid_months: int,
        prepay_cfg,
        override: Optional[float]
) -> float:
    """根据已还期数和配置计算违约金比例"""
    if override is not None:
        return override
    if paid_months >= prepay_cfg.free_after_months:
        return 0.0
    return prepay_cfg.penalty_rate


def _calc_shorten_term(
        principal: float,
        monthly_rate: float,
        total_months: int,
        paid_months: int,
        prepay_amount: float,
        method: str
) -> dict:
    """
    缩短期限方案：月供不变，计算新期限、节省利息等。
    说明：由于期限需为整数月，新期限向上取整，实际最后一期会有少量调整。
    """
    new_principal = principal - prepay_amount
    if new_principal <= 0:
        # 全部结清
        return {
            "new_principal": 0.0,
            "new_term_months": 0,
            "new_monthly_payment": 0.0,
            "original_remaining_interest": _calc_remaining_interest(
                principal, monthly_rate, total_months - paid_months, method
            ),
            "saved_interest": _calc_remaining_interest(
                principal, monthly_rate, total_months - paid_months, method
            )
        }

    remaining_months = total_months - paid_months
    # 原方案剩余总利息
    original_interest = _calc_remaining_interest(principal, monthly_rate, remaining_months, method)

    if method == "等额本息":
        # 首先计算原月供（基于当前剩余本金和剩余期限）
        if monthly_rate == 0:
            monthly_payment = principal / remaining_months
        else:
            factor = (1 + monthly_rate) ** remaining_months
            monthly_payment = principal * monthly_rate * factor / (factor - 1)
        # 保持月供不变，反算新期限（精确月数）
        if monthly_rate == 0 or monthly_payment <= new_principal * monthly_rate:
            # 极端情况，直接设为1期
            new_term = 1
        else:
            new_term = log(monthly_payment / (monthly_payment - new_principal * monthly_rate)) / log(1 + monthly_rate)
        new_term = max(1, int(new_term) + (1 if new_term % 1 > 0 else 0))  # 向上取整
        # 新方案总利息：新期限内所有月供之和 - 新本金
        new_interest = monthly_payment * new_term - new_principal
        saved_interest = original_interest - new_interest
        return {
            "new_principal": round(new_principal, 2),
            "new_term_months": new_term,
            "new_monthly_payment": round(monthly_payment, 2),
            "original_remaining_interest": round(original_interest, 2),
            "saved_interest": round(saved_interest, 2)
        }
    else:  # 等额本金
        # 原月供计算方式不同，缩短期限需保持每月本金不变？等额本金下缩短期限需调整月供还是保持月供不变？
        # 常见做法：等额本金缩短期限，仍保持每月偿还相同本金额，但期限变短导致每月本金增加。
        # 为简化，等额本金暂不支持缩短期限方案，可返回错误提示或降级为减少月供逻辑。
        raise ToolExecutionException(f"等额本金暂不支持缩短期限试算，请使用减少月供方案或咨询客户经理",ToolErrorType.PARAMETER_ERROR)


def _calc_reduce_payment(
        principal: float,
        monthly_rate: float,
        total_months: int,
        paid_months: int,
        prepay_amount: float,
        method: str
) -> dict:
    """
    减少月供方案：期限不变，重新计算新月供。
    """
    new_principal = principal - prepay_amount
    if new_principal <= 0:
        return {
            "new_principal": 0.0,
            "new_term_months": 0,
            "new_monthly_payment": 0.0,
            "original_remaining_interest": _calc_remaining_interest(
                principal, monthly_rate, total_months - paid_months, method
            ),
            "saved_interest": _calc_remaining_interest(
                principal, monthly_rate, total_months - paid_months, method
            )
        }

    remaining_months = total_months - paid_months
    original_interest = _calc_remaining_interest(principal, monthly_rate, remaining_months, method)

    if method == "等额本息":
        if monthly_rate == 0:
            new_monthly = new_principal / remaining_months
            new_interest = 0.0
        else:
            factor = (1 + monthly_rate) ** remaining_months
            new_monthly = new_principal * monthly_rate * factor / (factor - 1)
            new_interest = new_monthly * remaining_months - new_principal
    else:  # 等额本金
        # 等额本金下，月供 = 每月固定本金 + 剩余本金×月利率，提前还款后每月本金减少（期限不变）
        monthly_principal_new = new_principal / remaining_months
        # 首月月供
        new_monthly = monthly_principal_new + new_principal * monthly_rate
        # 总利息简化计算
        total_interest = 0.0
        rem = new_principal
        for _ in range(remaining_months):
            interest = rem * monthly_rate
            total_interest += interest
            rem -= monthly_principal_new
        new_interest = total_interest

    saved_interest = original_interest - new_interest
    return {
        "new_principal": round(new_principal, 2),
        "new_term_months": remaining_months,
        "new_monthly_payment": round(new_monthly, 2),
        "original_remaining_interest": round(original_interest, 2),
        "saved_interest": round(saved_interest, 2)
    }


def _calc_remaining_interest(principal: float, monthly_rate: float, months: int, method: str) -> float:
    """计算剩余期限的总利息（不提前还款的情况）"""
    if method == "等额本息":
        if monthly_rate == 0:
            return 0.0
        factor = (1 + monthly_rate) ** months
        monthly = principal * monthly_rate * factor / (factor - 1)
        return monthly * months - principal
    else:  # 等额本金
        interest = 0.0
        rem = principal
        monthly_principal = principal / months
        for _ in range(months):
            interest += rem * monthly_rate
            rem -= monthly_principal
        return interest
