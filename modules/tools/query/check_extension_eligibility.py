# author hgh
# version 1.0
"""
extension eligibility check tool
"""
from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field

from config.models.bank_global_config import BankGlobalConfig
from modules.agent.constants import AgentName


class CheckExtensionEligibilityInput(BaseModel):
    total_months: int = Field(..., gt=0, le=600, description="贷款总期数（月）")
    paid_months: int = Field(..., ge=0, le=600, description="已还期数（月）")
    overdue_count: int = Field(0, ge=0, description="历史逾期次数")
    has_current_overdue: bool = Field(False, description="是否有当前逾期")
    loan_type: str = Field("消费贷款", description="贷款类型，如：住房贷款、消费贷款、经营贷款")

@tool(
    "check_extension_eligibility",
    description="检查借款人是否具备申请贷款展期的资格。返回是否合格、原因、所需材料清单。",
    args_schema=CheckExtensionEligibilityInput,
    extras={"version": "1.0.0", "tags": [AgentName.AFTER_LOAN.value]}
)
def check_extension_eligibility(
    input: CheckExtensionEligibilityInput,
    bank_global_config: Annotated[BankGlobalConfig, InjectedToolArg],
) -> dict:
    rules = bank_global_config.extension
    reasons = []

    # 1. 检查贷款类型是否支持展期
    if input.loan_type not in rules.supported_loan_types:
        reasons.append(f"贷款类型 '{input.loan_type}' 不支持展期。支持的类型：{', '.join(rules.supported_loan_types)}")
        return {
            "eligible": False,
            "reasons": reasons,
            "required_materials": [],
            "suggestion": "请咨询客户经理了解其他解决方案。"
        }

    # 2. 检查已还期数是否满足最低还款月数
    if input.paid_months < rules.min_paid_months:
        reasons.append(f"正常还款月数不足：已还 {input.paid_months} 个月，要求至少 {rules.min_paid_months} 个月。")

    # 3. 检查是否有当前逾期
    if input.has_current_overdue:
        reasons.append("存在当前逾期，不符合展期条件。请先还清逾期欠款。")

    # 4. 检查逾期记录是否允许
    if not rules.allow_with_overdue and input.overdue_count > 0:
        reasons.append(f"存在历史逾期记录（{input.overdue_count}次），当前政策不允许展期。")

    # 5. 检查展期后期限是否超过上限
    max_total = int(input.total_months * (1 + rules.max_extension_ratio))
    # 此检查在申请时具体计算，此处只提示
    if input.paid_months >= max_total:
        reasons.append(f"已还期数过多，展期后总期限将超过上限（最长 {max_total} 个月）。")

    eligible = len(reasons) == 0

    return {
        "eligible": eligible,
        "reasons": reasons if not eligible else ["符合展期申请条件"],
        "required_materials": [
            "展期申请书",
            "身份证（原件+复印件）",
            "收入证明（近6个月银行流水）",
            "困难情况说明（如失业、疾病证明等）"
        ],
        "suggestion": "请携带上述材料到经办支行申请，最终以银行审批为准。" if eligible else "请先解决上述问题后再申请展期。"
    }
