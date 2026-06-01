# author hgh
# version 1.0
"""
Loan Settlement Certificate Generation Tool
"""
from datetime import datetime

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from modules.agent.constants import AgentName


class GenerateSettlementCertificateInput(BaseModel):
    user_name: str = Field(..., description="客户姓名")
    loan_id: str = Field(..., description="贷款编号")
    loan_type: str = Field(None, description="贷款类型（如住房贷款、消费贷款），可选")
    payoff_date: str = Field(None, description="结清日期，格式 YYYY-MM-DD，默认当天")
    branch_name: str = Field("本行", description="经办支行名称")

@tool(
    "generate_settlement_certificate",
    description="生成贷款结清证明文本模板。需要提供客户姓名、贷款编号，可选贷款类型、结清日期、经办支行。",
    args_schema=GenerateSettlementCertificateInput,
    extras={"version": "1.0.0", "tags": [AgentName.AFTER_LOAN.value]}
)
def generate_settlement_certificate(input: GenerateSettlementCertificateInput) -> dict:
    payoff_date = input.payoff_date or datetime.now().strftime("%Y-%m-%d")
    loan_type_str = input.loan_type + " " if input.loan_type else ""

    certificate = (
        f"贷款结清证明\n\n"
        f"兹证明客户{input.user_name}在我行办理的{loan_type_str}贷款（编号：{input.loan_id}）"
        f"已于{payoff_date}全额结清。\n"
        f"特此证明。\n\n"
        f"经办支行：{input.branch_name}\n"
        f"日期：{payoff_date}\n"
        f"（银行盖章有效）"
    )

    return {
        "certificate_text": certificate,
        "disclaimer": "此证明为模板，需加盖银行公章方为有效。请持此证明及身份证件前往网点办理解除抵押等后续手续。"
    }