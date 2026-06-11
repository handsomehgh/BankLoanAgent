# author hgh
# version 1.0
from typing import Annotated, Any
from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field

from infra.data_model.loan_interest import LoanInterest
from modules.agent.constants import AgentName
from modules.tools.error_handler import with_tool_error_handling


class QueryLoanInterestInput(BaseModel):
    """查询贷款意向输入"""
    loan_type: str = Field(..., description="贷款类型：住房贷款/消费贷款/经营贷款")


@tool(
    "query_loan_interest",
    description="根据用户ID和贷款类型，查询用户是否已登记该类型的贷款意向。返回意向的当前状态及详情，包括申请编号、金额、期限、还款方式等，供LLM根据状态生成相应回复（如“处理中”则告知等待，“已处理”则询问是否重新提交）",
    args_schema=QueryLoanInterestInput,
    extras={"version": "1.0.0", "tags": [AgentName.LOAN_ADVISOR.value]}
)
@with_tool_error_handling
def query_loan_interest(
    input: QueryLoanInterestInput,
    user_id: Annotated[str, InjectedToolArg],
    db_session: Annotated[Any, InjectedToolArg]
) -> dict:
    """查询用户对指定贷款类型的意向记录"""
    record = db_session.query(LoanInterest).filter(
        LoanInterest.user_id == user_id,
        LoanInterest.loan_type == input.loan_type
    ).first()

    if not record:
        return {
            "signal": "not_found",
            "message": f"您目前还没有登记过{input.loan_type}的意向。"
        }

    return {
        "signal": "found",
        "application_no": record.application_no,
        "loan_type": record.loan_type,
        "desired_amount": record.desired_amount,
        "term_years": record.term_years,
        "repayment_method": record.repayment_method or "未指定",
        "loan_purpose": record.loan_purpose or "未指定",
        "status": record.status,
        "urgency": record.urgency,
        "created_at": record.created_at.isoformat() if record.created_at else None
    }
