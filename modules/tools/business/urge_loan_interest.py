# author hgh
# version 1.0
from datetime import datetime
from typing import Annotated, Any
from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field

from infra.data_model.loan_interest import LoanInterest
from modules.agent.constants import AgentName
from modules.tools.error_handler import with_tool_error_handling


class UrgeLoanInterestInput(BaseModel):
    """催促贷款意向输入"""
    loan_type: str = Field(..., description="贷款类型：住房贷款/消费贷款/经营贷款")


@tool(
    "urge_loan_interest",
    description="",
    args_schema=UrgeLoanInterestInput,
    extras={"version": "1.0.0", "tags": [AgentName.LOAN_ADVISOR.value]}
)
@with_tool_error_handling
def urge_loan_interest(
    input: UrgeLoanInterestInput,
    user_id: str,
    db_session: Annotated[Any, InjectedToolArg]
) -> dict:
    """标记意向为加急处理"""
    record = db_session.query(LoanInterest).filter(
        LoanInterest.user_id == user_id,
        LoanInterest.loan_type == input.loan_type
    ).first()

    if not record:
        return {
            "signal": "not_found",
            "message": f"您目前还没有登记过{input.loan_type}的意向。"
        }

    if record.status in ('待处理', '处理中'):
        record.urgency = 1
        record.last_urged_at = datetime.now()
        db_session.commit()
        return {
            "signal": "urged",
            "application_no": record.application_no,
            "status": record.status,
            "message": "已为您标记为加急处理，我们会优先处理您的意向。"
        }

    return {
        "signal": "status_blocked",
        "application_no": record.application_no,
        "status": record.status,
        "message": f"您的意向当前状态为「{record.status}」，无法进行催促操作。"
    }
