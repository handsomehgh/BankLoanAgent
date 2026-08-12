# author hgh
# version 1.0
from typing import Annotated
from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field

from infra.database.mysql_manager import DatabaseManager
from infra.repository.LoanInterestRepository import LoanInterestRepository
from modules.agent.constants import AgentName
from modules.tools.error_handler import with_tool_error_handling


class UrgeLoanInterestInput(BaseModel):
    """催促贷款意向输入"""
    loan_type: str = Field(..., description="贷款类型：住房贷款/消费贷款/经营贷款")


@tool(
    "urge_loan_interest",
    description="催促加急处理已登记的贷款意向。根据用户ID和贷款类型将意向标记为加急，仅当意向状态为待处理/处理中时可催促。"
                "返回 signal（urged 已加急/not_found 无意向/status_blocked 状态不允许催促）及说明，供LLM据此生成回复。",
    args_schema=UrgeLoanInterestInput,
    extras={"version": "1.0.0", "tags": [AgentName.LOAN_ADVISOR.value]}
)
@with_tool_error_handling
def urge_loan_interest(
        input: UrgeLoanInterestInput,
        user_id: str,
        db_manager: Annotated[DatabaseManager, InjectedToolArg],
) -> dict:
    """标记意向为加急处理"""
    session = db_manager.create_session()

    try:
        repository = LoanInterestRepository(session)
        record = repository.find_by_user_and_type(user_id, input.loan_type)
        if not record:
            return {
                "signal": "not_found",
                "message": f"您目前还没有登记过{input.loan_type}的意向。"
            }

        if record.status in ('待处理', '处理中'):
            repository.mark_urgent(record)
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
    except Exception as e:
        if session:
            session.rollback()
        raise e
    finally:
        if session:
            session.close()
