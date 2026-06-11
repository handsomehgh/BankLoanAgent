# author hgh
# version 1.0
from datetime import datetime
import uuid
from typing import Annotated, Any, Optional
from langchain_core.tools import tool, InjectedToolArg
from pydantic import BaseModel, Field
from sqlalchemy.orm import session

from infra.data_model.loan_interest import LoanInterest
from modules.agent.constants import AgentName
from modules.tools.error_handler import with_tool_error_handling


class UpsertLoanInterestInput(BaseModel):
    """提交或更新贷款意向输入"""
    loan_type: str = Field(..., description="贷款类型：住房贷款/消费贷款/经营贷款")
    desired_amount: float = Field(..., description="期望贷款金额（元）")
    term_years: int = Field(..., description="期望贷款期限（年）")
    loan_purpose: Optional[str] = Field(None, description="贷款用途描述")
    repayment_method: Optional[str] = Field(None, description="还款方式：等额本息/等额本金")
    confirmed: bool = Field(False, description="用户是否确认修改（处理中/已处理状态时需要）")


@tool(
    "upsert_loan_interest",
    description="统一处理贷款意向的新建、更新和状态变更。根据是否存在已有记录及其当前状态，自动执行对应的业务逻辑——包括信息完整性校验、新建记录、更新记录、确认提示、重新激活已取消记录等。通过数据库唯一索引保证同一用户同一类型仅保留一条记录",
    args_schema=UpsertLoanInterestInput,
    extras={"version": "1.0.0", "tags": [AgentName.LOAN_ADVISOR.value]}
)
@with_tool_error_handling
def upsert_loan_interest(
        input: UpsertLoanInterestInput,
        user_id: Annotated[str, InjectedToolArg],
        conversation_summary: Annotated[str, InjectedToolArg],
        profile_summary: Annotated[str, InjectedToolArg],
        trace_id: Annotated[str, InjectedToolArg],
        db_session: Annotated[session, InjectedToolArg]
) -> dict:
    """提交或更新贷款意向，内部处理新建、更新、确认逻辑"""

    # 1. 查询已有记录
    existing = db_session.query(LoanInterest).filter(
        LoanInterest.user_id == user_id,
        LoanInterest.loan_type == input.loan_type
    ).first()

    # 2. 无记录 → 新建
    if not existing:
        application_no = _generate_application_no()
        record = LoanInterest(
            application_no=application_no,
            user_id=user_id,
            loan_type=input.loan_type,
            desired_amount=input.desired_amount,
            term_years=input.term_years,
            loan_purpose=input.loan_purpose,
            repayment_method=input.repayment_method,
            conversation_summary=conversation_summary,
            profile_summary=profile_summary,
            trace_id=trace_id,
            status='待处理',
            urgency=0
        )
        db_session.add(record)
        db_session.commit()
        return {
            "signal": "created",
            "application_no": application_no,
            "status": "待处理",
            "message": "您的贷款意向已登记，客户经理将在1个工作日内与您联系。",
            "details": {
                "loan_type": input.loan_type,
                "desired_amount": input.desired_amount,
                "term_years": input.term_years,
                "repayment_method": input.repayment_method or "未指定"
            }
        }

    # 3. 有记录，根据状态决策
    if existing.status in ('待处理',):
        # 直接更新
        existing.desired_amount = input.desired_amount
        existing.term_years = input.term_years
        existing.loan_purpose = input.loan_purpose
        existing.repayment_method = input.repayment_method
        existing.conversation_summary = conversation_summary
        existing.profile_summary = profile_summary
        db_session.commit()
        return {
            "signal": "updated",
            "application_no": existing.application_no,
            "status": existing.status,
            "message": "您的贷款意向已更新。",
            "details": {
                "loan_type": input.loan_type,
                "desired_amount": input.desired_amount,
                "term_years": input.term_years,
                "repayment_method": input.repayment_method or "未指定"
            }
        }

    elif existing.status == '处理中':
        if not input.confirmed:
            return {
                "signal": "need_confirm",
                "application_no": existing.application_no,
                "current_status": "处理中",
                "message": "您的意向已有工作人员在处理中，确认修改后可能需要重新审核。是否继续？"
            }
        # 确认后更新并重置状态
        existing.desired_amount = input.desired_amount
        existing.term_years = input.term_years
        existing.loan_purpose = input.loan_purpose
        existing.repayment_method = input.repayment_method
        existing.conversation_summary = conversation_summary
        existing.profile_summary = profile_summary
        existing.status = '待处理'
        existing.status_updated_at = datetime.now()
        db_session.commit()
        return {
            "signal": "updated",
            "application_no": existing.application_no,
            "status": "待处理",
            "message": "您的意向已更新，将重新进入处理流程。",
            "details": {
                "loan_type": input.loan_type,
                "desired_amount": input.desired_amount,
                "term_years": input.term_years,
                "repayment_method": input.repayment_method or "未指定"
            }
        }

    elif existing.status == '已处理':
        if not input.confirmed:
            return {
                "signal": "need_confirm",
                "application_no": existing.application_no,
                "current_status": "已处理",
                "message": "您之前的意向已处理完毕，是否重新提交？"
            }
        existing.desired_amount = input.desired_amount
        existing.term_years = input.term_years
        existing.loan_purpose = input.loan_purpose
        existing.repayment_method = input.repayment_method
        existing.conversation_summary = conversation_summary
        existing.profile_summary = profile_summary
        existing.status = '待处理'
        existing.status_updated_at = datetime.now()
        db_session.commit()
        return {
            "signal": "updated",
            "application_no": existing.application_no,
            "status": "待处理",
            "message": "您的意向已重新提交。",
            "details": {
                "loan_type": input.loan_type,
                "desired_amount": input.desired_amount,
                "term_years": input.term_years,
                "repayment_method": input.repayment_method or "未指定"
            }
        }

    elif existing.status == '已取消':
        # 自动重新激活
        existing.desired_amount = input.desired_amount
        existing.term_years = input.term_years
        existing.loan_purpose = input.loan_purpose
        existing.repayment_method = input.repayment_method
        existing.conversation_summary = conversation_summary
        existing.profile_summary = profile_summary
        existing.status = '待处理'
        existing.status_updated_at = datetime.now()
        db_session.commit()
        return {
            "signal": "reactivated",
            "application_no": existing.application_no,
            "status": "待处理",
            "message": "已重新激活您之前的贷款意向。",
            "details": {
                "loan_type": input.loan_type,
                "desired_amount": input.desired_amount,
                "term_years": input.term_years,
                "repayment_method": input.repayment_method or "未指定"
            }
        }


def _generate_application_no() -> str:
    """生成意向编号"""
    date_str = datetime.now().strftime("%Y%m%d")
    seq = str(uuid.uuid4().int)[:6]
    return f"LON{date_str}{seq}"
