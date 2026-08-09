# author hgh
# version 1.0
from datetime import datetime
from typing import Optional
from sqlalchemy.orm import Session
from infra.data_model.loan_interest import LoanInterest


class LoanInterestRepository:

    def __init__(self, db_session: Session):
        self._session = db_session

    # ==================== 查询 ====================

    def find_by_user_and_type(self, user_id: str, loan_type: str) -> Optional[LoanInterest]:
        """根据用户ID和贷款类型查询已有意向记录"""
        return self._session.query(LoanInterest).filter_by(
            user_id=user_id,
            loan_type=loan_type
        ).first()

    def find_by_user(self, user_id: str) -> list:
        """查询用户名下全部意向记录（供主动邀请gate做重复登记检查）"""
        return self._session.query(LoanInterest).filter_by(user_id=user_id).all()

    def find_by_application_no(self, application_no: str) -> Optional[LoanInterest]:
        """根据意向编号查询"""
        return self._session.query(LoanInterest).filter_by(
            application_no=application_no
        ).first()

    # ==================== 写入 ====================

    def create(self, record: LoanInterest) -> LoanInterest:
        self._session.add(record)
        self._session.flush()
        return record

    def update(self, record: LoanInterest, **kwargs) -> None:
        """更新意向记录的字段"""
        for field, value in kwargs.items():
            if hasattr(record, field):
                setattr(record, field, value)
        self._session.flush()

    def update_status(self, record: LoanInterest, new_status: str) -> None:
        """更新意向状态"""
        record.status = new_status
        record.status_updated_at = datetime.now()
        self._session.flush()

    def mark_urgent(self, record: LoanInterest) -> None:
        """标记为加急"""
        record.urgency = 1
        record.last_urged_at = datetime.now()
        self._session.flush()

    # ==================== 删除（软删除） ====================

    def cancel(self, record: LoanInterest) -> None:
        """取消意向（软删除，修改状态）"""
        record.status = '已取消'
        record.status_updated_at = datetime.now()
        self._session.flush()