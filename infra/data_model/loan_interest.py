# author hgh
# version 1.0
from sqlalchemy import Column, BigInteger, String, Text, JSON, DateTime, Numeric, Integer, SmallInteger, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

from sqlalchemy.sql.ddl import CreateTable, CreateIndex

Base = declarative_base()

class LoanInterest(Base):
    __tablename__ = 'loan_interests'

    # ========== 主键与业务标识 ==========
    id                  = Column(BigInteger, primary_key=True, autoincrement=True, comment='物理主键')
    application_no      = Column(String(32), nullable=False, unique=True, comment='意向编号，系统自动生成，如 LON20260610001')

    # ========== 用户标识（仅关联，不存敏感信息） ==========
    user_id             = Column(String(64), nullable=False, comment='用户唯一标识，关联认证系统与记忆系统')

    # ========== 意向核心信息 ==========
    loan_type           = Column(String(32), nullable=False, comment='贷款类型：住房贷款/消费贷款/经营贷款')
    loan_purpose        = Column(String(128), comment='贷款用途描述，从对话中提取')
    desired_amount      = Column(Numeric(12,2), comment='期望贷款金额（元）')
    term_years          = Column(Integer, comment='期望贷款期限（年）')
    repayment_method    = Column(String(16), comment='期望还款方式：等额本息/等额本金')
    preferred_contact_period = Column(String(32),comment="期望联系时间段")
    contact_time_note = Column(String(64),comment="联系时间备注")

    # ========== 需求场景上下文 ==========
    conversation_summary = Column(Text, comment='用户表达意向时的对话摘要（LLM生成）')
    profile_summary      = Column(JSON, comment='提交时的用户画像快照（月收入、负债、征信等，不含手机号等敏感字段）')
    trace_id             = Column(String(64), comment='关联的对话会话ID，用于审计回溯')

    # ========== 状态机 ==========
    status              = Column(String(16), nullable=False, default='待处理', comment='意向状态：待处理/处理中/已处理/已取消')
    status_updated_at   = Column(DateTime, comment='状态最后更新时间')

    # ========== 紧急程度（用户可设置） ==========
    urgency             = Column(SmallInteger, default=0, comment='紧急程度：0-普通，1-加急')
    last_urged_at       = Column(DateTime, comment='最近一次催促时间')

    # ========== 下游认领（下游系统写入，当前Agent不交互） ==========
    claimed_by          = Column(String(64), comment='认领人邮箱或工号')
    claimed_at          = Column(DateTime, comment='认领时间')

    # ========== 审计字段 ==========
    created_at          = Column(DateTime, nullable=False, default=datetime.utcnow, comment='创建时间')
    updated_at          = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow, comment='最后更新时间')

    # ========== 索引定义 ==========
    __table_args__ = (
        # user_id + loan_type 联合唯一索引（业务幂等 + 数据幂等）
        Index('uk_user_id_loan_type', 'user_id', 'loan_type', unique=True),
        {'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4', 'comment': '贷款意向登记表'}
    )

if __name__ == '__main__':
    print(CreateTable(LoanInterest.__table__))

    # 生成索引创建 SQL
    for index in LoanInterest.__table__.indexes:
        print(CreateIndex(index))