# author hgh
# version 1.0
from enum import Enum

class ProfileEntityKey(str, Enum):
    # ========== 核心授信维度 ==========
    INCOME = "income"                             # 年收入或月收入信息
    OCCUPATION = "occupation"                     # 职业、职位、行业信息
    LOAN_PURPOSE = "loan_purpose"                 # 贷款用途
    LOAN_AMOUNT = "loan_amount"                   # 期望贷款金额
    CREDIT_SCORE = "credit_score"                 # 信用评分或征信状况（含逾期记录）
    LIABILITY = "liability"                       # 负债情况
    LOAN_TERM = "loan_term"                       # 期望贷款期限
    MARITAL_STATUS = "marital_status"             # 婚姻状况
    DEPENDENTS = "dependents"                     # 供养人数
    WORK_YEARS = "work_years"                     # 工作年限
    ASSET = "asset"                               # 资产状况
    EXISTING_BANK_RELATIONSHIP = "existing_bank_relationship"  # 本行业务关系
    LOAN_EXPERIENCE = "loan_experience"           # 贷款经历

    # ========== 补充信息维度 ==========
    CONTACT = "contact"                           # 联系方式
    HOUSEHOLD_REGISTRATION = "household_registration"  # 户籍所在地
    INSURANCE_STATUS = "insurance_status"         # 社保/公积金缴纳情况
    EDUCATION = "education"                       # 最高学历
    RESIDENCE_TYPE = "residence_type"             # 居住情况

    @classmethod
    def to_list(cls) -> list:
        return [e.value for e in cls]

class EvidenceType(str, Enum):
    EXPLICIT_STATEMENT = "explicit_statement"
    BANK_STATEMENT = "bank_statement"
    CREDIT_REPORT = "credit_report"
    TAX_DOCUMENT = "tax_document"
    INFERRED = "inferred"

class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"
    DELETE = "delete"

class MemorySource(str, Enum):
    CHAT_EXTRACTION = "chat_extraction"
    EXPLICIT_CORRECTION = "explicit_correction"
    BANK_STATEMENT = "bank_statement"
    CREDIT_REPORT = "credit_report"
    TAX_DOCUMENT = "tax_document"
    FORM_SUBMISSION = "form_submission"
    AUTO_SUMMARY = "auto_summary"
    ADMIN_IMPORT = "admin_import"

class InteractionEventType(str, Enum):
    INQUIRY = "inquiry"
    COMPLAINT = "complaint"
    APPLICATION_STARTED = "application_started"
    APPLICATION_COMPLETED = "application_completed"
    HANDOFF_REQUESTED = "handoff_requested"
    FEEDBACK = "feedback"

class InteractionSentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    ANXIOUS = "anxious"
    FRUSTRATED = "frustrated"




