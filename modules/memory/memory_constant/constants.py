# author hgh
# version 1.0
from enum import Enum

class ProfileEntityKey(str, Enum):
    # ========== 核心身份 ==========
    AGE = "age"  # 年龄
    MARITAL_STATUS = "marital_status"  # 婚姻状况 (已婚/未婚/离异)
    EDUCATION = "education"  # 最高学历
    HOUSEHOLD_REGISTRATION = "household_registration"  # 户籍所在地
    DEPENDENTS = "dependents"  # 供养人数
    OCCUPATION = "occupation"  # 职业/行业/职位

    # ========== 财务健康 ==========
    ANNUAL_INCOME = "annual_income"  # 年收入
    MONTHLY_INCOME = "monthly_income"  # 月收入 (其实bank更看重月收入)
    INCOME_SOURCE = "income_source"  # 收入来源 (工资/经营/投资)
    CREDIT_HISTORY = "credit_history"  # 征信简况 (良好/有逾期记录/白户)
    CREDIT_OVERDUE_DETAIL = "credit_overdue_detail"  # 逾期详情 (如"连三累六")
    LIABILITY_AMOUNT = "liability_amount"  # 总负债金额
    LIABILITY_DETAIL = "liability_detail"  # 负债构成 (房贷/车贷/信用卡)
    MONTHLY_DEBT = "monthly_debt"  # 月还款额
    DTI_RATIO = "dti_ratio"  # 负债率 (可计算得出，但提取也接受)

    # ========== 资产与担保 ==========
    REAL_ESTATE = "real_estate"  # 房产 (套数/位置/状态)
    VEHICLE = "vehicle"  # 车辆
    DEPOSIT = "deposit"  # 存款/理财/股票
    INSURANCE = "insurance"  # 保单 (现金价值)
    SOCIAL_SECURITY = "social_security"  # 社保/医保 (年限)
    HOUSING_FUND = "housing_fund"  # 公积金 (缴存/余额/年限)

    # ========== 贷款关联 ==========
    LOAN_PURPOSE = "loan_purpose"  # 贷款用途
    DESIRED_AMOUNT = "desired_amount"  # 期望贷款金额
    LOAN_TERM = "loan_term"  # 期望期限
    EXISTING_BANK_RELATION = "existing_bank_relation"  # 本行业务关系
    LOAN_EXPERIENCE = "loan_experience"  # 贷款经历

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




