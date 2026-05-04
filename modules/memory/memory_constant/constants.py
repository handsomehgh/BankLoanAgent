# author hgh
# version 1.0
from enum import Enum

class ProfileEntityKey(str, Enum):
    INCOME = "income"
    OCCUPATION = "occupation"
    LOAN_PURPOSE = "loan_purpose"
    LOAN_AMOUNT = "loan_amount"
    CREDIT_SCORE = "credit_score"
    HAS_OVERDUE_HISTORY = "has_overdue_history"
    LIABILITY = "liability"
    CONTACT = "contact"
    WORK_YEARS = "work_years"
    HOUSEHOLD_REGISTRATION = "household_registration"
    INDUSTRY_TYPE = "industry_type"
    LOAN_EXPERIENCE = "loan_experience"
    LOAN_TERM = "loan_term"
    MARITAL_STATUS = "marital_status"
    DEPENDENTS = "dependents"
    ASSET = "asset"
    EXISTING_BANK_RELATIONSHIP = "existing_bank_relationship"
    PREFERENCE = "preference"
    INSURANCE_STATUS = "insurance_status"
    EDUCATION = "education"
    RESIDENCE_TYPE = "residence_type"

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

class ChromaOperator(str, Enum):
    AND = "$and"
    EQ = "$eq"
    GTE = "$gte"

