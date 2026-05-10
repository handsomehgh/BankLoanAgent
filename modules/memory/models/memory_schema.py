import logging
from datetime import datetime
from typing import Optional, List, Any

from pydantic import Field, field_validator

from config.context_settings import get_enum_strictness
from config.global_constant.constants import ComplianceAction, ComplianceSeverity
from modules.memory.memory_constant.constants import MemorySource, EvidenceType, MemoryStatus, InteractionEventType, \
    InteractionSentiment, ProfileEntityKey
from modules.memory.models.memory_base import MemoryBase

logger = logging.getLogger(__name__)


# ==================== user profile ====================
class UserProfileMemory(MemoryBase):
    source: MemorySource = Field(default=MemorySource.CHAT_EXTRACTION, description="source of memory")
    entity_key: ProfileEntityKey = Field(..., description="entity type")
    evidence_type: EvidenceType = Field(default=EvidenceType.EXPLICIT_STATEMENT, description="evidence type")
    effective_date: datetime = Field(default_factory=datetime.now, description="effective date")
    expires_at: Optional[datetime] = Field(default=None, description="expiration time")
    superseded_by: Optional[str] = Field(default=None, description="replaced by which memory id")

    def supersede(self, new_id: str) -> None:
        self.status = MemoryStatus.SUPERSEDED
        self.superseded_by = new_id

    @field_validator("evidence_type", mode="before")
    @classmethod
    def validate_evidence_type(cls, v: Any) -> EvidenceType:
        if not isinstance(v, str):
            return v
        try:
            return EvidenceType(v)
        except ValueError:
            if get_enum_strictness():
                raise ValueError(f"Invalid evidence_type '{v}'. Must be one of {[e.value for e in EvidenceType]}")
            logger.warning(f"Invalid evidence_type '{v}', fallback to EXPLICIT_STATEMENT")
            return EvidenceType.EXPLICIT_STATEMENT


# ==================== interaction log ====================
class InteractionLogMemory(MemoryBase):
    source: MemorySource = Field(default=MemorySource.AUTO_SUMMARY, description="source of interaction log")
    event_type: InteractionEventType = Field(default=InteractionEventType.INQUIRY, description="event type")
    session_id: str = Field(..., description="connected session id")
    sentiment: Optional[InteractionSentiment] = Field(default=InteractionSentiment.NEUTRAL, description="emotion")
    key_entities: List[str] = Field(default_factory=list, description="list of key entities")
    timestamp: datetime = Field(default_factory=datetime.now, description="actual interaction time")

    @field_validator("event_type", mode="before")
    @classmethod
    def validate_event_type(cls, v: Any) -> InteractionEventType:
        if not isinstance(v, str):
            return v
        try:
            return InteractionEventType(v)
        except ValueError:
            if get_enum_strictness():
                raise ValueError(f"Invalid event_type '{v}'. Must be one of {[e.value for e in InteractionEventType]}")
            logger.warning(f"Invalid event_type '{v}', fallback to INQUIRY")
            return InteractionEventType.INQUIRY

    @field_validator("sentiment", mode="before")
    @classmethod
    def validate_sentiment(cls, v: Any) -> Optional[InteractionSentiment]:
        if v is None:
            return None
        if not isinstance(v, str):
            return v
        try:
            return InteractionSentiment(v)
        except ValueError:
            if get_enum_strictness():
                raise ValueError(f"Invalid sentiment '{v}'. Must be one of {[e.value for e in InteractionSentiment]}")
            logger.warning(f"Invalid sentiment '{v}', ignoring sentiment")
            return None


# ==================== compliance rule ====================
class ComplianceRuleMemory(MemoryBase):
    source: str = Field(..., description="source of compliance rule")
    rule_id: str = Field(..., description="unique rule id")
    rule_name: str = Field(..., description="rule name")
    rule_type: str = Field(..., description="type of rule")
    pattern: str = Field(..., description="regular expression")
    action: ComplianceAction = Field(..., description="action after trigger")
    severity: ComplianceSeverity = Field(default=ComplianceSeverity.MEDIUM, description="severity")
    priority: int = Field(default=100, ge=1, le=1000, description="priority")
    version: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"), description="version")
    effective_from: datetime = Field(default_factory=datetime.now, description="effective from")
    effective_to: Optional[datetime] = Field(default=None, description="effective to")
    template: Optional[str] = Field(default=None, description="template text for APPEND action")
    description: Optional[str] = Field(default=None, description="description text for rule")
    superseded_by: Optional[str] = Field(default=None, description="replaced by which rule id")

    def supersede(self, new_id: str) -> None:
        self.status = MemoryStatus.SUPERSEDED
        self.superseded_by = new_id

    @field_validator("action", mode="before")
    @classmethod
    def validate_action(cls, v: Any) -> ComplianceAction:
        if not isinstance(v, str):
            return v
        try:
            return ComplianceAction(v)
        except ValueError:
            if get_enum_strictness():
                raise ValueError(f"Invalid action '{v}'. Must be one of {[e.value for e in ComplianceAction]}")
            logger.warning(f"Invalid action '{v}', fallback to WARN")
            return ComplianceAction.WARN

    @field_validator("severity", mode="before")
    @classmethod
    def validate_severity(cls, v: Any) -> ComplianceSeverity:
        if not isinstance(v, str):
            return v
        try:
            return ComplianceSeverity(v)
        except ValueError:
            if get_enum_strictness():
                raise ValueError(f"Invalid severity '{v}'. Must be one of {[e.value for e in ComplianceSeverity]}")
            logger.warning(f"Invalid severity '{v}', fallback to MEDIUM")
            return ComplianceSeverity.MEDIUM
