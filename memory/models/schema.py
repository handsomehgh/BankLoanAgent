# author hgh
# version 1.0
from datetime import datetime
from typing import Dict, Any, Optional, List

from pydantic import Field, model_validator

from memory.constant.constants import MemoryType, ProfileEntityKey, EvidenceType, MetadataFields, InteractionEventType, \
    InteractionSentiment, ComplianceSeverity, ComplianceAction, ComplianceRuleFields
from memory.models.memory_meta import MemoryMetadata


class UserProfileMetadata(MemoryMetadata):
    memory_type: MemoryType = Field(MemoryType.USER_PROFILE, frozen=True)
    entity_key: ProfileEntityKey = Field(..., description="entity type")
    evidence_type: EvidenceType = Field(default=EvidenceType.EXPLICIT_STATEMENT, description="evidence type")
    effective_date: datetime = Field(default_factory=datetime.now, description="effective date")
    expires_at: datetime = Field(default=None, description="expires date")

    @model_validator(mode="after")
    def validate_memory_type(self) -> "UserProfileMetadata":
        if self.memory_type != MemoryType.USER_PROFILE:
            raise ValueError(f"UserProfileMetadata 要求 memory_type 为 {MemoryType.USER_PROFILE}")
        return self

    def to_chroma_dict(self) -> Dict[str, Any]:
        data = super().to_chroma_dict()
        data[MetadataFields.ENTITY_KEY.value] = self.entity_key.value
        data[MetadataFields.EVIDENCE_TYPE.value] = self.evidence_type.value
        data[MetadataFields.EFFECTIVE_DATE.value] = self.effective_date.isoformat()
        if self.expires_at:
            data[MetadataFields.EXPIRES_AT.value] = self.expires_at.isoformat()
        return data

    @classmethod
    def from_chrom_dict(cls, data: Dict[str, Any]) -> "UserProfileMetadata":
        base = MemoryMetadata.from_chrom_dict(data)
        return cls(
            **base.model_dump(exclude={MetadataFields.MEMORY_TYPE.value, MetadataFields.EXTRA.value}),
            entity_key=ProfileEntityKey(
                data[MetadataFields.ENTITY_KEY.value]) if MetadataFields.ENTITY_KEY.value in data else None,
            evidence_type=EvidenceType(
                data[MetadataFields.EVIDENCE_TYPE.value]) if MetadataFields.EVIDENCE_TYPE.value in data else None,
            effective_date=datetime.fromisoformat(data[
                                                      MetadataFields.EFFECTIVE_DATE.value]) if MetadataFields.EFFECTIVE_DATE.value in data else datetime.now(),
            expires_at=datetime.fromisoformat(data[MetadataFields.EXPIRES_AT.value]) if data.get(
                MetadataFields.EXPIRES_AT.value) else None,
            extra=base.extra
        )


class InteractionLogMetadata(MemoryMetadata):
    """交互轨迹记忆元数据"""

    memory_type: MemoryType = Field(default=MemoryType.INTERACTION_LOG, frozen=True)
    event_type: InteractionEventType = Field(default=InteractionEventType.INQUIRY, description="event type")
    session_id: str = Field(..., description="associated session id")
    sentiment: Optional[InteractionSentiment] = Field(default=InteractionSentiment.NEUTRAL,
                                                      description="sentiment label")
    key_entities: List[str] = Field(default_factory=list, description="key entities involved")
    timestamp: datetime = Field(default_factory=datetime.now, description="interaction occurrence time")

    @model_validator(mode="after")
    def validate_memory_type(self) -> "InteractionLogMetadata":
        if self.memory_type != MemoryType.INTERACTION_LOG:
            raise ValueError(f"InteractionLogMetadata 要求 memory_type 为 {MemoryType.INTERACTION_LOG}")
        return self

    def to_chroma_dict(self) -> Dict[str, Any]:
        data = super().to_chroma_dict()
        data[MetadataFields.EVENT_TYPE.value] = self.event_type.value
        data[MetadataFields.SESSION_ID.value] = self.session_id
        if self.sentiment:
            data[MetadataFields.SENTIMENT.value] = self.sentiment.value
        data[MetadataFields.KEY_ENTITIES.value] = self.key_entities
        data[MetadataFields.TIMESTAMP.value] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_chroma_dict(cls, data: Dict[str, Any]) -> "InteractionLogMetadata":
        base = MemoryMetadata.from_chroma_dict(data)
        return cls(
            **base.model_dump(exclude={MetadataFields.MEMORY_TYPE.value, MemoryMetadata.EXTRA.value}),
            event_type=InteractionEventType(
                data[
                    MetadataFields.EVENT_TYPE.value]) if MetadataFields.EVENT_TYPE.value in data else InteractionEventType.INQUIRY,
            session_id=data.get(MetadataFields.SESSION_ID.value, ""),
            sentiment=InteractionSentiment(
                data[MetadataFields.SENTIMENT.value]) if MetadataFields.SENTIMENT.value in data else None,
            key_entities=data.get(MetadataFields.KEY_ENTITIES.value, []),
            timestamp=datetime.fromisoformat(
                data[MetadataFields.TIMESTAMP.value]) if MetadataFields.TIMESTAMP.value in data else datetime.now(),
            extra=base.extra
        )


class ComplianceRuleMetadata(MemoryMetadata):
    memory_type: MemoryType = Field(default=MemoryType.COMPLIANCE_RULE, frozen=True)
    rule_id: str = Field(..., description="globally unique rule number")
    rule_name: str = Field(..., description="rule name")
    rule_type: str = Field(..., description="rule type,such as forbidden_phrase,sensitive_keyword...")
    pattern: str = Field(..., description="regular expression or key word")
    action: ComplianceAction = Field(..., description="action after trigger")
    severity: ComplianceSeverity = Field(default=ComplianceSeverity.MEDIUM, description="severity level")
    priority: int = Field(default=100, ge=1, le=1000, description="priority level")
    version: str = Field(default_factory=lambda: datetime.now().strftime("Y%-%m-%d"), description="version number")
    effective_from: datetime = Field(default_factory=datetime.now, description="effective time")
    effective_to: Optional[datetime] = Field(default=None, description="expiration Date")
    template: Optional[str] = Field(..., description="template text for append action")

    @model_validator(mode="after")
    def validate_memory_type(self) -> "ComplianceRuleMetadata":
        if self.memory_type != MemoryType.COMPLIANCE_RULE:
            raise ValueError(f"ComplianceRuleMetadata 要求 memory_type 为 {MemoryType.COMPLIANCE_RULE}")
        return self

    def to_chroma_dict(self) -> Dict[str, Any]:
        data = super().to_chroma_dict()
        data[ComplianceRuleFields.RULE_ID.value] = self.rule_id
        data[ComplianceRuleFields.RULE_NAME.value] = self.rule_name
        data[ComplianceRuleFields.RULE_TYPE.value] = self.rule_type
        data[ComplianceRuleFields.PATTERN.value] = self.pattern
        data[ComplianceRuleFields.ACTION.value] = self.action.value
        data[ComplianceRuleFields.SEVERITY.value] = self.severity.value
        data[ComplianceRuleFields.PRIORITY.value] = self.priority
        data[ComplianceRuleFields.VERSION.value] = self.version
        data[ComplianceRuleFields.EFFECTIVE_FROM.value] = self.effective_from.isoformat()
        if self.effective_to:
            data[ComplianceRuleFields.EFFECTIVE_TO.value] = self.effective_to.isoformat()
        if self.template:
            data[ComplianceRuleFields.TEMPLATE.value] = self.template
        return data

    @classmethod
    def from_chroma_dict(cls, data: Dict[str, Any]) -> "ComplianceRuleMetadata":
        base = MemoryMetadata.from_chroma_dict(data)
        return cls(
            **base.model_dump(exclude={MetadataFields.MEMORY_TYPE.value, MetadataFields.EXTRA.value}),
            rule_id=data.get(ComplianceRuleFields.RULE_ID.value, ""),
            rule_name=data.get(ComplianceRuleFields.RULE_NAME.value, ""),
            rule_type=data.get(ComplianceRuleFields.RULE_TYPE.value, ""),
            pattern=data.get(ComplianceRuleFields.PATTERN.value, ""),
            action=ComplianceAction(data[
                                        ComplianceRuleFields.ACTION.value]) if ComplianceRuleFields.ACTION.value in data else ComplianceAction.WARN,
            severity=ComplianceSeverity(data[
                                            ComplianceRuleFields.SEVERITY.value]) if ComplianceRuleFields.SEVERITY.value in data else ComplianceSeverity.MEDIUM,
            priority=data.get(ComplianceRuleFields.PRIORITY.value, 100),
            version=data.get(ComplianceRuleFields.VERSION.value, datetime.now().strftime("%Y-%m-%d")),
            effective_from=datetime.fromisoformat(
                data[
                    ComplianceRuleFields.EFFECTIVE_FROM.value]) if ComplianceRuleFields.EFFECTIVE_FROM.value in data else datetime.now(),
            effective_to=datetime.fromisoformat(data[ComplianceRuleFields.EFFECTIVE_TO.value]) if data.get(
                ComplianceRuleFields.EFFECTIVE_TO.value) else None,
            template=data.get(ComplianceRuleFields.TEMPLATE.value),
            extra=base.extra
        )
