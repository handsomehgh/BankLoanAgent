# author hgh
# version 1.0
from datetime import datetime
from typing import Dict, Any

from pydantic import BaseModel, Field, field_validator

from memory.constant.constants import MemoryType, MemorySource, MemoryStatus, MetadataFields, MemoryModelFields


class MemoryMetadata(BaseModel):
    user_id: str = Field(..., description="user unique identifier")
    memory_type: MemoryType = Field(..., description="memory type")

    # general optional fields
    source: MemorySource = Field(default=MemorySource.CHAT_EXTRACTION, description="the source of memory")
    confidence: float = Field(default=0.8, gt=0.0, lt=1.0, description="memory reliability")
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE, description="the status of memory")
    permanent: bool = Field(default=False, description="whether the memory is permanent")
    create_at: datetime = Field(default_factory=datetime.now, description="the time of creation")
    lass_accessed_at: datetime = Field(default_factory=datetime.now, description="the time of last access")

    # associated fields, used for conflict detection
    superseded_by: str = Field(default=None, description="new memory id that overwrites this memory")

    # extended field container
    extra: Dict[str, Any] = Field(default_factory=dict, exclude=True)

    @field_validator(MetadataFields.CONFIDENCE.value)
    def validate_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, value))

    def to_chroma_dict(self) -> Dict[str, Any]:
        """
        convert to a flat dict that acceptable by chroma(covert enum values to strings,covert time to ISO format)
        """
        data = self.model_dump(exclude={MetadataFields.MEMORY_TYPE.value, MetadataFields.EXTRA.value})
        data[MetadataFields.TYPE.lower()] = self.memory_type.value
        data[MetadataFields.SOURCE.value] = self.source.value
        data[MetadataFields.STATUS.value] = self.status.value
        data[MetadataFields.CREATE_AT.value] = self.create_at.isoformat()
        data[MetadataFields.LAST_ACCESS_AT.value] = self.lass_accessed_at.isoformat()
        if self.superseded_by:
            data[MetadataFields.SUPERSEDED_BY.value] = self.superseded_by
        data.update(self.extra)
        return data

    @classmethod
    def from_chrom_dict(cls, data: Dict[str, Any]) -> "MemoryMetadata":
        """
        deserialize the dict returned from chroma into a model instance
        """
        known_fields = {
            MetadataFields.USER_ID.value: data.get(MemoryMetadata.USER_ID.value),
            MetadataFields.MEMORY_TYPE.value: data.get(
                MetadataFields.TYPE.value) if MetadataFields.TYPE.value in data else MemoryType.USER_PROFILE.value,
            MetadataFields.SOURCE.value: data.get(
                MetadataFields.SOURCE.value) if MetadataFields.SOURCE.value in data else MemorySource.CHAT_EXTRACTION.value,
            MetadataFields.CONFIDENCE.value: data.get(MetadataFields.CONFIDENCE.value, 0.8),
            MetadataFields.STATUS.value: data.get(
                MetadataFields.STATUS.value) if MetadataFields.STATUS.value in data else MemoryStatus.ACTIVE.value,
            MetadataFields.PERMANENT.value: data.get(MetadataFields.PERMANENT.value, False),
            MetadataFields.CREATE_AT.value: datetime.fromisoformat(
                data[MetadataFields.CREATE_AT.value]) if MetadataFields.CREATE_AT.value in data else datetime.now(),
            MetadataFields.LAST_ACCESS_AT.value: datetime.fromisoformat(data[
                                                                            MetadataFields.LAST_ACCESS_AT.value]) if MetadataFields.LAST_ACCESS_AT.value in data else datetime.now(),
            MetadataFields.SUPERSEDED_BY.value: data.get(MetadataFields.SUPERSEDED_BY.value)
        }

        extra = {k: v for k, v in data.items() if
                 k not in known_fields and k not in [MetadataFields.TYPE.value, MetadataFields.SOURCE.value,
                                                     MetadataFields.STATUS.value, MetadataFields.CREATE_AT.value,
                                                     MetadataFields.LAST_ACCESS_AT.value]}

        return cls(**known_fields, extra=extra)

    def touch(self) -> None:
        self.lass_accessed_at = datetime.now()
