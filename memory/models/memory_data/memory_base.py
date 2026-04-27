import logging
from datetime import datetime
from typing import Dict, Any

from pydantic import BaseModel, Field, field_validator

from config.settings import config
from config.constants import MemoryStatus, GeneralFieldNames

logger = logging.getLogger(__name__)


class MemoryBase(BaseModel):
    """common memory fields"""

    user_id: str = Field(..., description="unique user id")
    confidence: float = Field(default=0.8, gt=0.0, le=1.0, description="confidence score")
    status: MemoryStatus = Field(default=MemoryStatus.ACTIVE, description="status of memory")
    permanent: bool = Field(default=False, description="whether memory is permanent")
    created_at: datetime = Field(default_factory=datetime.now, description="creation time")
    last_accessed_at: datetime = Field(default_factory=datetime.now, description="last access time")
    extra: Dict[str, Any] = Field(default_factory=dict, description="dynamic additional data")

    def touch(self) -> None:
        self.last_accessed_at = datetime.now()

    def forget(self) -> None:
        if not self.permanent:
            self.status = MemoryStatus.FORGOTTEN

    @field_validator(GeneralFieldNames.CONFIDENCE)
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, value))

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v: Any) -> MemoryStatus:
        if not isinstance(v, str):
            return v
        try:
            return MemoryStatus(v)
        except ValueError:
            if config.STRICT_ENUM_VALIDATION:
                raise ValueError(f"Invalid status '{v}'. Must be one of {[e.value for e in MemoryStatus]}")
            logger.warning(f"Invalid status '{v}', fallback to ACTIVE")
            return MemoryStatus.ACTIVE