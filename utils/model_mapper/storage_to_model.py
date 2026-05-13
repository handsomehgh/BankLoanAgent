# author hgh
# version 1.0
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Any, get_origin

from google._upb._message import RepeatedScalarContainer

from config.global_constant.constants import MemoryType
from exceptions.exception import MappingError
from modules.memory.models.memory_schema import UserProfileMemory, InteractionLogMemory, \
    ComplianceRuleMemory
from modules.retrieval.knowledge_model import BusinessKnowledge

logger = logging.getLogger(__name__)

MODEL_CLASS_MAP = {
    MemoryType.USER_PROFILE: UserProfileMemory,
    MemoryType.INTERACTION_LOG: InteractionLogMemory,
    MemoryType.COMPLIANCE_RULE: ComplianceRuleMemory,
    MemoryType.BUSINESS_KNOWLEDGE: BusinessKnowledge,
}

REQUIRED_FIELDS = {
    MemoryType.USER_PROFILE: ["user_id", "entity_key"],
    MemoryType.INTERACTION_LOG: ["user_id", "session_id"],
    MemoryType.COMPLIANCE_RULE: ["user_id", "rule_id", "rule_name", "rule_type", "action"],
}

class StorageToMemoryMapper:
    """storage dict → domain model"""

    @staticmethod
    def from_db_dict(data: Dict[str, Any], memory_type: MemoryType) -> Any:
        model_class = MODEL_CLASS_MAP[memory_type]
        init_data = {}
        required = set(REQUIRED_FIELDS.get(memory_type, []))

        for field_name, field_info in model_class.model_fields.items():
            raw = data.get(field_name)
            if raw is None:
                continue

            annotation = field_info.annotation
            try:
                value = deserialize_field(field_name, raw, annotation)
                if value is not None:
                    init_data[field_name] = value
                elif field_name in required:
                    raise MappingError(
                        f"Required field '{field_name}' deserialized to None for {memory_type}"
                    )
            except MappingError:
                raise
            except Exception as e:
                logger.error(f"Unexpected error deserializing field '{field_name}': {e}")
                if field_name in required:
                    raise MappingError(f"Failed to deserialize required field '{field_name}'") from e

        missing = required - set(init_data.keys())
        if missing:
            raise MappingError(f"Missing required fields for {memory_type}: {missing}")

        try:
            return model_class(**init_data)
        except Exception as e:
            logger.error(f"Model construction failed for {memory_type}: {e}")
            raise MappingError(f"Model instantiation failed") from e

# ---------- 反序列化辅助函数 ----------
def deserialize_field(
        field_name: str,
        raw: Any,
        annotation: Any
) -> Any:
    """将存储值反序列化为模型字段期望的类型。"""
    # datetime type
    is_datetime = (annotation == datetime)
    is_optional_datetime = (
            hasattr(annotation, '__args__') and
            len(annotation.__args__) == 2 and
            annotation.__args__[1] == type(None) and
            datetime in annotation.__args__
    ) if not is_datetime else False

    if is_datetime or is_optional_datetime:
        if isinstance(raw, str) and raw.strip() in ("", "null", "None"):
            return None
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw)
            except ValueError as e:
                logger.error(
                    f"Field '{field_name}' invalid datetime string '{raw}': {e}. Returning None."
                )
                return None
        logger.warning(f"Field '{field_name}' expected datetime, got {type(raw).__name__}. Returning None.")
        return None

    # enum types
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if isinstance(raw, str):
            try:
                return annotation(raw)
            except ValueError:
                logger.warning(
                    f"Field '{field_name}' has invalid enum value '{raw}'. Letting model's field_validator handle it."
                )
                return raw
        elif isinstance(raw, annotation):
            return raw
        else:
            logger.warning(f"Enum field '{field_name}' got {type(raw).__name__}. Returning raw value.")
            return raw

    # 列表类型
    if annotation == list or (get_origin(annotation) is list):
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Field '{field_name}' cannot be parsed as list: {e}. Returning [].")
                return []
        if isinstance(raw, RepeatedScalarContainer):
            return list(raw)  # type: ignore[arg-type]
        if isinstance(raw, list):
            return raw
        logger.warning(f"Field '{field_name}' expected list, got {type(raw).__name__}. Returning [].")
        return []

    # 字典类型
    if annotation == dict or (get_origin(annotation) is dict):
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Field '{field_name}' cannot be parsed as dict: {e}. Returning {{}}.")
                return {}
        if isinstance(raw, dict):
            return raw
        logger.warning(f"Field '{field_name}' expected dict, got {type(raw).__name__}. Returning {{}}.")
        return {}

    return raw