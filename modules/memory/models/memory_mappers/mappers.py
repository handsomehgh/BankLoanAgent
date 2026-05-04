"""
Production-Grade Model - Storage Mapper
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Union, get_origin, get_args

from config.global_constant.constants import MemoryType
from exceptions.exception import MappingError
from modules.memory.models.memory_data.memory_base import MemoryBase
from modules.memory.models.memory_data.memory_schema import (
    UserProfileMemory,
    InteractionLogMemory,
    ComplianceRuleMemory, BusinessKnowledge,
)

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


# ---------- 序列化辅助函数 ----------
def serialize_field(field_name: str, value: Any) -> Any:
    """
    将 Python 值转换为存储友好的格式。
    返回基本类型（str/int/float/bool）或 None。
    - None      → None
    - Enum      → value
    - datetime  → ISO 字符串
    - list/dict → JSON 字符串
    - 其他      → str(value) 并记录警告
    """
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError as e:
            logger.warning(
                f"Field '{field_name}' is list/dict but cannot be JSON‑serialized: {e}. Falling back to str()."
            )
            return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    logger.warning(
        f"Field '{field_name}' has unsupported type {type(value).__name__}. Converting to str."
    )
    return str(value)


# ---------- 反序列化辅助函数 ----------
def deserialize_field(
        field_name: str,
        raw: Any,
        annotation: Any,
        memory_type: MemoryType
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


# ---------- 映射器 ----------
class MemoryToStorageMapper:
    """领域模型 → 存储字典"""

    @staticmethod
    def _infer_type(annotation) -> type:
        """从 Optional[X] 或直接类型中提取基础类型 X。"""
        if get_origin(annotation) is Union:
            args = [a for a in get_args(annotation) if a != type(None)]
            return args[0] if args else str
        return annotation

    @classmethod
    def _default_for_milvus(cls, field_name: str, field_info) -> Any:
        """根据字段的 Pydantic 类型注解返回 Milvus 安全的空值。"""
        annotation = field_info.annotation
        base_type = cls._infer_type(annotation)

        if base_type == str:
            return ""
        if base_type == int:
            return 0
        if base_type == float:
            return 0.0
        if base_type == bool:
            return False
        if base_type == datetime:
            # Milvus 中 datetime 存为 ISO 字符串，空字符串表示缺失
            return ""
        if base_type == list or base_type == dict:
            return "[]"  # 空集合序列化为 JSON 字符串
        if isinstance(base_type, type) and issubclass(base_type, Enum):
            # 取第一个枚举值
            return list(base_type)[0].value
        # 兜底
        logger.warning(f"Unknown type {annotation} for field '{field_name}', using empty string.")
        return ""

    @staticmethod
    def to_db_meta(model: Any, target_db: str = "chroma") -> Dict[str, Any]:
        """
        Convert the model to a storage dictionary.
        :param target_db: "chroma" or "milvus"
            - chroma: completely remove fields with value None
            - milvus: replace None with safe empty values ("" / 0 / 0.0 / False, etc.)
        """
        result = {}
        for field_name, field_info in model.model_fields.items():
            try:
                value = getattr(model, field_name)
                serialized = serialize_field(field_name, value)  # 可能为 None

                if target_db == "chroma":
                    if serialized is None:
                        continue
                elif target_db == "milvus":
                    if serialized is None:
                        serialized = MemoryToStorageMapper._default_for_milvus(field_name, field_info)
                result[field_name] = serialized
            except Exception as e:
                logger.error(f"Failed to serialize field '{field_name}': {e}")
                raise MappingError(f"Serialization failed for field '{field_name}'") from e
        return result


class StorageToMemoryMapper:
    """存储字典 → 领域模型"""

    @staticmethod
    def from_db_dict(data: Dict[str, Any], memory_type: MemoryType) -> MemoryBase:
        model_class = MODEL_CLASS_MAP[memory_type]
        init_data = {}
        required = set(REQUIRED_FIELDS.get(memory_type, []))

        for field_name, field_info in model_class.model_fields.items():
            raw = data.get(field_name)
            if raw is None:
                continue

            annotation = field_info.annotation
            try:
                value = deserialize_field(field_name, raw, annotation, memory_type)
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
