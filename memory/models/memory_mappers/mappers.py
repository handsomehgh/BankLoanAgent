"""
生产级模型-存储映射器
基于 Pydantic 模型元信息自动完成序列化/反序列化。
包含完善的错误处理、日志记录与异常包装。
"""

import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from exception import MappingError
from memory.models.memory_data.memory_meta import MemoryBase
from memory.models.memory_data.schema import (
    UserProfileMemory,
    InteractionLogMemory,
    ComplianceRuleMemory,
)
from memory.models.memory_constant.constants import MemoryType

logger = logging.getLogger(__name__)

# 模型类映射
MODEL_CLASS_MAP = {
    MemoryType.USER_PROFILE: UserProfileMemory,
    MemoryType.INTERACTION_LOG: InteractionLogMemory,
    MemoryType.COMPLIANCE_RULE: ComplianceRuleMemory,
}

# 必填字段集合（不同记忆类型的必填字段）
REQUIRED_FIELDS = {
    MemoryType.USER_PROFILE: ["user_id", "entity_key"],
    MemoryType.INTERACTION_LOG: ["user_id", "session_id"],
    MemoryType.COMPLIANCE_RULE: ["user_id", "rule_id", "rule_name", "rule_type", "action"],
}

# 核心枚举字段（解析失败应抛出异常，而非静默降级）
CRITICAL_ENUM_FIELDS = {
    "status", "action", "severity"
}


def serialize_field(field_name: str, value: Any) -> Optional[str]:
    """
    将字段值转为 Chroma 安全格式（str / int / float / bool 之一）。
    - None 或空集合 → 返回 None（调用方跳过）
    - datetime → ISO 字符串
    - Enum → value 字符串
    - list / dict → JSON 字符串
    - 未知类型 → str(value) 并记警告
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
                f"Field '{field_name}' is list/dict but cannot be JSON-serialized: {e}. "
                f"Falling back to str()."
            )
            return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    # 未知复杂类型，降级为字符串并记录
    logger.warning(
        f"Field '{field_name}' has unsupported type {type(value).__name__}. "
        f"Converting to str."
    )
    return str(value)


def deserialize_field(
    field_name: str,
    raw: Any,
    annotation: Any,
    memory_type: MemoryType,   # 保留参数以兼容旧调用，实际本函数未使用
) -> Any:
    """
    将存储值反序列化为模型字段期望的类型。
    - 时间字段：空值返回 None；解析失败返回 None（让模型使用默认值）。
    - 枚举字段：仅尝试转换，失败时返回原始字符串，交给模型的 field_validator 处理。
    - 列表/字典字段：JSON 字符串 → 对应类型；解析失败返回空集合。
    - 其他基础类型直接返回。
    """
    # ---------- 时间类型 ----------
    # 判断是否为 datetime 或 Optional[datetime]
    is_datetime = (annotation == datetime)
    is_optional_datetime = (
        hasattr(annotation, '__args__') and
        len(annotation.__args__) == 2 and
        annotation.__args__[1] == type(None) and
        datetime in annotation.__args__
    ) if not is_datetime else False

    if is_datetime or is_optional_datetime:
        # 空字符串、null 字符串统一返回 None
        if isinstance(raw, str) and raw.strip() in ("", "null", "None"):
            return None
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw)
            except ValueError as e:
                logger.error(
                    f"Field '{field_name}' contains invalid datetime string '{raw}': {e}. "
                    f"Returning None to let model apply default."
                )
                return None
        # 其他类型无法处理，返回 None
        logger.warning(
            f"Field '{field_name}' expected datetime, got {type(raw).__name__}. Returning None."
        )
        return None

    # ---------- 枚举类型 ----------
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if isinstance(raw, str):
            try:
                return annotation(raw)
            except ValueError:
                logger.warning(
                    f"Field '{field_name}' has invalid enum value '{raw}' for {annotation.__name__}. "
                    f"Letting model's field_validator handle it."
                )
                return raw   # 传回原始字符串，由模型校验器接管
        elif isinstance(raw, annotation):
            return raw
        else:
            logger.warning(
                f"Enum field '{field_name}' got unexpected type {type(raw).__name__}. "
                f"Returning raw value."
            )
            return raw

    # ---------- 列表类型 ----------
    if annotation == list or annotation == List[str]:
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return parsed
                logger.warning(f"Field '{field_name}' parsed as list but got {type(parsed).__name__}.")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Field '{field_name}' cannot be parsed as list: {e}. Returning [].")
            return []
        if isinstance(raw, list):
            return raw
        logger.warning(f"Field '{field_name}' expected list, got {type(raw).__name__}. Returning [].")
        return []

    # ---------- 字典类型 ----------
    if annotation == dict or annotation == Dict[str, Any]:
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
                logger.warning(f"Field '{field_name}' parsed as dict but got {type(parsed).__name__}.")
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"Field '{field_name}' cannot be parsed as dict: {e}. Returning {{}}.")
            return {}
        if isinstance(raw, dict):
            return raw
        logger.warning(f"Field '{field_name}' expected dict, got {type(raw).__name__}. Returning {{}}.")
        return {}

    # ---------- 其他基础类型（str, int, float, bool）----------
    return raw

class MemoryToStorageMapper:
    """领域模型 → 存储字典"""

    @staticmethod
    def to_db_meta(model: MemoryBase) -> Dict[str, Any]:
        """
        自动将模型转换为存储字典。
        跳过空值和非基本类型，确保 Chroma 兼容。
        """
        result = {}
        for field_name, field_info in model.model_fields.items():
            try:
                value = getattr(model, field_name)
                # 跳过 None
                if value is None:
                    continue
                # 跳过空集合
                if isinstance(value, (list, dict)) and not value:
                    continue

                serialized = serialize_field(field_name, value)
                if serialized is not None:
                    result[field_name] = serialized
            except Exception as e:
                logger.error(f"Failed to serialize field '{field_name}' for model {model.__class__.__name__}: {e}")
                raise MappingError(f"Serialization failed for field '{field_name}'") from e
        return result


class StorageToMemoryMapper:
    """存储字典 → 领域模型"""

    @staticmethod
    def from_db_dict(data: Dict[str, Any], memory_type: MemoryType) -> MemoryBase:
        model_class = MODEL_CLASS_MAP[memory_type]
        init_data = {}

        # 获取当前类型的必填字段
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
                else:
                    # 字段被跳过（枚举降级等），如果是必填字段则报错
                    if field_name in required:
                        msg = f"Required field '{field_name}' could not be deserialized for memory type {memory_type}"
                        logger.error(msg)
                        raise MappingError(msg)
            except MappingError:
                raise  # 重新抛出已知异常
            except Exception as e:
                logger.error(f"Unexpected error deserializing field '{field_name}': {e}")
                if field_name in required:
                    raise MappingError(f"Failed to deserialize required field '{field_name}'") from e
                # 可选字段跳过，继续

        # 检查必填字段是否都已存在
        missing = required - set(init_data.keys())
        if missing:
            msg = f"Missing required fields for {memory_type}: {missing}"
            logger.error(msg)
            raise MappingError(msg)

        # 构建模型实例，Pydantic 会执行最终校验
        try:
            return model_class(**init_data)
        except Exception as e:
            logger.error(f"Model construction failed for {memory_type}: {e}")
            raise MappingError(f"Model instantiation failed") from e