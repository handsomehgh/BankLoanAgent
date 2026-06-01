# author hgh
# version 1.0
import json
import logging
from datetime import datetime
from enum import Enum
from typing import get_origin, Union, get_args, Any, Dict

from exceptions.exception import MappingError

logger = logging.getLogger(__name__)

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