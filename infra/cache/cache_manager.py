# author hgh
# version 1.0
"""
unified multi_level cache manager,
supporting namespaces,serialization,compression,null marked,and avalanche protection(TTL random offset)
"""
import datetime
import hashlib
import json
import logging
import random
import zlib
from enum import Enum
from typing import Any, Optional, List

from pydantic import BaseModel

from infra.cache.cache_backend import CacheBackend

logger = logging.getLogger(__name__)

_NULL_MARKER = b"__NULL__"


class CacheManager:
    def __init__(
            self,
            namespace: str,
            compression_threshold: int = 4096,
            ttl_jitter: float = 0.1,
            default_ttl: int = 3600,
            null_ttl: int = 60,
            version: int = 1,
            l1_backend: Optional[CacheBackend] = None,
            l2_backend: Optional[CacheBackend] = None,
    ):
        self.namespace = namespace
        self.ttl_jitter = ttl_jitter
        self.default_ttl = default_ttl
        self.null_ttl = null_ttl
        self.version = version
        self.compression_threshold = compression_threshold
        self.l1 = l1_backend
        self.l2 = l2_backend

    def build_key(self, *parts: str) -> str:
        segments = [self.namespace] + list(parts) + [f"v{self.version}"]
        return ":".join(segments)

    def _apply_jitter(self, ttl: int) -> int:
        if not ttl or ttl <= 0:
            return ttl
        jitter = int(ttl * self.ttl_jitter)
        offset = random.randint(-jitter, jitter)
        return max(1, ttl + offset)

    def get(self, key: str) -> Optional[Any]:
        raw = None

        if self.l1:
            try:
                raw = self.l1.get(key)
                if raw is not None:
                    logger.debug(f"[Cache] L1 hit: {key}")
            except Exception as e:
                logger.warning(f"[Cache] L1 get failed, key={key}: {e}")

        if raw is None and self.l2:
            try:
                raw = self.l2.get(key)
                if raw is not None:
                    logger.debug(f"[Cache] L2 hit: {key}")
                    # 回填 L1（回填失败仅警告）
                    if self.l1:
                        try:
                            self.l1.set(key, raw)
                        except Exception as e:
                            logger.warning(f"[Cache] L1 backfill failed, key={key}: {e}")
            except Exception as e:
                logger.warning(f"[Cache] L2 get failed, key={key}: {e}")

        if raw is None:
            return None

        if isinstance(raw, bytes) and raw == _NULL_MARKER:
            logger.debug(f"[Cache] null value hit: {key}")
            return _NULL_MARKER

        try:
            return self._deserialize(raw)
        except Exception as e:
            logger.warning(f"[Cache] Deserialize failed, key={key}: {e}")
            self._delete(key)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if isinstance(value, bytes) and value == _NULL_MARKER:
            serialized = _NULL_MARKER
        else:
            try:
                serialized = self._serialize(value)
                if len(serialized) > self.compression_threshold:
                    serialized = zlib.compress(serialized)
            except Exception as e:
                logger.warning(f"[Cache] Serialize failed, key={key}: {e}")
                return

        if self.l1:
            try:
                self.l1.set(key, serialized)
            except Exception as e:
                logger.warning(f"[Cache] L1 set failed, key={key}: {e}")

        if self.l2:
            try:
                ttl = ttl or self.default_ttl
                ttl = self._apply_jitter(ttl)
                self.l2.set(key, serialized, ttl)
            except Exception as e:
                logger.warning(f"[Cache] L2 set failed, key={key}: {e}")

    def set_null(self, key: str, ttl: Optional[int] = None) -> None:
        self.set(key, _NULL_MARKER, ttl or self.null_ttl)

    def _delete(self, key: str) -> None:
        if self.l1:
            try:
                self.l1.delete(key)
            except Exception as e:
                logger.warning(f"[Cache] L1 delete failed, key={key}: {e}")

        if self.l2:
            try:
                self.l2.delete(key)
            except Exception as e:
                logger.warning(f"[Cache] L2 delete failed, key={key}: {e}")

    def invalidate(self,func_name: str,*args,ignore_args: Optional[List[int]] = None,**kwargs) -> None:
        filtered_args = list(args)
        if ignore_args:
            for idx in sorted(ignore_args,reverse=True):
                if 0 <= idx < len(filtered_args):
                    filtered_args.pop(idx)

        params = {"args": filtered_args, "kwargs": kwargs}
        raw = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
        param_hash = hashlib.md5(raw.encode()).hexdigest()
        full_key = self.build_key(func_name, param_hash)
        self._delete(full_key)
        logger.info("Cache invalidated for function=%s, key=%s", func_name, full_key)

    def _serialize(self, value: Any) -> bytes:
        """
        生产级通用序列化：将任意业务对象安全转换为 JSON 字节。
        处理顺序：Pydantic v2/v1 → dataclass/attrs → dict 属性 → 自定义函数 → 降级字符串。
        """

        def _convert(obj: Any) -> Any:
            # 1. Pydantic v2 (model_dump) 或 v1 (dict)
            if isinstance(obj, BaseModel):
                return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj.dict()

            # 2. dataclass / attrs
            if hasattr(obj, "__dataclass_fields__"):
                from dataclasses import asdict
                return asdict(obj)

            # 3. 通用 dict 化接口（如 SQLAlchemy 的 to_dict）
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                return obj.to_dict()

            # 4. 拥有 __dict__ 的普通类实例（排除内置类型和类本身）
            if hasattr(obj, "__dict__") and not isinstance(obj, type):
                return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}

            # 5. 常见不可序列化类型兜底
            if isinstance(obj, (datetime, datetime.date, datetime.time)):
                return obj.isoformat()
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, (set, frozenset)):
                return list(obj)
            if isinstance(obj, bytes):
                return obj.decode("utf-8", errors="replace")

            try:
                json.dumps(obj, ensure_ascii=False)
                return obj
            except (TypeError, ValueError):
                return str(obj)

        # 外部统一使用 default 参数处理复杂对象
        try:
            json_bytes = json.dumps(
                value,
                ensure_ascii=False,
                default=_convert
            ).encode("utf-8")
        except Exception as e:
            logger.error(f"Serialization fallback to str: {e}")
            json_bytes = json.dumps(
                str(value),
                ensure_ascii=False
            ).encode("utf-8")

        # 压缩
        if len(json_bytes) > self.compression_threshold:
            json_bytes = zlib.compress(json_bytes)
        return json_bytes

    def _deserialize(self, value: bytes) -> Any:
        try:
            data = zlib.decompress(value)
        except zlib.error:
            data = value
        return json.loads(data.decode("utf-8"))
