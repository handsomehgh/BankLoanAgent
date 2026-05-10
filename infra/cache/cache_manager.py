# author hgh
# version 1.0
"""
unified multi_level cache manager,
supporting namespaces,serialization,compression,null marked,and avalanche protection(TTL random offset)
"""
import json
import logging
import random
import zlib
from typing import Any, Optional

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

        ttl = ttl or self.default_ttl
        ttl = self._apply_jitter(ttl)

        if self.l1:
            try:
                self.l1.set(key, serialized)
            except Exception as e:
                logger.warning(f"[Cache] L1 set failed, key={key}: {e}")

        if self.l2:
            try:
                self.l2.set(key, serialized, ttl)
            except Exception as e:
                logger.warning(f"[Cache] L2 set failed, key={key}: {e}")

    def set_null(self, key: str, ttl: Optional[int] = None) -> None:
        self.set(key, _NULL_MARKER, ttl or self.null_ttl)

    def _delete(self, key: str) -> None:
        final_key = self.build_key(key)
        if self.l1:
            try:
                self.l1.delete(final_key)
            except Exception as e:
                logger.warning(f"[Cache] L1 delete failed, key={final_key}: {e}")

        if self.l2:
            try:
                self.l2.delete(key)
            except Exception as e:
                logger.warning(f"[Cache] L2 delete failed, key={key}: {e}")

    def _serialize(self, value: Any) -> bytes:
        return json.dumps(value, ensure_ascii=False).encode("utf-8")

    def _deserialize(self, value: bytes) -> Any:
        try:
            data = zlib.decompress(value)
        except zlib.error:
            pass
        return json.loads(data.decode("utf-8"))
