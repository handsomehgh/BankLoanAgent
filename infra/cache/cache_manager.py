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

    def _make_key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def _apply_jitter(self, ttl: int) -> int:
        if not ttl or ttl <= 0:
            return ttl
        jitter = int(ttl * self.ttl_jitter)
        offset = random.randint(-jitter, jitter)
        return max(1, ttl + offset)

    def get(self, key: str) -> Optional[Any]:
        full_key = self._make_key(key)
        raw = None

        if self.l1:
            raw = self.l1.get(full_key)

        if raw is None and self.l2:
            raw = self.l2.get(full_key)
            if raw is not None and self.l1:
                self.l1.set(full_key, raw)
        if raw is None:
            return None
        if raw == _NULL_MARKER:
            return _NULL_MARKER
        return self._deserialize(raw)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        full_key = self._make_key(key)
        if isinstance(value, bytes) and value == _NULL_MARKER:
            serialized = _NULL_MARKER
        else:
            serialized = self._serialize(value)
            if len(serialized) > self.compression_threshold:
                serialized = zlib.compress(serialized)
        ttl = ttl or self.default_ttl
        ttl = self._apply_jitter(ttl)
        if self.l1:
            self.l1.set(full_key, serialized)
        if self.l2:
            self.l2.set(full_key, serialized, ttl)

    def set_null(self, key: str, ttl: Optional[int] = None) -> None:
        self.set(key, _NULL_MARKER, self.null_ttl)

    def _delete(self, key: str) -> None:
        full_key = self._make_key(key)
        if self.l1:
            self.l1.delete(full_key)
        if self.l2:
            self.l2.delete(full_key)

    def _serialize(self, value: Any) -> bytes:
        return json.dumps(value, ensure_ascii=False).encode("utf-8")

    def _deserialize(self, value: bytes) -> Any:
        try:
            data = zlib.decompress(value)
        except zlib.error:
            pass
        return json.loads(data.decode("utf-8"))
