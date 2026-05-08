# author hgh
# version 1.0
import threading
from typing import Optional

from cachetools import TTLCache

from infra.cache.cache_backend import CacheBackend


class MemoryCacheBackend(CacheBackend):
    def __init__(self,maxsize: int = 512,ttl: int = 600):
        self.cache = TTLCache(maxsize,ttl)
        self._lock = threading.Lock()

    def get(self,key: str) -> Optional[bytes]:
        with self._lock:
            return self.cache.get(key)

    def set(self,key: str,value: bytes,ttl: Optional[int] = None) -> None:
        with self._lock:
            self.cache[key] = value

    def delete(self,key: str) -> None:
        with self._lock:
            self.cache.pop(key)

    def exists(self,key: str) -> bool:
        with self._lock:
            return key in self.cache
