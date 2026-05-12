# author hgh
# version 1.0
import logging
from typing import Optional

from redis import RedisError

from infra.cache.cache_backend import CacheBackend
from infra.redis_manager import RedisManager

logger = logging.getLogger(__name__)

_CURRENT_DATA_VERSION = b"v1:"

class RedisCacheBackend(CacheBackend):
    def __init__(self,max_value_size: int = 1_048_576):
        self._manager = RedisManager()
        self._max_value_size = max_value_size

    def _client(self):
        return self._manager.get_client()

    def get(self,key: str) -> Optional[bytes]:
        client = self._client()
        if client is None:
            return None

        try:
            raw = client.get(key)
            if raw is None:
                return None
            if not raw.startswith(_CURRENT_DATA_VERSION):
                logger.warning("Unsupported data version in key %s", key)
                return None
            return raw[len(_CURRENT_DATA_VERSION):]
        except Exception as e:
            logger.warning("Redis GET failed for %s: %s", key, e)
            return None

    def set(self,key: str,value: bytes,ttl: Optional[int] = None) -> None:
        if len(value) > self._max_value_size:
            logger.warning("Value size %d exceeds limit, not caching key %s", len(value), key)
            return

        client = self._client()
        if client is None:
            return

        data = _CURRENT_DATA_VERSION + value
        try:
            if ttl:
                client.setex(key, ttl, data)
            else:
                client.set(key, data)
        except Exception as e:
            logger.warning("Redis SET failed for %s: %s", key, e)

    def delete(self,key: str) -> None:
        client = self._client()
        if client is None:
            return
        try:
            client.delete(key)
        except Exception as e:
            logger.warning("Redis DELETE failed for %s: %s", key, e)

    def exists(self,key: str) -> bool:
        client = self._client()
        if client is None:
            return False

        try:
            return client.exists(key) > 0
        except (RedisError,TimeoutError,OSError) as e:
            return False





