# author hgh
# version 1.0
import logging
from typing import Optional

import redis
from redis import RedisError

from infra.cache.cache_backend import CacheBackend

logger = logging.getLogger(__name__)

_CURRENT_DATA_VERSION = b"v1:"

class RedisCacheBackend(CacheBackend):
    def __init__(
            self,
            host: str = "localhost",
            port: int = 6379,
            db: int = 0,
            password: Optional[str] = None,
            max_connections: int = 20,
            socket_timeout: float = 0.1,
            max_value_size: int = 1_048_576,
            extra_pool_kwargs: Optional[dict] = None
    ):
        pool_kwargs = {
            "host": host,
            "port": port,
            "db": db,
            "max_connections": max_connections,
            "socket_timeout": socket_timeout
        }
        if password is not None:
            pool_kwargs["password"] = password
        if extra_pool_kwargs:
            pool_kwargs.update(extra_pool_kwargs)

        self._pool = redis.ConnectionPool(**pool_kwargs)
        self._client = redis.Redis(connection_pool=self._pool)
        self._max_value_size = max_value_size

    def get(self,key: str) -> Optional[bytes]:
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            if not raw.startswith(_CURRENT_DATA_VERSION):
                logger.warning(f"Unsupported data version in key {key}, ignoring")
                return None
            return raw[len(_CURRENT_DATA_VERSION):]
        except (RedisError,TimeoutError,OSError) as e:
            logger.warning(f"Redis get error for key {key}: {e}")
            return None

    def set(self,key: str,value: bytes,ttl: Optional[int] = None) -> None:
        if len(value) > self._max_value_size:
            logger.warning(f"value size {len(value)} exceeds limit {self._max_value_size},not caching key {key}")
            return
        data = _CURRENT_DATA_VERSION + value
        try:
            logger.info(f"Preparing to write to RAG redis  cache,key -> {key},value -> {data}")
            if ttl:
                self._client.setex(key,ttl,data)
            else:
                self._client.set(key,data)
        except (RedisError,TimeoutError,OSError) as e:
            logger.warning(f"Redis set error for key {key}: {e}")

    def delete(self,key: str) -> None:
        try:
            self._client.delete(key)
        except (RedisError,TimeoutError,OSError) as e:
            logger.warning(f"Redis delete error for key {key}: {e}")

    def exists(self,key: str) -> bool:
        try:
            return self._client.exists(key) > 0
        except (RedisError,TimeoutError,OSError) as e:
            return False





