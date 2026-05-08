# author hgh
# version 1.0
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field

from infra.cache.memory_cache_backend import MemoryCacheBackend
from infra.cache.redis_cache_backend import RedisCacheBackend

class L1Config(BaseModel):
    maxsize: int = 512
    ttl: int = 600

    def create_backend(self) -> MemoryCacheBackend:
        return MemoryCacheBackend(maxsize=self.maxsize, ttl=self.ttl)


class L2Config(BaseModel):
    enabled: bool = False
    backend: str = "redis"
    host: str = "localhost"
    port: int = 6379
    db: int = 1
    password: Optional[str] = None
    ttl: int = 3600
    connection_pool: Dict[str, Any] = Field(default_factory=dict)
    socket_timeout: float = 0.1

    def create_backend(self) -> Optional[RedisCacheBackend]:
        if not self.enabled:
            return None

        pool_cfg = self.connection_pool
        max_connections = pool_cfg.get("max_connections", 20)
        extra_pool_kwargs = {k: v for k, v in pool_cfg.items() if k != "max_connections"}

        return RedisCacheBackend(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password,
            max_connections=max_connections,
            socket_timeout=self.socket_timeout,
            extra_pool_kwargs=extra_pool_kwargs,
        )

class CacheConfig(BaseModel):
    l1_defaults: L1Config = Field(default_factory=L1Config)
    l2_defaults: L2Config = Field(default_factory=L2Config)
    knowledge_base_version: int = 1
    compression_threshold: int = 4096
    null_ttl: int = 60
    ttl_jitter: float = 0.1
    namespaces: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
