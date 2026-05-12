# author hgh
# version 1.0
from config.models.cache_config import CacheConfig
from infra.cache.cache_manager import CacheManager
from infra.cache.memory_cache_backend import MemoryCacheBackend
from infra.cache.redis_cache_backend import RedisCacheBackend


class CacheFactory:
    def __init__(self, config: CacheConfig):
        self.config = config

    def create(self, namespace: str) -> CacheManager:
        # namespace config
        ns_cfg = self.config.namespaces.get(namespace, {})
        if ns_cfg is None:
            raise KeyError(f"Namespace '{namespace}' not found in cache configuration")

        # L1
        l1_backend = None
        if ns_cfg.enable_l1:
            if ns_cfg.l1 is None:
                raise ValueError(f"Namespace '{namespace}' enabled L1 but missing l1 config")
            l1_backend = MemoryCacheBackend(maxsize=ns_cfg.l1.maxsize, ttl=ns_cfg.l1.ttl)

        # L2
        l2_backend = None
        if ns_cfg.enable_l2:
            if ns_cfg.l2 is None or any(conf is None for conf in ns_cfg.model_dump().values):
                raise ValueError(f"Namespace '{namespace}' enabled L2 but missing l2 config")
            l2_backend = RedisCacheBackend(max_value_size=ns_cfg.l2.max_value_size)

        if l1_backend is None and l2_backend is None:
            raise ValueError(f"Namespace '{namespace}' must enable at least L1 or L2")

        return CacheManager(
            namespace=namespace,
            l1_backend=l1_backend,
            l2_backend=l2_backend,
            compression_threshold=ns_cfg.l2.compression_threshold,
            ttl_jitter=ns_cfg.l2.ttl_jitte,
            default_ttl=ns_cfg.l2.ttl,
            null_ttl=ns_cfg.l2.null_ttl,
            version=self.config.knowledge_base_version,
        )
