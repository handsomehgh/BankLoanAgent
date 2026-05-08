# author hgh
# version 1.0
from config.models.cache_config import CacheConfig
from infra.cache.cache_manager import CacheManager


class CacheFactory:
    def __init__(self, config: CacheConfig):
        self.global_config = config

    def create(self, namespace: str) -> CacheManager:
        # namespace config
        ns_cfg = self.global_config.namespaces.get(namespace, {})

        # merge l1 and l2 config
        l1_config = self.global_config.l1_defaults.model_copy(update=ns_cfg.get("l1", {}))
        l2_config = self.global_config.l2_defaults.model_copy(update=ns_cfg.get("l2", {}))

        # create cache instance
        l1_backend = l1_config.create_backend()
        l2_backend = l2_config.create_backend()

        return CacheManager(
            namespace=namespace,
            l1_backend=l1_backend,
            l2_backend=l2_backend,
            compression_threshold=self.global_config.compression_threshold,
            ttl_jitter=self.global_config.ttl_jitter,
            default_ttl=l2_config.ttl if l2_config.enabled else l1_config.ttl,
            null_ttl=self.global_config.null_ttl,
            version=self.global_config.knowledge_base_version,

        )
