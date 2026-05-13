# author hgh
# version 1.0
from pathlib import Path

from config.global_constant.constants import RegistryModules
from config.models.cache_config import CacheConfig
from config.models.file_process_config import FileProcessConfig
from config.models.llm_config import LLMConfig
from config.models.memory_config import MemorySystemConfig
from config.models.retrieval_config import RetrievalConfig
from config.registry import ConfigRegistry
from config.settings import GlobalSettings
from main import inject_sensitive_fields

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_config() -> ConfigRegistry:
    settings = GlobalSettings()
    registry = ConfigRegistry()

    path1 = Path(PROJECT_ROOT / "config" / "rules" / "memory_system_config.yaml")
    path2 = Path(PROJECT_ROOT / "config" / "rules" / "llm_config.yaml")
    path3 = Path(PROJECT_ROOT / "config" / "rules" / "retrieval_config.yaml")
    path4 = Path(PROJECT_ROOT / "config" / "rules" / "file_process_config.yaml")
    path5 = Path(PROJECT_ROOT / "config" / "rules" / "cache.yaml")

    registry.register_model(RegistryModules.MEMORY_SYSTEM, MemorySystemConfig, path1)
    registry.register_model(RegistryModules.LLM, LLMConfig, path2)
    registry.register_model(RegistryModules.RETRIEVAL, RetrievalConfig, path3)
    registry.register_model(RegistryModules.FILE_PROCESS, FileProcessConfig, path4)
    registry.register_model(RegistryModules.CACHE, CacheConfig, path5)
    registry.load_all()

    inject_sensitive_fields(registry, settings)

    return registry
