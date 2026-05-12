from pathlib import Path

from config.context_settings import set_enum_strictness
from config.global_constant.constants import RegistryModules
from config.models.cache_config import CacheConfig
from config.models.file_process_config import FileProcessConfig
from config.models.redis_config import RedisConfig
from config.registry import ConfigRegistry
from config.settings import GlobalSettings
from config.models.memory_config import MemorySystemConfig
from config.models.retrieval_config import RetrievalConfig
from config.models.llm_config import LLMConfig


PROJECT_ROOT = Path(__file__).resolve().parent

def inject_sensitive_fields(cfg_registry: ConfigRegistry, settings: GlobalSettings):
    """
    将环境变量中的敏感信息注入到对应配置模块。
    LLM: API Key, base URL
    Retrieval: milvus_uri
    """
    # LLM
    llm_cfg = cfg_registry.get_config(RegistryModules.LLM)
    llm_cfg.deepseek_api_key = settings.deepseek_api_key
    llm_cfg.alibaba_api_key = settings.alibaba_api_key
    llm_cfg.log_level = settings.log_level
    cfg_registry.update_config(RegistryModules.LLM, llm_cfg)

def load_config():
    settings = GlobalSettings()
    registry = ConfigRegistry()

    # 注册所有模块
    registry.register_model(
        RegistryModules.MEMORY_SYSTEM, MemorySystemConfig,
        Path(PROJECT_ROOT  / "config/rules/memory_system_config.yaml")
    )
    registry.register_model(
        RegistryModules.RETRIEVAL, RetrievalConfig,
        Path(PROJECT_ROOT / "config/rules/retrieval_config.yaml")
    )
    registry.register_model(
        RegistryModules.LLM, LLMConfig,
        Path(PROJECT_ROOT / "config/rules/llm_config.yaml")
    )
    registry.register_model(
        RegistryModules.FILE_PROCESS, FileProcessConfig,
        Path(PROJECT_ROOT / "config/rules/file_process_config.yaml")
    )
    registry.register_model(
        RegistryModules.CACHE, CacheConfig,
        Path(PROJECT_ROOT / "config/rules/cache.yaml")
    )
    registry.register_model(
        RegistryModules.REDIS, RedisConfig,
        Path(PROJECT_ROOT / "config/rules/redis.yaml")
    )

    # 加载 YAML
    registry.load_all()

    # set context config
    memory_config = registry.get_config(RegistryModules.MEMORY_SYSTEM)
    set_enum_strictness(memory_config.strict_enum_validation)

    # 用环境变量补丁
    inject_sensitive_fields(registry, settings)

    # 启动热更新
    observer = registry.start_hot_reload()
