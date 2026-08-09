# author hgh
# version 1.0
from pathlib import Path

from config.global_constant.constants import RegistryModules
from config.models.agent_config import SupervisorConfig, AgentsConfig
from config.models.cache_config import CacheConfig
from config.models.datasource_config import DataSourceConfig
from config.models.file_process_config import FileProcessConfig
from config.models.llm_config import LLMConfig
from config.models.memory_config import MemorySystemConfig
from config.models.prompt_library import PromptLibrary
from config.models.retrieval_config import RetrievalConfig
from config.models.tool_config import ToolRegistryConfig
from config.registry import ConfigRegistry
from config.settings import GlobalSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def inject_sensitive_fields(registry, settings):
    """Inject environment variables into the configuration"""
    llm_cfg = registry.get_config(RegistryModules.LLM)
    llm_cfg.deepseek_api_key = settings.deepseek_api_key
    llm_cfg.alibaba_api_key = settings.alibaba_api_key
    llm_cfg.log_level = settings.log_level
    registry.update_config(RegistryModules.LLM, llm_cfg)

def get_config() -> ConfigRegistry:
    settings = GlobalSettings()
    registry = ConfigRegistry()

    path1 = Path(PROJECT_ROOT / "config" / "rules" / "memory_system_config.yaml")
    path2 = Path(PROJECT_ROOT / "config" / "rules" / "llm_config.yaml")
    path3 = Path(PROJECT_ROOT / "config" / "rules" / "retrieval_config.yaml")
    path4 = Path(PROJECT_ROOT / "config" / "rules" / "file_process_config.yaml")
    path5 = Path(PROJECT_ROOT / "config" / "rules" / "cache.yaml")
    path6 = Path(PROJECT_ROOT / "config" / "rules" / "supervisor.yaml")
    path7 = Path(PROJECT_ROOT / "config" / "rules" / "agents.yaml")
    path10 = Path(PROJECT_ROOT / "config" / "rules" / "tool_registry.yaml")
    path11 = Path(PROJECT_ROOT / "config" / "rules" / "datasource_config.yaml")
    path12 = Path(PROJECT_ROOT / "config" / "rules" / "prompts_retrieval.yaml")
    path13 = Path(PROJECT_ROOT / "config" / "rules" / "prompts_memory.yaml")
    path14 = Path(PROJECT_ROOT / "config" / "rules" / "prompts_agent.yaml")

    registry.register_model(RegistryModules.MEMORY_SYSTEM, MemorySystemConfig, path1)
    registry.register_model(RegistryModules.LLM, LLMConfig, path2)
    registry.register_model(RegistryModules.RETRIEVAL, RetrievalConfig, path3)
    registry.register_model(RegistryModules.FILE_PROCESS, FileProcessConfig, path4)
    registry.register_model(RegistryModules.CACHE, CacheConfig, path5)
    registry.register_model(RegistryModules.SUPERVISOR, SupervisorConfig, path6)
    registry.register_model(RegistryModules.AGENTS, AgentsConfig, path7)
    registry.register_model(RegistryModules.TOOL_REGISTRY, ToolRegistryConfig, path10)
    registry.register_model(RegistryModules.DATASOURCE, DataSourceConfig, path11)
    registry.register_model(RegistryModules.PROMPTS_RETRIEVAL, PromptLibrary, path12)
    registry.register_model(RegistryModules.PROMPTS_MEMORY, PromptLibrary, path13)
    registry.register_model(RegistryModules.PROMPTS_AGENT, PromptLibrary, path14)
    registry.load_all()

    inject_sensitive_fields(registry, settings)

    return registry

if __name__ == '__main__':
    registry = get_config()
    print(registry.get_config(RegistryModules.SUPERVISOR))
    print(registry.get_config(RegistryModules.AGENTS))
    print(registry.get_config(RegistryModules.TOOL_REGISTRY))
