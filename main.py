from pathlib import Path
from config.registry import ConfigRegistry
from config.settings import GlobalSettings
from config.models.memory_system import MemorySystemConfig
from config.models.retrieval import RetrievalConfig
from config.models.llm import LLMConfig

def inject_sensitive_fields(cfg_registry: ConfigRegistry, settings: GlobalSettings):
    """
    将环境变量中的敏感信息注入到对应配置模块。
    LLM: API Key, base URL
    Retrieval: milvus_uri
    """
    # LLM
    llm_cfg = cfg_registry.get_config("llm")
    llm_cfg.deepseek_api_key = settings.deepseek_api_key
    llm_cfg.deepseek_base_url = settings.deepseek_base_url
    llm_cfg.deepseek_llm_name = settings.deepseek_llm_name
    llm_cfg.alibaba_api_key = settings.alibaba_api_key
    llm_cfg.alibaba_base_url = settings.alibaba_base_url
    llm_cfg.alibaba_emb_name = settings.alibaba_emb_name
    llm_cfg.alibaba_emb_backup = settings.alibaba_emb_backup
    llm_cfg.qwen_llm_name = settings.qwen_llm_name
    cfg_registry.update_config("llm", llm_cfg)

    # Retrieval
    ret_cfg = cfg_registry.get_config("retrieval")
    ret_cfg.milvus_uri = settings.milvus_uri
    ret_cfg.sqlite_db_path = settings.sqlite_db_path
    cfg_registry.update_config("retrieval", ret_cfg)

def load_config():
    settings = GlobalSettings()
    registry = ConfigRegistry()

    # 注册所有模块
    registry.register_model(
        "memory_system", MemorySystemConfig,
        Path("config/rules/memory_system.yaml")
    )
    registry.register_model(
        "retrieval", RetrievalConfig,
        Path("config/rules/retrieval.yaml")
    )
    registry.register_model(
        "llm", LLMConfig,
        Path("config/rules/llm.yaml")
    )

    # 加载 YAML
    registry.load_all()

    # 用环境变量补丁
    inject_sensitive_fields(registry, settings)

    # 启动热更新
    observer = registry.start_hot_reload()

    # 示例使用
    mem = registry.get_config("memory_system")
    print(mem)
    ret = registry.get_config("retrieval")
    print(ret)

    try:
        import time
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main()