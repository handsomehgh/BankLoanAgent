from pathlib import Path

from config.models.datasource_config import DataSourceConfig
from config.models.llm_config import LLMConfig
from config.models.memory_config import MemorySystemConfig
from config.models.retrieval_config import RetrievalConfig
from config.registry import ConfigRegistry
from config.settings import GlobalSettings
from infra.database.milvus_client import MilvusClientManager
from infra.database.redis_manager import RedisManager
from modules.memory.memory_business_store.long_term_memory_store import LongTermMemoryStore
from modules.memory.memory_utils.cursor_manager import CursorManager
from modules.memory.memory_vector_store.milvus_memory_vector_store import MilvusMemoryVectorStore
from modules.module_services.embeddings import RobustEmbeddings, RobustLocalEmbeder
from utils.config_utils.get_config import inject_sensitive_fields

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def create_test_memory_store(uri: str | None = None, config_path: str | None = None) -> LongTermMemoryStore:
    settings = GlobalSettings()
    milvus_uri = uri or settings.milvus_uri

    registry = ConfigRegistry()
    # 如果未提供路径，使用默认
    path1 = Path(config_path or  PROJECT_ROOT / "config" / "rules" / "memory_system_config.yaml")
    path2 = Path(config_path or  PROJECT_ROOT / "config" / "rules" / "llm_config.yaml")
    path3 = Path(config_path or  PROJECT_ROOT / "config" / "rules" / "retrieval_config.yaml")
    path4 = Path(config_path or  PROJECT_ROOT / "config" / "rules" / "datasource_config.yaml")

    if not registry._models:  # 避免重复注册
        registry.register_model("memory_system", MemorySystemConfig, path1)
        registry.register_model("llm", LLMConfig, path2)
        registry.register_model("retrieval", RetrievalConfig, path3)
        registry.register_model("datasource", DataSourceConfig, path4)
        registry.load_all()
    inject_sensitive_fields(registry, settings)
    memory_config = registry.get_config("memory_system")
    redis_config = registry.get_config("datasource")
    cfg = registry.get_config("llm")

    milvus_client = MilvusClientManager(uri=milvus_uri)
    embed = RobustLocalEmbeder(
        base_url="http://127.0.0.1:8001/v1/bge",
        model_name="no_need"
    )

    vec_store = MilvusMemoryVectorStore(milvus_client=milvus_client,embeder=embed, config=memory_config)
    return LongTermMemoryStore(vector_store=vec_store,config=memory_config,cursor_manager=CursorManager(redis_manager=RedisManager.from_config(redis_config.redis)))