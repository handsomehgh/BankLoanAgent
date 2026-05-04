# modules/memory/factory.py
from pathlib import Path

from config.models.llm import LLMConfig
from config.models.memory_system import MemorySystemConfig
from config.models.retrieval import RetrievalConfig
from config.registry import ConfigRegistry
from config.settings import GlobalSettings
from infra.milvus_client import MilvusClientManager
from main import inject_sensitive_fields
from modules.memory.memory_business_store.long_term_memory_store import LongTermMemoryStore
from modules.memory.memory_vector_store.milvus_memory_vector_store import MilvusMemoryVectorStore
from modules.module_services.embeddings import get_embeddings, RobustEmbeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def create_test_memory_store(uri: str | None = None, config_path: str | None = None) -> LongTermMemoryStore:
    settings = GlobalSettings()
    milvus_uri = uri or settings.milvus_uri

    registry = ConfigRegistry()
    # 如果未提供路径，使用默认
    path1 = Path(config_path or  PROJECT_ROOT / "config" / "rules" / "memory_system.yaml")
    path2 = Path(config_path or  PROJECT_ROOT / "config" / "rules" / "llm.yaml")
    path3 = Path(config_path or  PROJECT_ROOT / "config" / "rules" / "retrieval.yaml")
    if not registry._models:  # 避免重复注册
        registry.register_model("memory_system", MemorySystemConfig, path1)
        registry.register_model("llm", LLMConfig, path2)
        registry.register_model("retrieval", RetrievalConfig, path3)
        registry.load_all()
    inject_sensitive_fields(registry, settings)
    memory_config = registry.get_config("memory_system")

    milvus_client = MilvusClientManager(uri=milvus_uri)
    embed = RobustEmbeddings()
    vec_store = MilvusMemoryVectorStore(milvus_client=milvus_client,embed=embed, config=memory_config)
    return LongTermMemoryStore(vector_store=vec_store,config=memory_config)