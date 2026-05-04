# config/settings.py
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class GlobalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # ---- LLM 密钥 ----
    deepseek_api_key: str = Field(..., validation_alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field("https://api.deepseek.com", validation_alias="DEEPSEEK_BASE_URL")
    deepseek_llm_name: str = Field("deepseek-chat", validation_alias="DEEPSEEK_LLM_NAME")
    alibaba_api_key: str = Field(..., validation_alias="ALIBABA_API_KEY")
    alibaba_base_url: str = Field("https://dashscope.aliyuncs.com/compatible-mode/v1", validation_alias="ALIBABA_BASE_URL")
    alibaba_emb_name: str = Field("text-embedding-v4", validation_alias="QWEN_EMB_NAME")
    alibaba_emb_backup: str = Field("text-embedding-v3", validation_alias="QWEN_EMB_NAME_BACKUP")
    qwen_llm_name: str = Field("qwen3-max", validation_alias="QWEN_LLM_NAME")

    # ---- 向量数据库连接 ----
    milvus_uri: str = Field("http://192.168.24.128:19530", validation_alias="MILVUS_URI")

    # ---- 基础设施路径 ----
    sqlite_db_path: str = Field("./checkpoints.db", validation_alias="SQLITE_DB_PATH")

    # ---- 全局运行参数 ----
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")


_global_settings = None

def get_settings() -> GlobalSettings:
    global _global_settings
    if _global_settings is None:
        _global_settings = GlobalSettings()
    return _global_settings

if __name__ == '__main__':
    print(get_settings())