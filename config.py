# author hgh
# version 1.0
import os
import sys

from pydantic import Field
from pydantic_core import ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from exception import ConfigurationError


class BankLoanAgentConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".env"),  # 修正拼写
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # qwen ll
    qwen_llm_name: str = Field(..., validation_alias="QWEN_LLM_NAME")
    qwen_emb_name: str = Field(..., validation_alias="QWEN_EMB_NAME")
    alibaba_api_key: str = Field(..., validation_alias="ALIBABA_API_KEY")
    alibaba_base_url: str = Field("https://dashscope.aliyuncs.com/compatible-mode/v1",
                                  validation_alias="ALIBABA_BASE_URL")

    # deepseek llm
    deepseek_api_key: str = Field(..., validation_alias="DEEPSEEK_API_KEY")
    deepseek_base_url: str = Field("https://api.deepseek.com", validation_alias="DEEPSEEK_BASE_URL")
    deepseek_llm_name: str = Field(..., validation_alias="DEEPSEEK_LLM_NAME")

    # llm provider
    llm_provider: str = Field(..., validation_alias="LLM_PROVIDER")
    openai_provider: str = Field("openai", validation_alias="OPENAI_PROVIDER")
    deepseek_provider: str = Field("deepseek", validation_alias="deepseek")

    # storage
    chroma_persist_dir: str = Field("./chroma_data", validation_alias="CHROMA_PERSIST_DIR")
    sqlite_db_path: str = Field("./checkpoints.db", validation_alias="SQLITE_DB_PATH")

    # agent behaviour
    max_rewrite_attempts: int = 1
    self_eval_threshold: float = 0.7
    max_context_messages: int = 20

    # memory decay
    decay_factor: float = Field(0.01, validation_alias="DECAY_LAMBDA")
    decay_threshold: float = Field(0.3, validation_alias="DECAY_THRESHOLD")
    cleanup_interval_hours: int = Field(24, validation_alias="CLEANUP_INTERVAL_HOURS")

    # rag metrics
    retrieval_top_k: int = Field(5, validation_alias="RETRIEVAL_TOP_K")
    retrieval_fetch_k: int = Field(10, validation_alias="RETRIEVAL_FETCH_K")

    #memory dlq path
    memory_dlq_path: str = Field(...,validation_alias="MEMORY_DLQ_PATH")

    # Logging
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")

    def __str__(self):
        infos = []
        for name, value in self.model_dump().items():
            infos.append(f"{name}:{value}")
        return "\n".join(infos)


try:
    config = BankLoanAgentConfig()
except ValidationError as e:
    print("❌ 配置验证失败，请检查 .env 文件是否存在且包含以下必填项：")
    print(e)
    sys.exit(1)

if __name__ == '__main__':
    print(config)
