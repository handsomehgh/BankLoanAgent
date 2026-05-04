"""
LLM / Embedding 提供商配置（非敏感参数）
"""
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    llm_provider: str = "deepseek"
    deepseek_provider: str = "deepseek"
    deepseek_api_key: str = ""                 # 运行时注入
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_llm_name: str = "deepseek-chat"
    alibaba_api_key: str = ""                  # 运行时注入
    alibaba_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    alibaba_emb_name: str = "text-embedding-v4"
    alibaba_emb_backup: str = "text-embedding-v3"
    qwen_llm_name: str = "qwen3-max"
    qwen_provider: str = "openai"