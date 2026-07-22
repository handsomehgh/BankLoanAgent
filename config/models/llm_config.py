from pydantic import BaseModel


class LLMConfig(BaseModel):
    creative_temperature: float = 0.7
    precise_temperature: float = 0.1
    dimension: int = 1024
    deepseek_provider: str = "deepseek"
    openai_provider: str = "openai"
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    local_qwen_url: str = "http://localhost:8006/v1"
    local_qwen_name: str = "qwen2.5-7b"
    deepseek_llm_name: str = "deepseek-chat"
    alibaba_api_key: str = ""
    alibaba_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    alibaba_emb_name: str = "text-embedding-v4"
    alibaba_emb_backup: str = "text-embedding-v3"
    qwen_llm_name: str = "qwen3-max"
    log_level: str = "DEBUG"
    loan_embeder_url: str = "http://192.168.24.128:8000/v1"
    loan_embeder_name: str = "bge-small-loan"
    loan_embeder_dimension: int = 512