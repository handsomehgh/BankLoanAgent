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
    alibaba_api_key: str = Field(..., validation_alias="ALIBABA_API_KEY")
    log_level: str = Field("INFO", validation_alias="LOG_LEVEL")
