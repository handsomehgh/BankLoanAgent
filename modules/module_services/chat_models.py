"""
LLM Factory - Production-grade singleton with config from registry
"""
import logging
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from config.global_constant.constants import RegistryModules
from config.registry import ConfigRegistry
from config.models.llm_config import LLMConfig
from exceptions.exception import LLMTimeoutError, LLMRateLimitError, LLMError

logger = logging.getLogger(__name__)

# ---- module-level singleton cache ----
_llm_instance: Optional['RobustLLM'] = None


def _get_llm_config() -> LLMConfig:
    """获取LLM配置（每次调用可能返回热更新后的最新配置）"""
    return ConfigRegistry().get_config(RegistryModules.LLM)


def _create_llm_instance(api_key: str, base_url: str, model: str, temperature: float,
                         model_provider: str) -> BaseChatModel:
    """创建底层 LangChain 聊天模型实例"""
    return init_chat_model(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        model_provider=model_provider,
        timeout=60.0,
        max_retries=0
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((LLMTimeoutError, LLMRateLimitError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
def _invoke_with_retry(llm: BaseChatModel, messages):
    """同步调用，自动重试"""
    try:
        return llm.invoke(messages)
    except Exception as e:
        error_msg = str(e).lower()
        if "timeout" in error_msg or "timed out" in error_msg:
            raise LLMTimeoutError(f"LLM timeout: {e}") from e
        elif "rate limit" in error_msg or "429" in error_msg:
            raise LLMRateLimitError(f"LLM rate limited: {e}") from e
        else:
            raise LLMError(f"LLM call failed: {e}") from e


class RobustLLM:
    """
    生产级 LLM 包装器，支持单例、动态温度、重试、回退。
    """

    def __init__(self, temperature: float = 0.6, llm_config: Optional[LLMConfig] = None):
        self.temperature = temperature
        self.llm_config = llm_config or _get_llm_config()
        self.llm = self._create_llm()

    def _create_llm(self) -> BaseChatModel:
        provider = self.llm_config.provider
        if provider == "deepseek":
            return _create_llm_instance(
                self.llm_config.deepseek_api_key,
                self.llm_config.deepseek_base_url,
                self.llm_config.deepseek_llm_name,
                self.temperature,
                self.llm_config.deepseek_provider
            )
        elif provider == "qwen":
            return _create_llm_instance(
                self.llm_config.alibaba_api_key,
                self.llm_config.alibaba_base_url,
                self.llm_config.qwen_llm_name,
                self.temperature,
                self.llm_config.qwen_provider
            )
        else:
            raise LLMError(f"Unsupported provider: {provider}")

    def invoke(self, messages):
        return _invoke_with_retry(self.llm, messages)

    def invoke_with_fallback(self, messages, fallback_response: str = None):
        try:
            return self.invoke(messages)
        except LLMError as e:
            logger.error(f"LLM invocation failed, using fallback: {e}")
            from langchain_core.messages import AIMessage
            fallback = fallback_response or "抱歉，我暂时无法处理您的请求，请稍后再试。"
            return AIMessage(content=fallback)

    def update_temperature(self, new_temp: float):
        """动态调整温度，重建底层模型（如果温度变化需要重新创建）"""
        if new_temp != self.temperature:
            self.temperature = new_temp
            self.llm = self._create_llm()
            logger.info(f"LLM temperature changed to {new_temp}")


def get_llm(temperature: float = 0.7, force_new: bool = False) -> RobustLLM:
    """
    获取全局唯一的 LLM 客户端实例。
    :param temperature: 初始温度。如果已存在实例且温度不同，会更新温度。
    :param force_new: 是否强制重新创建实例（慎用）。
    """
    global _llm_instance
    if _llm_instance is None or force_new:
        _llm_instance = RobustLLM(temperature=temperature)
    elif temperature != _llm_instance.temperature:
        _llm_instance.update_temperature(temperature)
    return _llm_instance