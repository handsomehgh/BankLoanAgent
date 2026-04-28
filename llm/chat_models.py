# author hgh
# version 1.0
"""
LLm factory
"""
import logging

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

from config.settings import config
from exceptions.exception import LLMTimeoutError, LLMRateLimitError, LLMError

logger = logging.getLogger(__name__)


def _create_llm_instance(api_key: str, base_url: str, model: str, temperature: float,
                         model_provider: str) -> BaseChatModel:
    """create llm instance"""
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
    """actual invocation llm with retry"""
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
    LLM wrapper with retries and fallback
    """

    def __init__(self, temperature: float = 0.6):
        self.temperature = temperature
        self.llm = self._create_llm()

    def _create_llm(self) -> BaseChatModel:
        if config.llm_provider == "deepseek":
            return _create_llm_instance(
                config.deepseek_api_key,
                config.deepseek_base_url,
                config.deepseek_llm_name,
                self.temperature,
                config.deepseek_provider,
            )
        elif config.llm_provider == "qwen":
            return _create_llm_instance(
                config.alibaba_api_key,
                config.alibaba_base_url,
                config.qwen_llm_name,
                self.temperature,
                config.openai_provider,
            )
        else:
            raise LLMError(f"Unsupported provider: {config.llm_provider}")

    def invoke(self, messages):
        return _invoke_with_retry(self.llm,messages)

    def invoke_with_fallback(self, messages, fallback_response: str = None):
        try:
            return self.invoke(messages)
        except LLMError as e:
            logger.error(f"LLM invocation failed, using fallback: {e}")
            from langchain_core.messages import AIMessage
            fallback = fallback_response if fallback_response else "Sorry, I am temporarily unable to handle your request. Please try again later."
            return AIMessage(content=fallback)


def get_llm(temperature: float = 0.7):
    return RobustLLM(temperature=temperature)



if __name__ == '__main__':
    llm = get_llm()
    res = llm.invoke(messages="给我介绍一下生成式人工智能")
    print(res)