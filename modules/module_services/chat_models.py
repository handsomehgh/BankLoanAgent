"""
LLM Factory - Production-grade singleton with config from registry
"""
import logging
from typing import Optional, List,TypeVar

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log
from exceptions.exception import LLMTimeoutError, LLMRateLimitError, LLMError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class RobustLLM:
    """
    生产级 LLM 包装器，支持单例、动态温度、重试、回退。
    """

    def __init__(self, temperature=0.6, api_key=None, base_url=None, model=None, provider=None, timeout: int = None):
        self.temperature = temperature
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.provider = provider
        self.timeout = timeout
        self.llm = self._create_llm_instance()

    def _create_llm_instance(self) -> BaseChatModel:
        """创建底层 LangChain 聊天模型实例"""
        return init_chat_model(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=self.temperature,
            model_provider=self.provider,
            timeout=self.timeout
        )

    def invoke_with_fallback(self, messages, fallback_response: str = None):
        try:
            return self._invoke_with_retry(messages)
        except LLMError as e:
            logger.error(f"LLM invocation failed, using fallback: {e}")
            from langchain_core.messages import AIMessage
            fallback = fallback_response or "抱歉，我暂时无法处理您的请求，请稍后再试。"
            return AIMessage(content=fallback)

    def invoke(self, messages,tools: Optional[List[BaseTool]] = None,schema: Optional[T] = None,**kwargs):
        return self._invoke_with_retry(messages, tools=tools,schema=schema,**kwargs)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((LLMTimeoutError, LLMRateLimitError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def _invoke_with_retry(self, messages,tools,schema,**kwargs):
        """同步调用，自动重试"""
        try:
            final_llm = self.llm
            if schema:
                final_llm = self.llm.with_structured_output(schema)
            if tools:
                final_llm = final_llm.bind_tools(tools)
            return final_llm.invoke(messages,**kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if "timeout" in error_msg or "timed out" in error_msg:
                raise LLMTimeoutError(f"LLM timeout: {e}") from e
            elif "rate limit" in error_msg or "429" in error_msg:
                raise LLMRateLimitError(f"LLM rate limited: {e}") from e
            else:
                raise LLMError(f"LLM call failed: {e}") from e
