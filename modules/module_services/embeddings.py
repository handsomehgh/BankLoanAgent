import logging
from typing import List, Optional

from langchain_community.embeddings import DashScopeEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.global_constant.constants import RegistryModules
from config.models.llm_config import LLMConfig
from config.registry import ConfigRegistry
from exceptions.exception import EmbeddingTimeoutError, EmbeddingRateLimitError, EmbeddingError

logger = logging.getLogger(__name__)


def _get_llm_config() -> LLMConfig:
    """获取LLM配置（每次调用可能返回热更新后的最新配置）"""
    return ConfigRegistry().get_config(RegistryModules.LLM)


class RobustEmbeddings:
    def __init__(self,
                 model_name: Optional[str] = None,
                 backup_model_name: Optional[str] = None,
                 dimensions: int = 1024,
                 llm_config: Optional[LLMConfig] = None):
        self.llm_config = llm_config or _get_llm_config()
        self.model_name = model_name or self.llm_config.alibaba_emb_name
        self.backup_model_name = backup_model_name or self.llm_config.alibaba_emb_backup
        self.dimensions = dimensions
        self._primary = None
        self._backup = None

    def _create(self, model_name: str) -> DashScopeEmbeddings:
        """创建 embedding 实例，**model 参数必须使用传入的 model_name**"""
        return DashScopeEmbeddings(
            dashscope_api_key=self.llm_config.alibaba_api_key,
            model=model_name
        )

    @property
    def primary(self):
        if self._primary is None:
            self._primary = self._create(self.model_name)
        return self._primary

    @property
    def backup(self):
        if self.backup_model_name is None:
            return None
        if self._backup is None:
            self._backup = self._create(self.backup_model_name)
        return self._backup

    def _call_with_retry(self, emb: DashScopeEmbeddings, texts: List[str]) -> List[List[float]]:
        """带重试的调用，返回向量结果"""

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((EmbeddingTimeoutError, EmbeddingRateLimitError)),
            reraise=True,
        )
        def _invoke():
            try:
                return emb.embed_documents(texts)
            except Exception as e:
                error_msg = str(e).lower()
                if "timeout" in error_msg or "timed out" in error_msg:
                    raise EmbeddingTimeoutError(f"Embedding timeout: {e}") from e
                elif "rate limit" in error_msg or "429" in error_msg:
                    raise EmbeddingRateLimitError(f"Rate limited: {e}") from e
                else:
                    raise EmbeddingError(f"Embedding failed: {e}") from e

        return _invoke()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """主入口，主模型 → 备份模型 → 零向量降级"""
        # 尝试主模型
        try:
            return self._call_with_retry(self.primary, texts)
        except (EmbeddingTimeoutError, EmbeddingRateLimitError, EmbeddingError) as e:
            logger.warning(f"Primary model {self.model_name} failed: {e}")

        # 尝试备份模型（如果配置了）
        backup_instance = self.backup  # 触发懒加载，若未配置返回 None
        if backup_instance is not None:
            try:
                return self._call_with_retry(backup_instance, texts)
            except Exception as e:
                logger.error(f"Backup model also failed: {e}")
        else:
            logger.warning("No backup embedding model configured.")

        # 最终降级
        logger.warning("Falling back to zero vectors.")
        return [[0.0] * self.dimensions for _ in texts]


_robust_embeddings_instance = None


def get_embeddings(model_name: Optional[str] = None) -> RobustEmbeddings:
    global _robust_embeddings_instance
    if _robust_embeddings_instance is None:
        _robust_embeddings_instance = RobustEmbeddings(model_name=model_name)
    return _robust_embeddings_instance


if __name__ == '__main__':
    client = get_embeddings()
    res = client.embed_documents(["hello world"])
    print(len(res[0]))
