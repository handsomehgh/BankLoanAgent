import logging
from typing import List, Optional

from langchain_community.embeddings import DashScopeEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from exceptions.exception import EmbeddingTimeoutError, EmbeddingRateLimitError, EmbeddingError

logger = logging.getLogger(__name__)


class RobustEmbeddings:
    def __init__(self,
                 api_key: str = None,
                 model_name: Optional[str] = None,
                 backup_model_name: Optional[str] = None,
                 dimensions: int = 1024):
        self.api_key = api_key
        self.model_name = model_name
        self.backup_model_name = backup_model_name
        self.dimensions = dimensions
        self._primary = None
        self._backup = None

    def _create(self, model_name: str) -> DashScopeEmbeddings:
        """创建 embedding 实例，**model 参数必须使用传入的 model_name**"""
        return DashScopeEmbeddings(
            dashscope_api_key=self.api_key,
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

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """主入口，主模型 → 备份模型 → 零向量降级"""
        # 尝试主模型
        try:
            return self._invoke_with_retry(self.primary, texts)
        except (EmbeddingTimeoutError, EmbeddingRateLimitError, EmbeddingError) as e:
            logger.warning(f"Primary model {self.model_name} failed: {e}")

        # 尝试备份模型（如果配置了）
        backup_instance = self.backup  # 触发懒加载，若未配置返回 None
        if backup_instance is not None:
            try:
                return self._invoke_with_retry(backup_instance, texts)
            except Exception as e:
                logger.error(f"Backup model also failed: {e}")
        else:
            logger.warning("No backup embedding model configured.")

        # 最终降级
        logger.warning("Falling back to zero vectors.")
        return [[0.0] * self.dimensions for _ in texts]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((EmbeddingTimeoutError, EmbeddingRateLimitError)),
        reraise=True,
    )
    def _invoke_with_retry(self, emb: DashScopeEmbeddings, texts: List[str]) -> List[List[float]]:
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
