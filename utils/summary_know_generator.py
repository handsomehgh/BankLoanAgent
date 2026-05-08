# author hgh
# version 1.0
import hashlib
import logging
from typing import Dict, Optional

from config.models.retrieval_config import RetrievalConfig

logger = logging.getLogger(__name__)

SUMMARY_PROMPT_TEMPLATE = """请用一句简洁的话概括以下银行个人贷款业务知识片段的核心内容，保留关键产品名称和重要数字。不要添加任何解释，只输出摘要。

知识片段：
{content}

摘要："""


class SummaryKnowledgeGenerator:
    def __init__(self, config: RetrievalConfig,llm_client):
        summary_config = config.multi_vector.summary_config
        self.enabled_sources = summary_config.enabled_sources or []
        self.min_chunk_length = summary_config.min_chunk_length or 200
        self.max_output_tokens = summary_config.max_output_tokens or 80
        self.fallback_to_original = summary_config.fallback_to_original or True
        self.enable_source_filter = summary_config.enable_source_filter or True

        self.llm_client = llm_client

        self._cache: Dict[str, str] = {}

    def should_generate(self, source_type: str, content_len: int) -> bool:
        if self.enable_source_filter and source_type not in self.enabled_sources:
            return False
        if content_len < self.min_chunk_length:
            return False
        return True

    def generate_summary(self, text: str) -> Optional[str]:
        """
        generate summary，automatically cache,
        Decide whether to return None on failure based on the configuration.
        returning None indicates that the caller should fall back to the original text vector。
        """
        if not text or len(text.strip()) < 50:
            return None

        key = hashlib.md5(text.encode("utf-8")).hexdigest()
        if key in self._cache:
            logger.debug("hit summary cache")
            return self._cache[key]

        prompt = SUMMARY_PROMPT_TEMPLATE.format(content=text[:2000])

        try:
            response = self.llm_client.invoke(prompt)

            if hasattr(response, "content"):
                summary = response.content.strip()
            else:
                summary = str(response).strip()

            if summary.startswith("摘要："):
                summary = summary[3:].strip()
            if len(summary) > self.max_output_tokens * 2:
                summary = summary[:self.max_output_tokens * 2]

            if summary:
                self._cache[key] = summary
                logger.debug("Summary generated successfully，length %d", len(summary))
                return summary
            else:
                logger.warning("LLM return none")
                return None if self.fallback_to_original else text

        except Exception as e:
            logger.warning(
                f"Summary generation failed，{'fall back to text vector' if self.fallback_to_original else 'skip'}: {e}")
            return None if self.fallback_to_original else text

    def cache_size(self) -> int:
        return len(self._cache)
