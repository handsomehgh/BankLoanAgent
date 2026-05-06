# author hgh
# version 1.0
import hashlib
import logging
from typing import Dict, Optional

from config.models.retrieval_config import RetrievalConfig
from config.prompts.faq_similar_prompt import FAQ_SIMILAR_PROMPT_TEMPLATE
from modules.module_services.chat_models import get_llm

logger = logging.getLogger(__name__)


class FaqSimilarGenerator:
    def __init__(self, config: RetrievalConfig):
        self.config = config
        multi_vector_cfg = self.config.multi_vector
        self.num_variants = multi_vector_cfg.faq_similar_config.num_variants or 3
        self.temperature = multi_vector_cfg.faq_similar_config.temperature or 0.3
        self.max_output_tokens = multi_vector_cfg.faq_similar_config.max_output_tokens or 150
        self.fallback_to_original = multi_vector_cfg.faq_similar_config.fallback_to_original or True

        self.llm_client = get_llm(self.temperature)

        self._cache: Dict[str, str] = {}

    def generate_faq(self, question: str) -> Optional[str]:
        """generate similar question,if failed return none"""
        if not question or len(question) <= 5:
            return None

        key = hashlib.md5(question.encode("utf-8")).hexdigest()
        if key in self._cache:
            logger.debug("FAQ similar question hit cache")
            return self._cache[key]

        prompt = FAQ_SIMILAR_PROMPT_TEMPLATE.format(num_variants=self.num_variants, quesiton=question)

        try:
            response = self.llm_client.invoke(prompt)
            if hasattr(response, "content"):
                variants_text = response.content.strip()
            else:
                variants_text = str(response).strip()

            lines = [
                line.strip().lstrip("0123456789.、- ").strip()
                for line in variants_text.split("\n")
                if line.strip()
            ]

            filtered = [line for line in lines if line]
            if filtered:
                combined = "。".join(filtered)
                self._cache[key] = combined
                logger.debug(f"FAQ similar question generation successful，number：{len(filtered)}")
                return combined
            else:
                logger.warning("The similar questions returned by the LLM for the FAQ are empty")
                return None if self.fallback_to_original else question
        except Exception as e:
            logger.warning(
                f"FAQ similar question generation failed，{'回退到原文' if self.fallback_to_original else '跳过'}: {e}")
            return None if self.fallback_to_original else question

    def cache_size(self) -> int:
        return len(self._cache)
