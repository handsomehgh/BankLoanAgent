# author hgh
# version 1.0
import logging
import re
from typing import List, Dict

from sentence_transformers import CrossEncoder

from config.global_constant.fields import CommonFields
from config.models.retrieval_config import CompressorConfig

logger = logging.getLogger(__name__)

"""
split the original document content into sentences,
and user a cross encoder to rank and select the most appropriate content
"""

class ContextCompressor:
    def __init__(self, config: CompressorConfig):
        self.config = config
        self.model = CrossEncoder(model_name_or_path=config.model_name, max_length=512)

    def compress(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        compress the text of each document,retaining key sentences,and if it fails,keep the original text
        """
        if not self.config.enabled:
            logger.debug("Context compression is disabled, returning %d documents as-is", len(documents))
            return documents

        original_count = len(documents)
        compressed_count = 0
        for item in documents:
            text = item.get(CommonFields.TEXT, "")
            if len(text) <= self.config.max_context_tokens * 2:
                continue
            try:
                sentences = self._split_sentences(text)
                if len(sentences) <= self.config.sentences_to_keep:
                    continue
                pairs = [[query, sentence] for sentence in sentences]
                scores = self.model.predict(pairs, batch_size=8, show_progress_bar=False)
                top_indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)[:self.config.sentences_to_keep]
                compressed = "。".join(sentences[i] for i in top_indices) + "。"
                item[CommonFields.TEXT] = compressed

                compressed_count += 1
                logger.debug("Compressed document %s from %d to %d characters",
                             item.get(CommonFields.ID, "unknown"),
                             len(text), len(compressed))
            except Exception as e:
                logger.warning("Compression failed for document %s, keep original. Error: %s",item.get(CommonFields.ID, "unknown"), e, exc_info=True)
                if self.config.fallback_to_full:
                    continue

        logger.info("Context compression completed: %d/%d documents were compressed",
                    compressed_count, original_count)
        return documents

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r'[。！？]', text) if s.strip()]
