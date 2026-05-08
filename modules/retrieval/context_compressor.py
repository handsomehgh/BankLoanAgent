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
            return documents
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
            except Exception as e:
                logger.warning(f"Compression failed, keep original. Error: {e}")
                if self.config.fallback_to_full:
                    continue
        return documents

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [s.strip() for s in re.split(r'[。！？]', text) if s.strip()]
