# author hgh
# version 1.0
"""
reranker module: user cross_encoder for fine ranking
"""
import logging
from typing import List, Dict

from sentence_transformers import CrossEncoder

from config.global_constant.fields import CommonFields
from config.models.retrieval_config import RerankerConfig

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, config: RerankerConfig):
        self.config = config
        self.model = CrossEncoder(self.config.model_name, max_length=512)
        logger.info("Reranker initialized with model=%s, max_length=%d", self.config.model_name, 512)

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        logger.debug("Reranker called with %d candidates, query='%s...'", len(candidates), query[:60])
        if not candidates:
            logger.debug("Reranker received empty candidate list, returning empty")
            return []
        try:
            pairs = [[query, item[CommonFields.TEXT]] for item in candidates]
            scores = self.model.predict(pairs, batch_size=self.config.batch_size, show_progress_bar=False)
            for item, score in zip(candidates, scores):
                item["rerank_score"] = float(score)

            # descending order
            candidates.sort(key=lambda x: x.get("rerank_score"), reverse=True)
            top_results = candidates[:self.config.top_k]
            logger.info("Reranker completed: %d -> %d results (top_k=%d)",
                        len(candidates), len(top_results), self.config.top_k)
            return top_results
        except Exception as e:
            logger.warning("Reranker failed, keep original order and return top %d. Error: %s", self.config.top_k, e,
                           exc_info=True)
            return candidates[:self.config.top_k]
