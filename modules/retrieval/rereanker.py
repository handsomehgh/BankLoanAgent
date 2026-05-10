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
    def __init__(self,config: RerankerConfig):
        self.config = config
        self.model = CrossEncoder(self.config.model_name,max_length=512)

    def rerank(self,query: str,candidates: List[Dict]) -> List[Dict]:
        if not candidates:
            return []
        try:
            pairs = [[query, item[CommonFields.TEXT]] for item in candidates]
            scores = self.model.predict(pairs, batch_size=self.config.batch_size, show_progress_bar=False)
            for item, score in zip(candidates, scores):
                item["rerank_score"] = float(score)

            # descending order
            candidates.sort(key=lambda x: x.get("rerank_score"), reverse=True)
            return candidates[:self.config.top_k]
        except Exception as e:
            logger.warning(f"Reranker failed, keep original order. Error: {e}")
            return candidates[:self.config.top_k]

