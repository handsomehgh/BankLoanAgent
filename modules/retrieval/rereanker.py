# author hgh
# version 1.0
"""
reranker module: user cross_encoder for fine ranking
"""
import logging
import os
from typing import List, Dict

import requests
from sentence_transformers import CrossEncoder

from config.global_constant.fields import CommonFields
from config.models.retrieval_config import RerankerConfig

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, config: RerankerConfig):
        self.config = config
        remote_url = os.getenv("RERANKER_API_URL", config.remote_url)
        if remote_url:
            self._mode = "remote"
            self.remote_url = config.remote_url.rstrip("/")
            logger.info("Reranker initialized as remote service at %s", self.remote_url)
        else:
            self._mode = "local"
            self.model = CrossEncoder(self.config.model_name, max_length=512, local_files_only=True)
            logger.info("Reranker initialized with model=%s, max_length=%d", self.config.model_name, 512)

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        logger.debug("Reranker called with %d candidates, query='%s...'", len(candidates), query[:60])

        if not candidates:
            logger.debug("Reranker received empty candidate list, returning empty")
            return []

        try:
            documents = [item[CommonFields.TEXT] for item in candidates]
            if self._mode == "remote":
                scores = self._rerank_remote(query, documents)
            # if self._mode == "local":
            #     scores = self._rerank_local(query, documents)

            for item, score in zip(candidates, scores):
                item["rerank_score"] = float(score)

            candidates.sort(key=lambda x: x.get("rerank_score"), reverse=True)
            top_results = candidates[:self.config.top_k]
            logger.info("Reranker completed: %d -> %d results (top_k=%d)",
                        len(candidates), len(top_results), self.config.top_k)
            return top_results
        except Exception as e:
            logger.warning("Reranker failed, keep original order and return top %d. Error: %s", self.config.top_k, e,
                           exc_info=True)
            return candidates[:self.config.top_k]

    def _rerank_remote(self, query: str, documents: List[str]) -> List[float]:
        resp = requests.post(f"{self.remote_url}/rerank", json={"query": query, "documents": documents}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        scores = [0.0] * len(documents)
        for idx, score in zip(data["sorted_indices"], data["scores"]):
            scores[idx] = score
        logger.info(f"Reranker 分数: {scores[:5]}")
        return scores

    def _rerank_local(self, query: str, documents: List[str]) -> List[float]:
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs, batch_size=self.config.batch_size, show_progress_bar=False)
        return scores.tolist()
