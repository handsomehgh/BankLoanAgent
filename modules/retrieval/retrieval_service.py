# author hgh
# version 1.0
import asyncio
import logging
import threading
import time
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Optional, Dict

from config.global_constant.constants import MemoryType, CacheNamespace
from config.models.retrieval_config import RetrievalConfig
from modules.retrieval.context_compressor import ContextCompressor
from modules.retrieval.knowledge_model import BusinessKnowledge
from modules.retrieval.knowledge_vector_store.knowledge_search_engine import KnowledgeSearchEngine
from modules.retrieval.query_filter import QueryFilter
from modules.retrieval.query_rewriter import QueryRewriter
from modules.retrieval.rereanker import Reranker
from modules.retrieval.router.retrieval_base_router import RetrievalRouter
from modules.retrieval.rrf_fusion import rrf_fusion
from utils.cache_utils.cache_decorator import custom_cached
from utils.model_mapper.storage_to_model import StorageToMemoryMapper
from utils.monitor_utils.metrics import record_retrieval_metrics

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(
            self,
            engine: KnowledgeSearchEngine,
            rewriter: QueryRewriter,
            filter: QueryFilter,
            reranker: Reranker,
            compressor: ContextCompressor,
            config: RetrievalConfig,
            retrieve_router: Optional[RetrievalRouter] = None,
    ):
        self.engine = engine
        self.rewriter = rewriter
        self.filter = filter
        self.reranker = reranker
        self.compressor = compressor
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        self.config = config
        self.retrieve_router = retrieve_router
        self._executor = ThreadPoolExecutor(max_workers=9)
        logger.info("RetrievalService initialized with router=%s, number of workers=%d",
                    type(self.retrieve_router).__name__ if self.retrieve_router else "None", 3)

    @custom_cached(
        namespace=CacheNamespace.RAG.value,
        ttl=1800,
        null_ttl=60,
        converter=lambda data: [BusinessKnowledge(**item) for item in data] if data else [],
        empty_result_factory=list,
        ignore_args=[0]
    )
    def retrieve(self, query: str, context: Optional[Dict] = None, filter_expr: Optional[str] = None) -> List[
        BusinessKnowledge]:
        logger.info("Incoming retrieve request: query='%s...', context=%s", query[:80],
                    "available" if context else "absent")

        # router
        # if self.config.retrieval_routing.enabled and self.retrieve_router and not self.retrieve_router.should_retrieve(
        #         query):
        #     logger.info("Query skipped by retrieve_router: %s", query[:80])
        #     record_retrieval_metrics({}, route_skipped=True)
        #     return []

        logger.info("Start retrieval for query: %s", query[:80])
        results = asyncio.run(self._retrieve_async(query, context, filter_expr))
        logger.info("Retrieval completed: %d results returned", len(results))
        return results

    async def _retrieve_async(self, query: str, context: Optional[Dict] = None, filter_expr: Optional[str] = None) -> \
    List[BusinessKnowledge]:
        # rewrite query
        total_start = time.monotonic()

        # query rewrite
        logger.debug("Entering _retrieve_async for query: %s", query[:80])
        queries = [query]
        if self.config.rewriter.enabled:
            queries = self.rewriter.rewrite(query, context)

        # extract conditions
        if filter_expr is None and self.config.filter.enabled:
            filter_expr = self.filter.extract(query)
            logger.debug("Filter expression extracted: %s", filter_expr)
        else:
            logger.debug("Filter extraction disabled")

        # three-way parallel recall
        loop = asyncio.get_event_loop()
        all_dense, all_sparse, all_term = [], [], []

        async def recall_for_single_query(q: str):
            tasks = [
                loop.run_in_executor(self._executor, self.engine.dense_search, q, None, filter_expr),
                loop.run_in_executor(self._executor, self.engine.sparse_search, q, None, filter_expr),
            ]
            if self.config.multi_vector.term_vector:
                tasks.append(loop.run_in_executor(self._executor, self.engine.term_search, q, None, filter_expr))
            return await asyncio.gather(*tasks, return_exceptions=True)

        all_recall_results = await asyncio.gather(*[recall_for_single_query(q) for q in queries])

        for recall_results in all_recall_results:
            dense_raw = recall_results[0] if not isinstance(recall_results[0], Exception) else []
            sparse_raw = recall_results[1] if not isinstance(recall_results[1], Exception) else []
            term_raw = recall_results[2] if not isinstance(recall_results[2], Exception) else [] if len(
                recall_results) > 2 else []

            if isinstance(recall_results[0], Exception):
                logger.error("Dense search failed: %s", recall_results[0], exc_info=True)
            if isinstance(recall_results[1], Exception):
                logger.error("Sparse search failed: %s", recall_results[1], exc_info=True)
            if len(recall_results) > 2 and isinstance(recall_results[2], Exception):
                logger.error("Term search failed: %s", recall_results[2], exc_info=True)

            all_dense.extend(dense_raw if dense_raw else [])
            all_sparse.extend(sparse_raw if sparse_raw else [])
            all_term.extend(term_raw if term_raw else [])

        logger.debug("Total recall: dense=%d, sparse=%d, term=%d", len(all_dense), len(all_sparse), len(all_term))

        # RRF fusion
        fused_raw = rrf_fusion([all_dense, all_sparse, all_term], k=self.config.fusion.k)
        logger.debug("RRF fusion produced %d candidates", len(fused_raw))
        candidate_pool = fused_raw[: self.config.reranker.top_k * 3]
        logger.debug("Candidate pool size after top-N cut: %d", len(candidate_pool))

        # reranker
        reranked = self.reranker.rerank(query, candidate_pool)
        logger.debug("After reranking, selected %d results", len(reranked))

        # context compressor
        compressed = self.compressor.compress(query, reranked)
        logger.debug("Context compression completed")

        # output metrics
        dense_hits = len(all_dense)
        sparse_hits = len(all_sparse)
        term_hits = len(all_term) if self.config.multi_vector.term_vector else 0
        fused_count = len(fused_raw)
        rerank_count = len(reranked)
        duration_ms = (time.monotonic() - total_start) * 1000
        orig_total = sum(len(item.get("text", "")) for item in reranked)
        comp_total = sum(len(item.get("text", "")) for item in compressed)
        if orig_total > 0:
            comp_ratio = 1.0 - (comp_total / orig_total)
        else:
            comp_ratio = 0.0
        record_retrieval_metrics({
            'duration_ms': duration_ms,
            'dense': dense_hits,
            'sparse': sparse_hits,
            'term': term_hits,
            'fused': fused_count,
            'rerank': rerank_count,
            'comp_ratio': comp_ratio,
        })

        final_results = [StorageToMemoryMapper.from_db_dict(item, MemoryType.BUSINESS_KNOWLEDGE) for item in compressed]
        logger.info("Retrieval pipeline finished: %d final results", len(final_results))
        return final_results
