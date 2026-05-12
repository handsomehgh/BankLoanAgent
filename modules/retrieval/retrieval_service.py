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
        self._executor = ThreadPoolExecutor(max_workers=3)
        logger.info("RetrievalService initialized with router=%s, cache=%s, number of workers=%d",
                    type(self.retrieve_router).__name__ if self.retrieve_router else "None",3)

    def retrieve(self, query: str, context: Optional[Dict] = None) -> List[BusinessKnowledge]:
        logger.info("Incoming retrieve request: query='%s...', context=%s", query[:80],
                    "available" if context else "absent")

        #router
        if self.config.retrieval_routing.enabled and self.retrieve_router and not self.retrieve_router.should_retrieve(query):
            logger.info("Query skipped by retrieve_router: %s", query[:80])
            record_retrieval_metrics({}, route_skipped=True)
            return []

        logger.info("Start retrieval for query: %s", query[:80])
        results = asyncio.run(self._retrieve_async(query, context))
        logger.info("Retrieval completed: %d results returned", len(results))
        return results

    @custom_cached(
        namespace=CacheNamespace.RAG.value,
        ttl=1800,
        null_ttl=60,
        converter=lambda data: [BusinessKnowledge(**item) for item in data] if data else [],
        empty_result_factory=list,
        ignore_args=[0,2]
    )
    async def _retrieve_async(self, query: str, context: Optional[Dict] = None) -> List[BusinessKnowledge]:
        # rewrite query
        total_start = time.monotonic()

        #query rewrite
        logger.debug("Entering _retrieve_async for query: %s", query[:80])
        queries = [query]
        if self.config.rewriter.enabled:
            queries = self.rewriter.rewrite(query, context)

        # extract conditions
        filter_expr = None
        if self.config.filter.enabled:
            filter_expr = self.filter.extract(query)
            logger.debug("Filter expression extracted: %s", filter_expr)
        else:
            logger.debug("Filter extraction disabled")

        # three-way parallel recall
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                self._executor,
                self.engine.dense_search,
                queries[0],
                None,
                filter_expr
            ),
            loop.run_in_executor(
                self._executor,
                self.engine.sparse_search,
                query,
                None,
                filter_expr
            ),
        ]

        if self.config.multi_vector.term_vector:
            tasks.append(
                loop.run_in_executor(
                    self._executor,
                    self.engine.term_search,
                    query,
                    None,
                    filter_expr
                )
            )
            logger.debug("Term vector search enabled")
        else:
            logger.debug("Term vector search disabled")

        # execute concurrently
        logger.debug("Starting parallel recall (dense + sparse + term)")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        dense_raw = results[0] if not isinstance(results[0], Exception) else []
        sparse_raw = results[1] if not isinstance(results[1], Exception) else []
        if len(results) > 2:
            term_raw = results[2] if not isinstance(results[2], Exception) else []
        else:
            term_raw = []

        # record fail log
        if isinstance(results[0], Exception):
            logger.error("Dense search failed: %s", results[0], exc_info=True)
        if isinstance(results[1], Exception):
            logger.error("Sparse search failed: %s", results[1], exc_info=True)
        if len(results) > 2 and isinstance(results[2], Exception):
            logger.error("Term search failed: %s", results[2], exc_info=True)

        logger.debug("Recall counts: dense=%d, sparse=%d, term=%d",
                     len(dense_raw) if dense_raw else 0,
                     len(sparse_raw) if sparse_raw else 0,
                     len(term_raw) if term_raw else 0)

        # RRF fusion
        fused_raw = rrf_fusion([dense_raw, sparse_raw, term_raw], k=self.config.fusion.k)
        logger.debug("RRF fusion produced %d candidates", len(fused_raw))
        candidate_pool = fused_raw[: self.config.reranker.top_k * 3]
        logger.debug("Candidate pool size after top-N cut: %d", len(candidate_pool))

        # reranker
        reranked = self.reranker.rerank(query, candidate_pool)
        logger.debug("After reranking, selected %d results", len(reranked))

        # context compressor
        compressed = self.compressor.compress(query, reranked)
        logger.debug("Context compression completed")

        #output metrics
        dense_hits = len(dense_raw)
        sparse_hits = len(sparse_raw)
        term_hits = len(term_raw) if self.config.multi_vector.term_vector else 0
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
