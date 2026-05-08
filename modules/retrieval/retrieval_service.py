# author hgh
# version 1.0
import asyncio
import logging
from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Optional, Dict

from config.global_constant.constants import MemoryType
from config.models.retrieval_config import RetrievalConfig
from modules.retrieval.context_compressor import ContextCompressor
from modules.retrieval.knowledge_model import BusinessKnowledge
from modules.retrieval.knowledge_vector_store.knowledge_search_engine import KnowledgeSearchEngine
from modules.retrieval.query_filter import QueryFilter
from modules.retrieval.query_rewriter import QueryRewriter
from modules.retrieval.rereanker import Reranker
from modules.retrieval.rrf_fusion import rrf_fusion
from utils.model_mapper.storage_to_model import StorageToMemoryMapper

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(
            self,
            engine: KnowledgeSearchEngine,
            rewriter: QueryRewriter,
            filter: QueryFilter,
            reranker: Reranker,
            compressor: ContextCompressor,
            config: RetrievalConfig):
        self.engine = engine
        self.rewriter = rewriter
        self.filter = filter
        self.reranker = reranker
        self.compressor = compressor
        self.config = config
        self._executor = ThreadPoolExecutor(max_workers=3)

    def retrieve(self,query: str,context: Optional[Dict] = None) -> List[BusinessKnowledge]:
        return asyncio.run(self._retrieve_async(query,context))

    async def _retrieve_async(self, query: str, context: Optional[Dict] = None) -> List[BusinessKnowledge]:
        #rewrite query
        queries = self.rewriter.rewrite(query)

        #extract conditions
        filter_expr = None
        if self.config.filter.enabled:
            filter_expr = self.filter.extract(query)

        #three-way parallel recall
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

        #execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        dense_raw = results[0] if not isinstance(results[0], Exception) else []
        sparse_raw = results[1] if not isinstance(results[1], Exception) else []
        if len(results) > 2:
            term_raw = results[2] if not isinstance(results[2], Exception) else []
        else:
            term_raw = []

        # record fail log
        if isinstance(results[0], Exception):
            logger.error(f"Dense search failed: {results[0]}")
        if isinstance(results[1], Exception):
            logger.error(f"Sparse search failed: {results[1]}")
        if len(results) > 2 and isinstance(results[2], Exception):
            logger.error(f"Term search failed: {results[2]}")

        #RRF fusion
        fused_raw = rrf_fusion([dense_raw, sparse_raw, term_raw],k=self.config.fusion.k)
        candidate_pool = fused_raw[: self.config.reranker.top_k * 3]

        #reranker
        reranked = self.reranker.rerank(query,candidate_pool)

        #context compressor
        compressed = self.compressor.compress(query,reranked)

        final_results = [StorageToMemoryMapper.from_db_dict(item, MemoryType.BUSINESS_KNOWLEDGE) for item in compressed]
        return final_results









