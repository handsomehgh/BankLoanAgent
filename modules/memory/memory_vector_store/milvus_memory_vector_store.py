# author hgh
# version 1.0
import logging
import re
from typing import List, Any, Dict, Optional

import numpy as np
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from pymilvus import MilvusException, AnnSearchRequest, RRFRanker

from config.global_constant.constants import MemoryType, VectorQueryFields, SearchStrategy
from config.models.memory_config import MemorySystemConfig
from infra.collections import CollectionNames
from infra.milvus_client import MilvusClientManager
from modules.memory.memory_constant.fields import MemoryFields
from modules.module_services.embeddings import RobustEmbeddings
from modules.memory.models.memory_schema import UserProfileMemory, InteractionLogMemory, \
    ComplianceRuleMemory
from modules.memory.memory_vector_store.base_vector_store import BaseVectorStore
from utils.model_mapper.model_to_storage import MemoryToStorageMapper
from utils.query_utils.milvus_query_builder import MilvusQueryBuilder
from utils.query_utils.query_model import Query
from utils.retry import retry_on_failure

logger = logging.getLogger(__name__)

_MODEL_CLASSES = {
    MemoryType.USER_PROFILE: UserProfileMemory,
    MemoryType.INTERACTION_LOG: InteractionLogMemory,
    MemoryType.COMPLIANCE_RULE: ComplianceRuleMemory
}


class MilvusMemoryVectorStore(BaseVectorStore):
    """
    milvus vector storage implementation
    support hybrid search,keyword search,MMR retrieve and has the ability to choose search strategies dynamically
    """

    def __init__(self, milvus_client: MilvusClientManager, embed: RobustEmbeddings, config: MemorySystemConfig):
        """
        initialize milvus vector store

        Args:
            milvus_client: the client of milvus
        """
        self.client = milvus_client
        self.config = config
        self._embedding_model = embed
        self._query_builder = MilvusQueryBuilder()
        logger.info("MilvusMemoryVectorStore initialized")

    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        """convert text to dense vector"""
        vectors = self._embedding_model.embed_documents(texts)
        logger.debug(f"Embedded {len(texts)} texts into vectors")
        return vectors

    def _parse_search_results(self, results, output_fields=None) -> List[Dict]:
        formatted = []
        for hits in results:
            for hit in hits:
                item = {
                    MemoryFields.ID: hit.id,
                    VectorQueryFields.DISTANCE: hit.distance,
                    VectorQueryFields.SCORE: hit.score,
                }

                if output_fields:
                    for field in output_fields:
                        value = hit.get(field)
                        if value is not None:
                            item[field] = value
                formatted.append(item)

        return formatted

    def _get_all_output_fields(self, memory_type: MemoryType) -> List[str]:
        model_class = _MODEL_CLASSES[memory_type]
        fields = set(model_class.model_fields.keys())
        fields.add(MemoryFields.TEXT)
        return list(fields)

    @retry_on_failure(max_retries=3, initial_delay=0.3, exceptions=(MilvusException,))
    def add(
            self,
            memory_type: MemoryType,
            ids: List[str],
            texts: List[str],
            models: List[Any]
    ) -> None:
        """batch insert into vector data"""

        # get collection by memory type
        coll_name = CollectionNames.for_type(memory_type)
        collection = self.client.get_collection(coll_name)
        logger.debug(f"Preparing to insert {len(ids)} records into {coll_name}")

        # vectorized texts
        dense_vectors = self._embed_text(texts)

        # organize insert data
        meta_dicts = [MemoryToStorageMapper.to_db_meta(m, target_db=self.config.vector_backend) for m in models]
        rows = []
        for i, mem_id in enumerate(ids):
            row = {MemoryFields.ID: mem_id, MemoryFields.TEXT: texts[i],
                   VectorQueryFields.DENSE_VECTOR.value: dense_vectors[i]}
            # merge metadata
            for meta in meta_dicts:
                for key, value in meta.items():
                    row.setdefault(key, value)
            rows.append(row)

        try:
            collection.insert(data=rows)
            collection.flush()
            logger.info(
                f"Inserted {len(ids)} records into collection '{coll_name}' (memory_type={memory_type.value})")  # 新增
        except MilvusException:
            logger.exception(f"Failed to insert {len(ids)} records into {coll_name}")  # 新增
            raise

    @retry_on_failure(max_retries=3, initial_delay=0.2, exceptions=(MilvusException,))
    def search(
            self,
            memory_type: MemoryType,
            query: str,
            where: Optional[Query] = None,
            limit: int = 5
    ) -> List[Dict]:
        """unified search entrance"""
        search_strategy = SearchStrategy(self.config.default_search_strategy)
        # confirm the final strategy
        if search_strategy == SearchStrategy.AUTO:
            strategy = self._infer_strategy(query, memory_type)
        else:
            strategy = search_strategy

        coll_name = CollectionNames.for_type(memory_type)
        logger.info(
            f"Search in {coll_name} (strategy={strategy.value}) for query '{query[:60]}...'"
        )

        output_fields = self._get_all_output_fields(memory_type)
        expr = self._query_builder.build(where) if where else None

        # execute retrieve
        try:
            if strategy == SearchStrategy.HYBRID:
                results = self._hybrid_search(memory_type, query, expr, limit, output_fields)
            elif strategy == SearchStrategy.SEMANTIC:
                results = self._dense_search(memory_type, query, expr, limit, output_fields)
            elif strategy == SearchStrategy.KEYWORD:
                results = self._keyword_search(memory_type, query, expr, limit)
            elif strategy == SearchStrategy.MMR:
                results = self._mmr_search(memory_type, query, expr, limit)
            else:
                raise ValueError(f"Unsupported search strategy: {strategy}")
            logger.info(f"Search returned {len(results)} results from {coll_name}")
            return results
        except Exception:
            logger.exception(f"Search failed in {coll_name}")
            raise

    @retry_on_failure(max_retries=3, initial_delay=0.2, exceptions=(MilvusException,))
    def get(
            self,
            memory_type: MemoryType,
            where: Optional[Query] = None,
            ids: Optional[List[str]] = None,
            limit: Optional[int] = None,
    ) -> List[Dict]:
        # obtain collection
        coll_name = CollectionNames.for_type(memory_type)
        collection = self.client.get_collection(coll_name)

        # organize query_utils conditions
        parts = []
        if where:
            from_where = self._query_builder.build(where)
            if from_where:
                parts.append(f"{from_where}")
        if ids:
            quoted_ids = [f'"{id}"' for id in ids]
            parts.append(f"{MemoryFields.ID} in [{', '.join(quoted_ids)}]")
        expr = " AND ".join(parts) if parts else f"{MemoryFields.ID} != ''"

        output_fields = self._get_all_output_fields(memory_type)

        query_params = {
            "expr": expr,
            "output_fields": output_fields
        }
        if limit is not None:
            query_params["limit"] = limit

        # execute query_utils
        logger.debug(f"Get query on {coll_name} with expr='{expr}', limit={limit}")
        try:
            results = collection.query(**query_params)
            logger.info(f"Get returned {len(results)} records from {coll_name}")
            return results
        except MilvusException:
            logger.exception(f"Get query failed on {coll_name}")
            raise

    @retry_on_failure(max_retries=3, initial_delay=0.2, exceptions=(MilvusException,))
    def update(
            self,
            memory_type: MemoryType,
            ids: List[str],
            metadatas: List[Dict[str, Any]]
    ) -> None:
        if not ids or not metadatas:
            return

        coll_name = CollectionNames.for_type(memory_type)
        collection = self.client.get_collection(coll_name)

        data = []
        for mem_id, meta in zip(ids, metadatas):
            row = {MemoryFields.ID: mem_id}
            row.update(meta)
            data.append(row)

        try:
            collection.upsert(data, partial_update=True)
            collection.flush()
            logger.info(f"Partial updated {len(data)} records in {coll_name}")
        except MilvusException:
            logger.exception(f"Partial update failed on {coll_name}")

    @retry_on_failure(max_retries=3, initial_delay=0.2, exceptions=(MilvusException,))
    def delete(
            self,
            memory_type: MemoryType,
            where: Optional[Query] = None
    ) -> None:
        coll_name = CollectionNames.for_type(memory_type)
        collection = self.client.get_collection(coll_name)
        if where:
            expr = self._query_builder.build(where)
            if expr:
                try:
                    collection.delete(expr)
                    logger.info(f"Deleted records matching '{expr}' from {coll_name}")
                except MilvusException:
                    logger.exception(f"Delete failed on {coll_name}")
                    raise
        else:
            logger.warning(f"Delete called on {coll_name} with no condition")

    def _dense_search(
            self,
            memory_type: MemoryType,
            query: str,
            expr: Optional[str],
            limit: int,
            output_fields: List[str]
    ) -> List[Dict]:
        """pure dense vector search"""
        coll_name = CollectionNames.for_type(memory_type)
        collection = self.client.get_collection(coll_name)

        # vectorize query_utils content
        dense_vec = self._embed_text([query])[0]

        # organize search param
        search_params = {"metric_type": "COSINE", "params": {"ef": 60}}
        logger.debug(f"Dense search on {coll_name} with limit={limit}")

        try:
            results = collection.search(
                data=[dense_vec],
                anns_field=VectorQueryFields.DENSE_VECTOR.value,
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=output_fields
            )
            return self._parse_search_results(results, output_fields)
        except MilvusException:
            logger.exception(f"Dense search failed on {coll_name}")
            raise

    def _keyword_search(
            self,
            memory_type: MemoryType,
            query: str,
            expr: Optional[str],
            limit: int
    ) -> List[Dict]:
        """full-text keyword-search"""
        coll_name = CollectionNames.for_type(memory_type)
        collection = self.client.get_collection(coll_name)

        # use TEXT_MATCH for keyword search
        filter_expr = f"TEXT_MATCH({MemoryFields.TEXT},'{query}')"
        if expr:
            filter_expr = f"({filter_expr}) AND {expr}"
        logger.debug(f"Keyword search on {coll_name} with filter='{filter_expr}'")

        try:
            results = collection.query(
                expr=filter_expr,
                limit=limit,
                output_fields=self._get_all_output_fields(memory_type)
            )
            return results
        except MilvusException:
            logger.exception(f"Keyword search failed on {coll_name}")
            raise

    def _hybrid_search(
            self,
            memory_type: MemoryType,
            query: str,
            expr: Optional[str],
            limit: int,
            output_fields: List[str]
    ) -> List[Dict]:
        """dense vector and sparse vector search"""
        coll_name = CollectionNames.for_type(memory_type)
        collection = self.client.get_collection(coll_name)

        # dense vector search
        dense_vec = self._embed_text([query])[0]
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field=VectorQueryFields.DENSE_VECTOR.value,
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit * 2,
            expr=expr
        )

        # vetor search
        sparse_vec = AnnSearchRequest(
            data=[query],
            anns_field=VectorQueryFields.SPARSE_VECTOR.value,
            param={"metric_type": "BM25"},
            limit=limit * 2,
            expr=expr
        )

        # hybrid_search
        logger.debug(f"Hybrid search on {coll_name} with limit={limit}")
        try:
            reranker = RRFRanker()
            results = collection.hybrid_search(
                reqs=[dense_req, sparse_vec],
                rerank=reranker,
                limit=limit,
                output_fields=output_fields
            )
            return self._parse_search_results(results, output_fields)
        except MilvusException:
            logger.exception(f"Hybrid search failed on {coll_name}")
            raise

    def _mmr_search(
            self,
            memory_type: MemoryType,
            query: str,
            expr: Optional[str],
            limit: int,
            diversity: float = 0.7,
    ) -> List[Dict]:
        """MMR diversity search"""
        candidates = self._hybrid_search(
            memory_type, query, expr, limit * 2,
            output_fields=self._get_all_output_fields(memory_type)
        )

        if not candidates:
            return []
        logger.debug(f"MMR search on {CollectionNames.for_type(memory_type)} with {len(candidates)} candidates")

        query_emb = np.array(self._embed_text([query]))[0]
        candidate_embs = [
            self._embed_text([c[MemoryFields.TEXT]])[0] for c in candidates
        ]

        select_indices = maximal_marginal_relevance(query_emb, candidate_embs, diversity, limit)
        return [candidates[i] for i in select_indices]

    def _infer_strategy(self, query, memory_type) -> SearchStrategy:
        """dynamically select retrieval strategies based on query_utils content and memory type"""
        # global configuration override
        if self.config.default_search_strategy != SearchStrategy.AUTO.value:
            return SearchStrategy(self.config.default_search_strategy)

        # query_utils feature recognition
        if len(query) < 5 and any(c.isdigit() or c.isascii() for c in query):
            return SearchStrategy.KEYWORD

        if re.search(r"\b[A-Z]{2,}-\d+\b", query):
            return SearchStrategy.KEYWORD

        if re.search(r'(什么是|如何|怎么|推荐|比较|有没有|区别|哪个|哪款)', query):
            return SearchStrategy.HYBRID

        # revert to the default memory_type strategy
        return self._default_strategy(memory_type)

    def _default_strategy(self, memory_type) -> SearchStrategy:
        if memory_type == MemoryType.COMPLIANCE_RULE:
            return SearchStrategy.KEYWORD
        return SearchStrategy.HYBRID
