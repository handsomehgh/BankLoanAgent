# author hgh
# version 1.0
import logging
import re
from typing import List, Any, Dict, Optional

import numpy as np
from langchain_community.vectorstores.utils import maximal_marginal_relevance
from pymilvus import connections, MilvusException, Collection, AnnSearchRequest, RRFRanker
from pymilvus.orm import utility

from config.constants import MemoryType, SearchStrategy, GeneralFieldNames, CollectionNames
from config.settings import agentConfig
from llm.embeddings import get_embeddings
from query.milvus_query_builder import MilvusQueryBuilder
from query.query_model import Query
from memory.models.memory_data.memory_schema import UserProfileMemory, InteractionLogMemory, ComplianceRuleMemory, \
    BusinessKnowledge
from memory.models.memory_mappers.mappers import MemoryToStorageMapper
from memory.memory_vector_store.base_vector_store import BaseVectorStore
from llm.retry import retry_on_failure

logger = logging.getLogger(__name__)

_MODEL_CLASSES = {
    MemoryType.USER_PROFILE: UserProfileMemory,
    MemoryType.INTERACTION_LOG: InteractionLogMemory,
    MemoryType.COMPLIANCE_RULE: ComplianceRuleMemory,
    MemoryType.BUSINESS_KNOWLEDGE: BusinessKnowledge
}


class MilvusVectorStore(BaseVectorStore):
    """
    milvus vector storage implementation
    support hybrid search,keyword search,MMR retrieve and has the ability to choose search strategies dynamically
    """

    def __init__(self, uri: str):
        """
        initialize milvus vector store

        Args:
            uri: milvus connection address
        """
        self.uri = uri
        self._collections: Dict[MemoryType, Collection] = {}
        self._embedding_model = None
        self._query_builder = MilvusQueryBuilder()
        self._connect()

    def _connect(self):
        """build milvus connection(singleton)"""
        try:
            if not connections.has_connection("default"):
                connections.connect(alias="default", uri=self.uri, timeout=30)
                logger.info(f"Connected to Milvus at {self.uri}")
        except MilvusException as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def get_embedding_model(self):
        """obtain embedding model"""
        if self._embedding_model is None:
            self._embedding_model = get_embeddings()
        return self._embedding_model

    def _embed_text(self, texts: List[str]) -> List[List[float]]:
        """convert text to dense vector"""
        model = self.get_embedding_model()
        return model.embed_documents(texts)

    def _get_collection(self, memory_type: MemoryType) -> Collection:
        """get collection by memory type"""
        if memory_type not in self._collections:
            coll_name = CollectionNames.for_type(memory_type)
            if not utility.has_collection(coll_name):
                raise RuntimeError(f"Collection {coll_name} not found, please check your configuration")
            col = Collection(coll_name)
            col.load()
            self._collections[memory_type] = col
        return self._collections[memory_type]

    def _parse_search_results(self, results, output_fields=None) -> List[Dict]:
        formatted = []
        for hits in results:
            for hit in hits:
                item = {
                    GeneralFieldNames.ID: hit.id,
                    GeneralFieldNames.DISTANCE: hit.distance,
                    GeneralFieldNames.SCORE: hit.score,
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
        fields.add(GeneralFieldNames.TEXT)
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
        collection = self._get_collection(memory_type)

        # vectorized texts
        dense_vectors = self._embed_text(texts)

        # organize insert data
        meta_dicts = [MemoryToStorageMapper.to_db_meta(m,target_db=agentConfig.vector_backend) for m in models]
        rows = []
        for i, mem_id in enumerate(ids):
            row = {GeneralFieldNames.ID: mem_id, GeneralFieldNames.TEXT: texts[i],
                   GeneralFieldNames.DENSE_VECTOR: dense_vectors[i]}
            # merge metadata
            for meta in meta_dicts:
                for key, value in meta.items():
                    row.setdefault(key, value)
            rows.append(row)

        collection.insert(data=rows)
        collection.flush()
        logger.debug(f"Inserted {len(ids)} records into {collection.name}")

    @retry_on_failure(max_retries=3, initial_delay=0.2, exceptions=(MilvusException,))
    def search(
            self,
            memory_type: MemoryType,
            query: str,
            where: Optional[Query] = None,
            limit: int = 5
    ) -> List[Dict]:
        """unified search entrance"""
        search_strategy = agentConfig.default_search_strategy
        # confirm the final strategy
        if search_strategy == SearchStrategy.AUTO:
            strategy = self._infer_strategy(query, memory_type)
        else:
            strategy = search_strategy

        logger.info(
            f"Search strategy: {strategy.value} for query "
            f"'{query[:60]}...' in {CollectionNames.for_type(memory_type)}"
        )

        output_fields = self._get_all_output_fields(memory_type)
        expr = self._query_builder.build(where) if where else None

        # execute retrieve
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

        return results

    @retry_on_failure(max_retries=3, initial_delay=0.2, exceptions=(MilvusException,))
    def get(
            self,
            memory_type: MemoryType,
            where: Optional[Query] = None,
            ids: Optional[List[str]] = None,
            limit: Optional[int] = None,
    ) -> List[Dict]:
        # obtain collection
        collection = self._get_collection(memory_type)

        # organize query conditions
        parts = []
        if where:
            from_where = self._query_builder.build(where)
            if from_where:
                parts.append(f"{from_where}")
        if ids:
            quoted_ids = [f'"{id}"' for id in ids]
            parts.append(f"{GeneralFieldNames.ID} in [{', '.join(quoted_ids)}]")
        expr = " AND ".join(parts) if parts else f"{GeneralFieldNames.ID} != ''"

        output_fields = self._get_all_output_fields(memory_type)

        query_params = {
            "expr": expr,
            "output_fields": output_fields
        }
        if limit is not None:
            query_params["limit"] = limit

        # execute query
        results = collection.query(**query_params)
        return results

    @retry_on_failure(max_retries=3, initial_delay=0.2, exceptions=(MilvusException,))
    def update(
            self,
            memory_type: MemoryType,
            ids: List[str],
            metadatas: List[Dict[str, Any]]
    ) -> None:
        if not ids or not metadatas:
            return

        collection = self._get_collection(memory_type)

        data = []
        for mem_id, meta in zip(ids, metadatas):
            row = {GeneralFieldNames.ID: mem_id}
            row.update(meta)
            data.append(row)

        # 执行部分更新
        collection.upsert(data, partial_update=True)
        collection.flush()
        logger.debug(
            f"Partial updated {len(data)} records in {CollectionNames.for_type(memory_type)}"
        )
    @retry_on_failure(max_retries=3, initial_delay=0.2, exceptions=(MilvusException,))
    def delete(
            self,
            memory_type: MemoryType,
            where: Optional[Query] = None
    ) -> None:
        collection = self._get_collection(memory_type)
        if where:
            expr = self._query_builder.build(where)
            if expr:
                collection.delete(expr)
                logger.debug(f"Deleted records matching '{expr}' in {CollectionNames.for_type(memory_type)}")

    def _dense_search(
            self,
            memory_type: MemoryType,
            query: str,
            expr: Optional[str],
            limit: int,
            output_fields: List[str]
    ) -> List[Dict]:
        """pure dense vector search"""
        collection = self._get_collection(memory_type)

        # vectorize query content
        dense_vec = self._embed_text([query])[0]

        # organize search param
        search_params = {"metric_type": "COSINE", "params": {"ef": 60}}

        results = collection.search(
            data=[dense_vec],
            anns_field=GeneralFieldNames.DENSE_VECTOR,
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=output_fields
        )
        return self._parse_search_results(results, output_fields)

    def _keyword_search(
            self,
            memory_type: MemoryType,
            query: str,
            expr: Optional[str],
            limit: int
    ) -> List[Dict]:
        """full-text keyword-search"""
        collection = self._get_collection(memory_type)

        # use TEXT_MATCH for keyword search
        filter_expr = f"TEXT_MATCH({GeneralFieldNames.TEXT},'{query}')"
        if expr:
            filter_expr = f"({filter_expr}) AND {expr}"
        results = collection.query(
            expr=filter_expr,
            limit=limit,
            output_fields=self._get_all_output_fields(memory_type)
        )
        return results

    def _hybrid_search(
            self,
            memory_type: MemoryType,
            query: str,
            expr: Optional[str],
            limit: int,
            output_fields: List[str]
    ) -> List[Dict]:
        """dense vector and sparse vector search"""
        collection = self._get_collection(memory_type)

        # dense vector search
        dense_vec = self._embed_text([query])[0]
        dense_req = AnnSearchRequest(
            data=[dense_vec],
            anns_field=GeneralFieldNames.DENSE_VECTOR,
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=limit * 2,
            expr=expr
        )

        # vetor search
        sparse_vec = AnnSearchRequest(
            data=[query],
            anns_field=GeneralFieldNames.SPARSE_VECTOR,
            param={"metric_type": "BM25"},
            limit=limit * 2,
            expr=expr
        )

        # hybrid_search
        reranker = RRFRanker()
        results = collection.hybrid_search(
            reqs=[dense_req, sparse_vec],
            rerank=reranker,
            limit=limit,
            output_fields=output_fields
        )
        return self._parse_search_results(results, output_fields)

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

        query_emb = np.array(self._embed_text([query]))[0]
        candidate_embs = [
            self._embed_text([c[GeneralFieldNames.TEXT]])[0] for c in candidates
        ]

        select_indices = maximal_marginal_relevance(query_emb, candidate_embs, diversity, limit)
        return [candidates[i] for i in select_indices]

    def _infer_strategy(self, query, memory_type) -> SearchStrategy:
        """dynamically select retrieval strategies based on query content and memory type"""
        # global configuration override
        if agentConfig.default_search_strategy != SearchStrategy.AUTO:
            return agentConfig.default_search_strategy

        # query feature recognition
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
