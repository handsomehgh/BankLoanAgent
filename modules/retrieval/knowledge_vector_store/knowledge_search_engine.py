# author hgh
# version 1.0
import logging
from typing import Optional, List, Dict

from pymilvus import MilvusException

from config.global_constant.constants import MemoryType, VectorQueryFields
from config.models.retrieval_config import RetrievalConfig
from infra.collections import CollectionNames
from infra.milvus_client import MilvusClientManager
from modules.module_services.embeddings import RobustEmbeddings
from modules.retrieval.knowledge_model import BusinessKnowledge

logger = logging.getLogger(__name__)

class KnowledgeSearchEngine:
    def __init__(self,milvus_client: MilvusClientManager,embedder: RobustEmbeddings,config: RetrievalConfig):
        """
        knowledge search engine initialization

        Args:
            milvus_client: the client for milvus
            embedder: the embedder for milvus
            config: retrieval config
        """
        self.client = milvus_client
        self.embedder = embedder
        self.config = config
        self.collection = self.client.get_collection(CollectionNames.for_type(MemoryType.BUSINESS_KNOWLEDGE))

    def _parse_search_result(self, results,output_fields: List[str]) -> List[Dict]:
        formatted_res = []
        for hits in results:
            for hit in hits:
                item = {
                    VectorQueryFields.DISTANCE.value: hit.distance,
                    VectorQueryFields.SCORE.value: hit.score,
                }

                if output_fields:
                    for field in output_fields:
                        value = hit.get(field)
                        if value is not None:
                            item[field] = value
                formatted_res.append(item)
        return formatted_res

    def dense_search(self,query: str,top_k: Optional[int] = None,filter_expr: Optional[str] = None) -> List[Dict]:
        """
        dense vector search

        Args:
            query: the query_utils to search for
            top_k: the top k results to return
            filter_expr: the filter expression

        Returns:
            the model list of knowledge
        """
        top_k = top_k or self.config.search.dense_top_k
        try:
            query_vector = self.embedder.embed_documents([query])[0]
            search_params = {
                "metric_type": self.config.search.dense_metric_type,
                "params": {"ef": self.config.search.dense_ef}
            }
            output_fields = list(set(BusinessKnowledge.model_fields.keys()))
            result = self.collection.search(
                data=[query_vector],
                anns_field=VectorQueryFields.DENSE_VECTOR.value,
                limit=top_k,
                param=search_params,
                expr=filter_expr,
                output_fields=output_fields
            )

            return self._parse_search_result(result,output_fields)
        except MilvusException as e:
            logger.error(f"Dense search failed: {e}")
            return []

    def sparse_search(self,query: str,top_k: Optional[int] = None,filter_expr: Optional[str] = None) -> List[Dict]:
        top_k = top_k or self.config.search.sparse_top_k
        try:
            search_params = {
                "metric_type": self.config.search.bm25_metric_type
            }
            output_fields = list(set(BusinessKnowledge.model_fields.keys()))
            results = self.collection.search(
                data=[query],
                anns_field=VectorQueryFields.SPARSE_VECTOR.value,
                limit=top_k,
                param=search_params,
                expr=filter_expr,
                output_fields=output_fields
            )
            return self._parse_search_result(results,output_fields)
        except MilvusException as e:
            logger.error(f"Sparse (BM25) search failed: {e}")
            return []

    def term_search(self,query: str,top_k: Optional[int] = None,filter_expr: Optional[str] = None) -> List[Dict]:
        top_k = top_k or self.config.search.term_top_k
        try:
            query_vector = self.embedder.embed_documents([query])[0]
            search_params = {
                "metric_type": self.config.search.dense_metric_type,
                "params": {"ef": self.config.search.term_ef}
            }
            output_fields = list(set(BusinessKnowledge.model_fields.keys()))
            results = self.collection.search(
                data=[query_vector],
                anns_field=VectorQueryFields.TERM_VECTOR.value,
                limit=top_k,
                param=search_params,
                expr=filter_expr,
                output_fields=output_fields
            )
            return self._parse_search_result(results,output_fields)
        except MilvusException as e:
            logger.error(f"Term search failed: {e}")
            return []




