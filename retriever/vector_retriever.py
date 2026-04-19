# author hgh
# version 1.0
import logging
from typing import Optional, List, Dict, Any

from config import config
from memory.base import BaseMemoryStore
from retriever.base import BaseRetriever

logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    def __init__(self, memory_store: BaseMemoryStore):
        self.memory_store = memory_store

    def retriever(
            self,
            query: str,
            user_id: str,
            top_k: int = None,
            memory_types: Optional[List[str]] = None,
            **kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        top_k = top_k if top_k else config.retrieval_top_k
        types = memory_types or ["user_profile"]
        results = {}

        for mem_type in types:
            search_user_id = "GLOBAL" if mem_type in ["business_knowledge", "compliance_rule"] else user_id
            try:
                memories = self.memory_store.search_memory(
                    user_id=search_user_id, query=query,
                    limit=top_k, memory_type=mem_type, apply_decay=True
                )
            except Exception as e:
                logger.error(f"Retrieval failed for type: {mem_type}")
                memories = []
            results[mem_type] = memories
        return results
