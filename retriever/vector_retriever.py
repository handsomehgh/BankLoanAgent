# author hgh
# version 1.0
import logging
from typing import Optional, List, Dict, Any

from config import config
from memory.base import BaseMemoryStore
from models.constant.constants import MemoryType
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
            memory_types: Optional[List[MemoryType]] = None,
            **kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        retrieve multi-source memory
        - user_profile: semantic retrieval and decayed re-ranking
        - compliance_rule: get all active rules and sorted by severity
        - interaction_log: do not perform semantic retrieval, directly return the most recent N entries (in reverse chronological order)
        """
        top_k = top_k if top_k else config.retrieval_top_k
        types = memory_types or [MemoryType.USER_PROFILE, MemoryType.INTERACTION_LOG, MemoryType.COMPLIANCE_RULE]

        results = {}
        for mem_type in types:
            if mem_type == MemoryType.INTERACTION_LOG:
                # interaction memory
                results[mem_type.value] = self.memory_store.get_recent_interactions(user_id, limit=top_k)
            elif mem_type == MemoryType.COMPLIANCE_RULE:
                # compliance_rule memory
                results[mem_type.value] = self.memory_store.get_active_compliance_rules(limit=top_k * 2)
            else:
                # business_knowledge memory or user profile memory
                try:
                    results[mem_type.value] = self.memory_store.search_memory(
                        user_id=user_id,
                        query=query,
                        memory_type=mem_type,
                        limit=top_k,
                        apply_decay=(mem_type == MemoryType.USER_PROFILE)
                    )
                except Exception as e:
                    logger.error(f"Memory retrieval failed for {mem_type.value}: {e}")
                    results[mem_type.value] = []
        return results
