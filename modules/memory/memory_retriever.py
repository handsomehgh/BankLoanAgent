# author hgh
# version 1.0
import logging
from typing import Optional, List, Dict, Any

from config.global_constant.constants import MemoryType
from config.models.memory_config import MemorySystemConfig
from modules.memory.base import BaseRetriever
from modules.memory.memory_business_store.base_memory_store import BaseMemoryStore
from modules.memory.memory_constant.fields import MemoryFields

logger = logging.getLogger(__name__)


class MemoryVectorRetriever(BaseRetriever):
    def __init__(self, memory_store: BaseMemoryStore, memory_config: MemorySystemConfig):
        self.memory_store = memory_store
        self.memory_config = memory_config

    def retrieve(
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
        top_k = top_k if top_k else self.memory_config.memory_top_k
        types = memory_types or [MemoryType.USER_PROFILE, MemoryType.INTERACTION_LOG, MemoryType.COMPLIANCE_RULE]

        logger.info(
            "Memory retrieval started for user=%s, query='%s...', top_k=%d, memory_types=%s",
            user_id, query[:60], top_k, [t.value for t in types]
        )

        results = {}
        for mem_type in types:
            try:
                if mem_type == MemoryType.USER_PROFILE:
                    results[mem_type] = self.memory_store.search_memory(
                        user_id=user_id,
                        query=query,
                        memory_type=mem_type,
                        limit=top_k,
                        apply_decay=True
                    )

                    # minimum similarity filtering
                    if len(results[mem_type]) != 0:
                        min_sim = self.memory_config.memory_min_similarity
                        filtered = [r for r in results if r.get(MemoryFields.DECAYED_SIMILARITY, 0) >= min_sim]
                        results[mem_type.value] = filtered[:top_k]
                        logger.info(
                            "User profile retrieval: %d raw, %d filtered (min_sim=%.2f)",
                            len(results[mem_type]), len(filtered), min_sim
                        )
                    else:
                        results[mem_type.value] = []
                        logger.info("User profile retrieval returned empty")

                elif mem_type == MemoryType.INTERACTION_LOG:
                    results[mem_type.value] = self.memory_store.get_recent_interactions(
                        user_id=user_id,
                        limit=top_k
                    )
                    logger.info("Interaction log retrieval returned %d results for user=%s",
                                len(results[mem_type.value]), user_id)
                elif mem_type == MemoryType.COMPLIANCE_RULE:
                    results[mem_type.value] = self.memory_store.get_active_compliance_rules(
                        limit=top_k * 2
                    )
                    logger.info("Compliance rule retrieval returned %d active rules", len(results[mem_type.value]))
                else:
                    logger.warning("Unsupported memory type: %s, skipped", mem_type)
                    results[mem_type.value] = []
            except Exception as e:
                logger.error("Retrieval failed for memory_type=%s, user=%s: %s", mem_type.value, user_id, e, exc_info=True)
                results[mem_type.value] = []  # degrade: return an empty list, do not block the overall process
        return results
