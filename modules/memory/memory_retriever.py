# author hgh
# version 1.0
import logging
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Optional, List, Dict, Any

from config.global_constant.constants import MemoryType
from config.global_constant.fields import CommonFields
from config.models.memory_config import MemorySystemConfig
from modules.agent.constants import AgentName
from modules.memory.base import BaseRetriever
from modules.memory.memory_business_store.base_memory_store import BaseMemoryStore
from modules.memory.memory_constant.fields import MemoryFields
from utils.query_utils.milvus_query_builder import MilvusQueryBuilder
from utils.query_utils.query_model import Query, Condition

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
        with ThreadPoolExecutor(max_workers=len(types)) as executor:
            future_to_type = {}
            for mem_type in types:
                if mem_type == MemoryType.USER_PROFILE:
                    future = executor.submit(self.memory_store.get_all_user_profile_memories, user_id)
                elif mem_type == MemoryType.INTERACTION_LOG:
                    future = executor.submit(self.memory_store.get_recent_interactions, user_id, top_k)
                elif mem_type == MemoryType.SUB_INTERACTION_LOG:
                    future = executor.submit(self.memory_store.get_sub_recent_interactions, user_id, top_k)
                else:
                    logger.warning("Unsupported memory type: %s, skipped", mem_type)
                    results[mem_type.value] = []
                    continue
                future_to_type[future] = mem_type

            for future in as_completed(future_to_type):
                mem_type = future_to_type[future]
                try:
                    data = future.result()
                    results[mem_type.value] = data
                    logger.info("%s retrieval returned %d results", mem_type.value, len(data))
                except Exception as e:
                    logger.error("Retrieval failed for memory_type=%s, user=%s: %s", mem_type.value, user_id, e,
                                 exc_info=True)
                    results[mem_type.value] = []
        return results


if __name__ == '__main__':
    where = Query(conditions=[
        Condition(field=MemoryFields.USER_ID, op="==", value="111"),
        Condition(field=MemoryFields.STATUS, op="==", value="active"),
        Condition(field=CommonFields.SOURCE, op="in",
                  value=[AgentName.AFTER_LOAN.value, AgentName.AFTER_LOAN.value, AgentName.RISK_ASSESSMENT.value])
    ])

    print(MilvusQueryBuilder().build(where))
