# author hgh
# version 1.0
import asyncio
import logging
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

    def _fetch_single_type(self, mem_type: MemoryType, user_id: str, top_k: int) -> tuple:
        """Fetch memory for a single type. Returns (mem_type, data_list)."""
        if mem_type == MemoryType.USER_PROFILE:
            data = self.memory_store.get_all_user_profile_memories(user_id)
        elif mem_type == MemoryType.INTERACTION_LOG:
            data = self.memory_store.get_recent_interactions(user_id)
        elif mem_type == MemoryType.SUB_INTERACTION_LOG:
            data = self.memory_store.get_sub_recent_interactions(user_id, top_k)
        else:
            logger.warning("Unsupported memory type: %s, skipped", mem_type)
            data = []
        return mem_type, data

    def retrieve(
            self,
            query: str,
            user_id: str,
            top_k: int = None,
            memory_types: Optional[List[MemoryType]] = None,
            **kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        retrieve multi-source memory (sync, sequential — for Skill tool callers)
        """
        top_k = top_k if top_k else self.memory_config.memory_top_k
        types = memory_types or [MemoryType.USER_PROFILE, MemoryType.INTERACTION_LOG, MemoryType.COMPLIANCE_RULE]

        logger.info(
            "Memory retrieval started for user=%s, query='%.60s', top_k=%d, memory_types=%s",
            user_id, query, top_k, [t.value for t in types]
        )

        results = {}
        for mem_type in types:
            try:
                mem_type_key, data = self._fetch_single_type(mem_type, user_id, top_k)
                results[mem_type_key.value] = data
                logger.info("%s retrieval returned %d results", mem_type_key.value, len(data))
            except Exception as e:
                logger.error("Retrieval failed for memory_type=%s, user=%s: %s", mem_type.value, user_id, e,
                             exc_info=True)
                results[mem_type.value] = []
        return results

    async def aretrieve(
            self,
            query: str,
            user_id: str,
            top_k: int = None,
            memory_types: Optional[List[MemoryType]] = None,
            **kwargs
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        retrieve multi-source memory (async, parallel via asyncio.gather + to_thread)
        — no nested ThreadPoolExecutor, uses the event loop's default thread pool instead.
        """
        top_k = top_k if top_k else self.memory_config.memory_top_k
        types = memory_types or [MemoryType.USER_PROFILE, MemoryType.INTERACTION_LOG, MemoryType.COMPLIANCE_RULE]

        logger.info(
            "Async memory retrieval started for user=%s, query='%.60s', top_k=%d, memory_types=%s",
            user_id, query, top_k, [t.value for t in types]
        )

        async def _fetch(mem_type: MemoryType) -> tuple:
            return await asyncio.to_thread(self._fetch_single_type, mem_type, user_id, top_k)

        futures = [_fetch(t) for t in types]
        raw_results = await asyncio.gather(*futures, return_exceptions=True)

        results = {}
        for res in raw_results:
            if isinstance(res, Exception):
                logger.error("Async retrieval failed: %s", res, exc_info=True)
                continue
            mem_type_key, data = res
            results[mem_type_key.value] = data
            logger.info("%s retrieval returned %d results", mem_type_key.value, len(data))
        return results


if __name__ == '__main__':
    where = Query(conditions=[
        Condition(field=MemoryFields.USER_ID, op="==", value="111"),
        Condition(field=MemoryFields.STATUS, op="==", value="active"),
        Condition(field=CommonFields.SOURCE, op="in",
                  value=[AgentName.AFTER_LOAN.value, AgentName.AFTER_LOAN.value, AgentName.RISK_ASSESSMENT.value])
    ])

    print(MilvusQueryBuilder().build(where))
