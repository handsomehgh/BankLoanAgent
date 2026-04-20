# author hgh
# version 1.0
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from config import config
from memory.base import BaseMemoryStore
from models.constant.constants import MemoryType, MetadataFields, MemoryModelFields, ChromaOperator, ChromaResFields, \
    ComplianceSeverity, MemoryStatus
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
        """
        retrieve multi-source memory
        - user_profile: semantic retrieval and decayed re-ranking
        - business_knowledge: semantic retrieval(no decay)
        - interaction_log: do not perform semantic retrieval, directly return the most recent N entries (in reverse chronological order)
        """
        top_k = top_k if top_k else config.retrieval_top_k
        types = memory_types or [MemoryType.USER_PROFILE.value, MemoryType.INTERACTION_LOG.value,
                                 MemoryType.COMPLIANCE_RULE.value]
        results = {}

        for mem_type in types:
            if mem_type == MemoryType.INTERACTION_LOG.value:
                # interaction memory
                results[mem_type] = self._retrieve_recent_interactions(
                    user_id=user_id,
                    limit=top_k
                )
            elif mem_type == MemoryType.COMPLIANCE_RULE.value:
                # compliance_rule memory
                results[mem_type] = self._retrieve_compliance_rule(limit=top_k)
                pass
            else:
                # business_knowledge memory or user profile memory
                search_user_id = "GLOBAL"
                try:
                    results[mem_type] = self.memory_store.search_memory(
                        user_id=search_user_id,
                        query=query,
                        limit=top_k,
                        memory_type=mem_type,
                        apply_decay=True if mem_type == MemoryType.USER_PROFILE.value else False
                    )
                except Exception as e:
                    logger.error(f"memory retrieval failed for {mem_type}: {e}")
                    continue
        return results

    def _retrieve_recent_interactions(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """obtain the user's recent interaction history memory,sorted in reverse chronological order by timestamp"""
        try:
            collection = self.memory_store.collection
            where_filter = {
                ChromaOperator.AND.value: [
                    {MetadataFields.USER_ID.value: {ChromaOperator.EQ.value: user_id}},
                    {MetadataFields.TYPE.value: {ChromaOperator.EQ.value: MemoryType.INTERACTION_LOG.value}},
                    {MetadataFields.STATUS.value: {ChromaOperator.EQ.value: MemoryStatus.ACTIVE.value}}
                ]
            }
            fetch_limit = limit * 3
            results = collection.get(
                where=where_filter,
                limit=fetch_limit,
                include=[ChromaResFields.DOCUMENTS.value, ChromaResFields.METADATAS.value]
            )

            memories = []
            if results[ChromaResFields.IDS.value]:
                for i, doc in enumerate(results[ChromaResFields.DOCUMENTS.value]):
                    meta = results[ChromaResFields.METADATAS.value][i]
                    memories.append({
                        MemoryModelFields.ID.value: results[ChromaResFields.IDS.value][i],
                        MemoryModelFields.CONTENT.value: doc,
                        MemoryModelFields.METADATA.value: meta,
                        MemoryModelFields.SIMILARITY.value: 1.0,
                        MemoryModelFields.DECAYED_SIMILARITY.value: 1.0
                    })

            def get_timestamp(mem):
                ts_str = mem[MemoryModelFields.METADATA.value].get(MetadataFields.TIMESTAMP.value)
                if ts_str:
                    try:
                        return datetime.fromisoformat(ts_str)
                    except Exception as e:
                        pass
                return datetime.min

            memories.sort(key=get_timestamp, reverse=True)

            return memories[:limit]
        except Exception as e:
            logger.error(f"Failed to retrieve interaction logs: {e}")
            return []

    def _retrieve_compliance_rule(self, limit: int = 10) -> List[Dict[str, Any]]:
        """get all active compliance rules(sorted by severity)"""
        try:
            # query all effective rules
            collection = self.memory_store.collection
            where_filter = {
                ChromaOperator.AND.value: [
                    {MetadataFields.TYPE.value: {ChromaOperator.EQ.value: MemoryType.COMPLIANCE_RULE.value}},
                    {MetadataFields.STATUS.value: {ChromaOperator.EQ.value: MemoryStatus.ACTIVE.value}}
                ]
            }
            results = collection.get(
                where=where_filter,
                include=[ChromaResFields.DOCUMENTS.value, ChromaResFields.METADATAS.value]
            )

            # the rule of order
            severity_order = {ComplianceSeverity.CRITICAL.value: 0, ComplianceSeverity.HIGH.value: 1,
                              ComplianceSeverity.MEDIUM.value: 2, ComplianceSeverity.LOW.value: 3,
                              ComplianceSeverity.MANDATORY.value: 0}

            # rules are sorted according to severity
            rules = []
            if results[ChromaResFields.IDS.value]:
                for i, doc in enumerate(results[ChromaResFields.DOCUMENTS.value]):
                    meta = results[ChromaResFields.METADATAS.value][i]
                    rules.append({MemoryModelFields.ID.value: results[ChromaResFields.IDS.value][i],
                                  MemoryModelFields.CONTENT.value: doc, MemoryModelFields.METADATA.value: meta})
            rules.sort(key=lambda r: severity_order.get(
                r[MemoryModelFields.METADATA.value].get(MetadataFields.SEVERITY.value, ComplianceSeverity.LOW.value),
                4))

            # return results
            return rules[:limit]
        except Exception as e:
            logger.error(f"Failed to retrieve compliance rules: {e}")
            return []
