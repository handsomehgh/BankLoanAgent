# author hgh
# version 1.0
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from config.constants import MemoryType


class BaseMemoryStore(ABC):
    @abstractmethod
    def add_memory(
            self,
            user_id: str,
            content: str,
            memory_type: MemoryType,
            entity_key: Optional[str] = None,
            metadata: Optional[Dict[str,Any]] = None,
            permanent: bool = False
    ) -> str:
        """
        add a long memory

        Args:
            user_id: user unique id
            content: memory content
            memory_type: memory type
            entity_key: entity type key(such as "income","occupation"),used for conflict detection
            metadata: additional metadata dictionary(such as type,confidence,source,etc.)
            permanent: whether this memory is permanent(not affected by decay or forgetting)

        Returns:
              the unique id of the newly created memory
        """
        pass

    @abstractmethod
    def search_memory(
            self,
            user_id: str,
            query: str,
            memory_type: MemoryType,
            limit: int = 3,
            min_confidence: Optional[float] = None,
            apply_decay: bool=True
    ) -> List[Dict[str,Any]]:
        """
        semantic retrieval memory

        Args:
            user_id: user unique id
            query: search query text
            limit: number of results to return
            memory_type: filter by memory type(such as "user_profile")
            min_confidence: minimum confidence threshold
            apply_decay: whether to apply time decay re-ranking
        """
        pass

    @abstractmethod
    def get_memory_by_entity(
            self,user_id: str,
            entity_key: str,
            status: str = "active"
    ) -> List[Dict[str,Any]]:
        """
        retrieval memory using physical keys(for conflict detection)

        Args:
            user_id: user unique id
            entity_key: entity type key(such as "income","occupation")
            status: memory status("active","superseded","forgotten")

        Returns:
            list of qualifying memories
        """
        pass

    @abstractmethod
    def update_memory_status(
            self,
            memory_id: str,
            new_status: str,
            metadate_updates: Optional[Dict[str,Any]]
    ) -> bool:
        """
        update memory status(soft delete,mark as overwritten)

        Args:
            memory_id: unique memory id
            new_status: new memory status
            metadate_updates: other metadata that nees to be updated

        Returns:
            True if update was successful, False otherwise
        """
        pass

    @abstractmethod
    def apply_forgetting(
            self,
            memory_type: MemoryType,
            user_id: Optional[str] = None,
            threshold: Optional[float] = None
    ) -> int:
        """
        apply forgetting rules to mark memories with weights that have decayed below the threshold as forgotten

        Args:
            memory_type: memory type
            user_id: specify user,None means all users
            threshold: forgetting threshold,None means using global configuration

        Returns:
            the number of forgotten memories
        """
        pass

    @abstractmethod
    def delete_user_memories(self,user_id: str,memory_type: Optional[MemoryType] = None) -> bool:
        """
        delete all user's memories(permanent deletion,use with caution)

        Args:
            memory_type: memory type
            user_id: user unique id

        Returns:
            True if delete was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_recent_interactions(self,user_id: str,limit: int = 5) -> List[Dict[str,Any]]:
        """
        retrieve the user's recent interaction logs,sorted in reverse chronological order

        Args:
            user_id: user unique id
            limit: number of results to return

        Returns:
            list of interaction logs
        """
        pass

    @abstractmethod
    def get_active_compliance_rules(self,limit: int = 10) -> List[Dict[str,Any]]:
        """
        get all active compliance rules,sorted by severity

        Args:
            limit: number of results to return

        Returns:
            list of compliance rules
        """
        pass

    @abstractmethod
    def get_all_user_profile_memories(self, user_id: str, status: str = "active") -> List[Dict[str, Any]]:
        """obtain all user's profile memories(used for interface display)"""
        pass

