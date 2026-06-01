# author hgh
# version 1.0
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class RetrievalRouter(ABC):
    @abstractmethod
    def should_retrieve(self,query: str) -> bool:
        """Determine whether it is necessary to perform a knowledge base search"""
        pass
