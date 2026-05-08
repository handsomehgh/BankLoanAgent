# author hgh
# version 1.0
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class RetrievalRouter(ABC):
    @abstractmethod
    def should_retrieve(self,query: str,context: Optional[Dict] = None) -> bool:
        """Determine whether it is necessary to perform a knowledge base search"""
        pass
