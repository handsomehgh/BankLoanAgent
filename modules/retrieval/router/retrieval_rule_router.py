# author hgh
# version 1.0
import logging
import re
from typing import List, Optional, Dict

from config.models.retrieval_config import RuleBasedRouterConfig
from modules.retrieval.router.retrieval_base_router import RetrievalRouter

logger = logging.getLogger(__name__)


class RuleBaseRetrievalRouter(RetrievalRouter):
    def __init__(self, config: RuleBasedRouterConfig):
        self.config = config
        self._stop_patterns: List[re.Pattern] = [re.compile(p) for p in self.config.stop_patterns]

    def should_retrieve(self, query: str, context: Optional[Dict] = None) -> bool:
        if not query:
            return False

        # strong keywords
        if any(kw in query for kw in self.config.strong_keywords):
            return True

        # stop sentences
        for pattern in self._stop_patterns:
            if pattern.search(query):
                logger.debug(f"Query matched stop pattern: {pattern.pattern}")
                return False

        # weak keywords and sufficient length
        if any(kw in query for kw in self.config.weak_keywords):
            if len(query) > self.config.max_query_length:
                return True
            return False

        return False
