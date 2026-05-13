# author hgh
# version 1.0
import logging
import re
from typing import List

from config.models.retrieval_config import RuleBasedRouterConfig
from modules.retrieval.router.retrieval_base_router import RetrievalRouter

logger = logging.getLogger(__name__)


class RuleBaseRetrievalRouter(RetrievalRouter):
    def __init__(self, config: RuleBasedRouterConfig):
        self.config = config
        self._stop_patterns: List[re.Pattern] = [re.compile(p) for p in self.config.stop_patterns]

        self._has_strong = lambda q: any(kw in q for kw in config.strong_keywords)

        self.weak_signals: dict[str, int] = {}
        for item in self.config.weak_signals:
            score = item.score
            for word in item.words:
                self.weak_signals[word.lower()] = score

    def should_retrieve(self, query: str) -> bool:
        if not query:
            return False

        # stop sentences
        for pattern in self._stop_patterns:
            if pattern.search(query):
                logger.debug(f"Query matched stop pattern: {pattern.pattern}")
                return False

        # strong keywords
        if self._has_strong(query):
            return True

        # 3. 弱信号累积
        score = 0
        for word, point in self.weak_signals.items():
            if re.search(r'\b' + re.escape(word) + r'\b', query, re.IGNORECASE):
                score += point
                if score >= self.config.weak_signal_threshold:
                    return True

        return False
