# author hgh
# version 1.0
import hashlib
import logging
from typing import List, Dict

from langchain_core.prompts import ChatPromptTemplate

from modules.memory.memory_constant.constants import EvidenceType
from modules.module_services.chat_models import RobustLLM

logger = logging.getLogger(__name__)


class EvidenceTypeInfer:
    """证据类型推断器，规则优先 + LLM 兜底 + 内存缓存"""

    def __init__(
            self,
            llm_client: RobustLLM,
            strong_keywords: Dict[str, List[str]],
            prompt: ChatPromptTemplate
    ):
        """
        Args:
            llm_client: Low-temperature LLM client (for precise classification)
            strong_keywords: Evidence Type Keyword Mapping (from memory_config.evidence_rules.strong_keywords)
        """
        self.llm_client = llm_client
        self.strong_keywords = strong_keywords
        self.prompt = prompt
        self._cache: Dict[str, str] = {}

    def infer(self, content: str, user_messages: List[str]) -> str:
        """
        infer evidence type。
        """
        # 1. Rules take priority
        combined = (content + " " + " ".join(user_messages[-3:])).lower()
        for evidence_type, keywords in self.strong_keywords.items():
            if any(kw in combined for kw in keywords):
                logger.debug(f"Evidence type '{evidence_type}' determined by keyword rule")
                return evidence_type

        # 2. cache check
        cache_key = hashlib.md5(content.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 3. LLM judge
        conversation = "\n".join(user_messages[-3:])
        valid_types = [e.value for e in EvidenceType]
        messages = self.prompt.invoke({
            "valid_types": valid_types,
            "conversation": conversation
        }).to_messages()

        try:
            response = self.llm_client.invoke(messages).content.strip().lower()
            logger.info(f"LLM classified evidence type: {response}")
            if response in valid_types:
                self._cache[cache_key] = response
                logger.info(f"LLM classified final evidence type: {response}")
                return response
        except Exception as e:
            logger.error(f"Failed to infer evidence type: {e}")

        # 4. 最终兜底
        return EvidenceType.EXPLICIT_STATEMENT.value

    def clear_cache(self):
        """清空内部缓存"""
        self._cache.clear()
