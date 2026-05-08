# author hgh
# version 1.0
import hashlib
import logging
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate

from config.global_constant.fields import CommonFields
from modules.memory.memory_constant.constants import InteractionSentiment
from modules.module_services.chat_models import RobustLLM

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """sentiment analyser"""

    def __init__(
            self,
            llm_client: RobustLLM,
            strong_keywords: Dict[str, List[str]],
            prompt: ChatPromptTemplate
    ):
        """
        Args:
            llm_client: Low-temperature LLM client (precise inference)
            strong_keywords: Emotional keyword configuration (from memory_config.sentiment_rules.strong_keywords)
            prompt: prompt
        """
        self.llm_client = llm_client
        self.strong_keywords = strong_keywords
        self.prompt = prompt
        self._cache: Dict[str, str] = {}

    def analyze(self, text: str) -> str:
        """
        Analyze the sentiment of the text.
        Args:
            text: Text to be analyzed (such as a conversation summary)
        Returns:
            Emotional tags (such as "positive", "anxious", "frustrated", "neutral")
        """
        if not text or not text.strip():
            return InteractionSentiment.NEUTRAL.value

        try:
            text_lower = text.lower()

            # get from cache
            cache_key = hashlib.md5(text_lower.encode()).hexdigest()
            if cache_key in self._cache:
                return self._cache[cache_key]

            for sentiment, keywords in self.strong_keywords.items():
                if any(kw in text_lower for kw in keywords):
                    logger.debug(f"Sentiment '{sentiment}' determined by keyword rule")
                    return sentiment

            # invoke llm
            messages = self.prompt.invoke({CommonFields.TEXT: text_lower[:500]}).to_messages()
            response = self.llm_client.invoke(messages).content.strip().lower()

            valid_sentiments = [s.value for s in InteractionSentiment]
            if response in valid_sentiments:
                self._cache[cache_key] = response
            logger.debug(f"LLM classified sentiment: {response}")
            return response
        except Exception as e:
            logger.error(f"Failed to detect sentiment: {e}")
        return InteractionSentiment.NEUTRAL
