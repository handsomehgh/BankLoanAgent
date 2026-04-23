# author hgh
# version 1.0
import hashlib
import logging
from typing import Dict

from langchain_core.messages import HumanMessage

from memory.classifiers.rules.rules_loader import get_sentiment_loader
from memory.constant.constants import InteractionSentiment
from prompt.detect_setiment_prompt import DETECT_SENTIMENT_PROMPT
from utils.llm import get_llm

logger = logging.getLogger(__name__)

_sentiment_cache: Dict[str, str] = {}

llm = get_llm()

def detect_sentiment(text:str) -> str:
    """
    sentiment analysis

    Args:
        text (str): text to be analyzed

    Returns:
        sentiment enum string
    """
    text_lower = text.lower()

    loader = get_sentiment_loader()
    strong_keywords = loader.get_strong_keywords()
    for sentiment,keywords in strong_keywords.items():
        if any(kw in text_lower for kw in keywords):
            logger.debug(f"Sentiment '{sentiment}' determined by keyword rule")
            return sentiment

    #get from cache
    cache_key = hashlib.md5(text_lower.encode()).hexdigest()
    if cache_key in _sentiment_cache:
        return _sentiment_cache[cache_key]

    prompt = DETECT_SENTIMENT_PROMPT.format(text[:500])
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        valid_sentiments = [s.value for s in InteractionSentiment]
        if response in valid_sentiments:
            _sentiment_cache[cache_key] = response
            logger.debug(f"LLM classified sentiment: {response}")
            return response
    except Exception as e:
        logger.error(f"Failed to detect sentiment: {e}")
    return InteractionSentiment.NEUTRAL.value





