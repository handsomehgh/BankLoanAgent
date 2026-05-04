# author hgh
# version 1.0
import hashlib
import json
import logging
import re
from typing import Optional, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage

from config.prompts.detect_evidence_prompt import DETECT_EVIDENCE_PROMPT
from config.prompts.detect_setiment_prompt import DETECT_SENTIMENT_PROMPT
from modules.agent.constants import MessageCommonFields
from modules.memory.memory_constant.constants import EvidenceType, InteractionSentiment
from modules.module_services.chat_models import get_llm

logger = logging.getLogger(__name__)


def format_message(msg: BaseMessage) -> str:
    if isinstance(msg, HumanMessage):
        return f"用户: {msg.content}"
    elif isinstance(msg, AIMessage):
        prefix = "助手"
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            prefix += " [调用工具]"
        return f"{prefix}: {msg.content}"
    elif isinstance(msg, ToolMessage):
        return f"工具结果({msg.name}): {msg.content}"
    else:
        return f"系统: {msg.content}"


def get_message_index(msg: BaseMessage) -> Optional[int]:
    """安全获取消息的全局序号"""
    if hasattr(msg, MessageCommonFields.ADDITIONAL_KWARGS.value) and isinstance(msg.additional_kwargs, dict):
        return msg.additional_kwargs.get(MessageCommonFields.MESSAGE_INDEX.value)
    return None


def safe_parse_extraction_output(raw_output: str) -> list:
    """安全解析 LLM 提取输出，处理常见格式变异"""
    # 1. 尝试直接解析
    try:
        return json.loads(raw_output.strip())
    except json.JSONDecodeError:
        pass

    # 2. 尝试提取被 Markdown 代码块包裹的 JSON
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_output)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. 尝试找到第一个 [ 和最后一个 ] 之间的内容
    start = raw_output.find("[")
    end = raw_output.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_output[start:end + 1])
        except json.JSONDecodeError:
            pass

    # 4. 解析失败，返回空列表
    logger.warning(f"无法解析提取输出，已返回空列表。原始输出: {raw_output[:200]}")
    return []


_evidence_cache: Dict[str, str] = {}
llm = get_llm()


def infer_evidence_type(content: str, user_messages: List[str], strong_keywords: Dict[str, List[str]]) -> str:
    """
    inference evidence type

    Args:
        content (str): currently extracted user profile
        user_messages (List[str]): recent rounds of user message list(for context)

    Returns:
        evidence type enum type
    """
    combined = (content + " " + " ".join(user_messages[-3:])).lower()

    for evidence_type, keywords in strong_keywords.items():
        if any(kw in combined for kw in keywords):
            logger.debug(f"Evidence type '{evidence_type}' determined by keyword rule")
            return evidence_type

    # LLM judge
    cache_key = hashlib.md5(content.encode()).hexdigest()
    if cache_key in _evidence_cache:
        return _evidence_cache[cache_key]

    conversation = "\n".join(user_messages[-3:])
    prompt = DETECT_EVIDENCE_PROMPT.format(conversation=conversation)
    try:
        response = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
        valid_types = [e.value for e in EvidenceType]
        if response in valid_types:
            _evidence_cache[cache_key] = response
            logger.debug(f"LLM classified evidence type: {response}")
            return response
    except Exception as e:
        logger.error(f"Failed to infer evidence type: {e}")
    return EvidenceType.EXPLICIT_STATEMENT.value


_sentiment_cache: Dict[str, str] = {}


def detect_sentiment(text: str, strong_keywords: Dict[str, List[str]]) -> str:
    """
    sentiment analysis

    Args:
        text (str): text to be analyzed
        strong_keywords: sentiment strong key word

    Returns:
        sentiment enum string
    """
    text_lower = text.lower()

    for sentiment, keywords in strong_keywords.items():
        if any(kw in text_lower for kw in keywords):
            logger.debug(f"Sentiment '{sentiment}' determined by keyword rule")
            return sentiment

    # get from cache
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
