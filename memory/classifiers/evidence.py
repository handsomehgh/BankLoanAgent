# author hgh
# version 1.0
import hashlib
import logging
from typing import Dict, List

from langchain_core.messages import HumanMessage

from memory.classifiers.rules.rules_loader import get_evidence_loader
from memory.constant.constants import EvidenceType
from prompt.detect_evidence_prompt import DETECT_EVIDENCE_PROMPT
from utils.llm import get_llm

logger = logging.getLogger(__name__)

_evidence_cache: Dict[str, str] = {}

llm = get_llm()


def infer_evidence_type(content: str, user_messages: List[str]) -> str:
    """
    inference evidence type

    Args:
        content (str): currently extracted user profile
        user_messages (List[str]): recent rounds of user message list(for context)

    Returns:
        evidence type enum type
    """
    combined = (content + " " + " ".join(user_messages[-3:])).lower()

    loader = get_evidence_loader()
    strong_keywords = loader.get_strong_keywords()

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

def get_evidence_weights() -> Dict[str,str]:
    loader = get_evidence_loader()
    return loader.get_evidence_weights()
