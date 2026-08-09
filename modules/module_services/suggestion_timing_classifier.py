# author hgh
# version 1.0
"""
suggestion timing classifier: judges whether NOW is the right moment to proactively
invite the user to register a loan intent,after a calculation tool has succeeded.
follows the same "focused input -> structured output" pattern as SentimentAnalyzer/EvidenceTypeInfer
"""
import json
import logging
import re
from typing import Dict, Any, Set

from langchain_core.messages import SystemMessage, HumanMessage

from config.prompt_hub import PromptHub
from modules.module_services.chat_models import RobustLLM

logger = logging.getLogger(__name__)

_NEGATIVE_RESULT: Dict[str, Any] = {"should_suggest": False, "loan_type": "", "reason": "判断失败，保守不建议"}

_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class SuggestionTimingClassifier:
    """loan-intent suggestion timing classifier,backed by a low-temperature local LLM"""

    def __init__(self, llm_client: RobustLLM, prompt_hub: PromptHub,
                 prompt_name: str = "loan_advisor_suggestion_gate"):
        """
        Args:
            llm_client: local LLM client (precise inference,does not consume the main-link quota)
            prompt_hub: 提示词统一入口
            prompt_name: 提示词库条目名（原 loan_advisor.yaml 的 suggestion_gate_prompt）
        """
        self.llm_client = llm_client
        self.prompt_hub = prompt_hub
        self.prompt_name = prompt_name

    async def judge(
            self,
            recent_conversation: str,
            conversation_summary: str,
            user_profile: str,
            tool_facts: str,
            registered_types: Set[str]
    ) -> Dict[str, Any]:
        """
        Returns:
            {"should_suggest": bool, "loan_type": str, "reason": str}
            any failure degrades to a conservative negative result
        """
        try:
            prompt = self.prompt_hub.render_text(
                self.prompt_name,
                recent_conversation=recent_conversation or "无",
                conversation_summary=conversation_summary or "无",
                user_profile=user_profile or "无",
                tool_facts=tool_facts or "无",
                registered_types="、".join(registered_types) if registered_types else "无",
            )
            messages = [SystemMessage(content=prompt), HumanMessage(content="请开始判断。")]
            response = await self.llm_client.ainvoke(messages)
            result = self._parse(response.content)
            logger.info("SuggestionTimingClassifier judged: %s", result)
            return result
        except Exception as e:
            logger.exception("SuggestionTimingClassifier judge failed: %s", e)
            return dict(_NEGATIVE_RESULT)

    @staticmethod
    def _parse(content: str) -> Dict[str, Any]:
        """extract the first JSON object from the LLM output,tolerating stray text around it"""
        if not content:
            return dict(_NEGATIVE_RESULT)
        try:
            raw = json.loads(content.strip())
        except (json.JSONDecodeError, ValueError):
            match = _JSON_PATTERN.search(content)
            if not match:
                logger.warning("SuggestionTimingClassifier output has no JSON: %s", content[:200])
                return dict(_NEGATIVE_RESULT)
            try:
                raw = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                logger.warning("SuggestionTimingClassifier output invalid JSON: %s", content[:200])
                return dict(_NEGATIVE_RESULT)

        should_suggest = raw.get("should_suggest")
        if isinstance(should_suggest, str):
            should_suggest = should_suggest.strip().lower() == "true"
        return {
            "should_suggest": bool(should_suggest),
            "loan_type": str(raw.get("loan_type") or "").strip(),
            "reason": str(raw.get("reason") or "").strip(),
        }
