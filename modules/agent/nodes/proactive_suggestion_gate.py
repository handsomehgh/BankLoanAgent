# author hgh
# version 1.0
"""
proactive suggestion gate(loan_advisor only): sits between decision and response,
decides whether the reply should carry a proactive loan-intent registration invitation.
hard rules(stage/whitelist -> cooldown -> duplicate check) pre-filter cheaply,
only qualifying turns pay for one SuggestionTimingClassifier call.
the gate never emits user-visible text,it only writes proactive_hint into state;
any internal failure degrades to "no hint" and never blocks the reply.
"""
import asyncio
import logging
from typing import Dict, Any, Optional, Set

from langchain_core.runnables import RunnableConfig

from infra.cache.cache_manager import CacheManager
from infra.database.mysql_manager import DatabaseManager
from infra.repository.LoanInterestRepository import LoanInterestRepository
from modules.agent.constants import StateFields, ReplyStage
from modules.agent.multi_agent_state import AgentContext
from modules.module_services.suggestion_timing_classifier import SuggestionTimingClassifier

logger = logging.getLogger(__name__)

# only calculation-class tools may trigger an invitation,pure queries(comparing products etc.) are excluded
TRIGGER_TOOLS = frozenset({
    "calculate_monthly_payment",
    "calculate_max_loan_amount",
    "calculate_loan_total_cost",
    "generate_repayment_schedule",
    "check_loan_eligibility",
})
# once an invitation is emitted,stay silent for a full day regardless of the user's response
COOLDOWN_TTL_SECONDS = 24 * 3600
# records in these statuses count as "already registered"
ACTIVE_STATUSES = {"待处理", "处理中"}
NO_HINT = "无"
# keep the classifier input bounded so a huge tool output cannot blow up the prompt
TOOL_FACTS_MAX_CHARS = 1500


class ProactiveSuggestionGate:

    def __init__(
            self,
            classifier: SuggestionTimingClassifier,
            cache_manager: CacheManager,
            db_manager: DatabaseManager
    ):
        self.classifier = classifier
        self.cache_manager = cache_manager
        self.db_manager = db_manager

    async def check(self, state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        """gate node entry:always returns a proactive_hint value,never raises"""
        try:
            hint = await self._evaluate(state)
        except Exception as e:
            logger.exception("proactive suggestion gate failed,pass through without hint: %s", e)
            hint = None
        return {StateFields.PROACTIVE_HINT.value: hint or NO_HINT}

    async def _evaluate(self, state: Dict[str, Any]) -> Optional[str]:
        """returns the hint text when an invitation should be emitted,otherwise None"""
        # 1. only a successful tool turn can be a candidate
        if state.get(StateFields.REPLY_STAGE.value) != ReplyStage.FINAL.value:
            return None
        payload = state.get(StateFields.REPLY_PAYLOAD.value) or {}
        tool_names = payload.get("tool_names") or []
        if not TRIGGER_TOOLS.intersection(tool_names):
            return None

        context: AgentContext = state.get(StateFields.AGENT_CONTEXT.value)
        user_id = context.user_id

        # 2. cheapest check first:cooldown in Redis(sync backend,offload to thread)
        cooldown_key = self.cache_manager.build_key("invite", user_id)
        if await asyncio.to_thread(self.cache_manager.get, cooldown_key):
            logger.debug("user %s in suggestion cooldown,skip", user_id)
            return None

        # 3. duplicate registration check:one indexed query for all records of the user
        records = await asyncio.to_thread(self._find_user_records, user_id)
        registered_types: Set[str] = {r.loan_type for r in records if r.status in ACTIVE_STATUSES}

        # 4. semantic judgment:information completeness / user attitude / refusal memory
        result = await self.classifier.judge(
            recent_conversation=context.recent_conversation or "",
            conversation_summary=context.conversation_summary,
            user_profile=context.user_profile_summary,
            tool_facts=str(payload.get("tool_facts_text") or "")[:TOOL_FACTS_MAX_CHARS],
            registered_types=registered_types,
        )
        if not result.get("should_suggest"):
            logger.info("suggestion declined for user %s: %s", user_id, result.get("reason"))
            return None

        # 5. anti-hallucination:the classifier's loan_type must not collide with registered records
        loan_type = result.get("loan_type") or ""
        if not loan_type or loan_type in registered_types:
            logger.info("suggestion dropped,colliding loan_type=%r for user %s", loan_type, user_id)
            return None

        # 6. emit:write cooldown first so a crash right after cannot double-invite
        await asyncio.to_thread(self.cache_manager.set, cooldown_key, "1", COOLDOWN_TTL_SECONDS)
        logger.info("proactive suggestion emitted for user %s, loan_type=%s", user_id, loan_type)
        return (
            f"用户刚完成一次{loan_type}的测算，且对结果未表现出不满，已知信息足以登记贷款意向。"
            f"请在本轮回复的结尾，用一句自然、温和的话询问用户是否需要登记贷款意向单，方便客户经理跟进，"
            f"并给用户留出拒绝的余地。此时不要询问联系时间，联系时间只在用户明确同意登记后再追问。"
        )

    def _find_user_records(self, user_id: str):
        """sync DB access,follows the same session pattern as upsert_loan_interest tool"""
        session = self.db_manager.create_session()
        try:
            repository = LoanInterestRepository(session)
            return repository.find_by_user(user_id)
        finally:
            session.close()
