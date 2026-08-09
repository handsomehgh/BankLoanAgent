"""
unit tests for ProactiveSuggestionGate(loan_advisor only).
the key contracts under test:
- the gate only writes proactive_hint,it never raises and never generates user-visible text
- hard rules pre-filter in cost order:stage/whitelist -> redis cooldown -> mysql duplicate check -> classifier
- a successful invite must write the 24h cooldown before returning
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

from modules.agent import nodes as gate_module_pkg  # noqa: F401
from modules.agent.constants import StateFields, ReplyStage
from modules.agent.multi_agent_state import AgentContext
from modules.agent.nodes import proactive_suggestion_gate as gate_module
from modules.agent.nodes.proactive_suggestion_gate import ProactiveSuggestionGate, COOLDOWN_TTL_SECONDS

_NO = "无"
_TRIGGER_TOOL = "calculate_monthly_payment"


def _make_context() -> AgentContext:
    return AgentContext(
        user_id="u1", session_id="s1", trace_id="t1",
        current_query="帮我算一下月供",
        compliance_warnings=[], retrieved_knowledge=None,
        agent_instruction="", audit_logger=None,
        sub_conversation="暂无相关信息", conversation_summary="暂无相关信息",
        recent_conversation="用户:挺合理的", user_profile_summary="暂无相关信息",
    )


def _make_state(tool_names=None, stage=ReplyStage.FINAL.value):
    return {
        StateFields.AGENT_CONTEXT.value: _make_context(),
        StateFields.REPLY_STAGE.value: stage,
        StateFields.REPLY_PAYLOAD.value: {
            "tool_facts_text": "月供1000元",
            "all_messages": [],
            "tool_names": tool_names or [_TRIGGER_TOOL],
        },
    }


class FakeCacheManager:
    def __init__(self, hit=False, raise_on_get=False):
        self.store = {}
        self.hit = hit
        self.raise_on_get = raise_on_get
        self.set_calls = []

    def build_key(self, *parts):
        return ":".join(parts)

    def get(self, key):
        if self.raise_on_get:
            raise RuntimeError("redis down")
        return self.store.get(key) if not self.hit else "1"

    def set(self, key, value, ttl=None):
        self.store[key] = value
        self.set_calls.append((key, value, ttl))


class FakeRepository:
    """patched into the gate module in place of LoanInterestRepository"""
    records = []

    def __init__(self, session):
        pass

    def find_by_user(self, user_id):
        return self.records


def _make_db_manager():
    session = SimpleNamespace(close=MagicMock())
    db_manager = MagicMock()
    db_manager.create_session.return_value = session
    return db_manager, session


def _make_gate(classifier_result=None, cache=None, records=None):
    classifier = MagicMock()
    classifier.judge = AsyncMock(return_value=classifier_result or {
        "should_suggest": True, "loan_type": "住房贷款", "reason": "信息完整且用户认可"
    })
    cache = cache or FakeCacheManager()
    FakeRepository.records = records or []
    gate = ProactiveSuggestionGate(
        classifier=classifier, cache_manager=cache, db_manager=_make_db_manager()[0]
    )
    return gate, classifier, cache


def test_non_final_stage_passes_through_without_any_check(monkeypatch):
    monkeypatch.setattr(gate_module, "LoanInterestRepository", FakeRepository)
    gate, classifier, cache = _make_gate()

    result = asyncio.run(gate.check(_make_state(stage=ReplyStage.CLARIFY.value), config={}))

    assert result[StateFields.PROACTIVE_HINT.value] == _NO
    classifier.judge.assert_not_awaited()
    assert cache.set_calls == []


def test_non_trigger_tool_passes_through(monkeypatch):
    monkeypatch.setattr(gate_module, "LoanInterestRepository", FakeRepository)
    gate, classifier, cache = _make_gate()

    result = asyncio.run(gate.check(_make_state(tool_names=["query_interest_rate"]), config={}))

    assert result[StateFields.PROACTIVE_HINT.value] == _NO
    classifier.judge.assert_not_awaited()


def test_cooldown_blocks_before_db_and_classifier(monkeypatch):
    monkeypatch.setattr(gate_module, "LoanInterestRepository", FakeRepository)
    gate, classifier, _ = _make_gate(cache=FakeCacheManager(hit=True))

    result = asyncio.run(gate.check(_make_state(), config={}))

    assert result[StateFields.PROACTIVE_HINT.value] == _NO
    classifier.judge.assert_not_awaited()


def test_classifier_decline_emits_no_hint(monkeypatch):
    monkeypatch.setattr(gate_module, "LoanInterestRepository", FakeRepository)
    gate, _, cache = _make_gate(classifier_result={"should_suggest": False, "loan_type": "", "reason": "用户在比价"})

    result = asyncio.run(gate.check(_make_state(), config={}))

    assert result[StateFields.PROACTIVE_HINT.value] == _NO
    assert cache.set_calls == []


def test_registered_loan_type_blocks_hallucinated_suggestion(monkeypatch):
    monkeypatch.setattr(gate_module, "LoanInterestRepository", FakeRepository)
    records = [SimpleNamespace(loan_type="住房贷款", status="待处理")]
    gate, _, cache = _make_gate(records=records)

    result = asyncio.run(gate.check(_make_state(), config={}))

    assert result[StateFields.PROACTIVE_HINT.value] == _NO
    assert cache.set_calls == []


def test_full_pass_emits_hint_and_writes_cooldown(monkeypatch):
    monkeypatch.setattr(gate_module, "LoanInterestRepository", FakeRepository)
    records = [SimpleNamespace(loan_type="消费贷款", status="待处理")]
    gate, classifier, cache = _make_gate(records=records)

    result = asyncio.run(gate.check(_make_state(), config={}))

    hint = result[StateFields.PROACTIVE_HINT.value]
    assert hint != _NO
    assert "住房贷款" in hint
    assert "联系时间" in hint
    # classifier must see the already-registered types for its own judgement
    assert classifier.judge.await_args.kwargs["registered_types"] == {"消费贷款"}
    # cooldown written exactly once with the 24h ttl
    assert cache.set_calls == [("invite:u1", "1", COOLDOWN_TTL_SECONDS)]


def test_internal_exception_degrades_to_no_hint(monkeypatch):
    monkeypatch.setattr(gate_module, "LoanInterestRepository", FakeRepository)
    gate, _, _ = _make_gate(cache=FakeCacheManager(raise_on_get=True))

    # must not raise
    result = asyncio.run(gate.check(_make_state(), config={}))

    assert result[StateFields.PROACTIVE_HINT.value] == _NO
