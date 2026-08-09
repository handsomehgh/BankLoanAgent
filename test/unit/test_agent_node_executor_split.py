"""
regression tests for the AgentDecisionNode/AgentReplyNode split.
the key contract under test:
- AgentDecisionNode.decide() never calls the LLM to produce user-visible text,it only routes reply_stage/reply_payload
- AgentReplyNode.reply() is the only place that ends up producing the final AIMessage content
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, AsyncMock

import pytest
from langchain_core.messages import AIMessage

from config.global_constant.constants import RegistryModules
from config.models.agent_config import AgentsConfig, AgentConfig
from config.models.prompt_library import PromptLibrary
from exceptions.exception import ToolErrorType
from modules.agent import nodes as agent_nodes_pkg  # noqa: F401  (ensures package import works)
from modules.agent.constants import StateFields, ReplyStage
from modules.agent.multi_agent_state import AgentContext
from modules.agent.nodes import agent_decision_node as decision_module
from modules.agent.nodes.agent_decision_node import AgentDecisionNode
from modules.agent.nodes.agent_reply_node import AgentReplyNode
from modules.tools.tool_constatnt import ToolResult


def _make_context() -> AgentContext:
    return AgentContext(
        user_id="u1", session_id="s1", trace_id="t1",
        current_query="我想申请贷款",
        compliance_warnings=[], retrieved_knowledge=None,
        agent_instruction="", audit_logger=None,
        sub_conversation="暂无相关信息", conversation_summary="暂无相关信息",
        recent_conversation=None, user_profile_summary="暂无相关信息",
    )


def _make_agents_cfg():
    """提示词已迁移至提示词库,agent配置只保留行为开关；decision节点经 resolve() 按 agent 键取配置"""
    return AgentsConfig(
        default=AgentConfig(use_bert_classifier=False, tool_exposure="skills")
    )


def _make_prompt_library():
    """节点默认自建 PromptHub(registry) 查提示词库,这里提供 test_module 的桩提示词"""
    entry = lambda text: {"version": "1.0", "text": text}
    return PromptLibrary(version="1.0", prompts={
        "test_module_judge": entry("TOOLS:{tools_metadata}"),
        "test_module_execute": entry("EXEC:{tool_name}"),
        "test_module_clarify": entry("CLARIFY_ROLE"),
        "test_module_direct": entry("DIRECT_ROLE"),
        "test_module_res": entry("RES_ROLE"),
        "test_module_param_error": entry("PARAM_ERR:{error_msg}"),
        "system_prompt": entry("{agent_role}"),
    })


def _make_registry(tool_fallbacks=None):
    agents_cfg = _make_agents_cfg()
    executor_cfg = SimpleNamespace(
        circuit_breaker=SimpleNamespace(enabled=False, failure_threshold=5, recovery_timeout_sec=60),
        fallback_messages={},
        tool_fallbacks=tool_fallbacks or {"calc_tool": {"message": "计算服务暂时不可用。"}},
        response_handlers={},
    )
    prompt_library = _make_prompt_library()
    prompt_modules = {
        RegistryModules.PROMPTS_RETRIEVAL,
        RegistryModules.PROMPTS_MEMORY,
        RegistryModules.PROMPTS_AGENT,
    }

    def get_config(module):
        if module == RegistryModules.AGENT_EXECUTOR.value:
            return executor_cfg
        if module == RegistryModules.AGENTS.value:
            return agents_cfg
        if module in prompt_modules:
            return prompt_library
        return agents_cfg

    registry = MagicMock()
    registry.get_config.side_effect = get_config
    return registry


def _make_decision_node(llm_client, tool_executor=None, tools=None):
    fake_tool = SimpleNamespace(name="calc_tool", description="计算月供")
    tool_selector = MagicMock()
    tool_selector.get_tools.return_value = tools if tools is not None else [fake_tool]
    return AgentDecisionNode(
        agent_module="test_module",
        agent_name="TestAgent",
        registry=_make_registry(),
        llm_client=llm_client,
        tool_executor=tool_executor or MagicMock(),
        tool_selector=tool_selector,
    )


def _make_reply_node(llm_client):
    """reply node only needs registry/llm_client/seq_generator,no tool-related deps at all"""
    return AgentReplyNode(
        agent_module="test_module",
        agent_name="TestAgent",
        registry=_make_registry(),
        llm_client=llm_client,
        seq_generator=MagicMock(next_seq=MagicMock(return_value=1)),
    )


@pytest.fixture(autouse=True)
def _isolate_train_file(tmp_path, monkeypatch):
    """decide() writes a training sample under PROJECT_ROOT/data/wheel/<agent>/train.jsonl,
    redirect it to a tmp dir so the test suite doesn't pollute the real data folder"""
    monkeypatch.setattr(decision_module, "PROJECT_ROOT", tmp_path)


def test_decide_routes_clarify_without_generating_text():
    llm_client = MagicMock(provider="mock")
    llm_client.ainvoke = AsyncMock(return_value=AIMessage(content="CLARIFY"))
    decision_node = _make_decision_node(llm_client)
    context = _make_context()

    result = asyncio.run(decision_node.decide({StateFields.AGENT_CONTEXT.value: context}, config={}))

    assert result[StateFields.REPLY_STAGE.value] == ReplyStage.CLARIFY.value
    assert result[StateFields.REPLY_PAYLOAD.value] == {}
    # decide() must call the LLM exactly once (the judge call),it must never generate the
    # clarify reply text itself-that's reply()'s job
    assert llm_client.ainvoke.await_count == 1


def test_decide_falls_back_to_canned_message_on_judge_exception():
    llm_client = MagicMock(provider="mock")
    llm_client.ainvoke = AsyncMock(side_effect=RuntimeError("llm down"))
    decision_node = _make_decision_node(llm_client)
    context = _make_context()

    result = asyncio.run(decision_node.decide({StateFields.AGENT_CONTEXT.value: context}, config={}))

    assert result[StateFields.REPLY_STAGE.value] == ReplyStage.CANNED_FALLBACK.value
    assert "抱歉" in result[StateFields.REPLY_PAYLOAD.value]["text"]


def test_decide_routes_final_after_successful_tool_call():
    llm_client = MagicMock(provider="mock")
    tool_call_response = AIMessage(
        content="",
        tool_calls=[{"name": "calc_tool", "args": {"amount": 100000}, "id": "call1", "type": "tool_call"}],
    )
    llm_client.ainvoke = AsyncMock(side_effect=[
        AIMessage(content="calc_tool"),  # judge round
        tool_call_response,  # execute round
    ])
    tool_executor = MagicMock()
    tool_executor.execute.return_value = ToolResult(success=True, data={"monthly_payment": 1000}, summary="月供1000")
    decision_node = _make_decision_node(llm_client, tool_executor=tool_executor)
    context = _make_context()

    result = asyncio.run(decision_node.decide({StateFields.AGENT_CONTEXT.value: context}, config={}))

    assert result[StateFields.REPLY_STAGE.value] == ReplyStage.FINAL.value
    assert "月供1000" in result[StateFields.REPLY_PAYLOAD.value]["tool_facts_text"] or \
           result[StateFields.REPLY_PAYLOAD.value]["all_messages"]
    # exactly judge + execute,no extra LLM call for text generation inside decide()
    assert llm_client.ainvoke.await_count == 2


def test_decide_routes_param_error_on_tool_failure():
    llm_client = MagicMock(provider="mock")
    tool_call_response = AIMessage(
        content="",
        tool_calls=[{"name": "calc_tool", "args": {}, "id": "call1", "type": "tool_call"}],
    )
    llm_client.ainvoke = AsyncMock(side_effect=[
        AIMessage(content="calc_tool"),
        tool_call_response,
    ])
    tool_executor = MagicMock()
    tool_executor.execute.return_value = ToolResult(
        success=False, error="缺少必填参数amount", error_type=ToolErrorType.PARAMETER_ERROR
    )
    decision_node = _make_decision_node(llm_client, tool_executor=tool_executor)
    context = _make_context()

    result = asyncio.run(decision_node.decide({StateFields.AGENT_CONTEXT.value: context}, config={}))

    assert result[StateFields.REPLY_STAGE.value] == ReplyStage.PARAM_ERROR.value
    assert result[StateFields.REPLY_PAYLOAD.value]["error_msg"] == "缺少必填参数amount"


def test_reply_generates_final_text_from_decide_output():
    llm_client = MagicMock(provider="mock")
    llm_client.ainvoke = AsyncMock(return_value=AIMessage(content="您的月供预计是1000元，仅供参考。"))
    reply_node = _make_reply_node(llm_client)
    context = _make_context()

    state = {
        StateFields.AGENT_CONTEXT.value: context,
        StateFields.REPLY_STAGE.value: ReplyStage.FINAL.value,
        StateFields.REPLY_PAYLOAD.value: {"tool_facts_text": "月供1000元", "all_messages": []},
    }
    result = asyncio.run(reply_node.reply(state, config={}))

    agent_response = result[StateFields.FINAL_RESPONSE.value]
    assert agent_response.content == "您的月供预计是1000元，仅供参考。"
    assert llm_client.ainvoke.await_count == 1


def test_reply_uses_canned_text_without_calling_llm():
    llm_client = MagicMock(provider="mock")
    llm_client.ainvoke = AsyncMock(return_value=AIMessage(content="不应该被调用"))
    reply_node = _make_reply_node(llm_client)
    context = _make_context()

    state = {
        StateFields.AGENT_CONTEXT.value: context,
        StateFields.REPLY_STAGE.value: ReplyStage.CANNED_FALLBACK.value,
        StateFields.REPLY_PAYLOAD.value: {"text": "抱歉，我暂时无法处理您的问题，请稍后再试。"},
    }
    result = asyncio.run(reply_node.reply(state, config={}))

    agent_response = result[StateFields.FINAL_RESPONSE.value]
    assert agent_response.content == "抱歉，我暂时无法处理您的问题，请稍后再试。"
    llm_client.ainvoke.assert_not_awaited()
