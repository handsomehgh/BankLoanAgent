# 提示词回归的离线断言：不依赖 LLM，在 CI/本地即可拦住"提示词改动导致标签空间退化"。
# 与 pipelines/evaluator/eval_prompt_routing.py 共用同一份 golden set，
# 在线评估测"分类效果"，这里测"标签完整性/槽位完备性"这类结构性契约。
import json
import re
import string
from pathlib import Path

import pytest
import yaml

from modules.agent.constants import RouteTarget

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_LIBRARY = PROJECT_ROOT / "config" / "rules" / "prompts_agent.yaml"
EVAL_DIR = PROJECT_ROOT / "pipelines" / "data" / "eval"

AGENT_FILES = {
    "loan_advisor": "golden_intent_loan_advisor.jsonl",
    "risk_assessment": "golden_intent_risk_assessment.jsonl",
    "after_loan": "golden_intent_after_loan.jsonl",
}
# judge 提示词中不会以 **tool** 形式列出的固定标签
FIXED_LABELS = {"DIRECT_REPLY", "CLARIFY"}
# system_prompt 必须保留的全部槽位（decision/reply 节点渲染时都会传入）
SYSTEM_PROMPT_SLOTS = {
    "user_profile", "compliance_rule", "interaction_log", "business_knowledge",
    "tool_conversation", "recent_conversation", "tool_facts", "proactive_hint", "agent_role",
}
BUSINESS_AGENTS = {RouteTarget.LOAN_ADVISOR.value, RouteTarget.RISK_ASSESSMENT.value,
                   RouteTarget.AFTER_LOAN.value}
# router 提示词约定的输出标签：问候/闲聊由前置 DirectReplyNode 拦截，不会到达 supervisor，
# 所以 router 标签空间没有 direct（direct 仅是节点解析层的容错分支）
PROMPT_TARGETS = BUSINESS_AGENTS | {RouteTarget.UNKNOWN.value, RouteTarget.HUMAN_HANDOFF_NOTIFY.value}
PARSEABLE_TARGETS = PROMPT_TARGETS | {RouteTarget.DIRECT.value}


def _load_agent_library():
    data = yaml.safe_load(AGENT_LIBRARY.read_text(encoding="utf-8"))
    return data["prompts"]


def _extract_tool_labels(judge_text: str):
    """judge 提示词用 **tool_name** 粗体标记定义每个工具/技能标签"""
    return set(re.findall(r"\*\*([a-z][a-z0-9_]*)\*\*", judge_text))


def _load_golden(file_name: str):
    path = EVAL_DIR / file_name
    assert path.exists(), f"golden set 缺失：{path}"
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("agent_key,file_name", list(AGENT_FILES.items()))
def test_golden_labels_exist_in_judge_prompt(agent_key, file_name):
    """golden set 的每个标签必须出现在对应 judge 提示词的标签空间里,
    否则说明提示词改动删掉了标签或 golden set 过期,在线评估必然全错"""
    prompts = _load_agent_library()
    judge_text = prompts[f"{agent_key}_judge"]["text"]
    label_space = _extract_tool_labels(judge_text) | FIXED_LABELS

    samples = _load_golden(file_name)
    assert len(samples) >= 10, f"{file_name} 样本量不足，golden set 疑似被截断"
    for item in samples:
        assert item["expected"] in label_space, (
            f"{file_name} 标签 [{item['expected']}] 不在 {agent_key}_judge 标签空间中，"
            f"提示词与 golden set 已不同步"
        )


def test_judge_prompts_declare_all_tools_registered_in_dataset():
    """反向检查：golden set 应覆盖 judge 提示词声明的每个标签,
    未被覆盖的标签会给出提醒(软断言收集后统一报错),防止新增标签忘了补样本"""
    prompts = _load_agent_library()
    gaps = []
    for agent_key, file_name in AGENT_FILES.items():
        judge_text = prompts[f"{agent_key}_judge"]["text"]
        declared = _extract_tool_labels(judge_text) | FIXED_LABELS
        covered = {item["expected"] for item in _load_golden(file_name)}
        missing = declared - covered
        if missing:
            gaps.append(f"{agent_key}: 缺样本标签 {sorted(missing)}")
    assert not gaps, "golden set 标签覆盖不完整，请补充样本：" + "；".join(gaps)


def test_supervisor_golden_targets_are_parseable():
    """supervisor golden set 的目标必须落在可解析的目标空间内"""
    samples = _load_golden("golden_supervisor_routing.jsonl")
    assert len(samples) >= 15, "supervisor golden set 样本量不足"
    for item in samples:
        assert item["expected"], f"样本 expected 为空: {item['query']}"
        for target in item["expected"]:
            assert target in PARSEABLE_TARGETS, f"supervisor golden set 含非法目标 [{target}]"


def test_supervisor_router_prompt_mentions_all_targets():
    """router 提示词必须提及全部约定输出标签,否则 LLM 无从输出"""
    prompts = _load_agent_library()
    router_text = prompts["supervisor_router"]["text"].lower()
    for target in PROMPT_TARGETS:
        assert target.lower() in router_text, f"supervisor_router 提示词未提及目标 [{target}]"


def test_system_prompt_slots_complete():
    """system_prompt 的 9 个槽位必须全部存在,缺槽位会导致节点渲染时 KeyError"""
    prompts = _load_agent_library()
    text = prompts["system_prompt"]["text"]
    slots = {field.split("[")[0].split(".")[0]
             for _, field, _, _ in string.Formatter().parse(text) if field}
    missing = SYSTEM_PROMPT_SLOTS - slots
    assert not missing, f"system_prompt 缺少槽位 {sorted(missing)}，节点渲染将抛 KeyError"
