# author hgh
# version 1.0
"""
提示词回归评估流水线（意图分类 + supervisor 路由，golden set 离线驱动）

背景：
    提示词统一迁入 config/rules/prompts_*.yaml 后，任何对 judge/router 提示词的改动
    都需要回归验证"分类标签没有退化"。本脚本用人工标注的 golden set 驱动真实 LLM，
    计算精确匹配准确率，并把"提示词版本 + 指标"一起落盘，形成版本归因的评估历史。

评估对象：
    1. supervisor_router：用户问题 -> 路由目标（LoanAdvisor/RiskAssessment/AfterLoan/
       direct/unknown/human_handoff_notify，支持多目标逗号分隔）
    2. {agent}_judge：用户问题 -> 工具/技能标签（含 DIRECT_REPLY/CLARIFY）

说明：
    - judge 提示词文本本身已内置完整标签定义，渲染时 tools_metadata 传空串即可，
      分类标签空间不依赖运行时工具注册表
    - 需要本地 LLM 服务在线（vLLM），离线环境下请跑 test/unit/test_prompt_eval_offline.py

用法：
    python -m pipelines.evaluator.eval_prompt_routing                 # 全量评估
    python -m pipelines.evaluator.eval_prompt_routing --task supervisor
    python -m pipelines.evaluator.eval_prompt_routing --task intent --agent loan_advisor
"""
import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from config.global_constant.constants import RegistryModules
from config.prompt_hub import PromptHub
from modules.agent.constants import RouteTarget
from modules.module_services.chat_models import RobustLLM
from utils.config_utils.get_config import get_config
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
REPORT_PATH = EVAL_DIR / "eval_prompt_routing_report.jsonl"

AGENT_FILES = {
    "loan_advisor": "golden_intent_loan_advisor.jsonl",
    "risk_assessment": "golden_intent_risk_assessment.jsonl",
    "after_loan": "golden_intent_after_loan.jsonl",
}
BUSINESS_AGENTS = [RouteTarget.LOAN_ADVISOR.value, RouteTarget.RISK_ASSESSMENT.value,
                   RouteTarget.AFTER_LOAN.value]

SYSTEM_PROMPT_DEFAULTS = {
    "user_profile": "暂无相关信息",
    "compliance_rule": "暂无相关信息",
    "interaction_log": "暂无相关信息",
    "business_knowledge": "暂无相关信息",
    "tool_conversation": "暂无相关信息",
    "tool_facts": "暂无",
    "proactive_hint": "无",
}


def load_golden_set(file_name: str) -> List[Dict]:
    path = EVAL_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"golden set 不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_supervisor_decision(decision: str) -> List[str]:
    """与 SupervisorRouteNode._llm_route 保持一致的解析规则"""
    decision_lower = decision.lower()
    if RouteTarget.HUMAN_HANDOFF_NOTIFY.value in decision_lower:
        return [RouteTarget.HUMAN_HANDOFF_NOTIFY.value]
    if RouteTarget.DIRECT.value in decision_lower:
        return [RouteTarget.DIRECT.value]
    if RouteTarget.UNKNOWN.value in decision_lower:
        return [RouteTarget.UNKNOWN.value]
    targets = [t.strip() for t in decision.split(",") if t.strip() in BUSINESS_AGENTS]
    return targets or [RouteTarget.UNKNOWN.value]


async def eval_supervisor(hub: PromptHub, llm: RobustLLM) -> Dict:
    samples = load_golden_set("golden_supervisor_routing.jsonl")
    system_prompt = hub.get_text("supervisor_router")
    hits, details = 0, []
    for item in samples:
        human_parts = []
        if item.get("recent_conversation"):
            human_parts.append(f"最近几轮对话:\n{item['recent_conversation']}")
        human_parts.append(f"用户当前问题: {item['query']}")
        human_parts.append("请决定路由目标:")
        messages = [SystemMessage(content=system_prompt), HumanMessage(content="\n".join(human_parts))]
        try:
            response = await llm.ainvoke(messages)
            predicted = parse_supervisor_decision(response.content.strip())
        except Exception as e:
            logger.warning("supervisor 评估样本调用失败: %s", e)
            predicted = ["LLM_error"]
        expected = item["expected"]
        ok = set(predicted) == set(expected)
        hits += int(ok)
        details.append({"query": item["query"], "expected": expected,
                        "predicted": predicted, "correct": ok})
    return {
        "task": "supervisor_routing",
        "prompt_name": "supervisor_router",
        "prompt_version": hub.version_of("supervisor_router"),
        "total": len(samples), "hits": hits,
        "accuracy": round(hits / len(samples), 4) if samples else 0.0,
        "details": details,
    }


async def eval_intent(hub: PromptHub, llm: RobustLLM, agent_key: str) -> Dict:
    samples = load_golden_set(AGENT_FILES[agent_key])
    # judge 提示词文本自带标签定义，tools_metadata 置空不影响标签空间
    judge_role = hub.render_text(f"{agent_key}_judge", tools_metadata="")
    hits, details = 0, []
    for item in samples:
        format_kwargs = {
            **SYSTEM_PROMPT_DEFAULTS,
            "recent_conversation": item.get("recent_conversation", "暂无相关信息"),
            "agent_role": judge_role,
        }
        system_prompt = hub.render_text("system_prompt", **format_kwargs)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=item["query"])]
        try:
            response = await llm.ainvoke(messages)
            predicted = response.content.strip()
        except Exception as e:
            logger.warning("[%s] 评估样本调用失败: %s", agent_key, e)
            predicted = "LLM_error"
        ok = predicted == item["expected"]
        hits += int(ok)
        details.append({"query": item["query"], "expected": item["expected"],
                        "predicted": predicted, "correct": ok})
    return {
        "task": f"intent_classification_{agent_key}",
        "prompt_name": f"{agent_key}_judge",
        "prompt_version": hub.version_of(f"{agent_key}_judge"),
        "total": len(samples), "hits": hits,
        "accuracy": round(hits / len(samples), 4) if samples else 0.0,
        "details": details,
    }


async def run(args):
    registry = get_config()
    hub = PromptHub(registry)
    llm_config = registry.get_config(RegistryModules.LLM)
    # 分类任务要求稳定输出，使用 precise 温度
    llm = RobustLLM(
        temperature=llm_config.precise_temperature,
        api_key="No need",
        base_url=llm_config.local_qwen_url,
        model=llm_config.local_qwen_name,
        provider=llm_config.openai_provider,
    )

    results = []
    if args.task in ("all", "supervisor"):
        logger.info("===== 评估 supervisor 路由 =====")
        results.append(await eval_supervisor(hub, llm))
    if args.task in ("all", "intent"):
        for agent_key in ([args.agent] if args.agent else AGENT_FILES):
            logger.info("===== 评估 %s 意图分类 =====", agent_key)
            results.append(await eval_intent(hub, llm, agent_key))

    print("\n" + "=" * 72)
    print(" 提示词回归评估报告（golden set 驱动）")
    print("=" * 72)
    print(f"{'任务':<36}{'提示词版本':<12}{'准确率':<10}{'命中'}")
    for r in results:
        print(f"{r['task']:<36}{r['prompt_version']:<12}{r['accuracy']:<10.4f}{r['hits']}/{r['total']}")
        for d in r["details"]:
            if not d["correct"]:
                print(f"    错例: {d['query'][:40]}  期望={d['expected']}  预测={d['predicted']}")
    print("=" * 72)

    report = {
        "timestamp": datetime.now().isoformat(),
        "library_versions": {
            "prompts_agent": hub.registry.get_config(RegistryModules.PROMPTS_AGENT).version,
        },
        "results": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")
    logger.info("评估报告已追加保存至 %s", REPORT_PATH)


def main():
    parser = argparse.ArgumentParser(description="提示词回归评估：意图分类 + supervisor 路由")
    parser.add_argument("--task", choices=["all", "supervisor", "intent"], default="all")
    parser.add_argument("--agent", choices=list(AGENT_FILES), default=None,
                        help="task=intent 时可指定单个 agent，默认全跑")
    args = parser.parse_args()
    setup_logging(log_level="INFO")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
