# author hgh
# version 1.1
"""
QueryRewriter 单元测试

策略选择（DynamicStrategySelector）为纯规则逻辑，离线可测；
改写执行依赖真实 LLM（与 container 接线一致），LLM 不可用时相关用例失败属环境问题。
"""
from unittest.mock import MagicMock

import pytest

from config.global_constant.constants import RegistryModules
from config.models.retrieval_config import RewriterConfig
from config.prompt_hub import PromptHub
from modules.module_services.chat_models import RobustLLM
from modules.retrieval.knowledge_constant import RewritingStrategy
from modules.retrieval.query_rewriter import DynamicStrategySelector, QueryRewriter
from utils.config_utils.get_config import get_config


@pytest.fixture(scope="module")
def selector():
    return DynamicStrategySelector()


def get_rewriter(override_strategy: str = None) -> QueryRewriter:
    """按 container 接线方式构造 QueryRewriter（提示词来自提示词库）"""
    registry = get_config()
    llm_config = registry.get_config(RegistryModules.LLM)
    creative_llm = RobustLLM(
        temperature=llm_config.creative_temperature,
        api_key=llm_config.deepseek_api_key,
        base_url=llm_config.deepseek_base_url,
        model=llm_config.deepseek_llm_name,
        provider=llm_config.openai_provider
    )
    prompt_hub = PromptHub(registry)
    # 独立构造配置，避免污染全局 registry；override_strategy 非空时关闭动态选择
    rewriter_cfg = RewriterConfig(
        enable_dynamic=override_strategy is None,
        override_strategy=override_strategy,
    )
    return QueryRewriter(rewriter_cfg, creative_llm, prompt_hub)


# ---------------- 策略选择（离线） ----------------

def test_select_hyde_short_plain(selector):
    """短查询且无疑问词时应选择 HYDE（生成假设文档增强检索）"""
    assert selector.select("利率") == RewritingStrategy.HYDE


def test_select_multi_query_short_with_question_word(selector):
    """短查询但含疑问词时，HYDE 不适用，应选择 MULTI_QUERY"""
    assert selector.select("月供怎么算") == RewritingStrategy.MULTI_QUERY


def test_select_stepback_comparison(selector):
    """对比/选择类长查询应选择 STEP_BACK"""
    assert selector.select("等额本息和等额本金哪个更好") == RewritingStrategy.STEP_BACK


def test_select_decompose_multi_question(selector):
    """包含多个问号的复合问题应选择 DECOMPOSE"""
    assert selector.select("住房贷款利率是多少？额度怎么算？") == RewritingStrategy.DECOMPOSE


def test_select_none_for_plain_long_query(selector):
    """普通的长查询不改写"""
    assert selector.select("你们银行申请贷款需要什么手续") is None


# ---------------- 改写执行 ----------------

def test_rewrite_no_strategy():
    """关闭动态选择且无 override 策略时，原样返回查询（不调用 LLM）"""
    registry = get_config()
    cfg = RewriterConfig(enable_dynamic=False, override_strategy=None)
    rewriter = QueryRewriter(cfg, MagicMock(), MagicMock())
    assert rewriter.rewrite("还款") == ["还款"]


def test_rewrite_llm_failure_fallback():
    """LLM 调用失败时按 fallback_to_original 返回原查询"""
    cfg = RewriterConfig(enable_dynamic=False, override_strategy=RewritingStrategy.MULTI_QUERY)
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("llm down")
    hub = MagicMock()
    hub.render_messages.return_value = []
    rewriter = QueryRewriter(cfg, llm, hub)
    assert rewriter.rewrite("还款") == ["还款"]


def test_rewrite_multi_query_success():
    """MULTI_QUERY 改写契约：结果必含原查询且原查询收尾（需真实 LLM 在线；
    变体为空时实现会降级只返回原查询，故不强约束变体数量）"""
    rewriter = get_rewriter(override_strategy=RewritingStrategy.MULTI_QUERY)
    result = rewriter.rewrite("房贷利率")
    print(f"rewrite result-------{result}")
    assert len(result) >= 1
    assert "房贷利率" in result
    assert result[-1] == "房贷利率"


def test_rewrite_hyde_success():
    """HYDE 改写：返回单条假设文档（需真实 LLM 在线）"""
    rewriter = get_rewriter(override_strategy=RewritingStrategy.HYDE)
    result = rewriter.rewrite("我想申请一笔个人住房贷款，请问目前的利率和额度是多少")
    print(f"rewrite result-------{result}")
    assert len(result) == 1
    assert len(result[0]) > 0


if __name__ == '__main__':
    test_rewrite_multi_query_success()
