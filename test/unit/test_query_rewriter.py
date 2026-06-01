# author hgh
# version 1.0
from config.global_constant.constants import RegistryModules
from modules.retrieval.knowledge_constant import RewritingStrategy
from modules.retrieval.query_rewriter import DynamicStrategySelector, QueryRewriter
from utils.config_utils.get_config import get_config


def get_rewriter():
    config = get_config()
    retrieval_config = config.get_config(RegistryModules.RETRIEVAL)
    rewriter = QueryRewriter(retrieval_config.rewriter)
    return rewriter

def test_select_multi_query_short():
    """长度 <=5 应选择 MULTI_QUERY"""
    assert DynamicStrategySelector.select("利率") == RewritingStrategy.MULTI_QUERY

def test_select_multi_query_digit():
    """包含数字应选择 MULTI_QUERY"""
    assert DynamicStrategySelector.select("贷款10万") == RewritingStrategy.MULTI_QUERY

def test_select_multi_query_keyword():
    """包含特定缩写应选择 MULTI_QUERY"""
    assert DynamicStrategySelector.select("LPR是多少") == RewritingStrategy.MULTI_QUERY

def test_select_stepback_intro():
    """包含'介绍'应选择 STEP_BACK"""
    assert DynamicStrategySelector.select("介绍一下住房贷款") == RewritingStrategy.STEP_BACK

def test_select_hyde_long_and_keyword():
    """长度>30 且包含产品词应选择 HYDE"""
    long_query = "我想申请一笔个人住房贷款，请问目前的利率和额度是多少？"
    assert DynamicStrategySelector.select(long_query) == RewritingStrategy.HYDE

def test_rewrite_no_strategy():
    rewriter = get_rewriter()
    res = rewriter.rewrite("还款")
    print(f"rewrite result-------{res}")
    assert res == ["还款"]

def test_rewrite_multi_query_success():
    rewriter = get_rewriter()
    result = rewriter.rewrite("房贷利率")
    print(f"rewrite result-------{result}")
    assert "房贷利率" in result
    assert len(result) >= 2
    assert result[-1] == "房贷利率"

def test_rewrite_stepback_success():
    rewriter = get_rewriter()
    result = rewriter.rewrite("等额本息和等额本金哪个好？")
    print(f"rewrite result-------{result}")
    assert isinstance(result, list)
    assert len(result) >= 1
    assert result[0] == "住房贷款的还款方式有哪些？"

def test_rewrite_hyde_success():
    rewriter = get_rewriter()
    result = rewriter.rewrite("我想申请一笔个人住房贷款，请问目前的利率和额度是多少")
    print(f"rewrite result-------{result}")
    assert len(result) == 1
    assert "LPR" in result[0] or "利率" in result[0]


if __name__ == '__main__':
    # test_select_multi_query_short()
    # test_select_multi_query_digit()
    # test_select_multi_query_keyword()
    # test_select_stepback_intro()
    # test_select_hyde_long_and_keyword()
    # test_rewrite_no_strategy()
    # test_rewrite_multi_query_success()
    # test_rewrite_stepback_success()
    test_rewrite_hyde_success()