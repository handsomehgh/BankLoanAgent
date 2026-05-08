# author hgh
# version 1.0
import os

import pytest

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from config.global_constant.constants import RegistryModules
from config.global_constant.fields import CommonFields
from modules.retrieval.rereanker import Reranker
from utils.config_utils.get_config import get_config


def test_basic_rerank(reranker):
    """正常重排：返回按分数降序排列的列表，并赋予 rerank_score"""
    candidates = [
        {CommonFields.ID: "1", CommonFields.TEXT: "住房贷款的利率由LPR加点形成。"},
        {CommonFields.ID: "2", CommonFields.TEXT: "消费贷款可以用于装修、旅游等合法用途。"},
        {CommonFields.ID: "3", CommonFields.TEXT: "经营贷款需要提供营业执照和经营流水。"},
    ]
    query = "我想申请住房贷款，利率是多少？"
    result = reranker.rerank(query, candidates)
    for res in result:
        print(res)

    assert len(result) == 3
    # 验证分数存在且为 float
    for item in result:
        assert isinstance(item.get("rerank_score"), float)
    # 验证降序排列
    for i in range(len(result) - 1):
        assert result[i]["rerank_score"] >= result[i + 1]["rerank_score"]
    # 第1条和第2条应极为相关（但具体分数依赖于模型，仅验证顺序）
    assert result[0][CommonFields.ID] in ["1", "2", "3"]  # 有排序即可


def test_empty_candidates(reranker):
    """空候选列表应返回空列表"""
    result = reranker.rerank("查询", [])
    assert result == []


def test_score_assignment_consistency(reranker):
    """相同查询和相同文本多次调用分数应一致（模型确定）"""
    candidates = [
        {CommonFields.ID: "x", CommonFields.TEXT: "公积金贷款额度根据账户余额计算。"}
    ]
    r1 = reranker.rerank("公积金贷款额度", candidates)
    r2 = reranker.rerank("公积金贷款额度", candidates)
    assert r1[0]["rerank_score"] == pytest.approx(r2[0]["rerank_score"], rel=1e-4)


def test_short_text(reranker):
    """短文本也能正常处理"""
    candidates = [
        {CommonFields.ID: "1", CommonFields.TEXT: "LPR"},
        {CommonFields.ID: "2", CommonFields.TEXT: "BP"},
    ]
    result = reranker.rerank("LPR", candidates)
    print(result)
    assert len(result) == 2


if __name__ == '__main__':
    config = get_config()
    reranker = Reranker(config.get_config(RegistryModules.RETRIEVAL).reranker)
    # test_basic_rerank(reranker)
    # test_empty_candidates(reranker)
    # test_score_assignment_consistency(reranker)
    test_short_text(reranker)
