# author hgh
# version 1.0
from config.global_constant.fields import CommonFields
from modules.retrieval.rrf_fusion import rrf_fusion


def test_basic_fusion_two_lists():
    """两路检索结果正常融合"""
    list_a = [
        {CommonFields.ID: "A1", "text": "doc A1", "score": 0.9},
        {CommonFields.ID: "A2", "text": "doc A2", "score": 0.5},
    ]
    list_b = [
        {CommonFields.ID: "B1", "text": "doc B1", "score": 0.8},
        {CommonFields.ID: "A1", "text": "doc A1 (duplicate)", "score": 0.6},
    ]
    fused = rrf_fusion([list_a, list_b], k=60)
    for f in fused:
        print(f)

    # 应去重，A1只出现一次
    ids = [item[CommonFields.ID] for item in fused]
    assert ids.count("A1") == 1
    # 应包含所有唯一ID：A1, A2, B1
    assert set(ids) == {"A1", "A2", "B1"}
    # A1在两路中排名都很高，应排在第一
    assert fused[0][CommonFields.ID] == "A1"
    # 验证融合分数存在
    for item in fused:
        assert "rrf_score" in item


def test_fusion_with_three_lists():
    """三路检索结果正常融合"""
    list_a = [
        {CommonFields.ID: "X", "score": 1.0},
        {CommonFields.ID: "Y", "score": 0.9},
    ]
    list_b = [
        {CommonFields.ID: "Z", "score": 0.8},
        {CommonFields.ID: "X", "score": 0.7},
    ]
    list_c = [
        {CommonFields.ID: "Y", "score": 0.6},
        {CommonFields.ID: "X", "score": 0.5},
    ]
    fused = rrf_fusion([list_a, list_b, list_c], k=60)
    for f in fused:
        print(f)
    ids = [item[CommonFields.ID] for item in fused]
    assert ids == ["X", "Y", "Z"]  # X 在两路中排第1和第2，胜出


def test_all_lists_empty():
    """各子列表都为空时返回空列表"""
    fused = rrf_fusion([[], []], k=60)
    assert fused == []


def test_large_corpus():
    """模拟较大规模的融合测试，验证性能与正确性"""
    import random
    random.seed(42)
    # 创建5路检索，每路返回20个文档
    result_lists = []
    all_ids = [f"doc_{i}" for i in range(100)]
    for _ in range(5):
        random.shuffle(all_ids)
        lst = [{CommonFields.ID: _id, "score": 1 - i * 0.01} for i, _id in enumerate(all_ids[:20])]
        result_lists.append(lst)
    fused = rrf_fusion(result_lists, k=60)
    for f in fused:
        print(f)
    assert len(fused) <= 100  # 最多100个唯一ID
    # 验证融合分数降序
    for i in range(len(fused) - 1):
        assert fused[i]["rrf_score"] >= fused[i + 1]["rrf_score"]


if __name__ == '__main__':
    # test_basic_fusion_two_lists()
    # test_fusion_with_three_lists()
    # test_all_lists_empty()
    test_large_corpus()