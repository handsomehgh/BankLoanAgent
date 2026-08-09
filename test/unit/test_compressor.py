# tests/unit/test_context_compressor.py
"""
ContextCompressor 单元测试

当前压缩语义：LLM 对候选文档按相关性整体排序（llm_rerank 提示词），
保留排序后的前 compress_top_k 篇；LLM 失败/返回空时降级取前 3 篇。
（早期"分句 + cross-encoder 保留关键句"的语义已废弃）
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from config.global_constant.fields import CommonFields
from config.models.retrieval_config import CompressorConfig
from modules.retrieval.context_compressor import ContextCompressor


def _make_compressor(llm_mock=None, enabled=True, top_k=2):
    config = CompressorConfig(enabled=enabled, compress_top_k=top_k)
    prompt_hub = MagicMock()
    prompt_hub.render_messages.return_value = []
    return ContextCompressor(config=config, llm_client=llm_mock, prompt_hub=prompt_hub)


def _docs(*ids):
    return [{CommonFields.ID: i, CommonFields.TEXT: f"文档 {i} 的内容"} for i in ids]


def _llm_returning(sorted_ids):
    llm = MagicMock()
    # compress 只检查 sorted_ids 属性，无需构造 pydantic 模型（避免 int/str 强转）
    llm.invoke.return_value = SimpleNamespace(sorted_ids=sorted_ids)
    return llm


class TestContextCompressor:
    """上下文压缩测试"""

    def test_compress_disabled_returns_as_is(self):
        """压缩关闭时原样返回全部文档"""
        compressor = _make_compressor(enabled=False)
        docs = _docs("1", "2", "3")
        assert compressor.compress("查询", docs) == docs

    def test_compress_empty_documents(self):
        """空文档列表返回空列表"""
        compressor = _make_compressor(llm_mock=_llm_returning([]))
        assert compressor.compress("查询", []) == []

    def test_compress_orders_by_llm_and_truncates_top_k(self):
        """按 LLM 排序结果重排文档，并截断到 compress_top_k"""
        compressor = _make_compressor(llm_mock=_llm_returning(["3", "1", "2"]), top_k=2)
        compressed = compressor.compress("住房贷款的利率是多少？", _docs("1", "2", "3"))
        assert [d[CommonFields.ID] for d in compressed] == ["3", "1"]

    def test_compress_skips_unknown_ids(self):
        """LLM 返回了不存在的文档 id 时应跳过，而不是报错"""
        compressor = _make_compressor(llm_mock=_llm_returning(["9", "2"]), top_k=5)
        compressed = compressor.compress("查询", _docs("1", "2"))
        assert [d[CommonFields.ID] for d in compressed] == ["2"]

    def test_compress_llm_empty_result_fallback_top3(self):
        """LLM 返回空排序结果时降级取前 3 篇"""
        compressor = _make_compressor(llm_mock=_llm_returning([]), top_k=2)
        compressed = compressor.compress("查询", _docs("1", "2", "3", "4"))
        assert [d[CommonFields.ID] for d in compressed] == ["1", "2", "3"]

    def test_compress_llm_error_fallback_top3(self):
        """LLM 调用异常时降级取前 3 篇，不抛异常"""
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("llm down")
        compressor = _make_compressor(llm_mock=llm, top_k=2)
        compressed = compressor.compress("查询", _docs("1", "2", "3", "4"))
        assert [d[CommonFields.ID] for d in compressed] == ["1", "2", "3"]
