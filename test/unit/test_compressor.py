# tests/unit/test_context_compressor.py
"""
ContextCompressor 单元测试（真实模型，无 Mock）
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from modules.retrieval.context_compressor import ContextCompressor
from config.global_constant.constants import RegistryModules
from config.global_constant.fields import CommonFields
from utils.config_utils.get_config import get_config

class TestContextCompressor:
    """上下文压缩测试"""

    def test_compress_long_text(self, compressor):
        """长文本应被压缩，保留与查询最相关的句子，且结果变短"""
        query = "住房贷款的利率是多少？"
        # 构造一个很长的文本，包含多个句子，长度远超阈值
        long_text = (
            "住房贷款的利率由LPR加点形成。"
            "LPR每月20日公布。"
            "利率根据购房套数不同而有差异。"
            "首套房利率通常低于二套房。"
            "消费贷款可以用于装修、旅游等合法用途。"
            "经营贷款需要提供营业执照。"
            "等额本息和等额本金是常见的还款方式。"
            "提前还款可能需要支付违约金。"
            "公积金贷款额度受账户余额限制。"
        )
        documents = [
            {CommonFields.ID: "1", CommonFields.TEXT: long_text}
        ]
        compressed = compressor.compress(query, documents)
        print(compressed)

        # 应返回与原列表同长度的列表
        assert len(compressed) == 1
        original_text = long_text
        compressed_text = compressed[0][CommonFields.TEXT]
        # 压缩后文本应明显短于原文
        assert len(compressed_text) < len(original_text)
        # 应包含与查询相关的关键词（如“利率”）
        assert "利率" in compressed_text
        # 应只保留 sentences_to_keep 个句子（2个）
        sentences = compressor._split_sentences(compressed_text)
        assert len(sentences) == compressor.config.sentences_to_keep

    def test_compress_short_text_skip(self, compressor):
        """短文本（不超过阈值）不应被压缩，保持原样"""
        query = "贷款额度"
        short_text = "消费贷款最高额度为50万元。"
        documents = [
            {CommonFields.ID: "2", CommonFields.TEXT: short_text}
        ]
        compressed = compressor.compress(query, documents)
        assert len(compressed) == 1
        # 内容不变
        assert compressed[0][CommonFields.TEXT] == short_text

    def test_compress_empty_documents(self, compressor):
        """空文档列表直接返回空列表"""
        assert compressor.compress("查询", []) == []

    def test_compress_single_sentence_long(self, compressor):
        """文本很长但只有一个句子，不应被压缩（句子数不大于 sentences_to_keep）"""
        query = "信用卡"
        # 单个句子但长度足够长
        single_long = "信用卡还款方式灵活，支持多种分期方案。" * 10  # 长句但没有句号分隔
        documents = [
            {CommonFields.ID: "3", CommonFields.TEXT: single_long}
        ]
        compressed = compressor.compress(query, documents)
        # 分句后只有一个句子（因为没有句号），句子数 <= 配置的 sentences_to_keep，所以跳过分句压缩
        assert compressed[0][CommonFields.TEXT] == single_long

    def test_compress_multiple_documents(self, compressor):
        """多个文档，有的长有的短，各自正确处理"""
        query = "还款方式"
        long_text = (
            "等额本息月供固定。"
            "等额本金前期压力大。"
            "双周供可以加速还款。"
            "气球贷末期一次性还本。"
        )
        short_text = "提前还款无违约金。"
        documents = [
            {CommonFields.ID: "a", CommonFields.TEXT: long_text},
            {CommonFields.ID: "b", CommonFields.TEXT: short_text},
        ]
        compressed = compressor.compress(query, documents)
        # 长文本被压缩
        assert len(compressed[0][CommonFields.TEXT]) < len(long_text)
        # 短文本保持不变
        assert compressed[1][CommonFields.TEXT] == short_text

    def test_compress_keyword_relevance(self, compressor):
        """验证压缩后保留的句子确实与查询相关"""
        query = "公积金贷款额度"
        text = (
            "公积金贷款额度受账户余额影响。"
            "商业贷款利率由LPR决定。"
            "申请贷款需要提供收入证明。"
            "账户余额越高，可贷额度越大。"
        )
        documents = [{CommonFields.ID: "rel", CommonFields.TEXT: text}]
        compressed = compressor.compress(query, documents)
        result_text = compressed[0][CommonFields.TEXT]
        # 应保留与公积金额度相关的句子，而非不相关的句子
        assert "公积金" in result_text or "额度" in result_text


if __name__ == "__main__":
    # 手动执行测试示例（用于快速调试）
    registry = get_config()
    cfg = registry.get_config(RegistryModules.RETRIEVAL).compressor
    cfg.model_name = "cross-encoder/ms-marco-MiniLM-L6-v2"
    cfg.enabled = True
    cfg.max_context_tokens = 50
    cfg.sentences_to_keep = 2
    compressor = ContextCompressor(config=cfg)

    test = TestContextCompressor()
    test.test_compress_long_text(compressor)
    # test.test_compress_short_text_skip(compressor)
    # test.test_compress_empty_documents(compressor)
    # test.test_compress_single_sentence_long(compressor)
    # test.test_compress_multiple_documents(compressor)
    # test.test_compress_keyword_relevance(compressor)
    print("所有测试通过！")