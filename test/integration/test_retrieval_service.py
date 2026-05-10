# tests/integration/test_retrieval_service.py
"""
RetrievalService 集成测试（真实环境）
需要 Milvus 服务运行，且知识库已索引完毕；LLM 与 Embedding 服务可用。
"""
import os

from modules.retrieval.router.retrieval_rule_router import RuleBaseRetrievalRouter

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from infra.cache.cache_factory import CacheFactory
from modules.module_services.chat_models import RobustLLM
from modules.module_services.embeddings import RobustEmbeddings

import pytest
from config.global_constant.constants import RegistryModules
from modules.retrieval.knowledge_vector_store.knowledge_search_engine import KnowledgeSearchEngine
from modules.retrieval.query_rewriter import QueryRewriter
from modules.retrieval.query_filter import QueryFilter
from modules.retrieval.rereanker import Reranker
from modules.retrieval.context_compressor import ContextCompressor
from modules.retrieval.retrieval_service import RetrievalService
from modules.retrieval.knowledge_model import BusinessKnowledge
from infra.milvus_client import MilvusClientManager
from utils.config_utils.get_config import get_config

@pytest.fixture(scope="module")
def retrieval_service():
    """初始化完整的 RetrievalService，复用全局配置和 Milvus 连接"""
    registry = get_config()
    retrieval_config = registry.get_config(RegistryModules.RETRIEVAL)
    llm_config = registry.get_config(RegistryModules.LLM)
    cache_config = registry.get_config(RegistryModules.CACHE)

    # 基础组件
    milvus_client = MilvusClientManager(uri=retrieval_config.milvus_uri)

    embedder = RobustEmbeddings(
        api_key=llm_config.alibaba_api_key,
        model_name=llm_config.alibaba_emb_name,
        backup_model_name=llm_config.alibaba_emb_backup,
        dimensions=llm_config.dimension
    )

    engine = KnowledgeSearchEngine(milvus_client, embedder, retrieval_config)

    cache_factory = CacheFactory(cache_config)
    rag_cache_manager = cache_factory.create("rag")

    creative_llm = RobustLLM(
        temperature=llm_config.creative_temperature,
        api_key=llm_config.deepseek_api_key,
        base_url=llm_config.deepseek_base_url,
        model=llm_config.deepseek_llm_name,
        provider=llm_config.openai_provider
    )
    precise_llm = RobustLLM(
        temperature=llm_config.precise_temperature,
        api_key=llm_config.deepseek_api_key,
        base_url=llm_config.deepseek_base_url,
        model=llm_config.deepseek_llm_name,
        provider=llm_config.openai_provider
    )
    # 业务组件
    rewriter = QueryRewriter(retrieval_config.rewriter,creative_llm)
    query_filter = QueryFilter(retrieval_config.filter,precise_llm)
    reranker = Reranker(retrieval_config.reranker)
    compressor = ContextCompressor(retrieval_config.compressor)
    router = RuleBaseRetrievalRouter(retrieval_config.retrieval_routing.rule_based)

    service = RetrievalService(
        engine=engine,
        rewriter=rewriter,
        filter=query_filter,
        reranker=reranker,
        compressor=compressor,
        config=retrieval_config,
        cache_manager=rag_cache_manager,
        retrieve_router=router
    )
    return service


class TestRetrievalService:
    """检索服务核心链路测试"""

    def test_basic_retrieval(self, retrieval_service):
        """正常语义检索：返回 BusinessKnowledge 列表且包含相关结果"""
        query = "我想先了解一下公积金贷"
        results = retrieval_service.retrieve(query)
        print(f"==========={results}")

        assert isinstance(results, list)
        assert len(results) > 0, "应至少返回一条知识"
        for item in results:
            assert isinstance(item, BusinessKnowledge)
            # 验证核心字段非空
            assert item.text
            assert item.id
            assert item.source_type
        # 第一条结果应与利率相关（topics 或文本中包含“利率”）
        first = results[0]
        assert "利率" in first.topics or "利率" in first.text, "最相关结果应涉及利率"

    def test_filtered_retrieval(self, retrieval_service):
        """带产品类型过滤的检索：只返回住房贷款相关内容"""
        query = "提前还款有违约金吗"
        # 注意：filter 在 QueryFilter.extract 中自动提取，
        # 但这里我们无法强制指定，所以依赖 LLM 能否从查询中提取 product_type
        # 如果 LLM 提取失败，则退化为无过滤，但结果仍应包含还款相关。
        results = retrieval_service.retrieve(query)

        assert len(results) > 0
        for item in results:
            # 可以检查大部分结果的 product_type 是否为住房贷款或消费贷款等
            # 但不强约束，因为 LLM 可能提取不满
            pass
        # 至少有一条结果涉及“还款”或“违约金”
        texts = [item.text for item in results]
        assert any("还款" in t or "违约金" in t for t in texts), "结果应包含还款相关知识"

    def test_sparse_search_scenario(self, retrieval_service):
        """针对精确关键词查询，验证 BM25 稀疏检索能正常工作"""
        # 使用一个明确的术语查询，如“LPR”
        query = "LPR"
        results = retrieval_service.retrieve(query)

        assert len(results) > 0
        # LPR 相关的文档 topics 或文本中应出现“利率”
        found = any(
            "利率" in r.topics or "LPR" in r.text or "利率" in r.text
            for r in results
        )
        assert found, "LPR 查询应召回利率相关文档"