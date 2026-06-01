# author hgh
# version 1.0
from utils.logging_config import setup_logging
import json
import logging
from pathlib import Path
from typing import List, Dict

from config.global_constant.constants import RegistryModules
from infra.database.milvus_client import MilvusClientManager
from modules.module_services.chat_models import RobustLLM
from modules.module_services.embeddings import RobustEmbeddings, RobustLocalEmbeder
from modules.retrieval.context_compressor import ContextCompressor
from modules.retrieval.knowledge_vector_store.knowledge_search_engine import KnowledgeSearchEngine
from modules.retrieval.query_filter import QueryFilter
from modules.retrieval.query_rewriter import QueryRewriter
from modules.retrieval.rereanker import Reranker
from modules.retrieval.retrieval_service import RetrievalService
from modules.retrieval.router.retrieval_rule_router import RuleBaseRetrievalRouter
from utils.config_utils.get_config import get_config

logger = logging.getLogger(__name__)

def load_test_data(path: Path) -> List[Dict]:
    with open(path,"r",encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def evaluate_retrieval(retrieval_service, test_data, k=5):
    total = len(test_data)
    if total == 0:
        return {"Recall@5": 0.0, "Precision@5": 0.0, "MRR@5": 0.0, "total_queries": 0}

    hits = 0
    mrr = 0.0
    precision_sum = 0.0
    hit_positions = []  # 记录命中时正例的排名位置

    for i, item in enumerate(test_data):
        query = item["query"]
        relevant_ids = set(item["relevant_doc_ids"])
        results = retrieval_service.retrieve(query)
        top_k_results = results[:k]
        retrieved_ids = [doc.id for doc in top_k_results]

        matched = set(retrieved_ids) & relevant_ids
        if matched:
            hits += 1
            for rank, doc_id in enumerate(retrieved_ids, 1):
                if doc_id in relevant_ids:
                    mrr += 1.0 / rank
                    hit_positions.append(rank)
                    break

        relevant_count = len(matched)
        precision = relevant_count / len(retrieved_ids) if retrieved_ids else 0.0
        precision_sum += precision

        # 打印前 3 个查询的详细信息用于诊断
        if i < 3:
            logger.info(f"查询 {i+1}: {query[:60]}...")
            logger.info(f"  正例 IDs: {relevant_ids}")
            logger.info(f"  返回 Top-5 IDs: {retrieved_ids}")
            logger.info(f"  匹配数: {relevant_count}, Precision@5: {precision:.4f}")

    recall = hits / total
    mrr = mrr / total
    avg_precision = precision_sum / total

    # 统计命中时的排名分布
    if hit_positions:
        logger.info(f"命中时的排名分布: 平均={sum(hit_positions)/len(hit_positions):.2f}, "
                    f"第1位={hit_positions.count(1)}, 第2位={hit_positions.count(2)}, "
                    f"第3位={hit_positions.count(3)}, 第4位={hit_positions.count(4)}, "
                    f"第5位={hit_positions.count(5)}")

    return {
        "Recall@5": round(recall, 4),
        "Precision@5": round(avg_precision, 4),
        "MRR@5": round(mrr, 4),
        "total_queries": total,
        "hits": hits
    }

def main():
    setup_logging(log_level="DEBUG")
    registry = get_config()
    llm_config = registry.get_config(RegistryModules.LLM)
    retrieval_config = registry.get_config(RegistryModules.RETRIEVAL)
    memory_config = registry.get_config(RegistryModules.MEMORY_SYSTEM)

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
    embedder = RobustLocalEmbeder(
        base_url=llm_config.loan_embeder_url,
        model_name=llm_config.loan_embeder_name,
        dimensions=llm_config.loan_embeder_dimension
    )

    knowledge_client = MilvusClientManager(retrieval_config.milvus_uri)
    knowledge_engine = KnowledgeSearchEngine(knowledge_client, embedder, retrieval_config)
    rewriter = QueryRewriter(retrieval_config.rewriter, llm_client=creative_llm)
    query_filter = QueryFilter(retrieval_config.filter, llm_client=precise_llm)
    reranker = Reranker(retrieval_config.reranker)
    compressor = ContextCompressor(retrieval_config.compressor)
    router = RuleBaseRetrievalRouter(retrieval_config.retrieval_routing.rule_based)

    retrieval_service = RetrievalService(
        engine=knowledge_engine,
        rewriter=rewriter,
        filter=query_filter,
        reranker=reranker,
        compressor=compressor,
        config=retrieval_config,
        retrieve_router=router
    )

    test_data_path = Path(__file__).resolve().parent.parent / "data/eval/auto_generated.jsonl"
    if not test_data_path.exists():
        logger.error(f"测试数据文件不存在：{test_data_path}，请先运行人工校验。")
        return

    test_data = load_test_data(test_data_path)
    logger.info(f"加载 {len(test_data)} 条测试样本")

    # 执行评估
    metrics = evaluate_retrieval(retrieval_service, test_data, k=5)
    logger.info(f"评估结果：{metrics}")

    # 输出报告
    print("\n" + "=" * 40)
    print(" RAG 检索质量评估报告")
    print("=" * 40)
    print(f"测试样本数：{metrics['total_queries']}")
    print(f"Recall@5 ：{metrics['Recall@5']:.2%}")
    print(f"MRR@5    ：{metrics['MRR@5']:.4f}")
    print(f"命中样本数：{metrics['hits']}/{metrics['total_queries']}")


if __name__ == "__main__":
    main()




