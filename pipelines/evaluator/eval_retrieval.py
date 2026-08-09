# author hgh
# version 2.0
"""
RAG 检索质量评估流水线（数据生成 -> Rerank 前后 Ablation 对比 -> 报告落盘）

流程：
1. 数据生成：从 Milvus 知识库抽取活跃 chunk，用 LLM 转写为口语化用户问题，
   构建 (query, relevant_doc_ids) 评估集（若已存在评估集则默认跳过，可用 --regenerate 强制重新生成）
2. Ablation 对比：复用同一套 dense/sparse/term 召回 + RRF 融合 + 查询改写/过滤组件，
   仅替换 reranker 组件，搭建两条严格对照的检索管道：
     - baseline：RRF 融合后直接截断 top_k（不做 Cross-Encoder 重排序）
     - full    ：RRF 融合 + Cross-Encoder 重排序
3. 指标计算：对两条管道分别计算 Recall@k / Precision@k / MRR@k，输出对比表并追加保存到
   pipelines/data/eval/eval_report.json，形成可追溯的评估历史记录

用法：
    python -m pipelines.evaluator.eval_retrieval                       # 使用已有评估集，直接跑对比
    python -m pipelines.evaluator.eval_retrieval --regenerate          # 从知识库重新生成评估集后再跑对比
    python -m pipelines.evaluator.eval_retrieval --regenerate --num-samples 150 --top-k 5
"""
import argparse
import json
import logging
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List

from config.global_constant.constants import MemoryType, RegistryModules
from config.models.retrieval_config import CompressorConfig
from infra.database.collections_type import CollectionNames
from infra.database.milvus_client import MilvusClientManager
from modules.module_services.chat_models import RobustLLM
from modules.module_services.embeddings import RobustLocalEmbeder
from modules.retrieval.context_complete import ContextComplete
from modules.retrieval.context_compressor import ContextCompressor
from modules.retrieval.knowledge_vector_store.knowledge_search_engine import KnowledgeSearchEngine
from modules.retrieval.query_filter import QueryFilter
from modules.retrieval.query_rewriter import QueryRewriter
from modules.retrieval.rereanker import Reranker
from modules.retrieval.retrieval_service import RetrievalService
from pipelines.evaluator.generate_eval_dataset import TOPIC_BASED_QUESTION_PROMPT
from utils.config_utils.get_config import get_config
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

EVAL_DATA_PATH = Path(__file__).resolve().parent.parent / "data/eval/auto_generated.jsonl"
REPORT_PATH = Path(__file__).resolve().parent.parent / "data/eval/eval_report4.json"


class PassThroughReranker:
    """
    Ablation 对照组：不调用 Cross-Encoder，直接沿用 RRF 融合后的原始顺序截断 top_k。
    与 Reranker.rerank 保持相同的调用接口，可直接替换注入 RetrievalService，
    用于和真实 Reranker 做"有/无重排序"的效果对比。
    """

    def __init__(self, top_k: int):
        self.top_k = top_k

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        logger.debug("PassThroughReranker: skip cross-encoder, truncate to top_%d directly", self.top_k)
        return candidates[: self.top_k]


def load_test_data(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_eval_dataset(llm: RobustLLM, retrieval_config, num_samples: int, output_path: Path) -> None:
    """从 Milvus 知识库抽取活跃 chunk，转写为口语化问题，生成 (query, relevant_doc_ids) 评估集"""
    milvus_client = MilvusClientManager(retrieval_config.milvus_uri)
    collection = milvus_client.get_collection(CollectionNames.for_type(MemoryType.BUSINESS_KNOWLEDGE))
    chunks = collection.query(
        expr="status == 'active'",
        output_fields=["id", "text", "source_type", "topics", "product_type"],
        limit=5000,
    )
    logger.info("从知识库获取到 %d 个活跃 chunk", len(chunks))
    if not chunks:
        logger.error("知识库中没有活跃 chunk，无法生成评估集")
        return

    random.shuffle(chunks)
    sampled = chunks[:num_samples]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in sampled:
            chunk_text = chunk.get("text", "")
            chunk_id = chunk.get("id", "")
            if not chunk_text or not chunk_id:
                continue

            prompt = TOPIC_BASED_QUESTION_PROMPT.format(chunk_text=chunk_text[:800])
            try:
                response = llm.invoke(prompt)
                question = response.content.strip() if hasattr(response, "content") else str(response).strip()
                question = re.sub(r"^\d+[\.\、\s]+", "", question)
            except Exception as e:
                logger.warning("生成问题失败 (chunk %s): %s", chunk_id, e)
                continue
            if not question:
                continue

            record = {
                "query": question,
                "relevant_doc_ids": [chunk_id],
                "ground_truth_answer": chunk_text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            logger.info("已生成 %d/%d: %s...", written, len(sampled), question[:50])

    logger.info("评估集已生成 %d 条，保存至 %s", written, output_path)


def evaluate_retrieval(retrieve_fn: Callable[[str], List], test_data: List[Dict], k: int = 5) -> Dict:
    """
    对给定检索函数 retrieve_fn(query) -> List[BusinessKnowledge] 计算 Recall@k / Precision@k / MRR@k
    """
    total = len(test_data)
    if total == 0:
        return {f"Recall@{k}": 0.0, f"Precision@{k}": 0.0, f"MRR@{k}": 0.0, "total_queries": 0, "hits": 0}

    hits = 0
    mrr = 0.0
    precision_sum = 0.0
    hit_positions = []  # 记录命中时正例的排名位置

    for i, item in enumerate(test_data):
        query = item["query"]
        relevant_ids = set(item["relevant_doc_ids"])
        results = retrieve_fn(query)
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

        if i < 3:
            logger.info("查询 %d: %s...", i + 1, query[:60])
            logger.info("  正例 IDs: %s", relevant_ids)
            logger.info("  返回 Top-%d IDs: %s", k, retrieved_ids)
            logger.info("  匹配数: %d, Precision@%d: %.4f", relevant_count, k, precision)

    recall = hits / total
    mrr = mrr / total
    avg_precision = precision_sum / total

    if hit_positions:
        logger.info(
            "命中时的排名分布: 平均=%.2f, 第1位=%d, 第2位=%d, 第3位=%d, 第4位=%d, 第5位=%d",
            sum(hit_positions) / len(hit_positions),
            hit_positions.count(1), hit_positions.count(2),
            hit_positions.count(3), hit_positions.count(4), hit_positions.count(5),
        )

    return {
        f"Recall@{k}": round(recall, 4),
        f"Precision@{k}": round(avg_precision, 4),
        f"MRR@{k}": round(mrr, 4),
        "total_queries": total,
        "hits": hits,
    }


def main():
    parser = argparse.ArgumentParser(description="RAG 检索质量评估：Rerank 前后 Recall@k/MRR@k 对比")
    parser.add_argument("--regenerate", action="store_true", help="从知识库重新生成评估数据集")
    parser.add_argument("--num-samples", type=int, default=100, help="重新生成评估集时的采样数量")
    parser.add_argument("--top-k", type=int, default=5, help="评估的 top-k")
    args = parser.parse_args()

    setup_logging(log_level="INFO")
    registry = get_config()
    llm_config = registry.get_config(RegistryModules.LLM)
    retrieval_config = registry.get_config(RegistryModules.RETRIEVAL)

    creative_llm = RobustLLM(
        temperature=llm_config.creative_temperature,
        api_key="No need",
        base_url=llm_config.local_qwen_url,
        model=llm_config.local_qwen_name,
        provider=llm_config.openai_provider,
    )
    precise_llm = RobustLLM(
        temperature=llm_config.precise_temperature,
        api_key="No need",
        base_url=llm_config.local_qwen_url,
        model=llm_config.local_qwen_name,
        provider=llm_config.openai_provider,
    )
    embedder = RobustLocalEmbeder(
        base_url=llm_config.loan_custom_embeder_url,
        model_name=llm_config.loan_custom_embeder_name,
        dimensions=llm_config.loan_embeder_dimension,
    )

    # ---------- 1. 数据生成（可选，从知识库到评估集） ----------
    if args.regenerate or not EVAL_DATA_PATH.exists():
        logger.info("开始从知识库生成评估数据集...")
        generate_eval_dataset(creative_llm, retrieval_config, args.num_samples, EVAL_DATA_PATH)

    if not EVAL_DATA_PATH.exists():
        logger.error("评估数据文件不存在：%s，且自动生成失败，请检查知识库连接", EVAL_DATA_PATH)
        return

    test_data = load_test_data(EVAL_DATA_PATH)
    logger.info("加载 %d 条测试样本", len(test_data))
    if not test_data:
        logger.error("评估数据集为空，终止评估")
        return

    # ---------- 2. 组装共享检索组件（除 reranker 外，两条管道完全一致，确保对比公平） ----------
    knowledge_client = MilvusClientManager(retrieval_config.milvus_uri)
    knowledge_engine = KnowledgeSearchEngine(knowledge_client, embedder, retrieval_config)
    rewriter = QueryRewriter(retrieval_config.rewriter, llm_client=creative_llm)
    query_filter = QueryFilter(retrieval_config.filter, llm_client=precise_llm)
    context_complete = ContextComplete(retrieval_config, precise_llm)
    # 评估只聚焦"检索 + 排序"质量，关闭上下文压缩，避免压缩阶段的 LLM 重选噪声混入 Recall/MRR
    compressor = ContextCompressor(CompressorConfig(enabled=False), llm_client=precise_llm)

    def make_service(reranker) -> RetrievalService:
        return RetrievalService(
            engine=knowledge_engine,
            rewriter=rewriter,
            filter=query_filter,
            reranker=reranker,
            compressor=compressor,
            config=retrieval_config,
            context_complete=context_complete,
        )

    baseline_service = make_service(PassThroughReranker(top_k=args.top_k))
    full_service = make_service(Reranker(retrieval_config.reranker))

    # ---------- 3. 分别评估 baseline（仅RRF）与 full（RRF+Rerank） ----------
    logger.info("===== 评估 Baseline（RRF 融合，无 Rerank）=====")
    baseline_metrics = evaluate_retrieval(baseline_service.retrieve, test_data, k=args.top_k)

    logger.info("===== 评估 Full Pipeline（RRF 融合 + Cross-Encoder Rerank）=====")
    full_metrics = evaluate_retrieval(full_service.retrieve, test_data, k=args.top_k)

    # ---------- 4. 输出对比报告 ----------
    r_key, p_key, m_key = f"Recall@{args.top_k}", f"Precision@{args.top_k}", f"MRR@{args.top_k}"
    print("\n" + "=" * 64)
    print(" RAG 检索质量评估报告（Rerank 前后 Ablation 对比）")
    print("=" * 64)
    print(f"测试样本数：{baseline_metrics['total_queries']}")
    print(f"{'指标':<14}{'无 Rerank(Baseline)':<22}{'有 Rerank(Full)':<20}{'提升':<10}")
    for name in (r_key, p_key, m_key):
        b, fv = baseline_metrics[name], full_metrics[name]
        print(f"{name:<14}{b:<22.4f}{fv:<20.4f}{fv - b:+.4f}")
    print("=" * 64)
    print(f"命中样本数：Baseline {baseline_metrics['hits']}/{baseline_metrics['total_queries']}"
          f"  |  Full {full_metrics['hits']}/{full_metrics['total_queries']}")

    # ---------- 5. 落盘保存，形成可追溯的评估历史记录 ----------
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_data_path": str(EVAL_DATA_PATH),
        "total_queries": baseline_metrics["total_queries"],
        "top_k": args.top_k,
        "baseline_no_rerank": baseline_metrics,
        "full_with_rerank": full_metrics,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False) + "\n")
    logger.info("评估报告已追加保存至 %s", REPORT_PATH)


if __name__ == "__main__":
    main()
