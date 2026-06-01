#!/usr/bin/env python3
import argparse, json, logging, math, os
from collections import defaultdict
from typing import List
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "models-BAAI-small-zh-v1.5"
OUTPUT_DIR = "./bge-small-loan"
EVAL_TOP_K = [1, 3, 5, 10]
PATIENCE = 3

def load_data(file_path: str) -> List[InputExample]:
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            anchor = data.get("anchor") or data.get("query")
            positive = data.get("positive")
            negatives = data.get("negatives", [])
            if not anchor or not positive:
                continue
            negatives = [n for n in negatives if n]
            examples.append(InputExample(texts=[anchor, positive] + negatives))
    logger.info(f"从 {file_path} 加载了 {len(examples)} 条有效数据")
    return examples

def build_evaluator(val_examples: List[InputExample]) -> evaluation.InformationRetrievalEvaluator:
    queries = {}
    corpus = {}
    relevant_docs = defaultdict(set)
    doc_set = set()
    for example in val_examples:
        if len(example.texts) < 2:
            continue
        positive = example.texts[1]
        if positive:
            doc_set.add(positive)
    doc_to_id = {doc: str(idx) for idx, doc in enumerate(doc_set)}
    for doc, did in doc_to_id.items():
        corpus[did] = doc
    for i, example in enumerate(val_examples):
        if len(example.texts) < 2:
            continue
        query = example.texts[0]
        positive = example.texts[1]
        if not query or not positive:
            continue
        qid = str(i)
        queries[qid] = query
        pos_id = doc_to_id.get(positive)
        if pos_id:
            relevant_docs[qid].add(pos_id)
    logger.info(f"评估器已构建：{len(queries)} 个查询, {len(corpus)} 个文档")
    return evaluation.InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        main_score_function='cos_sim',
        mrr_at_k=EVAL_TOP_K,
        ndcg_at_k=EVAL_TOP_K,
        precision_recall_at_k=EVAL_TOP_K,
        accuracy_at_k=EVAL_TOP_K,
        map_at_k=EVAL_TOP_K,
        show_progress_bar=True,
    )

def main(args):
    train_examples = load_data(args.train_file)
    evaluator = None
    if args.val_file and os.path.exists(args.val_file):
        evaluator = build_evaluator(load_data(args.val_file))

    model = SentenceTransformer(MODEL_NAME)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=args.batch_size,
        collate_fn=model.smart_batching_collate,
    )

    total_steps = math.ceil(len(train_examples) / args.batch_size) * args.epochs
    warmup_steps = int(total_steps * 0.1)

    best_score, no_improve = 0, 0
    def early_stopping_callback(score, epoch, step):
        nonlocal best_score, no_improve
        if score > best_score:
            best_score = score
            no_improve = 0
        else:
            no_improve += 1
            logger.info(f"评估分数未提升 ({no_improve}/{PATIENCE})")
        if no_improve >= PATIENCE:
            logger.info("触发早停，训练终止")
            return True
        return False

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': args.learning_rate},
        output_path=OUTPUT_DIR,
        save_best_model=True,
        show_progress_bar=True,
        use_amp=torch.cuda.is_available(),
        callback=early_stopping_callback if evaluator else None,
    )

    if evaluator:
        logger.info("最终评估 (最佳模型)...")
        final_metrics = evaluator(model, output_path=None)
        if isinstance(final_metrics, dict):
            for k, v in final_metrics.items():
                if isinstance(v, float):
                    logger.info(f"  {k}: {v:.4f}")
                else:
                    logger.info(f"  {k}: {v}")
        else:
            logger.info(f"主评估分数: {final_metrics:.4f}")

    logger.info(f"训练完成！模型已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="train_data.jsonl")
    parser.add_argument("--val_file", default="eval_data.jsonl")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    main(args)