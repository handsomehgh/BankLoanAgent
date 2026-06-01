import argparse
import json
import logging
import os
import random
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# -------------------- 配置 --------------------
MODEL_NAME = "./models-BAAI-bge-reranker-base"
OUTPUT_DIR = "/root/models/bge-loan-reranker"
MAX_LENGTH = 512
RANDOM_SEED = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# -------------------- 自定义 Dataset --------------------
class RerankerDataset(Dataset):
    """Reranker 成对数据集"""

    def __init__(self, file_path: str, tokenizer, max_length: int = MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load(file_path)

    def _load(self, file_path: str) -> List[Dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))
        logger.info(f"从 {file_path} 加载了 {len(samples)} 条样本")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoded = self.tokenizer(
            sample["query"],
            sample["document"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"]),
            "attention_mask": torch.tensor(encoded["attention_mask"]),
            "label": torch.tensor(sample["label"], dtype=torch.float),
        }


# -------------------- 评估指标 --------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = (torch.sigmoid(torch.tensor(logits)) >= 0.5).int().numpy()
    labels = labels.astype(int)
    acc = accuracy_score(labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# -------------------- 主函数 --------------------
def main(args):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # 1. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 2. 加载数据集
    train_dataset = RerankerDataset(args.train_file, tokenizer)
    val_dataset = RerankerDataset(args.val_file, tokenizer)
    logger.info(f"训练集: {len(train_dataset)} 条, 验证集: {len(val_dataset)} 条")

    # 3. 加载模型（二分类，num_labels=1，输出 logit）
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1)
    logger.info(f"模型加载完成: {MODEL_NAME}")

    # 4. 训练参数
    total_steps = (len(train_dataset) // args.batch_size) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        report_to="none",
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
    )

    # 5. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    # 6. 训练
    logger.info("开始训练...")
    trainer.train()

    # 7. 保存模型
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 8. 最终评估
    logger.info("最终评估结果:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info(f"训练完成，模型已保存至 {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reranker 微调 (HuggingFace Trainer)")
    parser.add_argument("--train_file", type=str, required=True, help="训练集 JSONL")
    parser.add_argument("--val_file", type=str, required=True, help="验证集 JSONL")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--logging_steps", type=int, default=50)
    args = parser.parse_args()
    main(args)