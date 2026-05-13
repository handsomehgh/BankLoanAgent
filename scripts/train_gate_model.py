# author hgh
# version 1.0
# scripts/train_gate_model.py
"""
画像提取门控模型训练脚本
功能：基于标注数据微调一个微型 BERT 模型，用于判断是否触发画像提取。
"""

import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------- 配置管理 --------------------
@dataclass
class GateModelConfig:
    """门控模型训练配置"""
    # 数据
    train_data_path: str = str(PROJECT_ROOT / "data/gate_train.jsonl")
    test_size: float = 0.2
    random_seed: int = 42
    # 模型
    base_model_name: str = "shibing624/text2vec-base-chinese"
    num_labels: int = 2
    max_seq_length: int = 64
    # 训练超参
    output_dir: str = str(PROJECT_ROOT / "models/gate_model_output")
    final_model_dir: str = str(PROJECT_ROOT / "models/gate_model_final")
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    num_train_epochs: int = 5
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2
    fp16: bool = torch.cuda.is_available()
    dataloader_num_workers: int = 0
    save_total_limit: int = 2
    # 标签映射
    label_map: dict = field(default_factory=lambda: {0: "no_extract", 1: "extract"})

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.final_model_dir).mkdir(parents=True, exist_ok=True)


# -------------------- 数据加载与处理 --------------------
def load_and_prepare_data(config: GateModelConfig):
    """加载 JSONL 数据，划分训练/验证集"""
    logger.info("Loading training data from %s", config.train_data_path)
    try:
        with open(config.train_data_path, 'r', encoding='utf-8') as f:
            raw_data = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        logger.error("训练数据文件不存在: %s", config.train_data_path)
        sys.exit(1)

    if len(raw_data) < 50:
        logger.warning("训练数据量较少 (%d 条)，模型可能欠拟合", len(raw_data))

    texts = [item["text"] for item in raw_data]
    labels = [item["label"] for item in raw_data]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=config.test_size, random_state=config.random_seed, stratify=labels
    )
    logger.info("训练集: %d 条, 验证集: %d 条", len(train_texts), len(val_texts))

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
    return train_dataset, val_dataset


def tokenize_function(examples, tokenizer, max_length):
    """分词处理函数"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,   # 将在 DataCollator 中动态填充
    )


# -------------------- 评估指标 --------------------
def compute_metrics(eval_pred):
    """计算准确率、精确率、召回率、F1"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# -------------------- 主训练流程 --------------------
def main():
    config = GateModelConfig()
    set_seed(config.random_seed)
    logger.info("Using device: %s", "cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据
    train_dataset, val_dataset = load_and_prepare_data(config)

    # 2. 加载模型和分词器
    logger.info("Loading base model from %s", config.base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_name,
        num_labels=config.num_labels,
    )

    # 3. 数据预处理
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config.max_seq_length),
        batched=True,
    )
    tokenized_val = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, config.max_seq_length),
        batched=True,
    )
    # 移除原始文本列，仅保留模型需要的列
    tokenized_train = tokenized_train.remove_columns(["text"])
    tokenized_val = tokenized_val.remove_columns(["text"])
    tokenized_train.set_format("torch")
    tokenized_val.set_format("torch")

    # 4. 动态 padding 的 DataCollator
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        save_total_limit=config.save_total_limit,
        report_to="none",  # 不上报 wandb/mlflow
        seed=config.random_seed,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    # 7. 训练
    logger.info("开始训练...")
    trainer.train()

    # 8. 评估
    eval_results = trainer.evaluate()
    logger.info("验证集评估结果: %s", eval_results)

    # 9. 保存最终模型与元数据
    logger.info("保存最终模型到 %s", config.final_model_dir)
    model.save_pretrained(config.final_model_dir)
    tokenizer.save_pretrained(config.final_model_dir)

    # 保存配置和标签映射
    with open(Path(config.final_model_dir) / "gate_config.json", 'w') as f:
        json.dump({
            "base_model": config.base_model_name,
            "max_seq_length": config.max_seq_length,
            "label_map": config.label_map,
        }, f, ensure_ascii=False, indent=2)

    logger.info("训练完毕，模型已保存至 %s", config.final_model_dir)


if __name__ == "__main__":
    main()