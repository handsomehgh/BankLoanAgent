# author hgh
# version 1.0
import argparse
import os
from collections import Counter

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, EarlyStoppingCallback, \
    Trainer

os.environ["TENSORBOARD_LOGGING_DIR"] = "./logs"

MODEL_NAME = "hf1/chinese-roberta-wwm-ext"
MAX_LENGTH = 512
OUTPUT_DIR = "./loan_advisor_classifier_model"
LOGGING_DIR = "./logs"

LABELS = [
     "DIRECT_REPLY",
    "CLARIFY",
    "apply_home_loan_skill",
    "apply_consumer_loan_skill",
    "calculate_monthly_payment",
    "query_interest_rate",
    "calculate_loan_total_cost",
    "calculate_max_loan_amount",
    "check_loan_eligibility",
    "compare_loan_products",
    "generate_repayment_schedule",
    "general_search_knowledge",
    "query_loan_interest",
    "upsert_loan_interest",
    "urge_loan_interest",
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs,return_outputs=False,**kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(labels.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss,outputs) if return_outputs else loss

def preprocess_function(examples, tokenizer):
    texts_a = examples.get("text_a", [""] * len(examples["text_a"]))
    texts_b = examples["text_b"]

    return tokenizer(
        texts_a,
        texts_b,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


def convert_labels(examples):
    examples["label"] = [LABEL2ID[label] for label in examples["label"]]
    return examples


def load_data(train_path: str, val_path: str):
    dataset = load_dataset(
        "json",
        data_files={"train": train_path, "validation": val_path},
    )
    return dataset


def compute_metrics(eval_pred):
    """
    计算评估指标：准确率、宏平均 F1、每个类别的 F1
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_per_class = f1_score(labels, predictions, average=None)

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
    }

    # 每个类别的 F1
    for i, f1 in enumerate(f1_per_class):
        metrics[f"f1_{ID2LABEL[i]}"] = f1

    return metrics


def main(args):
    print("=" * 60)
    print("LoanAdvisor BERT 分类器训练")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n加载训练数据: {args.train_file}")
    print(f"加载验证数据: {args.val_file}")
    dataset = load_data(args.train_file, args.val_file)

    # 打印数据分布
    train_labels = [item["label"] for item in dataset["train"]]
    print("\n训练集标签分布:")
    for label, count in Counter(train_labels).most_common():
        print(f"  {label}: {count}")

    # 2. 加载分词器
    print(f"\n加载分词器: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained("./chinese-roberta-wwm-ext")

    # 3. 数据预处理
    print("预处理数据...")

    # 转换标签
    dataset = dataset.map(convert_labels, batched=True)

    # 分词
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    # 设置数据集格式
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # 加载模型
    print(f"加载模型: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        "./chinese-roberta-wwm-ext",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    total_samples = len(dataset["train"])
    total_steps = (total_samples // args.batch_size) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
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
        seed=42,
        data_seed=42
    )

    # 6. 早停回调
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience
    )

    # 7. 创建 Traine
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["validation"],
    #     compute_metrics=compute_metrics,
    #     callbacks=[early_stopping]
    # )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )

    # 8. 开始训练
    print("\n开始训练...")
    trainer.train()

    # 9. 保存最佳模型
    print(f"\n保存最佳模型到: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 10. 最终评估
    print("\n最终评估结果:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        if "f1" in key or "accuracy" in key:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value:.4f}")

    print("\n训练完成！")
    return trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练 LoanAdvisor 意图分类器")
    parser.add_argument("--train_file", type=str, default="advisor_train.jsonl", help="训练集文件路径")
    parser.add_argument("--val_file", type=str, default="advisor_val.jsonl", help="验证集文件路径")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--batch_size", type=int, default=16, help="训练批次大小")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数 (默认: 5)")
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="早停耐心值 (默认: 2)")
    parser.add_argument("--logging_steps", type=int, default=50, help="日志记录步数间隔 (默认: 50)")
    args = parser.parse_args()
    main(args)
