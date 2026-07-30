import argparse
import json
import os
from functools import partial

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from sklearn.utils import compute_class_weight
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

# ================= 标签定义 =================
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

LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}
NUM_CLASSES = len(LABELS)


# ================= 数据集定义 =================
class IntentDataset(Dataset):
    def __init__(self, file_path: str):
        self.samples = []
        self.labels = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                text_a = item.get("text_a", "")
                text_b = item.get("text_b", "")
                label_str = item.get("label", "")
                # 跳过空文本（两个文本都为空）
                if not text_a and not text_b:
                    continue
                # 跳过未知标签
                if label_str not in LABEL2ID:
                    print(f"Warning: unknown label '{label_str}' in {file_path}, skipping.")
                    continue
                label = LABEL2ID[label_str]
                self.labels.append(label)
                self.samples.append({
                    "text_a": text_a,
                    "text_b": text_b,
                    "label": label,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ================= 批处理函数 =================
def collate_fn(batch, tokenizer, max_length: int):
    text_a_batch = [item.get("text_a", "") for item in batch]
    text_b_batch = [item.get("text_b", "") for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    encoded = tokenizer(
        text_a_batch,
        text_b_batch,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels
    }


# ================= 模型与分词器加载 =================
def load_model_and_tokenizer(model_name, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 处理 pad_token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            # 复用 eos_token 作为 pad_token，无需调整词表
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # 新增 [PAD] token，需要调整模型 embedding 大小
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            print("Added [PAD] token and resized model embeddings.")

    # 同步 model 的 pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    return model.to(device), tokenizer


# ================= 训练一个 epoch（修复梯度累积与 total_loss） =================
def train_epoch(model, loader: DataLoader, optimizer, scheduler, criterion, scaler,
                max_grad_norm: float, gradient_accumulation_steps: int, device: torch.device):
    model.train()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for step, batch in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(dtype=torch.float16):
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

        # 损失归一化（适配梯度累积）
        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

        # 累积步数达到或已经到了最后一个 batch
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(loader):
            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # 优化器更新
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * gradient_accumulation_steps
        preds = torch.argmax(outputs.logits, dim=-1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


# ================= 验证/评估 =================
def eval_epoch(model, loader: DataLoader, criterion, device: torch.device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", total=len(loader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    unique_labels = sorted(set(all_labels))
    f1_per_class = f1_score(all_labels, all_preds, average=None, labels=unique_labels, zero_division=0)
    precision_per_class = precision_score(all_labels, all_preds, average=None, labels=unique_labels, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, labels=unique_labels, zero_division=0)
    report = classification_report(all_labels, all_preds, labels=unique_labels,
                                   target_names=[ID2LABEL[i] for i in unique_labels], zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)

    f1_details = {ID2LABEL[i]: round(f1, 4) for i, f1 in zip(unique_labels, f1_per_class)}
    precision_details = {ID2LABEL[i]: round(p, 4) for i, p in zip(unique_labels, precision_per_class)}
    recall_details = {ID2LABEL[i]: round(r, 4) for i, r in zip(unique_labels, recall_per_class)}

    return {
        "loss": total_loss / len(loader),
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_per_class": f1_details,
        "precision_per_class": precision_details,
        "recall_per_class": recall_details,
        "classification_report": report,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "unique_labels": unique_labels,
    }


# ================= 主流程 =================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    # 构建 Dataset 和 DataLoader
    train_dataset = IntentDataset(args.train_path)
    eval_dataset = IntentDataset(args.val_path)

    train_collate = partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length)
    eval_collate = partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_collate,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=eval_collate,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 类别权重（用于不平衡数据）
    unique_labels = np.unique(train_dataset.labels)
    class_weight_raw = compute_class_weight(class_weight="balanced", classes=unique_labels, y=train_dataset.labels)
    class_weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for lbl, w in zip(unique_labels, class_weight_raw):
        class_weights[lbl] = w
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 优化器、损失函数、调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 注意：考虑到梯度累积，总训练步数要除以 accumulation_steps
    total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    scaler = GradScaler()

    best_f1 = 0.0
    patience_counter = 0
    os.makedirs(args.output_path, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch} / {args.epochs}")
        print(f"{'=' * 60}")

        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler,
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            device=device
        )
        val_metrics = eval_epoch(model, eval_loader, criterion, device)

        print(f"\n[Train] Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1(macro): {train_f1:.4f}")
        print(
            f"[Val]   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f} | F1(macro): {val_metrics['f1_macro']:.4f}")

        print("\n[Val] Per-Class Metrics:")
        for label in val_metrics["f1_per_class"].keys():
            print(f"  {label:>30s} | P: {val_metrics['precision_per_class'][label]:.4f} | "
                  f"R: {val_metrics['recall_per_class'][label]:.4f} | F1: {val_metrics['f1_per_class'][label]:.4f}")

        print("\n[Val] Classification Report:")
        print(val_metrics["classification_report"])

        # 早停与模型保存
        current_f1 = val_metrics["f1_macro"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            model.save_pretrained(args.output_path)
            tokenizer.save_pretrained(args.output_path)
            print(f"✅ New best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{args.early_stopping_patience}")
            if patience_counter >= args.early_stopping_patience:
                print(f"⏹ Early stopping triggered after {epoch} epochs")
                break

    # 最终评估最佳模型
    print(f"\n{'=' * 60}")
    print(f"Loading best model from {args.output_path}...")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        args.output_path, num_labels=NUM_CLASSES, id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)
    final_metrics = eval_epoch(best_model, eval_loader, criterion, device)

    print(f"\n{'=' * 60}")
    print("FINAL EVALUATION (Best Model)")
    print(f"{'=' * 60}")
    print(f"Loss: {final_metrics['loss']:.4f}")
    print(f"Accuracy: {final_metrics['acc']:.4f}")
    print(f"F1 (macro): {final_metrics['f1_macro']:.4f}")

    print("\nPer-Class F1:")
    for label, f1 in final_metrics["f1_per_class"].items():
        print(f"  {label:>30s}: {f1:.4f}")

    print("\nClassification Report:")
    print(final_metrics["classification_report"])

    cm = final_metrics["confusion_matrix"]
    unique_labels = final_metrics["unique_labels"]
    label_names = [ID2LABEL[i] for i in unique_labels]
    print("\nConfusion Matrix (row=true, col=pred):")
    header = " " * 30 + "".join([f"{name:>15s}" for name in label_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{v:>5d}" for v in row])
        print(f"{label_names[i]:>30s} [{row_str}]")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train intent classifier with RoBERTa")
    parser.add_argument("--train_path", type=str, default="./advisor_train.jsonl", help="Training data path")
    parser.add_argument("--val_path", type=str, default="./advisor_val.jsonl", help="Validation data path")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext", help="Pretrained model name")
    parser.add_argument("--output_path", type=str, default="./intent_classifier", help="Directory to save best model")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio of total steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()
    main(args)
