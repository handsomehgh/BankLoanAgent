import argparse
import json
import math
import os
from functools import partial

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from sklearn.utils import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

# AMP 兼容：torch>=2.3 推荐 torch.amp，旧版本回退 torch.cuda.amp
try:
    from torch.amp import GradScaler, autocast
    _AMP_NEW = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    _AMP_NEW = False


def amp_context(device: torch.device):
    enabled = device.type == "cuda"
    if _AMP_NEW:
        return autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)
    return autocast(enabled=enabled, dtype=torch.float16)


def make_scaler(device: torch.device):
    enabled = device.type == "cuda"
    if _AMP_NEW:
        return GradScaler("cuda", enabled=enabled)
    return GradScaler(enabled=enabled)


# ================= 标签定义 =================
# 三个子图分类器的标签集，顺序必须与对应 model_server 的 ID2LABEL 完全一致
# （推理服务端按 argmax -> ID2LABEL[id] 映射，id 错位即线上标签错乱）
TASK_LABELS = {
    "advisor": [
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
    ],
    "after": [
        "DIRECT_REPLY",
        "CLARIFY",
        "prepayment_evaluation_skill",
        "extension_management_skill",
        "overdue_handling_skill",
        "repayment_method_switch_skill",
        "check_extension_eligibility",
        "calculate_extension_plan",
        "generate_repayment_schedule",
        "generate_settlement_certificate",
        "calculate_monthly_payment",
        "general_search_knowledge",
    ],
    "risk": [
        "DIRECT_REPLY",
        "CLARIFY",
        "risk_assessment_skill",
        "calculate_dti",
        "calculate_ltv",
        "calculate_dscr",
        "estimate_credit_score",
        "query_regulation",
        "general_search_knowledge",
    ],
}


# ================= 数据集定义 =================
class IntentDataset(Dataset):
    def __init__(self, file_path: str, label2id: dict):
        self.samples = []
        self.labels = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line.strip())
                text_a = item.get("text_a", "") or ""
                text_b = item.get("text_b", "") or ""
                label_str = item.get("label", "")
                # 跳过空文本（两个文本都为空）
                if not text_a and not text_b:
                    continue
                # 跳过未知标签
                if label_str not in label2id:
                    print(f"Warning: unknown label '{label_str}' in {file_path}, skipping.")
                    continue
                label = label2id[label_str]
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
    # text_a 截断到 500 字符，与 model_server 推理侧的 text_a[:500] 对齐
    text_a_batch = [item.get("text_a", "")[:500] for item in batch]
    text_b_batch = [item.get("text_b", "") for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    encoded = tokenizer(
        text_a_batch,
        text_b_batch,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt"
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels
    }


# ================= 模型与分词器加载 =================
def load_model_and_tokenizer(model_name, num_classes, id2label, label2id, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id
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


# ================= 训练一个 epoch =================
def train_epoch(model, loader: DataLoader, optimizer, scheduler, criterion, scaler,
                max_grad_norm: float, gradient_accumulation_steps: int, device: torch.device):
    model.train()
    all_preds, all_labels = [], []
    total_loss = 0.0

    optimizer.zero_grad()
    step = -1
    for step, batch in enumerate(tqdm(loader, desc="Training", total=len(loader))):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with amp_context(device):
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

    if step < 0:
        raise ValueError("训练集为空，请检查 --train_path")
    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


# ================= 验证/评估 =================
@torch.no_grad()
def eval_epoch(model, loader: DataLoader, criterion, device: torch.device, id2label: dict):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for batch in tqdm(loader, desc="Evaluating", total=len(loader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with amp_context(device):
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
        total_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    unique_labels = sorted(set(all_labels))
    f1_per_class = f1_score(all_labels, all_preds, average=None, labels=unique_labels, zero_division=0)
    precision_per_class = precision_score(all_labels, all_preds, average=None, labels=unique_labels, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, labels=unique_labels, zero_division=0)
    report = classification_report(all_labels, all_preds, labels=unique_labels,
                                   target_names=[id2label[i] for i in unique_labels], zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=unique_labels)

    f1_details = {id2label[i]: round(f1, 4) for i, f1 in zip(unique_labels, f1_per_class)}
    precision_details = {id2label[i]: round(p, 4) for i, p in zip(unique_labels, precision_per_class)}
    recall_details = {id2label[i]: round(r, 4) for i, r in zip(unique_labels, recall_per_class)}

    return {
        "loss": total_loss / len(loader),
        "acc": acc,
        "f1_macro": f1_macro,
        "f1_per_class": f1_details,
        "precision_per_class": precision_details,
        "recall_per_class": recall_details,
        "classification_report": report,
        "confusion_matrix": cm,
        "unique_labels": unique_labels,
    }


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_f1, patience_counter):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_f1": best_f1,
        "patience_counter": patience_counter,
    }, path)


# ================= 主流程 =================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    labels = TASK_LABELS[args.task]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    num_classes = len(labels)
    print(f"任务: {args.task} | 类别数: {num_classes}")

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, num_classes, id2label, label2id, device
    )

    # 构建 Dataset 和 DataLoader
    train_dataset = IntentDataset(args.train_path, label2id)
    eval_dataset = IntentDataset(args.val_path, label2id)
    print(f"训练集 {len(train_dataset)} 条，验证集 {len(eval_dataset)} 条")

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
        batch_size=args.eval_batch_size,
        collate_fn=eval_collate,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 类别权重（用于不平衡数据）；训练集未覆盖的类别权重保持 1
    unique_labels = np.unique(train_dataset.labels)
    class_weight_raw = compute_class_weight(class_weight="balanced", classes=unique_labels, y=train_dataset.labels)
    class_weights = np.ones(num_classes, dtype=np.float32)
    for lbl, w in zip(unique_labels, class_weight_raw):
        class_weights[lbl] = w
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 优化器、损失函数、调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # 向上取整，避免残余 batch 的额外 step 超出调度器总步数后 lr 归零
    total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    scaler = make_scaler(device)

    best_f1 = 0.0
    patience_counter = 0
    start_epoch = 1
    os.makedirs(args.output_path, exist_ok=True)
    checkpoint_path = os.path.join(args.output_path, "latest_checkpoint.pt")

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt["best_f1"]
        patience_counter = ckpt["patience_counter"]
        print(f"已从 {args.resume} 恢复训练：epoch={start_epoch}, best_f1={best_f1:.4f}")

    train_log = []
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch} / {args.epochs}")
        print(f"{'=' * 60}")

        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, scaler,
            max_grad_norm=args.max_grad_norm,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            device=device
        )
        val_metrics = eval_epoch(model, eval_loader, criterion, device, id2label)

        print(f"\n[Train] Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1(macro): {train_f1:.4f}")
        print(
            f"[Val]   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f} | F1(macro): {val_metrics['f1_macro']:.4f}")

        print("\n[Val] Per-Class Metrics:")
        for label in val_metrics["f1_per_class"].keys():
            print(f"  {label:>35s} | P: {val_metrics['precision_per_class'][label]:.4f} | "
                  f"R: {val_metrics['recall_per_class'][label]:.4f} | F1: {val_metrics['f1_per_class'][label]:.4f}")

        print("\n[Val] Classification Report:")
        print(val_metrics["classification_report"])

        train_log.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "train_f1_macro": train_f1,
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"],
            "val_f1_macro": val_metrics["f1_macro"], "val_f1_per_class": val_metrics["f1_per_class"],
        })
        with open(os.path.join(args.output_path, "train_log.json"), "w", encoding="utf-8") as f:
            json.dump(train_log, f, ensure_ascii=False, indent=2)

        # 早停与模型保存
        current_f1 = val_metrics["f1_macro"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            model.save_pretrained(args.output_path)
            tokenizer.save_pretrained(args.output_path)
            print(f"New best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{args.early_stopping_patience}")

        save_checkpoint(checkpoint_path, model, optimizer, scheduler, scaler,
                        epoch, best_f1, patience_counter)

        if patience_counter >= args.early_stopping_patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    # 最终评估最佳模型
    print(f"\n{'=' * 60}")
    print(f"Loading best model from {args.output_path}...")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        args.output_path, num_labels=num_classes, id2label=id2label, label2id=label2id
    ).to(device)
    final_metrics = eval_epoch(best_model, eval_loader, criterion, device, id2label)

    print(f"\n{'=' * 60}")
    print("FINAL EVALUATION (Best Model)")
    print(f"{'=' * 60}")
    print(f"Loss: {final_metrics['loss']:.4f}")
    print(f"Accuracy: {final_metrics['acc']:.4f}")
    print(f"F1 (macro): {final_metrics['f1_macro']:.4f}")

    print("\nPer-Class F1:")
    for label, f1 in final_metrics["f1_per_class"].items():
        print(f"  {label:>35s}: {f1:.4f}")

    print("\nClassification Report:")
    print(final_metrics["classification_report"])

    cm = final_metrics["confusion_matrix"]
    unique_labels = final_metrics["unique_labels"]
    label_names = [id2label[i] for i in unique_labels]
    print("\nConfusion Matrix (row=true, col=pred):")
    header = " " * 35 + "".join([f"{name:>15s}" for name in label_names])
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join([f"{v:>5d}" for v in row])
        print(f"{label_names[i]:>35s} [{row_str}]")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="三个子图意图分类器统一训练（advisor/after/risk）")
    parser.add_argument("--task", type=str, default="advisor", choices=list(TASK_LABELS.keys()),
                        help="选择子图分类器任务，标签集与对应 model_server 的 ID2LABEL 对齐")
    parser.add_argument("--train_path", type=str, default="./advisor_train.jsonl", help="Training data path")
    parser.add_argument("--val_path", type=str, default="./advisor_val.jsonl", help="Validation data path")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext", help="Pretrained model name")
    parser.add_argument("--output_path", type=str, default="./intent_classifier", help="Directory to save best model")
    parser.add_argument("--resume", type=str, default="",
                        help="Checkpoint path to resume training，如 output_path/latest_checkpoint.pt（不传则从头训）")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
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
