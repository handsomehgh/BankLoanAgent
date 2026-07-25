import argparse
import json
import os
from functools import partial

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from sklearn.utils import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from scripts.model_script.train_bert_context_torch import NUM_CLASSES

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

LABEL2ID = {label: id for id, label in enumerate(LABELS)}
ID2LABEL = {id: label for id, label in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)


class IntentDataset(Dataset):
    def __init__(self, file_path: str):
        self.samples = []
        self.labels = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                text_a = item.get("text_a", "")
                text_b = item.get("text_b", "")
                label_str = item["label"]
                if not label_str or label_str not in LABEL2ID:
                    print(f"Warning: unknown label '{label_str}', skipping.")
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


def collate_fn(batch, tokenizer, max_length: int):
    text_a_list = [item["text_a"] for item in batch]
    text_b_list = [item["text_b"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    encoding = tokenizer(text_a_list, text_b_list, truncation=True, max_length=max_length, padding="max_length",
                         return_tensors="pt")

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": labels
    }


def load_model_and_tokenizer(model_name, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else "[PAD]"
        print(f"pad_token set to: {tokenizer.pad_token}")
    model.config.pad_token_id = tokenizer.pad_token_id

    return model.to(device), tokenizer


def train_epoch(model, loader: DataLoader, optimizer, scheduler, criterion, device: torch.device, max_grad_norm):
    model.train()
    all_preds, all_labels = [], []
    total_loss = 0.0

    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with autocast(dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

        # 直接反向传播（梯度不会下溢）
        loss.backward()

        # 梯度裁剪（此时梯度是真实的 BF16 精度，裁剪有效）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # 更新参数
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(loader), acc, f1


def eval_epoch(model, loader: DataLoader, criterion, device: torch.device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro")

        unique_labels = sorted(set(all_labels))
        f1_per_class = f1_score(all_labels, all_preds, average="None", labels=unique_labels)
        precision_per_class = precision_score(all_labels, all_preds, average="None", labels=unique_labels,
                                              zero_division=0)
        recall_per_class = recall_score(
            all_labels, all_preds, average=None, labels=unique_labels, zero_division=0
        )
        report = classification_report(
            all_labels,
            all_preds,
            labels=unique_labels,
            target_names=[ID2LABEL[i] for i in unique_labels],
            zero_division=0,
        )
        cm = confusion_matrix(all_labels, all_preds, unique_labels)
        f1_details = {
            ID2LABEL[i]: round(f1, 4) for i, f1 in zip(unique_labels, f1_per_class)
        }
        precision_details = {
            ID2LABEL[i]: round(p, 4)
            for i, p in zip(unique_labels, precision_per_class)
        }
        recall_details = {
            ID2LABEL[i]: round(r, 4)
            for i, r in zip(unique_labels, recall_per_class)
        }
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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    train_dataset = IntentDataset(args.train_path)
    val_dataset = IntentDataset(args.val_path)

    train_collate = partial(collate_fn, tokenizer, args.max_length)
    eval_collate = train_collate

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_collate,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    eval_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=eval_collate,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    unique_labels = np.unique(train_dataset.labels)
    class_weight_raw = compute_class_weight(class_weight="balanced", classes=unique_labels, y=train_dataset.labels)
    class_weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for lbl, w in zip(unique_labels, class_weight_raw):
        class_weights[lbl] = w
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    scaler = GradScaler()

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    best_f1 = 0.0
    patience_counter = 0
    os.makedirs(args.output_path, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch} / {args.epochs}")
        print(f"{'=' * 60}")

        # 🔧 修复：传入 max_grad_norm 参数
        train_loss, train_acc, train_f1 = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            scaler,
            device,
            max_grad_norm=args.max_grad_norm,
        )
        val_metrics = eval_epoch(model, eval_loader, criterion, device)

        print(
            f"\n[Train] Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1(macro): {train_f1:.4f}"
        )
        print(
            f"[Val]   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f} | F1(macro): {val_metrics['f1_macro']:.4f}"
        )

        print("\n[Val] Per-Class Metrics:")
        for label in val_metrics["f1_per_class"].keys():
            print(
                f"  {label:>30s} | P: {val_metrics['precision_per_class'][label]:.4f} | "
                f"R: {val_metrics['recall_per_class'][label]:.4f} | "
                f"F1: {val_metrics['f1_per_class'][label]:.4f}"
            )

        print("\n[Val] Classification Report:")
        print(val_metrics["classification_report"])

        current_f1 = val_metrics["f1_macro"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            model.save_pretrained(args.output_path)
            tokenizer.save_pretrained(args.output_path)
            print(f"✅ New best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            print(
                f"Early stopping counter: {patience_counter}/{args.early_stopping_patience}"
            )
            if patience_counter >= args.early_stopping_patience:
                print(f"⏹ Early stopping triggered after {epoch} epochs")
                break

    print(f"\n{'=' * 60}")
    print(f"Loading best model from {args.output_path}...")
    best_model = AutoModelForSequenceClassification.from_pretrained(
        args.output_path,num_labels=NUM_CLASSES, id2label=ID2LABEL,label2id=LABEL2ID
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
    parser = argparse.ArgumentParser(description="Train BERT Classifier on BERT dataset")
    parser.add_argument("--train_path", type=str, default="./advisor_train.jsonl")
    parser.add_argument("--val_path", type=str, default="./advisor_val.jsonl")
    parser.add_argument("--model_name", type=str, default="hf1/chinese-roberta-wwm-ext")
    parser.add_argument("--output_path", type=str, default="./intent_classifier")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
