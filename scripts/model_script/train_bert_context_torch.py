# author hgh
# version 1.0
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import compute_class_weight
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = [
    "COMPLETE",
    "NEED_CONTEXT"
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
NUM_CLASSES = len(LABELS)


class IntentDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len):
        self.encodings = []
        self.labels = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                text_a = item.get("text_a", None)
                text_b = item.get("text_b", None)
                label_str = item["label"]
                if label_str not in LABEL2ID:
                    print(f"Warning: unknown label '{label_str}', skipping.")
                    continue
                label = LABEL2ID[label_str]

                encoding = tokenizer(
                    text_a,
                    text_b,
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                    return_tensors="pt"
                )
                self.encodings.append({
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze()
                })
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.encodings[idx]
        label = self.labels[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "label": torch.tensor(label, dtype=torch.long)
        }


def load_data(train_path, val_path, tokenizer, max_length, batch_size, eval_batch_size):
    train_dataset = IntentDataset(train_path, tokenizer, max_length)
    val_dataset = IntentDataset(val_path, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    return train_loader, val_loader, train_dataset.labels


def get_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    return model.to(DEVICE)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(loader), acc, f1_macro


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Validation"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(loader), acc, f1_macro


def main(args):
    global DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader, val_loader, train_labels = load_data(args.train_path, args.val_path, tokenizer, args.max_length,
                                                       args.batch_size, args.eval_batch_size)

    # class_weight = compute_class_weight(
    #     class_weight="balance",
    #     classes=np.array(LABELS),
    #     y=np.array([ID2LABEL[l] for l in train_labels])
    # )

    criterion = nn.CrossEntropyLoss()
    model = get_model(args.model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_f1 = 0.0
    patience_counter = 0
    os.makedirs(args.output_path, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch} / {args.epochs}")
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            # 保存完整模型和分词器
            model.save_pretrained(args.output_path)
            tokenizer.save_pretrained(args.output_path)
            print(f"  -> New best model saved to {args.output_path} (F1={best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    print(f"\nLoading best model from {args.output_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(args.output_path).to(DEVICE)
    final_loss, final_acc, final_f1 = eval_epoch(model, val_loader, criterion, DEVICE)
    print(f"\n{'=' * 50}")
    print(f"Final validation on best model:")
    print(f"Loss: {final_loss:.4f} | Acc: {final_acc:.4f} | F1: {final_f1:.4f}")
    print(f"{'=' * 50}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="上下文完整性 BERT 二分类训练")
    parser.add_argument("--train_path", type=str, default="context_train.jsonl", help="训练集文件路径")
    parser.add_argument("--val_path", type=str, default="context_val.jsonl", help="验证集文件路径")
    parser.add_argument("--model_name", type=str, default="hfl/chinese-roberta-wwm-ext", help="预训练模型名称")
    parser.add_argument("--output_path", type=str, default="./context_classifier_model", help="模型保存路径")
    parser.add_argument("--max_length", type=int, default=512, help="最大输入长度")
    parser.add_argument("--batch_size", type=int, default=16, help="批次大小")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="验证批次大小")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--early_stopping_patience", type=int, default=2, help="早停耐心值")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    main(args)