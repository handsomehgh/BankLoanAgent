import argparse
import json
import math
import os
from functools import partial

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import GradScaler, nn
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


class RerankerDataSet(Dataset):
    def __init__(self, file_path: str):
        self.samples = []
        with open(file_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line.strip())
                query = item.get("query", "")
                doc = item.get("document", "")
                label = item.get("label", None)
                if not query or not doc or not label:
                    continue
                self.samples.append(
                    {
                        "query": query,
                        "document": doc,
                        "label": label
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_length):
    query_list = [item["query"] for item in batch]
    doc_list = [item["document"] for item in batch]
    label_list = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    encoded = tokenizer(query_list, doc_list, max_length=max_length, padding="max_length", truncation=True,
                        return_tensors="pt")

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": label_list
    }


def load_model_and_tokenizer(model_name, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            model.resize_token_embeddings(len(tokenizer))
            print("Added [PAD] token and resized model embeddings.")

    model.config.pad_token_id = tokenizer.pad_token_id
    return model.to(device), tokenizer


def compute_metrics(all_labels, all_probs):
    labels = np.array(all_labels).astype(int)
    preds = (np.array(all_probs) > 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_positive = f1_score(labels, preds, pos_label=1)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, all_probs)
    except ValueError:
        auc = 0.5

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_positive": f1_positive,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }


def train_epoch(model, loader: DataLoader, optimizer, scheduler, scaler, criterion, max_grad_norm,
                gradient_accumulation_steps, device):
    model.train()
    all_labels, all_probs = [], []
    total_loss = 0.0

    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(dtype=torch.float16):
            outputs = model(inut_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels)

        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss * gradient_accumulation_steps
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    if (step + 1) % gradient_accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0

    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(dtype=torch.float16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, tokenizer = load_model_and_tokenizer(args.model_name, device)

    train_dataset = RerankerDataSet(args.train_file)
    val_dataset = RerankerDataSet(args.val_file)

    train_collate = partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length)
    val_collate = partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_collate,
                            num_workers=args.num_workers, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        model.parameters(),
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    scaler = GradScaler('cuda')
    criterion = nn.BCEWithLogitsLoss()

    best_metric = 0.0
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, criterion,
            args.max_grad_norm, args.gradient_accumulation_steps, device
        )
        print(f"Train Loss: {train_loss:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"F1(macro): {train_metrics['f1_macro']:.4f} | "
              f"F1(pos): {train_metrics['f1_positive']:.4f} | "
              f"Precision: {train_metrics['precision']:.4f} | "
              f"Recall: {train_metrics['recall']:.4f} | "
              f"AUC: {train_metrics['auc']:.4f}")

        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)
        print(f"Val   Loss: {val_loss:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"F1(macro): {val_metrics['f1_macro']:.4f} | "
              f"F1(pos): {val_metrics['f1_positive']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | "
              f"AUC: {val_metrics['auc']:.4f}")

        current_metric = val_metrics['auc']
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            print(f"  => saved best model (AUC {best_metric:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(args.output_dir, "best_model")
    ).to(device)
    _, final_metrics = eval_epoch(model, val_loader, criterion, device)
    print(f"\nFinal Results -> "
          f"Acc: {final_metrics['accuracy']:.4f} | "
          f"F1(macro): {final_metrics['f1_macro']:.4f} | "
          f"F1(pos): {final_metrics['f1_positive']:.4f} | "
          f"Precision: {final_metrics['precision']:.4f} | "
          f"Recall: {final_metrics['recall']:.4f} | "
          f"AUC: {final_metrics['auc']:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Reranker fine-tuning with Cross-Encoder")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-reranker-base",
                        help="Pretrained model name or path")
    parser.add_argument("--train_file", type=str, default="./train_reranker.jsonl",
                        help="Path to training JSONL file")
    parser.add_argument("--val_file", type=str, default="./val_reranker.jsonl",
                        help="Path to validation JSONL file")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/models/torch_reranker",
                        help="Directory to save model")
    parser.add_argument("--max_length", type=int, default=512, help="Max token length for tokenizer")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio of total steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
