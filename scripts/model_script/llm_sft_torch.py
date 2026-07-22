import argparse
import gc
import json
import math
import os
from typing import Dict, List

import numpy as np
import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torch import nn, GradScaler
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, \
    get_linear_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--train_file", type=str, required=True, default="./model/train.jsonl")
parser.add_argument("--val_file", type=str, required=True, default="./model/val.jsonl")
parser.add_argument("--output_dir", type=str, default="./qlora_sft_output")
parser.add_argument("--resume_from", type=str, default=None, help="断点续训 checkpoint 路径")
parser.add_argument("--max_length", type=int, default=2048)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--learning_rate", type=float, default=2e-4)
parser.add_argument("--warmup_ratio", type=float, default=0.03)
parser.add_argument("--weight_decay", type=float, default=0.001)
parser.add_argument("--max_grad_norm", type=float, default=0.3)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--fp16", action="store_true", help="启用 FP16 混合精度")
parser.add_argument("--bf16", action="store_true", help="启用 BF16 混合精度")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--patience", type=int, default=2, help="早停耐心值（无验证集时忽略）")
parser.add_argument("--use_chat_template", action="store_true", default=True,
                    help="使用 tokenizer.apply_chat_template 自动拼接（推荐）")
parser.add_argument("--no_cuda", action="store_true")
args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
torch.manual_seed(args.seed)
np.random.seed(args.seed)


def format_sample(sample: dict, tokenizer) -> str:
    """
        将一条样本转换为训练所需的文本字符串。
        支持两种格式：
          1) {"instruction": "...", "input": "...", "output": "..."}  （单轮）
          2) {"messages": [{"role": "user", "content": "..."}, ...]} （多轮）
        返回完整的对话文本，可直接 tokenize。
    """
    if "messages" in sample:
        return tokenizer.apply_chat_template(
            sample["messages"],
            tokenizer=False,
            add_generation_prompt=False
        )
    else:
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output = sample.get("output", "")
        if not instruction or not output:
            return None
        if input_text:
            user_content = f"{instruction}\n{input_text}"
        else:
            user_content = instruction
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenizer=False,
            add_generation_prompt=False
        )


class SFTDataset(Dataset):
    def __init__(self, file_path: str, tokenizer):
        self.samples = []
        skipped = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                text = format_sample(item, tokenizer)
                if text is None:
                    skipped += 1
        print(f"Loaded {len(self.samples)} samples from {file_path}" +
              (f" (skipped {skipped})" if skipped else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[str], tokenizer, max_length: int) -> Dict[str, torch.Tensor]:
    enc = tokenizer(batch, padding=False, truncation=False, max_length=max_length)

    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    assistant_marker = tokenizer.encode("assistant", add_special_tokens=False)

    for raw_ids in enc["input_ids"]:
        if len(raw_ids) > max_length:
            head = raw_ids[:512]
            tail = raw_ids[-(max_length - 512):]
            raw_ids = head + tail

        seq_len = len(raw_ids)
        pad_len = max_length - seq_len
        input_ids = raw_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len

        labels = [-100] * max_length
        assistant_start = None
        for i in range(len(raw_ids) - len(attention_mask) + 1):
            if raw_ids[i:i + len(assistant_marker)] == assistant_marker:
                assistant_start = i + len(assistant_marker)
                break
        if assistant_start is None:
            assistant_start = 0
        for i in range(assistant_start, seq_len):
            labels[i] = raw_ids[i]

        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        labels_batch.append(labels)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
    }


def load_model_and_tokenizer(model_name: str, resume_ckpt=None):
    compute_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules="all-linear",
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def compute_loss_and_metrics(logits, labels, ignore_index=-100):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., :1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    active_mask = (shift_labels != ignore_index)
    if active_mask.sum() > 0:
        preds = shift_logits.argmax(dim=-1)
        correct = (preds == shift_labels) & active_mask
        token_acc = correct.sum().float() / active_mask.sum().float()
    else:
        token_acc = 0.0

    return loss, token_acc


def train_epoch(model, loader, optimizer, scheduler, scaler, device, accumulation_steps, max_grad_norm):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    recent_losses = []
    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(enabled=args.fp16 or args.bf16):
            outputs = model(input_ids, attention_mask=attention_mask)
            loss, token_acc = compute_loss_and_metrics(outputs.logits, labels, ignore_index=-100)

        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscaler_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        real_loss = loss.item() * accumulation_steps
        total_loss += real_loss
        total_acc += token_acc.item()
        recent_losses.append(real_loss)
        if len(recent_losses) > 50:
            recent_losses.pop(0)

        if (step + 1) % (10 * accumulation_steps) == 0:
            avg_loss = sum(recent_losses) / len(recent_losses)
            print(
                f"  Step {step + 1} | Loss: {avg_loss:.4f} | Token Acc: {token_acc.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast(enabled=args.fp16 or args.bf16):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss, token_acc = compute_loss_and_metrics(outputs.logits, labels)

        total_loss += loss.item()
        total_acc += token_acc.item()

    avg_loss = total_loss / len(loader)
    avg_acc = total_acc / len(loader)
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, avg_acc, perplexity


def main():
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.resume_from)

    # 数据
    train_dataset = SFTDataset(args.train_file, tokenizer)
    val_dataset = SFTDataset(args.val_file, tokenizer) if args.val_file else None

    from functools import partial
    train_collate = partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length)
    val_collate = train_collate

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=train_collate, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=val_collate, num_workers=4, pin_memory=True
    ) if val_dataset else None

    # 优化器与调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    scaler = GradScaler(enabled=args.fp16)

    # 断点续训
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        patience_counter = checkpoint.get('patience_counter', 0)
        print(f"Resumed from {args.resume_from}, starting epoch {start_epoch}")

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'=' * 60}\nEpoch {epoch}/{args.epochs}\n{'=' * 60}")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, DEVICE,
            args.gradient_accumulation_steps, args.max_grad_norm
        )
        print(f"Train Loss: {train_loss:.4f} | Train Token Acc: {train_acc:.4f}")

        if val_loader:
            val_loss, val_acc, val_ppl = eval_epoch(model, val_loader, DEVICE)
            print(f"Val Loss: {val_loss:.4f} | Val Token Acc: {val_acc:.4f} | Val Perplexity: {val_ppl:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型（LoRA 权重）
                model.save_pretrained(os.path.join(args.output_dir, "best_model"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
                # 保存训练状态
                state = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'patience_counter': patience_counter,
                }
                torch.save(state, os.path.join(args.output_dir, "training_state.pt"))
                print(f"  => saved best model (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        else:
            model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-epoch{epoch}"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, f"checkpoint-epoch{epoch}"))

        # 清理缓存
        torch.cuda.empty_cache()
        gc.collect()

    print("Training finished. Model saved to", args.output_dir)


if __name__ == "__main__":
    main()
