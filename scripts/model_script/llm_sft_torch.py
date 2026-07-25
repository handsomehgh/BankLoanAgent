import argparse
import gc
import json
import math
import os
from functools import partial
from typing import Optional, List

import torch
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


def format_sample(sample: dict, tokenizer) -> Optional[str]:
    if "messages" in sample:
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False
        )
        return text
    else:
        instruction = sample.get("instruction", "")
        input_text = sample.get("input", "")
        output_text = sample.get("output", "")

        if not instruction or not output_text:
            return None

        user_content = f"{instruction}\n{input_text}" if input_text else instruction
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_text}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenizer=False,
            add_generation_prompt=False
        )
        return text


class SFTDataset(Dataset):
    def __init__(self, file_path: str, tokenizer):
        self.samples = []
        skipped = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line.strip())
                text = format_sample(item, tokenizer)
                if text is None:
                    skipped += 1
                else:
                    self.samples.append(text)

        print(f"Loaded {len(self.samples)} samples from {file_path}" +
              (f" (skipped {skipped})" if skipped else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_fn(batch: List[str], tokenizer, max_length):
    encoded = tokenizer(batch, add_special_tokens=False, max_length=None, truncation=False, padding=False)

    assistant_marker = tokenizer.encode("[/INST]", add_special_tokens=False)

    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []

    for raw_id in encoded["input_ids"]:
        if len(raw_id) > max_length:
            raw_id = raw_id[:max_length]

        seq_len = len(raw_id)
        pad_len = max_length - seq_len

        input_ids = raw_id + [tokenizer.pad_token_id] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len

        labels = [-100] * max_length

        assistant_start = None
        marker_len = len(assistant_marker)
        for i in range(seq_len - marker_len + 1):
            if raw_id[i:i + marker_len] == assistant_marker:
                assistant_start = i + marker_len
                break

        if assistant_start is not None:
            for i in range(assistant_start, seq_len):
                labels[i] = raw_id[i]

        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        labels_batch.append(labels)

    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long)
    }


def load_model_and_tokenizer(args):
    compute_type = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_type
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb_config, device_map="auto",
                                                 trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.unk_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q", "v"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CASUAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def compute_loss_and_metrics(logits, labels, ignore_index=-100):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
    loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    active_mask = (shift_labels != ignore_index)
    if active_mask.sum() > 0:
        preds = shift_logits.argmax(-1)
        correct = (preds == shift_labels) & active_mask
        token_acc = correct.sum().float() / active_mask.sum().float()
    else:
        token_acc = 0

    return loss, token_acc


def train_epoch(args, model, loader, optimizer, scheduler, scaler, device, accumulation_steps, max_grad_norm):
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
            loss, token_acc = compute_loss_and_metrics(outputs.logits, labels)

        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
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
def eval_epoch(args, model, loader, device):
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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    model, tokenizer = load_model_and_tokenizer(args)

    train_dataset = SFTDataset(args.train_file, tokenizer)
    val_dataset = SFTDataset(args.val_file, tokenizer) if args.val_file else None

    train_collate = partial(collate_fn, tokenizer=tokenizer, max_length=args.max_length)
    val_collate = train_collate

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_collate,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=val_collate, num_workers=4, pin_memory=True
    ) if val_dataset else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_step = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = int(total_step * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_step)
    scaler = GradScaler(enabled=args.fp16)

    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
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
            args, model, train_loader, optimizer, scheduler, scaler, device,
            args.gradient_accumulation_steps, args.max_grad_norm
        )
        print(f"Train Loss: {train_loss:.4f} | Train Token Acc: {train_acc:.4f}")

        if val_loader:
            val_loss, val_acc, val_ppl = eval_epoch(args, model, val_loader, device)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Llama-2 LoRA SFT 训练脚本")

    # 模型与数据
    parser.add_argument("--model_name", type=str, required=True,
                        help="模型路径或HuggingFace模型名称")
    parser.add_argument("--train_file", type=str, required=True,
                        help="训练数据 JSONL 文件路径")
    parser.add_argument("--val_file", type=str, default=None,
                        help="验证数据 JSONL 文件路径（可选）")
    parser.add_argument("--output_dir", type=str, default="./lora_output",
                        help="输出目录（保存模型和训练状态）")

    # 训练超参数
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="每GPU批大小（微批次）")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                        help="预热步数占比")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪阈值")
    parser.add_argument("--max_length", type=int, default=512,
                        help="输入序列最大长度")

    # 混合精度
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="启用 FP16 混合精度")
    parser.add_argument("--bf16", action="store_true", default=False,
                        help="启用 BF16 混合精度")
    parser.add_argument("--no_cuda", action="store_true", default=False,
                        help="禁用 CUDA（强制使用 CPU）")

    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=8,
                        help="LoRA 秩")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="LoRA 缩放因子")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout 概率")

    # 早停与恢复
    parser.add_argument("--patience", type=int, default=3,
                        help="早停耐心值（验证集无改善的 epoch 数）")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从训练状态文件恢复（如 training_state.pt）")

    args = parser.parse_args()
    main(args)
