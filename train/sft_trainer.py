#!/usr/bin/env python3
"""
SFT 微调训练器 —— 在预训练模型基础上做指令微调。

功能：
  - 加载预训练 checkpoint，处理词表扩展后的 shape 不匹配
  - 使用 <user>/<assistant> Chat 模板 + Loss Masking
  - 自动切分验证集（默认 10%）
  - Early Stopping（验证 loss 连续 N 个 epoch 未下降则停）
  - 复用预训练训练器的优化器 / 调度器 / 混合精度逻辑
  - 每 epoch 输出训练 loss、验证 loss 和 perplexity
  - Checkpoint 保存与恢复

用法示例：
    python train/sft_trainer.py \
        --tokenizer tokenizer/sft_tokenizer.json \
        --sft-data data/sft_data.jsonl \
        --pretrain-checkpoint checkpoints/checkpoint-latest.pt \
        --output-dir sft_checkpoints \
        --epochs 10 --lr 2e-4 --batch-size 8 --grad-accum-steps 4

    # 恢复训练：
    python train/sft_trainer.py ... --resume sft_checkpoints/checkpoint-latest.pt
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# ── 项目根目录导入 ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.model import GPT, print_param_count
from train.dataset import SFTChatDataset, sft_collate
from train.trainer import separate_weight_decay_params, get_cosine_warmup_scheduler


# ======================================================================
# SFT Trainer
# ======================================================================

class SFTTrainer:
    """SFT 微调训练器。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)

        # ── Tokenizer ──
        print(f"\n{'='*60}")
        print(f"  Loading tokenizer from {args.tokenizer}")
        from tokenizers import Tokenizer as HFTokenizer
        self.tokenizer = HFTokenizer.from_file(str(args.tokenizer))
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.pad_id = self.tokenizer.token_to_id("<pad>") or \
                      self.tokenizer.token_to_id("[PAD]") or 0
        self.eos_id = self.tokenizer.token_to_id("[EOS]") or 3
        print(f"  Vocab size: {self.vocab_size}, pad_id={self.pad_id}, eos_id={self.eos_id}")

        # ── Model ──
        self.model = self._build_model(args)
        print_param_count(self.model)

        # ── Mixed Precision ──
        self.amp_dtype: Optional[torch.dtype] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self._setup_mixed_precision()

        if self.device.type == "cuda":
            self._warmup_cuda()

        # ── Data ──
        self.train_loader, self.val_loader = self._build_dataloaders()

        # ── Optimizer & Scheduler ──
        param_groups = separate_weight_decay_params(
            self.model, args.weight_decay, args.lr
        )
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )

        total_steps = args.epochs * len(self.train_loader)
        self.scheduler = get_cosine_warmup_scheduler(
            self.optimizer, args.warmup_steps, total_steps
        )

        # ── 状态 ──
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.patience_counter = 0

        # ── Resume ──
        if args.resume:
            self.load_checkpoint(args.resume)

        # ── TensorBoard ──
        self.writer = None
        if args.tensorboard and args.log_dir:
            from torch.utils.tensorboard import SummaryWriter
            log_path = Path(args.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_path)
            print(f"  TensorBoard: {log_path.resolve()}")

    # ── Model Building ──────────────────────────────────────────────────

    def _build_model(self, args) -> GPT:
        """创建模型并加载预训练权重（处理词表扩展）。"""
        model = GPT(
            vocab_size=self.vocab_size,
            d_model=args.d_model,
            n_layer=args.n_layer,
            n_head=args.n_head,
            ffn_hidden=args.ffn_hidden,
            max_seq_len=args.max_length,
        )

        if args.pretrain_checkpoint:
            print(f"\n  Loading pretrain checkpoint: {args.pretrain_checkpoint}")
            ckpt = torch.load(args.pretrain_checkpoint, map_location="cpu")
            pretrained_state = ckpt["model_state_dict"]

            # 加载匹配的权重，处理 shape 不匹配（embedding / output head 维度变化）
            model_state = model.state_dict()
            loaded_keys = 0
            skipped_keys = 0
            extended_keys = 0

            for key, pretrained_param in pretrained_state.items():
                if key in model_state:
                    if pretrained_param.shape == model_state[key].shape:
                        model_state[key] = pretrained_param
                        loaded_keys += 1
                    elif key in ("token_embedding.weight", "output_head.weight"):
                        # 词表维度扩展：复制旧词表部分，新 token 保持随机 init
                        old_dim = pretrained_param.shape[0]
                        model_state[key][:old_dim] = pretrained_param
                        extended_keys += 1
                        print(f"    [extend] {key}: {pretrained_param.shape} → {model_state[key].shape}")
                    else:
                        skipped_keys += 1
                else:
                    skipped_keys += 1

            model.load_state_dict(model_state)
            print(f"  Pretrain weights loaded: {loaded_keys} loaded, "
                  f"{extended_keys} extended, {skipped_keys} skipped")

        self.model = model.to(self.device)
        return self.model

    def _setup_mixed_precision(self):
        """配置混合精度。"""
        if self.device.type != "cuda":
            print("  No CUDA → FP32")
            return
        mp = self.args.mixed_precision
        if mp == "bf16" and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            print("  Mixed precision: BF16")
        elif mp == "fp16":
            self.amp_dtype = torch.float16
            self.scaler = torch.cuda.amp.GradScaler()
            print("  Mixed precision: FP16 + GradScaler")
        else:
            print("  Mixed precision: FP32")

    def _warmup_cuda(self):
        dummy = torch.randint(0, 100, (2, 64), device=self.device)
        with torch.amp.autocast("cuda", dtype=self.amp_dtype or torch.float32,
                                enabled=self.amp_dtype is not None):
            _ = self.model(dummy).sum()
        print("  CUDA warmup done.")

    # ── Data ────────────────────────────────────────────────────────────

    def _build_dataloaders(self):
        """加载 SFT 数据，按比例切分训练 / 验证集。"""
        args = self.args
        full_dataset = SFTChatDataset(
            self.tokenizer,
            data_path=args.sft_data,
            max_length=args.max_length,
            verbose=True,
        )

        val_ratio = args.val_ratio
        val_size = max(1, int(len(full_dataset) * val_ratio))
        train_size = len(full_dataset) - val_size

        train_ds, val_ds = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )

        print(f"\n  Data split: train={train_size}, val={val_size} "
              f"(val_ratio={val_ratio:.0%})")

        def collate(batch):
            return sft_collate(batch, self.pad_id)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate, num_workers=args.num_workers,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate, num_workers=args.num_workers,
        )

        return train_loader, val_loader

    # ── Loss ────────────────────────────────────────────────────────────

    def compute_loss(self, model, input_ids, labels):
        """计算 SFT loss（已包含 loss masking）。"""
        logits = model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    # ── Training Step ───────────────────────────────────────────────────

    def train_step(self, batch) -> float:
        """一个优化步（含 gradient accumulation）。"""
        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        grad_accum = min(self.args.grad_accum_steps, input_ids.size(0))
        micro_batch = max(1, input_ids.size(0) // grad_accum)
        grad_accum = max(1, input_ids.size(0) // micro_batch)
        total_loss = 0.0

        for micro_idx in range(grad_accum):
            start = micro_idx * micro_batch
            end = min(start + micro_batch, input_ids.size(0))
            if start >= end:
                break
            mb_input = input_ids[start:end]
            mb_labels = labels[start:end]

            with torch.amp.autocast(
                "cuda" if self.device.type == "cuda" else "cpu",
                dtype=self.amp_dtype,
                enabled=self.amp_dtype is not None,
            ):
                loss = self.compute_loss(self.model, mb_input, mb_labels)
                loss = loss / grad_accum

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.item()

        # Gradient Clipping
        if self.args.grad_clip > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

        # Optimizer Step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        return total_loss

    # ── Validation ──────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> tuple:
        """在验证集上计算 loss 和 perplexity。"""
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            logits = self.model(input_ids)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += (shift_labels != -100).sum().item()

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 100))
        self.model.train()
        return avg_loss, perplexity

    # ── Checkpoint ──────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "args": vars(self.args),
        }

        latest_path = output_dir / "sft-latest.pt"
        torch.save(ckpt, latest_path)
        print(f"    [Checkpoint] saved: {latest_path} (epoch {epoch})")

        if epoch % self.args.save_interval == 0:
            step_path = output_dir / f"sft-epoch-{epoch:03d}.pt"
            torch.save(ckpt, step_path)
            print(f"    [Checkpoint] saved: {step_path}")

        if is_best:
            best_path = output_dir / "sft-best.pt"
            torch.save(ckpt, best_path)
            print(f"    [Checkpoint] best model: {best_path}")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.best_epoch = ckpt.get("best_epoch", -1)
        print(f"  [Resume] loaded from {path} (epoch {ckpt['epoch']})")

    # ── Generate Sample ─────────────────────────────────────────────────

    @torch.no_grad()
    def generate_sample(self, rough_input: str, max_new_tokens: int = 80) -> str:
        """为 rough_input 生成精炼提示词。"""
        prompt_text = f"<user>{rough_input}</user><assistant>"
        encoding = self.tokenizer.encode(prompt_text)
        input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=self.device)

        # 超出长度从左侧截断
        if input_ids.size(1) > self.args.max_length:
            input_ids = input_ids[:, -self.args.max_length:]

        out = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=self.args.gen_temperature,
            top_k=self.args.gen_top_k,
        )

        full_text = self.tokenizer.decode(out[0].tolist())
        return extract_assistant_response(full_text)

    # ── Main Loop ──────────────────────────────────────────────────────

    def train(self):
        args = self.args
        epochs = args.epochs
        print(f"\n{'='*60}")
        print(f"  SFT Training")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {epochs}")
        print(f"  Train batches/epoch: {len(self.train_loader)}")
        print(f"  Val batches/epoch:   {len(self.val_loader)}")
        eff_batch = args.batch_size * args.grad_accum_steps
        print(f"  Effective batch size: {eff_batch}")
        print(f"  LR: {args.lr} | Warmup: {args.warmup_steps} | Weight decay: {args.weight_decay}")
        print(f"  Early stopping patience: {args.patience}")
        print(f"{'='*60}\n")

        global_step = 0

        for epoch in range(self.start_epoch, epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start = time.time()

            # ── Train ──
            for batch in self.train_loader:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                global_step += 1

                # 打印损失
                if global_step % args.log_interval == 0:
                    avg = epoch_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    print(f"    Step {global_step:>6d} | loss {avg:.4f} | lr {lr:.2e}")

            avg_train_loss = epoch_loss / max(num_batches, 1)

            # ── Validate ──
            val_loss, val_ppl = self.evaluate()
            elapsed = time.time() - epoch_start

            improved = val_loss < self.best_val_loss - 1e-6
            self.best_val_loss = min(self.best_val_loss, val_loss)

            # ── Log ──
            print()
            print(f"  ╔══ Epoch {epoch+1:>2d}/{epochs} ═══════════════════════════════╗")
            print(f"  ║ Train loss: {avg_train_loss:.4f}  Val loss: {val_loss:.4f}    ║")
            print(f"  ║ Perplexity: {val_ppl:.2f}  Time: {elapsed:.1f}s            ║")
            if improved:
                print(f"  ║ ↓ Best val loss! (prev best: {self.best_epoch+1})        ║")
                self.best_epoch = epoch
                self.patience_counter = 0
                print(f"  ║ Patience reset.                                       ║")
            else:
                self.patience_counter += 1
                print(f"  ║ No improvement ({self.patience_counter}/{args.patience})         ║")
            print(f"  ╚══════════════════════════════════════════════════╝")
            print()

            # ── 生成样例 ──
            try:
                sample = self.generate_sample(
                    "帮我写一个Python脚本来读取和处理CSV文件",
                    max_new_tokens=60,
                )
                print(f"  ── Generation Sample ──")
                print(f"  {sample[:300]}")
                print(f"  ──────────────────────")
                print()
            except Exception as e:
                print(f"  [Warn] Generation failed: {e}")

            # ── TensorBoard ──
            if self.writer:
                self.writer.add_scalar("sft/train_loss", avg_train_loss, epoch)
                self.writer.add_scalar("sft/val_loss", val_loss, epoch)
                self.writer.add_scalar("sft/val_ppl", val_ppl, epoch)
                self.writer.add_scalar("sft/lr", self.scheduler.get_last_lr()[0], epoch)

            # ── Save ──
            self.save_checkpoint(epoch, is_best=improved)

            # ── Early Stopping ──
            if self.patience_counter >= args.patience:
                print(f"  Early stopping triggered after {epoch+1} epochs "
                      f"(no improvement for {args.patience} consecutive epochs).")
                print(f"  Best epoch: {self.best_epoch+1} (val_loss={self.best_val_loss:.4f})")
                break

        # ── Final ──
        print(f"\n{'='*60}")
        print(f"  SFT Training Complete!")
        print(f"  Best epoch: {self.best_epoch+1} (val_loss={self.best_val_loss:.4f})")
        print(f"{'='*60}")

        if self.writer:
            self.writer.close()


# ======================================================================
# 辅助：提取 assistant 回复
# ======================================================================

def extract_assistant_response(full_text: str) -> str:
    """从模型生成的完整文本中提取 <assistant> 和 </assistant> 之间的内容。"""
    marker = "<assistant>"
    start = full_text.find(marker)
    if start == -1:
        return full_text.strip()
    content = full_text[start + len(marker):]
    end_marker = "</assistant>"
    end = content.find(end_marker)
    if end != -1:
        content = content[:end]
    return content.strip()


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT 微调训练器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 模型 ──
    mg = parser.add_argument_group("Model")
    mg.add_argument("--d-model", type=int, default=256)
    mg.add_argument("--n-layer", type=int, default=6)
    mg.add_argument("--n-head", type=int, default=8)
    mg.add_argument("--ffn-hidden", type=int, default=1024)
    mg.add_argument("--max-length", type=int, default=512)
    mg.add_argument("--rope-theta", type=float, default=10000.0)
    mg.add_argument("--pretrain-checkpoint", default=None,
                    help="预训练 checkpoint 路径（加载模型权重）")

    # ── 数据 ──
    dg = parser.add_argument_group("Data")
    dg.add_argument("--tokenizer", required=True,
                    help="Tokenizer JSON 路径（建议使用扩展后的 sft_tokenizer.json）")
    dg.add_argument("--sft-data", required=True,
                    help="SFT 数据 JSONL 路径（含 rough_input, refined_prompt）")
    dg.add_argument("--val-ratio", type=float, default=0.1,
                    help="验证集比例 (default: 0.1)")
    dg.add_argument("--num-workers", type=int, default=0)

    # ── 训练 ──
    tg = parser.add_argument_group("Training")
    tg.add_argument("--epochs", type=int, default=10, help="最大训练 epoch 数")
    tg.add_argument("--batch-size", type=int, default=8)
    tg.add_argument("--grad-accum-steps", type=int, default=4)
    tg.add_argument("--lr", type=float, default=2e-4,
                    help="学习率（比预训练阶段小）")
    tg.add_argument("--weight-decay", type=float, default=0.1)
    tg.add_argument("--adam-beta1", type=float, default=0.9)
    tg.add_argument("--adam-beta2", type=float, default=0.95)
    tg.add_argument("--adam-eps", type=float, default=1e-8)
    tg.add_argument("--warmup-steps", type=int, default=100)
    tg.add_argument("--grad-clip", type=float, default=1.0)
    tg.add_argument("--mixed-precision", choices=["bf16", "fp16", "no"], default="bf16")

    # ── Early Stopping ──
    esg = parser.add_argument_group("Early Stopping")
    esg.add_argument("--patience", type=int, default=3,
                     help="验证 loss 连续 N 个 epoch 不下降则停")

    # ── 生成 ──
    gg = parser.add_argument_group("Generation")
    gg.add_argument("--gen-temperature", type=float, default=0.8)
    gg.add_argument("--gen-top-k", type=int, default=10)

    # ── 日志 / 保存 ──
    lg = parser.add_argument_group("Logging & Saving")
    lg.add_argument("--log-dir", default="runs/sft")
    lg.add_argument("--output-dir", default="sft_checkpoints")
    lg.add_argument("--log-interval", type=int, default=10)
    lg.add_argument("--save-interval", type=int, default=1, help="每 N 个 epoch 保存一次")
    lg.add_argument("--tensorboard", action="store_true", default=True)
    lg.add_argument("--no-tensorboard", action="store_false", dest="tensorboard")

    # ── 其他 ──
    og = parser.add_argument_group("Other")
    og.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    og.add_argument("--seed", type=int, default=42)
    og.add_argument("--resume", default=None, help="从 checkpoint 恢复")

    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    return args


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print()
    print("=" * 60)
    print("  SFT Config")
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:25s} = {v}")
    print("=" * 60)

    trainer = SFTTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
