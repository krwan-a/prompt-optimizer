#!/usr/bin/env python3
"""
预训练训练器 —— 完整的 GPT 预训练流程。

功能：
  - AdamW（weight decay 只施加在 Linear 层，不施加在 Norm / Bias）
  - Cosine LR Schedule + Warmup
  - 梯度裁剪
  - BF16 / FP16 混合精度训练
  - Gradient Accumulation
  - 定期验证（loss + perplexity + 生成样例）
  - Checkpoint 保存与恢复 (model + optimizer + scheduler + step)
  - TensorBoard / 纯 print 双模式

用法示例：
    python train/trainer.py \
        --tokenizer tokenizer/prompt_opt_tokenizer.json \
        --train-corpus data/clean_corpus.txt \
        --val-corpus data/clean_corpus.txt \
        --batch-size 16 --grad-accum-steps 4 \
        --max-steps 10000 --lr 3e-4 \
        --output-dir checkpoints --log-dir runs

    # 从 checkpoint 恢复：
    python train/trainer.py ... --resume checkpoints/checkpoint-latest.pt
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── 将项目根目录加入 sys.path ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.model import GPT, print_param_count


# ======================================================================
# 参数分组 —— 哪些参数施加 weight decay
# ======================================================================

def separate_weight_decay_params(model: nn.Module, weight_decay: float, lr: float):
    """
    将参数分为两组：
      - decay: 所有 2D+ 权重（Linear 层）— 施加 weight_decay
      - no_decay: 所有 1D 参数（norm 权重）和 bias — 不施加 weight_decay

    Returns: param_groups for AdamW
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 1D 参数（RMSNorm weight）和任何名为 bias 的参数 ←→ no_decay
        if param.ndim == 1 or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params,    "weight_decay": weight_decay, "lr": lr},
        {"params": no_decay_params, "weight_decay": 0.0,         "lr": lr},
    ]

    total_decay = sum(p.numel() for p in decay_params)
    total_no_decay = sum(p.numel() for p in no_decay_params)

    print(f"  Weight decay applied to:  {total_decay:>10,} params ({len(decay_params)} tensors)")
    print(f"  No weight decay:          {total_no_decay:>10,} params ({len(no_decay_params)} tensors)")

    return groups


# ======================================================================
# Cosine Schedule + Warmup (LambdaLR)
# ======================================================================

def get_cosine_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Cosine decay with linear warmup."""
    def lr_lambda(step: int) -> float:
        # Warmup
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ======================================================================
# Trainer
# ======================================================================

class Trainer:
    """GPT 预训练器。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(args.device)

        # ── Tokenizer ──
        print(f"\n{'='*60}")
        print(f"  Loading tokenizer from {args.tokenizer}")
        from tokenizers import Tokenizer as HFTokenizer
        self.tokenizer = HFTokenizer.from_file(str(args.tokenizer))

        # ── Model ──
        self.model = GPT(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layer=args.n_layer,
            n_head=args.n_head,
            ffn_hidden=args.ffn_hidden,
            max_seq_len=args.max_length,
            rope_theta=args.rope_theta,
        ).to(self.device)
        print_param_count(self.model)

        # ── Mixed Precision ──
        self.amp_dtype: Optional[torch.dtype] = None
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None
        self._setup_mixed_precision()

        # 如果有 GPU，先做 warmup forward（初始化 CUDA 内核）
        if self.device.type == "cuda":
            self._warmup_cuda()

        # ── Data ──
        self.train_loader, self.val_loader, self.bos_id = self._build_dataloaders()

        # ── Optimizer ──
        param_groups = separate_weight_decay_params(
            self.model, args.weight_decay, args.lr
        )
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
        )

        # ── Scheduler ──
        self.scheduler = get_cosine_warmup_scheduler(
            self.optimizer, args.warmup_steps, args.max_steps
        )

        # ── 状态 ──
        self.start_step = 0
        self.best_val_loss = float("inf")
        self.best_val_ppl = float("inf")

        # ── Resume ──
        if args.resume:
            self.load_checkpoint(args.resume)

        # ── Logger (TensorBoard) ──
        self.use_tensorboard = args.tensorboard and args.log_dir is not None
        self.writer = None
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            log_path = Path(args.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_path)
            print(f"  TensorBoard log: {log_path.resolve()}")

    # ── AMP ────────────────────────────────────────────────────────────

    def _setup_mixed_precision(self):
        """配置混合精度。"""
        if self.device.type != "cuda":
            print("  No CUDA → 使用 FP32")
            return

        mp = self.args.mixed_precision
        if mp == "bf16" and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
            print(f"  Mixed precision: BF16")
        elif mp == "fp16":
            self.amp_dtype = torch.float16
            self.scaler = torch.cuda.amp.GradScaler()
            print(f"  Mixed precision: FP16 + GradScaler")
        else:
            print(f"  Mixed precision: FP32 (amp={mp})")

    def _warmup_cuda(self):
        """空 forward/backward 触发 CUDA 初始化。"""
        print("  Warming up CUDA...")
        dummy = torch.randint(0, 100, (2, 64), device=self.device)
        with torch.amp.autocast("cuda", dtype=self.amp_dtype or torch.float32, enabled=self.amp_dtype is not None):
            _ = self.model(dummy).sum()
        print("  CUDA ready.")

    # ── Data ───────────────────────────────────────────────────────────

    def _build_dataloaders(self):
        """构建训练和验证 DataLoader。"""
        from train.dataset import PretrainDataset, create_dataloader

        args = self.args
        bos_id = self.tokenizer.token_to_id("[BOS]")
        if bos_id is None:
            bos_id = self.tokenizer.token_to_id("<s>") or 2
            print(f"  [Warn] [BOS] not found, using fallback ID={bos_id}")

        # 训练集
        print(f"\n  Loading training data: {args.train_corpus}")
        train_ds = PretrainDataset(
            self.tokenizer,
            corpus_path=args.train_corpus,
            max_length=args.max_length,
            stride=args.max_length // 2,
            verbose=True,
        )
        train_loader = create_dataloader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            is_sft=False,
            num_workers=args.num_workers,
        )

        # 验证集
        val_loader = None
        if args.val_corpus:
            print(f"\n  Loading validation data: {args.val_corpus}")
            val_ds = PretrainDataset(
                self.tokenizer,
                corpus_path=args.val_corpus,
                max_length=args.max_length,
                stride=args.max_length,
                verbose=True,
            )
            val_loader = create_dataloader(
                val_ds,
                batch_size=args.batch_size,
                shuffle=False,
                is_sft=False,
                num_workers=args.num_workers,
            )

        return train_loader, val_loader, bos_id

    # ── Loss ───────────────────────────────────────────────────────────

    def compute_loss(
        self, model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 autoregressive LM loss。
        对 logits 做 shift（predict next token）。
        """
        logits = model(input_ids)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    # ── Training Step ──────────────────────────────────────────────────

    def train_step(self, batch) -> float:
        """一次参数更新（含 gradient accumulation）。"""
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

        # ── Gradient Clipping ──
        if self.args.grad_clip > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.args.grad_clip
            )

        # ── Optimizer Step ──
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
    def evaluate(self) -> Tuple[float, float, str]:
        """
    在验证集上计算平均 loss 和 perplexity，并生成一段样例文本。

    Returns:
        (avg_loss, perplexity, generated_text)
        """
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
        perplexity = math.exp(min(avg_loss, 100))  # 防止 exp 爆炸

        # ── 生成样例 ──
        generated_text = ""
        try:
            # 从 BOS token 开始自回归生成
            prompt = torch.full((1, 1), self.bos_id, dtype=torch.long, device=self.device)
            out_ids = self.model.generate(
                prompt,
                max_new_tokens=self.args.max_length - 1,
                temperature=self.args.gen_temperature,
                top_k=self.args.gen_top_k,
            )
            generated_text = self.tokenizer.decode(out_ids[0].tolist())
        except Exception as e:
            generated_text = f"[Generation failed: {e}]"

        self.model.train()
        return avg_loss, perplexity, generated_text

    # ── Checkpoint ──────────────────────────────────────────────────────

    def save_checkpoint(self, step: int, is_best: bool = False):
        """保存完整训练状态。"""
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ckpt: Dict[str, Any] = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_ppl": self.best_val_ppl,
            "args": vars(self.args),
        }

        # 保存最新
        latest_path = output_dir / "checkpoint-latest.pt"
        torch.save(ckpt, latest_path)
        print(f"  [Checkpoint] saved to {latest_path} (step {step})")

        # 定期保存 step 标记
        if step % self.args.save_interval == 0:
            step_path = output_dir / f"checkpoint-{step:07d}.pt"
            torch.save(ckpt, step_path)
            print(f"  [Checkpoint] saved to {step_path}")

        # 最佳模型
        if is_best:
            best_path = output_dir / "checkpoint-best.pt"
            torch.save(ckpt, best_path)
            print(f"  [Checkpoint] best model saved to {best_path}")

    def load_checkpoint(self, path: str):
        """从 checkpoint 恢复训练状态。"""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_step = ckpt["step"] + 1
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.best_val_ppl = ckpt.get("best_val_ppl", float("inf"))

        print(f"  [Resume] loaded from {path} (step {ckpt['step']})")
        print(f"  [Resume] best val loss: {self.best_val_loss:.4f}, "
              f"perplexity: {self.best_val_ppl:.4f}")

    # ── Main Loop ──────────────────────────────────────────────────────

    def train(self):
        args = self.args
        print(f"\n{'='*60}")
        print(f"  训练开始")
        print(f"  设备: {self.device}")
        print(f"  最大步数: {args.max_steps}")
        print(f"  Batch size (per GPU): {args.batch_size}")
        print(f"  Gradient accumulation: {args.grad_accum_steps}")
        eff_batch = args.batch_size * args.grad_accum_steps
        eff_tokens = eff_batch * args.max_length
        print(f"  Effective batch size: {eff_batch}")
        print(f"  Effective tokens/step: {eff_tokens:,}")
        print(f"  Learning rate: {args.lr}")
        print(f"  Warmup steps: {args.warmup_steps}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  Gradient clip: {args.grad_clip}")
        print(f"{'='*60}\n")

        # ── 数据循环迭代器 ──
        def infinite_loader(dl):
            while True:
                for batch in dl:
                    yield batch

        train_iter = infinite_loader(self.train_loader)
        total_steps = args.max_steps

        # 计时变量
        accum_time = 0.0
        accum_loss = 0.0
        start_time = time.time()

        for step in range(self.start_step, total_steps):
            # ── Train ──
            loss = self.train_step(next(train_iter))
            accum_loss += loss
            accum_time += time.time() - start_time
            start_time = time.time()

            # ── Log ──
            if step % args.log_interval == 0:
                avg_loss = accum_loss / args.log_interval
                ppl = math.exp(min(avg_loss, 100))
                lr = self.scheduler.get_last_lr()[0]
                ms_per_step = accum_time / args.log_interval * 1000

                print(
                    f"  Step {step:>7d}/{total_steps:<7d} | "
                    f"loss {avg_loss:>7.4f} | "
                    f"ppl {ppl:>7.2f} | "
                    f"lr {lr:.2e} | "
                    f"{ms_per_step:.1f} ms/step"
                )

                if self.writer:
                    self.writer.add_scalar("train/loss", avg_loss, step)
                    self.writer.add_scalar("train/ppl", ppl, step)
                    self.writer.add_scalar("train/lr", lr, step)

                accum_loss = 0.0
                accum_time = 0.0

            # ── Eval ──
            if step % args.eval_interval == 0 and step > 0 and self.val_loader is not None:
                val_loss, val_ppl, generated = self.evaluate()
                improved = val_loss < self.best_val_loss
                self.best_val_loss = min(self.best_val_loss, val_loss)
                self.best_val_ppl = min(self.best_val_ppl, val_ppl)

                print()
                print(f"  ╔══ EVAL @ step {step} ═══╗")
                print(f"  ║ Val loss:    {val_loss:>8.4f}    ║")
                print(f"  ║ Perplexity:  {val_ppl:>8.2f}    ║")
                if improved:
                    print(f"  ║ ↓ Best loss!           ║")
                print(f"  ╚══════════════════════════╝")
                print()

                # 打印生成样例（截断到合理长度）
                gen_preview = generated[:500] if len(generated) > 500 else generated
                print(f"  ── Generated Sample ──")
                print(f"  {gen_preview}")
                print(f"  ──────────────────────")
                print()

                if self.writer:
                    self.writer.add_scalar("val/loss", val_loss, step)
                    self.writer.add_scalar("val/ppl", val_ppl, step)
                    self.writer.add_text("val/generated", gen_preview, step)

                # 保存最佳
                if improved:
                    self.save_checkpoint(step, is_best=True)

            # ── Save ──
            if step % args.save_interval == 0 and step > 0:
                self.save_checkpoint(step)

        # ── 最终保存 ──
        self.save_checkpoint(total_steps - 1)

        # 最终验证
        if self.val_loader is not None:
            val_loss, val_ppl, generated = self.evaluate()
            print(f"\n  Final eval: loss={val_loss:.4f}, ppl={val_ppl:.2f}")

        print(f"\n{'='*60}")
        print(f"  训练完成！")
        print(f"{'='*60}")

        if self.writer:
            self.writer.close()


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT 预训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 模型参数 ──────────────────────────────────────────────────────
    model_group = parser.add_argument_group("Model Architecture")
    model_group.add_argument("--vocab-size", type=int, default=8000,
                             help="词表大小")
    model_group.add_argument("--d-model", type=int, default=256,
                             help="模型维度")
    model_group.add_argument("--n-layer", type=int, default=6,
                             help="Transformer 层数")
    model_group.add_argument("--n-head", type=int, default=8,
                             help="注意力头数")
    model_group.add_argument("--ffn-hidden", type=int, default=1024,
                             help="SwiGLU FFN 隐藏维度")
    model_group.add_argument("--max-length", type=int, default=512,
                             help="最大序列长度 (context length)")
    model_group.add_argument("--rope-theta", type=float, default=10000.0,
                             help="RoPE base frequency")

    # ── 数据参数 ──────────────────────────────────────────────────────
    data_group = parser.add_argument_group("Data")
    data_group.add_argument("--tokenizer", required=True,
                            help="Tokenizer JSON 路径")
    data_group.add_argument("--train-corpus", required=True,
                            help="训练语料文件（纯文本）")
    data_group.add_argument("--val-corpus", default=None,
                            help="验证语料文件（缺省时不验证）")
    data_group.add_argument("--num-workers", type=int, default=0,
                            help="DataLoader 子进程数")

    # ── 训练参数 ──────────────────────────────────────────────────────
    train_group = parser.add_argument_group("Training")
    train_group.add_argument("--batch-size", type=int, default=16,
                             help="每个 step 的 batch size（per GPU）")
    train_group.add_argument("--grad-accum-steps", type=int, default=4,
                             help="梯度累积步数")
    train_group.add_argument("--max-steps", type=int, default=10000,
                             help="最大训练步数")
    train_group.add_argument("--lr", type=float, default=3e-4,
                             help="最大学习率")
    train_group.add_argument("--weight-decay", type=float, default=0.1,
                             help="Weight decay")
    train_group.add_argument("--adam-beta1", type=float, default=0.9,
                             help="Adam beta1")
    train_group.add_argument("--adam-beta2", type=float, default=0.95,
                             help="Adam beta2")
    train_group.add_argument("--adam-eps", type=float, default=1e-8,
                             help="Adam epsilon")
    train_group.add_argument("--warmup-steps", type=int, default=500,
                             help="学习率预热步数")
    train_group.add_argument("--grad-clip", type=float, default=1.0,
                             help="梯度裁剪最大范数 (0=disable)")
    train_group.add_argument("--mixed-precision", type=str,
                             choices=["bf16", "fp16", "no"], default="bf16",
                             help="混合精度模式")

    # ── 生成参数 ──────────────────────────────────────────────────────
    gen_group = parser.add_argument_group("Generation (eval)")
    gen_group.add_argument("--gen-temperature", type=float, default=0.8,
                           help="生成采样温度")
    gen_group.add_argument("--gen-top-k", type=int, default=10,
                           help="生成 Top-K 截断")

    # ── 日志 / 保存 ──────────────────────────────────────────────────
    log_group = parser.add_argument_group("Logging & Saving")
    log_group.add_argument("--log-dir", default="runs",
                           help="TensorBoard 日志目录（None=不写 TensorBoard）")
    log_group.add_argument("--output-dir", default="checkpoints",
                           help="Checkpoint 保存目录")
    log_group.add_argument("--log-interval", type=int, default=10,
                           help="日志打印间隔（步数）")
    log_group.add_argument("--eval-interval", type=int, default=200,
                           help="验证间隔（步数）")
    log_group.add_argument("--save-interval", type=int, default=1000,
                           help="Checkpoint 保存间隔（步数）")
    log_group.add_argument("--tensorboard", action="store_true", default=True,
                           help="启用 TensorBoard")
    log_group.add_argument("--no-tensorboard", action="store_false",
                           dest="tensorboard",
                           help="禁用 TensorBoard，仅用 print")
    log_group.add_argument("--resume", default=None,
                           help="从 checkpoint 恢复训练")

    # ── 其他 ──────────────────────────────────────────────────────────
    other_group = parser.add_argument_group("Other")
    other_group.add_argument("--device", default="auto",
                             choices=["auto", "cuda", "cpu"],
                             help="训练设备")
    other_group.add_argument("--seed", type=int, default=42,
                             help="随机种子")

    args = parser.parse_args()

    # ── 设备解析 ──
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    # 随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 打印配置
    print()
    print("=" * 60)
    print("  配置 (Configuration)")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"  {key:25s} = {value}")
    print("=" * 60)

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
