#!/usr/bin/env python3
"""
Decoder-Only Transformer（缩小版 Llama 架构）

包含：RMSNorm, RoPE, SwiGLU FFN, Causal Self-Attention, Pre-Norm

架构参数：
  vocab_size=8000, context_length=512, d_model=256,
  n_layer=6, n_head=8 (head_dim=32), ffn_hidden=1024
  总参数 ≈ 10.4M

用法：
    from model.model import GPT, print_param_count
    model = GPT(vocab_size=8000, d_model=256, n_layer=6, n_head=8, ffn_hidden=1024)
    print_param_count(model)
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# RMSNorm
# ======================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (https://arxiv.org/abs/1910.07467)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ======================================================================
# RoPE — Rotary Position Embedding
# ======================================================================

def precompute_rope_freqs(
    head_dim: int, max_len: int, theta: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 sin / cos 频率表。

    Args:
        head_dim: 每个注意力头的维度（必须为偶数）
        max_len: 最大序列长度
        theta: RoPE base frequency

    Returns:
        cos: (max_len, head_dim/2)  余弦值
        sin: (max_len, head_dim/2)  正弦值
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # (max_len, head_dim//2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 Q 和 K 施加旋转位置编码。

    Args:
        xq: (B, n_head, L, head_dim)  query
        xk: (B, n_head, L, head_dim)  key
        cos: (max_len, head_dim/2)    预计算余弦
        sin: (max_len, head_dim/2)    预计算正弦

    Returns:
        (xq_rotated, xk_rotated)    形状与输入相同
    """
    seq_len = xq.shape[-2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, L, D/2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    # 将最后一维拆成两半做复数旋转
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# ======================================================================
# Causal Self-Attention  (用 scaled_dot_product_attention)
# ======================================================================

class CausalSelfAttention(nn.Module):
    """标准因果自注意力 + RoPE。"""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"

        self.n_head = n_head
        self.head_dim = d_model // n_head

        # 独立投影（无 bias，类似 Llama）
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        B, L, D = x.shape

        q = self.q_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_head, self.head_dim).transpose(1, 2)

        # RoPE
        q, k = apply_rotary_emb(q, k, cos, sin)

        # Flash-Attention / 标准 causal attention (PyTorch 2.0+)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
        )

        # 合并头
        out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


# ======================================================================
# SwiGLU FFN
# ======================================================================

class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈网络。

    公式: output = down_proj(silu(gate_proj(x)) * up_proj(x))
    三个独立投影，相比标准 FFN 多一个 gate_proj，无 bias。
    """

    def __init__(self, d_model: int, ffn_hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, ffn_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, ffn_hidden, bias=False)
        self.down_proj = nn.Linear(ffn_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ======================================================================
# Transformer Block (Pre-Norm)
# ======================================================================

class TransformerBlock(nn.Module):
    """单层 Transformer Block: Attention → Residual → FFN → Residual (Pre-Norm)。"""

    def __init__(self, layer_id: int, d_model: int, n_head: int, ffn_hidden: int):
        super().__init__()
        self.layer_id = layer_id

        # Attention 模块
        self.input_norm = RMSNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_head)

        # FFN 模块
        self.post_attn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_hidden)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        # Pre-Norm Attention + Residual
        x = x + self.attention(self.input_norm(x), cos, sin)
        # Pre-Norm FFN + Residual
        x = x + self.ffn(self.post_attn_norm(x))
        return x


# ======================================================================
# GPT 模型
# ======================================================================

class GPT(nn.Module):
    """
    Decoder-Only Transformer。

    配置（默认 ~10.4M 参数）:
        vocab_size=8000, d_model=256, n_layer=6, n_head=8,
        ffn_hidden=1024, max_seq_len=512
    """

    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        ffn_hidden: int = 1024,
        max_seq_len: int = 512,
        rope_theta: float = 10000.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.ffn_hidden = ffn_hidden
        self.max_seq_len = max_seq_len

        # Token Embedding（不与输出头共享权重）
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(i, d_model, n_head, ffn_hidden)
            for i in range(n_layer)
        ])

        # Final Norm
        self.norm = RMSNorm(d_model)

        # Output Head（不共享）
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        # 预计算 RoPE 频率（作为 buffer，不参与梯度）
        cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len, rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """权重初始化。"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L)  token IDs

        Returns:
            logits: (B, L, vocab_size)  每个位置的 logits
        """
        B, L = input_ids.shape
        assert L <= self.max_seq_len, \
            f"序列长度 {L} 超过 max_seq_len {self.max_seq_len}"

        # Token Embedding
        h = self.token_embedding(input_ids)  # (B, L, D)

        # 逐层前向
        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin)

        # Final Norm
        h = self.norm(h)

        # Output Projection
        logits = self.output_head(h)  # (B, L, V)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        自回归生成。

        Args:
            input_ids: (1, L0)  初始 token 序列
            max_new_tokens: 生成的最大新 token 数
            temperature: 采样温度（1.0 = 标准 softmax，<1.0 更确定）
            top_k: top-k 截断采样（None = 不截断）

        Returns:
            (1, L0 + max_new_tokens)  完整序列
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 若超出 context_length，取最后 max_seq_len 个 token
            if input_ids.size(1) > self.max_seq_len:
                ctx = input_ids[:, -self.max_seq_len:]
            else:
                ctx = input_ids

            logits = self(ctx)  # (1, L, V)
            next_logits = logits[0, -1, :] / temperature  # (V,)

            # Top-K 过滤
            if top_k is not None:
                topk_vals, topk_idx = torch.topk(next_logits, top_k)
                next_logits[next_logits < topk_vals[-1]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)  # (1, 1)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids


# ======================================================================
# 参数量统计
# ======================================================================

def print_param_count(model: GPT):
    """打印详细的参数量统计。"""
    total = 0
    print()
    print("=" * 52)
    print("  参数量统计 (Parameter Count)")
    print("=" * 52)

    # --- Embedding ---
    emb = sum(p.numel() for n, p in model.token_embedding.named_parameters())
    total += emb
    print(f"  Embedding")
    print(f"    token_embedding.weight:     {emb:>10,}")
    print(f"  {'─' * 40}")
    print()

    # --- 逐层 ---
    all_attn = 0
    all_ffn = 0
    all_norm = 0
    for i, layer in enumerate(model.layers):
        attn = sum(p.numel() for _, p in layer.attention.named_parameters())
        ffn = sum(p.numel() for _, p in layer.ffn.named_parameters())
        norm = sum(p.numel() for _, p in layer.input_norm.named_parameters())
        norm += sum(p.numel() for _, p in layer.post_attn_norm.named_parameters())
        all_attn += attn
        all_ffn += ffn
        all_norm += norm

        if i == 0:  # 只打印第一层的明细
            print(f"  Per Layer (×{len(model.layers)}):")
            print(f"    q_proj.weight:     {model.layers[0].attention.q_proj.weight.numel():>10,}")
            print(f"    k_proj.weight:     {model.layers[0].attention.k_proj.weight.numel():>10,}")
            print(f"    v_proj.weight:     {model.layers[0].attention.v_proj.weight.numel():>10,}")
            print(f"    o_proj.weight:     {model.layers[0].attention.o_proj.weight.numel():>10,}")
            print(f"    ─ Attention Subtotal:  {attn:>10,}")
            print(f"    gate_proj.weight:  {model.layers[0].ffn.gate_proj.weight.numel():>10,}")
            print(f"    up_proj.weight:    {model.layers[0].ffn.up_proj.weight.numel():>10,}")
            print(f"    down_proj.weight:  {model.layers[0].ffn.down_proj.weight.numel():>10,}")
            print(f"    ─ FFN (SwiGLU) Subtotal:  {ffn:>10,}")
            print(f"    input_norm.weight: {model.layers[0].input_norm.weight.numel():>10,}")
            print(f"    post_attn_norm.weight: {model.layers[0].post_attn_norm.weight.numel():>10,}")
            print(f"    ─ RMSNorm Subtotal:     {norm:>10,}")
            print(f"  {'─' * 40}")
            print(f"  Layer Total:         {attn + ffn + norm:>10,}")

    layers_total = all_attn + all_ffn + all_norm
    total += layers_total
    print(f"  ×{len(model.layers)} layers:         {layers_total:>10,}")
    print()

    # --- Output Head ---
    head = sum(p.numel() for _, p in model.output_head.named_parameters())
    total += head
    print(f"  Output Head (no tying)")
    print(f"    output_head.weight: {head:>10,}")
    print()

    # --- Final Norm ---
    fn = sum(p.numel() for _, p in model.norm.named_parameters())
    total += fn
    print(f"  Final Norm")
    print(f"    norm.weight:        {fn:>10,}")
    print()

    # === Total ===
    print(f"  {'=' * 40}")
    print(f"  TOTAL PARAMETERS:    {total:>10,}")
    print(f"  {'=' * 40}")
    print()

    return total


# ======================================================================
# 快速测试
# ======================================================================

def _test_model():
    """创建模型并测试前向 + 参数量。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = GPT(
        vocab_size=8000,
        d_model=256,
        n_layer=6,
        n_head=8,
        ffn_hidden=1024,
        max_seq_len=512,
    ).to(device)

    print_param_count(model)

    # 测试前向
    x = torch.randint(0, 100, (2, 64), device=device)
    with torch.no_grad():
        logits = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {logits.shape}")
    print(f"  Output range: [{logits.min():.2f}, {logits.max():.2f}]")

    # 测试生成
    x0 = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(x0, max_new_tokens=10, temperature=0.8, top_k=10)
    print(f"  Generate: {x0.shape} → {out.shape}")
    print()

    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:>10,}")
    print(f"  Trainable params: {trainable_params:>10,}")
    print()

    return model


if __name__ == "__main__":
    _test_model()
