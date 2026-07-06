#!/usr/bin/env python3
"""
数据集类 —— 将清洗后的文本 / SFT 数据转换为定长 (max_length=512) 训练样本。

预训练数据 (PretrainDataset):
  对整个语料做滑窗切分，每个样本为固定长度的连续 token 序列。
  用于 autoregressive language modeling (labels = input_ids)。

SFT 数据 (SFTDataset):
  将 (rough_input, refined_prompt) 格式化为：
    "[INST] {rough_input} [/INST]\\n{refined_prompt}"
  通过 character offset 定位 response 起始位置，loss 仅在 response 部分计算（输入部分 mask 为 -100）。

DataLoader 工厂:
  create_dataloader() 自动选择 pretrain / SFT 的 collate 函数。

用法示例：
    from tokenizers import Tokenizer
    from train.dataset import PretrainDataset, SFTDataset, create_dataloader

    tokenizer = Tokenizer.from_file("tokenizer/prompt_opt_tokenizer.json")

    # 预训练
    ds = PretrainDataset("data/clean_corpus.txt", tokenizer, max_length=512)
    loader = create_dataloader(ds, batch_size=8, shuffle=True)

    # SFT
    ds = SFTDataset("data/sft_data.jsonl", tokenizer, max_length=512)
    loader = create_dataloader(ds, batch_size=8, shuffle=True,
                               pad_token_id=0, is_sft=True)
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# PretrainDataset —— 滑窗切分
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    """
    预训练数据集：将整个语料 tokenize 后做滑窗切分。

    每个样本返回:
        input_ids: [max_length]  — 模型输入
        labels:    [max_length]  — 与 input_ids 相同（autoregressive LM）
    """

    def __init__(
        self,
        tokenizer,
        corpus_path: Union[str, Path],
        max_length: int = 512,
        stride: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Args:
            tokenizer: 已训练的 tokenizer（需有 .encode() 和 .decode() 方法）
            corpus_path: 语料文件（纯文本，UTF-8）
            max_length: 每个样本的最大 token 数
            stride: 滑窗步长 (default: max_length // 2，即 50% 重叠)
            verbose: 是否打印统计信息
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride if stride is not None else max_length // 2

        # 读取
        if verbose:
            print(f"[PretrainDataset] 加载语料: {corpus_path}")
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()

        if verbose:
            print(f"[PretrainDataset] 字符数: {len(text):,}")

        # Tokenize 整个语料
        encoding = tokenizer.encode(text)
        self.tokens = encoding.ids
        if verbose:
            print(f"[PretrainDataset] Token 数: {len(self.tokens):,}")

        # 滑窗切分
        self.chunks = []
        for i in range(0, len(self.tokens) - max_length + 1, self.stride):
            self.chunks.append(self.tokens[i:i + max_length])

        if verbose:
            print(f"[PretrainDataset] 训练样本数: {len(self.chunks):,} "
                  f"(max_len={max_length}, stride={self.stride})")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.chunks[idx]
        return {
            "input_ids": torch.tensor(chunk, dtype=torch.long),
            "labels": torch.tensor(chunk, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# SFTDataset —— 带 loss masking 的指令微调数据
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """
    SFT 数据集：将 (rough_input, refined_prompt) 用模板拼接后 tokenize，
    通过 character offset 定位 response 起始 token，loss 仅施加在 response 部分。

    每个样本返回:
        input_ids: [...]  — 模型输入
        labels:    [...]  — 非 response 位置为 -100（忽略 loss）
    """

    INST_TOKEN = "[INST]"
    END_INST_TOKEN = "[/INST]"

    def __init__(
        self,
        tokenizer,
        data_path: Union[str, Path],
        max_length: int = 512,
        verbose: bool = True,
    ):
        """
        Args:
            tokenizer: 已训练的 tokenizer
            data_path: JSONL 文件路径（每行 {"rough_input": ..., "refined_prompt": ...}）
            max_length: 最大序列长度（从 input 侧截断以尽量保留 response）
            verbose: 是否打印统计信息
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        if verbose:
            print(f"[SFTDataset] 加载 SFT 数据: {data_path}")

        self.examples = self._load_and_process(data_path, verbose)

        if verbose:
            print(f"[SFTDataset] SFT 样本数: {len(self.examples):,}")
            # 打印一个示例
            if self.examples:
                ex = self.examples[0]
                inp_len = len(ex["input_ids"])
                lab_resp = sum(1 for l in ex["labels"] if l != -100)
                print(f"[SFTDataset] 示例: input_len={inp_len}, response_tokens={lab_resp}")

    def _load_and_process(
        self, data_path: Union[str, Path], verbose: bool
    ) -> List[Dict]:
        """加载 JSONL 并逐条处理。"""
        examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        iterator = tqdm(lines, desc="  Processing") if verbose else lines
        for line in iterator:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                rough = item.get("rough_input", "")
                refined = item.get("refined_prompt", "")
                if rough and refined:
                    processed = self._process_example(rough, refined)
                    if processed is not None:
                        examples.append(processed)
            except json.JSONDecodeError:
                if verbose:
                    print(f"  [Warn] JSON 解析失败，跳过: {line[:80]}")

        return examples

    def _process_example(
        self, rough_input: str, refined_prompt: str
    ) -> Optional[Dict]:
        """
        处理单个 (rough, refined) 对。

        步骤：
          1. 构造完整文本: "[INST] {rough} [/INST]\\n{refined}"
          2. Tokenize 并利用 character offset 确定 response 起始 token
          3. 超长时从 input 侧截断
          4. labels: input 位置 = -100（不计算 loss）
        """
        # 构造文本
        prefix = f"{self.INST_TOKEN} {rough_input} {self.END_INST_TOKEN}\n"
        full_text = prefix + refined_prompt

        # Tokenize
        encoding = self.tokenizer.encode(full_text)
        input_ids = encoding.ids

        # 用 character offset 定位 response 起始 token 位置
        prefix_end = len(prefix)  # refined_prompt 开始的字符位置
        response_start = len(input_ids)  # fallback
        for i, (start, end) in enumerate(encoding.offsets):
            if start >= prefix_end:
                response_start = i
                break

        # 超长处理：从 input 侧截断，尽量保留 response
        if len(input_ids) > self.max_length:
            excess = len(input_ids) - self.max_length
            truncate_from = min(excess, response_start)
            input_ids = input_ids[truncate_from:]
            response_start = max(0, response_start - truncate_from)
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                response_start = min(response_start, self.max_length)

        # labels: -100 表示该位置不参与 loss 计算
        labels = [-100] * response_start + input_ids[response_start:]

        return {"input_ids": input_ids, "labels": labels}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.examples[idx]


# ---------------------------------------------------------------------------
# Collate 函数
# ---------------------------------------------------------------------------

def pretrain_collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """预训练 collate：所有序列等长，直接 stack。"""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def sft_collate(
    batch: List[Dict], pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """SFT collate：padding 至 batch 内最大长度，生成 attention_mask。"""
    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]

    max_len = max(len(ids) for ids in input_ids)

    padded_input_ids, attention_masks, padded_labels = [], [], []
    for ids, lbls in zip(input_ids, labels):
        pad = max_len - len(ids)
        padded_input_ids.append(ids + [pad_token_id] * pad)
        attention_masks.append([1] * len(ids) + [0] * pad)
        padded_labels.append(lbls + [-100] * pad)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# DataLoader 工厂
# ---------------------------------------------------------------------------

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    pad_token_id: int = 0,
    is_sft: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """
    创建 DataLoader，自动选择 collate 函数。

    Args:
        dataset: PretrainDataset 或 SFTDataset
        batch_size: 批次大小
        shuffle: 是否打乱
        pad_token_id: padding token ID（仅 SFT 需要）
        is_sft: 是否为 SFT 数据
        num_workers: 数据加载子进程数
    """
    collate_fn = sft_collate_factory(pad_token_id) if is_sft else pretrain_collate
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def sft_collate_factory(pad_token_id: int):
    """返回绑定了 pad_token_id 的 sft_collate。"""
    def collate(batch):
        return sft_collate(batch, pad_token_id)
    return collate


# ---------------------------------------------------------------------------
# SFTChatDataset —— 使用 <user>/<assistant> Chat 模板 + Loss Masking
# ---------------------------------------------------------------------------

class SFTChatDataset(Dataset):
    """
    SFT 数据集（Chat 模板版）。

    格式: "<user>{rough_input}</user><assistant>{refined_prompt}</assistant>[EOS]"

    Loss 仅在 <assistant> 标签之后的 token 位置计算，
    <user> 部分和 <assistant> 标签本身的 label = -100。
    """

    def __init__(
        self,
        tokenizer,
        data_path: Union[str, Path],
        max_length: int = 512,
        verbose: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_id = tokenizer.token_to_id("[EOS]") or 3

        if verbose:
            print(f"[SFTChatDataset] 加载 SFT 数据: {data_path}")

        self.examples = self._load_and_process(data_path, verbose)

        if verbose:
            print(f"[SFTChatDataset] 样本数: {len(self.examples):,}")
            if self.examples:
                ex = self.examples[0]
                inp_len = len(ex["input_ids"])
                lab_resp = sum(1 for l in ex["labels"] if l != -100)
                print(f"[SFTChatDataset] 示例: input_len={inp_len}, response_tokens={lab_resp}")

    def _load_and_process(self, data_path: Union[str, Path], verbose: bool) -> List[Dict]:
        """加载 JSONL 并逐条处理。"""
        examples = []
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        iterator = tqdm(lines, desc="  Processing") if verbose else lines
        for line in iterator:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                rough = item.get("rough_input", "")
                refined = item.get("refined_prompt", "")
                if rough and refined:
                    processed = self._process_example(rough, refined)
                    if processed is not None:
                        examples.append(processed)
            except json.JSONDecodeError:
                if verbose:
                    print(f"  [Warn] JSON 解析失败，跳过: {line[:80]}")
        return examples

    def _process_example(self, rough_input: str, refined_prompt: str) -> Optional[Dict]:
        """
        处理单个 (rough, refined) 对。

        Loss masking: 通过 character offset 定位 <assistant> 标签后的起始 token，
        之前的所有 token（<user> 部分、<assistant> 标签等）label = -100。
        """
        # 1) 构造带 [EOS] 的完整文本
        full_text = f"<user>{rough_input}</user><assistant>{refined_prompt}</assistant>"

        # 2) Tokenize
        encoding = self.tokenizer.encode(full_text)
        input_ids = encoding.ids

        # 3) 用 character offset 找到 <assistant> 结束位置（response 起始）
        marker = "<assistant>"
        resp_char_start = full_text.find(marker) + len(marker)
        response_start = len(input_ids)  # fallback
        for i, (s, e) in enumerate(encoding.offsets):
            if s >= resp_char_start:
                response_start = i
                break

        # 4) 超长处理：从 input 侧截断，尽量保留 response
        if len(input_ids) > self.max_length:
            excess = len(input_ids) - self.max_length
            truncate_from = min(excess, response_start)
            input_ids = input_ids[truncate_from:]
            response_start = max(0, response_start - truncate_from)
            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                response_start = min(response_start, self.max_length)

        # 5) labels: 只有 <assistant> 之后的 token 才参与 loss
        labels = [-100] * response_start + input_ids[response_start:]

        return {"input_ids": input_ids, "labels": labels}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.examples[idx]


# ---------------------------------------------------------------------------
# 简单测试
# ---------------------------------------------------------------------------

def _test_pretrain(tokenizer_path: str, corpus_path: str):
    """测试 PretrainDataset + DataLoader。"""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    ds = PretrainDataset(tokenizer, corpus_path, max_length=128, verbose=True)
    loader = create_dataloader(ds, batch_size=4, shuffle=False, is_sft=False)
    batch = next(iter(loader))
    print(f"\n  Pretrain batch: input_ids {batch['input_ids'].shape}, "
          f"labels {batch['labels'].shape}")
    return ds, loader


def _test_sft(tokenizer_path: str, data_path: str):
    """测试 SFTDataset + DataLoader。"""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tokenizer_path)
    pad_id = tokenizer.token_to_id("[PAD]") if tokenizer.token_to_id("[PAD]") is not None else 0

    ds = SFTDataset(tokenizer, data_path, max_length=128, verbose=True)
    loader = create_dataloader(ds, batch_size=4, shuffle=False,
                               pad_token_id=pad_id, is_sft=True)
    batch = next(iter(loader))
    print(f"\n  SFT batch: input_ids {batch['input_ids'].shape}, "
          f"attention_mask {batch['attention_mask'].shape}, "
          f"labels {batch['labels'].shape}")
    print(f"  Response tokens (non -100) in first sample: "
          f"{(batch['labels'][0] != -100).sum().item()}")

    return ds, loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据集类测试")
    parser.add_argument("--mode", choices=["pretrain", "sft"], required=True,
                        help="测试模式")
    parser.add_argument("--tokenizer", required=True,
                        help="tokenizer JSON 路径")
    parser.add_argument("--data", required=True,
                        help="数据路径（pretrain: 文本文件 / sft: JSONL）")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)

    args = parser.parse_args()

    if args.mode == "pretrain":
        _test_pretrain(args.tokenizer, args.data)
    else:
        _test_sft(args.tokenizer, args.data)

    print("\n✓ 测试通过！")
