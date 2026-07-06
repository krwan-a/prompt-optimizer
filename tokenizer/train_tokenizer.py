#!/usr/bin/env python3
"""
BPE 分词器训练脚本 —— 基于收集到的中英混合语料训练 BPE tokenizer。

使用 HuggingFace `tokenizers` 库，支持 ByteLevel BPE。
vocab_size=8000，中英混合，自动处理 Unicode。

用法示例：
    # 从清洗后的语料训练
    python tokenizer/train_tokenizer.py \\
        --corpus-path data/clean_corpus.txt \\
        --output tokenizer/prompt_opt_tokenizer.json \\
        --vocab-size 8000

    # 从多个文件训练
    python tokenizer/train_tokenizer.py \\
        --corpus-path data/corpus1.txt data/corpus2.txt \\
        --output tokenizer/prompt_opt_tokenizer.json
"""

import argparse
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
from tokenizers.normalizers import NFC


# ---------------------------------------------------------------------------
# 特殊 Token 定义
# ---------------------------------------------------------------------------

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

# 各 token 的 ID（按上述顺序）
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# 语料读取
# ---------------------------------------------------------------------------

def corpus_lines(corpus_paths: List[Path]) -> List[str]:
    """读取语料文件，每行为一个 text unit（用于 tokenizer 训练）。"""
    all_lines: List[str] = []
    for path in corpus_paths:
        if path.is_dir():
            files = sorted(path.rglob("*.txt"))
        elif path.is_file():
            files = [path]
        else:
            print(f"  [Warn] 跳过: {path}")
            continue

        for fpath in files:
            try:
                with fpath.open("r", encoding="utf-8", errors="ignore") as f:
                    all_lines.extend(line.strip() for line in f if line.strip())
            except Exception as e:
                print(f"  [Warn] 读取失败 {fpath}: {e}")

    return all_lines


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def train_tokenizer(
    corpus_paths: List[Path],
    vocab_size: int = 8000,
    output_path: Path = Path("tokenizer/tokenizer.json"),
    special_tokens: Optional[List[str]] = None,
    min_frequency: int = 2,
):
    """
    训练 ByteLevel BPE tokenizer。

    Args:
        corpus_paths: 语料文件或目录列表
        vocab_size: 词表大小 (default: 8000)
        output_path: 输出路径
        special_tokens: 特殊 token 列表
        min_frequency: 最小出现频率
    """
    if special_tokens is None:
        special_tokens = SPECIAL_TOKENS

    # 1. 读取语料
    print("=" * 50)
    print("  读取语料...")
    print("=" * 50)
    lines = corpus_lines(corpus_paths)
    total_chars = sum(len(l) for l in lines)
    print(f"  文件数: {len(corpus_paths):,}")
    print(f"  文本行数: {len(lines):,}")
    print(f"  总字符数: {total_chars:,}")

    if not lines:
        print("[Error] 未读取到任何文本，请检查 --corpus-path")
        exit(1)

    # 2. 初始化 tokenizer
    print("\n" + "=" * 50)
    print("  初始化 ByteLevel BPE Tokenizer...")
    print("=" * 50)

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # Normalizer: NFC Unicode 标准化
    tokenizer.normalizer = normalizers.Sequence([NFC()])

    # Pre-tokenizer: ByteLevel（处理所有语言的统一方式，GPT-2 等模型采用）
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder: ByteLevel
    tokenizer.decoder = decoders.ByteLevel()

    # 3. 训练
    print(f"\n  训练 BPE tokenizer (vocab_size={vocab_size}, min_freq={min_frequency})...")

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        show_progress=True,
    )

    tokenizer.train_from_iterator(lines, trainer)
    print(f"  训练完成！词表大小: {tokenizer.get_vocab_size()}")

    # 4. 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    print(f"\n  已保存: {output_path}")

    # 5. 测试
    print("\n" + "=" * 50)
    print("  Tokenizer 测试")
    print("=" * 50)
    test_texts = [
        "你好，世界！Hello, World!",
        "请用Python编写一个数据分析脚本，支持读取CSV和Excel文件。",
        "Write a detailed, well-structured prompt for code generation tasks.",
        "中英文混合的文本Chinese and English mixed text适合用来测试tokenizer效果。",
        "[INST] 写个Python脚本读CSV [/INST]\n请用pandas编写一个CSV读取脚本。",
    ]

    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids)
        tokens = encoded.tokens

        print(f"\n  Input  : {text}")
        print(f"  Tokens ({len(encoded.ids)}):")
        # 分行显示 token，更易读
        for i, (tok, tid) in enumerate(zip(tokens, encoded.ids)):
            print(f"    [{i:3d}] ID={tid:4d}  token=「{tok}」")
        print(f"  Decoded: {decoded}")
        print(f"  Match  : {'✓' if decoded == text else '✗'}")

    # 6. 词表统计
    print("\n" + "=" * 50)
    print("  词表统计")
    print("=" * 50)
    vocab = tokenizer.get_vocab()
    special = {k: v for k, v in vocab.items() if k.startswith("[")}
    chinese = {k: v for k, v in vocab.items() if any("一" <= c <= "鿿" for c in k)}
    byte_tokens = {k: v for k, v in vocab.items() if k.startswith("Ġ")}
    print(f"  总词表: {len(vocab)}")
    print(f"  特殊 token: {len(special)} ({special})")
    print(f"  中文相关 token: {len(chinese)}")
    print(f"  ByteLevel 空格前缀 token (Ġ): {len(byte_tokens)}")

    return tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="训练 BPE tokenizer（中英混合，ByteLevel）"
    )
    parser.add_argument(
        "--corpus-path", "-c", nargs="+", required=True,
        help="语料路径，可指定文件或目录（多个用空格分隔）",
    )
    parser.add_argument(
        "--output", "-o", default="tokenizer/prompt_opt_tokenizer.json",
        help="输出路径 (default: tokenizer/prompt_opt_tokenizer.json)",
    )
    parser.add_argument(
        "--vocab-size", "-v", type=int, default=8000,
        help="词表大小 (default: 8000)",
    )
    parser.add_argument(
        "--min-frequency", type=int, default=2,
        help="最小出现频率 (default: 2)",
    )
    parser.add_argument(
        "--special-tokens", nargs="+",
        default=SPECIAL_TOKENS,
        help="特殊 token 列表",
    )

    args = parser.parse_args()

    corpus_paths = [Path(p) for p in args.corpus_path]

    train_tokenizer(
        corpus_paths=corpus_paths,
        vocab_size=args.vocab_size,
        output_path=Path(args.output),
        special_tokens=args.special_tokens,
        min_frequency=args.min_frequency,
    )
