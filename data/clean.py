#!/usr/bin/env python3
"""
文本清洗与去重模块 —— 清洗收集到的预训练语料。

清洗管线：
  1. Unicode 标准化 (NFKC)
  2. 移除/替换特殊空白字符
  3. 可选：剥离 HTML 标签
  4. 可选：URL / Email 移除
  5. 长度过滤（min / max chars）
  6. 重复率过滤（去重后 unique/total 比例）
  7. 精确去重（MD5 hash）
  8. 可选：MinHash 近似去重

用法示例：
    python data/clean.py --input data/raw_corpus.txt --output data/clean_corpus.txt \\
        --min-length 100 --max-length 50000 --dedup
"""

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import List, Set, Optional

from tqdm import tqdm


# ---------------------------------------------------------------------------
# 清洗函数
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """NFKC Unicode 标准化 + 特殊空白处理。"""
    text = unicodedata.normalize("NFKC", text)
    # 替换特殊空白字符
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f​-‏ - ﻿]", " ", text)
    return text


def normalize_whitespace(text: str) -> str:
    """合并连续空白为单个空格，去除首尾空白。"""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_html(text: str) -> str:
    """剥离 HTML 标签及常见实体。"""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"&#\d+;", " ", text)
    return text


def strip_urls(text: str) -> str:
    """移除 URL。"""
    return re.sub(r"https?://\S+", "", text)


def strip_emails(text: str) -> str:
    """移除 Email 地址。"""
    return re.sub(r"\S+@\S+\.\S+", "", text)


# ---------------------------------------------------------------------------
# 过滤器
# ---------------------------------------------------------------------------

def filter_by_length(text: str, min_chars: int = 50, max_chars: int = 100_000) -> bool:
    """按字符长度过滤。"""
    return min_chars <= len(text) <= max_chars


def filter_by_repetition(text: str, max_dup_ratio: float = 0.6) -> bool:
    """按行级重复率过滤：去重后行数 / 总行数 < max_dup_ratio 则过滤。"""
    lines = text.split("\n")
    if len(lines) < 3:
        return True  # 太短的文本跳过此检查
    unique_ratio = len(set(lines)) / len(lines)
    return unique_ratio >= (1 - max_dup_ratio)


def filter_by_char_repetition(text: str, max_repeat: int = 80) -> bool:
    """过滤包含过长连续重复字符的文本。"""
    # 检查连续相同字符
    repeat_pattern = re.compile(r"(.)\1{%d,}" % (max_repeat - 1))
    return not bool(repeat_pattern.search(text))


# ---------------------------------------------------------------------------
# 去重
# ---------------------------------------------------------------------------

def exact_dedup(docs: List[str]) -> List[str]:
    """基于 MD5 的精确去重，保留顺序。"""
    seen: Set[str] = set()
    result: List[str] = []
    for doc in tqdm(docs, desc="  Exact dedup"):
        sig = hashlib.md5(doc.encode("utf-8")).hexdigest()
        if sig not in seen:
            seen.add(sig)
            result.append(doc)
    return result


def simple_near_dedup(docs: List[str], threshold: float = 0.85, n: int = 8) -> List[str]:
    """
    基于 n-gram Jaccard 相似度的近似去重（O(n²) 警告 —— 仅适合小规模语料）。
    若文档与任何已保留文档的 Jaccard 相似度 > threshold，则丢弃。
    """
    def ngrams(text: str, n: int) -> Set[str]:
        chars = text.replace(" ", "")
        return {chars[i:i + n] for i in range(len(chars) - n + 1)}

    def jaccard(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    print(f"  Near-dedup (threshold={threshold}, ngram={n}) — O(n²), only suitable for small corpus...")
    kept: List[str] = []
    kept_ngrams: List[Set[str]] = []

    for doc in tqdm(docs, desc="  Near dedup"):
        ng = ngrams(doc[:2000], n)  # 取前 2000 字符算 ngram
        too_similar = False
        for existing in kept_ngrams:
            if jaccard(ng, existing) > threshold:
                too_similar = True
                break
        if not too_similar:
            kept.append(doc)
            kept_ngrams.append(ng)

    return kept


# ---------------------------------------------------------------------------
# 管线
# ---------------------------------------------------------------------------

def clean_corpus(
    input_path: Path,
    output_path: Path,
    min_length: int = 50,
    max_length: int = 100_000,
    strip_html_tags: bool = True,
    strip_urls_flag: bool = True,
    do_dedup: bool = True,
    near_dedup: bool = False,
    near_dedup_threshold: float = 0.85,
):
    """完整的清洗管线。"""
    # 读取
    print(f"Reading: {input_path}")
    docs = [
        line.strip()
        for line in input_path.open("r", encoding="utf-8")
        if line.strip()
    ]
    print(f"  Input: {len(docs):,} documents")

    # 逐条清洗
    cleaned: List[str] = []
    for doc in tqdm(docs, desc="  Cleaning"):
        text = doc
        if strip_html_tags:
            text = strip_html(text)
        if strip_urls_flag:
            text = strip_urls(text)
        text = normalize_unicode(text)
        text = normalize_whitespace(text)

        if not filter_by_length(text, min_length, max_length):
            continue
        if not filter_by_char_repetition(text):
            continue
        if not filter_by_repetition(text):
            continue

        cleaned.append(text)

    print(f"  After cleaning & filtering: {len(cleaned):,} documents")

    # 精确去重
    if do_dedup:
        cleaned = exact_dedup(cleaned)
        print(f"  After exact dedup: {len(cleaned):,} documents")

    # 近似去重
    if near_dedup and len(cleaned) < 50_000:
        cleaned = simple_near_dedup(cleaned, threshold=near_dedup_threshold)
        print(f"  After near dedup: {len(cleaned):,} documents")
    elif near_dedup:
        print(f"  [Warn] 语料过大 ({len(cleaned):,})，跳过近似去重")

    # 写出
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for doc in cleaned:
            f.write(doc + "\n")

    print(f"\nSaved: {output_path}")
    print(f"  {len(cleaned):,} documents | total chars: {sum(len(d) for d in cleaned):,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="清洗和去重预训练语料")

    parser.add_argument("--input", "-i", required=True,
                        help="输入文件（一行一篇文档）")
    parser.add_argument("--output", "-o", required=True,
                        help="输出文件路径")
    parser.add_argument("--min-length", type=int, default=100,
                        help="最小字符数 (default: 100)")
    parser.add_argument("--max-length", type=int, default=100_000,
                        help="最大字符数 (default: 100000)")
    parser.add_argument("--no-html-strip", action="store_true",
                        help="跳过 HTML 标签剥离")
    parser.add_argument("--no-url-strip", action="store_true",
                        help="跳过 URL 移除")
    parser.add_argument("--no-dedup", action="store_true",
                        help="跳过精确去重")
    parser.add_argument("--near-dedup", action="store_true",
                        help="启用近似去重（仅适合小规模语料）")
    parser.add_argument("--near-dedup-threshold", type=float, default=0.85,
                        help="近似去重阈值 (default: 0.85)")

    args = parser.parse_args()

    clean_corpus(
        input_path=Path(args.input),
        output_path=Path(args.output),
        min_length=args.min_length,
        max_length=args.max_length,
        strip_html_tags=not args.no_html_strip,
        strip_urls_flag=not args.no_url_strip,
        do_dedup=not args.no_dedup,
        near_dedup=args.near_dedup,
        near_dedup_threshold=args.near_dedup_threshold,
    )
