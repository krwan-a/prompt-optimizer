#!/usr/bin/env python3
"""
预训练语料收集模块 —— 从多个来源收集领域文本，输出为纯文本行（每行一篇文档）。

支持的数据源：
  - huggingface : 从 HuggingFace Datasets 加载
  - local       : 读取本地 .txt 文件
  - web         : 从 URL 列表抓取网页正文

用法示例：
    # 从 HuggingFace 加载英文 wiki + 中文百科
    python data/collect_pretrain.py \\
        --source huggingface --dataset wikipedia --split train --text-field text --max-samples 50000 \\
        --output data/raw_corpus.txt

    # 从本地文件读取
    python data/collect_pretrain.py \\
        --source local --input-dir ./my_texts/ --extension .txt \\
        --output data/raw_corpus.txt

    # 从网页抓取
    python data/collect_pretrain.py \\
        --source web --urls-file urls.txt --max-pages 200 \\
        --output data/raw_corpus.txt
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional, List

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Source: HuggingFace Datasets
# ---------------------------------------------------------------------------

def collect_huggingface(
    dataset_name: str,
    split: str = "train",
    text_field: str = "text",
    max_samples: Optional[int] = None,
) -> List[str]:
    """从 HuggingFace 加载文本数据集。"""
    from datasets import load_dataset

    print(f"  Loading dataset '{dataset_name}' (split={split})...")
    ds = load_dataset(dataset_name, split=split, streaming=True)

    docs: List[str] = []
    for i, example in enumerate(tqdm(ds, desc=f"  Reading {dataset_name}")):
        text = example.get(text_field, "")
        if text and isinstance(text, str) and text.strip():
            docs.append(text.strip())
        if max_samples and len(docs) >= max_samples:
            break

    print(f"  Collected {len(docs):,} documents from {dataset_name}")
    return docs


# ---------------------------------------------------------------------------
# Source: Local files
# ---------------------------------------------------------------------------

def collect_local(
    input_dir: Path,
    extension: str = ".txt",
    max_samples: Optional[int] = None,
) -> List[str]:
    """读取本地文本文件。"""
    files = sorted(input_dir.rglob(f"*{extension}"))
    if not files:
        print(f"  [Warn] 在 {input_dir} 下未找到 *{extension} 文件")
        return []

    print(f"  Found {len(files):,} files")
    random.shuffle(files)

    docs: List[str] = []
    for fpath in tqdm(files, desc="  Reading files"):
        try:
            text = fpath.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append(text)
        except Exception as e:
            print(f"  [Warn] 跳过 {fpath}: {e}")
        if max_samples and len(docs) >= max_samples:
            break

    print(f"  Collected {len(docs):,} documents from {input_dir}")
    return docs


# ---------------------------------------------------------------------------
# Source: Web scraping
# ---------------------------------------------------------------------------

def extract_text_from_url(url: str, timeout: int = 15) -> Optional[str]:
    """从 URL 提取正文文本。"""
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        # 检测编码
        if resp.encoding and resp.encoding.lower() != "utf-8":
            resp.encoding = resp.apparent_encoding or "utf-8"

        soup = BeautifulSoup(resp.text, "html.parser")
        # 移除无用标签
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "noscript", "iframe", "form"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        return text if len(text) > 100 else None
    except Exception as e:
        print(f"  [Warn] 抓取失败 {url}: {e}")
        return None


def collect_web(
    urls_file: Path,
    max_pages: int = 200,
    delay: float = 1.0,
) -> List[str]:
    """从 URL 列表抓取网页正文。"""
    urls = [
        line.strip() for line in urls_file.open(encoding="utf-8")
        if line.strip() and not line.startswith("#")
    ]
    if not urls:
        print(f"  [Warn] {urls_file} 中没有有效 URL")
        return []

    print(f"  Loaded {len(urls)} URLs, will scrape up to {max_pages}")
    random.shuffle(urls)

    docs: List[str] = []
    for url in tqdm(urls[:max_pages], desc="  Scraping"):
        text = extract_text_from_url(url)
        if text:
            docs.append(text)
        time.sleep(delay)  # 礼貌延迟

    print(f"  Collected {len(docs):,} documents from web scraping")
    return docs


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="收集预训练语料，输出为一行一篇文档的纯文本文件"
    )
    parser.add_argument("--output", "-o", default="data/raw_corpus.txt",
                        help="输出文件路径 (default: data/raw_corpus.txt)")

    # Source 选择
    parser.add_argument("--source", choices=["huggingface", "local", "web"],
                        default="huggingface", help="数据源")

    # HuggingFace 参数
    parser.add_argument("--dataset", default="wikitext",
                        help="HuggingFace 数据集名 (--source huggingface 时使用)")
    parser.add_argument("--split", default="train", help="数据集 split")
    parser.add_argument("--text-field", default="text",
                        help="文本字段名 (如 'text', 'content')")

    # 本地文件参数
    parser.add_argument("--input-dir", type=Path,
                        help="输入目录 (--source local 时使用)")
    parser.add_argument("--extension", default=".txt",
                        help="文件扩展名 (default: .txt)")

    # Web 参数
    parser.add_argument("--urls-file", type=Path,
                        help="URL 列表文件 (--source web 时使用，每行一个 URL)")
    parser.add_argument("--max-pages", type=int, default=200,
                        help="最大抓取页数 (default: 200)")

    # 公共参数
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大文档数限制")
    parser.add_argument("--shuffle", action="store_true", default=True,
                        help="打乱输出顺序")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()
    random.seed(args.seed)

    # 收集
    print(f"Data source: {args.source}")
    if args.source == "huggingface":
        docs = collect_huggingface(
            dataset_name=args.dataset,
            split=args.split,
            text_field=args.text_field,
            max_samples=args.max_samples,
        )
    elif args.source == "local":
        if not args.input_dir:
            print("[Error] --source local 需要 --input-dir")
            exit(1)
        docs = collect_local(
            input_dir=args.input_dir,
            extension=args.extension,
            max_samples=args.max_samples,
        )
    elif args.source == "web":
        if not args.urls_file:
            print("[Error] --source web 需要 --urls-file")
            exit(1)
        docs = collect_web(
            urls_file=args.urls_file,
            max_pages=args.max_pages,
        )
    else:
        docs = []

    if not docs:
        print("[Error] 未收集到任何文档，退出。")
        exit(1)

    # 打乱
    if args.shuffle:
        random.shuffle(docs)

    # 写出
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            # 将多行文本压缩为单行（替换换行为空格）
            line = doc.replace("\n", " ").replace("\r", " ").strip()
            f.write(line + "\n")

    print(f"\nSaved {len(docs):,} documents to {output_path}")
    # 简单统计
    total_chars = sum(len(d) for d in docs)
    print(f"Total characters: {total_chars:,}")


if __name__ == "__main__":
    main()
