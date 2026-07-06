#!/usr/bin/env python3
"""
Tokenizer 扩展模块 —— 为 SFT 微调添加 Chat 特殊 Token。

在已有 BPE tokenizer 的词表中追加：
  <user> </user> <assistant> </assistant> <pad>

并提供 SFT 数据格式化函数：
  format_chat(rough_input, refined_prompt) -> "<user>...</user><assistant>...</assistant>"

用法：
    python tokenizer/extend_tokenizer.py \
        --input-tokenizer tokenizer/prompt_opt_tokenizer.json \
        --output-tokenizer tokenizer/sft_tokenizer.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from tokenizers import AddedToken, Tokenizer


# ---------------------------------------------------------------------------
# 新增的 Chat 特殊 Token
# ---------------------------------------------------------------------------

CHAT_SPECIAL_TOKENS = [
    AddedToken("<user>", single_word=True, normalized=False),
    AddedToken("</user>", single_word=True, normalized=False),
    AddedToken("<assistant>", single_word=True, normalized=False),
    AddedToken("</assistant>", single_word=True, normalized=False),
    AddedToken("<pad>", single_word=True, normalized=False),
]

CHAT_TOKEN_NAMES = ["<user>", "</user>", "<assistant>", "</assistant>", "<pad>"]


# ---------------------------------------------------------------------------
# 格式化函数
# ---------------------------------------------------------------------------

def format_chat(rough_input: str, refined_prompt: str) -> str:
    """
    将 (rough_input, refined_prompt) 转换成 Chat 拼接文本。

    >>> format_chat("写个Python脚本", "请编写脚本...")
    '<user>写个Python脚本</user><assistant>请编写脚本...</assistant>'
    """
    return f"<user>{rough_input}</user><assistant>{refined_prompt}</assistant>"


def format_chat_prompt(rough_input: str) -> str:
    """
    仅构建 Prompt 部分（用于生成时作为初始上下文）。

    >>> format_chat_prompt("写个Python脚本")
    '<user>写个Python脚本</user><assistant>'
    """
    return f"<user>{rough_input}</user><assistant>"


def extract_assistant_response(full_text: str) -> str:
    """
    从模型生成的完整文本中提取 <assistant>...</assistant> 之间的内容。

    如果找不到闭合标签，返回 <assistant> 之后的所有文本。
    """
    marker = "<assistant>"
    start = full_text.find(marker)
    if start == -1:
        return full_text  # fallback: 返回全文
    content = full_text[start + len(marker):]

    end_marker = "</assistant>"
    end = content.find(end_marker)
    if end != -1:
        content = content[:end]

    return content.strip()


# ---------------------------------------------------------------------------
# 扩展 Tokenizer
# ---------------------------------------------------------------------------

def extend_tokenizer(
    input_path: Path,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> Tokenizer:
    """
    加载已有 tokenizer，追加 chat 特殊 token，保存并返回。
    """
    if verbose:
        print(f"[extend] 加载 tokenizer: {input_path}")

    tokenizer = Tokenizer.from_file(str(input_path))
    old_size = tokenizer.get_vocab_size()

    if verbose:
        print(f"[extend] 原始词表大小: {old_size}")

    # 检查哪些 token 已存在
    for tok_name in CHAT_TOKEN_NAMES:
        existing_id = tokenizer.token_to_id(tok_name)
        if existing_id is not None:
            if verbose:
                print(f"[extend]   token '{tok_name}' 已存在（ID={existing_id}），跳过")

    # 只添加不存在的 token
    tokens_to_add = [
        t for t in CHAT_SPECIAL_TOKENS
        if tokenizer.token_to_id(t.content) is None
    ]

    added_count = tokenizer.add_tokens(tokens_to_add)
    new_size = tokenizer.get_vocab_size()

    if verbose:
        print(f"[extend] 新增 {added_count} 个 token")
        print(f"[extend] 扩展后词表大小: {new_size}")
        for tok_name in CHAT_TOKEN_NAMES:
            tid = tokenizer.token_to_id(tok_name)
            print(f"[extend]   {tok_name:15s} → ID={tid}")

    # 测试编码
    test_text = format_chat("写个Python脚本", "请用pandas编写脚本。")
    encoded = tokenizer.encode(test_text)
    if verbose:
        print(f"\n[extend] 编码测试:")
        print(f"  输入: {test_text}")
        print(f"  Token 数: {len(encoded.ids)}")
        print(f"  Tokens: {encoded.tokens}")
        print(f"  解码: {tokenizer.decode(encoded.ids)}")
        assert tokenizer.decode(encoded.ids) == test_text, "编码-解码自洽性检查失败！"
        print(f"  编码-解码自洽性: ✓")

    # 保存
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(output_path))
        if verbose:
            print(f"\n[extend] 扩展 tokenizer 已保存: {output_path}")

    return tokenizer


# ---------------------------------------------------------------------------
# SFT 数据加载与格式化
# ---------------------------------------------------------------------------

def load_sft_jsonl(path: Path) -> List[Dict[str, str]]:
    """加载 SFT JSONL 数据。"""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "rough_input" in item and "refined_prompt" in item:
                    examples.append(item)
            except json.JSONDecodeError:
                print(f"  [Warn] JSON 解析失败: {line[:80]}")
    return examples


def convert_sft_to_chat_jsonl(
    input_path: Path,
    output_path: Path,
    tokenizer_path: Optional[Path] = None,
):
    """
    将原始 SFT JSONL 转换为 Chat 格式化文本并保存。
    输出每行: {"text": "<user>...</user><assistant>...</assistant>", "rough_input": "...", "refined_prompt": "..."}
    """
    examples = load_sft_jsonl(input_path)
    print(f"[convert] 加载 {len(examples)} 条 SFT 数据")

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in examples:
            chat_text = format_chat(ex["rough_input"], ex["refined_prompt"])
            out = {
                "text": chat_text,
                "rough_input": ex["rough_input"],
                "refined_prompt": ex["refined_prompt"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"[convert] 已保存 {len(examples)} 条 Chat 格式数据至: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="扩展 Tokenizer 并测试 Chat 模板"
    )
    parser.add_argument("--input-tokenizer", "-i",
                        default="tokenizer/prompt_opt_tokenizer.json",
                        help="输入 tokenizer JSON 路径")
    parser.add_argument("--output-tokenizer", "-o",
                        default="tokenizer/sft_tokenizer.json",
                        help="输出 tokenizer JSON 路径")
    parser.add_argument("--convert-sft", nargs=2, metavar=("INPUT_JSONL", "OUTPUT_JSONL"),
                        help="将 SFT JSONL 转换为 Chat 格式")
    parser.add_argument("--test", action="store_true",
                        help="加载扩展后 tokenizer 并打印测试")

    args = parser.parse_args()

    # 扩展 tokenizer
    tokenizer = extend_tokenizer(
        input_path=Path(args.input_tokenizer),
        output_path=Path(args.output_tokenizer),
    )

    # 转换 SFT 数据
    if args.convert_sft:
        convert_sft_to_chat_jsonl(
            input_path=Path(args.convert_sft[0]),
            output_path=Path(args.convert_sft[1]),
        )
