#!/usr/bin/env python3
"""
交互式提示词优化 —— 加载训好的 SFT 模型，输入粗糙需求，输出精炼提示词。

用法：
    python interact.py

    # 或单次使用：
    python interact.py --input "帮我写个Python脚本"
"""

import argparse
import sys
from pathlib import Path

# ── 项目根目录 ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tokenizers import Tokenizer
from model.model import GPT


def load_model(
    checkpoint_path="sft_checkpoints/sft-best.pt",
    tokenizer_path="tokenizer/sft_tokenizer.json",
    device="cpu",
):
    """加载 SFT 模型和 tokenizer。"""
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()

    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    if isinstance(args, dict):
        d_model = args.get("d_model", 256)
        n_layer = args.get("n_layer", 6)
        n_head = args.get("n_head", 8)
        ffn_hidden = args.get("ffn_hidden", 1024)
        max_len = args.get("max_length", 512)
    else:
        d_model, n_layer, n_head, ffn_hidden, max_len = 256, 6, 8, 1024, 512

    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layer=n_layer,
        n_head=n_head,
        ffn_hidden=ffn_hidden,
        max_seq_len=max_len,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total:,} params on {device}")
    return model, tokenizer


def optimize_prompt(
    model,
    tokenizer,
    rough_input,
    max_new_tokens=200,
    temperature=0.8,
    top_k=10,
    device="cpu",
):
    """对粗糙需求生成精炼提示词。"""
    prompt_text = f"<user>{rough_input}</user><assistant>"
    encoding = tokenizer.encode(prompt_text)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

    if input_ids.size(1) > model.max_seq_len:
        input_ids = input_ids[:, -model.max_seq_len:]

    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    full_text = tokenizer.decode(out_ids[0].tolist())

    # 提取 <assistant> 标签之间的内容
    start = full_text.find("<assistant>")
    if start == -1:
        return full_text.strip()
    content = full_text[start + len("<assistant>"):]
    end = content.find("</assistant>")
    if end != -1:
        content = content[:end]
    return content.strip()


# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="交互式提示词优化")
    parser.add_argument("--checkpoint", default="sft_checkpoints/sft-best.pt")
    parser.add_argument("--tokenizer", default="tokenizer/sft_tokenizer.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--input", "-i", help="单次输入（不进入交互模式）")

    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model(args.checkpoint, args.tokenizer, args.device)

    if args.input:
        result = optimize_prompt(
            model, tokenizer, args.input,
            args.max_new_tokens, args.temperature, args.top_k, args.device,
        )
        print(f"\n{'='*60}")
        print(f"  粗糙需求: {args.input}")
        print(f"{'='*60}")
        print(f"  精炼提示词:")
        print(f"  {result}")
        print(f"{'='*60}")
        return

    # ── 交互模式 ──
    print(f"\n{'='*60}")
    print(f"  提示词优化助手（输入 Ctrl+C 或 q 退出）")
    print(f"{'='*60}")
    print()

    while True:
        try:
            rough = input(">>> 粗糙需求: ").strip()
            if not rough or rough.lower() in ("q", "quit", "exit"):
                break

            result = optimize_prompt(
                model, tokenizer, rough,
                args.max_new_tokens, args.temperature, args.top_k, args.device,
            )
            print(f"\n  {'─'*55}")
            print(f"  精炼提示词:")
            for line in result.split("\n"):
                print(f"  {line}")
            print(f"  {'─'*55}")
            print()

        except KeyboardInterrupt:
            print()
            break
        except Exception as e:
            print(f"  [Error] {e}")
            print()

    print("bye!")


if __name__ == "__main__":
    main()
