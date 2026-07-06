#!/usr/bin/env python3
"""
SFT 模型评估脚本 —— 对测试 rough_input 生成精炼提示词，并用 LLM 裁判打分。

评估流水线：
  1. 加载 SFT 模型 + 扩展 tokenizer
  2. 对每条 test rough_input 做 <user>...</user><assistant> 生成
  3. 提取 <assistant> 标签之间的生成内容
  4. 将 (rough_input, 模型输出) 发给 Claude API 按 rubric 打分
  5. 解析 JSON 分数并汇总报告

Rubric 四个维度（各 1-5 分）：
  A. 任务目标是否明确清晰
  B. 是否包含必要的上下文/背景信息
  C. 是否给出了输出格式或结构约束
  D. 整体语言是否清晰无歧义

用法示例：
    python eval/evaluate.py \
        --checkpoint sft_checkpoints/sft-best.pt \
        --tokenizer tokenizer/sft_tokenizer.json \
        --test-data data/sft_data.jsonl \
        --num-samples 20 \
        --output-dir eval_results

    # 只生成不评分（跳过 LLM 裁判）：
    python eval/evaluate.py ... --no-judge
"""

import argparse
import json
import os
import re
import sys
import time
import concurrent.futures
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ── 项目根目录导入 ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.model import GPT

# 并发上限
MAX_CONCURRENT = 3


# ======================================================================
# 数据结构
# ======================================================================

@dataclass
class ScoreEntry:
    """单条评估记录。"""
    rough_input: str
    generated_prompt: str
    scores: Dict[str, int] = field(default_factory=dict)
    reason: str = ""
    judge_error: str = ""


@dataclass
class EvalReport:
    """汇总报告。"""
    total: int = 0
    scored: int = 0
    avg_scores: Dict[str, float] = field(default_factory=dict)
    overall_avg: float = 0.0
    best: List[ScoreEntry] = field(default_factory=list)
    worst: List[ScoreEntry] = field(default_factory=list)
    entries: List[ScoreEntry] = field(default_factory=list)


# ======================================================================
# 生成器
# ======================================================================

def load_model_and_tokenizer(
    checkpoint_path: str, tokenizer_path: str, device: str
) -> Tuple[GPT, "Tokenizer"]:
    """加载 SFT 模型和扩展 tokenizer。"""
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    print(f"  Tokenizer vocab size: {vocab_size}")

    # 从 checkpoint 获取模型配置
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_args = ckpt.get("args", {})
    if isinstance(model_args, dict):
        d_model = model_args.get("d_model", 256)
        n_layer = model_args.get("n_layer", 6)
        n_head = model_args.get("n_head", 8)
        ffn_hidden = model_args.get("ffn_hidden", 1024)
        max_len = model_args.get("max_length", 512)
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

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {total_params:,} params on {device}")

    return model, tokenizer


def generate_prompt(
    model: GPT,
    tokenizer: "Tokenizer",
    rough_input: str,
    max_new_tokens: int = 128,
    temperature: float = 0.8,
    top_k: int = 10,
    max_length: int = 512,
    device: str = "cpu",
) -> str:
    """
    为 rough_input 生成精炼提示词。

    1. 格式化为 <user>{rough}</user><assistant>
    2. 生成直到 </assistant> 或 max_new_tokens
    3. 提取 <assistant> 和 </assistant> 之间的内容
    """
    prompt_text = f"<user>{rough_input}</user><assistant>"
    encoding = tokenizer.encode(prompt_text)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

    # 超长截断（从左侧）
    if input_ids.size(1) > max_length:
        input_ids = input_ids[:, -max_length:]

    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    full_text = tokenizer.decode(out[0].tolist())

    # 提取 assistant 回复
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
# LLM 裁判（本地 Ollama 版）
# ======================================================================

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_JUDGE_MODEL = "qwen2.5:7b-instruct"

JUDGE_PROMPT_TEMPLATE = """You are an expert prompt engineering evaluator. Your task is to assess the quality of an **optimized prompt** generated from a rough user request.

Evaluate the optimized prompt on these four criteria, each scored **1 to 5**:

| Score | Meaning |
|-------|---------|
| 1     | Very poor — missing or completely inadequate |
| 2     | Below average — partially addressed but insufficient |
| 3     | Average — adequate but not remarkable |
| 4     | Good — well addressed with minor room for improvement |
| 5     | Excellent — fully and precisely addressed |

### Criterion A: Task Goal Clarity (任务目标明确清晰)
Does the prompt clearly state what the AI should do? Can the AI understand the exact task without ambiguity?

### Criterion B: Context & Background (包含必要的上下文/背景信息)
Does the prompt provide sufficient context, background, constraints, or examples needed to complete the task well?

### Criterion C: Output Structure & Constraints (输出格式/结构约束)
Does the prompt specify the desired output format, structure, length, or other concrete constraints?

### Criterion D: Language Clarity (语言清晰无歧义)
Is the language precise, well-structured, and free of ambiguity? Are instructions logically ordered?

---

### Original rough request:
```
{rough_input}
```

### Optimized prompt to evaluate:
```
{generated_prompt}
```

---

Return ONLY a valid JSON object with this exact structure (no markdown, no extra text):
{{"scores": {{"task_clarity": <1-5>, "context_completeness": <1-5>, "output_structure": <1-5>, "language_clarity": <1-5>}}, "reason": "<one-sentence justification>"}}"""


def _is_valid_judge_response(text: str) -> bool:
    """检查裁判返回内容是否合规。"""
    if not text or len(text.strip()) < 20:
        return False
    # 检查是否包含评分关键词（防止模型乱答非所问）
    has_score_keywords = any(kw in text.lower() for kw in
                             ["task_clarity", "context_completeness", "output_structure", "language_clarity"])
    return has_score_keywords


def call_judge_ollama(
    rough_input: str,
    generated_prompt: str,
    model: str = OLLAMA_JUDGE_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    max_retries: int = 3,
) -> Tuple[Optional[Dict[str, int]], str]:
    """
    调用本地 Ollama 模型作为裁判，返回 (scores_dict, reason)。

    内部包含：
      - JSON 解析容错（调用 _parse_judge_response）
      - 空内容 / 乱码检测
      - 最多 max_retries 次重试
    """
    from openai import OpenAI

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        rough_input=rough_input[:500],
        generated_prompt=generated_prompt[:1000],
    )

    client = OpenAI(api_key="ollama-placeholder", base_url=base_url)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",
                     "content": "You are a precise evaluator. Always return valid JSON only. "
                                "Do NOT wrap JSON in markdown code blocks."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.2,
                timeout=120,
            )
            text = (resp.choices[0].message.content or "").strip()

            # 空内容检测
            if not text:
                print(f"    [Judge retry {attempt+1}/{max_retries}] 空响应")
                time.sleep(1)
                continue

            # 内容合规检测
            if not _is_valid_judge_response(text):
                print(f"    [Judge retry {attempt+1}/{max_retries}] 响应不含评分关键词")
                time.sleep(1)
                continue

            scores, reason = _parse_judge_response(text)
            if scores:
                return scores, reason
            else:
                print(f"    [Judge retry {attempt+1}/{max_retries}] JSON 解析失败")

        except Exception as e:
            print(f"    [Judge retry {attempt+1}/{max_retries}] 请求异常: {e}")
            time.sleep(2 ** attempt)

    return None, f"Judge failed after {max_retries} retries"


def _parse_judge_response(text: str) -> Tuple[Optional[Dict[str, int]], str]:
    """解析裁判输出，提取 JSON 分数。"""
    # 先试直接 JSON 解析
    # 移除可能的 markdown 代码块标记
    text_clean = text.strip()
    m = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", text_clean)
    if m:
        text_clean = m.group(1).strip()

    try:
        data = json.loads(text_clean)
        scores = data.get("scores", data)
        reason = data.get("reason", "")

        # 验证 score 字段
        required_keys = ["task_clarity", "context_completeness", "output_structure", "language_clarity"]
        parsed = {}
        for key in required_keys:
            val = scores.get(key)
            if val is not None and isinstance(val, (int, float)) and 1 <= val <= 5:
                parsed[key] = int(val)

        if len(parsed) == 4:
            return parsed, reason
        else:
            return None, f"Missing or invalid score fields in: {parsed}"
    except json.JSONDecodeError:
        pass

    # 尝试用正则提取 JSON 对象
    m = re.search(r"\{[^{}]*\"task_clarity\"[^{}]*\}", text_clean)
    if m:
        try:
            data = json.loads(m.group())
            scores = data.get("scores", data)
            reason = data.get("reason", "")
            parsed = {}
            for key in ["task_clarity", "context_completeness", "output_structure", "language_clarity"]:
                val = scores.get(key)
                if isinstance(val, (int, float)) and 1 <= val <= 5:
                    parsed[key] = int(val)
            if len(parsed) == 4:
                return parsed, reason
        except (json.JSONDecodeError, ValueError):
            pass

    return None, f"Could not parse JSON from response: {text[:200]}"


# ======================================================================
# 汇总报告
# ======================================================================

def generate_report(entries: List[ScoreEntry], top_k: int = 5) -> EvalReport:
    """生成汇总报告。"""
    scored = [e for e in entries if e.scores]
    report = EvalReport()
    report.total = len(entries)
    report.scored = len(scored)

    if not scored:
        return report

    # 平均分
    dims = ["task_clarity", "context_completeness", "output_structure", "language_clarity"]
    totals = {d: 0 for d in dims}
    for e in scored:
        for d in dims:
            totals[d] += e.scores.get(d, 0)

    report.avg_scores = {d: totals[d] / len(scored) for d in dims}
    report.overall_avg = sum(report.avg_scores.values()) / len(dims)

    # 计算每个条目总分排序
    def overall(e: ScoreEntry) -> float:
        if not e.scores:
            return 0
        return sum(e.scores.values()) / len(e.scores)

    scored_sorted = sorted(scored, key=overall, reverse=True)

    report.best = scored_sorted[:top_k]
    # 过滤掉 0 分的最差案例
    scored_with_scores = [e for e in scored_sorted if overall(e) > 0]
    report.worst = scored_with_scores[-top_k:] if len(scored_with_scores) >= top_k else scored_with_scores[::-1]
    report.entries = entries

    return report


def print_report(report: EvalReport):
    """打印人类可读的报告。"""
    print()
    print("=" * 58)
    print("  Evaluation Report")
    print("=" * 58)
    print(f"  Total examples:    {report.total}")
    print(f"  Successfully judged: {report.scored}")
    print(f"  Judge failures:    {report.total - report.scored}")
    print()

    if report.scored == 0:
        print("  [No scored examples to report.]")
        return

    print(f"  ── Average Scores ──")
    for dim, avg in report.avg_scores.items():
        label_map = {
            "task_clarity": "A. Task Goal Clarity",
            "context_completeness": "B. Context & Background",
            "output_structure": "C. Output Structure",
            "language_clarity": "D. Language Clarity",
        }
        bar = "█" * int(avg) + "░" * (5 - int(avg))
        print(f"    {label_map.get(dim, dim):30s}  {avg:.2f}  {bar}")
    print(f"    {'─' * 45}")
    print(f"    {'Overall Average':30s}  {report.overall_avg:.2f}")
    print()

    # Top cases
    if report.best:
        print(f"  ── Best Examples (Top {len(report.best)}) ──")
        for i, e in enumerate(report.best, 1):
            avg = sum(e.scores.values()) / len(e.scores)
            print(f"    {i}. Score: {avg:.1f} | Rough: {e.rough_input[:60]}...")
            print(f"       Prompt: {e.generated_prompt[:100]}...")
            print(f"       Reason: {e.reason[:120]}")
            print()

    # Worst cases
    if report.worst:
        print(f"  ── Worst Examples (Bottom {len(report.worst)}) ──")
        for i, e in enumerate(report.worst, 1):
            avg = sum(e.scores.values()) / len(e.scores)
            print(f"    {i}. Score: {avg:.1f} | Rough: {e.rough_input[:60]}...")
            print(f"       Prompt: {e.generated_prompt[:100]}...")
            if e.reason:
                print(f"       Reason: {e.reason[:120]}")
            print()

    print("=" * 58)


def save_report(report: EvalReport, output_dir: Path):
    """保存报告到文件。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSONL 原始数据
    jsonl_path = output_dir / "eval_scores.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for e in report.entries:
            f.write(json.dumps({
                "rough_input": e.rough_input,
                "generated_prompt": e.generated_prompt,
                "scores": e.scores,
                "reason": e.reason,
                "judge_error": e.judge_error,
            }, ensure_ascii=False) + "\n")
    print(f"  Saved: {jsonl_path}")

    # TXT 报告
    txt_path = output_dir / "eval_report.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        # 重定向 print 到文件
        import io
        buf = io.StringIO()
        # 手动构建报告内容
        f.write("=" * 58 + "\n")
        f.write("  Evaluation Report\n")
        f.write("=" * 58 + "\n")
        f.write(f"  Total examples:      {report.total}\n")
        f.write(f"  Successfully judged: {report.scored}\n")
        f.write(f"  Judge failures:      {report.total - report.scored}\n\n")

        if report.scored > 0:
            f.write("  ── Average Scores ──\n")
            for dim, avg in report.avg_scores.items():
                label_map = {
                    "task_clarity": "A. Task Goal Clarity",
                    "context_completeness": "B. Context & Background",
                    "output_structure": "C. Output Structure",
                    "language_clarity": "D. Language Clarity",
                }
                f.write(f"    {label_map.get(dim, dim):30s}  {avg:.2f}\n")
            f.write(f"    {'─' * 45}\n")
            f.write(f"    {'Overall Average':30s}  {report.overall_avg:.2f}\n\n")

            if report.best:
                f.write("  ── Best Examples ──\n")
                for i, e in enumerate(report.best, 1):
                    avg = sum(e.scores.values()) / len(e.scores)
                    f.write(f"    {i}. Score: {avg:.1f}\n")
                    f.write(f"       Rough: {e.rough_input[:100]}\n")
                    f.write(f"       Prompt: {e.generated_prompt[:200]}\n")
                    if e.reason:
                        f.write(f"       Reason: {e.reason[:150]}\n")
                    f.write("\n")

            if report.worst:
                f.write("  ── Worst Examples ──\n")
                for i, e in enumerate(report.worst, 1):
                    avg = sum(e.scores.values()) / len(e.scores)
                    f.write(f"    {i}. Score: {avg:.1f}\n")
                    f.write(f"       Rough: {e.rough_input[:100]}\n")
                    f.write(f"       Prompt: {e.generated_prompt[:200]}\n")
                    if e.reason:
                        f.write(f"       Reason: {e.reason[:150]}\n")
                    f.write("\n")

    print(f"  Saved: {txt_path}")

    # JSON 汇总
    summary_path = output_dir / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total": report.total,
            "scored": report.scored,
            "avg_scores": report.avg_scores,
            "overall_avg": report.overall_avg,
        }, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {summary_path}")


# ======================================================================
# 主流水线
# ======================================================================

def load_test_data(path: Path, num_samples: Optional[int] = None) -> List[Dict]:
    """加载测试数据 JSONL。"""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if "rough_input" in item:
                    examples.append(item)
            except json.JSONDecodeError:
                pass
    if num_samples is not None and num_samples < len(examples):
        examples = examples[:num_samples]
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="SFT 模型评估：生成 + LLM 裁判打分",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--checkpoint", required=True,
                        help="SFT checkpoint 路径")
    parser.add_argument("--tokenizer", required=True,
                        help="扩展 tokenizer JSON 路径")
    parser.add_argument("--test-data", required=True,
                        help="测试数据 JSONL（含 rough_input 字段）")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="测试样本数（从数据集中取前 N 条）")
    parser.add_argument("--output-dir", default="eval_results",
                        help="输出目录")

    # 生成参数
    gen_group = parser.add_argument_group("Generation")
    gen_group.add_argument("--max-new-tokens", type=int, default=128,
                           help="每个样本最大生成 token 数")
    gen_group.add_argument("--temperature", type=float, default=0.7,
                           help="生成温度")
    gen_group.add_argument("--top-k", type=int, default=10,
                           help="Top-K 采样")

    # 裁判参数
    judge_group = parser.add_argument_group("Judge (Local Ollama)")
    judge_group.add_argument("--judge-model", default=OLLAMA_JUDGE_MODEL,
                             help=f"裁判模型 (default: {OLLAMA_JUDGE_MODEL})")
    judge_group.add_argument("--ollama-base-url", default=OLLAMA_BASE_URL,
                             help=f"Ollama 服务地址 (default: {OLLAMA_BASE_URL})")
    judge_group.add_argument("--judge-retries", type=int, default=3,
                             help="裁判重试次数 (default: 3)")
    judge_group.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT,
                             help=f"最大并发裁判请求 (default: {MAX_CONCURRENT})")
    judge_group.add_argument("--no-judge", action="store_true",
                             help="只生成不评分")

    # 其他
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(args.seed)

    # ── 1. 加载模型 ──
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        args.checkpoint, args.tokenizer, args.device,
    )

    # ── 2. 加载测试数据 ──
    print(f"\nLoading test data from {args.test_data}")
    test_examples = load_test_data(Path(args.test_data), args.num_samples)
    print(f"  Loaded {len(test_examples)} test examples")

    if not test_examples:
        print("[Error] No test examples found.")
        exit(1)

    # ── 3. 生成 ──
    print(f"\n{'='*60}")
    print(f"  Generating prompts...")
    print(f"{'='*60}")

    entries: List[ScoreEntry] = []
    for i, ex in enumerate(test_examples):
        rough = ex["rough_input"]
        print(f"  [{i+1}/{len(test_examples)}] Generating...", end=" ", flush=True)

        try:
            generated = generate_prompt(
                model=model,
                tokenizer=tokenizer,
                rough_input=rough,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=args.device,
            )
            print(f"({len(generated)} chars)")
        except Exception as e:
            print(f"[Error: {e}]")
            generated = f"[Generation failed: {e}]"

        entries.append(ScoreEntry(
            rough_input=rough,
            generated_prompt=generated,
        ))

    # ── 4. 裁判打分（并发调用本地 Ollama） ──
    if not args.no_judge:
        print(f"\n{'='*60}")
        print(f"  Judging with local Ollama ({args.judge_model}) ...")
        print(f"  Concurrent workers: {args.max_concurrent}")
        print(f"{'='*60}")

        # 收集待评分条目
        to_judge = []
        for i, entry in enumerate(entries):
            if not entry.generated_prompt or entry.generated_prompt.startswith("[Generation failed"):
                entry.judge_error = "Generation failed, skipping judge"
                print(f"  [{i+1}/{len(entries)}] Skipping judge (generation failed)")
            else:
                to_judge.append((i, entry))

        judged_count = 0
        fail_count = 0

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_concurrent
        ) as executor:
            future_map = {}
            for idx, entry in to_judge:
                future = executor.submit(
                    call_judge_ollama,
                    entry.rough_input,
                    entry.generated_prompt,
                    args.judge_model,
                    args.ollama_base_url,
                    args.judge_retries,
                )
                future_map[future] = (idx, entry)

            for future in concurrent.futures.as_completed(future_map):
                idx, entry = future_map[future]
                try:
                    scores, reason = future.result(timeout=180)
                    if scores:
                        entry.scores = scores
                        entry.reason = reason
                        judged_count += 1
                        avg = sum(scores.values()) / len(scores)
                        print(f"  [{idx+1}/{len(entries)}] Judged avg={avg:.1f}")
                    else:
                        entry.judge_error = reason
                        fail_count += 1
                        print(f"  [{idx+1}/{len(entries)}] Judge fail: {reason[:60]}")
                except Exception as e:
                    entry.judge_error = str(e)
                    fail_count += 1
                    print(f"  [{idx+1}/{len(entries)}] Judge exception: {e}")

        print(f"\n  Judge done: {judged_count} success, {fail_count} failed")

    # ── 5. 报告 ──
    print(f"\n{'='*60}")
    print(f"  Generating report...")
    print(f"{'='*60}")

    output_dir = Path(args.output_dir)
    report = generate_report(entries, top_k=5)
    print_report(report)
    save_report(report, output_dir)

    print(f"\nDone! Report saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
