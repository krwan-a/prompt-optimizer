#!/usr/bin/env python3
"""
SFT 数据生成模块 —— 调用 LLM API（OpenAI / Anthropic）批量生成合成 SFT 数据。

输出格式（JSONL，每行一条）：
    {"rough_input": "...", "refined_prompt": "...", "task_type": "...", "source": "..."}

用法示例：
    python data/collect_sft.py --output data/sft_data.jsonl \\
        --api-provider openai --model gpt-4o \\
        --task-types code_writing copywriting data_analysis \\
        --num-per-task 50 --batch-size 5
"""

import json
import os
import re
import time
import argparse
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple

from tqdm import tqdm

# 并发上限 —— 避免打满本地推理服务
MAX_CONCURRENT = 3

# JSON 解析修复统计
_parse_stats = {"repair_success": 0, "repair_fail": 0}

# ---------------------------------------------------------------------------
# 任务类型定义
# ---------------------------------------------------------------------------

TASK_TYPES = {
    "code_writing":    "编写代码/脚本",
    "copywriting":     "文案撰写与编辑",
    "data_analysis":   "数据分析与可视化",
    "role_playing":    "角色扮演模拟",
    "translation":     "翻译任务",
    "summarization":   "文本总结与摘要",
    "brainstorming":   "头脑风暴与创意生成",
    "teaching":        "教学辅导与解释",
    "creative_writing":"创意写作（故事/诗歌）",
    "email_writing":   "邮件撰写",
    "planning":        "计划制定与项目管理",
    "debate":          "辩论与观点分析",
}

# ---------------------------------------------------------------------------
# API 提示词模板
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_ZH = """你是一个提示词(prompt)优化专家。你的任务是为一个正在学习提示词优化的大语言模型生成训练数据。

你需要为任务类型「{task_type_cn}」生成 {num_examples} 组 (粗糙需求, 精炼提示词) 配对数据。

## 要求
1. **粗糙需求 (rough_input)**：模拟真实用户的原始输入——简短、模糊、缺少上下文、可能语法不完整、口语化。
2. **精炼提示词 (refined_prompt)**：优化后的高质量提示词——清晰、具体、包含必要约束和上下文、有结构。

## 示例
```json
[
  {{"rough_input": "帮我写个Python脚本读CSV", "refined_prompt": "请用Python编写一个CSV文件读取脚本，要求：\n1. 使用pandas库\n2. 支持通过命令行参数指定文件路径\n3. 包含基本的错误处理（文件不存在、空文件等）\n4. 输出前5行数据的基本统计信息"}},
  {{"rough_input": "写个文案推广这个产品", "refined_prompt": "请为以下产品撰写一则面向年轻职场人士的推广文案，要求简洁有力、突出核心卖点。\n\n产品名称：{{product_name}}\n目标用户：{{target_audience}}\n核心卖点：1. {{feature_1}} 2. {{feature_2}}\n\n文案风格：小红书/社交媒体风格，使用emoji和短句。"}}
]
```

请生成 {num_examples} 个 realistic 且多样化的示例。每个示例的 rough_input 要不一样，覆盖不同的表达方式。

## 输出格式
直接输出 JSON 数组，不要包含其他内容：
[
  {{"rough_input": "...", "refined_prompt": "..."}},
  ...
]"""

SYSTEM_PROMPT_EN = """You are a prompt engineering expert. Your task is to generate training data for a language model learning to optimize prompts.

Generate {num_examples} pairs of (rough_input, refined_prompt) for the task type: **{task_type}**.

## Requirements
1. **rough_input**: Short, vague, poorly structured request that sounds like a real user's first draft — incomplete context, typos, informal.
2. **refined_prompt**: Well-crafted, detailed, effective prompt with clear instructions, constraints, and structure.

Generate {num_examples} diverse and realistic examples. Each rough_input should be different.

## Output Format (JSON array only — no extra text):
[
  {{"rough_input": "...", "refined_prompt": "..."}},
  ...
]"""


# ---------------------------------------------------------------------------
# API 调用（本地 Ollama，OpenAI 兼容接口）
# ---------------------------------------------------------------------------

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
OLLAMA_DEFAULT_MODEL = "qwen2.5:7b-instruct"


def call_ollama(
    system_prompt: str,
    model: str = OLLAMA_DEFAULT_MODEL,
    base_url: str = OLLAMA_DEFAULT_BASE_URL,
    temperature: float = 0.8,
    max_tokens: int = 4096,
    timeout: int = 120,
) -> Optional[str]:
    """调用本地 Ollama API（OpenAI 兼容接口）。"""
    from openai import OpenAI

    client = OpenAI(api_key="ollama-placeholder", base_url=base_url)
    user_msg = "请生成数据。" if "你是一个" in system_prompt else "Please generate the data."

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            content = resp.choices[0].message.content
            if content and len(content.strip()) > 10:
                return content
            print(f"  [Ollama] attempt {attempt+1}/3: 返回内容过短 ({len(content or '')} chars)")
        except Exception as e:
            print(f"  [Ollama] attempt {attempt+1}/3 failed: {e}")
            time.sleep(2 ** attempt)
    return None


def _is_valid_generation(text: str, min_chars: int = 5) -> bool:
    """
    检查生成内容是否合规。

    过滤规则：
      - 非空且长度大于 min_chars
      - 不包含大量重复字符（如 "aaaaaaaa"）
      - 不包含大量重复词（如 "abc abc abc"）
    """
    if not text or len(text.strip()) < min_chars:
        return False

    # 字符级重复检测
    if len(text) > 20:
        char_set_ratio = len(set(text)) / max(len(text), 1)
        if char_set_ratio < 0.05:  # 超过 95% 字符是重复的
            return False

    # 词级重复检测
    words = text.split()
    if len(words) > 5:
        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio < 0.1:  # 超过 90% 的词是重复的
            return False

    return True


def _call_and_parse_batch(
    system_prompt: str,
    model: str,
    base_url: str,
    temperature: float,
    task_type: str,
    api_provider: str,
    max_retries: int = 2,
) -> List[Dict]:
    """
    调用 Ollama → 解析 JSON → 验证合规，失败时重试。

    返回有效示例列表（可能为空），不抛出异常。
    """
    for attempt in range(max_retries):
        response = call_ollama(system_prompt, model, base_url, temperature)
        if not response:
            print(f"    [Retry {attempt+1}/{max_retries}] API 返回空")
            time.sleep(1)
            continue

        examples = parse_json_response(response)
        valid = []
        for ex in examples:
            rough = (ex.get("rough_input") or "").strip()
            refined = (ex.get("refined_prompt") or "").strip()
            if not (_is_valid_generation(rough, 5) and _is_valid_generation(refined, 10)):
                continue
            ex["rough_input"] = rough
            ex["refined_prompt"] = refined
            ex["task_type"] = task_type
            ex["source"] = f"{api_provider}/{model}"
            valid.append(ex)

        if valid:
            return valid
        else:
            print(f"    [Retry {attempt+1}/{max_retries}] 解析后无有效示例")
            time.sleep(1)

    return []  # 所有重试耗尽


# ---------------------------------------------------------------------------
# 响应解析（含容错修复）
# ---------------------------------------------------------------------------

def _repair_json(raw: str) -> str:
    """
    修复 JSON 字符串中未转义的换行符。

    使用简单状态机追踪双引号字符串边界，只处理字符串内部的
    \\n / \\r → 转义为 \\\\n / \\\\r，不破坏 JSON 结构本身。
    """
    result = []
    in_string = False
    escaped = False
    for ch in raw:
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
            elif ch == '\n':
                result.append('\\n')
                continue
            elif ch == '\r':
                result.append('\\r')
                continue
        else:
            if ch == '"':
                in_string = True
        result.append(ch)
    return ''.join(result)


def _try_parse_with_fallback(content: str):
    """
    多层 fallback 解析: 标准 JSON → 修复换行后解析 → json5 → demjson3。
    返回 (parsed_list_or_None, was_repaired_bool)。
    """
    # 1) 标准 json.loads
    try:
        data = json.loads(content)
        return (data if isinstance(data, list) else [data]), False
    except json.JSONDecodeError:
        pass

    # 2) 修复字符串内裸换行后重试
    repaired = _repair_json(content)
    try:
        data = json.loads(repaired)
        return (data if isinstance(data, list) else [data]), True
    except json.JSONDecodeError:
        pass

    # 3) json5（允许尾逗号、单引号、多行字符串等）
    try:
        import json5
        data = json5.loads(content)
        return (data if isinstance(data, list) else [data]), True
    except ImportError:
        pass
    except Exception:
        pass

    # 4) demjson3（更宽松）
    try:
        import demjson3
        data = demjson3.decode(content)
        return (data if isinstance(data, list) else [data]), True
    except ImportError:
        pass
    except Exception:
        pass

    return None, False


def parse_json_response(response: Optional[str]) -> List[Dict[str, str]]:
    """
    从 API 响应中解析 JSON 数组。

    容错管线：
      markdown 代码块剥离
      → 标准 json.loads
      → 换行符修复 + json.loads
      → json5（如可用）
      → demjson3（如可用）
      → 返回空列表
    """
    global _parse_stats

    if not response:
        return []

    # 剥离 markdown 代码块
    m = re.search(r'```(?:json)?\s*\n([\s\S]*?)```', response)
    content = m.group(1).strip() if m else response.strip()

    # 多层容错解析
    parsed, was_repaired = _try_parse_with_fallback(content)
    if parsed is not None:
        if was_repaired:
            _parse_stats["repair_success"] += 1
        return parsed

    # 再用正则兜底：尝试从原文提取 JSON 数组片段
    m2 = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
    if m2:
        parsed2, was_repaired2 = _try_parse_with_fallback(m2.group())
        if parsed2 is not None:
            if was_repaired2:
                _parse_stats["repair_success"] += 1
            return parsed2

    _parse_stats["repair_fail"] += 1
    print(f"  [Warn] 所有 JSON 解析方式均失败（前 200 字符）：{response[:200]}")
    return []


def print_parse_stats():
    """打印 JSON 解析修复的统计信息。"""
    total = _parse_stats["repair_success"] + _parse_stats["repair_fail"]
    if total > 0:
        hit_rate = _parse_stats["repair_success"] / total * 100
        print(f"\n  JSON 解析修复统计: "
              f"成功 {_parse_stats['repair_success']} / "
              f"失败 {_parse_stats['repair_fail']} / "
              f"总次数 {total} / "
              f"命中率 {hit_rate:.0f}%")


# ---------------------------------------------------------------------------
# 主生成逻辑
# ---------------------------------------------------------------------------

def generate_sft_data(
    task_types: List[str],
    num_per_task: Dict[str, int],
    output_path: Path,
    api_provider: str = "ollama",
    api_key: str = "",
    model: str = OLLAMA_DEFAULT_MODEL,
    temperature: float = 0.8,
    batch_size: int = 5,
    language: str = "zh",
    resume: bool = False,
    ollama_base_url: str = OLLAMA_DEFAULT_BASE_URL,
    max_concurrent: int = MAX_CONCURRENT,
):
    """为指定任务类型生成 SFT 数据，增量写入 JSONL。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 统计已有数据（--resume 模式）---
    existing_counts: Dict[str, int] = {}
    if resume and output_path.exists():
        for line in output_path.open(encoding="utf-8"):
            try:
                item = json.loads(line)
                task = item.get("task_type", "unknown")
                existing_counts[task] = existing_counts.get(task, 0) + 1
            except json.JSONDecodeError:
                pass
        for task in task_types:
            done = existing_counts.get(task, 0)
            needed = num_per_task[task]
            if done >= needed:
                print(f"  [{task}] 已有 {done} 条 >= 目标 {needed}，跳过")
            else:
                print(f"  [{task}] 已有 {done} 条，还需 {needed - done} 条")
        print()

    total_generated = 0

    for task_type in task_types:
        task_cn = TASK_TYPES.get(task_type, task_type)
        target = num_per_task[task_type]
        done = existing_counts.get(task_type, 0)
        remaining = max(0, target - done)
        if remaining == 0:
            continue

        print(f"\n{'='*60}")
        print(f"  任务: {task_type} ({task_cn})  需生成: {remaining}  "
              f"(并发={max_concurrent}, batch={batch_size})")
        print(f"{'='*60}")

        generated = 0
        pbar = tqdm(total=remaining, desc=task_type, unit="条")

        # ── 并发批量生成 ──────────────────────────────────────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Future → batch_size
            pending: Dict[concurrent.futures.Future, int] = {}

            while generated < remaining or pending:
                # 填充待提交的任务
                while len(pending) < max_concurrent and generated < remaining:
                    num = min(batch_size, remaining - generated)

                    if language == "zh":
                        sys_prompt = SYSTEM_PROMPT_ZH.format(
                            task_type_cn=task_cn, num_examples=num,
                        )
                    else:
                        sys_prompt = SYSTEM_PROMPT_EN.format(
                            task_type=task_type if language == "en" else task_cn,
                            num_examples=num,
                        )

                    future = executor.submit(
                        _call_and_parse_batch,
                        sys_prompt, model, ollama_base_url,
                        temperature, task_type, api_provider,
                    )
                    pending[future] = num
                    generated += num  # 乐观计数

                # 等待至少一个完成
                done_set, _ = concurrent.futures.wait(
                    pending.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                    timeout=120,
                )

                for future in done_set:
                    batch_num = pending.pop(future, batch_size)
                    try:
                        valid = future.result(timeout=5)
                    except Exception as e:
                        valid = []
                        print(f"  [Error] batch 异常: {e}")

                    if valid:
                        with output_path.open("a", encoding="utf-8") as f:
                            for ex in valid:
                                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        pbar.update(len(valid))
                        total_generated += len(valid)

        pbar.close()

    print(f"\n{'='*60}")
    print(f"  完成！共生成 {total_generated} 条，保存至 {output_path}")
    print_parse_stats()
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="生成 SFT 数据（默认使用本地 Ollama 服务）"
    )

    parser.add_argument("--output", "-o", default="data/sft_data.jsonl",
                        help="输出 JSONL 路径 (default: data/sft_data.jsonl)")
    parser.add_argument("--api-provider", choices=["ollama", "openai", "anthropic"],
                        default="ollama", help="API 提供商 (default: ollama)")
    parser.add_argument("--api-key", default="",
                        help="API Key（外部 API 时使用；ollama 模式无需填写）")
    parser.add_argument("--model", default=OLLAMA_DEFAULT_MODEL,
                        help=f"模型名 (default: {OLLAMA_DEFAULT_MODEL})")
    parser.add_argument("--ollama-base-url", default=OLLAMA_DEFAULT_BASE_URL,
                        help=f"Ollama 服务地址 (default: {OLLAMA_DEFAULT_BASE_URL})")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT,
                        help=f"最大并发请求数 (default: {MAX_CONCURRENT})")
    parser.add_argument("--task-types", nargs="+",
                        default=list(TASK_TYPES.keys()),
                        choices=list(TASK_TYPES.keys()) + ["all"],
                        help="任务类型列表 (default: 全部)")
    parser.add_argument("--num-per-task", type=int, default=100,
                        help="每种任务类型生成条数 (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="生成温度 (default: 0.8)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="每次 API 调用生成条数 (default: 5)")
    parser.add_argument("--language", choices=["zh", "en"], default="zh",
                        help="生成语言 (default: zh)")
    parser.add_argument("--resume", action="store_true",
                        help="续写模式：统计已有数据，只补足差额")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # ── API Key 检查（ollama 模式不需要） ──
    api_key = args.api_key
    if args.api_provider != "ollama":
        if not api_key:
            env_var = f"{args.api_provider.upper()}_API_KEY"
            api_key = os.environ.get(env_var, "")
        if not api_key:
            print(f"[Error] 外部 API ({args.api_provider}) 需要 --api-key 或 {env_var}")
            exit(1)

    # 解析任务类型
    task_types = list(TASK_TYPES.keys()) if "all" in args.task_types else args.task_types
    num_per_task = {t: args.num_per_task for t in task_types}

    generate_sft_data(
        task_types=task_types,
        num_per_task=num_per_task,
        output_path=Path(args.output),
        api_provider=args.api_provider,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        batch_size=args.batch_size,
        language=args.language,
        resume=args.resume,
        ollama_base_url=args.ollama_base_url,
        max_concurrent=args.max_concurrent,
    )
