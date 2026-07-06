#!/usr/bin/env python3
"""
用本地 Ollama 生成提示词工程领域文本，扩充预训练语料。

用法：
    python data/generate_pretrain_ollama.py \
        --output data/domain_augmented.txt \
        --num-docs 200 \
        --concurrent 3
"""

import argparse
import json
import os
import time
import concurrent.futures
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

OLLAMA_BASE = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

TOPICS = [
    "什么是提示词工程（Prompt Engineering）？请用300字左右解释其核心概念和重要性。",
    "什么是思维链提示（Chain-of-Thought Prompting）？请用200-300字解释原理并给出一个例子。",
    "什么是Few-shot提示？请用200-300字解释其工作原理和适用场景。",
    "什么是Zero-shot提示？与Few-shot有什么区别？请用200-300字说明。",
    "提示词中system prompt和user prompt各自的作用是什么？请用200-300字解释。",
    "写提示词时应该遵循哪些基本原则？请列出5-8条原则并简要解释每条。",
    "什么是角色扮演提示（Role Prompting）？请用200-300字解释并给出一个例子。",
    "什么是思维树提示（Tree-of-Thought Prompting）？请用200-300字解释。",
    "在提示词中如何有效地使用示例？请用200-300字说明最佳实践。",
    "什么是负向提示（Negative Prompting）？请用200-300字解释其用途。",
    "如何编写清晰的指令型提示词？请用200-300字说明关键要素。",
    "提示词中的格式控制（如JSON输出）如何实现？请用200-300字说明。",
    "什么是迭代式提示优化？请用200-300字解释其流程。",
    "如何处理大模型输出中的幻觉问题？请用200-300字说明提示词层面的方法。",
    "什么是上下文长度（Context Window）？在写提示词时需要注意什么？请用200-300字说明。",
    "如何通过提示词控制输出风格和语气？请用200-300字说明方法。",
    "什么是结构化提示？请用200-300字解释并给出一个模板示例。",
    "提示词中的条件逻辑如何实现？请用200-300字说明。",
    "什么是多轮对话中的提示词管理？请用200-300字说明策略。",
    "如何评估一个提示词的质量？请用200-300字列出评估维度和方法。",
    "什么是自动提示优化（Automatic Prompt Optimization）？请用200-300字介绍。",
    "在代码生成任务中如何设计有效的提示词？请用200-300字说明。",
    "在文本总结任务中如何设计有效的提示词？请用200-300字说明。",
    "在翻译任务中如何设计有效的提示词？请用200-300字说明。",
    "在数据分析任务中如何设计有效的提示词？请用200-300字说明。",
    "在创意写作任务中如何设计有效的提示词？请用200-300字说明。",
    "什么是系统1和系统2提示？请用200-300字解释其概念。",
    "如何通过提示词让大模型进行事实核查？请用200-300字说明。",
    "什么是提示链（Prompt Chaining）？请用200-300字解释其应用场景。",
    "什么是自适应提示？请用200-300字解释其概念和优势。",
    "如何处理提示词中的敏感内容？请用200-300字说明最佳实践。",
    "什么是元提示（Meta-Prompting）？请用200-300字解释。",
    "什么是情感提示（Emotional Prompting）？请用200-300字解释其效果。",
    "如何利用提示词让模型更好地理解和遵循约束条件？请用200-300字说明。",
    "在提示词中如何使用分隔符提高清晰度？请用200-300字说明。",
]

SYSTEM_PROMPT = "你是一个提示词工程(Prompt Engineering)领域的写作助手。请用中文写一段200-300字的说明文字，要求内容准确、语言清晰、结构完整。不要使用markdown格式，直接输出纯文本段落。"


def generate_document(topic: str, model: str, base_url: str) -> Optional[str]:
    """调用 Ollama 生成一篇领域文档。"""
    from openai import OpenAI
    client = OpenAI(api_key="placeholder", base_url=base_url)

    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": topic},
                ],
                max_tokens=600,
                temperature=0.7,
                timeout=60,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text and len(text) > 100:
                return text.replace("\n", " ").replace("\r", " ").strip()
        except Exception as e:
            if attempt == 0:
                time.sleep(2)
    return None


def main():
    parser = argparse.ArgumentParser(description="生成提示词工程领域文本")
    parser.add_argument("--output", default="data/domain_augmented.txt")
    parser.add_argument("--num-docs", type=int, default=len(TOPICS),
                       help=f"生成文档数 (max {len(TOPICS)})")
    parser.add_argument("--concurrent", type=int, default=3)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--base-url", default=OLLAMA_BASE)
    args = parser.parse_args()

    topics = TOPICS[:args.num_docs]
    print(f"准备生成 {len(topics)} 篇提示词工程领域文档 (并发={args.concurrent})")

    docs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as ex:
        fut_to_topic = {ex.submit(generate_document, t, args.model, args.base_url): t for t in topics}
        for fut in tqdm(concurrent.futures.as_completed(fut_to_topic),
                       total=len(fut_to_topic), desc="Generating"):
            topic = fut_to_topic[fut]
            try:
                doc = fut.result(timeout=120)
                if doc:
                    docs.append(doc)
                else:
                    print(f"  [Warn] 生成失败: {topic[:30]}...")
            except Exception as e:
                print(f"  [Error] {topic[:30]}...: {e}")

    if not docs:
        print("未生成任何文档，退出")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc + "\n")

    print(f"\n生成完成: {len(docs)} 篇文档")
    print(f"总字符: {sum(len(d) for d in docs):,}")
    print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
