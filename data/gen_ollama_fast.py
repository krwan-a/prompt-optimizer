#!/usr/bin/env python3
"""
快速版 Ollama 领域文本生成 —— 严格超时控制 + 增量保存。

用法：
    python data/gen_ollama_fast.py --output data/ollama_docs.txt --num-docs 40
"""

import argparse, concurrent.futures, json, os, time
from pathlib import Path
from openai import OpenAI

OLLAMA_BASE = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

TOPICS = [
    "请用中文写一段300-500字的说明文，解释什么是提示词工程及其重要性。",
    "请用中文写一段300-500字的说明文，介绍思维链提示（Chain-of-Thought）的原理和使用方法。",
    "请用中文写一段300-500字的说明文，介绍Few-shot提示的工作原理和适用场景。",
    "请用中文写一段300-500字的说明文，介绍Zero-shot提示及其与Few-shot的区别。",
    "请用中文写一段300-500字的说明文，解释system prompt和user prompt的区别和用法。",
    "请用中文写一段300-500字的说明文，列出编写高质量提示词的基本原则。",
    "请用中文写一段300-500字的说明文，介绍角色扮演提示的设计方法。",
    "请用中文写一段300-500字的说明文，介绍如何在提示词中有效使用示例。",
    "请用中文写一段300-500字的说明文，解释什么是负向提示及其用途。",
    "请用中文写一段300-500字的说明文，介绍如何编写清晰的指令型提示词。",
    "请用中文写一段300-500字的说明文，介绍提示词中的格式控制方法。",
    "请用中文写一段300-500字的说明文，介绍提示词优化的迭代流程。",
    "请用中文写一段300-500字的说明文，介绍如何处理大模型输出中的幻觉问题。",
    "请用中文写一段300-500字的说明文，解释上下文长度对提示词设计的影响。",
    "请用中文写一段300-500字的说明文，介绍如何通过提示词控制输出风格。",
    "请用中文写一段300-500字的说明文，介绍结构化提示的设计方法。",
    "请用中文写一段300-500字的说明文，介绍多轮对话中的提示词管理策略。",
    "请用中文写一段300-500字的说明文，介绍如何评估一个提示词的质量。",
    "请用中文写一段300-500字的说明文，介绍代码生成任务中的提示词设计。",
    "请用中文写一段300-500字的说明文，介绍文本总结任务中的提示词设计。",
    "请用中文写一段300-500字的说明文，介绍翻译任务中的提示词设计。",
    "请用中文写一段300-500字的说明文，介绍数据分析任务中的提示词设计。",
    "请用中文写一段300-500字的说明文，介绍创意写作任务中的提示词设计。",
    "请用中文写一段300-500字的说明文，解释什么是提示链及其应用场景。",
    "请用中文写一段300-500字的说明文，介绍如何设计提示词做情感分析。",
    "请用中文写一段300-500字的说明文，介绍如何设计提示词做文本分类。",
    "请用中文写一段300-500字的说明文，介绍如何设计提示词做代码解释。",
    "请用中文写一段300-500字的说明文，介绍如何设计提示词做文案撰写。",
    "请用中文写一段300-500字的说明文，介绍如何设计提示词做头脑风暴。",
    "请用中文写一段300-500字的说明文，介绍如何设计提示词做故事创作。",
    "请用中文写一段300-500字的说明文，介绍如何设计提示词做角色扮演。",
    "请用中文写一段300-500字的说明文，介绍什么是RAG检索增强生成。",
    "请用中文写一段300-500字的说明文，介绍提示词中的温度参数如何影响输出。",
    "请用中文写一段300-500字的说明文，介绍如何设计提示词让模型输出结构化数据。",
    "请用中文写一段300-500字的说明文，介绍如何利用提示词让模型一步步思考。",
    "请用中文写一段300-500字的说明文，介绍思维树提示的原理和应用。",
    "请用中文写一段300-500字的说明文，介绍元提示的概念和优势。",
    "请用中文写一段300-500字的说明文，介绍情感提示的效果和使用方法。",
    "请用中文写一段300-500字的说明文，介绍提示词测试驱动开发的方法。",
    "请用中文写一段300-500字的说明文，介绍CoT思维链和ToT思维树的区别。",
    "请用中文写一段300-500字的说明文，介绍Agent提示词设计的特殊性。",
    "请用中文写一段300-500字的说明文，介绍ReAct提示模式的工作原理。",
    "请用中文写一段300-500字的说明文，介绍什么是微调及其与提示词工程的关系。",
    "请用中文写一段300-500字的说明文，介绍嵌入向量的概念和作用。",
    "请用中文写一段300-500字的说明文，介绍Transformer注意力机制的基本原理。",
]

CLIENT = OpenAI(api_key="placeholder", base_url=OLLAMA_BASE)
SYSTEM_PROMPT = "你是一个技术写作助手。请用中文写一段300-500字的说明文，内容准确、结构完整。不要使用markdown格式，直接输出纯文本段落。"


def generate_one(topic):
    """调用一次 Ollama，严格 90s 超时。"""
    try:
        resp = CLIENT.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": topic}],
            max_tokens=800,
            temperature=0.7,
            timeout=90,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text and len(text) > 100:
            return text.replace("\n", " ").replace("\r", " ").strip()
    except Exception as e:
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/ollama_docs.txt")
    parser.add_argument("--num-docs", type=int, default=len(TOPICS))
    parser.add_argument("--concurrent", type=int, default=2)
    args = parser.parse_args()

    topics = TOPICS[:args.num_docs]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"共 {len(topics)} 个主题，并发 {args.concurrent}")
    done_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as ex:
        futures = {ex.submit(generate_one, t): t for t in topics}

        for fut in concurrent.futures.as_completed(futures):
            try:
                text = fut.result(timeout=100)
                if text:
                    with open(output_path, "a", encoding="utf-8") as f:
                        f.write(text + "\n")
                    done_count += 1
                    print(f"  ✓ 已生成 {done_count} 篇", end="\r", flush=True)
            except Exception:
                pass

    print(f"\n完成！共生成 {done_count} 篇，保存至 {output_path}")
    print(f"总字符: {sum(len(l) for l in output_path.read_text(encoding='utf-8').splitlines())}")


if __name__ == "__main__":
    main()
