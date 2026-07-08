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
    # ── 提示词工程基础 ──
    "什么是提示词工程（Prompt Engineering）？请用300字左右解释其核心概念和重要性。",
    "什么是思维链提示（Chain-of-Thought Prompting）？请用300字左右解释原理并给出一个例子。",
    "什么是Few-shot提示？请用300字左右解释其工作原理和适用场景。",
    "什么是Zero-shot提示？与Few-shot有什么区别？请用300字左右说明。",
    "提示词中system prompt和user prompt各自的作用是什么？请用300字左右解释。",
    "写提示词时应该遵循哪些基本原则？请列出5-8条原则并简要解释每条。",
    "什么是角色扮演提示（Role Prompting）？请用300字左右解释并给出一个例子。",
    "什么是思维树提示（Tree-of-Thought Prompting）？请用300字左右解释。",
    "在提示词中如何有效地使用示例？请用300字左右说明最佳实践。",
    "什么是负向提示（Negative Prompting）？请用300字左右解释其用途。",
    "如何编写清晰的指令型提示词？请用300字左右说明关键要素。",
    "提示词中的格式控制（如JSON输出）如何实现？请用300字左右说明。",
    "什么是迭代式提示优化？请用300字左右解释其流程。",
    "如何处理大模型输出中的幻觉问题？请用300字左右说明提示词层面的方法。",
    "什么是上下文长度（Context Window）？在写提示词时需要注意什么？请用300字左右说明。",
    "如何通过提示词控制输出风格和语气？请用300字左右说明方法。",
    "什么是结构化提示？请用300字左右解释并给出一个模板示例。",
    "提示词中的条件逻辑如何实现？请用300字左右说明。",
    "什么是多轮对话中的提示词管理？请用300字左右说明策略。",
    "如何评估一个提示词的质量？请用300字左右列出评估维度和方法。",
    "什么是自动提示优化（Automatic Prompt Optimization）？请用300字左右介绍。",
    "在代码生成任务中如何设计有效的提示词？请用300字左右说明。",
    "在文本总结任务中如何设计有效的提示词？请用300字左右说明。",
    "在翻译任务中如何设计有效的提示词？请用300字左右说明。",
    "在数据分析任务中如何设计有效的提示词？请用300字左右说明。",
    "在创意写作任务中如何设计有效的提示词？请用300字左右说明。",
    "什么是系统1和系统2提示？请用300字左右解释其概念。",
    "如何通过提示词让大模型进行事实核查？请用300字左右说明。",
    "什么是提示链（Prompt Chaining）？请用300字左右解释其应用场景。",
    "什么是自适应提示？请用300字左右解释其概念和优势。",
    "如何处理提示词中的敏感内容？请用300字左右说明最佳实践。",
    "什么是元提示（Meta-Prompting）？请用300字左右解释。",
    "什么是情感提示（Emotional Prompting）？请用300字左右解释其效果。",
    "如何利用提示词让模型更好地理解和遵循约束条件？请用300字左右说明。",
    "在提示词中如何使用分隔符提高清晰度？请用300字左右说明。",
    # ── 实战技巧 ──
    "如何在提示词中处理长文本输入？请用300字左右说明分段和摘要策略。",
    "提示词中的温度参数（temperature）如何影响输出质量？请用300字左右说明。",
    "什么是Top-P和Top-K采样？如何在提示词设计中考虑这些参数？",
    "如何设计提示词让模型输出JSON格式？请用300字左右说明。",
    "如何设计提示词让模型输出表格？请用300字左右说明。",
    "在提示词中如何给出负面约束（不要做什么）？请用300字左右说明。",
    "如何利用提示词让模型一步步思考？请用300字左右说明Step-by-Step技巧。",
    "什么是反思链提示？请用300字左右解释其工作原理。",
    "如何设计提示词来纠正模型的错误输出？请用300字左右说明。",
    "在提示词中使用重复和强调的有效方法是什么？请用300字左右说明。",
    "什么是提示词中的锚定效应？如何利用它改进输出？",
    "如何设计多步骤提示词来处理复杂任务？请用300字左右说明。",
    "提示词中如何处理模糊性和不确定性？请用300字左右说明。",
    "如何设计提示词来控制输出的详细程度？请用300字左右说明。",
    # ── 应用场景 ──
    "如何设计提示词来做情感分析？请用300字左右说明。",
    "如何设计提示词来做文本分类？请用300字左右说明。",
    "如何设计提示词来做命名实体识别？请用300字左右说明。",
    "如何设计提示词来做问答系统？请用300字左右说明。",
    "如何设计提示词来做代码解释？请用300字左右说明。",
    "如何设计提示词来做代码调试？请用300字左右说明。",
    "如何设计提示词来做代码优化？请用300字左右说明。",
    "如何设计提示词来做数据库查询生成？请用300字左右说明。",
    "如何设计提示词来做正则表达式生成？请用300字左右说明。",
    "如何设计提示词来做文案撰写？请用300字左右说明。",
    "如何设计提示词来做邮件撰写？请用300字左右说明。",
    "如何设计提示词来做报告生成？请用300字左右说明。",
    "如何设计提示词来做对话系统？请用300字左右说明。",
    "如何设计提示词来做知识问答？请用300字左右说明。",
    "如何设计提示词来做教育辅导？请用300字左右说明。",
    "如何设计提示词来做头脑风暴？请用300字左右说明。",
    "如何设计提示词来做决策支持？请用300字左右说明。",
    "如何设计提示词来做辩论模拟？请用300字左右说明。",
    "如何设计提示词来做角色扮演？请用300字左右说明。",
    "如何设计提示词来做故事创作？请用300字左右说明。",
    # ── 进阶主题 ──
    "什么是多模态提示词设计？请用300字左右说明。",
    "什么是提示词版本管理？为什么它重要？请用300字左右说明。",
    "如何构建提示词模板库？请用300字左右说明最佳实践。",
    "什么是提示词测试驱动开发？请用300字左右说明。",
    "如何评估提示词的鲁棒性？请用300字左右说明方法。",
    "什么是提示词对抗性攻击？如何防御？请用300字左右说明。",
    "如何设计无偏见的提示词？请用300字左右说明注意事项。",
    "提示词中的多语言设计策略是什么？请用300字左右说明。",
    "如何针对不同大模型（GPT、Claude、Llama）调整提示词策略？",
    "什么是Agent提示词设计？请用300字左右说明其特殊性。",
    "如何设计工具使用（Tool Use）的提示词？请用300字左右说明。",
    "什么是ReAct提示模式？请用300字左右解释。",
    "什么是Plan-and-Solve提示模式？请用300字左右解释。",
    "什么是Self-Consistency提示策略？请用300字左右解释。",
    "什么是Active Prompting？请用300字左右解释其原理。",
    "什么是Directional Stimulus Prompting？请用300字左右解释。",
    "如何设计提示词来处理超长上下文（100K+ tokens）？",
    "如何利用提示词让模型进行自我纠错？请用300字左右说明。",
    "什么是提示词中的角色锚定？请用300字左右解释其效果。",
    "如何设计提示词来模拟专家角色？请用300字左右说明。",
    # ── AI基础概念（帮助模型理解领域上下文）──
    "什么是大语言模型的训练过程？请用300字左右简要说明。",
    "什么是监督学习和强化学习？请用300字左右说明区别。",
    "什么是Transformer架构中的注意力机制？请用300字左右说明。",
    "什么是Token和分词器？请用300字左右说明其作用。",
    "什么是嵌入向量（Embedding）？请用300字左右说明。",
    "什么是微调（Fine-tuning）？与提示词工程有什么关系？",
    "什么是RAG检索增强生成？请用300字左右说明其工作原理。",
    "什么是模型的温度和随机性？如何影响生成结果？",
]

SYSTEM_PROMPT = "你是一个提示词工程(Prompt Engineering)领域的写作助手。请用中文写一段300-500字的说明文字，要求内容准确、语言清晰、结构完整。不要使用markdown格式，直接输出纯文本段落。"


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
                max_tokens=1000,
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
