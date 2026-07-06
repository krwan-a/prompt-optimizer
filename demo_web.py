#!/usr/bin/env python3
"""
Web 演示界面 —— 一键启动，生成公网链接，老师浏览器打开就能用。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from tokenizers import Tokenizer
from model.model import GPT
import gradio as gr
import time


def load_model():
    ckpt_path = "sft_checkpoints/sft-best.pt"
    tok_path = "tokenizer/sft_tokenizer.json"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_file(tok_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    if isinstance(args, dict):
        d_model = args.get("d_model", 256)
        n_layer = args.get("n_layer", 6)
        n_head = args.get("n_head", 8)
        ffn_hidden = args.get("ffn_hidden", 1024)
        max_len = args.get("max_length", 512)
    else:
        d_model, n_layer, n_head, ffn_hidden, max_len = 256, 6, 8, 1024, 512

    model = GPT(vocab_size=tokenizer.get_vocab_size(),
                d_model=d_model, n_layer=n_layer, n_head=n_head,
                ffn_hidden=ffn_hidden, max_seq_len=max_len)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    return model, tokenizer, device, total


model, tokenizer, device, param_count = load_model()


def optimize(rough, temperature, max_tokens, top_k):
    if not rough.strip():
        return "请输入粗糙需求。"
    prompt_text = f"<user>{rough}</user><assistant>"
    encoding = tokenizer.encode(prompt_text)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)
    if input_ids.size(1) > model.max_seq_len:
        input_ids = input_ids[:, -model.max_seq_len:]

    start = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=int(max_tokens),
            temperature=temperature,
            top_k=int(top_k),
        )
    elapsed = time.time() - start

    full = tokenizer.decode(out_ids[0].tolist())
    s = full.find("<assistant>")
    content = full[s + len("<assistant>"):] if s != -1 else full
    e = content.find("</assistant>")
    if e != -1:
        content = content[:e]

    return f"{content}\n\n（{elapsed:.1f}s，{param_count:,} 参数模型）"


with gr.Blocks(title="Prompt Optimizer - 提示词优化") as demo:
    gr.Markdown(f"""
    # 🎓 Prompt Optimizer
    从零训练的 10M 参数 GPT Decoder-Only Transformer

    **参数量**: {param_count:,} | **词表**: 8,005 | **上下文**: 512 tokens

    输入一个粗糙需求，模型会尝试生成精炼的提示词。
    """)
    with gr.Row():
        with gr.Column(scale=2):
            rough_input = gr.Textbox(
                label="粗糙需求",
                placeholder="例如：帮我写个Python脚本读CSV文件",
                lines=3,
            )
            submit_btn = gr.Button("✨ 生成精炼提示词", variant="primary")
        with gr.Column(scale=1):
            temperature = gr.Slider(0.1, 1.5, value=0.8, label="温度")
            max_tokens = gr.Slider(50, 400, value=200, step=10, label="最大生成长度")
            top_k = gr.Slider(1, 50, value=10, step=1, label="Top-K")
    output = gr.Textbox(label="精炼提示词", lines=8, interactive=False)

    submit_btn.click(fn=optimize, inputs=[rough_input, temperature, max_tokens, top_k], outputs=output)
    rough_input.submit(fn=optimize, inputs=[rough_input, temperature, max_tokens, top_k], outputs=output)

    gr.Markdown("""
    ---
    **项目结构**: `data/` 数据收集 · `tokenizer/` 分词器 · `model/` 模型架构 · `train/` 训练循环 · `eval/` 评估
    """)

demo.launch(share=True, server_name="0.0.0.0")
