# Prompt Optimizer — 从零训练 10M 参数 GPT Decoder-Only Transformer

一个学习项目：从零训练一个约 **1000 万参数**的 GPT 风格 Decoder-Only Transformer，
专门用于**提示词优化**任务（输入粗糙需求 → 输出精炼提示词）。

## 🚀 立即使用（无需训练）

```bash
# 1. 克隆
git clone https://github.com/krwan-a/prompt-optimizer.git
cd prompt-optimizer

# 2. 安装依赖
pip install torch tokenizers tqdm numpy

# 3. 运行交互式提示词优化
python interact.py
```

然后输入粗糙需求（如 `帮我写个Python脚本读CSV`），模型会输出精炼提示词。

也可单次运行：
```bash
python interact.py --input "帮我写个Python脚本读CSV"
```

## 项目结构

```
prompt-optimizer/
├── __init__.py
├── requirements.txt
├── README.md
│
├── data/                       # 🔵 数据收集与清洗（本阶段完成）
│   ├── __init__.py
│   ├── collect_sft.py          # 调用 LLM API 生成 SFT 合成数据
│   ├── collect_pretrain.py     # 从多种来源收集预训练语料
│   └── clean.py                # 文本清洗与去重
│
├── tokenizer/                  # 🟡 分词器（本阶段完成）
│   ├── __init__.py
│   └── train_tokenizer.py      # 训练 BPE tokenizer (vocab_size=8000)
│
├── model/                      # 🟠 模型架构（待实现）
│   └── __init__.py
│
├── train/                      # 🟠 训练（本阶段仅包含数据集类）
│   ├── __init__.py
│   └── dataset.py              # PretrainDataset + SFTDataset + DataLoader
│
└── eval/                       # 🟠 评估（待实现）
    └── __init__.py
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 收集与清洗预训练语料

```bash
# 从 HuggingFace 加载 wikitext 作为预训练语料
python data/collect_pretrain.py \
    --source huggingface \
    --dataset wikitext \
    --split train \
    --text-field text \
    --max-samples 50000 \
    --output data/raw_corpus.txt

# 清洗与去重
python data/clean.py \
    --input data/raw_corpus.txt \
    --output data/clean_corpus.txt \
    --min-length 100 --max-length 50000 \
    --dedup
```

### 3. 生成 SFT 数据

```bash
# 使用 OpenAI API 生成 SFT 数据（需要设置 OPENAI_API_KEY）
python data/collect_sft.py \
    --output data/sft_data.jsonl \
    --api-provider openai \
    --model gpt-4o \
    --task-types code_writing copywriting data_analysis \
    --num-per-task 100 \
    --batch-size 5

# 或使用 Claude
python data/collect_sft.py \
    --api-provider anthropic \
    --model claude-sonnet-4-6 \
    --task-types all \
    --num-per-task 50
```

### 4. 训练分词器

```bash
python tokenizer/train_tokenizer.py \
    --corpus-path data/clean_corpus.txt \
    --output tokenizer/prompt_opt_tokenizer.json \
    --vocab-size 8000
```

### 5. 测试数据集

```bash
# 测试预训练数据集
python train/dataset.py --mode pretrain \
    --tokenizer tokenizer/prompt_opt_tokenizer.json \
    --data data/clean_corpus.txt \
    --max-length 512 --batch-size 4

# 测试 SFT 数据集
python train/dataset.py --mode sft \
    --tokenizer tokenizer/prompt_opt_tokenizer.json \
    --data data/sft_data.jsonl \
    --max-length 512 --batch-size 4
```

## 模块说明

### data/collect_sft.py

调用 LLM API（OpenAI / Anthropic）批量生成 `{"rough_input", "refined_prompt"}` 配对数据。

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--output` | 输出 JSONL 路径 | `data/sft_data.jsonl` |
| `--api-provider` | API 提供商 (`openai` / `anthropic`) | `openai` |
| `--model` | 模型名 | `gpt-4o` |
| `--task-types` | 任务类型列表（支持 `all`） | 全部 12 种 |
| `--num-per-task` | 每种任务生成条数 | 100 |
| `--batch-size` | 每次 API 调用生成条数 | 5 |
| `--language` | 生成语言 (`zh` / `en`) | `zh` |
| `--resume` | 续写模式（统计已有，补足差额） | `False` |

内置 12 种任务类型：`code_writing` `copywriting` `data_analysis` `role_playing`
`translation` `summarization` `brainstorming` `teaching` `creative_writing`
`email_writing` `planning` `debate`

### data/collect_pretrain.py

支持三种数据源：

| Source | 参数 | 说明 |
|---|---|---|
| `huggingface` | `--dataset`, `--split`, `--text-field`, `--max-samples` | 从 HuggingFace Datasets 加载 |
| `local` | `--input-dir`, `--extension` | 读取本地 `.txt` 文件 |
| `web` | `--urls-file`, `--max-pages` | 从 URL 列表抓取网页正文 |

### data/clean.py

清洗管线：Unicode 标准化 → 可选 HTML 剥离 → URL 移除 → 长度过滤 → 重复率过滤 → 精确去重 → 可选近似去重。

### tokenizer/train_tokenizer.py

基于 HuggingFace `tokenizers` 库训练 **ByteLevel BPE** tokenizer。

- vocab_size=8000（可根据需求调整）
- 中英混合，自动处理 Unicode
- 特殊 tokens: `[PAD]` `[UNK]` `[BOS]` `[EOS]`
- 训练完成后会进行一系列编码/解码测试

### train/dataset.py

两个 Dataset 类 + DataLoader 工厂：

**PretrainDataset**
- 读取整个语料 → tokenize → 滑窗切分（512 token，重叠 50%）
- 返回 `input_ids` 和 `labels`（标准 autoregressive LM）

**SFTDataset**
- 读取 JSONL → 格式化为 `[INST] {rough} [/INST]\n{refined}`
- 通过 **character offset** 精确定位 response 起始 token
- 非 response 位置 label = `-100`（不参与 loss 计算）
- 超长时从 input 侧截断以尽量保留 response

**DataLoader**
- `create_dataloader()` 工厂函数，自动选 collate：
  - Pretrain: 等长序列直接 stack
  - SFT: padding 到 batch 内最大长度，生成 `attention_mask`

## Pipeline 工作流

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ 预训练语料收集 │───→│ 清洗与去重     │───→│ 训练 BPE Tokenizer │
│ (huggingface │    │ (clean.py)   │    │ (vocab_size=8k)  │
│  /local/web)  │    └──────────────┘    └─────────────────┘
└─────────────┘                               │
                                              ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ SFT 数据生成   │───→│ 数据集类      │───→│ 模型训练（TODO）   │
│ (API调用)     │    │ (dataset.py) │    │                 │
└─────────────┘    └──────────────┘    └─────────────────┘
```

## License

MIT
