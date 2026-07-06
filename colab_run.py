"""
Google Colab 一键跑预训练 —— 用法：

1. 在 Colab 中: 运行 -> 运行时类型 -> T4 GPU
2. 执行以下命令（每个代码块一个单元格）：
"""

# ====== 单元格 1：安装依赖 ======
"""
!pip install torch tokenizers tqdm numpy sentencepiece
!pip install requests beautifulsoup4 datasets
"""

# ====== 单元格 2：挂载 Google Drive ======
"""
from google.colab import drive
drive.mount('/content/drive')
"""

# ====== 单元格 3：复制项目和数据到 Colab ======
"""
import os, shutil
# 如果之前已经复制过，跳过
if not os.path.exists('/content/prompt-optimizer'):
    # 从 Google Drive 复制项目包
    # 先把 prompt-optimizer 整个目录上传到你的 Google Drive 的根目录
    shutil.copytree('/content/drive/MyDrive/prompt-optimizer',
                    '/content/prompt-optimizer')
    print("Project copied to Colab")
os.chdir('/content/prompt-optimizer')
print(f"Working dir: {os.getcwd()}")
!ls -la data/
"""

# ====== 单元格 4：从 HuggingFace 镜像拉中文百科 ======
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
from tqdm import tqdm

# 检查是否已有语料
corpus_path = 'data/full_corpus.txt'
if not os.path.exists(corpus_path):
    print("Downloading Chinese Wikipedia...")
    ds = load_dataset('wikimedia/wikipedia', '20231101.zh',
                      split='train', streaming=True)

    with open('data/wiki_cn.txt', 'w', encoding='utf-8') as f:
        count = 0
        for example in tqdm(ds):
            text = example.get('text', '')
            if text and len(text.strip()) > 100:
                line = text.replace('\\n', ' ').replace('\\r', ' ').strip()
                f.write(line + '\\n')
                count += 1
            if count >= 3000:
                break
    print(f"Downloaded {count} articles")

    # 加载已有领域语料（如果有）
    domain_path = 'data/domain_corpus.txt'
    if os.path.exists(domain_path):
        !cat data/domain_corpus.txt data/wiki_cn.txt > data/full_corpus.txt
        print("Merged with domain corpus")
    else:
        !cp data/wiki_cn.txt data/full_corpus.txt

    # 清洗
    from data.clean import clean_corpus
    from pathlib import Path
    clean_corpus(
        input_path=Path('data/full_corpus.txt'),
        output_path=Path('data/full_corpus_clean.txt'),
        min_length=100,
    )
    !mv data/full_corpus_clean.txt data/full_corpus.txt
else:
    print(f"Corpus already exists ({os.path.getsize(corpus_path)//1024//1024}MB)")
"""

# ====== 单元格 5：训练 Tokenizer（如果不存在） ======
"""
import os
if not os.path.exists('tokenizer/prompt_opt_tokenizer.json'):
    !python tokenizer/train_tokenizer.py \\
        --corpus-path data/full_corpus.txt \\
        --output tokenizer/prompt_opt_tokenizer.json \\
        --vocab-size 8000
else:
    print("Tokenizer already exists")
"""

# ====== 单元格 6：预训练（关键！T4 GPU 加速） ======
"""
# 检查 GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
!echo "---"
!python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

# 运行预训练（T4 GPU 上约 10 分钟）
!python train/trainer.py \\
    --tokenizer tokenizer/prompt_opt_tokenizer.json \\
    --train-corpus data/full_corpus.txt \\
    --max-steps 5000 \\
    --batch-size 16 \\
    --grad-accum-steps 4 \\
    --lr 3e-4 \\
    --warmup-steps 200 \\
    --output-dir checkpoints \\
    --log-dir runs \\
    --log-interval 50 \\
    --eval-interval 500 \\
    --save-interval 1000 \\
    --device cuda
"""

# ====== 单元格 7：将训练好的 checkpoint 复制回 Google Drive ======
"""
import os, shutil
output_dir = '/content/drive/MyDrive/prompt-optimizer/checkpoints'
os.makedirs(output_dir, exist_ok=True)
for f in os.listdir('checkpoints'):
    shutil.copy(f'checkpoints/{f}', f'{output_dir}/{f}')
print(f"Checkpoints saved to Google Drive: {output_dir}")
!ls -lh {output_dir}
"""
