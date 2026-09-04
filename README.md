<div align="center">

# Zhuangzi LLM · 庄子对话模型

**A Zhuangzi-style dialogue model built by hand-writing a GPT-2 from scratch, plugging in a Chinese backbone, and fine-tuning with 554 Zhuangzi Q&A samples.**

*Hand-written Transformer → Chinese GPT-2 (uer) → SFT → hand-written GRPO/RLAIF*

</div>

---

## 📖 English

### What is this?

This project fine-tunes a **GPT-2 architecture from scratch** into a chatbot that answers in the voice and worldview of **Zhuangzi (庄子)**, the ancient Chinese Daoist philosopher.

The unusual part: the Transformer is **not** loaded from a framework — it is **hand-written** (~700 lines: attention, layers, model, pretraining loop, weight loader), then connected to a **Chinese pre-trained backbone** ([`uer/gpt2-chinese-cluecorpussmall`](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall), 102M, BERT-style tokenizer) because the original English GPT-2 cannot tokenize Chinese.

### What's inside

| Component | File | What it does |
|---|---|---|
| Multi-head attention | `s1_attention.py` | Causal self-attention from scratch |
| GPT-2 model | `s2_model.py` | 12-layer / 768-dim Transformer, vocab 21128 |
| Text generation | `s3_generate.py` | top-k/top-p sampling, eos handling |
| Training loop | `s4_pretraining.py` | `train_model_simple` — cross-entropy trainer |
| Load Chinese backbone | `s5_load_uer.py` | Download/load the uer Chinese GPT-2 weights |
| Weight loader | `s6_load_weights.py` | Map uer's flat HF keys into the hand-written model |
| Dataset | `s7_dataset.py` | Completion-only SFT collate (labels only on the answer) |
| **SFT fine-tune** | `run1_sft.py` | Supervised fine-tuning on Zhuangzi Q&A |
| **RLAIF** | `run2_rlaif.py` | Hand-written GRPO policy-gradient refinement |
| **Chat** | `run3_chat.py` | Talk to the trained model |

### Data

`data/data.jsonl` — **554 hand-curated samples** in 3 types:

| type | count | meaning |
|---|---|---|
| `style` | 242 | Original sentences from the *Inner Chapters* of the Zhuangzi (one per answer) |
| `identity` | 162 | Western-philosophy questions (self/existence/death/freedom/God…) answered in Zhuangzi's voice |
| `chat` | 150 | Everyday worries (friends/family/exams/love) answered with Zhuangzi-style thinking |

### Reproduce

```bash
# 1. setup (Python 3.10 + CUDA 12.8)
conda create -n zhuangzi python=3.10 -y
pip install torch transformers datasets accelerate   # use your CUDA wheel source

# 2. fine-tune (SFT)
python run1_sft.py        # → zhuangzi-sft.pth

# 3. chat
python run3_chat.py       # talk to your Zhuangzi

# 4. (optional) RLAIF with hand-written GRPO
#    needs a local Ollama judge model:
#    ollama pull qwen2.5vl:7b
python run2_rlaif.py      # → zhuangzi-rlaif.pth
```

> ⚠️ The fine-tuned weights (`.pth`, ~500 MB) are **not** committed (GitHub's 100 MB limit). Run `run1_sft.py` to regenerate them — the code and data are complete.

### Honest notes / lessons learned

This project was built to **understand deep learning from first principles**, not just to ship a demo. Key things it taught:

1. **Why you don't re-pretrain a language model.** Pretraining the Zhuangzi text (~17k chars) is a drop in the ocean vs. the trillion-token scale real models need. Use a pre-trained backbone; fine-tune the *style* instead.
2. **`uer`'s "GPT-2" is actually a BERT tokenizer.** Vocab 21128, all special tokens empty — must manually set `[PAD]` / `[unused1]` or `trl` crashes.
3. **Small models plateau on "aesthetic" RL.** After SFT, samples all *look* like Zhuangzi, so a judge scores them ~equal → GRPO advantages collapse to ~0 → RL can't learn. RL needs *verifiable* signals (math, format), not taste.
4. **A 102M model + subjective style = honest ceiling.** It mimics the *tone* but can't reach Zhuangzi's philosophical *depth* on adversarial questions.

### Tech stack

`Python · PyTorch · transformers · uer/gpt2-chinese · GPT-2 architecture · SFT · GRPO (hand-written) · Ollama`

---

## 📖 中文

### 这是什么？

把一个**手写的 GPT-2** 接到中文底座上，用 **554 条庄子问答**微调，做出一个**用庄子腔调说话**的对话模型。

特别之处：Transformer 不是调库，是**手写**的（注意力、层、模型、预训练循环、权重加载器约 700 行）。然后接上**中文预训练底座** [`uer/gpt2-chinese-cluecorpussmall`](https://huggingface.co/uer/gpt2-chinese-cluecorpussmall)（102M，BERT 式分词）——因为英文 GPT-2 的分词器不会切中文。

### 项目结构

| 模块 | 文件 | 作用 |
|---|---|---|
| 多头注意力 | `s1_attention.py` | 从零写因果自注意力 |
| GPT-2 模型 | `s2_model.py` | 12 层 / 768 维，词表 21128 |
| 文本生成 | `s3_generate.py` | top-k/top-p 采样 |
| 训练循环 | `s4_pretraining.py` | `train_model_simple` 交叉熵训练器 |
| 加载中文底座 | `s5_load_uer.py` | 下载/加载 uer 中文 GPT-2 权重 |
| 权重加载器 | `s6_load_weights.py` | 把 uer 的扁平 HF 键名灌进手写模型 |
| 数据加载 | `s7_dataset.py` | 只对答案段算 loss 的 SFT collate |
| **SFT 微调** | `run1_sft.py` | 在庄子问答上做监督微调 |
| **RLAIF** | `run2_rlaif.py` | 手写 GRPO 策略梯度精修 |
| **对话** | `run3_chat.py` | 跟训好的模型聊天 |

### 数据

`data/data.jsonl` — **554 条人工把关数据**，三类：

| type | 条数 | 含义 |
|---|---|---|
| `style` | 242 | 庄子内篇原文（逐句一条） |
| `identity` | 162 | 西哲根本问题（我/存在/死亡/自由/神…）×庄子式文言回答 |
| `chat` | 150 | 日常烦恼（朋友/家人/考试/恋爱）×庄子式思维化解 |

### 复现步骤

```bash
# 1. 环境（Python 3.10 + CUDA 12.8）
conda create -n zhuangzi python=3.10 -y
pip install torch transformers datasets accelerate   # 按你的 CUDA 换源

# 2. 微调（SFT）
python run1_sft.py        # → zhuangzi-sft.pth

# 3. 对话
python run3_chat.py       # 跟你的庄子聊两句

# 4.（可选）RLAIF 手写 GRPO
#    需要本地 Ollama 当裁判：
#    ollama pull qwen2.5vl:7b
python run2_rlaif.py      # → zhuangzi-rlaif.pth
```

> ⚠️ 微调权重（.pth，约 500MB）**未提交**（GitHub 单文件 100MB 上限）。代码和数据是完整的，跑 `run1_sft.py` 即可重新生成。

### 诚实记录：这个项目教了我什么

做这个项目是为了**从第一性原理理解深度学习**，而不只是交付一个 demo。几个关键认知：

1. **为什么不能自己重新预训练**：庄子原文约 1.7 万字，对预训练是"一滴水填海"（真实模型用万亿 token）。要用现成底座，只微调**风格**。
2. **uer 的"GPT-2"其实是 BERT 分词器**：词表 21128，特殊 token 全空，必须手动设 `[PAD]`/`[unused1]`，否则 trl 直接崩。
3. **小模型在"审美"型 RL 上会撞墙**：SFT 之后采样结果"表面都像庄子"，裁判打分成片趋同 → GRPO 优势坍缩到 0 → RL 学不动。RL 需要**可验证的信号**（数学对错、格式），不是口味。
4. **102M + 主观风格 = 诚实的上限**：它学到了庄子的"腔调"，但对刁钻问题的哲学深度够不着。

### 技术栈

`Python · PyTorch · transformers · uer/gpt2-chinese · GPT-2 架构 · SFT · GRPO(手写) · Ollama`

---

## 📜 License

MIT (code). The Zhuangzi text extracts are from the public-domain classical text; fine-tuned weights are not included.
