<div align="center">

# Zhuangzi LLM · 庄子对话模型

**A Zhuangzi-style dialogue model built by hand-writing a GPT-2 from scratch, plugging in a Chinese backbone, and fine-tuning with 554 Zhuangzi Q&A samples.**

*Hand-written Transformer → Chinese GPT-2 (uer) → SFT → hand-written GRPO/RLAIF → held-out evaluation*

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
| **Evaluation** | `run4_eval.py` | Held-out eval: SFT vs RLAIF on questions the model never trained on |

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

# 5. (optional) held-out evaluation, SFT vs RLAIF
python run4_eval.py       # prints per-group Δ and saves eval_results.json (git-ignored)
```

> ⚠️ The fine-tuned weights (`.pth`, ~500 MB) are **not** committed (GitHub's 100 MB limit). Run `run1_sft.py` to regenerate them — the code and data are complete.

### What the evaluation showed

`run4_eval.py` compares SFT vs RLAIF on questions **neither model trained on**, same seed per question so sampling noise doesn't mask the real difference:

- **A — 16 everyday questions** (regression: did RLAIF break what SFT already did?)
- **B — 6 new, harder philosophical questions** (effect: did RLAIF get better at the hard kind it was actually trained for?)

Judge criteria (identical to training): does the answer actually address the question, and is it plain rather than piling up stock allusions to cover emptiness?

| group | SFT | RLAIF | Δ |
|---|---|---|---|
| A · everyday (16) | 4.19 | 4.88 | **+0.69** |
| B · unseen hard (6) | 5.50 | 6.00 | **+0.50** |

Both groups improved — including on questions never seen in RL training — so the gain is **generalization, not memorization** of the 9 training prompts. The small Δ also means RL nudged a weak model rather than transformed it.

### Honest notes

This project was built to **understand deep learning from first principles**, not to ship a demo. What it actually taught:

1. **Why you don't re-pretrain a language model.** The Zhuangzi text (~17k chars) is a drop in the ocean next to the trillion-token scale real models need. Use a pre-trained backbone and fine-tune the *style*.

2. **`uer`'s "GPT-2" is a BERT tokenizer.** Vocab 21128, all special tokens empty — you must set `[PAD]` / `[unused1]` yourself or the training loop breaks. Small things that cost an afternoon.

3. **My first RL run looked like a wall — it was a misdiagnosis.** I first scored answers by *"how much this sounds like Zhuangzi"*. After SFT everything already sounds like Zhuangzi, so scores bunched at 7–8, GRPO advantages collapsed to ~0, and — running only 8 steps — I concluded RL was hopeless for aesthetic tasks. Two things were wrong. The reward had **no discrimination**: once style saturates, an aesthetic rubric can't separate good from better. Switching the judge to criteria that *do* have relative right/wrong (does it answer the question; is it plain or padding with stock allusions) gave the signal back. And 8 gradient steps at lr 5e-6 barely moves a 124M model. At 100 steps with the discriminating judge, held-out scores improved on both groups. The real lesson: RL needs a reward with signal, enough steps to learn, and an evaluation on questions the model never trained on.

4. **102M + subjective style = an honest ceiling.** Absolute scores stayed low (~4–5/10 on everyday answers). The model mimics Zhuangzi's *tone* — reusing his aphorisms, replying in aphorisms — but can't reach his *philosophical depth* on adversarial questions. Tone is a surface the backbone already provides; depth needs a bigger model. RL made a weak model a little better; it didn't make it wise.

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
| **评测** | `run4_eval.py` | Held-out 评测：SFT vs RLAIF，在模型没训过的题上对比 |

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

# 5.（可选）held-out 评测：SFT vs RLAIF
python run4_eval.py       # 打印两组 Δ，并存 eval_results.json（已 gitignore）
```

> ⚠️ 微调权重（.pth，约 500MB）**未提交**（GitHub 单文件 100MB 上限）。代码和数据是完整的，跑 `run1_sft.py` 即可重新生成。

### 评测结果说明

`run4_eval.py` 在**两个模型都没训过的题**上对比 SFT 和 RLAIF，同一道题用同一 seed 采样，免得采样噪声盖过真实差异：

- **A 组 — 16 道日常题**（回归测试：RLAIF 有没有把 SFT 已会的搞坏）
- **B 组 — 6 道新刁钻哲学题**（效果测试：RLAIF 在它真正练的那类难题上有没有变强）

裁判标准（与训练时一致）：回答是否真的在回应问题、是否朴素实在而非堆砌典故掩盖空洞。

| 组 | SFT | RLAIF | Δ |
|---|---|---|---|
| A · 日常（16） | 4.19 | 4.88 | **+0.69** |
| B · 新刁钻（6） | 5.50 | 6.00 | **+0.50** |

两组都涨了——包括 RL 从没训过的题——说明增益是**泛化**，不是把那 9 道训练题背下来了。Δ 不大也说明：RL 把一个弱模型推好了一点，不是脱胎换骨。

### 诚实记录：这个项目教了我什么

做这个项目是为了**从第一性原理理解深度学习**，而不只是交付一个 demo。几个关键认知：

1. **为什么不能自己重新预训练**：庄子原文约 1.7 万字，对预训练是"一滴水填海"（真实模型用万亿 token）。要用现成底座，只微调**风格**。

2. **uer 的"GPT-2"其实是 BERT 分词器**：词表 21128，特殊 token 全空，必须手动设 `[PAD]`/`[unused1]`，否则训练直接崩。都是些能让你搭进去一下午的小坑。

3. **我第一次 RL 撞的墙，其实是误诊**：最初裁判按"像不像庄子"打分——可 SFT 之后采样出来都像庄子，分数全挤在 7-8，GRPO 优势坍缩到 0；再加只跑了 8 步，就下了"审美类任务 RL 没救"的结论。错在两处。一是 reward 没有区分度：风格饱和后，"美不美"分不出高下；把裁判换成有相对对错的标准（答没答到点上、是朴素还是拿典故堆空洞），信号就回来了。二是 8 步、lr 5e-6，对一个 124M 模型根本挪不动。换成区分度够的裁判、跑够 100 步，held-out 两组分都涨了。真正的教训：RL 要带信号的 reward、足够的步数，而且得用模型没训过的题来验收。

4. **102M + 主观风格 = 诚实的上限**：绝对分仍然低（日常题约 4-5 分）。它学到了庄子的"腔调"——复用他的格言、用格言式句子回应——但对刁钻问题的哲学深度够不着。腔调是底座本来就会的表层；深度要更大的模型。RL 让一个弱模型好了一点，没让它变聪明。
