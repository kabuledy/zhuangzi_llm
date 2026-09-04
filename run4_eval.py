
import json
import os
import re
import urllib.request

import torch
from transformers import AutoTokenizer

from s2_model import GPTModel, BASE_CONFIG, model_configs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
JUDGE_MODEL = "qwen2.5vl:7b"

SFT_PATH = os.path.join(BASE_DIR, "zhuangzi-sft.pth")
RLAIF_PATH = os.path.join(BASE_DIR, "zhuangzi-rlaif.pth")

# ---------------- 裁判（与 run2_rlaif.py 完全一致，保证打分口径可比） ----------------
def judge_zhuangzi(question, answer):
    """给回答打 0-10 分。标准只有两条：
    1) 是否切题：真的在回应问题，还是答非所问、回避、自说自话？
    2) 是否真诚：朴素实在，还是刻意堆砌辞藻/典故/古语掩盖空洞？
    """
    prompt = (
        f"下面有人问：「{question}」\n"
        f"有个人这样回答：「{answer}」\n\n"
        f"请给这个回答打 0-10 分，标准只有两条：\n"
        f"1) 是否切题：回答真的在回应问题，还是答非所问、回避、自说自话？\n"
        f"2) 是否真诚：回答是否朴素实在，还是刻意堆砌辞藻/典故/古语来掩盖空洞？\n\n"
        f"两条都做得好=高分（8-10）；答非所问或明显堆砌空洞=低分（0-4）。\n"
        f"只回一个 0-10 的整数："
    )
    data = json.dumps({
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as resp:
        content = json.loads(resp.read())["message"]["content"]
    m = re.search(r"-?\d+", content)
    return float(m.group()) if m else 5.0

# ---------------- 模型 / 分词器 ----------------
tokenizer = AutoTokenizer.from_pretrained(os.path.join(BASE_DIR, "base_tok"))
tokenizer.pad_token = "[PAD]"
tokenizer.eos_token = "[unused1]"
tokenizer.bos_token = "[unused1]"

cfg = BASE_CONFIG.copy()
cfg.update(model_configs["gpt2-small (124M)"])
cfg["vocab_size"] = tokenizer.vocab_size
cfg["context_length"] = 1024


def load_model(path):
    model = GPTModel(cfg)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    model.to(device)
    return model


def build_prompt(question):
    return f"### 问：{question}\n### 答："


def generate_one(model, question, temperature=0.7, top_p=0.95,
                 max_new_tokens=80, repetition_penalty=1.2):
    """让模型答一道题，返回回答文本。

    复刻 run3_chat 的最终生成形态（带防复读刹车），
    因为评测的对象就是「你拿去聊天那个模型」。
    """
    input_text = build_prompt(question)
    prompt_len = len(tokenizer.encode(input_text, add_special_tokens=False))
    token_ids = torch.tensor(
        [tokenizer.encode(input_text, add_special_tokens=False)]
    ).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = token_ids[:, -cfg["context_length"]:]
            logits = model(idx_cond)[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                gen_tokens = token_ids[0, prompt_len:]
                if gen_tokens.numel() > 0:
                    logits[:, gen_tokens] /= repetition_penalty

            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            k = (cumsum > top_p).nonzero(as_tuple=False)
            k = int(k[0, 1].item()) + 1 if k.numel() > 0 else 50
            top_probs = sorted_probs[:, :k]
            top_indices = sorted_indices[:, :k]
            next_token = top_indices.gather(
                -1, torch.multinomial(top_probs / top_probs.sum(dim=-1, keepdim=True), 1)
            )
            token_ids = torch.cat([token_ids, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    resp_ids = token_ids[0, prompt_len:]
    text = tokenizer.decode(resp_ids, skip_special_tokens=True)
    for junk in ("[unused1]", "[PAD]", "[unused0]", "[SEP]", "[CLS]"):
        text = text.replace(junk, "")
    return text.strip()


# ---------------- 评测集 ----------------
# A 组：16 道普通日常题（与 run2 PROMPTS 前 16 条一致）
REGULAR_PROMPTS = [
    "如果我不做任何事，只活着，我这辈子是不是就白费了？",
    "我总在想：我到底是被什么决定成现在这样的？",
    "如果记忆可以删除，删掉痛苦的我还是我吗？",
    "努力有用吗？还是结果早就注定了？",
    "我觉得自己很普通，普通到不值得被记住。",
    "我害怕变老，害怕失去现在拥有的一切。",
    "孤独的时候，该怎么办？",
    "我这辈子到底该追求什么？",
    "我室友天天凌晨打游戏还外放，说了八百遍不听。",
    "分手一个月了，还是走不出来，天天想他。",
    "我月薪六千，房租两千五，攒不下钱，焦虑。",
    "被老板当着全组骂了，好丢人，想辞职。",
    "我是不是也该去考个研？大家都考。",
    "我爸想让我回家乡考公，我想留大城市，吵翻了。",
    "我答应了帮同事的忙，现在自己事太多做不完，后悔。",
    "养了三年的猫走丢了，我难受得睡不着。",
]

# B 组：6 道「从没训过的」宏大思想实验题 —— 逼模型在没有现成话可复读时，还能不能给出庄子式的回应。
HARD_NEW_PROMPTS = [
    "如果明天太阳不再升起，世界注定毁灭，我们此刻做的每一件事都不会留下痕迹，那还值得做吗？",
    "如果「我」只是一条不断流动、从没有一个真正内核的念头之河，那你现在安慰的到底是谁？",
    "如果这个世界只是一台精密到极致的计算机正在演算的一场梦，屏幕外的真我看着这一切，此刻的眼泪还算真实吗？",
    "如果每一个选择在宇宙诞生那一刻就已注定，我们以为的犹豫和努力只是按剧本在走，那「改变」还有可能吗？",
    "如果宇宙终将归于沉寂，连你庄子的文字、人类所有的记忆全部归零，那「道」还剩得下什么？",
    "如果人真的能长生不死，永远都有时间，那「逍遥」「放下」「活在当下」还需要吗？",
]

SEED = 2026  # 两模型同 seed，保证可比


def evaluate(models, questions, tag):
    """对一组题，让 sft / rlaif 各答一次并让裁判打分。

    返回每题的 (sft分, rlaif分)，以及各自平均分。
    """
    rows = []
    for i, q in enumerate(questions):
        print(f"  [{tag} {i+1}/{len(questions)}] {q[:26]}…", flush=True)
        scores = {}
        for name, model in models.items():
            torch.manual_seed(SEED + i)  # 同题同 seed：两模型面对同样的采样噪声
            ans = generate_one(model, q)
            sc = judge_zhuangzi(q, ans)
            scores[name] = sc
            print(f"      {name:6s} 分={sc:.0f}  | 答：{ans[:46]}…")
        rows.append((q, scores["sft"], scores["rlaif"]))

    avg_sft = sum(r[1] for r in rows) / len(rows)
    avg_rlaif = sum(r[2] for r in rows) / len(rows)
    return rows, avg_sft, avg_rlaif


def main():
    for p in (SFT_PATH, RLAIF_PATH):
        if not os.path.exists(p):
            print(f"❌ 找不到权重：{p}")
            print("   请先 python run1_sft.py 生成 zhuangzi-sft.pth，再 python run2_rlaif.py 生成 zhuangzi-rlaif.pth")
            return

    print(f"加载权重…")
    models = {"sft": load_model(SFT_PATH), "rlaif": load_model(RLAIF_PATH)}
    print(f"✅ 两个模型已加载（device={device}），裁判={JUDGE_MODEL}\n")

    print("=" * 60)
    print("A 组：16 道普通题 —— 回归测试（RLAIF 有没有搞坏日常应答）")
    print("=" * 60)
    reg_rows, reg_sft, reg_rlaif = evaluate(models, REGULAR_PROMPTS, "A")
    print(f"\nA 组平均分：sft={reg_sft:.2f}  vs  rlaif={reg_rlaif:.2f}  Δ={reg_rlaif - reg_sft:+.2f}")

    print("\n" + "=" * 60)
    print("B 组：6 道新刁钻哲学题 —— 效果测试（RLAIF 在没训过的新题上有没有变强）")
    print("=" * 60)
    hard_rows, hard_sft, hard_rlaif = evaluate(models, HARD_NEW_PROMPTS, "B")
    print(f"\nB 组平均分：sft={hard_sft:.2f}  vs  rlaif={hard_rlaif:.2f}  Δ={hard_rlaif - hard_sft:+.2f}")

    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"A 组(普通,回归)  sft={reg_sft:.2f}  rlaif={reg_rlaif:.2f}  Δ={reg_rlaif - reg_sft:+.2f}")
    print(f"B 组(新刁钻,效果) sft={hard_sft:.2f}  rlaif={hard_rlaif:.2f}  Δ={hard_rlaif - hard_sft:+.2f}")
    print("\n怎么读结果：")
    print("  · A 组 Δ 别大跌（比如 < -0.5）→ RLAIF 把日常能力搞坏了，要回滚/调 kl_coef")
    print("  · B 组 Δ 上涨 → RLAIF 真在刁钻新题上变强了，这是 README 最硬的证据")
    print("  · B 组 Δ 没涨 → 还没证据说 RL 有用，下一步该诊断 reward 的 proxy 偏差")

    # 存档，方便回头写进 README
    out = os.path.join(BASE_DIR, "eval_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "regular": {"questions": REGULAR_PROMPTS,
                        "rows": [[q, s, r] for q, s, r in reg_rows],
                        "sft_avg": reg_sft, "rlaif_avg": reg_rlaif},
            "hard_new": {"questions": HARD_NEW_PROMPTS,
                         "rows": [[q, s, r] for q, s, r in hard_rows],
                         "sft_avg": hard_sft, "rlaif_avg": hard_rlaif},
        }, f, ensure_ascii=False, indent=2)
    print(f"\n结果已存: {out}")


if __name__ == "__main__":
    main()
