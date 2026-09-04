
import json
import os
import random
import re
import urllib.request

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from s2_model import GPTModel, BASE_CONFIG, model_configs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
JUDGE_MODEL = "qwen2.5vl:7b"

def judge_zhuangzi(question, answer):
    """让裁判给回答打分（0-10 整数）。

    判断标准：
      - 不刻意堆砌辞藻（是否真诚朴素、不空洞）
      - 与问题息息相关（是否切题、不答非所问）

    为什么用这个标准而不是「像不像庄子」？
    小模型经过 SFT 后，采样出来的回答表面都「像庄子腔」，
    用「像不像庄子」打分会全部挤在 7-8 分 → 组内没差异 → RL 学不动。
    但「切不切题」「有没有堆砌辞藻」是有相对对错的，
    答非所问、空洞堆砌会被打低分 → 分数拉开 → RL 才有信号。
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

tokenizer = AutoTokenizer.from_pretrained(os.path.join(BASE_DIR, "base_tok"))
tokenizer.pad_token = "[PAD]"
tokenizer.eos_token = "[unused1]"
tokenizer.bos_token = "[unused1]"

cfg = BASE_CONFIG.copy()
cfg.update(model_configs["gpt2-small (124M)"])
cfg["vocab_size"] = tokenizer.vocab_size
cfg["context_length"] = 1024

policy = GPTModel(cfg)
policy.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "zhuangzi-sft.pth"), map_location="cpu", weights_only=True
))
policy.to(device)

ref = GPTModel(cfg)
ref.load_state_dict(torch.load(
    os.path.join(BASE_DIR, "zhuangzi-sft.pth"), map_location="cpu", weights_only=True
))
ref.to(device)
ref.eval()
for p in ref.parameters():
    p.requires_grad = False

def build_prompt(question):
    return f"### 问：{question}\n### 答："

# ---------------- 采样：只记 token 序列（no_grad，不记 logprob） ----------------
def sample_answers(model, question, G=4, max_new_tokens=60, temperature=1.0, top_p=0.95):
    """对一个问题采样 G 个回答。

    返回 (answer_texts, seqs)：
      answer_texts = 解码出的回答文本（给裁判打分用）
      seqs         = 完整 token 序列 (prompt+回答)，训练时重算 logprob 用
    """
    prompt_text = build_prompt(question)
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt_text, add_special_tokens=False)]
    ).to(device)
    answer_texts, seqs = [], []
    prompt_len = prompt_ids.shape[1]

    model.eval()
    with torch.no_grad():
        for _ in range(G):
            ids = prompt_ids.clone()
            for _ in range(max_new_tokens):
                logits = model(ids[:, -cfg["context_length"]:])[:, -1, :] / temperature
                probs = F.softmax(logits, dim=-1)

                sorted_p, sorted_i = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_p, dim=-1)
                nz = (cum > top_p).nonzero(as_tuple=False)
                k = int(nz[0, 1].item()) + 1 if nz.numel() > 0 else 50
                sp, si = sorted_p[:, :k], sorted_i[:, :k]
                sp = sp / sp.sum(dim=-1, keepdim=True)
                nxt = si[0, torch.multinomial(sp[0], 1).item()].reshape(1, 1)
                ids = torch.cat([ids, nxt], dim=1)
                if nxt.item() == tokenizer.eos_token_id:
                    break

            resp_ids = ids[0, prompt_len:]
            text = tokenizer.decode(resp_ids, skip_special_tokens=True)

            for junk in ("[unused1]", "[PAD]", "[unused0]", "[SEP]", "[CLS]"):
                text = text.replace(junk, "")
            answer_texts.append(text.strip())
            seqs.append(ids[0].clone())
    return answer_texts, seqs

def _logp_sum(model, seq, prompt_len, grad):
    """算一个回答序列里「回答部分」的 logprob 之和。

    seq        = 完整 token 序列 (1D tensor)
    prompt_len = prompt 的 token 数 → 从这之后才是回答
    grad       = True 时保留梯度（训练用）；False 时只在 no_grad 下算（判 KL/奖用）
    原理：模型在位置 t 预测 seq[t+1]。
          回答首 token 是 seq[prompt_len]，由 logits[prompt_len-1] 预测；
          回答尾 token 是 seq[L-1]，由 logits[L-2] 预测。
    """
    L = seq.numel()
    if L <= prompt_len + 1:
        return torch.tensor(0.0, device=seq.device)
    logits = model(seq.unsqueeze(0))
    logp = F.log_softmax(logits, dim=-1)
    pos = torch.arange(prompt_len - 1, L - 1, device=seq.device)
    tok = seq[pos + 1]
    return logp[0, pos, tok].sum()

# ---------------- 一次 GRPO 更新 ----------------
def grpo_step(optimizer, questions, G=8, kl_coef=0.05, temperature=1.0, top_p=0.95,
              max_new_tokens=50):
    """一次 GRPO 更新：对每题采样 G 个回答→裁判打分→组内优势→策略梯度。

    返回 (loss, 平均优势, 平均奖励)。
    """
    total_loss = 0.0
    n_seqs = 0
    adv_all, reward_all = [], []

    for q in questions:
        prompt_len = len(tokenizer.encode(build_prompt(q), add_special_tokens=False))
        answer_texts, seqs = sample_answers(policy, q, G=G, max_new_tokens=max_new_tokens,
                                            temperature=temperature, top_p=top_p)

        rewards = [judge_zhuangzi(q, a) for a in answer_texts]
        r = torch.tensor(rewards, dtype=torch.float, device=device)

        kl_terms = []
        for seq in seqs:
            with torch.no_grad():
                pol_lp = _logp_sum(policy, seq, prompt_len, grad=False)
                ref_lp = _logp_sum(ref, seq, prompt_len, grad=False)
            n_resp = max(1, seq.numel() - prompt_len)
            kl_terms.append((pol_lp - ref_lp).item() / n_resp)
        kl_t = torch.tensor(kl_terms, device=device)
        eff_r = r - kl_coef * kl_t

        mean, std = eff_r.mean(), eff_r.std()
        if std < 1e-6:

            continue
        adv = (eff_r - mean) / (std + 1e-8)

        policy.train()
        for i, seq in enumerate(seqs):
            pol_lp = _logp_sum(policy, seq, prompt_len, grad=True)
            n_resp = max(1, seq.numel() - prompt_len)
            loss_i = -(adv[i].detach() * pol_lp) / n_resp
            total_loss = total_loss + loss_i
            n_seqs += 1
        adv_all.append(adv.mean().item())
        reward_all.append(r.mean().item())

    if n_seqs == 0:
        return 0.0, 0.0, 0.0

    total_loss = total_loss / n_seqs
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    return total_loss.item(), sum(adv_all) / len(adv_all), sum(reward_all) / len(reward_all)

PROMPTS = [

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

    "如果全世界只剩下你一个人，还有善恶吗？",
    "你讲的那些道理，自己真的做到过吗？",
    "我总觉得你说的「顺其自然」就是认命、就是躺平的借口。",
    "如果人人都学庄子躺平，社会不就倒退了吗？",
    "你整天讲无用之用，可饭还是要吃的，钱还是要赚的。",
    "你现在说的这些话，明天会不会自己也推翻？",
    "如果蝴蝶能说话，它会怎么反驳你？",
    "天道如果真的公正，为什么坏人过得比好人好？",
    "你叫我别争，可你不也在写书争一个道理吗？",
]

HARD_PROMPTS = PROMPTS[16:]

def main():
    optimizer = torch.optim.AdamW(policy.parameters(), lr=5e-6)
    num_steps = 8
    G = 6
    questions_per_step = 3
    max_new_tokens = 40

    print(f"开始 RLAIF：{num_steps} 步 × 每步 {questions_per_step} 题 × 每题 G={G}，裁判={JUDGE_MODEL}")
    print(f"只跑 {len(HARD_PROMPTS)} 条刁钻题（普通题没信号），采样 {max_new_tokens} token\n")

    for step in range(num_steps):
        random.seed(step)
        qs = random.sample(HARD_PROMPTS, min(questions_per_step, len(HARD_PROMPTS)))
        loss, adv_mean, r_mean = grpo_step(
            optimizer, qs, G=G, kl_coef=0.05, temperature=1.0, top_p=0.95,
            max_new_tokens=max_new_tokens,
        )

        if step % 5 == 0 or step == num_steps - 1:
            demo_q = PROMPTS[step % len(PROMPTS)]
            ans, _ = sample_answers(policy, demo_q, G=1, temperature=0.9, top_p=0.95)
            print(f"[step {step:3d}] loss={loss:.3f} | 平均优势={adv_mean:+.2f} | 平均裁判分={r_mean:.1f}")
            print(f"   问：{demo_q}")
            print(f"   答：{ans[0][:70]}")
            print()

    out = os.path.join(BASE_DIR, "zhuangzi-rlaif.pth")
    torch.save(policy.state_dict(), out)
    print(f"✅ RLAIF 完成，权重已保存: {out}")
    print("想对比效果：把 run6 里的 zhuangzi-sft.pth 改成 zhuangzi-rlaif.pth 再跑。")

if __name__ == "__main__":
    main()
