
import os

import torch
from transformers import AutoTokenizer

from s2_model import GPTModel, BASE_CONFIG, model_configs

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(os.path.join(BASE_DIR, "base_tok"))

tokenizer.pad_token = "[PAD]"
tokenizer.eos_token = "[unused1]"
tokenizer.bos_token = "[unused1]"

model_config = BASE_CONFIG.copy()
model_config.update(model_configs["gpt2-small (124M)"])
model_config["vocab_size"] = tokenizer.vocab_size
model_config["context_length"] = 1024

model = GPTModel(model_config)

state_dict = torch.load(
    os.path.join(BASE_DIR, "zhuangzi-sft.pth"),
    map_location=device,
    weights_only=True,
)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

print("✅ 庄子模型已加载，可以和它聊天了（Ctrl+C 退出）")

# ---------------- 3. 把问句包装成训练时的格式 ----------------
def format_question(question):
    """训练时问句格式是 '### 问：...\n### 答：'，生成时用同样格式它才接得上。"""
    return f"### 问：{question}\n### 答："

# ---------------- 4. 生成回答 ----------------
def _clean_response(text):
    """剥掉 eos/pad 标记（[unused1]/[PAD] 不是 transformers 的 special token，
    skip_special_tokens=True 删不掉它们，必须手动剥）。"""
    for junk in ("[unused1]", "[PAD]", "[unused0]", "[SEP]", "[CLS]"):
        text = text.replace(junk, "")
    return text.strip()

def ask_zhuangzi(question, max_new_tokens=120, temperature=0.8):
    """问庄子一个问题，返回它的回答。"""
    input_text = format_question(question)
    prompt_len = len(tokenizer.encode(input_text, add_special_tokens=False))

    token_ids = torch.tensor(
        [tokenizer.encode(input_text, add_special_tokens=False)]
    ).to(device)

    with torch.no_grad():
        for _ in range(max_new_tokens):

            idx_cond = token_ids[:, -model_config["context_length"]:]
            logits = model(idx_cond)[:, -1, :] / temperature

            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)

            k = (cumsum > 0.9).nonzero(as_tuple=False)
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
    response = tokenizer.decode(resp_ids, skip_special_tokens=True)
    return _clean_response(response)

if __name__ == "__main__":

    examples = [
        "我很焦虑，怕自己不够优秀，怎么办？",
        "什么是真正的自由？",
        "人为什么活着？",
    ]
    for q in examples:
        print(f"\n你：{q}")
        print(f"庄子：{ask_zhuangzi(q)}")

    while True:
        try:
            user_input = input("\n你（Ctrl+C 退出）：")
            if not user_input.strip():
                continue
            print(f"庄子：{ask_zhuangzi(user_input.strip())}")
        except (KeyboardInterrupt, EOFError):
            print("\n再见。安时而处顺，哀乐不能入也。")
            break
