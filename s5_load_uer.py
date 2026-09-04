

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def download_and_load_gpt2_uer(base_dir="."):
    """下载（或读本地）uer 中文 GPT-2，返回扁平参数 dict。

    优先读本地 base_model/ + base_tok/（已存在则不再联网）；
    没有则从 HuggingFace 下载（国内需 HF_ENDPOINT=https://hf-mirror.com）。

    返回:
        params: {键名: torch.Tensor}，键名 = uer safetensors 原始名字
        tokenizer: 配套的中文 tokenizer（已设好 pad/eos，供生成时用）
    """
    model_dir = os.path.join(base_dir, "base_model")
    tok_dir = os.path.join(base_dir, "base_tok")

    if os.path.isdir(model_dir) and os.path.isdir(tok_dir):
        print(f"本地 uer 底座已存在（{model_dir}），跳过下载。")
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
    else:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        name = "uer/gpt2-chinese-cluecorpussmall"
        print(f"从 HuggingFace 下载 {name} ...")
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
        tokenizer.save_pretrained(tok_dir)
        model.save_pretrained(model_dir)
        print("已保存到本地:", model_dir, tok_dir)

    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[unused1]"
    tokenizer.bos_token = "[unused1]"
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    params = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return params, tokenizer

if __name__ == "__main__":
    params, tok = download_and_load_gpt2_uer()
    print("总参数字典键数:", len(params))
    for k in list(params)[:12]:
        print("  ", k, tuple(params[k].shape))
