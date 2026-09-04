
import torch

def assign(left, right):
    """把 uer 的 torch.Tensor 包成 nn.Parameter，塞进模型的对应参数。

    left  = 模型里现有的参数（用来核对形状）
    right = uer 传来的权重张量
    形状对不上立刻报错（防手滑写错映射）。
    uer 传进来的已是 torch.Tensor，不用像原版那样 torch.tensor() 再转。
    """
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(right)

def load_weights_into_gpt(gpt, params):
    """把 uer 扁平权重 params 灌进手写 GPTModel gpt。

    形状对照（左=手写模型，右=uer）:
      手写 tok_emb(21128,768)  ← wte(21128,768)
      手写 pos_emb(1024,768)   ← wpe(1024,768)
      手写 out_head(768,21128) ← wte(21128,768) 再 .T（权重共享，tie_word_embeddings）
      每层:
        手写 att.W_query/key/value(768,768) ← c_attn.w(768,2304) 切成3段后各自 .T
        手写 att.out_proj(768,768)          ← c_proj.w(768,768) 再 .T
        手写 ff.layers[0](3072,768)         ← c_fc.w(768,3072) 再 .T
        手写 ff.layers[2](768,3072)         ← c_proj.w(3072,768) 再 .T
        手写 norm.scale(768)                ← ln_*.weight(768,)   ← 注意是 weight 不是 scale
        手写 norm.shift(768)                ← ln_*.bias(768,)
    """

    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["transformer.wte.weight"])

    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["transformer.wpe.weight"])

    n_layers = len([k for k in params if ".attn.c_attn.weight" in k])

    for b in range(n_layers):

        prefix = f"transformer.h.{b}."

        q_w, k_w, v_w = torch.split(
            params[f"{prefix}attn.c_attn.weight"], 768, dim=1
        )

        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = torch.split(
            params[f"{prefix}attn.c_attn.bias"], 768, dim=0
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params[f"{prefix}attn.c_proj.weight"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params[f"{prefix}attn.c_proj.bias"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params[f"{prefix}mlp.c_fc.weight"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params[f"{prefix}mlp.c_fc.bias"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params[f"{prefix}mlp.c_proj.weight"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params[f"{prefix}mlp.c_proj.bias"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params[f"{prefix}ln_1.weight"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params[f"{prefix}ln_1.bias"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params[f"{prefix}ln_2.weight"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params[f"{prefix}ln_2.bias"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["transformer.ln_f.weight"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["transformer.ln_f.bias"])

    gpt.out_head.weight = assign(gpt.out_head.weight, params["transformer.wte.weight"])

    print(f"✅ uer 权重已灌入手写 GPTModel（{n_layers} 层）")
