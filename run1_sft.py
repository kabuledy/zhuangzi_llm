
import json
import os
from functools import partial

import torch
from torch.utils.data import DataLoader

from s2_model import GPTModel, BASE_CONFIG, model_configs
from s4_pretraining import train_model_simple
from s5_load_uer import download_and_load_gpt2_uer
from s6_load_weights import load_weights_into_gpt
from s7_dataset import (
    InstructionDataset,
    custom_collate_fn,
    format_question,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params, tokenizer = download_and_load_gpt2_uer(BASE_DIR)

model_config = BASE_CONFIG.copy()
model_config.update(model_configs["gpt2-small (124M)"])
model_config["vocab_size"] = tokenizer.vocab_size
model_config["context_length"] = 1024

model = GPTModel(model_config)

load_weights_into_gpt(model, params)
model.to(device)
model.eval()

data = [json.loads(line) for line in open(os.path.join(BASE_DIR, "data.jsonl"), encoding="utf-8")]

train_portion = int(len(data) * 0.85)
test_portion = int(len(data) * 0.1)
train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

customized_collate_fn = partial(
    custom_collate_fn,
    pad_token_id=tokenizer.pad_token_id,
    allowed_max_length=256,
)

batch_size = 8
torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=0,
)

torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
num_epochs = 3

start_context = format_question(val_data[0]) if val_data else "### 问：什么是逍遥？\n### 答："

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=10, eval_iter=5,
    start_context=start_context, tokenizer=tokenizer,
)

out_path = os.path.join(BASE_DIR, "zhuangzi-sft.pth")
torch.save(model.state_dict(), out_path)
print(f"训练完成，权重已保存: {out_path}")
