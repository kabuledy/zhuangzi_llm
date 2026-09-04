
import torch
from torch.utils.data import Dataset

def format_question(entry):
    """把一条样本的问题部分格式化成模型看到的文本。"""
    return f"### 问：{entry['input']}\n### 答："

class InstructionDataset(Dataset):
    """uer 版指令数据集。

    每条样本 = token_ids(问题+答案拼接) + answer_start(答案从哪个位置开始)。
    存 answer_start 就是为了 collate 时能精确标出「答案段」，不靠 eos 计数。
    """

    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts, self.answer_starts = [], []

        for entry in data:
            q_text = format_question(entry)
            a_text = entry["output"]

            q_ids = tokenizer.encode(q_text, add_special_tokens=False)
            a_ids = tokenizer.encode(a_text, add_special_tokens=False)

            full_ids = q_ids + a_ids + [tokenizer.eos_token_id]

            self.encoded_texts.append(full_ids)
            self.answer_starts.append(len(q_ids))

    def __getitem__(self, index):
        return self.encoded_texts[index], self.answer_starts[index]

    def __len__(self):
        return len(self.data)

def custom_collate_fn(
    batch,
    pad_token_id=0,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """把一批 (token_ids, answer_start) 填成等长，构造 input/target 对。

    关键：target 里【只有答案段】保留真实 token 作监督，
    问题段和 padding 一律置 ignore_index(-100)，不参与 loss。
    """
    batch_max_length = max(len(item) for item, _ in batch)
    if allowed_max_length is not None:
        batch_max_length = min(batch_max_length, allowed_max_length)

    inputs_lst, targets_lst = [], []

    for item, ans_start in batch:
        n = len(item)

        if allowed_max_length is not None and n > allowed_max_length:
            item = item[:allowed_max_length]
            ans_start = min(ans_start, allowed_max_length - 1)
            n = allowed_max_length

        padded = item + [pad_token_id] * (batch_max_length - n)

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        for i in range(len(targets)):
            orig_idx = i + 1
            if not (ans_start <= orig_idx < n):
                targets[i] = ignore_index

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor
