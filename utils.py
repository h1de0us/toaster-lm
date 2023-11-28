import torch
from typing import List

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad_sequence(batch):
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
    return batch


def collate_fn(dataset_items: List[dict]):
    return {
        "texts": torch.as_tensor(pad_sequence([item[0] for item in dataset_items])),
        "lengths": torch.as_tensor([item[1] for item in dataset_items]),
    }
