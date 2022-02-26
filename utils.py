import torch
from torch.nn.utils.rnn import pad_sequence


def padding(data1, data2, pad_idx=0):
    len1, len2 = data1.shape[1], data2.shape[1]
    if len1 > len2:
        data2 = torch.cat([data2, torch.ones(data2.shape[0], len1 - len2).long() * pad_idx], dim=1)
    elif len2 > len1:
        data1 = torch.cat([data1, torch.ones(data1.shape[0], len2 - len1).long() * pad_idx], dim=1)
    return data1, data2


def batch_padding(data, pad_idx=0):
    max_len = 0
    for text in data:
        max_len = max(max_len, len(text))
    for i in range(len(data)):
        data[i] += [pad_idx] * (max_len - len(data[i]))
    return torch.tensor(data)


def collate_fn(batch, pad_idx=0):
    data_list = []
    length_list = []
    target_list = []
    for data, length, target in batch:
        data_list.append(torch.tensor(data))
        length_list.append(length)
        target_list.append(target)
    data = pad_sequence(data_list, batch_first=True, padding_value=pad_idx)
    length = torch.tensor(length_list)
    target = torch.tensor(target_list)
    return data, length, target
