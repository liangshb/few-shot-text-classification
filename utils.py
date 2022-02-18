import torch
from model import Encoder, CNNEncoder, LSTMEncoder, GRUEncoder, CNNHEncoder
from omegaconf import DictConfig


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


def get_encoder(conf: DictConfig, vocab_size, weights):
    num_classes = conf["class"]
    num_support = conf["support"]
    embed_dim = conf["embed_dim"]
    hidden_dim = conf["hidden_dim"]
    d_a = conf.get("d_a")
    num_filters = conf.get("num_filters")
    kernel_sizes = conf.get("kernel_sizes")
    dropout = conf.get("dropout")
    num_layers = conf.get("num_layers")
    model_type = conf.get("type")
    if model_type == "cnn":
        encoder = CNNEncoder(
            num_classes,
            num_support,
            vocab_size,
            embed_dim,
            num_filters,
            kernel_sizes,
            dropout,
            hidden_dim,
            weights
        )
    elif model_type == "cnnh":
        encoder = CNNHEncoder(
            num_classes,
            num_support,
            vocab_size,
            embed_dim,
            num_filters,
            kernel_sizes,
            num_layers,
            dropout,
            hidden_dim,
            weights
        )
    elif model_type == "lstm":
        encoder = LSTMEncoder(
            num_classes,
            num_support,
            vocab_size,
            embed_dim,
            hidden_dim,
            num_layers,
            weights
        )
    elif model_type == "gru":
        encoder = GRUEncoder(
            num_classes,
            num_support,
            vocab_size,
            embed_dim,
            hidden_dim,
            num_layers,
            weights
        )
    else:
        encoder = Encoder(num_classes, num_support, vocab_size, embed_dim, hidden_dim, d_a, weights)

    return encoder
