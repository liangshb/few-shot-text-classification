import torch
from torch import nn
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.nn.util import get_mask_from_sequence_lengths


class CNNEncoder(nn.Module):
    def __init__(self,
                 config,
                 num_support,
                 vocab_size,
                 weights,
                 pad_idx,
                 ):
        super(CNNEncoder, self).__init__()
        self.num_support = num_support

        self.embedding = nn.Embedding(vocab_size, config["embed_dim"], padding_idx=pad_idx)
        if weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(weights))

        self.encoder = CnnEncoder(
            embedding_dim=config["embed_dim"],
            num_filters=config["num_filters"],
            ngram_filter_sizes=config["kernel_sizes"],
            conv_layer_activation=config.get("conv_layer_activation"),
            output_dim=config.get("output_dim"),
        )
        if config.get("do_layer_norm"):
            self.norm = nn.LayerNorm(self.encoder.get_output_dim())
        else:
            self.norm = lambda tensor: tensor

    def get_output_dim(self):
        return self.encoder.get_output_dim()

    def forward(self, x, length):
        x = self.embedding(x)
        mask = get_mask_from_sequence_lengths(length, max_length=max(length))
        x = self.encoder(x, mask)
        x = self.norm(x)
        support, query = x[0: self.num_support], x[self.num_support:]
        return support, query
