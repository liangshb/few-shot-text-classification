import torch
from torch import nn
from allennlp.nn.util import get_mask_from_sequence_lengths
from models.modules.cnn_highway_mask import CnnHighwayMaskEncoder


class CNNHEncoder(nn.Module):
    def __init__(self,
                 config,
                 num_support,
                 vocab_size,
                 weights,
                 pad_idx
                 ):
        super(CNNHEncoder, self).__init__()
        self.num_support = num_support

        self.embedding = nn.Embedding(vocab_size, config["embed_dim"], padding_idx=pad_idx)
        if weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(weights))

        self.encoder = CnnHighwayMaskEncoder(
            embedding_dim=config["embed_dim"],
            num_filters=config["num_filters"],
            ngram_filter_sizes=config["kernel_sizes"],
            projection_dim=config["projection_dim"],
            num_highway=config.get("num_highway") or 1,
            activation=config.get("activation") or "relu",
            projection_location=config.get("projection_location") or "after_highway",
            do_layer_norm=config.get("do_layer_norm") or False
        )

    def get_output_dim(self):
        return self.encoder.get_output_dim()

    def forward(self, x, length):
        x = self.embedding(x)
        mask = get_mask_from_sequence_lengths(length, max_length=max(length))
        x = self.encoder(x, mask)
        support, query = x[0: self.num_support], x[self.num_support:]
        return support, query
