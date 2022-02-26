import torch
from torch import nn
from allennlp.modules.seq2seq_encoders import LstmSeq2SeqEncoder, GruSeq2SeqEncoder
from allennlp.nn.util import get_mask_from_sequence_lengths
from allennlp.modules.attention import AdditiveAttention
from models.modules.rnn_attn import RnnAttnEncoder

_RNN_TYPE = {
    "lstm": LstmSeq2SeqEncoder,
    "gru": GruSeq2SeqEncoder
}


class RNNATTNEncoder(nn.Module):
    def __init__(self, config, num_support, vocab_size, weights, pad_idx):
        super(RNNATTNEncoder, self).__init__()
        self.num_support = num_support

        self.embedding = nn.Embedding(vocab_size, config["embed_dim"], padding_idx=pad_idx)
        if weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(weights))

        seq2seq_encoder = _RNN_TYPE[config["rnn_type"]](
            input_size=config["embed_dim"],
            hidden_size=config["hidden_dim"],
            num_layers=config["num_layers"],
            bias=config.get("bias") or True,
            dropout=config.get("dropout") or 0.0,
            bidirectional=config.get("bidirectional") or True,
            stateful=config.get("stateful") or False
        )
        attention = AdditiveAttention(
            vector_dim=config["hidden_dim"] * 2,
            matrix_dim=config["hidden_dim"] * 2
        )
        self.encoder = RnnAttnEncoder(
            seq2seq_encoder=seq2seq_encoder,
            attention=attention
        )

    def get_output_dim(self):
        return self.encoder.get_output_dim()

    def forward(self, x, length):
        x = self.embedding(x)
        mask = get_mask_from_sequence_lengths(length, max_length=max(length))
        x = self.encoder(x, mask)
        support, query = x[0: self.num_support], x[self.num_support:]
        return support, query
