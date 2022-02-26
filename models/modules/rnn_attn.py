import torch
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.attention.attention import Attention
from allennlp.nn.util import weighted_sum


@Seq2VecEncoder.register("rnn-attn")
class RnnAttnEncoder(Seq2VecEncoder):
    def __init__(
            self,
            seq2seq_encoder: Seq2SeqEncoder,
            attention: Attention
    ):
        super(RnnAttnEncoder, self).__init__()
        self.encoder = seq2seq_encoder
        self.hidden_size = self.encoder.get_output_dim() // 2
        self.attention = attention

    def get_input_dim(self) -> int:
        return self.encoder.get_input_dim()

    def get_output_dim(self) -> int:
        return self.encoder.get_output_dim()

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor):
        encoded = self.encoder(tokens, mask)
        forward_last = encoded[:, -1, :self.hidden_size]
        backward_last = encoded[:, 0, -self.hidden_size:]
        hidden = torch.cat((forward_last, backward_last), dim=1)
        weights = self.attention(hidden, encoded, mask)
        encoded = weighted_sum(encoded, weights)
        return encoded
