import torch
from torch.nn.utils.rnn import pad_packed_sequence
from allennlp.common.checks import ConfigurationError
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from allennlp.modules.attention.attention import Attention
from allennlp.nn.util import weighted_sum


class RNN(Seq2SeqEncoder):
    def __init__(self, module: torch.nn.Module) -> None:
        super(RNN, self).__init__()
        self._module = module
        try:
            if not self._module.batch_first:
                raise ConfigurationError("Our encoder semantics assumes batch is always first!")
        except AttributeError:
            pass

        try:
            self._is_bidirectional = self._module.bidirectional
        except AttributeError:
            self._is_bidirectional = False
        if self._is_bidirectional:
            self._num_directions = 2
        else:
            self._num_directions = 1

    def get_input_dim(self) -> int:
        return self._module.input_size

    def get_output_dim(self) -> int:
        return self._module.hidden_size * self._num_directions

    def is_bidirectional(self) -> bool:
        return self._is_bidirectional

    def forward(
        self, inputs: torch.Tensor, mask: torch.BoolTensor, hidden_state: torch.Tensor = None
    ) -> torch.Tensor:
        if mask is None:
            res = self._module(inputs, hidden_state)[0]
            return res[0], res[0][:, -1, :]

        batch_size, total_sequence_length = mask.size()

        packed_sequence_output, state, restoration_indices = self.sort_and_run_forward(
            self._module, inputs, mask, hidden_state
        )

        # seq2seq
        unpacked_sequence_tensor, _ = pad_packed_sequence(packed_sequence_output, batch_first=True)
        num_valid = unpacked_sequence_tensor.size(0)
        if num_valid < batch_size:
            _, length, output_dim = unpacked_sequence_tensor.size()
            zeros = unpacked_sequence_tensor.new_zeros(batch_size - num_valid, length, output_dim)
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 0)

        sequence_length_difference = total_sequence_length - unpacked_sequence_tensor.size(1)
        if sequence_length_difference > 0:
            zeros = unpacked_sequence_tensor.new_zeros(
                batch_size, sequence_length_difference, unpacked_sequence_tensor.size(-1)
            )
            unpacked_sequence_tensor = torch.cat([unpacked_sequence_tensor, zeros], 1)

        # seq2vec
        if isinstance(state, tuple):
            state = state[0]

        num_layers_times_directions, num_valid, encoding_dim = state.size()
        if num_valid < batch_size:
            zeros = state.new_zeros(
                num_layers_times_directions, batch_size - num_valid, encoding_dim
            )
            state = torch.cat([state, zeros], 1)

        unsorted_state = state.transpose(0, 1).index_select(0, restoration_indices)
        try:
            last_state_index = 2 if self._module.bidirectional else 1
        except AttributeError:
            last_state_index = 1
        last_layer_state = unsorted_state[:, -last_state_index:, :]
        last_layer_state = last_layer_state.contiguous().view([-1, self.get_output_dim()])

        return unpacked_sequence_tensor.index_select(0, restoration_indices), last_layer_state


@Seq2SeqEncoder.register("blstm")
class BLSTMEncoder(RNN):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 dropout: float = 0.0,
                 ):
        module = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        super(BLSTMEncoder, self).__init__(module)


@Seq2SeqEncoder.register("bgru")
class BGRUEncoder(RNN):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 dropout: float = 0.0,
                 ):
        module = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        super(BGRUEncoder, self).__init__(module)


@Seq2VecEncoder.register("rnn-attn")
class RnnAttnEncoder(Seq2VecEncoder):
    def __init__(self, rnn_encoder: Seq2SeqEncoder, attention: Attention):
        super(RnnAttnEncoder, self).__init__()
        self.encoder = rnn_encoder
        self.attention = attention

    def get_input_dim(self) -> int:
        return self.encoder.get_input_dim()

    def get_output_dim(self) -> int:
        return self.encoder.get_output_dim()

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor):
        encoded, state = self.encoder(tokens, mask)
        weights = self.attention(state, encoded, mask)
        encoded = weighted_sum(encoded, weights)
        return encoded
