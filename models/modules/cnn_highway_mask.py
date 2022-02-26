from typing import Callable, Tuple

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.modules.highway import Highway
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn import Activation
from allennlp.nn.util import min_value_of_dtype
from torch.nn import Conv1d

_VALID_PROJECTION_LOCATIONS = {"after_cnn", "after_highway", None}


@Seq2VecEncoder.register("cnn-highway-mask")
class CnnHighwayMaskEncoder(Seq2VecEncoder):
    def __init__(
            self,
            embedding_dim: int,
            num_filters: int,
            ngram_filter_sizes: Tuple[int, ...],
            num_highway: int,
            projection_dim: int,
            activation: str = "relu",
            projection_location: str = "after_highway",
            do_layer_norm: bool = False,
    ) -> None:
        super().__init__()

        if projection_location not in _VALID_PROJECTION_LOCATIONS:
            raise ConfigurationError(f"unknown projection location: {projection_location}")

        self._embedding_dim = embedding_dim
        self._num_filters = num_filters
        self._ngram_filter_sizes = ngram_filter_sizes
        self._num_highway = num_highway
        self._projection_dim = projection_dim
        self._activation = Activation.by_name(activation)()
        self._projection_location = projection_location
        self._do_layer_norm = do_layer_norm

        self._convolution_layers = [
            Conv1d(
                in_channels=self._embedding_dim,
                out_channels=self._num_filters,
                kernel_size=ngram_size,
            )
            for ngram_size in self._ngram_filter_sizes
        ]
        for i, conv_layer in enumerate(self._convolution_layers):
            self.add_module("conv_layer_%d" % i, conv_layer)

        num_filters = self._num_filters * len(self._ngram_filter_sizes)
        if projection_location == "after_cnn":
            highway_dim = projection_dim
        else:
            # highway_dim is the number of cnn filters
            highway_dim = num_filters
        self._highways = Highway(highway_dim, num_highway, activation=torch.nn.functional.relu)

        # Projection layer: always num_filters -> projection_dim
        self._projection = torch.nn.Linear(num_filters, projection_dim, bias=True)

        # And add a layer norm
        if do_layer_norm:
            self._layer_norm: Callable = LayerNorm(self._projection_dim)
        else:
            self._layer_norm = lambda tensor: tensor

    def get_input_dim(self) -> int:
        return self._embedding_dim

    def get_output_dim(self) -> int:
        return self._projection_dim

    def forward(self, tokens: torch.Tensor, mask: torch.BoolTensor):
        if mask is not None:
            tokens = tokens * mask.unsqueeze(-1)
        else:
            mask = torch.ones(tokens.shape[0], tokens.shape[1], device=tokens.device).bool()

        tokens = torch.transpose(tokens, 1, 2)

        outputs = []
        batch_size = tokens.shape[0]
        last_unmasked_tokens = mask.sum(dim=1).unsqueeze(dim=-1)
        for i in range(len(self._convolution_layers)):
            convolution_layer = getattr(self, "conv_layer_{}".format(i))
            pool_length = tokens.shape[2] - convolution_layer.kernel_size[0] + 1

            activations = self._activation(convolution_layer(tokens))

            indices = (
                torch.arange(pool_length, device=activations.device)
                    .unsqueeze(0)
                    .expand(batch_size, pool_length)
            )
            activations_mask = indices.ge(
                last_unmasked_tokens - convolution_layer.kernel_size[0] + 1
            )
            activations_mask = activations_mask.unsqueeze(1).expand_as(activations)

            activations = activations + (activations_mask * min_value_of_dtype(activations.dtype))

            outputs.append(activations.max(dim=2)[0])

        outputs = torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]
        outputs[outputs == min_value_of_dtype(outputs.dtype)] = 0.0

        if self._projection_location == "after_cnn":
            outputs = self._projection(outputs)
        outputs = self._highways(outputs)
        if self._projection_location == "after_highway":
            outputs = self._projection(outputs)

        outputs = self._layer_norm(outputs)
        return outputs
