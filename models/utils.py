from model import Encoder
from models.rnn import RNNEncoder
from models.cnn import CNNEncoder
from models.cnnh import CNNHEncoder
from models.rnn_attn import RNNATTNEncoder

_MODELS = {
    "norm": Encoder,
    "rnn": RNNEncoder,
    "rnn_attn": RNNATTNEncoder,
    "cnn": CNNEncoder,
    "cnnh": CNNHEncoder
}


def get_encoder(model_config, vocab_size, weights, pad_idx):
    encoder_conf = model_config["encoder"]
    return _MODELS[encoder_conf["type"]](encoder_conf,
                                         model_config["class"] * model_config["support"],
                                         vocab_size,
                                         weights,
                                         pad_idx
                                         )
