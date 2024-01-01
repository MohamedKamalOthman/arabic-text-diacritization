import torch

from config import CONFIG
from encoder.arabic_encoder import ArabicEncoder
from models.cbhg import CBHGModel
from models.crnn import CRNNModel
from models.rnn import RNNModel
from models.rnncrf import RNNCRFModel
from models.transformer import TransformerModel


def load_model(model_name: str, encoder: ArabicEncoder) -> torch.nn.Module:
    if model_name == "rnn":
        return RNNModel(
            in_vocab_size=encoder.in_vocab_size,
            out_vocab_size=encoder.out_vocab_size,
            embedding_dim=CONFIG["rnn_embedding_dim"],
            hidden_dim=CONFIG["rnn_hidden_dim"],
            num_layers=CONFIG["rnn_num_layers"],
        )
    elif model_name == "crnn":
        return CRNNModel(
            in_vocab_size=encoder.in_vocab_size,
            out_vocab_size=encoder.out_vocab_size,
            embedding_dim=CONFIG["crnn_embedding_dim"],
            hidden_dim=CONFIG["crnn_hidden_dim"],
            num_layers=CONFIG["crnn_num_layers"],
            K=CONFIG["crnn_num_filters"],
            proj_dims=CONFIG["crnn_proj_dims"],
        )
    elif model_name == "cbhg":
        return CBHGModel(
            in_vocab_size=encoder.in_vocab_size,
            out_vocab_size=encoder.out_vocab_size,
            embedding_dim=CONFIG["cbhg_embedding_dim"],
            use_prenet=CONFIG["use_prenet"],
            prenet_dims=CONFIG["prenet_dims"],
            prenet_dropout=CONFIG["prenet_dropout"],
            cbhg_num_filters=CONFIG["cbhg_num_filters"],
            cbhg_proj_dims=CONFIG["cbhg_proj_dims"],
            cbhg_num_highway_layers=CONFIG["cbhg_num_highway_layers"],
            cbhg_gru_hidden_size=CONFIG["cbhg_gru_hidden_size"],
            lstm_hidden_dims=CONFIG["lstm_hidden_dims"],
            use_batch_norm_post_lstm=CONFIG["use_batch_norm_post_lstm"],
        )
    elif model_name == "rnncrf":
        return RNNCRFModel(
            in_vocab_size=encoder.in_vocab_size,
            out_vocab_size=encoder.out_vocab_size,
            embedding_dim=CONFIG["rnn_embedding_dim"],
            hidden_dim=CONFIG["rnn_hidden_dim"],
            num_layers=CONFIG["rnn_num_layers"],
        )
    elif model_name == "transformer":
        return TransformerModel(
            input_size=encoder.in_vocab_size,
            output_size=encoder.out_vocab_size,
            # max_seq_length=CONFIG["transformer_max_seq_length"],
            # num_heads=CONFIG["transformer_num_heads"],
            # hidden_size=CONFIG["transformer_hidden_size"],
            # num_layers=CONFIG["transformer_num_layers"],
        )

    raise ValueError(f"Unknown model {model_name}")
