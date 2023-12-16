from tracemalloc import start

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CONFIG
from dataset import DiacritizerDataset, get_dataloader
from encoder.arabic_encoder import ArabicEncoder
from model import CBHGModel


def train_CBHG():
    """Train CBHG model."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("WARNING: CUDA not available, using CPU")
        device = torch.device("cpu")

    encoder = ArabicEncoder()
    start_token_id = encoder.start_token_id

    model = CBHGModel(
        in_vocab_size=len(encoder.vocab),
        out_vocab_size=len(encoder.diacritics),
        embedding_dim=CONFIG["embedding_dim"],
        use_prenet=CONFIG["use_prenet"],
        prenet_dims=CONFIG["prenet_dims"],
        prenet_dropout=CONFIG["prenet_dropout"],
        cbhg_num_filters=CONFIG["cbhg_num_filters"],
        cbhg_proj_dims=CONFIG["cbhg_proj_dims"],
        cbhg_num_highway_layers=CONFIG["cbhg_num_highway_layers"],
        cbhg_gru_hidden_dim=CONFIG["cbhg_gru_hidden_dim"],
        lstm_hidden_dims=CONFIG["lstm_hidden_dims"],
        use_batch_norm_post_lstm=CONFIG["use_batch_norm_post_lstm"],
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        betas=CONFIG["betas"],
        weight_decay=CONFIG["weight_decay"],
    )

    step = 0
    resume_training = False

    scaler = torch.cuda.amp.GradScaler()
    training_set = DiacritizerDataset(data=encoder.data, encoder=encoder)
    train_iterator = get_dataloader(
        training_set,
        params={
            "batch_size": CONFIG["batch_size"],
            "shuffle": True,
            "num_workers": CONFIG["num_workers"],
        },
    )

    eval_set = DiacritizerDataset(data=encoder.data, encoder=encoder)
    eval_iterator = get_dataloader(
        eval_set,
        params={
            "batch_size": CONFIG["batch_size"],
            "shuffle": False,
            "num_workers": CONFIG["num_workers"],
        },
    )


if __name__ == "__main__":
    train_CBHG()
