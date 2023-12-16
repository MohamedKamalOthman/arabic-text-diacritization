CONFIG = {
    # data params
    "train_data_path": "dataset/train.txt",
    "val_data_path": "dataset/val.txt",
    "test_data_path": "dataset/val.txt",
    "max_length": 500,
    # model params
    "embedding_dim": 256,
    "prenet_dim": (512, 256),
    "fully_connected_dim": (256, 256),
    "cbhg_gru_hidden_dim": (256, 256),
    "conv_filters": 16,
    # training params
    "batch_size": 32,
    "epochs": 50_000,
    "lr": 0.001,
    "adam_betas": (0.9, 0.999),
    "decay": True,
    "prenet": True,
}
