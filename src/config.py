CONFIG = {
    # data params
    "train_data_path": "../dataset/train.txt",
    "val_data_path": "../dataset/val.txt",
    "test_data_path": "dataset/val.txt",
    "max_length": 500,
    # data loader params
    "num_workers": 2,
    # model params
    "embedding_dim": 128,
    "prenet_dim": (128, 64),
    "cbhg_num_filters": 8,
    "cbhg_proj_dims": (64, 64),
    "cbhg_gru_hidden_size": 32,
    "cbhg_num_highway_layers": 4,
    "lstm_hidden_dims": (64, 64),
    "use_batch_norm_post_lstm": True,
    "use_prenet": True,
    "prenet_dims": (128, 64),
    "prenet_dropout": 0.5,
    # training params
    "total_steps": 1_000,
    "batch_size": 32,
    "epochs": 50_000,
    "learning_rate": 0.001,
    "adam_betas": (0.9, 0.999),
    "weight_decay": 1e-6,
    # RNN Model
    "rnn_embedding_dim": 256,
    "rnn_hidden_dim": 128,
}
