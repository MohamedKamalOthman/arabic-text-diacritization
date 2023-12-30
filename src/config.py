CONFIG = {
    "models": ["cbhg", "rnn"],
    # data params
    "train_data_path": "../dataset/train_split.txt",
    "val_data_path": "../dataset/val.txt",
    "test_data_directory": "../dataset/tests",
    "test_model_name": "rnn",
    "test_model_path": "saved_models/1-rnn.pt",
    "infer_model_path": "dataset/test.txt",
    "max_length": 500,
    # data loader params
    "num_workers": 2,
    "eval_batch_size": 256,
    "test_batch_size": 256,
    # Log params
    "log_base_path": "logs",
    "save_every": 1,  # save every x epochs
    "load_model": False,
    "load_epoch": -1,
    # model params
    "embedding_dim": 256,
    "cbhg_num_filters": 16,
    "cbhg_proj_dims": (256, 256),
    "cbhg_gru_hidden_size": 32,
    "cbhg_num_highway_layers": 4,
    "lstm_hidden_dims": (256, 256),
    "use_batch_norm_post_lstm": True,
    "use_prenet": True,
    "prenet_dims": (512, 256),
    "prenet_dropout": 0.5,
    # training params
    "batch_size": 32,
    "epochs": 3,
    "eval_every": 1,  # eval every x epochs
    "learning_rate": 0.001,
    "adam_betas": (0.9, 0.999),
    "weight_decay": 0.0,
    "use_decay": True,
    # RNN Model
    "rnn_embedding_dim": 256,
    "rnn_hidden_dim": 128,
    "rnn_num_layers": 3,
    # inference params
    "inference_batch_size": 1,
}
