import argparse

from trainer import CBHGTrainer, RNNTrainer

trainers = {"cbhg": CBHGTrainer, "rnn": RNNTrainer, "crnn": RNNTrainer}
default_trainer = "rnn"


def train(model: str = default_trainer):
    trainer = trainers[model]
    if trainer == RNNTrainer:
        trainer = trainer(model_name=model)
    else:
        trainer = trainer()
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=trainers.keys(),
        default=default_trainer,
        help="Model to train",
    )
    args = parser.parse_args()
    train(args.model)
