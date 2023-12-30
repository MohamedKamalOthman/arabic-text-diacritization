import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import CONFIG
from dataset import DiacritizedDataset, get_dataloader
from encoder.arabic_encoder import ArabicEncoder
from models.cbhg import CBHGModel
from models.loader import load_model
from models.rnn import RNNModel
from utils import batch_accuracy, batch_diac_error


class Trainer:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print("WARNING: CUDA not available, using CPU")
            self.device = torch.device("cpu")

        self.encoder = ArabicEncoder()
        # self.start_token_id = self.encoder.start_token_id

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.encoder.padding_token_id)
        self.scaler = torch.cuda.amp.GradScaler()

        self.epoch = 0
        self.step = 0

        training_data = open(
            CONFIG["train_data_path"], "r", encoding="utf-8"
        ).readlines()
        training_set = DiacritizedDataset(data=training_data, encoder=self.encoder)
        self.train_iterator = get_dataloader(
            training_set,
            params={
                "batch_size": CONFIG["batch_size"],
                "shuffle": True,
                "num_workers": CONFIG["num_workers"],
            },
        )

        eval_data = open(CONFIG["val_data_path"], "r", encoding="utf-8").readlines()
        eval_set = DiacritizedDataset(data=eval_data, encoder=self.encoder)
        self.eval_iterator = get_dataloader(
            eval_set,
            params={
                "batch_size": CONFIG["eval_batch_size"],
                "shuffle": False,
                "num_workers": CONFIG["num_workers"],
            },
        )

        self.model: nn.Module | None = None
        self.optimizer: optim.Optimizer | None = None

    def log(self, name: str, log_string: str, path: str = "cbhg"):
        full_dir_path = os.path.join(CONFIG["log_base_path"], path)
        # create directory if not exists
        os.makedirs(full_dir_path, exist_ok=True)
        # write to log
        with open(os.path.join(full_dir_path, name + ".log"), "a") as file:
            file.write(log_string + "\n")

    def save(self, path: str = "cbhg"):
        if self.model is None:
            raise ValueError("Model is not initialized")
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized")
        full_dir_path = os.path.join(
            CONFIG["log_base_path"],
            path,
            "checkpoints",
        )
        # create directory if not exists
        os.makedirs(full_dir_path, exist_ok=True)
        # save model
        torch.save(
            {
                "epoch": self.epoch,
                "step": self.step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(full_dir_path, f"{self.epoch}-snapshot.pt"),
        )
        # write config
        with open(
            os.path.join(CONFIG["log_base_path"], path, "config.json"), "w"
        ) as file:
            file.write(json.dumps(CONFIG, indent=4))

    def load(self, path: str = "cbhg"):
        epoch = CONFIG.get("load_epoch", -1)
        if self.model is None:
            raise ValueError("Model is not initialized")
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized")
        # check if path exists
        full_dir_path = os.path.join(
            CONFIG["log_base_path"],
            path,
            "checkpoints",
        )
        if not os.path.exists(full_dir_path):
            print("WARNING: No checkpoints found, starting from scratch")
            return

        # get latest checkpoint
        if epoch == -1:
            checkpoints = os.listdir(full_dir_path)
            checkpoints = [int(checkpoint.split("-")[0]) for checkpoint in checkpoints]
            checkpoints.sort()
            epoch = checkpoints[-1]

        checkpoint = torch.load(
            os.path.join(full_dir_path, f"{epoch}-snapshot.pt"),
            map_location=self.device,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError("training_step is not implemented in base Trainer")

    def train(self):
        total_steps = CONFIG["epochs"] * len(self.train_iterator)
        training_tqdm = tqdm(
            iterable=range(total_steps),
            total=total_steps,
            leave=True,
            initial=self.step,
            desc="Training: ",
        )
        for epoch in range(self.epoch, CONFIG["epochs"]):
            self.epoch = epoch + 1
            for batch in self.train_iterator:
                if False and CONFIG["use_decay"]:
                    # TODO: implement learning rate decay
                    pass

                self.optimizer.zero_grad()
                result = self.training_step(batch)
                # get training loss
                loss = result["loss"]
                # get training batch accuracy
                acc = batch_accuracy(
                    result["pred"], result["gold"], self.encoder.padding_token_id
                ).item()
                loss.backward()
                # TODO: implement gradient clipping
                self.optimizer.step()

                self.step += 1

                # update tqdm
                training_tqdm.set_description_str(
                    f"Training: loss: {loss:.8f}, accuracy: {acc:.8f}, epoch: {self.epoch}"
                )
                training_tqdm.update()

            # save model
            if self.step % CONFIG["save_every"] == 0:
                self.save()

            # evaluate
            if self.step % CONFIG["eval_every"] == 0:
                self.evaluate()

        # final evaluation
        self.evaluate()
        training_tqdm.close()

    def evaluate(self):
        eval_tqdm = tqdm(
            iterable=self.eval_iterator,
            total=len(self.eval_iterator),
            leave=True,
            desc="Evaluate: ",
        )
        total_loss = 0
        total_acc = 0
        total_diac_corr = 0
        total_diac_err = 0
        self.model.eval()
        with torch.no_grad():
            for batch in self.eval_iterator:
                char_seq = batch["char_seq"].to(self.device)
                diac_seq = batch["diac_seq"].to(self.device)
                seq_lengths = batch["seq_lengths"].to("cpu")

                # forward pass
                pred = (
                    self.model(char_seq, seq_lengths)
                    .contiguous()
                    .view(-1, self.encoder.out_vocab_size)
                )

                # get res
                gold = diac_seq.contiguous().view(-1).to(self.device)

                # calculate loss
                loss = self.criterion(pred, gold)
                acc = batch_accuracy(pred, gold, self.encoder.padding_token_id)
                cor, err = batch_diac_error(
                    char_seq,
                    pred,
                    gold,
                    self.encoder.arabic_ids,
                    device=self.device,
                )
                total_diac_corr += cor.item()
                total_diac_err += err.item()
                total_loss += loss.item()
                total_acc += acc.item()
                eval_tqdm.update()

        eval_tqdm.set_description(
            f"Evaluate: loss: {total_loss/len(self.eval_iterator)}, accuracy: {total_acc/len(self.eval_iterator)}, DER: {total_diac_err/(total_diac_err+total_diac_corr)}"
        )
        # save to log
        self.log(
            name="eval",
            log_string=f"Epoch: {self.epoch}, Accuracy: {total_acc/len(self.eval_iterator)}, Loss: {total_loss/len(self.eval_iterator)}",
        )

        self.log(
            name="eval",
            log_string=f"Epoch: {self.epoch}, DER: {total_diac_err/(total_diac_err+total_diac_corr)}, error: {total_diac_err}, correct: {total_diac_corr}",
        )

        self.model.train()
        eval_tqdm.close()


class RNNTrainer(Trainer):
    def __init__(self):
        super(RNNTrainer, self).__init__()

        self.model: RNNModel = load_model("rnn", self.encoder).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=CONFIG["learning_rate"],
            betas=CONFIG["adam_betas"],
            weight_decay=CONFIG["weight_decay"],
        )

        # Load model if specified
        if CONFIG["load_model"]:
            self.load()

    def log(self, name: str, log_string: str):
        super(RNNTrainer, self).log(name=name, log_string=log_string, path="rnn")

    def load(self):
        super(RNNTrainer, self).load("rnn")

    def save(self):
        super(RNNTrainer, self).save("rnn")

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        char_seq = batch["char_seq"].to(self.device)
        diac_seq = batch["diac_seq"].to(self.device)
        seq_lengths = batch["seq_lengths"].to("cpu")
        # forward pass
        pred = (
            self.model(char_seq, seq_lengths)
            .contiguous()
            .view(-1, self.encoder.out_vocab_size)
        )

        # get res
        gold = diac_seq.contiguous().view(-1)

        # calculate loss
        loss = self.criterion(pred, gold)
        return {"loss": loss, "pred": pred, "gold": gold}


class CBHGTrainer(Trainer):
    def __init__(self):
        super(CBHGTrainer, self).__init__()
        self.model: CBHGModel = load_model("cbhg", self.encoder).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=CONFIG["learning_rate"],
            betas=CONFIG["adam_betas"],
            weight_decay=CONFIG["weight_decay"],
        )

        # Load model if specified
        if CONFIG["load_model"]:
            self.load()

    def log(self, name: str, log_string: str):
        super(CBHGTrainer, self).log(name=name, log_string=log_string, path="cbhg")

    def load(self):
        super(CBHGTrainer, self).load("cbhg")

    def save(self):
        super(CBHGTrainer, self).save("cbhg")

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        char_seq = batch["char_seq"].to(self.device)
        diac_seq = batch["diac_seq"].to(self.device)
        seq_lengths = batch["seq_lengths"].to("cpu")

        # forward pass
        pred = (
            self.model(char_seq, seq_lengths)
            .contiguous()
            .view(-1, self.encoder.out_vocab_size)
        )

        # backward pass
        gold = diac_seq.contiguous().view(-1)

        # calculate loss
        loss = self.criterion(pred.to(self.device), gold.to(self.device))
        return {"loss": loss, "pred": pred, "gold": gold}
