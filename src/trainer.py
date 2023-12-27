import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange

from config import CONFIG
from dataset import DiacritizerDataset, get_dataloader
from encoder.arabic_encoder import ArabicEncoder
from models.cbhg import CBHGModel
from models.rnn import RNNModel


def batch_accuracy(output, gold, pad_index, device="cuda"):
    predictions = output.argmax(dim=1, keepdim=True)
    non_pad_elements = torch.nonzero((gold != pad_index))
    correct = predictions[non_pad_elements].squeeze(1).eq(gold[non_pad_elements])
    return correct.sum() / torch.FloatTensor([gold[non_pad_elements].shape[0]]).to(
        device
    )


def _repeated_batches(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


class Trainer:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print("WARNING: CUDA not available, using CPU")
            self.device = torch.device("cpu")

        self.encoder = ArabicEncoder()
        self.start_token_id = self.encoder.start_token_id

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.encoder.padding_token_id)
        self.scaler = torch.cuda.amp.GradScaler()

        self.step = 0

        training_data = open(
            CONFIG["train_data_path"], "r", encoding="utf-8"
        ).readlines()
        training_set = DiacritizerDataset(data=training_data, encoder=self.encoder)
        self.train_iterator = get_dataloader(
            training_set,
            params={
                "batch_size": CONFIG["batch_size"],
                "shuffle": True,
                "num_workers": CONFIG["num_workers"],
            },
        )

        eval_data = open(CONFIG["val_data_path"], "r", encoding="utf-8").readlines()
        eval_set = DiacritizerDataset(data=eval_data, encoder=self.encoder)
        self.eval_iterator = get_dataloader(
            eval_set,
            params={
                "batch_size": CONFIG["batch_size"],
                "shuffle": False,
                "num_workers": CONFIG["num_workers"],
            },
        )

        self.model: nn.Module | None = None

    def evaluate(self):
        eval_tqdm = trange(self.step, len(self.eval_iterator), leave=True)
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        for batch in self.eval_iterator:
            with torch.no_grad():
                char_seq = batch["char_seq"].to(self.device)
                diac_seq = batch["diac_seq"].to(self.device)

                # forward pass
                pred = (
                    self.model(char_seq)
                    .contiguous()
                    .view(-1, len(self.encoder.diacritics))
                )

                # get res
                gold = diac_seq.contiguous().view(-1).to(self.device)

                # calculate loss
                loss = self.criterion(pred, gold)
                acc = batch_accuracy(pred, gold, self.encoder.padding_token_id)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                eval_tqdm.update()

        eval_tqdm.display(
            f"Evaluate: accuracy, {epoch_acc / len(self.eval_iterator)}, loss: {epoch_loss / len(self.eval_iterator)}",
            pos=4,
        )
        self.model.train()


class RNNTrainer(Trainer):
    def __init__(self):
        super(RNNTrainer, self).__init__()
        self.model = RNNModel(
            in_vocab_size=len(self.encoder.vocab),
            out_vocab_size=len(self.encoder.diacritics),
            embedding_dim=CONFIG["rnn_embedding_dim"],
            hidden_dim=CONFIG["rnn_hidden_dim"],
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=CONFIG["learning_rate"],
            betas=CONFIG["adam_betas"],
            weight_decay=CONFIG["weight_decay"],
        )

    def train(self):
        self.evaluate()
        training_tqdm = trange(self.step, CONFIG["total_steps"], leave=True)
        for batch in _repeated_batches(self.train_iterator):
            if False and CONFIG["use_decay"]:
                # TODO: implement learning rate decay
                pass

            self.optimizer.zero_grad()
            result = self.training_step(batch)
            loss = result["loss"]
            training_tqdm.display(f"loss: {loss}", pos=3)
            loss.backward()
            # TODO: implement gradient clipping
            self.optimizer.step()

            self.step += 1
            if self.step > CONFIG["total_steps"]:
                self.evaluate()
                return

            training_tqdm.update()

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        char_seq = batch["char_seq"].to(self.device)
        diac_seq = batch["diac_seq"].to(self.device)
        # seq_lengths = batch["seq_lengths"].to("cpu")
        # forward pass
        pred = self.model(char_seq).contiguous().view(-1, len(self.encoder.diacritics))

        # get res
        gold = diac_seq.contiguous().view(-1)

        # calculate loss
        loss = self.criterion(pred, gold)
        return {"loss": loss, "pred": pred}


class CBHGTrainer(Trainer):
    def __init__(self):
        super(CBHGTrainer, self).__init__()
        self.model = CBHGModel(
            in_vocab_size=len(self.encoder.vocab),
            out_vocab_size=len(self.encoder.diacritics),
            embedding_dim=CONFIG["embedding_dim"],
            use_prenet=CONFIG["use_prenet"],
            prenet_dims=CONFIG["prenet_dims"],
            prenet_dropout=CONFIG["prenet_dropout"],
            cbhg_num_filters=CONFIG["cbhg_num_filters"],
            cbhg_proj_dims=CONFIG["cbhg_proj_dims"],
            cbhg_num_highway_layers=CONFIG["cbhg_num_highway_layers"],
            cbhg_gru_hidden_size=CONFIG["cbhg_gru_hidden_size"],
            lstm_hidden_dims=CONFIG["lstm_hidden_dims"],
            use_batch_norm_post_lstm=CONFIG["use_batch_norm_post_lstm"],
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=CONFIG["learning_rate"],
            betas=CONFIG["adam_betas"],
            weight_decay=CONFIG["weight_decay"],
        )

    def train(self):
        self.evaluate()
        training_tqdm = trange(self.step, CONFIG["total_steps"], leave=True)
        for batch in _repeated_batches(self.train_iterator):
            if False and CONFIG["use_decay"]:
                # TODO: implement learning rate decay
                pass

            self.optimizer.zero_grad()
            result = self.training_step(batch)
            loss = result["loss"]
            training_tqdm.display(f"loss: {loss}", pos=3)
            loss.backward()
            # TODO: implement gradient clipping
            self.optimizer.step()

            self.step += 1
            if self.step > CONFIG["total_steps"]:
                self.evaluate()
                return

            training_tqdm.update()

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        char_seq = batch["char_seq"].to(self.device)
        diac_seq = batch["diac_seq"].to(self.device)
        seq_lengths = batch["seq_lengths"].to("cpu")

        # forward pass
        pred = (
            self.model(char_seq, seq_lengths)
            .contiguous()
            .view(-1, len(self.encoder.diacritics))
        )

        # backward pass
        gold = diac_seq.contiguous().view(-1)

        # calculate loss
        loss = self.criterion(pred.to(self.device), gold.to(self.device))
        return {"loss": loss, "pred": pred}

    def evaluate_with_error_rates(self, iterator, tqdm):
        all_orig = []
        all_predicted = []
        results = {}
        self.diacritizer.set_model(self.model)
        evaluated_batches = 0
        tqdm.set_description(f"Calculating DER/WER {self.global_step}: ")
        for batch in iterator:
            if evaluated_batches > int(self.config["error_rates_n_batches"]):
                break

            predicted = self.diacritizer.diacritize_batch(batch)
            all_predicted += predicted
            all_orig += batch["original"]
            tqdm.update()

        summary_texts = []
        orig_path = os.path.join(self.config_manager.prediction_dir, f"original.txt")
        predicted_path = os.path.join(
            self.config_manager.prediction_dir, f"predicted.txt"
        )

        with open(orig_path, "w", encoding="utf8") as file:
            for sentence in all_orig:
                file.write(f"{sentence}\n")

        with open(predicted_path, "w", encoding="utf8") as file:
            for sentence in all_predicted:
                file.write(f"{sentence}\n")

        for i in range(int(self.config["n_predicted_text_tensorboard"])):
            if i > len(all_predicted):
                break

            summary_texts.append(
                (f"eval-text/{i}", f"{ all_orig[i]} |->  {all_predicted[i]}")
            )

        results["DER"] = der.calculate_der_from_path(orig_path, predicted_path)
        results["DER*"] = der.calculate_der_from_path(
            orig_path, predicted_path, case_ending=False
        )
        results["WER"] = wer.calculate_wer_from_path(orig_path, predicted_path)
        results["WER*"] = wer.calculate_wer_from_path(
            orig_path, predicted_path, case_ending=False
        )
        tqdm.reset()
        return results, summary_texts
