import os

import torch
from tqdm import tqdm

from config import CONFIG
from dataset import DiacritizedDataset, get_dataloader
from encoder.arabic_encoder import ArabicEncoder
from models.loader import load_model
from utils import batch_accuracy


class Tester:
    def __init__(self, model_name, model_path):
        self.encoder = ArabicEncoder()
        self.model_name = model_name
        self.model = load_model(model_name=model_name, encoder=self.encoder)
        model_snapshot = torch.load(model_path)
        self.model.load_state_dict(model_snapshot["model_state_dict"])
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.encoder.padding_token_id
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def log(self, log_string: str, create_new: bool = False):
        full_dir_path = os.path.join(CONFIG["log_base_path"], self.model_name)
        # create directory if not exists
        os.makedirs(full_dir_path, exist_ok=True)
        # write to log
        with open(
            os.path.join(full_dir_path, "test.log"), "w" if create_new else "a"
        ) as file:
            file.write(log_string + "\n")

    def test_all(self, test_directory: str):
        test_files = os.listdir(test_directory)
        for test_file in test_files:
            test_file_path = os.path.join(test_directory, test_file)
            self.test(test_file_path)

    def test(self, test_file_path: str):
        test_filename = os.path.basename(test_file_path)
        test_data = open(test_file_path, "r", encoding="utf-8").readlines()
        test_set = DiacritizedDataset(data=test_data, encoder=self.encoder)
        test_iterator = get_dataloader(
            dataset=test_set,
            params={
                "batch_size": CONFIG["test_batch_size"],
                "shuffle": False,
                "num_workers": CONFIG["num_workers"],
            },
        )

        test_tqdm = tqdm(
            iterable=test_iterator,
            total=len(test_iterator),
            leave=True,
            desc=f"Testing {test_filename}: ",
        )

        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for batch in test_iterator:
                char_seq: torch.Tensor = batch["char_seq"].to(self.device)
                diac_seq: torch.Tensor = batch["diac_seq"].to(self.device)
                seq_lengths: torch.Tensor = batch["seq_lengths"].to("cpu")

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
                test_loss += loss.item()
                test_acc += acc.item()

                test_tqdm.update()

        test_loss /= len(test_iterator)
        test_acc /= len(test_iterator)
        log_string = (
            f"Testing {test_filename}: accuracy={test_acc:.6f}, loss={test_loss:.6f}"
        )
        test_tqdm.set_description(log_string)
        self.log(log_string)

        test_tqdm.close()
