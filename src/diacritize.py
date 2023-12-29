import argparse
import csv
import os
from pathlib import Path

import torch
from tqdm import tqdm

from config import CONFIG
from dataset import UndiacritizedDataset, get_dataloader
from encoder.arabic_encoder import ArabicEncoder
from models.loader import load_model

_models = ["cbhg", "rnn"]
_default_model = "rnn"


def infer(input_file: str, output_directory: str, model_name: str):
    # create directory if not exists
    os.makedirs(output_directory, exist_ok=True)

    output_directory = Path(output_directory)

    encoder = ArabicEncoder()
    inference_data = open(input_file, "r", encoding="utf-8").readlines()
    inference_set = UndiacritizedDataset(data=inference_data, encoder=encoder)
    inference_iterator = get_dataloader(
        dataset=inference_set,
        params={
            "batch_size": CONFIG["inference_batch_size"],
            "shuffle": False,
            "num_workers": CONFIG["num_workers"],
        },
        diacritized=False,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("WARNING: CUDA not available, using CPU")
        device = torch.device("cpu")
    model = load_model(model_name, encoder).to(device)
    saved_model = torch.load(
        f"{input_file}.pt",
        map_location=device,
    )
    model.load_state_dict(saved_model["model_state_dict"])
    with open(output_directory / f"{input_file}.csv", "w") as output_file:
        csv_writer = csv.writer(output_file, lineterminator="\n")
        csv_writer.writerow(["ID", "label"])
        infer_tqdm = tqdm(
            iterable=inference_iterator,
            total=len(inference_iterator),
            leave=True,
            desc="Infering: ",
        )

        model.eval()
        with torch.no_grad():
            for batch in infer_tqdm:
                char_seq: torch.Tensor = batch["char_seq"].to(device)
                indices: list[list[int]] = batch["char_indices"]
                # forward pass
                preds = model(char_seq).contiguous().argmax(dim=-1).cpu().numpy()

                # save to file
                result = []
                for i in range(len(indices)):
                    for j in range(len(indices[i])):
                        if indices[i][j] != -1:
                            result.append([indices[i][j], preds[i][j]])
                result = sorted(result)
                csv_writer.writerows(result)

        infer_tqdm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=str,
        help="Path to input file",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to output directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=_models,
        default=_default_model,
        help="Model to use",
    )
    args = parser.parse_args()

    infer(input_file=args.input, output_directory=args.output, model_name=args.model)
