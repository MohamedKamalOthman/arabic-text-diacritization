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


def infer(input_file: str, output_directory: str, model_path: str, model_name: str):
    # create directory if not exists
    os.makedirs(output_directory, exist_ok=True)

    output_directory = Path(output_directory)

    encoder = ArabicEncoder()
    inference_data = open(input_file, "r", encoding="utf-8").readlines()
    inference_set = UndiacritizedDataset(data=inference_data, encoder=encoder)
    with open(
        output_directory / f"{input_file}-inputs.csv", "w", encoding="utf-8"
    ) as input_file_csv:
        csv_writer = csv.writer(input_file_csv, lineterminator="\n")
        csv_writer.writerow(["id", "line_number", "letter"])
        for i, (chars, indices) in enumerate(inference_set):
            for j, letter in enumerate(chars):
                if indices[j] != -1:
                    csv_writer.writerow(
                        [indices[j], i, encoder.idx2char[letter.item()]]
                    )
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
        model_path,
        map_location=device,
    )
    model.load_state_dict(saved_model["model_state_dict"])
    with open(
        output_directory / f"{input_file}.csv", "w", encoding="utf-8"
    ) as output_file:
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
                            result.append([indices[i][j], preds[i][j] - 1])
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
        "model",
        type=str,
        help="Model to use",
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Name of model",
    )
    args = parser.parse_args()

    infer(
        input_file=args.input,
        output_directory=args.output,
        model_path=args.model,
        model_name=args.model_name,
    )
