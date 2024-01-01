import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CONFIG
from dataset import DiacritizerDataset, collate_diacritizer
from encoder.arabic_encoder import ArabicEncoder
from encoder.vocab import ARABIC_LETTERS
from models.loader import load_model


def diacritize(input_file: str, output_file: str, model_path: str, model_name: str):
    encoder = ArabicEncoder()
    inference_data = open(input_file, "r", encoding="utf-8").readlines()
    inference_set = DiacritizerDataset(data=inference_data, encoder=encoder)
    inference_iterator = DataLoader(
        dataset=inference_set,
        batch_size=CONFIG["inference_batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_diacritizer,
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
    with open(output_file, "w", encoding="utf-8") as of:
        diacritize_tqdm = tqdm(
            inference_iterator,
            desc="Diacritizing",
            total=len(inference_iterator),
            unit="batch",
        )
        for batch in diacritize_tqdm:
            char_seq = batch["char_seq"].to(device)
            char_indices = batch["char_indices"]
            texts_list = batch["txt"]
            texts_list = [list(text) for text in texts_list]
            # insert diacritics in text
            with torch.no_grad():
                diac_seq = model(char_seq, None)
                # get max
                diac_seq = torch.argmax(diac_seq, dim=-1)

            for i, text in enumerate(texts_list):
                for j, char_idx in enumerate(char_indices[i]):
                    if text[char_idx] in ARABIC_LETTERS:
                        text[char_idx] += encoder.idx2diac[diac_seq[i][j].item()]

            for i, text in enumerate(texts_list):
                diacritized_text = "".join(text)
                of.write(diacritized_text)


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
        "model_path",
        type=str,
        help="Model to use",
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Model name",
    )
    args = parser.parse_args()

    diacritize(
        input_file=args.input,
        output_file=args.output,
        model_path=args.model_path,
        model_name=args.model_name,
    )
