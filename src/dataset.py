import torch
from torch.utils.data import DataLoader, Dataset

from encoder.arabic_encoder import ArabicEncoder


class DiacritizerDataset(Dataset):
    def __init__(self, data, encoder: ArabicEncoder):
        self.data = data
        self.encoder = encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        item = self.encoder.clean(item)
        text, chars, diacritics = self.encoder.extract_diacritics(item)
        chars_vector = torch.tensor(self.encoder.chars_to_vector(chars))
        diacritics_vector = torch.tensor(self.encoder.diac_to_vector(diacritics))
        return chars_vector, diacritics_vector, text


def collate_fn(samples):
    # sort batch by descending length to use with pack_padded_sequence
    samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
    char_seq, diac_seq, text = zip(*samples)

    # pad sequences and extract lengths
    seq_lengths = [len(seq) for seq in char_seq]
    padded_char_seq = torch.zeros(len(char_seq), max(seq_lengths)).long()
    padded_diac_seq = torch.zeros(len(diac_seq), max(seq_lengths)).long()
    for i, (seq, diac) in enumerate(zip(char_seq, diac_seq)):
        assert len(seq) == len(diac)
        padded_char_seq[i, : seq_lengths[i]] = seq
        padded_diac_seq[i, : seq_lengths[i]] = diac

    batch = {
        "char_seq": padded_char_seq,
        "diac_seq": padded_diac_seq,
        "seq_lengths": torch.LongTensor(seq_lengths),
        "text": text,
    }
    return batch


def get_dataloader(dataset, params):
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        **params,
    )
    return dataloader
