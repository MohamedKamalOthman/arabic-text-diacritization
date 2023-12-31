import torch
from torch.utils.data import DataLoader, Dataset

from encoder.arabic_encoder import ArabicEncoder
from encoder.vocab import ARABIC_LETTERS


class DiacritizedWordDataset(Dataset):
    def __init__(self, data, encoder: ArabicEncoder):
        self.data = []
        for item in data:
            item = encoder.clean(item)
            if len(item) > 0:
                self.data.append(item)
        self.encoder = encoder

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        text, words, chars, diacritics = self.encoder.extract_diacritics_with_words(item)
        words_vector = torch.tensor(self.encoder.words_to_vector(words))
        chars_vector = torch.tensor(self.encoder.chars_to_vector(chars))
        diacritics_vector = torch.tensor(self.encoder.diac_to_vector(diacritics))
        return words_vector, chars_vector, diacritics_vector, text

class DiacritizedDataset(Dataset):
    def __init__(self, data, encoder: ArabicEncoder):
        self.data = []
        for item in data:
            item = encoder.clean(item)
            if len(item) > 0:
                self.data.append(item)
        self.encoder = encoder

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        text, chars, diacritics = self.encoder.extract_diacritics(item)
        chars_vector = torch.tensor(self.encoder.chars_to_vector(chars))
        diacritics_vector = torch.tensor(self.encoder.diac_to_vector(diacritics))
        return chars_vector, diacritics_vector, text

class UndiacritizedDataset(Dataset):
    def __init__(self, data, encoder: ArabicEncoder):
        self.data = data
        self.encoder = encoder
        current_idx = 0
        self.indices: list[list[int]] = []
        for i in range(len(data)):
            data[i] = self.encoder.clean(data[i])
            seq = data[i]
            self.indices.append([])
            chars_idx = self.indices[-1]
            for char in seq:
                if char in ARABIC_LETTERS:
                    chars_idx.append(current_idx)
                    current_idx += 1
                else:
                    chars_idx.append(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        chars = self.data[index]
        chars_vector = torch.tensor(self.encoder.chars_to_vector(chars))
        char_indices = self.indices[index]

        return chars_vector, char_indices


def collate_undiacritized(samples):
    # sort batch by descending length to use with pack_padded_sequence
    samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
    char_seq, char_indices = zip(*samples)

    # pad sequences and extract lengths
    seq_lengths = [len(seq) for seq in char_seq]
    padded_char_seq = torch.zeros(len(char_seq), max(seq_lengths)).long()
    for i, seq in enumerate(char_seq):
        assert len(seq) == len(char_indices[i])
        padded_char_seq[i, : seq_lengths[i]] = seq

    batch = {
        "char_seq": padded_char_seq,
        "char_indices": char_indices,
        "seq_lengths": torch.LongTensor(seq_lengths),
    }
    return batch

def collate_diacritized(samples):
    # sort batch by descending length to use with pack_padded_sequence
    samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
    char_seq, diac_seq, text = zip(*samples)

    # pad sequences and extract lengths
    seq_lengths = [len(seq) for seq in char_seq]
    max_seq_length = max(seq_lengths)
    padded_char_seq = torch.zeros(len(char_seq), max_seq_length).long()
    padded_diac_seq = torch.zeros(len(diac_seq), max_seq_length).long()
    for i, (seq, diac) in enumerate(zip(char_seq, diac_seq)):
        assert len(seq) == len(diac), f"{text[i]}, {i}, {seq}, {diac}"
        padded_char_seq[i, : seq_lengths[i]] = seq
        padded_diac_seq[i, : seq_lengths[i]] = diac

    batch = {
        "char_seq": padded_char_seq,
        "diac_seq": padded_diac_seq,
        "seq_lengths": torch.LongTensor(seq_lengths),
        "text": text,
    }
    return batch

def collate_undiacritized_with_words(samples):
    # sort batch by descending length to use with pack_padded_sequence
    samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
    word_seq, char_seq, char_indices = zip(*samples)

    # pad sequences and extract lengths
    seq_lengths = [len(seq) for seq in char_seq]
    padded_words_seq = torch.zeros(len(word_seq), max(seq_lengths)).long()
    padded_char_seq = torch.zeros(len(char_seq), max(seq_lengths)).long()
    for i, (seq, word) in enumerate(char_seq, word_seq):
        assert len(seq) == len(char_indices[i])
        padded_char_seq[i, : seq_lengths[i]] = seq
        padded_words_seq[i, : seq_lengths[i]] = word

    batch = {
        "words_seq": padded_words_seq,
        "char_seq": padded_char_seq,
        "char_indices": char_indices,
        "seq_lengths": torch.LongTensor(seq_lengths),
    }
    return batch

def collate_diacritized_with_words(samples):
    # sort batch by descending length to use with pack_padded_sequence
    samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
    words_seq, char_seq, diac_seq, text = zip(*samples)

    # pad sequences and extract lengths
    seq_lengths = [len(seq) for seq in char_seq]
    max_seq_length = max(seq_lengths)
    padded_words_seq = torch.zeros(len(words_seq), max_seq_length).long()
    padded_char_seq = torch.zeros(len(char_seq), max_seq_length).long()
    padded_diac_seq = torch.zeros(len(diac_seq), max_seq_length).long()
    for i, (seq, diac, word) in enumerate(zip(char_seq, diac_seq, words_seq)):
        assert len(seq) == len(diac), f"{text[i]}, {i}, {seq}, {diac}, {word}"
        padded_char_seq[i, : seq_lengths[i]] = seq
        padded_diac_seq[i, : seq_lengths[i]] = diac
        padded_words_seq[i, : seq_lengths[i]] = word

    batch = {
        "words_seq": padded_words_seq,
        "char_seq": padded_char_seq,
        "diac_seq": padded_diac_seq,
        "seq_lengths": torch.LongTensor(seq_lengths),
        "text": text,
    }
    return batch


def get_dataloader(dataset, params, diacritized=True, with_words=False):
    collate_fn = None
    if diacritized:
        if with_words:
            collate_fn = collate_diacritized_with_words
        else:
            collate_fn = collate_diacritized
    else:
        if with_words:
            collate_fn = collate_undiacritized_with_words
        else:
            collate_fn = collate_undiacritized

    dataloader = DataLoader(
        dataset,
        collate_fn = collate_fn,
        **params,
    )
    return dataloader
