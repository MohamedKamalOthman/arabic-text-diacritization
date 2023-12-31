import torch
from torch.utils.data import DataLoader, Dataset

from encoder.arabic_encoder import ArabicEncoder
from encoder.vocab import ARABIC_LETTERS
from features import bag_of_words, tf_idf


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


class DiacritizedDatasetWithFeatures(Dataset):
    def __init__(self, data, encoder: ArabicEncoder):
        self.encoder = encoder

        self.texts = []
        self.chars = []
        self.diacritics = []
        for item in data:
            item = self.encoder.clean(item)
            if len(item) > 0:
                text, chars, diacritics = self.encoder.extract_diacritics(item)
                self.texts.append(text)
                self.chars.append("".join(chars))
                self.diacritics.append(diacritics)

        self.encoder = encoder
        bow_vocab, self.bow = bag_of_words(self.chars)
        tfidf_vocab, self.tfidf = tf_idf(self.chars)
        self.bow2id = {bow: i for i, bow in enumerate(bow_vocab)}
        self.tfidf2id = {tfidf: i for i, tfidf in enumerate(tfidf_vocab)}
        # print("bow_vocab", len(bow_vocab))
        # print("tfidf_vocab", len(tfidf_vocab))
        # print("bow", self.bow.shape)
        # print("tfidf", self.tfidf.shape)

        # print(len(self.texts), len(self.chars), len(self.diacritics))
        print(max([len(text) for text in self.texts]))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text, chars, diacritics = (
            self.texts[index],
            self.chars[index],
            self.diacritics[index],
        )
        chars = list(chars)
        chars_vector = torch.tensor(self.encoder.chars_to_vector(chars))
        # get bow and tfidf features
        bow = [self.bow[index][self.bow2id[c]] for c in chars]
        tfidf = [self.tfidf[index][self.tfidf2id[c]] for c in chars]
        bow_vector = torch.tensor(bow)
        tfidf_vector = torch.tensor(tfidf)

        # feature concatenation each feature vector is of shape (seq_len, 3)
        feature_vector = torch.stack([chars_vector, bow_vector, tfidf_vector], dim=1)
        diacritics_vector = torch.tensor(self.encoder.diac_to_vector(diacritics))
        return feature_vector, diacritics_vector, text


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


def collate_diacritized_with_features(samples):
    # sort batch by descending length to use with pack_padded_sequence
    samples = sorted(samples, key=lambda x: len(x[0]), reverse=True)
    feature_seq, diac_seq, text = zip(*samples)

    # pad sequences and extract lengths
    seq_lengths = [len(seq) for seq in feature_seq]
    max_seq_length = max(seq_lengths)
    padded_feature_seq = torch.zeros(len(feature_seq), max_seq_length, 3).float()
    padded_diac_seq = torch.zeros(len(diac_seq), max_seq_length).long()
    for i, (seq, diac) in enumerate(zip(feature_seq, diac_seq)):
        assert len(seq) == len(diac), f"{text[i]}, {i}, {seq}, {diac}"
        padded_feature_seq[i, : seq_lengths[i]] = seq
        padded_diac_seq[i, : seq_lengths[i]] = diac

    batch = {
        "feature_seq": padded_feature_seq,
        "diac_seq": padded_diac_seq,
        "seq_lengths": torch.LongTensor(seq_lengths),
        "text": text,
    }
    return batch


def get_dataloader(dataset, params, diacritized=True):
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_diacritized_with_features
        if diacritized
        else collate_undiacritized,
        **params,
    )
    return dataloader
