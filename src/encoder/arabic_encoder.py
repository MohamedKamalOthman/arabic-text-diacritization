import re

from vocab import CHARACTERS, DIACRITICS


class ArabicEncoder:
    """Arabic text encoder."""

    def __init__(
        self, vocab: list[str] = CHARACTERS, diacritics: list[str] = DIACRITICS
    ):
        self.padding = "x"
        self.start = "s"
        self.vocab = vocab
        self.diacritics = diacritics + [self.start]
        # create dicionary mapping each character to its index and vice versa
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}
        # create dicionary mapping each diacritic to its index and vice versa
        self.diac2idx = {diac: idx for idx, diac in enumerate(self.diacritics)}
        self.idx2diac = {idx: diac for idx, diac in enumerate(self.diacritics)}

    def input_to_vector(self, input: str) -> list[str]:
        return [self.char2idx[s] for s in input if s != self.padding]

    def diac_to_vector(self, diac: str) -> list[str]:
        return [self.diac2idx[s] for s in diac if s != self.padding]

    def clean(self, text: str):
        valid_chars = set(self.vocab + self.diacritics)
        text = "".join([c for c in text if c in valid_chars])
        return re.sub(r"\s+", " ", text).strip()

    def normalize_diacritic(self, diacritics: list[str]):
        reverse_diacritic = "".join(reversed(diacritics))
        normal_diacritic = "".join(diacritics)
        # check both normal and reverse diacritics
        if normal_diacritic in DIACRITICS:
            return normal_diacritic
        if reverse_diacritic in DIACRITICS:
            return reverse_diacritic
        raise ValueError(f"{diacritics} list not known diacritic")

    def extract_diacritics(self, text: str):
        current_diacritics = []
        diacritics = []
        chars = []
        for char in text:
            if char in DIACRITICS:
                current_diacritics.append(char)
            else:
                diacritics.append(self.normalize_diacritic(current_diacritics))
                chars.append(char)
                current_diacritics = []

        if len(diacritics):
            del diacritics[0]

        diacritics.append(self.normalize_diacritic(current_diacritics))

        return text, chars, diacritics
