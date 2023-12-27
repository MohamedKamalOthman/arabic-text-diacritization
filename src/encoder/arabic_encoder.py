import re

from encoder.vocab import CHARACTERS, DIACRITICS


class ArabicEncoder:
    """Arabic text encoder."""

    def __init__(
        self, vocab: list[str] = CHARACTERS, diacritics: list[str] = DIACRITICS
    ):
        self.padding = "x"
        self.start = "s"
        self.vocab = vocab
        self.diacritics = diacritics + [self.start]
        self.diacritics_set = set(diacritics)
        # create dicionary mapping each character to its index and vice versa
        self.char2idx = {
            char: idx for idx, char in enumerate(self.vocab + [self.padding])
        }
        self.idx2char = {
            idx: char for idx, char in enumerate(self.vocab + [self.padding])
        }
        # create dicionary mapping each diacritic to its index and vice versa
        self.diac2idx = {diac: idx for idx, diac in enumerate(self.diacritics)}
        self.idx2diac = {idx: diac for idx, diac in enumerate(self.diacritics)}

        self.start_token_id = self.diac2idx[self.start]
        self.padding_token_id = self.char2idx[self.padding]

    def chars_to_vector(self, chars: list[str]) -> list[str]:
        return [self.char2idx[s] for s in chars if s != self.padding]

    def diac_to_vector(self, diac: list[str]) -> list[str]:
        return [self.diac2idx[s] for s in diac if s != self.padding]

    def clean(self, text: str):
        valid_chars = set(self.vocab + self.diacritics)
        text = "".join([c for c in text if c in valid_chars])
        return re.sub(r"\s+", " ", text).strip()

    def normalize_diacritic(self, diacritics: list[str]):
        reverse_diacritic = "".join(reversed(diacritics))
        normal_diacritic = "".join(diacritics)
        # check both normal and reverse diacritics
        if normal_diacritic in self.diacritics_set:
            return normal_diacritic
        if reverse_diacritic in self.diacritics_set:
            return reverse_diacritic
        raise ValueError(f"{diacritics} list not known diacritic")

    def extract_diacritics(self, text: str):
        current_diacritics = []
        diacritics = []
        chars = []
        for char in text:
            if char in self.diacritics_set:
                current_diacritics.append(char)
            else:
                diacritics.append(self.normalize_diacritic(current_diacritics))
                chars.append(char)
                current_diacritics = []

        if len(diacritics) > 0:
            diacritics.pop(0)

        diacritics.append(self.normalize_diacritic(current_diacritics))

        return text, chars, diacritics
