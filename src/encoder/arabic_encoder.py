import re

from encoder.vocab import CHARACTERS_LIST, DIACRITICS2ID


class ArabicEncoder:
    """Arabic text encoder."""

    def __init__(
        self,
        characters_list: list[str] = CHARACTERS_LIST,
        diacritics2id: dict[str, int] = DIACRITICS2ID,
    ):
        self.padding = "x"
        self.start = "s"
        # ensure that padding has id 0
        char_and_pad_list = [self.padding] + characters_list
        # create dicionary mapping each character to its index and vice versa
        self.char2idx = {char: idx for idx, char in enumerate(char_and_pad_list)}
        self.idx2char = {idx: char for idx, char in enumerate(char_and_pad_list)}
        # create dicionary mapping each diacritic to its index and vice versa
        self.diac2idx = diacritics2id
        self.idx2diac = {idx: diac for diac, idx in self.diac2idx.items()}

        self.valid_chars = set(diacritics2id) | set(characters_list)

        # self.start_token_id = self.diac2idx[self.start]
        self.padding_token_id = self.char2idx[self.padding]

        self.in_vocab_size = len(self.char2idx)
        self.out_vocab_size = len(self.diac2idx)

        assert self.padding_token_id == 0

    def chars_to_vector(self, chars: list[str]) -> list[str]:
        return [self.char2idx[s] for s in chars if s != self.padding]

    def diac_to_vector(self, diac: list[str]) -> list[str]:
        return [self.diac2idx[s] for s in diac if s != self.padding]

    def clean(self, text: str):
        text = "".join([c for c in text if c in self.valid_chars])
        return re.sub(r"\s+", " ", text).strip()

    def normalize_diacritic(self, diacritics: list[str]):
        reverse_diacritic = "".join(reversed(diacritics))
        normal_diacritic = "".join(diacritics)
        # check both normal and reverse diacritics
        if normal_diacritic in self.diac2idx:
            return normal_diacritic
        if reverse_diacritic in self.diac2idx:
            return reverse_diacritic
        raise ValueError(f"{diacritics} list not known diacritic")

    def extract_diacritics(self, text: str):
        if len(text) == 0:
            return text, [], []

        current_diacritics = []
        diacritics = []
        chars = []
        for char in text:
            if char in self.diac2idx:
                current_diacritics.append(char)
            else:
                diacritics.append(self.normalize_diacritic(current_diacritics))
                chars.append(char)
                current_diacritics = []

        if len(diacritics) > 0:
            diacritics.pop(0)

        diacritics.append(self.normalize_diacritic(current_diacritics))

        return text, chars, diacritics
