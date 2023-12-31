import re
import gensim

from encoder.vocab import ARABIC_LETTERS, CHARACTERS_LIST, DIACRITICS_LIST


class ArabicEncoder:
    """Arabic text encoder."""

    def __init__(
        self,
        characters_list: list[str] = CHARACTERS_LIST,
        diacritics_list: list[str] = DIACRITICS_LIST,
    ):
        self.word_embedding = gensim.models.Word2Vec.load('./encoder/full_grams_cbow_100_twitter.mdl')
        # self.word_embedding = gensim.models.Word2Vec.load()
        self.padding = "x"
        # self.start = "s"
        # ensure that padding has id 0
        char_and_pad_list = [self.padding] + characters_list
        diac_and_pad_list = [self.padding] + diacritics_list
        # create dicionary mapping each character to its index and vice versa
        self.char2idx = {char: idx for idx, char in enumerate(char_and_pad_list)}
        self.idx2char = {idx: char for idx, char in enumerate(char_and_pad_list)}
        # create dicionary mapping each diacritic to its index and vice versa
        self.diac2idx = {diac: idx for idx, diac in enumerate(diac_and_pad_list)}
        self.idx2diac = {idx: diac for idx, diac in enumerate(diac_and_pad_list)}

        self.valid_chars = set(diacritics_list) | set(characters_list)

        # self.start_token_id = self.diac2idx[self.start]
        self.padding_token_id = self.diac2idx[self.padding]

        self.in_vocab_size = len(self.char2idx)
        self.out_vocab_size = len(self.diac2idx)

        self.arabic_ids = [
            self.char2idx[c] for c in self.char2idx if c in ARABIC_LETTERS
        ]

        assert self.padding_token_id == 0

    def chars_to_vector(self, chars: list[str]) -> list[str]:
        return [self.char2idx[s] for s in chars if s != self.padding]

    def words_to_vector(self, words: list[str]) -> list[str]:
        count = 0
        for i in range(len(words)):
            if words[i] in self.word_embedding.wv:
                words[i] = self.word_embedding.wv[words[i]]
            else:
                count += 1
                words[i] = self.word_embedding.wv['مصر']
        # print(count * 100 / len(words))
        return words
        
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

    def remove_diacritics(self, text: str):
        result = ""
        for char in text:
            if char not in self.diac2idx:
                result += char
        return result
    
    def extract_diacritics_with_words(self, text: str):
        if len(text) == 0:
            return text, [], [], []

        current_diacritics = []
        diacritics = []
        chars = []
        words = []
        split_text = text.split(" ")
        for word in split_text:
            new_word = self.remove_diacritics(word)
            for char in word:
                if char in self.diac2idx:
                    current_diacritics.append(char)
                else:
                    diacritics.append(self.normalize_diacritic(current_diacritics))
                    chars.append(char)
                    words.append(new_word)
                    current_diacritics = []
        
        if len(diacritics) > 0:
            diacritics.pop(0)

        diacritics.append(self.normalize_diacritic(current_diacritics))
        # print(len(words), len(chars), len(diacritics))

        return text, words, chars, diacritics

    def combine_chars_diac(self, chars: list[str], diac: list[str]) -> list[str]:
        output = ""
        for i, input_id in enumerate(chars):
            if input_id == self.padding_token_id:
                break
            output += self.idx2char[input_id]
            output += self.idx2diac[diac[i]]
        return output
