import re


class ArabicEncoder:
    """Arabic text encoder."""

    def __init__(self) -> None:
        self.padding = "x"
        self.start = "s"
        self.vocab = [' ', '-', '.', ':', '،', '؛', '؟', 'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى', 'ي']
        self.diacritics = [self.start, '', 'ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ًّ', 'ٌّ', 'ٍّ', 'َّ', 'ُّ', 'ِّ', 'ْ']
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
        valid_chars = [' ', '-', '.', ':', '،', '؛', '؟', 'ء', 'آ', 'أ', 'ؤ', 'إ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ى', 'ي', 'ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ']
        text = ''.join([c for c in text if c in valid_chars])
        return re.sub(r'\s+', ' ', text).strip()

        