import inspect
import pickle
from pathlib import Path

_CURRENT_DIR = Path(inspect.getfile(inspect.currentframe())).parent

ARABIC_LETTERS = pickle.load(open(_CURRENT_DIR / "arabic_letters.pickle", "rb"))
_characters_set = {
    " ",
    "-",
    ".",
    ":",
    "،",
    "؛",
    "؟",
} | ARABIC_LETTERS
CHARACTERS_LIST = sorted(_characters_set)

DIACRITICS_CLASSES = pickle.load(open(_CURRENT_DIR / "diacritic2id.pickle", "rb"))
# sort dict into list by value
DIACRITICS_LIST: list[str] = [
    k for k, v in sorted(DIACRITICS_CLASSES.items(), key=lambda item: item[1])
]
