import inspect
import pickle
from pathlib import Path

_CURRENT_DIR = Path(inspect.getfile(inspect.currentframe())).parent


_characters_set = {
    " ",
    "-",
    ".",
    ":",
    "،",
    "؛",
    "؟",
} | pickle.load(open(_CURRENT_DIR / "arabic_letters.pickle", "rb"))
CHARACTERS2ID = {char: idx for idx, char in enumerate(sorted(_characters_set))}


DIACRITICS2ID: dict[str, str] = pickle.load(
    open(_CURRENT_DIR / "diacritic2id.pickle", "rb")
)
