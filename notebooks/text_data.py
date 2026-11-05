"""Offline text data for notebooks 16–17 (no downloads needed).

1. load_corpus()  — ~500k characters of English technical prose: the documentation strings of
   Python's standard library (PSF-licensed), saved to data/python_docs_corpus.txt. Used to train
   tokenisers, word embeddings and a tiny GPT.
2. make_reviews() — a synthetic product-review sentiment dataset built from templates. It
   deliberately includes NEGATION ("not good", "never disappointing") and word-order effects,
   which bag-of-words models get wrong and sequence models can learn.
"""
from pathlib import Path
import importlib, random, re

DATA_DIR = Path(__file__).with_name("data")
CORPUS_PATH = DATA_DIR / "python_docs_corpus.txt"

_MODULES = """collections itertools functools json re os pathlib datetime random math statistics string textwrap
typing unittest argparse logging csv email http.client urllib.request threading subprocess socket inspect
dataclasses enum heapq bisect decimal fractions zipfile tarfile shutil tempfile glob fnmatch pickle copy pprint
abc contextlib asyncio queue sqlite3 hashlib secrets base64 struct array calendar time difflib doctest pdb
timeit trace gzip configparser optparse getopt locale gettext codecs io html xml.dom.minidom wave colorsys
imaplib smtplib ftplib uuid ipaddress mimetypes webbrowser shelve weakref types operator numbers
cmath sched signal selectors ssl platform sysconfig warnings traceback gc dis ast tokenize keyword pkgutil
zipimport runpy graphlib zoneinfo tomllib reprlib unicodedata netrc plistlib quopri
binascii mailbox concurrent.futures multiprocessing""".split()


def _build_corpus():
    docs = []
    for name in _MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for obj in [mod] + [v for k, v in sorted(vars(mod).items()) if not k.startswith("_")]:
            d = getattr(obj, "__doc__", None)
            if isinstance(d, str) and len(d) > 200:
                docs.append(re.sub(r"[ \t]+", " ", d.strip()))
    text = "\n\n".join(dict.fromkeys(docs))                 # de-duplicate, keep order
    return "".join(ch for ch in text if ord(ch) < 128)       # plain ASCII keeps the character vocabulary small


def load_corpus():
    """Return the corpus text, building and caching it on first use."""
    if not CORPUS_PATH.exists():
        DATA_DIR.mkdir(exist_ok=True)
        CORPUS_PATH.write_text(_build_corpus())
    return CORPUS_PATH.read_text()


_POS = ["good", "great", "excellent", "amazing", "solid", "reliable", "comfortable", "fast", "useful", "perfect"]
_NEG = ["bad", "terrible", "awful", "poor", "broken", "flimsy", "slow", "useless", "disappointing", "cheap"]
_THINGS = ["battery", "screen", "keyboard", "camera", "speaker", "charger", "case", "app", "mouse", "blender"]
_FILLER = ["honestly", "overall", "to be fair", "after a week", "for the price", "so far", "in my opinion", ""]


def make_reviews(n=4000, seed=0):
    """Return (texts, labels) with label 1 = positive, 0 = negative.

    Templates (all with random filler words):
      plain      "the battery is great"                   → sentiment of the adjective
      negated    "the battery is not great"               → flipped
      contrast   "the case is bad but the screen is great" → sentiment of the clause AFTER "but"
                 (the first clause is random; the second may itself be negated: "... but the screen is not great")
    """
    rnd = random.Random(seed); texts, labels = [], []
    for _ in range(n):
        kind = rnd.choice(["plain", "negated", "contrast"])
        f = rnd.choice(_FILLER); t1, t2 = rnd.sample(_THINGS, 2)
        if kind == "plain":
            pos = rnd.random() < 0.5; adj = rnd.choice(_POS if pos else _NEG)
            s, y = f"{f} the {t1} is {adj}", int(pos)
        elif kind == "negated":
            pos_adj = rnd.random() < 0.5; adj = rnd.choice(_POS if pos_adj else _NEG)
            neg_word = rnd.choice(["not", "never", "not at all", "hardly"])
            s, y = f"{f} the {t1} is {neg_word} {adj}", int(not pos_adj)
        else:
            a1 = rnd.choice(_POS + _NEG)                              # first clause: random, irrelevant
            pos2 = rnd.random() < 0.5; a2 = rnd.choice(_POS if pos2 else _NEG)
            negate = rnd.random() < 0.4                               # the deciding clause may itself be negated
            s = f"{f} the {t1} is {a1} but the {t2} is {'not ' if negate else ''}{a2}"
            y = int(pos2 != negate)
        texts.append(" ".join(s.split())); labels.append(y)
    return texts, labels
