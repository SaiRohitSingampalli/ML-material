"""
================================================================================
 NLP & LLMs FROM SCRATCH — pure Python, zero libraries
================================================================================
 Only `math`, `random`, `re` and `collections`. No NLTK, no spaCy, no
 HuggingFace, no NumPy. A small ORIGINAL corpus is built in so everything runs
 offline.

 CONTENTS
 --------
 1. Text processing ....... normalization, regex tokenization, stop words,
                            suffix-stripping stemmer, Levenshtein edit
                            distance, noisy-channel spelling correction
 2. Subword tokenizers .... Byte-Pair Encoding (train/encode/decode),
                            WordPiece (likelihood-scored merges + greedy
                            longest-match encoding), character fallback
 3. Language models ....... n-gram MLE, add-k smoothing, interpolated
                            Kneser-Ney, perplexity
 4. Decoding .............. greedy, beam search (with length penalty),
                            temperature, top-k, top-p (nucleus), repetition
                            penalty
 5. Representations ....... bag of words, TF-IDF, BM25, PPMI + truncated SVD
                            embeddings, skip-gram with negative sampling,
                            cosine similarity, analogy, mean-pooled sentence
                            vectors
 6. Classic NLP tasks ..... Naive Bayes and logistic-regression text
                            classifiers, HMM part-of-speech tagger (Viterbi)
 7. LLM internals ......... RoPE, KV-cache cost accounting, int8/int4
                            quantization, LoRA parameter math, attention
                            temperature/entropy
 8. RAG ................... chunking with overlap, BM25 + dense retrieval,
                            reciprocal rank fusion, reranking, prompt assembly
 9. Evaluation ............ BLEU, ROUGE-1/2/L, exact match & token F1,
                            precision@k, recall@k, MRR, nDCG
 10. Demos ................ everything above, trained and measured
================================================================================
"""

import math
import random
import re
from collections import Counter, defaultdict

random.seed(0)
EPS = 1e-12


# =============================================================================
# BUILT-IN CORPORA (original text, written for this file)
# =============================================================================
CORPUS = [
    # animals
    "the cat sat on the warm mat near the window",
    "a small cat chased the grey mouse across the kitchen",
    "the dog barked at the cat in the garden",
    "my dog loves to run in the park every morning",
    "the puppy and the kitten played together all afternoon",
    "a loyal dog waited at the door for its owner",
    "the kitten slept on the soft blanket by the fire",
    "cats and dogs are the most popular pets in the city",
    # cooking
    "she baked fresh bread in the oven this morning",
    "the chef added garlic and onion to the hot pan",
    "we cooked pasta with tomato sauce for dinner",
    "fresh bread tastes best with butter and honey",
    "the soup needs more salt and a little pepper",
    "he chopped the onion and fried it in olive oil",
    "the oven must be hot before you bake the bread",
    "tomato sauce and garlic make a simple pasta dinner",
    # space
    "the rocket launched into orbit around the earth",
    "astronauts live on the space station for months",
    "the telescope captured images of a distant galaxy",
    "the moon orbits the earth once every month",
    "a new rocket engine will carry astronauts to the moon",
    "stars in the galaxy shine with different colors",
    "the space station orbits the earth every ninety minutes",
    "scientists use the telescope to study distant stars",
    # finance
    "the bank raised interest rates to slow inflation",
    "investors bought shares when the market fell",
    "the stock market rose after the bank report",
    "high inflation reduces the value of savings",
    "the company reported strong profits this quarter",
    "interest rates affect loans and mortgages",
    "investors watch the market and the interest rates closely",
    "the bank lends money to the company at a low rate",
    # sport
    "the team scored a late goal to win the match",
    "fans cheered loudly as the players entered the stadium",
    "the coach praised the team after the final match",
    "she trained every day to win the race",
    "the players ran onto the field before the match",
    "the striker scored twice in the second half",
    "the stadium was full for the championship final",
    "the runner finished the race in record time",
]

TOPICS = (["animals"] * 8 + ["cooking"] * 8 + ["space"] * 8 +
          ["finance"] * 8 + ["sport"] * 8)

SENTIMENT = [
    ("the food was delicious and the service was excellent", 1),
    ("i loved the movie the acting was brilliant", 1),
    ("what a wonderful day everything went perfectly", 1),
    ("the staff were friendly and very helpful", 1),
    ("great product works exactly as described", 1),
    ("the hotel room was clean and comfortable", 1),
    ("an amazing performance i would watch it again", 1),
    ("fast delivery and the quality is superb", 1),
    ("the view was beautiful and the price was fair", 1),
    ("highly recommend this friendly little cafe", 1),
    ("brilliant story with a happy ending", 1),
    ("the new update is fast and reliable", 1),
    ("the food was cold and the service was terrible", 0),
    ("i hated the movie the plot was boring", 0),
    ("what an awful day everything went wrong", 0),
    ("the staff were rude and very unhelpful", 0),
    ("poor product broke after one day", 0),
    ("the hotel room was dirty and noisy", 0),
    ("a dull performance i walked out early", 0),
    ("slow delivery and the quality is poor", 0),
    ("the view was ugly and the price was absurd", 0),
    ("never visit this unfriendly little cafe", 0),
    ("boring story with a sad pointless ending", 0),
    ("the new update is slow and buggy", 0),
]

# tiny hand-tagged POS corpus (DET NOUN VERB ADJ ADP PRON ADV)
POS_CORPUS = [
    [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB"), ("on", "ADP"),
     ("the", "DET"), ("mat", "NOUN")],
    [("a", "DET"), ("dog", "NOUN"), ("runs", "VERB"), ("fast", "ADV")],
    [("the", "DET"), ("small", "ADJ"), ("dog", "NOUN"), ("barked", "VERB")],
    [("she", "PRON"), ("baked", "VERB"), ("fresh", "ADJ"), ("bread", "NOUN")],
    [("the", "DET"), ("chef", "NOUN"), ("cooked", "VERB"), ("the", "DET"),
     ("soup", "NOUN")],
    [("he", "PRON"), ("runs", "VERB"), ("in", "ADP"), ("the", "DET"),
     ("park", "NOUN")],
    [("the", "DET"), ("rocket", "NOUN"), ("launched", "VERB"),
     ("quickly", "ADV")],
    [("a", "DET"), ("bright", "ADJ"), ("star", "NOUN"), ("shines", "VERB")],
    [("they", "PRON"), ("watched", "VERB"), ("the", "DET"), ("match", "NOUN")],
    [("the", "DET"), ("team", "NOUN"), ("won", "VERB"), ("the", "DET"),
     ("final", "ADJ"), ("match", "NOUN")],
    [("she", "PRON"), ("sat", "VERB"), ("on", "ADP"), ("a", "DET"),
     ("soft", "ADJ"), ("chair", "NOUN")],
    [("the", "DET"), ("bank", "NOUN"), ("raised", "VERB"), ("rates", "NOUN")],
    [("a", "DET"), ("cat", "NOUN"), ("slept", "VERB"), ("quietly", "ADV")],
    [("the", "DET"), ("warm", "ADJ"), ("bread", "NOUN"), ("smells", "VERB"),
     ("good", "ADJ")],
    [("he", "PRON"), ("cooked", "VERB"), ("in", "ADP"), ("the", "DET"),
     ("kitchen", "NOUN")],
    [("the", "DET"), ("players", "NOUN"), ("ran", "VERB"), ("fast", "ADV")],
]

# documents for the retrieval / RAG demo (original text)
KNOWLEDGE_BASE = {
    "doc_bpe": (
        "Byte pair encoding builds a subword vocabulary by repeatedly merging "
        "the most frequent adjacent pair of symbols. It starts from single "
        "characters, so any word can be encoded and there are no unknown "
        "tokens. The number of merges sets the vocabulary size."),
    "doc_rope": (
        "Rotary position embedding rotates query and key vectors by an angle "
        "that depends on the token position. Because both are rotated, their "
        "dot product depends only on the relative distance between tokens. "
        "Most modern large language models use rotary embeddings."),
    "doc_kv": (
        "During generation a model caches the keys and values of previous "
        "tokens. With this key value cache each new token only needs one "
        "attention row instead of recomputing the whole prefix. The cache "
        "grows linearly with context length and dominates memory at long "
        "context."),
    "doc_lora": (
        "Low rank adaptation freezes the pretrained weights and learns a "
        "small update written as the product of two thin matrices. With rank "
        "eight the trainable parameters drop by more than ninety nine percent "
        "while quality stays close to full fine tuning."),
    "doc_quant": (
        "Quantization stores weights with fewer bits, for example eight bit "
        "or four bit integers instead of sixteen bit floats. A scale factor "
        "maps integers back to real values. Outlier weights make quantization "
        "harder, which is why per channel or per group scales are used."),
    "doc_rag": (
        "Retrieval augmented generation fetches relevant passages from a "
        "document store and places them in the prompt. It reduces "
        "hallucination and lets the model use knowledge that changes after "
        "training without retraining the model."),
    "doc_bm25": (
        "BM25 is a ranking function for keyword search. It rewards rare terms "
        "through inverse document frequency, saturates repeated terms, and "
        "normalizes for document length. It remains a strong baseline next to "
        "dense retrieval."),
    "doc_temp": (
        "Sampling temperature divides the logits before the softmax. Low "
        "temperature makes the output more deterministic and repetitive, "
        "while high temperature makes it more diverse but less coherent. Top p "
        "sampling keeps the smallest set of tokens whose probability exceeds "
        "p."),
}


# =============================================================================
# 1. TEXT PROCESSING
# =============================================================================
STOP_WORDS = set("""a an the and or but if of at by for with about against between
into through during before after above below to from up down in out on off over
under again further then once here there when where why how all any both each few
more most other some such no nor not only own same so than too very s t can will
just don should now is are was were be been being have has had having do does did
i me my we our you your he him his she her it its they them their what which who
this that these those am""".split())


def normalize(text):
    """
    NORMALIZATION: lowercase, strip accents-lite, collapse whitespace.
    Every choice here is a trade-off: lowercasing merges "Apple" (company) with
    "apple" (fruit); stripping punctuation destroys "U.S." and emoticons. Modern
    subword tokenizers mostly skip normalization and let the model learn it.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text):
    """Regex word tokenizer: keeps contractions like "don't" as one token."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def remove_stopwords(tokens):
    """
    Stop words carry little topical content, so removing them helps BoW
    retrieval and topic models. It HURTS anything that needs syntax or
    negation: removing "not" flips "not good" into "good". Transformers keep
    everything.
    """
    return [t for t in tokens if t not in STOP_WORDS]


def stem(word):
    """
    A simplified SUFFIX-STRIPPING stemmer in the spirit of Porter (1980).

    Stemming chops suffixes by rule: fast, no dictionary, but crude and
    sometimes wrong ("university" and "universe" collapse together; the stem
    need not be a real word). LEMMATIZATION instead maps to the dictionary form
    using a vocabulary and the part of speech ("better" -> "good", "ran" ->
    "run") — accurate but slower and language-specific.
    """
    w = word
    if len(w) <= 3:
        return w
    for suf, rep in [("ational", "ate"), ("tional", "tion"), ("ization", "ize"),
                     ("fulness", "ful"), ("ousness", "ous"), ("iveness", "ive"),
                     ("ing", ""), ("edly", ""), ("ed", ""), ("ies", "y"),
                     ("ly", ""), ("ness", ""), ("ment", ""), ("es", ""),
                     ("s", "")]:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)] + rep
            break
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "lsz":
        w = w[:-1]                                  # running -> runn -> run
    return w


def edit_distance(a, b, sub_cost=1, return_ops=False):
    """
    LEVENSHTEIN DISTANCE by dynamic programming.

        D[i][j] = min( D[i-1][j]   + 1          deletion
                       D[i][j-1]   + 1          insertion
                       D[i-1][j-1] + (a_i != b_j) * sub_cost )   substitution

    O(len(a) * len(b)) time. The same DP skeleton, with different costs and a
    max instead of a min, is Needleman-Wunsch sequence alignment in biology and
    the core of word error rate (WER) in speech recognition.
    """
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1,
                          D[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else sub_cost))
    if not return_ops:
        return D[n][m]
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + \
                (0 if a[i - 1] == b[j - 1] else sub_cost):
            ops.append("keep " + a[i - 1] if a[i - 1] == b[j - 1]
                       else f"sub {a[i - 1]}->{b[j - 1]}")
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append("del " + a[i - 1])
            i -= 1
        else:
            ops.append("ins " + b[j - 1])
            j -= 1
    return D[n][m], ops[::-1]


class SpellCorrector:
    """
    NOISY-CHANNEL SPELLING CORRECTION (the Norvig formulation).

        correction = argmax_c  P(c) * P(typo | c)
                              prior    channel model

    P(c): word frequency from a corpus (a unigram language model).
    P(typo | c): approximated by edit distance — candidates at distance 1 are
    strongly preferred over distance 2.

    This is Bayes' rule doing real work, and it is the same decomposition used
    by classical speech recognition and machine translation: a language model
    times a channel model.
    """

    def __init__(self, corpus_tokens):
        self.freq = Counter(corpus_tokens)
        self.total = sum(self.freq.values())
        self.letters = "abcdefghijklmnopqrstuvwxyz"

    def _edits1(self, w):
        splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits if b for c in self.letters]
        inserts = [a + c + b for a, b in splits for c in self.letters]
        return set(deletes + transposes + replaces + inserts)

    def correct(self, w):
        if w in self.freq:
            return w
        e1 = sorted(c for c in self._edits1(w) if c in self.freq)
        if e1:
            return max(e1, key=lambda c: self.freq[c])
        e2 = sorted({c for e in self._edits1(w) for c in self._edits1(e)
                     if c in self.freq})
        if e2:
            return max(e2, key=lambda c: self.freq[c])
        return w


# =============================================================================
# 2. SUBWORD TOKENIZERS
# =============================================================================
class BPETokenizer:
    """
    BYTE-PAIR ENCODING (Sennrich et al. 2016; the GPT family's tokenizer).

    TRAIN
      1. Split every word into characters plus an end-of-word marker </w>.
      2. Count every adjacent symbol pair, weighted by word frequency.
      3. Merge the MOST FREQUENT pair into a new symbol. Record the merge.
      4. Repeat for `n_merges` rounds. Vocab size = base chars + merges.

    ENCODE a new word by replaying the learned merges IN THE ORDER THEY WERE
    LEARNED (merge rank), not by greedily matching the longest vocab entry.

    WHY SUBWORDS
      * Word-level vocabularies explode and still hit out-of-vocabulary words.
      * Character-level has no OOV but makes sequences ~4-5x longer, and
        attention is O(T^2).
      * Subwords sit between: frequent words stay whole ("the"), rare words
        decompose into reusable pieces ("un" + "believ" + "able"), and since
        the base alphabet is included NOTHING is ever out of vocabulary.
        Byte-level BPE (GPT-2) uses the 256 bytes as the base alphabet, so it
        can encode any string in any script.

    CONSEQUENCES people ask about: token counts drive API cost and context
    usage; languages under-represented in the training data get split into
    more tokens (a "tokenizer tax"); models struggle with character-level tasks
    ("how many r's in strawberry") because they never see characters.
    """

    def __init__(self, n_merges=100):
        self.n_merges = n_merges
        self.merges = []
        self.ranks = {}

    @staticmethod
    def _pairs(symbols):
        return {(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)}

    def train(self, texts, verbose_first=0):
        word_freq = Counter(w for t in texts for w in tokenize(t))
        vocab = {tuple(w) + ("</w>",): f for w, f in word_freq.items()}
        self.base = sorted({c for w in vocab for c in w})
        for i in range(self.n_merges):
            pair_counts = Counter()
            for word, f in vocab.items():
                for j in range(len(word) - 1):
                    pair_counts[(word[j], word[j + 1])] += f
            if not pair_counts:
                break
            best, count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
            if verbose_first and i < verbose_first:
                print(f"     merge {i + 1:2d}: {best[0]!r} + {best[1]!r} "
                      f"-> {best[0] + best[1]!r}   (count {count})")
            self.merges.append(best)
            new_vocab = {}
            for word, f in vocab.items():
                out, j = [], 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j + 1]) == best:
                        out.append(word[j] + word[j + 1])
                        j += 2
                    else:
                        out.append(word[j])
                        j += 1
                new_vocab[tuple(out)] = f
            vocab = new_vocab
        self.ranks = {p: i for i, p in enumerate(self.merges)}
        self.vocab = set(self.base) | {a + b for a, b in self.merges}
        return self

    def encode_word(self, word):
        symbols = list(word) + ["</w>"]
        while len(symbols) > 1:
            pairs = [(self.ranks.get((symbols[i], symbols[i + 1]), 1e9), i)
                     for i in range(len(symbols) - 1)]
            rank, i = min(pairs)
            if rank == 1e9:
                break                               # no learned merge applies
            symbols = symbols[:i] + [symbols[i] + symbols[i + 1]] + symbols[i + 2:]
        return symbols

    def encode(self, text):
        return [s for w in tokenize(text) for s in self.encode_word(w)]

    @staticmethod
    def decode(tokens):
        return "".join(tokens).replace("</w>", " ").strip()


class WordPieceTokenizer:
    """
    WORDPIECE (BERT's tokenizer).

    Two differences from BPE:

    1. MERGE SCORE. Instead of raw pair frequency, WordPiece merges the pair
       that most increases the corpus LIKELIHOOD under a unigram model:

           score(a, b) = freq(ab) / ( freq(a) * freq(b) )

       i.e. pointwise mutual information. It prefers pairs that occur together
       MORE than chance predicts, not merely pairs that are common because both
       halves are common.

    2. ENCODING is GREEDY LONGEST-MATCH-FIRST: take the longest vocabulary
       prefix, then continue on the remainder with "##"-prefixed continuation
       pieces. If no piece matches, emit [UNK].

    SentencePiece (T5, LLaMA) differs again: it treats the input as a raw
    stream including spaces (marked with a special character), so it needs no
    pre-tokenization and is language-agnostic; its Unigram variant starts from
    a large vocabulary and PRUNES pieces by likelihood.
    """

    def __init__(self, n_merges=100):
        self.n_merges = n_merges

    def train(self, texts):
        word_freq = Counter(w for t in texts for w in tokenize(t))
        # word split: first char plain, continuation chars get ##
        vocab = {tuple([w[0]] + ["##" + c for c in w[1:]]): f
                 for w, f in word_freq.items()}
        self.vocab = {s for w in vocab for s in w}
        for _ in range(self.n_merges):
            sym_freq, pair_freq = Counter(), Counter()
            for word, f in vocab.items():
                for s in word:
                    sym_freq[s] += f
                for j in range(len(word) - 1):
                    pair_freq[(word[j], word[j + 1])] += f
            if not pair_freq:
                break
            best = max(sorted(pair_freq), key=lambda p: (pair_freq[p] /
                                                         (sym_freq[p[0]] * sym_freq[p[1]]),
                                                         pair_freq[p]))
            merged = best[0] + best[1][2:]          # "##" dropped from the right
            self.vocab.add(merged)
            new_vocab = {}
            for word, f in vocab.items():
                out, j = [], 0
                while j < len(word):
                    if j < len(word) - 1 and (word[j], word[j + 1]) == best:
                        out.append(merged)
                        j += 2
                    else:
                        out.append(word[j])
                        j += 1
                new_vocab[tuple(out)] = f
            vocab = new_vocab
        return self

    def encode_word(self, word):
        pieces, start = [], 0
        while start < len(word):
            end, piece = len(word), None
            while start < end:
                cand = word[start:end] if start == 0 else "##" + word[start:end]
                if cand in self.vocab:
                    piece = cand
                    break
                end -= 1
            if piece is None:
                return ["[UNK]"]
            pieces.append(piece)
            start = end
        return pieces

    def encode(self, text):
        return [p for w in tokenize(text) for p in self.encode_word(w)]


# =============================================================================
# 3. N-GRAM LANGUAGE MODELS
# =============================================================================
BOS, EOS = "<s>", "</s>"


class NGramLM:
    """
    N-GRAM LANGUAGE MODEL

    Chain rule, then the MARKOV assumption (condition on the last n-1 words):
        P(w_1..w_T) = prod_t P(w_t | w_1..w_(t-1))
                   ~= prod_t P(w_t | w_(t-n+1)..w_(t-1))

    MLE:        P(w | h) = c(h, w) / c(h)
    Problem:    any unseen n-gram gets probability 0, so any test sentence
                containing one gets probability 0 and INFINITE perplexity.

    SMOOTHING moves probability mass from seen to unseen events:
      add-k      P = (c(h,w) + k) / (c(h) + k|V|)      simple, crude; k=1 is
                 Laplace and badly over-smooths large vocabularies
      Kneser-Ney (interpolated, bigram):
                 P_KN(w|h) = max(c(h,w) - d, 0)/c(h) + lambda(h) * P_cont(w)
                 lambda(h) = d * N1+(h, .) / c(h)       (the discounted mass)
                 P_cont(w) = N1+(., w) / N1+(., .)
       The key idea is P_cont — the CONTINUATION probability: how many
       DIFFERENT contexts w appears after, not how often it appears. "Francisco"
       is frequent but almost only follows "San", so it should be a poor guess
       in a novel context. KN was the best n-gram smoother for decades.

    PERPLEXITY = exp( -(1/N) sum log P(w_t | h_t) )
      = the exponentiated cross-entropy = the effective branching factor, i.e.
      "the model is as confused as if it chose uniformly among PP words". Lower
      is better. Perplexities are only comparable with the SAME tokenizer and
      vocabulary.
    """

    def __init__(self, n=2, smoothing="kn", k=0.1, discount=0.75):
        self.n, self.smoothing, self.k, self.d = n, smoothing, k, discount

    def fit(self, sentences):
        self.counts = Counter()
        self.ctx = Counter()
        self.vocab = set([EOS])
        for s in sentences:
            toks = [BOS] * (self.n - 1) + tokenize(s) + [EOS]
            self.vocab.update(toks[self.n - 1:])
            for i in range(self.n - 1, len(toks)):
                h = tuple(toks[i - self.n + 1:i])
                self.counts[(h, toks[i])] += 1
                self.ctx[h] += 1
        # Kneser-Ney continuation statistics
        self.cont = Counter()                     # N1+(., w)
        self.follow = Counter()                   # N1+(h, .)
        for (h, w), c in self.counts.items():
            self.cont[w] += 1
            self.follow[h] += 1
        self.n_bigram_types = len(self.counts)
        self.V = len(self.vocab)
        return self

    def prob(self, w, h):
        h = tuple(h[-(self.n - 1):]) if self.n > 1 else ()
        c_hw, c_h = self.counts.get((h, w), 0), self.ctx.get(h, 0)
        if self.smoothing == "mle":
            return c_hw / c_h if c_h else 0.0
        if self.smoothing == "addk":
            return (c_hw + self.k) / (c_h + self.k * self.V)
        # interpolated Kneser-Ney
        p_cont = (self.cont.get(w, 0) + 0.5) / (self.n_bigram_types + 0.5 * self.V)
        if c_h == 0:
            return p_cont
        lam = self.d * self.follow[h] / c_h
        return max(c_hw - self.d, 0) / c_h + lam * p_cont

    def next_distribution(self, h):
        # iterate in SORTED order: Python randomizes string-set ordering per
        # process (PYTHONHASHSEED), so iterating a raw set makes argmax ties —
        # and therefore greedy output — change from run to run
        return {w: self.prob(w, h) for w in sorted(self.vocab)}

    def perplexity(self, sentences):
        logp, N = 0.0, 0
        for s in sentences:
            toks = [BOS] * (self.n - 1) + tokenize(s) + [EOS]
            for i in range(self.n - 1, len(toks)):
                p = self.prob(toks[i], toks[:i])
                logp += math.log(p) if p > 0 else -float("inf")
                N += 1
        return math.exp(-logp / N) if logp > -float("inf") else float("inf")


# =============================================================================
# 4. DECODING STRATEGIES
# =============================================================================
def _normalize(dist):
    s = sum(dist.values())
    return {w: p / s for w, p in dist.items()} if s > 0 else dist


def apply_temperature(dist, T):
    """
    TEMPERATURE: p_i^(1/T), renormalized  (== softmax(logits / T)).
      T -> 0   : collapses onto the argmax (greedy)
      T = 1    : the model's distribution unchanged
      T > 1    : flattens toward uniform — more diverse, less coherent
    """
    if T <= 1e-6:
        best = max(dist, key=dist.get)
        return {w: (1.0 if w == best else 0.0) for w in dist}
    return _normalize({w: p ** (1.0 / T) for w, p in dist.items() if p > 0})


def top_k_filter(dist, k):
    """Keep the k most likely tokens, renormalize. A FIXED-size candidate set —
    too permissive when the model is confident, too strict when it isn't."""
    keep = sorted(dist, key=dist.get, reverse=True)[:k]
    return _normalize({w: dist[w] for w in keep})


def top_p_filter(dist, p):
    """
    NUCLEUS (top-p) SAMPLING (Holtzman et al. 2019): keep the SMALLEST set of
    tokens whose cumulative probability >= p. The candidate set ADAPTS: when the
    model is confident the nucleus is 1-2 tokens, when it is uncertain it widens.
    That adaptivity is why it replaced top-k as the default. It exists because
    the long unreliable tail of a softmax, summed, holds real probability mass —
    sample from it often enough and generation derails ("neural text
    degeneration").
    """
    items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    out, cum = {}, 0.0
    for w, pw in items:
        out[w] = pw
        cum += pw
        if cum >= p:
            break
    return _normalize(out)


def sample_from(dist, rng):
    r, acc = rng.random(), 0.0
    for w, p in dist.items():
        acc += p
        if r <= acc:
            return w
    return max(dist, key=dist.get)


def generate(lm, prompt, max_len=12, strategy="greedy", T=1.0, k=5, p=0.9,
             repetition_penalty=1.0, min_len=0, seed=0):
    """
    Autoregressive generation with a pluggable decoding strategy.

    REPETITION PENALTY divides the probability of already-generated tokens by a
    factor (> 1). Greedy decoding of a language model is notorious for loops
    ("the the the" or a repeated sentence) because once a phrase is likely, its
    own repetition makes it MORE likely; the penalty breaks the loop.

    MIN_LEN masks the end-of-sequence token until at least `min_len` tokens
    exist (HuggingFace's min_new_tokens). Needed here for a real reason: a
    Kneser-Ney model trained on a tiny corpus backs off heavily after a word
    like "the" (it is followed by many different words), and the backoff
    distribution favours </s>, which ends every sentence. Without the mask,
    greedy decoding stops after one word.
    """
    rng = random.Random(seed)
    toks = [BOS] * (lm.n - 1) + tokenize(prompt)
    generated = []
    for _ in range(max_len):
        dist = lm.next_distribution(toks)
        dist.pop(BOS, None)
        if len(generated) < min_len:
            dist.pop(EOS, None)
        if repetition_penalty != 1.0:
            dist = {w: (pw / repetition_penalty if w in generated else pw)
                    for w, pw in dist.items()}
        dist = _normalize(dist)
        if strategy == "greedy":
            w = max(dist, key=dist.get)
        else:
            if strategy in ("temperature", "top_k", "top_p"):
                dist = apply_temperature(dist, T)
            if strategy == "top_k":
                dist = top_k_filter(dist, k)
            elif strategy == "top_p":
                dist = top_p_filter(dist, p)
            w = sample_from(dist, rng)
        if w == EOS:
            break
        toks.append(w)
        generated.append(w)
    return " ".join(tokenize(prompt) + generated)


def beam_search(lm, prompt, beam_width=3, max_len=12, length_penalty=0.7,
                min_len=0):
    """
    BEAM SEARCH — keep the `beam_width` highest-scoring partial sequences.

        score(y) = sum log P(y_t | y_<t)  /  |y|^alpha        (length penalty)

    Greedy decoding commits to the locally best token and can never recover;
    beam search approximates the global argmax. Without the length penalty the
    sum of NEGATIVE log-probabilities always favours SHORT outputs, so beams
    end early. alpha ~ 0.6-1.0 compensates.

    Where it is used: translation and summarization (a single "correct" output).
    Where it is NOT: open-ended generation and chat — maximizing likelihood
    produces bland, repetitive text, which is why sampling (top-p) wins there.
    """
    start = [BOS] * (lm.n - 1) + tokenize(prompt)
    beams = [(0.0, start, False)]
    for _ in range(max_len):
        cand = []
        for logp, toks, done in beams:
            if done:
                cand.append((logp, toks, True))
                continue
            dist = lm.next_distribution(toks)
            dist.pop(BOS, None)
            if len(toks) - len(start) < min_len:
                dist.pop(EOS, None)
            for w, pw in sorted(dist.items(), key=lambda kv: -kv[1])[:beam_width * 2]:
                if pw <= 0:
                    continue
                cand.append((logp + math.log(pw), toks + [w], w == EOS))

        def norm_score(c):
            n_gen = len(c[1]) - len(start)
            return c[0] / (max(n_gen, 1) ** length_penalty)
        beams = sorted(cand, key=norm_score, reverse=True)[:beam_width]
        if all(b[2] for b in beams):
            break
    best = max(beams, key=lambda c: c[0] / (max(len(c[1]) - len(start), 1)
                                            ** length_penalty))
    return " ".join(t for t in best[1] if t not in (BOS, EOS))


# =============================================================================
# 5. REPRESENTATIONS
# =============================================================================
def cosine(u, v):
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    return sum(a * b for a, b in zip(u, v)) / (nu * nv + EPS)


class TfidfVectorizer:
    """
    tf(t,d)  = count(t,d) / |d|
    idf(t)   = log( (1 + N) / (1 + df(t)) ) + 1          smoothed
    tfidf    = tf * idf, then L2-normalized per document

    idf down-weights words that appear everywhere ("the") — a word in every
    document cannot discriminate between them. L2 normalization makes cosine
    similarity a plain dot product and removes document-length effects.
    """

    def __init__(self, stop_words=True, use_stem=False):
        self.stop, self.use_stem = stop_words, use_stem

    def _toks(self, text):
        t = tokenize(text)
        if self.stop:
            t = remove_stopwords(t)
        if self.use_stem:
            t = [stem(w) for w in t]
        return t

    def fit(self, docs):
        df = Counter()
        for d in docs:
            df.update(set(self._toks(d)))
        self.vocab = sorted(df)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        N = len(docs)
        self.idf = [math.log((1 + N) / (1 + df[w])) + 1 for w in self.vocab]
        return self

    def transform(self, docs):
        out = []
        for d in docs:
            toks = self._toks(d)
            vec = [0.0] * len(self.vocab)
            for w in toks:
                if w in self.index:
                    vec[self.index[w]] += 1
            if toks:
                vec = [v / len(toks) * self.idf[i] for i, v in enumerate(vec)]
            n = math.sqrt(sum(v * v for v in vec))
            out.append([v / n for v in vec] if n > EPS else vec)
        return out


class BM25:
    """
    OKAPI BM25 — the strongest classical ranking function, and still the
    baseline every dense retriever must beat.

      score(q, d) = sum_{t in q} IDF(t) * f(t,d)(k1 + 1) /
                                   ( f(t,d) + k1 (1 - b + b |d|/avgdl) )
      IDF(t)      = log( (N - n_t + 0.5)/(n_t + 0.5) + 1 )

    Three ideas in one formula:
      * IDF: rare query terms matter more.
      * TERM-FREQUENCY SATURATION via k1 (~1.2-2.0): the 10th occurrence of a
        word adds far less than the 1st. Raw TF-IDF grows linearly, which lets
        keyword stuffing win.
      * LENGTH NORMALIZATION via b (~0.75): a long document mentions everything
        once; don't let it beat a short, focused one.

    Its weakness is the flip side of its strength: EXACT lexical match. It
    cannot tell that "car" and "automobile" mean the same thing, which is the
    gap dense embeddings fill — hence hybrid retrieval.
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b

    def fit(self, docs):
        self.docs = [remove_stopwords(tokenize(d)) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {t: math.log((self.N - n + 0.5) / (n + 0.5) + 1)
                    for t, n in df.items()}
        return self

    def scores(self, query):
        q = remove_stopwords(tokenize(query))
        out = []
        for i, d in enumerate(self.docs):
            s = 0.0
            L = len(d)
            for t in q:
                f = self.tf[i].get(t, 0)
                if f:
                    s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / \
                        (f + self.k1 * (1 - self.b + self.b * L / self.avgdl))
            out.append(s)
        return out


class PPMIEmbeddings:
    """
    COUNT-BASED WORD EMBEDDINGS: PPMI + truncated SVD.

    1. Co-occurrence counts within a window of +/- w words.
    2. POSITIVE POINTWISE MUTUAL INFORMATION:
           PMI(w, c) = log  P(w, c) / ( P(w) P(c) )
           PPMI      = max(PMI, 0)
       Raw counts are dominated by frequent words ("the" co-occurs with
       everything). PMI asks whether w and c co-occur MORE than chance would
       predict. Negative PMI estimates are unreliable from small counts, so
       they are clipped to 0. Context counts are smoothed with alpha = 0.75,
       which boosts rare contexts — the same trick word2vec uses in its
       negative-sampling distribution.
    3. TRUNCATED SVD compresses the sparse |V|x|V| matrix into dense k-dim
       vectors, merging correlated context dimensions (so "cat" and "kitten"
       become close even if they never shared an exact context word).

    Levy & Goldberg (2014) showed skip-gram with negative sampling is
    implicitly factorizing a SHIFTED PMI matrix — count-based and
    prediction-based embeddings are two routes to the same object.
    """

    def __init__(self, window=3, dim=10, min_count=1, seed=0):
        self.window, self.dim, self.min_count, self.seed = window, dim, min_count, seed

    def fit(self, sentences):
        toks = [remove_stopwords(tokenize(s)) for s in sentences]
        freq = Counter(w for t in toks for w in t)
        self.vocab = sorted(w for w, c in freq.items() if c >= self.min_count)
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        C = [[0.0] * V for _ in range(V)]
        for t in toks:
            for i, w in enumerate(t):
                if w not in self.idx:
                    continue
                for j in range(max(0, i - self.window), min(len(t), i + self.window + 1)):
                    if i != j and t[j] in self.idx:
                        C[self.idx[w]][self.idx[t[j]]] += 1.0
        total = sum(sum(r) for r in C) + EPS
        row = [sum(r) for r in C]
        col_raw = [sum(C[i][j] for i in range(V)) for j in range(V)]
        col_s = [c ** 0.75 for c in col_raw]
        col_tot = sum(col_s) + EPS
        M = [[0.0] * V for _ in range(V)]
        for i in range(V):
            for j in range(V):
                if C[i][j] > 0:
                    pmi = math.log((C[i][j] / total) /
                                   ((row[i] / total) * (col_s[j] / col_tot)) + EPS)
                    M[i][j] = max(pmi, 0.0)
        self.vectors = self._svd(M, min(self.dim, V))
        return self

    def _svd(self, M, k):
        """Top-k left singular vectors * singular values via power iteration
        on M M^T, with deflation (see ml_from_scratch_part2.TruncatedSVD)."""
        rng = random.Random(self.seed)
        V = len(M)
        A = [[sum(M[i][t] * M[j][t] for t in range(V)) for j in range(V)]
             for i in range(V)]
        comps = []
        for _ in range(k):
            v = [rng.gauss(0, 1) for _ in range(V)]
            for _ in range(80):
                w = [sum(A[i][j] * v[j] for j in range(V)) for i in range(V)]
                n = math.sqrt(sum(x * x for x in w)) + EPS
                v = [x / n for x in w]
            lam = sum(v[i] * sum(A[i][j] * v[j] for j in range(V)) for i in range(V))
            comps.append((lam, v))
            for i in range(V):
                for j in range(V):
                    A[i][j] -= lam * v[i] * v[j]
        return {w: [math.sqrt(max(lam, 0)) * vec[self.idx[w]] for lam, vec in comps]
                for w in self.vocab}

    def most_similar(self, word, n=4):
        if word not in self.vectors:
            return []
        v = self.vectors[word]
        sims = [(w, cosine(v, u)) for w, u in self.vectors.items() if w != word]
        return sorted(sims, key=lambda t: -t[1])[:n]

    def sentence_vector(self, text):
        """MEAN POOLING: average the word vectors. A surprisingly strong
        baseline; it ignores word order entirely ("dog bites man" == "man bites
        dog"), which is exactly what contextual encoders fix."""
        vs = [self.vectors[w] for w in remove_stopwords(tokenize(text))
              if w in self.vectors]
        if not vs:
            return [0.0] * self.dim
        return [sum(col) / len(vs) for col in zip(*vs)]


class SkipGram:
    """
    WORD2VEC SKIP-GRAM WITH NEGATIVE SAMPLING (Mikolov et al. 2013)

    For each (centre, context) pair plus k random 'negative' words:
        L = log sigma(v_c . u_o) + sum_{n=1..k} log sigma(-v_c . u_n)

    The full softmax over the vocabulary is O(V) per update; negative sampling
    turns it into k+1 binary logistic regressions ("real context or random?").
    Negatives are drawn from unigram^0.75. Frequent-word SUBSAMPLING (dropping
    "the" with high probability) speeds training and improves rare-word vectors.

    Each word gets TWO vectors (input v, output u); the input vectors are the
    embeddings people use. The famous analogy geometry (king - man + woman ~
    queen) needs a large corpus — on a 40-sentence corpus you get topical
    neighbourhoods, not analogies.
    """

    def __init__(self, dim=16, window=2, negatives=5, lr=0.05, epochs=60, seed=0):
        self.dim, self.window, self.neg = dim, window, negatives
        self.lr, self.epochs, self.seed = lr, epochs, seed

    def fit(self, sentences):
        rng = random.Random(self.seed)
        toks = [remove_stopwords(tokenize(s)) for s in sentences]
        freq = Counter(w for t in toks for w in t)
        self.vocab = sorted(freq)
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        weights = [freq[w] ** 0.75 for w in self.vocab]
        tot = sum(weights)
        cum, acc = [], 0.0
        for w in weights:
            acc += w / tot
            cum.append(acc)
        self.Win = [[rng.uniform(-0.5, 0.5) / self.dim for _ in range(self.dim)]
                    for _ in range(V)]
        self.Wout = [[0.0] * self.dim for _ in range(V)]

        def neg_sample():
            r = rng.random()
            lo, hi = 0, V - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if cum[mid] < r:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        pairs = []
        for t in toks:
            ids = [self.idx[w] for w in t]
            for i, c in enumerate(ids):
                for j in range(max(0, i - self.window), min(len(ids), i + self.window + 1)):
                    if i != j:
                        pairs.append((c, ids[j]))

        def sig(x):
            return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))

        for _ in range(self.epochs):
            rng.shuffle(pairs)
            for c, o in pairs:
                v = self.Win[c]
                grad_v = [0.0] * self.dim
                for t, label in [(o, 1.0)] + [(neg_sample(), 0.0) for _ in range(self.neg)]:
                    u = self.Wout[t]
                    g = (label - sig(sum(a * b for a, b in zip(v, u)))) * self.lr
                    for d in range(self.dim):
                        grad_v[d] += g * u[d]
                        u[d] += g * v[d]
                for d in range(self.dim):
                    v[d] += grad_v[d]
        self.vectors = {w: self.Win[i] for w, i in self.idx.items()}
        return self

    def most_similar(self, word, n=4):
        if word not in self.vectors:
            return []
        v = self.vectors[word]
        return sorted(((w, cosine(v, u)) for w, u in self.vectors.items() if w != word),
                      key=lambda t: -t[1])[:n]


# =============================================================================
# 6. CLASSIC NLP TASKS
# =============================================================================
class NaiveBayesText:
    """
    MULTINOMIAL NAIVE BAYES for text:
        argmax_c  log P(c) + sum_{w in doc} log P(w | c)
        P(w | c) = (count(w, c) + alpha) / (total_c + alpha |V|)

    Fast, needs little data, strong baseline for short-text classification.
    The Laplace alpha is essential: one unseen word would otherwise zero out the
    whole product. Its probabilities are badly over-confident (the independence
    assumption double-counts correlated words), but the argmax is often right.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, texts, labels):
        self.classes = sorted(set(labels))
        self.prior, self.wc, self.total = {}, {}, {}
        self.vocab = set()
        for c in self.classes:
            docs = [tokenize(t) for t, l in zip(texts, labels) if l == c]
            self.prior[c] = math.log(len(docs) / len(texts))
            self.wc[c] = Counter(w for d in docs for w in d)
            self.total[c] = sum(self.wc[c].values())
            self.vocab |= set(self.wc[c])
        return self

    def log_scores(self, text):
        V = len(self.vocab)
        return {c: self.prior[c] + sum(
            math.log((self.wc[c].get(w, 0) + self.alpha) / (self.total[c] + self.alpha * V))
            for w in tokenize(text) if w in self.vocab) for c in self.classes}

    def predict(self, text):
        s = self.log_scores(text)
        return max(s, key=s.get)

    def top_features(self, c, other, n=5):
        V = len(self.vocab)

        def lp(cls, w):
            return math.log((self.wc[cls].get(w, 0) + self.alpha) /
                            (self.total[cls] + self.alpha * V))
        return sorted(sorted(self.vocab), key=lambda w: lp(c, w) - lp(other, w),
                      reverse=True)[:n]


class LogisticTextClassifier:
    """Binary logistic regression on TF-IDF features, trained by gradient
    descent on cross-entropy with L2. The discriminative counterpart of Naive
    Bayes: slower to converge with little data, better with more (Ng & Jordan)."""

    def __init__(self, lr=1.0, epochs=300, l2=1e-3):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, texts, labels):
        self.vec = TfidfVectorizer(stop_words=False).fit(texts)
        X = self.vec.transform(texts)
        d = len(X[0])
        self.w, self.b = [0.0] * d, 0.0
        n = len(X)
        for _ in range(self.epochs):
            gw, gb = [0.0] * d, 0.0
            for x, y in zip(X, labels):
                z = sum(a * b for a, b in zip(self.w, x)) + self.b
                p = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
                e = p - y
                for j in range(d):
                    if x[j]:
                        gw[j] += e * x[j]
                gb += e
            self.w = [w - self.lr * (g / n + self.l2 * w) for w, g in zip(self.w, gw)]
            self.b -= self.lr * gb / n
        return self

    def predict(self, text):
        x = self.vec.transform([text])[0]
        return 1 if sum(a * b for a, b in zip(self.w, x)) + self.b > 0 else 0


class HMMTagger:
    """
    HIDDEN MARKOV MODEL PART-OF-SPEECH TAGGER

        argmax_tags  prod_t  P(tag_t | tag_(t-1)) * P(word_t | tag_t)
                             transition            emission

    Supervised, so the parameters are just smoothed COUNTS from the tagged
    corpus (no EM needed). Decoding is VITERBI — dynamic programming over the
    trellis, O(T * K^2) for T words and K tags instead of O(K^T) brute force.

    Unknown words are the practical weak point: their emission probability
    must come from somewhere. Here: a small smoothing mass, plus a suffix
    heuristic (-ly -> ADV, -ed -> VERB). CRFs replaced HMMs because they are
    discriminative and can use arbitrary overlapping features of the whole
    sentence; BiLSTM-CRFs and then transformers replaced those.
    """

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, tagged):
        self.trans, self.emit = defaultdict(Counter), defaultdict(Counter)
        self.tags, self.words = set(), set()
        for sent in tagged:
            prev = BOS
            for w, t in sent:
                self.trans[prev][t] += 1
                self.emit[t][w] += 1
                self.tags.add(t)
                self.words.add(w)
                prev = t
            self.trans[prev][EOS] += 1
        self.tags = sorted(self.tags)
        return self

    def _pt(self, prev, t):
        c = self.trans[prev]
        return (c.get(t, 0) + self.alpha) / (sum(c.values()) + self.alpha * (len(self.tags) + 1))

    def _pe(self, t, w):
        c = self.emit[t]
        base = (c.get(w, 0) + self.alpha) / (sum(c.values()) + self.alpha * (len(self.words) + 1))
        if w not in self.words:                    # unknown-word heuristics
            if w.endswith("ly") and t == "ADV":
                base *= 20
            if w.endswith("ed") and t == "VERB":
                base *= 20
            if w.endswith("s") and t in ("NOUN", "VERB"):
                base *= 3
        return base

    def tag(self, words):
        T, K = len(words), len(self.tags)
        V = [[-float("inf")] * K for _ in range(T)]
        back = [[0] * K for _ in range(T)]
        for k, t in enumerate(self.tags):
            V[0][k] = math.log(self._pt(BOS, t)) + math.log(self._pe(t, words[0]))
        for i in range(1, T):
            for k, t in enumerate(self.tags):
                best, arg = -float("inf"), 0
                for j, tp in enumerate(self.tags):
                    s = V[i - 1][j] + math.log(self._pt(tp, t))
                    if s > best:
                        best, arg = s, j
                V[i][k] = best + math.log(self._pe(t, words[i]))
                back[i][k] = arg
        last = max(range(K), key=lambda k: V[T - 1][k] + math.log(self._pt(self.tags[k], EOS)))
        path = [last]
        for i in range(T - 1, 0, -1):
            path.append(back[i][path[-1]])
        return [self.tags[k] for k in reversed(path)]


# =============================================================================
# 7. LLM INTERNALS
# =============================================================================
def rope_rotate(vec, pos, base=10000.0):
    """
    ROTARY POSITION EMBEDDING (Su et al. 2021) — the LLaMA/Mistral/GPT-NeoX default.

    Pair up dimensions (x_2i, x_2i+1) and rotate each pair by an angle that
    grows with position:

        theta_i   = base^(-2i/d)
        [x'_2i  ]   [ cos(pos*theta_i)  -sin(pos*theta_i) ] [x_2i  ]
        [x'_2i+1] = [ sin(pos*theta_i)   cos(pos*theta_i) ] [x_2i+1]

    Apply it to Q and K (not V). The payoff:

        <R(m) q, R(n) k> = <q, R(n - m) k>

    because rotations compose and R(m)^T R(n) = R(n - m). Attention scores
    therefore depend only on the RELATIVE offset n - m, with no learned
    position parameters. Low-index pairs rotate fast (local position), high-
    index pairs slowly (long range) — a multi-scale clock. Context extension
    methods (position interpolation, NTK scaling, YaRN) work by rescaling
    these frequencies.
    """
    out = list(vec)
    d = len(vec)
    for i in range(0, d - 1, 2):
        theta = base ** (-i / d)
        c, s = math.cos(pos * theta), math.sin(pos * theta)
        x, y = vec[i], vec[i + 1]
        out[i], out[i + 1] = x * c - y * s, x * s + y * c
    return out


def kv_cache_report(n_layers=32, n_heads=32, n_kv_heads=None, head_dim=128,
                    seq_len=4096, bytes_per=2, batch=1):
    """
    KV-CACHE MEMORY

        bytes = 2 (K and V) * layers * kv_heads * head_dim * seq_len * batch * bytes

    Without a cache, generating token t re-runs attention over all t previous
    tokens, so producing T tokens costs O(T^2) projection work and O(T^3)
    attention work in total. With a cache, each step computes K,V for ONE new
    token and one attention row: O(T) projections and O(T^2) attention total.
    The price is memory that grows linearly with context — at long context the
    cache, not the weights, is what limits batch size.

    MULTI-QUERY / GROUPED-QUERY ATTENTION shrink it: all query heads (MQA) or
    groups of them (GQA) share one K,V head. LLaMA-2-70B uses 8 KV heads for 64
    query heads — an 8x smaller cache with little quality loss. PagedAttention
    (vLLM) manages the cache in fixed-size blocks, like OS virtual memory, to
    avoid fragmentation.
    """
    kv = n_kv_heads or n_heads
    per_token = 2 * n_layers * kv * head_dim * bytes_per
    total = per_token * seq_len * batch
    return per_token, total


def generation_cost(T, d=64):
    """Count multiply-adds for generating T tokens (1 layer, 1 head), with and
    without a KV cache, to make the asymptotics concrete."""
    no_cache = cache = 0
    for t in range(1, T + 1):
        no_cache += 3 * t * d * d + 2 * t * t * d      # re-project + full attention
        cache += 3 * d * d + 2 * t * d                  # one new token only
    return no_cache, cache


def quantize(weights, bits=8, group_size=None):
    """
    SYMMETRIC ABSMAX QUANTIZATION

        scale = max|w| / (2^(bits-1) - 1)
        q     = round(w / scale)        stored as a small integer
        w_hat = q * scale               dequantized on the fly

    Memory: 16-bit -> 8-bit halves it, 4-bit quarters it. A 70B model is ~140 GB
    in FP16 and ~35 GB in 4-bit.

    THE OUTLIER PROBLEM: one large weight sets the scale for everything sharing
    it, so all the small weights get crushed into a handful of integer levels.
    Transformers have systematic outlier features, which is why naive per-
    TENSOR quantization fails. Fixes:
      * per-CHANNEL / per-GROUP scales (e.g. one scale per 64 or 128 weights) —
        an outlier only damages its own group
      * LLM.int8(): keep outlier feature dimensions in FP16, quantize the rest
      * GPTQ: quantize column by column, updating the remaining columns to
        compensate using second-order (Hessian) information
      * AWQ: protect the ~1% of weights that matter most, found from ACTIVATION
        magnitudes

    PTQ (post-training, what this is) needs no retraining. QAT (quantization-
    aware training) simulates rounding during training and recovers more
    accuracy at low bit widths.
    """
    qmax = 2 ** (bits - 1) - 1
    groups = [weights] if not group_size else \
        [weights[i:i + group_size] for i in range(0, len(weights), group_size)]
    deq, scales = [], []
    for g in groups:
        scale = max(abs(w) for w in g) / qmax if any(g) else 1.0
        scales.append(scale)
        deq.extend(max(-qmax, min(qmax, round(w / scale))) * scale for w in g)
    mse = sum((a - b) ** 2 for a, b in zip(weights, deq)) / len(weights)
    return deq, scales, mse


def lora_param_count(d_in, d_out, rank):
    """Full fine-tune updates d_in*d_out params; LoRA trains A (r x d_in) and
    B (d_out x r) = r*(d_in + d_out)."""
    return d_in * d_out, rank * (d_in + d_out)


class LoRALinear:
    """
    LOW-RANK ADAPTATION (Hu et al. 2021)

        h = W0 x + (alpha / r) * B A x          W0 FROZEN;  A: r x d_in,  B: d_out x r

    Hypothesis: the weight CHANGE needed to adapt a pretrained model has low
    intrinsic rank, so it can be written as a product of two thin matrices.

    Two details that matter:
      * B is initialized to ZERO (A small random), so B A = 0 and the adapted
        model starts EXACTLY equal to the pretrained one — fine-tuning begins
        from a known-good point.
      * After training, B A can be MERGED into W0 (W = W0 + BA), so inference
        has zero extra latency. Or keep adapters separate and hot-swap many
        task adapters over one shared base model.

    QLoRA = a 4-bit quantized frozen base + LoRA adapters in 16-bit, which is
    how a 65B model can be fine-tuned on a single 48 GB GPU.
    """

    def __init__(self, W0, rank=2, alpha=None, seed=0):
        rng = random.Random(seed)
        self.W0 = W0
        self.d_out, self.d_in = len(W0), len(W0[0])
        self.r = rank
        self.scale = (alpha or rank) / rank
        self.A = [[rng.gauss(0, 0.1) for _ in range(self.d_in)] for _ in range(rank)]
        self.B = [[0.0] * rank for _ in range(self.d_out)]        # ZERO init

    def delta(self):
        return [[self.scale * sum(self.B[i][k] * self.A[k][j] for k in range(self.r))
                 for j in range(self.d_in)] for i in range(self.d_out)]

    def forward(self, x):
        D = self.delta()
        return [sum((self.W0[i][j] + D[i][j]) * x[j] for j in range(self.d_in))
                for i in range(self.d_out)]

    def fit(self, X, Y, lr=0.05, epochs=400):
        """Least squares on the adapter only; W0 never changes."""
        n = len(X)
        for _ in range(epochs):
            gA = [[0.0] * self.d_in for _ in range(self.r)]
            gB = [[0.0] * self.r for _ in range(self.d_out)]
            for x, y in zip(X, Y):
                pred = self.forward(x)
                err = [p - t for p, t in zip(pred, y)]
                Ax = [sum(self.A[k][j] * x[j] for j in range(self.d_in))
                      for k in range(self.r)]
                for i in range(self.d_out):
                    for k in range(self.r):
                        gB[i][k] += 2 * err[i] * self.scale * Ax[k] / n
                BtE = [sum(self.B[i][k] * err[i] for i in range(self.d_out))
                       for k in range(self.r)]
                for k in range(self.r):
                    for j in range(self.d_in):
                        gA[k][j] += 2 * self.scale * BtE[k] * x[j] / n
            for i in range(self.d_out):
                for k in range(self.r):
                    self.B[i][k] -= lr * gB[i][k]
            for k in range(self.r):
                for j in range(self.d_in):
                    self.A[k][j] -= lr * gA[k][j]
        return self


def softmax_list(z, T=1.0):
    m = max(z)
    e = [math.exp((v - m) / T) for v in z]
    s = sum(e)
    return [v / s for v in e]


def entropy(p):
    return -sum(x * math.log(x) for x in p if x > 0)


# =============================================================================
# 8. RETRIEVAL-AUGMENTED GENERATION
# =============================================================================
def chunk_text(doc_id, text, chunk_size=2, overlap=1):
    """
    CHUNKING by sentences with overlap.

    Too large: the chunk embedding averages several topics (diluted), and you
    waste context budget on irrelevant text. Too small: a chunk loses the
    context needed to answer ("it" with no antecedent). OVERLAP prevents a fact
    that straddles a boundary from being split in half. Production systems
    chunk by tokens (~256-1024) or by document structure (headings), and attach
    metadata (source, section, date) for filtering and citation.
    """
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    step = max(chunk_size - overlap, 1)
    chunks = []
    for i in range(0, len(sents), step):
        piece = " ".join(sents[i:i + chunk_size])
        chunks.append((f"{doc_id}#{len(chunks)}", doc_id, piece))
        if i + chunk_size >= len(sents):
            break
    return chunks


class LSARetriever:
    """
    A DENSE retriever: TF-IDF followed by truncated SVD (Latent Semantic
    Analysis). Query and chunks are projected into the same k-dim space and
    compared by cosine.

    It stands in for a neural bi-encoder (sentence embeddings). The structure is
    the same — embed documents offline, embed the query at request time,
    nearest-neighbour search — and so is the benefit: terms that co-occur
    ("cache", "keys", "values") are merged into shared dimensions, so a query
    can match a passage that never uses its exact words. Production systems use
    a trained encoder plus an approximate-nearest-neighbour index (HNSW, IVF-PQ).
    """

    def __init__(self, dim=6, seed=0):
        self.dim, self.seed = dim, seed

    def fit(self, texts):
        self.tfidf = TfidfVectorizer(stop_words=True, use_stem=True).fit(texts)
        X = self.tfidf.transform(texts)
        d = len(X[0])
        rng = random.Random(self.seed)
        C = [[sum(X[n][i] * X[n][j] for n in range(len(X))) for j in range(d)]
             for i in range(d)]
        self.comps = []
        for _ in range(min(self.dim, d)):
            v = [rng.gauss(0, 1) for _ in range(d)]
            for _ in range(60):
                w = [sum(C[i][j] * v[j] for j in range(d)) for i in range(d)]
                nrm = math.sqrt(sum(x * x for x in w)) + EPS
                v = [x / nrm for x in w]
            lam = sum(v[i] * sum(C[i][j] * v[j] for j in range(d)) for i in range(d))
            self.comps.append(v)
            for i in range(d):
                for j in range(d):
                    C[i][j] -= lam * v[i] * v[j]
        self.doc_vecs = [self._project(x) for x in X]
        return self

    def _project(self, x):
        return [sum(a * b for a, b in zip(c, x)) for c in self.comps]

    def scores(self, query):
        q = self._project(self.tfidf.transform([query])[0])
        return [cosine(q, d) for d in self.doc_vecs]


def rank(scores):
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def reciprocal_rank_fusion(rankings, k=60):
    """
    RRF (Cormack et al. 2009):   score(d) = sum_over_rankers  1 / (k + rank(d))

    Fuses rankers using RANKS only, never raw scores — so there is no need to
    calibrate BM25 scores (unbounded) against cosine similarities ([-1, 1]).
    k ~ 60 damps the influence of any single ranker's top positions. Simple,
    parameter-light, and consistently strong: the default way to build HYBRID
    lexical + dense search.
    """
    fused = defaultdict(float)
    for ranking in rankings:
        for r, doc in enumerate(ranking):
            fused[doc] += 1.0 / (k + r + 1)
    return sorted(fused, key=lambda d: -fused[d])


def rerank(query, candidates, texts):
    """
    A lightweight RERANKER standing in for a cross-encoder.

    Retrieval (bi-encoder / BM25) scores query and document INDEPENDENTLY, which
    is what makes it fast enough to search millions of documents. A CROSS-
    ENCODER reads query and document TOGETHER through one transformer, so it can
    model their interaction precisely — far more accurate, far too slow for the
    whole corpus. Hence the standard two-stage pipeline: cheap recall-oriented
    retrieval of ~50-100 candidates, then an expensive precision-oriented rerank
    of just those.

    Here the interaction features are stemmed query-term coverage and bigram
    overlap.
    """
    q = [stem(w) for w in remove_stopwords(tokenize(query))]
    qb = set(zip(q, q[1:]))
    scored = []
    for c in candidates:
        d = [stem(w) for w in remove_stopwords(tokenize(texts[c]))]
        ds = set(d)
        coverage = sum(1 for w in q if w in ds) / max(len(q), 1)
        bigrams = len(qb & set(zip(d, d[1:])))
        scored.append((coverage + 0.5 * bigrams, c))
    return [c for _, c in sorted(scored, key=lambda t: -t[0])]


def build_prompt(query, passages, max_chars=700):
    """
    PROMPT ASSEMBLY: numbered passages with sources, then the question, plus an
    explicit instruction to answer only from the context and cite. Putting the
    most relevant passage FIRST (or last) matters: models attend less to the
    middle of long contexts ("lost in the middle").
    """
    lines = ["Answer using ONLY the context below. Cite sources as [n]. "
             "If the context does not contain the answer, say so.", "", "Context:"]
    used = 0
    for i, (cid, text) in enumerate(passages, 1):
        if used + len(text) > max_chars:
            break
        lines.append(f"[{i}] ({cid}) {text}")
        used += len(text)
    lines += ["", f"Question: {query}", "Answer:"]
    return "\n".join(lines)


def extractive_answer(query, passages):
    """A stand-in 'generator': return the passage sentence with the highest
    stemmed overlap with the query, with its citation. A real system would send
    build_prompt(...) to an LLM."""
    q = set(stem(w) for w in remove_stopwords(tokenize(query)))
    best, cite = "", ""
    best_s = -1
    for i, (cid, text) in enumerate(passages, 1):
        for s in re.split(r"(?<=[.!?])\s+", text):
            ov = len(q & set(stem(w) for w in remove_stopwords(tokenize(s))))
            if ov > best_s:
                best_s, best, cite = ov, s, f"[{i}]"
    return f"{best} {cite}"


# =============================================================================
# 9. EVALUATION METRICS
# =============================================================================
def ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidate, references, max_n=4, smooth=True):
    """
    BLEU (Papineni et al. 2002) — machine translation.

        p_n  = CLIPPED n-gram precision: each candidate n-gram counts at most as
               many times as it appears in any single reference (stops "the the
               the the" scoring perfectly)
        BP   = 1 if c > r else exp(1 - r/c)        brevity penalty
        BLEU = BP * exp( (1/N) sum_n log p_n )     geometric mean, n = 1..4

    It is PRECISION-based, so without BP a one-word output of a correct word
    would score perfectly. Known weaknesses: no synonyms, no meaning, only
    moderately correlated with human judgement at the sentence level —
    designed for corpus-level comparison.
    """
    c = tokenize(candidate)
    refs = [tokenize(r) for r in references]
    if not c:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        cand = ngrams(c, n)
        max_ref = Counter()
        for r in refs:
            for g, v in ngrams(r, n).items():
                max_ref[g] = max(max_ref[g], v)
        clipped = sum(min(v, max_ref[g]) for g, v in cand.items())
        total = max(sum(cand.values()), 1)
        if smooth:
            p = (clipped + 1) / (total + 1)       # add-one smoothing (Lin & Och)
        else:
            p = clipped / total if clipped else 1e-9
        log_p += math.log(p) / max_n
    r = min((len(ref) for ref in refs), key=lambda L: (abs(L - len(c)), L))
    bp = 1.0 if len(c) > r else math.exp(1 - r / len(c))
    return bp * math.exp(log_p)


def rouge_n(candidate, reference, n=1):
    """ROUGE-N: n-gram RECALL against the reference (summarization: did you
    cover the important content?). Returns (precision, recall, f1)."""
    c, r = ngrams(tokenize(candidate), n), ngrams(tokenize(reference), n)
    overlap = sum((c & r).values())
    p = overlap / max(sum(c.values()), 1)
    rec = overlap / max(sum(r.values()), 1)
    return p, rec, 2 * p * rec / (p + rec + EPS)


def rouge_l(candidate, reference):
    """
    ROUGE-L: based on the LONGEST COMMON SUBSEQUENCE (in order, gaps allowed).
    Rewards correct word ORDER without requiring contiguous n-grams. LCS is the
    same DP family as edit distance.
    """
    a, b = tokenize(candidate), tokenize(reference)
    L = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            L[i][j] = L[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else \
                max(L[i - 1][j], L[i][j - 1])
    lcs = L[len(a)][len(b)]
    p, r = lcs / max(len(a), 1), lcs / max(len(b), 1)
    return p, r, 2 * p * r / (p + r + EPS)


def exact_match(pred, gold):
    return float(" ".join(tokenize(pred)) == " ".join(tokenize(gold)))


def token_f1(pred, gold):
    """SQuAD-style token-overlap F1 — partial credit for extractive QA."""
    p, g = tokenize(pred), tokenize(gold)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def retrieval_metrics(ranked, relevant, k=3):
    """
    precision@k  = relevant in top k / k
    recall@k     = relevant in top k / total relevant
    MRR          = 1 / rank of the FIRST relevant result (averaged over queries)
    nDCG@k       = DCG/IDCG,  DCG = sum rel_i / log2(i + 1)  — position-weighted,
                   so a hit at rank 1 is worth far more than a hit at rank 3
    For RAG, RECALL@k is usually the metric that matters: if the answer is not
    in the retrieved context, the generator cannot recover it.
    """
    top = ranked[:k]
    hits = [1 if d in relevant else 0 for d in top]
    p_at_k = sum(hits) / k
    r_at_k = sum(hits) / max(len(relevant), 1)
    rr = 0.0
    for i, d in enumerate(ranked, 1):
        if d in relevant:
            rr = 1.0 / i
            break
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return p_at_k, r_at_k, rr, (dcg / idcg if idcg else 0.0)


# =============================================================================
# 10. DEMOS
# =============================================================================
def _hdr(t):
    print("\n" + "=" * 76)
    print(t)
    print("=" * 76)


RAG_QUERIES = [
    ("what memory does the model keep while it writes each new word", "doc_kv"),
    ("why do models never see an unknown word", "doc_bpe"),
    ("how to make answers less random", "doc_temp"),
    ("adapt a big model cheaply by training a tiny add on", "doc_lora"),
    ("shrink the model to use smaller integer numbers", "doc_quant"),
    ("how does the model know the order of words", "doc_rope"),
    ("classic lexical ranking that penalizes long documents", "doc_bm25"),
    ("look things up in a database before answering", "doc_rag"),
    ("rotating queries and keys", "doc_rope"),
    ("scale factor for eight bit integers", "doc_quant"),
    ("merging frequent symbol pairs", "doc_bpe"),
    ("frozen weights plus two thin matrices", "doc_lora"),
    ("cache grows with context length", "doc_kv"),
    ("diverse but less coherent output", "doc_temp"),
]


def demo_text_processing():
    _hdr("1. TEXT PROCESSING — stemming, edit distance, spelling correction")
    words = ["running", "cats", "happily", "nationalization", "studies",
             "barked", "hopeful"]
    print("  stemming (rule-based suffix stripping):")
    print("   " + ", ".join(f"{w} -> {stem(w)}" for w in words))
    print("   note 'happily' -> 'happi': a stem need not be a real word.")
    print("   Lemmatization would give 'happy' — at the cost of a dictionary.\n")

    d, ops = edit_distance("kitten", "sitting", return_ops=True)
    print(f"  edit distance kitten -> sitting = {d}")
    print(f"   alignment: {', '.join(ops)}\n")

    sc = SpellCorrector([w for s in CORPUS for w in tokenize(s)])
    typos = ["cta", "gardn", "rockt", "teh", "inflaton", "stadum", "barkd"]
    print("  noisy-channel spelling correction  argmax_c P(c) * P(typo|c):")
    print("   " + ", ".join(f"{t} -> {sc.correct(t)}" for t in typos))

    s = "The dog did NOT like the rain, it wasn't happy."
    toks = tokenize(normalize(s))
    print(f"\n  tokens:              {toks}")
    print(f"  stop words removed:  {remove_stopwords(toks)}")
    print("   -> 'not' is gone: stop-word removal silently flips negation, which")
    print("      is why it helps topic retrieval and hurts sentiment analysis.")


def demo_tokenizers():
    _hdr("2. SUBWORD TOKENIZERS — BPE vs WordPiece")
    print("  BPE: the first merges learned from the corpus (most frequent pair):")
    bpe = BPETokenizer(150).train(CORPUS, verbose_first=6)
    wp = WordPieceTokenizer(150).train(CORPUS)
    print(f"\n  after 150 merges: BPE vocab {len(bpe.vocab)}, "
          f"WordPiece vocab {len(wp.vocab)}\n")
    print(f"  {'word':<16}{'BPE':<34}{'WordPiece'}")
    for w in ["the", "rocket", "rockets", "galaxy", "astronaut", "unbelievable",
              "xyz"]:
        print(f"  {w:<16}{' '.join(bpe.encode_word(w)):<34}"
              f"{' '.join(wp.encode_word(w))}")
    print("\n  The contrast is the point:")
    print("   * BPE merges the most FREQUENT pair, so 'the' becomes one token by")
    print("     merge #3 — frequent words are cheap.")
    print("   * WordPiece merges the pair with the highest freq(ab)/(freq(a)freq(b))")
    print("     — co-occurrence beyond chance — so it happily builds the RARE word")
    print("     'galaxy' whole while common letters stay split. (With a 30k vocab")
    print("     on a real corpus both end up keeping common words whole.)")
    print("   * BPE falls back to characters for 'xyz'; WordPiece emits [UNK]")
    print("     when no piece matches. Byte-level BPE never needs [UNK] at all.")
    text = "the astronauts launched a rocket toward a distant galaxy"
    n_words = len(tokenize(text))
    n_bpe = len(bpe.encode(text))
    n_char = len(text.replace(" ", "")) + n_words
    print(f"\n  sequence length for one sentence: words {n_words}, "
          f"BPE {n_bpe}, characters {n_char}")
    print("   -> subwords sit between word and character level: no OOV, and")
    print("      sequences far shorter than characters (attention is O(T^2)).")
    print(f"  round trip: '{bpe.decode(bpe.encode(text))}'")


def demo_language_models():
    _hdr("3. N-GRAM LANGUAGE MODELS — smoothing and perplexity")
    test = ["the dog sat on the mat", "the rocket orbits the moon",
            "investors watch the bank"]
    print("  held-out sentences (new word combinations):")
    for t in test:
        print(f"    '{t}'")
    print(f"\n  {'bigram smoothing':<22}{'train PPL':>12}{'test PPL':>12}")
    for sm, label in [("mle", "MLE (none)"), ("addk", "add-k (k=0.1)"),
                      ("kn", "Kneser-Ney")]:
        lm = NGramLM(2, sm).fit(CORPUS)
        tr, te = lm.perplexity(CORPUS), lm.perplexity(test)
        te_s = "inf" if te == float("inf") else f"{te:.2f}"
        print(f"  {label:<22}{tr:>12.2f}{te_s:>12}")
    print("\n  MLE memorizes the training set (lowest train PPL) and assigns")
    print("  probability ZERO to any unseen bigram, so test perplexity is infinite.")
    print("  Add-k fixes the zeros crudely; Kneser-Ney's continuation probability")
    print("  ('how many different words does w follow?') generalizes best.")


def demo_decoding():
    _hdr("4. DECODING STRATEGIES — same model, different outputs")
    lm = NGramLM(2, "kn").fit(CORPUS)
    rows = [
        ("greedy", generate(lm, "the", 12, "greedy", min_len=6)),
        ("greedy + repetition penalty",
         generate(lm, "the", 12, "greedy", min_len=6, repetition_penalty=3.0)),
        ("beam search (width 4)", beam_search(lm, "the", 4, 12, min_len=5)),
        ("temperature 0.3", generate(lm, "the", 12, "temperature", T=0.3,
                                     min_len=5, seed=4)),
        ("temperature 1.0", generate(lm, "the", 12, "temperature", T=1.0,
                                     min_len=5, seed=4)),
        ("temperature 2.0", generate(lm, "the", 12, "temperature", T=2.0,
                                     min_len=5, seed=4)),
        ("top-p 0.9", generate(lm, "the", 12, "top_p", p=0.9, min_len=5, seed=4)),
    ]
    for name, out in rows:
        print(f"  {name:<30} {out}")
    print("\n  Greedy LOOPS — it repeats the same phrase, because once a phrase is")
    print("  likely, repeating it keeps it likely. A repetition")
    print("  penalty breaks the loop; beam search finds a higher-likelihood")
    print("  sequence; temperature trades coherence for diversity.")

    print("\n  Why top-p ADAPTS where top-k cannot:")
    confident = {"paris": 0.90, "lyon": 0.04, "nice": 0.03, "lille": 0.02,
                 "metz": 0.01}
    flat = {w: 0.1 for w in "abcdefghij"}
    for name, dist in [("confident next-token dist", confident),
                       ("uncertain next-token dist", flat)]:
        print(f"    {name:<28} top-p(0.9) keeps {len(top_p_filter(dist, 0.9))} "
              f"tokens | top-k(5) keeps {len(top_k_filter(dist, 5))}")
    print("    -> the nucleus shrinks to 1 token when the model is sure and")
    print("       widens when it is not; top-k keeps 5 either way.")

    print("\n  Temperature reshapes the distribution (logits [2.0, 1.0, 0.5, 0.1]):")
    logits = [2.0, 1.0, 0.5, 0.1]
    for T in [0.25, 0.5, 1.0, 2.0, 5.0]:
        p = softmax_list(logits, T)
        print(f"    T={T:<5} probs {[round(x, 3) for x in p]}  "
              f"entropy {entropy(p):.3f} nats")
    print(f"    (maximum possible entropy for 4 tokens = ln 4 = {math.log(4):.3f})")


def demo_representations():
    _hdr("5. REPRESENTATIONS — sparse vectors, BM25, and learned embeddings")
    tf = TfidfVectorizer().fit(CORPUS)
    X = tf.transform(CORPUS)
    print("  TF-IDF cosine similarity between sentences:")
    pairs = [(0, 6, "cat / kitten sentences"), (8, 14, "two bread-baking sentences"),
             (0, 16, "cat vs rocket")]
    for i, j, label in pairs:
        print(f"    {label:<28} {cosine(X[i], X[j]):.3f}")
    print("    -> 'cat' and 'kitten' share NO surface form, so sparse vectors")
    print("       see little overlap. That is the gap embeddings close.\n")

    bm = BM25().fit(CORPUS)
    for q in ["rocket moon astronauts", "interest rates inflation"]:
        s = bm.scores(q)
        top = rank(s)[:2]
        print(f"  BM25 '{q}':")
        for i in top:
            print(f"    {s[i]:.3f}  {CORPUS[i]}")

    topic_of = {}
    for s, t in zip(CORPUS, TOPICS):
        for w in remove_stopwords(tokenize(s)):
            topic_of.setdefault(w, Counter())[t] += 1

    def purity(model, k=5):
        good = tot = 0
        for w in model.vectors:
            main = topic_of[w].most_common(1)[0][0]
            for x, _ in model.most_similar(w, k):
                tot += 1
                good += topic_of[x].most_common(1)[0][0] == main
        return good / tot

    pe = PPMIEmbeddings(window=3, dim=10).fit(CORPUS)
    sg = SkipGram(dim=16, window=5, epochs=60, seed=1).fit(CORPUS)
    print("\n  Word embeddings, nearest neighbours by cosine:")
    for w in ["cat", "rocket", "bank", "bread"]:
        print(f"    PPMI+SVD  {w:<7} {[x for x, _ in pe.most_similar(w, 4)]}")
        print(f"    skip-gram {w:<7} {[x for x, _ in sg.most_similar(w, 4)]}")
    print(f"\n  TOPIC PURITY — fraction of each word's 5 nearest neighbours drawn")
    print(f"  from the same topic (5 topics, so chance = 0.20):")
    print(f"    PPMI + SVD  {purity(pe):.3f}")
    print(f"    skip-gram   {purity(sg):.3f}")
    print("  Both recover the topic structure from co-occurrence alone. PPMI's")
    print("  neighbours are mostly same-SENTENCE words; skip-gram generalizes a")
    print("  little further (e.g. 'cat' finds 'barked', from a dog sentence).")
    print("  The famous analogy arithmetic needs millions of sentences, not 40.")

    a = pe.sentence_vector("the dog chased the cat")
    b = pe.sentence_vector("the cat chased the dog")
    print(f"\n  mean-pooled sentence vectors: cos('dog chased cat', "
          f"'cat chased dog') = {cosine(a, b):.3f}")
    print("   -> identical: averaging discards word order. Contextual encoders")
    print("      (BERT, sentence transformers) exist to fix exactly this.")


def demo_classification():
    _hdr("6. CLASSIC TASKS — sentiment classification and POS tagging")
    texts = [t for t, _ in SENTIMENT]
    labels = [l for _, l in SENTIMENT]
    tests = [("the service was excellent and fast", 1),
             ("terrible food and a boring evening", 0),
             ("friendly staff and delicious coffee", 1),
             ("dirty room and slow rude service", 0),
             ("the story was brilliant", 1),
             ("the update is buggy and slow", 0)]
    nb = NaiveBayesText().fit(texts, labels)
    lr = LogisticTextClassifier().fit(texts, labels)
    nb_acc = sum(nb.predict(t) == y for t, y in tests) / len(tests)
    lr_acc = sum(lr.predict(t) == y for t, y in tests) / len(tests)
    print(f"  held-out sentiment accuracy: Naive Bayes {nb_acc:.3f} | "
          f"logistic regression {lr_acc:.3f}")
    print(f"  most POSITIVE-indicative words (NB log-ratio): "
          f"{nb.top_features(1, 0, 6)}")
    print(f"  most NEGATIVE-indicative words (NB log-ratio): "
          f"{nb.top_features(0, 1, 6)}")
    print("\n  Negation — both models, on sentences whose sentiment words ARE in")
    print("  the training vocabulary:")
    for s, truth in [("the service was not excellent", 0),
                     ("the staff were not friendly", 0),
                     ("the room was not dirty", 1)]:
        print(f"    '{s}'  truth {truth} | NB {nb.predict(s)} | "
              f"LR {lr.predict(s)}")
    print("   -> every one is WRONG, in both directions. A bag of words sees")
    print("      'excellent' or 'dirty' and has no mechanism for what 'not' does")
    print("      to it. (A sentence like 'not good' can look right by luck when")
    print("      neither word was ever seen in training — always check which")
    print("      words the model actually knows before trusting a success.)")

    tagger = HMMTagger().fit(POS_CORPUS)
    print("\n  HMM part-of-speech tagging with Viterbi decoding:")
    for s in ["the dog sat on the chair", "she cooked the warm soup",
              "the rocket launched quickly", "a bright team played",
              "the kitten jumped happily"]:
        w = s.split()
        tags = tagger.tag(w)
        unk = [x for x in w if x not in tagger.words]
        print(f"    {' '.join(f'{a}/{b}' for a, b in zip(w, tags))}"
              + (f"   (unseen: {', '.join(unk)})" if unk else ""))
    print("   -> unseen words are tagged from context (transition probabilities)")
    print("      plus suffix clues: 'jumped' -> VERB via -ed, 'happily' -> ADV")
    print("      via -ly.")


def demo_llm_internals():
    _hdr("7. LLM INTERNALS — RoPE, KV cache, quantization, LoRA")
    rng = random.Random(0)
    q = [rng.gauss(0, 1) for _ in range(8)]
    k = [rng.gauss(0, 1) for _ in range(8)]
    print("  RoPE: score <R(m)q, R(n)k> for different absolute positions m, n")
    for m, n in [(0, 3), (5, 8), (100, 103), (2, 10), (50, 58)]:
        sc = sum(a * b for a, b in zip(rope_rotate(q, m), rope_rotate(k, n)))
        print(f"    m={m:<4} n={n:<4} offset {n - m:<3} score {sc:.6f}")
    print("   -> same offset gives the same score regardless of absolute position:")
    print("      attention sees RELATIVE position, with zero learned parameters.\n")

    per_tok, total = kv_cache_report()
    per_tok_g, total_g = kv_cache_report(n_kv_heads=8)
    print("  KV cache, 7B-class config (32 layers, 32 heads x 128, fp16):")
    print(f"    per token {per_tok / 1024:.0f} KB | at 4,096 tokens "
          f"{total / 2**30:.2f} GB per sequence")
    print(f"    with grouped-query attention (8 KV heads): per token "
          f"{per_tok_g / 1024:.0f} KB | {total_g / 2**30:.2f} GB  (4x smaller)")
    for T in [64, 256, 1024]:
        nc, c = generation_cost(T)
        print(f"    generating {T:5d} tokens: {nc:.2e} ops without cache vs "
              f"{c:.2e} with  ({nc / c:,.0f}x)")
    print("   -> the saving grows with length; the cost is memory that grows")
    print("      linearly with context and limits batch size.\n")

    w = [rng.gauss(0, 0.02) for _ in range(512)]
    print("  Quantization of 512 weights ~ N(0, 0.02^2):")
    print(f"    {'scheme':<30}{'MSE (normal)':>16}{'MSE (1 outlier)':>18}")
    w_out = list(w)
    w_out[37] = 1.5                                   # one outlier weight
    for bits, g, label in [(8, None, "int8 per-tensor"),
                           (4, None, "int4 per-tensor"),
                           (4, 64, "int4 per-group (64)")]:
        m1 = quantize(w, bits, g)[2]
        m2 = quantize(w_out, bits, g)[2]
        print(f"    {label:<30}{m1:>16.2e}{m2:>18.2e}")
    print("   -> ONE outlier sets the scale for its whole group and crushes the")
    print("      other weights. Per-group scales confine the damage — that is")
    print("      why every practical 4-bit scheme (GPTQ, AWQ, QLoRA) uses groups.\n")

    full, lora = lora_param_count(4096, 4096, 8)
    print(f"  LoRA on one 4096x4096 projection: full fine-tune {full:,} params,")
    print(f"    rank-8 LoRA {lora:,} params ({100 * lora / full:.2f}% of full)")
    d = 6
    W0 = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(d)]
    us = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(2)]
    vs = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(2)]
    Wt = [[W0[i][j] + 0.5 * sum(us[r][i] * vs[r][j] for r in range(2))
           for j in range(d)] for i in range(d)]
    X = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(60)]
    Y = [[sum(Wt[i][j] * x[j] for j in range(d)) for i in range(d)] for x in X]
    print("    Adapt a frozen W0 to a target whose true update is RANK 2:")
    for r in [1, 2, 3]:
        L = LoRALinear(W0, r, seed=1)
        before = sum(sum((a - b) ** 2 for a, b in zip(L.forward(x), y))
                     for x, y in zip(X, Y)) / len(X)
        L.fit(X, Y, lr=0.02, epochs=300)
        after = sum(sum((a - b) ** 2 for a, b in zip(L.forward(x), y))
                    for x, y in zip(X, Y)) / len(X)
        print(f"      rank {r}: error {before:.3f} -> {after:.5f}")
    print("   -> the error before training is identical for every rank because B")
    print("      starts at ZERO (the adapter begins as a no-op). Rank 1 cannot")
    print("      express a rank-2 change; rank 2 recovers it exactly.")


def demo_rag():
    _hdr("8. RAG — chunk, retrieve (lexical + dense), fuse, rerank, prompt")
    chunks = [c for did, t in KNOWLEDGE_BASE.items() for c in chunk_text(did, t, 2, 1)]
    texts = [c[2] for c in chunks]
    ids = [c[1] for c in chunks]
    print(f"  {len(KNOWLEDGE_BASE)} documents -> {len(chunks)} chunks "
          f"(2 sentences each, 1-sentence overlap)")
    bm = BM25().fit(texts)
    ls = LSARetriever(dim=6).fit(texts)

    def ranked_docs(order):
        out = []
        for i in order:
            if ids[i] not in out:
                out.append(ids[i])
        return out

    print(f"  {len(RAG_QUERIES)} test queries, many paraphrased so they share few "
          f"words with the source\n")
    print(f"  {'retriever':<24}{'hit@1':>8}{'recall@3':>10}{'MRR':>8}{'nDCG@3':>9}")
    for name in ["BM25 (lexical)", "LSA (dense)", "hybrid (RRF)",
                 "hybrid + rerank"]:
        tot = [0.0, 0.0, 0.0, 0.0]
        for qu, rel in RAG_QUERIES:
            rb, rd = rank(bm.scores(qu)), rank(ls.scores(qu))
            if name.startswith("BM25"):
                r = rb
            elif name.startswith("LSA"):
                r = rd
            else:
                r = reciprocal_rank_fusion([rb, rd])
                if "rerank" in name:
                    r = rerank(qu, r[:6], texts) + r[6:]
            docs = ranked_docs(r)
            h1 = retrieval_metrics(docs, {rel}, 1)[0]
            _, r3, rr, nd = retrieval_metrics(docs, {rel}, 3)
            tot = [a + b for a, b in zip(tot, [h1, r3, rr, nd])]
        n = len(RAG_QUERIES)
        print(f"  {name:<24}{tot[0]/n:>8.3f}{tot[1]/n:>10.3f}"
              f"{tot[2]/n:>8.3f}{tot[3]/n:>9.3f}")
    print("\n  Read it column by column — no single stage wins everything:")
    print("   * the RERANKER is best at the top (hit@1, MRR): reading query and")
    print("     passage together fixes the ORDER of the candidates.")
    print("   * the DENSE retriever has the best recall@3: on paraphrased queries")
    print("     it finds passages that share few exact words.")
    print("   * RRF fusion did NOT improve recall here. With only two rankers and")
    print("     14 chunks, the weaker ranker's votes can push a good passage out")
    print("     of the top 3. Fusion is usually a win at scale, but not for free.")
    print("  Lesson: measure every stage separately. For RAG, recall@k of the")
    print("  RETRIEVER bounds the whole system — an unretrieved answer cannot be")
    print("  generated — so tune retrieval recall first, then rerank for order.")

    q = "how does the model avoid recomputing earlier tokens when generating"
    r = rerank(q, reciprocal_rank_fusion([rank(bm.scores(q)), rank(ls.scores(q))])[:6],
               texts)[:2]
    passages = [(chunks[i][0], texts[i]) for i in r]
    print("\n  Assembled prompt for:", repr(q))
    print("  " + build_prompt(q, passages).replace("\n", "\n  "))
    print(f"\n  extractive stand-in answer: {extractive_answer(q, passages)}")


def demo_metrics():
    _hdr("9. EVALUATION METRICS — what each one rewards")
    ref = "the cat is sitting on the mat"
    cands = [
        ("the cat is sitting on the mat", "identical"),
        ("the cat sits on the mat", "paraphrase"),
        ("on the mat the cat is sitting", "same words, reordered"),
        ("the the the the the the", "degenerate repetition"),
        ("a dog", "short and wrong"),
    ]
    print(f"  reference: '{ref}'\n")
    print(f"  {'candidate':<34}{'BLEU':>7}{'R-1 F':>8}{'R-2 F':>8}{'R-L F':>8}"
          f"{'EM':>5}{'F1':>7}")
    for c, label in cands:
        print(f"  {label:<34}{bleu(c, [ref]):>7.3f}{rouge_n(c, ref, 1)[2]:>8.3f}"
              f"{rouge_n(c, ref, 2)[2]:>8.3f}{rouge_l(c, ref)[2]:>8.3f}"
              f"{exact_match(c, ref):>5.0f}{token_f1(c, ref):>7.3f}")
    print("\n  * Reordering keeps ROUGE-1 and token F1 at 1.0 (bags of words) but")
    print("    drops BLEU, ROUGE-2 and ROUGE-L, which reward order.")
    print("  * BLEU's CLIPPED precision caps the 'the the the' exploit: each 'the'")
    print("    counts only as often as it appears in the reference.")
    print("  * A valid paraphrase is penalized by every n-gram metric — they")
    print("    measure surface overlap, not meaning. That is why LLM outputs are")
    print("    now judged by humans, embedding similarity, or an LLM-as-judge.")


if __name__ == "__main__":
    print("=" * 76)
    print(" NLP & LLMs FROM SCRATCH — pure Python, no libraries")
    print("=" * 76)
    demo_text_processing()
    demo_tokenizers()
    demo_language_models()
    demo_decoding()
    demo_representations()
    demo_classification()
    demo_llm_internals()
    demo_rag()
    demo_metrics()
    print("\n" + "=" * 76)
    print("Every component implemented and measured with zero external libraries.")
    print("=" * 76)
