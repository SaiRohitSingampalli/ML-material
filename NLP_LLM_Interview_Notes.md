# NLP & LLMs — COMPLETE INTERVIEW NOTES

> Companion to `ML_Interview_Notes.md`, `DL_Interview_Notes.md`, and `RL_Interview_Notes.md`.
> Math in ASCII code blocks so it renders anywhere.
> Working code: `nlp_from_scratch.py` — every technique below, pure Python, runs in ~5 seconds, output identical on every run.
> Transformer and attention mechanics themselves live in the DL notes (Part 7) and `dl_from_scratch.py` (MiniGPT); this document covers everything around them.

---

## THE ARC OF NLP IN ONE TABLE

| Era | Representation | Model | Key limitation it hit |
|---|---|---|---|
| Rules (–1990s) | hand-written grammars | parsers, regex | brittle, doesn't scale |
| Statistical (1990s–2012) | counts, n-grams, TF-IDF | HMM, CRF, Naive Bayes, SVM | sparse features, no meaning |
| Static embeddings (2013–2017) | word2vec, GloVe | RNN/LSTM | one vector per word regardless of context |
| Contextual (2018–2019) | ELMo, BERT | pretrained transformers | fine-tune per task |
| LLMs (2020–) | tokens in context | decoder-only transformers at scale | cost, hallucination, context length |

**The 5 questions behind almost every NLP/LLM interview question:** How is text turned into units (**tokenization**)? How are those units represented (**sparse vs dense, static vs contextual**)? What is the **training objective**? How is output produced (**decoding**)? How is it **evaluated**, and does the metric measure what you care about?

---

# PART 0 — CHEAT SHEET

```
TOKENIZATION
  BPE          merge most FREQUENT adjacent pair; encode by replaying merge ranks
  WordPiece    merge pair maximizing freq(ab)/(freq(a)freq(b)); greedy longest-match
  Unigram LM   start big, PRUNE pieces by likelihood loss (SentencePiece)
  byte-level   256-byte base alphabet -> never an unknown token (GPT-2 onward)

LANGUAGE MODELING
  chain rule   P(w_1..T) = prod_t P(w_t | w_<t)
  n-gram       condition on the last n-1 words (Markov assumption)
  perplexity   PP = exp( -(1/N) sum log P(w_t|h_t) ) = exp(cross-entropy)
  Kneser-Ney   P = max(c-d,0)/c(h) + lambda(h) * P_cont(w)
               P_cont(w) = #distinct contexts w follows / #bigram types

DECODING
  greedy       argmax each step            -> loops, bland
  beam         keep top-B partial seqs, score = sum log p / |y|^alpha
  temperature  softmax(z / T)              T<1 sharper, T>1 flatter
  top-k        keep k most likely          fixed-size candidate set
  top-p        smallest set with mass >= p ADAPTIVE candidate set
  rep. penalty divide p(already generated) by a factor > 1

REPRESENTATIONS
  TF-IDF       tf * log((1+N)/(1+df)) + 1, L2-normalized
  BM25         sum IDF * f(k1+1) / (f + k1(1 - b + b|d|/avgdl))
  PMI          log P(w,c) / (P(w)P(c)) ;  PPMI = max(PMI, 0)
  skip-gram    log s(v.u_o) + sum_neg log s(-v.u_n)
  cosine       u.v / (|u||v|)

LLM EFFICIENCY
  KV cache     2 * layers * kv_heads * head_dim * seq * bytes
  LoRA         h = W0 x + (alpha/r) B A x ; params r(d_in + d_out) ; B init 0
  quantization scale = max|w|/(2^(b-1)-1) ; q = round(w/scale)
  RoPE         <R(m)q, R(n)k> = <q, R(n-m)k>  -> relative position for free

METRICS
  BLEU         BP * exp(mean log clipped n-gram precision, n=1..4)   translation
  ROUGE-N/L    n-gram / LCS recall                                    summarization
  EM, F1       exact match, token-overlap F1                          extractive QA
  recall@k     retrieved relevant / total relevant                   RAG retrieval
  MRR          mean of 1/rank of first relevant                      search
  nDCG@k       DCG/IDCG, DCG = sum rel_i/log2(i+1)                   ranking
```

**Which approach when:**

| Situation | Reach for |
|---|---|
| Short-text classification, little data, need speed | TF-IDF + logistic regression / Naive Bayes |
| Classification with a few hundred labels, high accuracy | fine-tune a small encoder (BERT-class) |
| No labels at all | zero/few-shot prompting an LLM, or embeddings + clustering |
| Keyword/exact-match search, IDs, codes | BM25 |
| Semantic search, paraphrases | dense embeddings + ANN index |
| Production search | hybrid (BM25 + dense) + cross-encoder rerank |
| Knowledge that changes / must be cited | RAG |
| Behaviour, tone, format | fine-tuning (LoRA) |
| Fits one GPU only barely | 4-bit quantization + LoRA (QLoRA) |

---

# PART 1 — TEXT PROCESSING

## 1.1 Normalization and tokenization
Every classical choice is a trade-off: lowercasing merges "Apple" the company with "apple" the fruit; stripping punctuation breaks "U.S." and emoticons; stop-word removal **deletes negation**.

Measured in the companion code: removing stop words from *"the dog did NOT like the rain"* yields `['dog', 'like', 'rain', ...]` — the meaning flips. Stop-word removal helps topic retrieval and hurts sentiment. Modern subword tokenizers skip most normalization and let the model learn it.

## 1.2 Stemming vs lemmatization
```
stemming       rule-based suffix stripping (Porter)  fast, crude, stem may not be a word
               running -> run, happily -> happi, university & universe -> univers
lemmatization  dictionary + part of speech            accurate, slower, language-specific
               better -> good, ran -> run
```

## 1.3 Edit distance — the DP to know cold
```
D[i][j] = min( D[i-1][j] + 1,                    deletion
               D[i][j-1] + 1,                    insertion
               D[i-1][j-1] + (a_i != b_j) )      substitution
O(|a| * |b|) time
```
kitten → sitting = 3 (sub k→s, sub e→i, insert g). The same DP family gives **word error rate** (speech), **Needleman-Wunsch** (sequence alignment), and **longest common subsequence** (ROUGE-L). Variants: Damerau adds transposition; weighted costs model keyboard proximity.

## 1.4 Noisy-channel spelling correction
```
correction = argmax_c  P(c)  *  P(typo | c)
                       prior    channel model (edit distance)
```
Bayes' rule doing real work — and the same decomposition (language model × channel model) that powered classical speech recognition and statistical machine translation. In the companion code: `cta→cat, gardn→garden, inflaton→inflation, stadum→stadium`.

---

# PART 2 — SUBWORD TOKENIZATION

## 2.1 Why subwords
```
word-level     huge vocab, still out-of-vocabulary words, no morphology sharing
char-level     no OOV, but sequences ~4-5x longer and attention is O(T^2)
subword        frequent words whole, rare words as reusable pieces, no OOV
```
Measured on one sentence in the companion code: 9 words → 23 BPE tokens (small training corpus) → 57 characters.

## 2.2 BPE (GPT family)
```
TRAIN: start from characters (+ end-of-word marker)
       repeat: count adjacent pairs, merge the MOST FREQUENT, record the merge
ENCODE: replay merges IN LEARNED ORDER (by rank) — not greedy longest match
```
In the companion code the first merges are `e+</w>`, `t+h`, `th+e</w>` — "the" is a single token by merge #3. **Byte-level BPE** uses the 256 bytes as the base alphabet, so any string in any script is encodable with no `[UNK]`.

## 2.3 WordPiece (BERT)
```
merge score = freq(ab) / ( freq(a) * freq(b) )      (pointwise mutual information)
encode      = greedy longest-match-first, continuation pieces prefixed "##"
```
It prefers pairs that co-occur **more than chance predicts**, not merely frequent pairs. In the companion code this produces a striking contrast: WordPiece builds the rare word `galaxy` as one piece while leaving `the` split as `th ##e`; BPE does the opposite. (With a real 30k vocabulary both keep common words whole.)

## 2.4 SentencePiece / Unigram
Treats input as a raw character stream **including spaces** (marked `▁`), so no pre-tokenization and truly language-agnostic. The Unigram variant starts from a large candidate vocabulary and **prunes** the pieces whose removal least hurts corpus likelihood; it can also sample alternative segmentations (subword regularization).

## 2.5 Tokenization consequences interviewers probe
- **Cost and context**: APIs bill per token; context windows are measured in tokens.
- **The tokenizer tax**: languages under-represented in tokenizer training split into far more tokens per word — higher cost, less effective context.
- **Character blindness**: "how many r's in strawberry" is hard because the model sees `straw|berry`, not letters. Same for arithmetic on multi-digit numbers split inconsistently.
- **Trailing whitespace / glitch tokens**: rare tokens that appeared in tokenizer training but barely in model training behave erratically.

---

# PART 3 — LANGUAGE MODELS

## 3.1 n-gram models
```
P(w_1..w_T) = prod_t P(w_t | w_<t)                       chain rule, exact
            ~= prod_t P(w_t | w_(t-n+1)..w_(t-1))        Markov assumption
MLE: P(w | h) = c(h, w) / c(h)
```
**The zero problem**: any unseen n-gram gets probability 0, so any test sentence containing one gets probability 0 and infinite perplexity.

## 3.2 Smoothing
```
add-k        (c + k) / (c(h) + k|V|)       k=1 is Laplace; badly over-smooths
backoff      use the (n-1)-gram if the n-gram is unseen (Katz)
interpolation  lambda_3 P3 + lambda_2 P2 + lambda_1 P1
Kneser-Ney   P_KN(w|h) = max(c(h,w) - d, 0)/c(h) + lambda(h) P_cont(w)
             lambda(h) = d * N1+(h,.) / c(h)
             P_cont(w) = N1+(., w) / N1+(., .)
```
**The Kneser-Ney insight is `P_cont`**: count how many **different** contexts a word follows, not how often it occurs. "Francisco" is frequent but almost only follows "San", so it's a poor guess after a novel context.

Measured in the companion code (bigram, held-out sentences with new word combinations):

| Smoothing | Train PPL | Test PPL |
|---|---|---|
| MLE (none) | 3.08 | **∞** |
| add-k (k=0.1) | 19.61 | 37.73 |
| Kneser-Ney | 8.39 | **17.53** |

MLE memorizes (lowest train perplexity) and fails completely on new data — overfitting in its purest form.

## 3.3 Perplexity
```
PP = exp( -(1/N) sum_t log P(w_t | h_t) ) = exp(cross-entropy)
```
The **effective branching factor**: a PP of 20 means the model is as uncertain as choosing uniformly among 20 words. Lower is better. **Perplexities are only comparable with the same tokenizer and vocabulary** — a character model and a subword model are measuring different things. Per-token perplexity also doesn't track downstream task quality perfectly.

## 3.4 Neural language models
```
Bengio 2003    feedforward over concatenated word embeddings — first to share
               statistical strength via embeddings
RNN/LSTM LM    unbounded history, sequential training
Transformer LM parallel training via causal masking (see DL notes Part 7)
```
Pretraining objectives:
```
causal LM (GPT)       predict next token          -> generation
masked LM (BERT)      predict 15% masked tokens   -> bidirectional understanding
span corruption (T5)  predict masked spans        -> seq2seq
ELECTRA               detect replaced tokens      -> sample-efficient
```

---

# PART 4 — DECODING

## 4.1 The strategies
```
greedy       argmax at every step
beam search  keep B best partial sequences; score = sum log p / |y|^alpha
temperature  p_i proportional to exp(z_i / T)
top-k        sample from the k most likely tokens
top-p        sample from the smallest set whose cumulative mass >= p
min-p        keep tokens with p >= min_p * p_max  (scales with confidence)
rep. penalty divide the probability of already-used tokens by a factor
```

## 4.2 What each one actually does
Measured in the companion code (same bigram model, prompt "the"):

| Strategy | Output |
|---|---|
| greedy | the bank lends money to the bank lends money to the |
| greedy + repetition penalty | the bank lends money to the earth |
| beam search (width 4) | the bank raised interest rates closely |
| temperature 0.3 | the championship final match the bank raised interest rates … |
| temperature 2.0 | the door best it championship astronauts investors use soup … |

- **Greedy loops**: once a phrase is likely, repeating it keeps it likely.
- **Beam search** approximates the global argmax. Without a length penalty it favours short outputs (every extra token adds a negative log-probability). Use it for **translation and summarization** (one right answer). **Not** for chat or open-ended writing — maximizing likelihood produces bland, repetitive text.
- **Temperature**: measured on logits `[2.0, 1.0, 0.5, 0.1]`, entropy rises from 0.111 nats at T=0.25 to 1.376 at T=5, approaching the maximum ln 4 = 1.386.

## 4.3 Why top-p replaced top-k
Top-k keeps a **fixed** number of candidates: too permissive when the model is confident, too strict when it is uncertain. Top-p **adapts**. In the companion code, a confident distribution (`paris: 0.90`) gives a nucleus of **1** token, a flat one gives **10** — while top-k keeps 5 in both cases.

The underlying problem (Holtzman et al., *neural text degeneration*): the long tail of a softmax is individually unlikely but collectively holds real mass; sample from it often enough and generation derails.

## 4.4 Other decoding knobs
```
min_new_tokens   mask end-of-sequence until a minimum length (needed in the
                 companion code: a Kneser-Ney model on a tiny corpus backs off
                 onto </s> after "the" and would stop after one word)
stop sequences   halt on a delimiter
constrained      grammar / JSON-schema constrained decoding masks invalid tokens
speculative      a small draft model proposes k tokens, the large model verifies
                 them in ONE forward pass; accepted tokens are exact, 2-3x faster
```

---

# PART 5 — REPRESENTATIONS

## 5.1 Sparse: bag of words, TF-IDF, BM25
```
TF-IDF   tf(t,d) * idf(t),  idf = log((1+N)/(1+df)) + 1,  L2-normalize
BM25     sum_{t in q} IDF(t) * f(t,d)(k1+1) / ( f(t,d) + k1(1 - b + b|d|/avgdl) )
```
BM25's three ideas: **IDF** (rare terms matter), **term-frequency saturation** via k1 (the 10th mention adds little — defeats keyword stuffing), **length normalization** via b (a long document shouldn't win by mentioning everything once).

Weakness: exact lexical match only. In the companion code, TF-IDF cosine between the "cat" sentence and the "kitten" sentence is **0.000** — no shared surface form. That gap is what embeddings close.

## 5.2 Static word embeddings
**Distributional hypothesis** (Firth): *you shall know a word by the company it keeps.*
```
PPMI + SVD   co-occurrence counts -> PPMI (clip negatives) -> truncated SVD
skip-gram    predict context from centre; negative sampling replaces the O(V)
             softmax with k+1 binary logistic problems
CBOW         predict centre from averaged context — faster, weaker on rare words
GloVe        weighted least squares on log co-occurrence counts
FastText     sum of character n-gram vectors -> handles morphology & OOV
```
**Levy & Goldberg (2014)**: skip-gram with negative sampling implicitly factorizes a shifted PMI matrix — count-based and prediction-based embeddings are two routes to the same object.

Tricks worth naming: context distribution smoothing (`count^0.75`), frequent-word subsampling, two vector sets (input/output — use the input vectors).

Measured in the companion code — **topic purity** (fraction of each word's 5 nearest neighbours from the same topic; 5 topics so chance = 0.20):

| Method | Topic purity |
|---|---|
| PPMI + SVD | **0.984** |
| skip-gram | **0.983** |

Both recover topic structure from co-occurrence alone. The famous `king − man + woman ≈ queen` analogy geometry needs millions of sentences, not 40.

## 5.3 Sentence embeddings and their limits
**Mean pooling** of word vectors is a strong baseline but ignores order: in the companion code, cos("the dog chased the cat", "the cat chased the dog") = **1.000**. Contextual encoders fix this:
```
bi-encoder     encode query and document SEPARATELY -> fast, indexable
               (Sentence-BERT, trained contrastively with in-batch negatives)
cross-encoder  encode the PAIR together -> accurate, not indexable (rerankers)
ColBERT        late interaction: keep per-token vectors, MaxSim at query time
```
Anisotropy: raw BERT embeddings occupy a narrow cone, making cosine similarity uninformative — contrastive fine-tuning spreads them out.

---

# PART 6 — CLASSIC NLP TASKS

## 6.1 Text classification
```
Naive Bayes     argmax_c log P(c) + sum_w log P(w|c), Laplace smoothing
Logistic reg.   on TF-IDF features, L2
Fine-tuned encoder   [CLS] representation -> linear head
Zero/few-shot LLM    prompt with instructions and examples
```
In the companion code both classical models get 6/6 ordinary held-out sentiment examples right — and then get **every** negated sentence wrong, in both directions: *"the service was not excellent"* → positive, *"the room was not dirty"* → negative. A bag of words sees the sentiment word and has no mechanism for what "not" does to it. Negation, sarcasm, and contrast need order and context.

A trap worth knowing: *"the food was not good"* happens to come out correct — but only because neither "not" nor "good" appears in the training data, so the prediction rests on "food" and "was". **Check which words the model actually knows before trusting a success.**

## 6.2 Sequence labeling
```
tasks      POS tagging, named-entity recognition, chunking
HMM        argmax prod P(tag_t | tag_t-1) P(word_t | tag_t)  -> Viterbi O(T K^2)
CRF        discriminative; arbitrary overlapping features of the whole sentence
BiLSTM-CRF neural features + CRF transition constraints
BERT + head  token classification (watch subword/label alignment)
BIO tags   B-PER I-PER O ... encode spans as per-token labels
```
The companion HMM tags unseen words from context plus suffix clues: `kitten/NOUN jumped/VERB happily/ADV`.

## 6.3 Other tasks to be able to define
Named-entity recognition, coreference resolution, dependency/constituency parsing, relation extraction, extractive vs abstractive summarization, machine translation, natural language inference (entailment/contradiction/neutral), question answering (extractive span vs generative), topic modeling (LDA: documents as mixtures of topics, topics as distributions over words, fit by Gibbs sampling or variational inference).

---

# PART 7 — LARGE LANGUAGE MODELS

## 7.1 The training pipeline
```
1. PRETRAINING   next-token prediction on trillions of tokens of web, code, books.
                 Learns language, facts, and reasoning patterns. ~all the compute.
2. SFT           supervised fine-tuning on (instruction, response) demonstrations.
                 Teaches the FORMAT of being an assistant.
3. PREFERENCE    RLHF: reward model on human pairwise preferences, then PPO with a
   TUNING        KL penalty to the SFT model.  Or DPO: optimize the preference
                 likelihood directly, no reward model, no RL loop.
                 Teaches WHICH responses are better (helpful, honest, harmless).
4. (optional)    RL on verifiable rewards (math answers, unit tests) for reasoning;
                 tool use; long-context extension.
```
See the RL notes (Part 7.4) for the RLHF/DPO math.

## 7.2 Scaling laws
```
loss(N, D) ~ E + A / N^alpha + B / D^beta         N = params, D = tokens
compute C ~ 6 N D FLOPs
```
Loss falls as a smooth **power law** in parameters, data, and compute. **Chinchilla** (2022) showed most earlier models were **undertrained**: for a fixed compute budget, scale data and parameters together — roughly **20 tokens per parameter**. Since inference cost depends on N, modern models are deliberately trained far past Chinchilla-optimal ("over-trained") to be smaller and cheaper to serve.

**Emergent abilities** — capabilities that appear abruptly at scale — are partly an artifact of discontinuous metrics (exact match); with continuous metrics many look smooth.

## 7.3 Architecture choices in modern LLMs
```
decoder-only        causal self-attention; dominates since GPT-3
pre-norm + RMSNorm  stable deep training, cheaper than LayerNorm
SwiGLU FFN          gated activation, better quality per parameter
RoPE                relative position via rotation (see 7.5)
GQA / MQA           share K,V heads -> much smaller KV cache
no bias terms       simpler, marginally better
Mixture of Experts  router sends each token to k of E expert FFNs: more total
                    parameters at the same per-token compute (Mixtral, likely
                    GPT-4). Cost: all experts in memory, load balancing loss,
                    harder to serve.
```

## 7.4 In-context learning and prompting
```
zero-shot          instruction only
few-shot           instruction + k examples in the prompt; no weight updates
chain-of-thought   "think step by step" / worked examples of reasoning ->
                   large gains on multi-step problems; intermediate tokens act
                   as extra computation
self-consistency   sample several reasoning chains, majority-vote the answer
ReAct              interleave reasoning with tool calls and observations
structured output  JSON schema + constrained decoding
```
Prompt sensitivity is real: example order, formatting, and label wording can swing accuracy substantially. **Lost in the middle**: models use information at the start and end of a long context better than the middle.

## 7.5 Position encoding: RoPE
```
rotate each pair (x_2i, x_2i+1) by angle pos * theta_i,  theta_i = base^(-2i/d)
<R(m) q, R(n) k> = <q, R(n - m) k>
```
Measured in the companion code: the attention score for offset 3 is **1.425579** at positions (0,3), (5,8), and (100,103) — identical — and **1.844912** for offset 8 at both (2,10) and (50,58). Relative position emerges from rotation composition with zero learned parameters. Context extension (position interpolation, NTK-aware scaling, YaRN) rescales these frequencies. **ALiBi** is the alternative: a linear distance penalty added to attention scores.

---

# PART 8 — LLM EFFICIENCY

## 8.1 KV cache
```
memory = 2 * layers * kv_heads * head_dim * seq_len * batch * bytes
```
Measured in the companion code for a 7B-class config (32 layers, 32×128 heads, fp16): **512 KB per token, 2.00 GB per 4,096-token sequence**. With grouped-query attention (8 KV heads): **128 KB / 0.50 GB** — 4× smaller.

Compute saved (1 layer, 1 head):

| Tokens generated | Without cache | With cache | Speed-up |
|---|---|---|---|
| 64 | 3.70e7 | 1.05e6 | 35× |
| 256 | 1.12e9 | 7.36e6 | 153× |
| 1,024 | 5.23e10 | 7.98e7 | **656×** |

Without a cache, generating T tokens costs O(T²) projections and O(T³) attention; with it, O(T) and O(T²). The price is memory that grows linearly with context and caps batch size. **PagedAttention** (vLLM) stores the cache in fixed-size blocks like virtual memory; **prefix caching** reuses the cache for shared prompt prefixes.

## 8.2 Serving: why inference is memory-bound
```
prefill   process the whole prompt in parallel      -> compute-bound
decode    one token at a time, reading ALL weights  -> memory-bandwidth-bound
```
Each decode step reads every weight from memory to do a small amount of math, so throughput is limited by memory bandwidth, not FLOPs. This is why **batching** (continuous batching across requests), **quantization** (fewer bytes to read), and **speculative decoding** (verify several tokens per weight read) all help so much.

## 8.3 Quantization
```
scale = max|w| / (2^(bits-1) - 1) ;  q = round(w / scale) ;  w_hat = q * scale
```
Memory: FP16 → INT8 halves it, INT4 quarters it (70B: ~140 GB → ~35 GB).

**The outlier problem**, measured in the companion code (512 weights, one outlier of 1.5 added):

| Scheme | MSE, normal weights | MSE, one outlier |
|---|---|---|
| int8 per-tensor | 2.65e-08 | 1.20e-05 |
| int4 per-tensor | 8.90e-06 | **4.20e-04** |
| int4 per-group (64) | 5.05e-06 | **6.60e-05** |

One outlier sets the scale for its whole group and crushes the other weights into a few integer levels; per-group scales confine the damage (6.4× lower error here). Real schemes:
```
LLM.int8()  keep outlier feature dimensions in FP16, quantize the rest
GPTQ        column-by-column, compensating remaining columns with 2nd-order info
AWQ         protect the ~1% most important weights, found from activations
PTQ vs QAT  post-training (no retraining) vs quantization-aware training
```

## 8.4 LoRA and parameter-efficient fine-tuning
```
h = W0 x + (alpha / r) B A x       W0 frozen, A: r x d_in, B: d_out x r, B = 0 at init
trainable params = r (d_in + d_out)
```
For one 4096×4096 projection, rank 8: **65,536 vs 16,777,216 params (0.39%)**.

Measured in the companion code — adapting a frozen W0 to a target whose true update is **rank 2**:

| LoRA rank | Error before | Error after |
|---|---|---|
| 1 | 18.403 | 1.54987 |
| 2 | 18.403 | **0.00004** |
| 3 | 18.403 | 0.00000 |

The "before" error is identical for every rank **because B starts at zero** — the adapter is a no-op until trained, so fine-tuning begins exactly at the pretrained model. Rank 1 cannot express a rank-2 change; rank 2 recovers it exactly. After training, `BA` can be **merged** into W0 for zero inference overhead, or kept separate to hot-swap many task adapters over one base model.
```
QLoRA        4-bit frozen base + 16-bit LoRA adapters -> fine-tune 65B on one GPU
adapters     small bottleneck MLPs inserted between layers
prefix/prompt tuning  learn virtual tokens prepended to the input
```

## 8.5 Other efficiency techniques
```
FlashAttention   exact attention, tiled to stay in SRAM, never materializes T x T
speculative dec. draft model proposes, target model verifies in parallel
distillation     train a small student on a large teacher's soft outputs
pruning          structured (whole heads/channels) pays off; unstructured needs
                 sparse kernels
```

---

# PART 9 — RETRIEVAL-AUGMENTED GENERATION

## 9.1 Why RAG
Pretrained knowledge is frozen at the training cutoff, can't be cited, and is hallucinated when missing. RAG retrieves relevant passages at query time and puts them in the prompt. **RAG vs fine-tuning**: RAG for *knowledge* (changing, citable, auditable); fine-tuning for *behaviour* (format, style, domain tone). They compose.

## 9.2 The pipeline
```
INDEX:   documents -> clean -> CHUNK (with overlap, + metadata) -> embed -> store
QUERY:   query -> (rewrite / expand) -> retrieve top-k (lexical + dense)
         -> fuse -> RERANK top-k -> assemble prompt -> generate with citations
```
**Chunking**: too large dilutes the embedding and wastes context; too small loses the context needed to answer ("it" with no antecedent). Overlap prevents a fact straddling a boundary from being split. Typical: 256–1024 tokens, or structure-aware (by heading), with source/section/date metadata for filtering and citation.

**Retrieval**:
```
lexical (BM25)   exact terms, IDs, rare names, codes
dense            paraphrases, semantic matches; bi-encoder + ANN index (HNSW, IVF-PQ)
hybrid           both, fused with RECIPROCAL RANK FUSION:
                 score(d) = sum_rankers 1 / (k + rank(d)),  k ~ 60
                 uses ranks only, so no score calibration between rankers needed
```
**Reranking**: a cross-encoder reads query and passage *together* — far more accurate, far too slow for the whole corpus. Standard two-stage pattern: cheap high-recall retrieval of 50–100 candidates, expensive high-precision rerank of just those.

## 9.3 What the measurements show
Measured in the companion code (8 documents → 14 chunks, 14 queries, most paraphrased):

| Retriever | hit@1 | recall@3 | MRR | nDCG@3 |
|---|---|---|---|---|
| BM25 (lexical) | 0.571 | 0.714 | 0.687 | 0.652 |
| LSA (dense) | 0.500 | **0.857** | 0.693 | **0.716** |
| hybrid (RRF) | 0.571 | 0.714 | 0.702 | 0.662 |
| hybrid + rerank | **0.714** | 0.714 | **0.774** | 0.714 |

Read it column by column — **no single stage wins everything**:
- The **reranker** is best at the top (hit@1, MRR): it fixes the *order* of candidates.
- The **dense** retriever has the best recall@3: it finds paraphrased passages sharing few exact words.
- **RRF fusion did not improve recall here.** With two rankers and 14 chunks, the weaker ranker's votes can push a good passage out of the top 3. Fusion usually wins at scale, but not for free.

**Lesson: measure every stage separately.** Retriever recall@k bounds the whole system — an answer that isn't retrieved can't be generated — so tune recall first, then rerank for order.

## 9.4 Prompt assembly and generation
Number the passages, include sources, instruct the model to answer **only** from the context, cite `[n]`, and say so when the context lacks the answer. Put the most relevant passage first or last (lost-in-the-middle).

## 9.5 RAG failure modes
```
retrieval miss        answer not in top-k -> hallucination or refusal
wrong chunk boundary  answer split across chunks
distractors           similar-but-wrong passages mislead the generator
stale index           documents changed, embeddings not refreshed
context overflow      too many passages; the model ignores the middle
unfaithfulness        model answers from parametric memory, not the context
```
Advanced patterns: query rewriting / HyDE (embed a hypothetical answer), multi-hop retrieval, parent-document retrieval (search small chunks, return the larger parent), metadata filters, GraphRAG, and agentic RAG where the model decides when and what to retrieve.

## 9.6 Evaluating RAG
Evaluate **retrieval** and **generation** separately:
```
retrieval    recall@k, MRR, nDCG against labelled relevant passages
generation   faithfulness / groundedness (is every claim supported by context?),
             answer relevance, citation accuracy, answer correctness
```

---

# PART 10 — EVALUATION

## 10.1 Classic generation metrics
```
BLEU     BP * exp( mean_n log p_n ),  p_n = CLIPPED n-gram precision, n = 1..4
         BP = exp(1 - r/c) if the candidate is shorter than the reference
ROUGE-N  n-gram recall (summarization: did you cover the content?)
ROUGE-L  longest common subsequence — rewards order, allows gaps
METEOR   adds stemming and synonym matching
chrF     character n-gram F-score — robust to morphology
EM / F1  exact match and token-overlap F1 (extractive QA)
```
Measured in the companion code against *"the cat is sitting on the mat"*:

| Candidate | BLEU | R-1 F | R-2 F | R-L F | EM | F1 |
|---|---|---|---|---|---|---|
| identical | 1.000 | 1.000 | 1.000 | 1.000 | 1 | 1.000 |
| paraphrase ("the cat sits on the mat") | 0.414 | 0.769 | 0.545 | 0.769 | 0 | 0.769 |
| same words, reordered | 0.691 | **1.000** | 0.833 | **0.571** | 0 | **1.000** |
| "the the the the the the" | 0.207 | 0.308 | 0.000 | 0.308 | 0 | 0.308 |
| "a dog" | 0.037 | 0.000 | 0.000 | 0.000 | 0 | 0.000 |

- Reordering keeps ROUGE-1 and token F1 at 1.0 (bags of words) but drops BLEU, ROUGE-2, and ROUGE-L, which reward order.
- BLEU's **clipped** precision caps the "the the the" exploit.
- A valid paraphrase is penalized by **every** n-gram metric — they measure surface overlap, not meaning.

## 10.2 Evaluating LLMs
```
embedding similarity   BERTScore: token-level cosine matching in embedding space
LLM-as-judge           a strong model grades outputs with a rubric; cheap and
                       scalable but biased toward position, length, and its own
                       style — randomize order, use pairwise comparison, calibrate
                       against human labels
human evaluation       gold standard; expensive, needs clear guidelines and
                       inter-annotator agreement (Cohen's kappa)
pairwise / Elo         Chatbot-Arena-style head-to-head preferences
benchmarks             MMLU (knowledge), GSM8K/MATH (reasoning), HumanEval (code,
                       pass@k), TruthfulQA, HellaSwag, long-context needle tests
```
**Benchmark contamination** — test items leaking into pretraining data — inflates scores; check n-gram overlap, use held-out or freshly created test sets. **pass@k** for code = probability that at least one of k samples passes the unit tests.

---

# PART 11 — AGENTS, TOOLS, AND SAFETY

## 11.1 Tool use and agents
```
function calling   model emits a structured call; the runtime executes it and
                   returns the result into the context
ReAct loop         thought -> action -> observation -> ... -> answer
planning           decompose into sub-goals; reflect and retry on failure
memory             short-term = context window; long-term = a retrieval store
```
Failure modes: compounding errors over long horizons, tool-call hallucination (invented arguments), infinite loops, cost blow-up. Mitigations: bounded step budgets, validation of tool arguments, human checkpoints for irreversible actions, evaluation on end-to-end task success.

## 11.2 Safety and reliability
```
hallucination     fluent but false claims -> RAG, citations, calibrated abstention
prompt injection  instructions hidden in retrieved content or user data hijack the
                  model -> treat retrieved/tool text as DATA, never as
                  instructions; privilege separation; output filtering
jailbreaks        adversarial prompts bypass alignment
bias & toxicity   inherited from training data -> filtering, evaluation, RLHF
privacy           training-data memorization and extraction
sycophancy        agreeing with the user over the truth — a side effect of
                  optimizing for human approval
```

---

# PART 12 — RAPID-FIRE Q&A

**Q: Why subword tokenization?** Word vocabularies explode and still hit OOV words; characters make sequences too long for O(T²) attention. Subwords keep frequent words whole, split rare ones into reusable pieces, and — with a character or byte base — never produce an unknown token.

**Q: BPE vs WordPiece?** BPE merges the most frequent pair and encodes by replaying merges in order. WordPiece merges the pair maximizing `freq(ab)/(freq(a)freq(b))` (co-occurrence beyond chance) and encodes greedily by longest match.

**Q: What is perplexity, and what are its limits?** The exponentiated per-token cross-entropy — an effective branching factor. It's only comparable across models with the same tokenizer, and low perplexity doesn't guarantee good downstream behaviour.

**Q: Why does greedy decoding repeat itself?** Once a phrase is probable, conditioning on it makes it probable again — a self-reinforcing loop. Fix with sampling, repetition/frequency penalties, or n-gram blocking.

**Q: Top-k vs top-p?** Top-k keeps a fixed number of candidates regardless of confidence; top-p keeps the smallest set covering probability p, so it shrinks when the model is sure and widens when it isn't.

**Q: When would you use beam search?** Tasks with one correct output — translation, summarization, structured extraction. Not for open-ended generation, where it produces bland, repetitive text.

**Q: What does the KV cache store and why does it matter?** The keys and values of all previous tokens, so each new token needs only one attention row. It turns quadratic re-computation into linear work per step — at the cost of memory that grows with context and limits batch size.

**Q: Why is LLM inference memory-bound?** During decoding each step reads every weight to produce one token, so memory bandwidth, not FLOPs, is the bottleneck. Batching, quantization, and speculative decoding all attack this.

**Q: Why initialize LoRA's B to zero?** So BA = 0 and the adapted model starts exactly equal to the pretrained one.

**Q: Why do outliers hurt quantization?** With a shared scale, one large weight forces a coarse step size on all the small weights sharing it. Per-channel/per-group scales, outlier handling (LLM.int8), and error-compensating methods (GPTQ, AWQ) fix it.

**Q: How does RoPE encode position?** It rotates query/key dimension pairs by position-dependent angles, so their dot product depends only on the relative offset.

**Q: BM25 or dense retrieval?** BM25 for exact terms, names, IDs, and codes; dense for paraphrase and semantic matching. Production uses both (hybrid) plus a cross-encoder reranker.

**Q: Bi-encoder vs cross-encoder?** Bi-encoders embed query and document separately — fast and indexable, used for retrieval. Cross-encoders process the pair together — more accurate, too slow for a corpus, used for reranking.

**Q: RAG or fine-tuning?** RAG for knowledge that changes or must be cited; fine-tuning for behaviour, format, and style. They combine well.

**Q: How do you evaluate a RAG system?** Separately: retrieval (recall@k, MRR, nDCG) and generation (faithfulness to the context, answer relevance, citation accuracy), plus end-to-end correctness.

**Q: Why are BLEU/ROUGE poor for LLMs?** They measure surface n-gram overlap, so they penalize valid paraphrases and can be gamed. Use human evaluation, embedding-based metrics, or calibrated LLM-as-judge.

**Q: What is prompt injection and how do you defend against it?** Instructions embedded in untrusted content (a web page, a document, a tool result) that hijack the model. Treat retrieved and tool content as data rather than instructions, separate privileges, validate actions, and require confirmation for consequential ones.

**Q: What did Chinchilla change?** It showed that for fixed compute, parameters and training tokens should scale together (~20 tokens/parameter), so earlier large models were undertrained.

**Q: Why decoder-only for LLMs?** One simple objective (next-token prediction) that scales cleanly, uses all tokens for training, and directly supports generation; with enough scale it handles understanding tasks too.

**Q: Your classifier gets 100% on your test set. Should you trust it?** Not yet: check class balance, leakage between splits, whether the test set is too easy or near-duplicate, and whether successes come from words the model actually learned. The companion code shows a bag-of-words model scoring 6/6 on ordinary sentences and then failing every negated one.

---

# PART 13 — COVERAGE INDEX

| Concept | Implementation in `nlp_from_scratch.py` | Demo |
|---|---|---|
| Normalization, regex tokenization | `normalize`, `tokenize` | 1 |
| Stop words (and the negation trap) | `remove_stopwords` | 1 |
| Suffix-stripping stemmer | `stem` | 1 |
| Levenshtein distance + alignment | `edit_distance` | 1 |
| Noisy-channel spelling correction | `SpellCorrector` | 1 |
| BPE train / encode / decode | `BPETokenizer` | 2 |
| WordPiece (PMI merges, longest-match) | `WordPieceTokenizer` | 2 |
| n-gram LM, MLE / add-k / Kneser-Ney | `NGramLM` | 3 |
| Perplexity | `NGramLM.perplexity` | 3 |
| Greedy, temperature, top-k, top-p | `generate`, filters | 4 |
| Repetition penalty, min length | `generate` | 4 |
| Beam search with length penalty | `beam_search` | 4 |
| TF-IDF | `TfidfVectorizer` | 5 |
| BM25 | `BM25` | 5, 8 |
| PPMI + SVD embeddings | `PPMIEmbeddings` | 5 |
| Skip-gram with negative sampling | `SkipGram` | 5 |
| Mean-pooled sentence vectors | `PPMIEmbeddings.sentence_vector` | 5 |
| Naive Bayes text classifier | `NaiveBayesText` | 6 |
| Logistic regression on TF-IDF | `LogisticTextClassifier` | 6 |
| HMM POS tagger + Viterbi | `HMMTagger` | 6 |
| RoPE | `rope_rotate` | 7 |
| KV-cache memory and compute | `kv_cache_report`, `generation_cost` | 7 |
| Quantization (per-tensor / per-group) | `quantize` | 7 |
| LoRA (param math + training) | `lora_param_count`, `LoRALinear` | 7 |
| Chunking with overlap | `chunk_text` | 8 |
| Dense retrieval (LSA) | `LSARetriever` | 8 |
| Reciprocal rank fusion | `reciprocal_rank_fusion` | 8 |
| Reranking | `rerank` | 8 |
| Prompt assembly with citations | `build_prompt` | 8 |
| BLEU, ROUGE-N, ROUGE-L, EM, F1 | `bleu`, `rouge_n`, `rouge_l`, ... | 9 |
| precision@k, recall@k, MRR, nDCG | `retrieval_metrics` | 8 |

**Covered elsewhere:** attention, transformer blocks, and a trained MiniGPT → `DL_Interview_Notes.md` Part 7 and `dl_from_scratch.py`. RLHF/DPO math → `RL_Interview_Notes.md` Part 7.4.

**Covered in these notes, not implemented** — be ready to explain: SentencePiece Unigram, GloVe/FastText, BERT/ELMo pretraining, CRFs, LDA, cross-encoders and ColBERT, MoE, FlashAttention, speculative decoding, GPTQ/AWQ internals, HyDE and agentic RAG, LLM-as-judge, agents and tool use.

---

# PART 14 — 10-DAY PLAN

```
Day 1   Text processing, edit distance DP, noisy channel. Run demo 1.
Day 2   Tokenization: BPE vs WordPiece vs SentencePiece; tokenizer consequences.
        Run demo 2.
Day 3   n-gram LMs, smoothing, Kneser-Ney's continuation idea, perplexity.
        Run demo 3.
Day 4   Decoding: greedy loops, beam + length penalty, temperature, top-k vs
        top-p. Run demo 4.
Day 5   TF-IDF, BM25's three ideas, PMI, word2vec, bi- vs cross-encoders.
        Run demo 5.
Day 6   Classic tasks: classification, sequence labeling, HMM/CRF, NER (BIO).
        Run demo 6.
Day 7   LLM pipeline: pretraining, SFT, RLHF/DPO, scaling laws, MoE, prompting.
Day 8   Efficiency: KV cache, GQA, quantization, LoRA/QLoRA, RoPE. Run demo 7.
Day 9   RAG end to end and its failure modes; RAG evaluation. Run demo 8.
Day 10  Metrics and LLM evaluation, agents, safety, rapid-fire Q&A. Run demo 9.
```

**Self-test — from memory, can you:**
1. Train BPE by hand on five words and encode a new one?
2. Explain why Kneser-Ney uses continuation counts, with the "San Francisco" example?
3. Define perplexity and say when two perplexities are not comparable?
4. Explain why greedy loops, why beam search is bland, and why top-p adapts?
5. Write BM25 and name its three ideas?
6. State what PMI measures and how skip-gram relates to it?
7. Explain bi-encoder vs cross-encoder and the two-stage retrieval pattern?
8. Compute KV-cache memory for a given model and explain why GQA helps?
9. Explain why decoding is memory-bound?
10. Explain LoRA, including why B starts at zero, and how QLoRA fits a big model on one GPU?
11. Explain the quantization outlier problem and two fixes?
12. Design a RAG pipeline, list five failure modes, and say how to evaluate it?

---
*Companion code: `nlp_from_scratch.py`. Related: `DL_Interview_Notes.md` (transformers), `RL_Interview_Notes.md` (RLHF), `ML_Interview_Notes.md`.*
