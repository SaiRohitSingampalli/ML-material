# ML SYSTEM DESIGN — COMPLETE INTERVIEW NOTES

> Companion to the ML, DL, RL, and NLP/LLM notes.
> Working code: `mlsys_from_scratch.py` — the failure modes below, simulated and measured. Pure Python, runs in ~8 seconds, identical output every run.

---

## WHAT THIS INTERVIEW IS ACTUALLY TESTING

A system design interview is **not** a modelling interview. The interviewer already assumes you can train a classifier. They want to see whether you can:

1. Turn a vague business goal into a well-posed ML problem — or recognize that ML is the wrong tool.
2. Pick metrics that actually track the goal, offline **and** online.
3. Reason about data: where labels come from, how they're delayed or biased, and how leakage sneaks in.
4. Design for latency, scale, and cost with explicit back-of-envelope numbers.
5. Anticipate how the system fails after launch — and how you'd know.

Most candidates over-invest in step 3's model choice and under-invest in steps 1, 2, and 5. Reverse that.

---

# PART 0 — THE FRAMEWORK (use it every time)

Spend roughly these proportions of a 45-minute interview. **Say the structure out loud at the start** — it signals seniority and lets the interviewer steer you.

```
1. CLARIFY            (~5 min)   goal, users, scale, constraints, what exists today
2. FRAME AS ML        (~5 min)   input -> output, task type, label source
3. METRICS            (~5 min)   business / online / offline, and how they connect
4. DATA & FEATURES    (~8 min)   sources, labels, features, leakage, freshness
5. MODEL              (~7 min)   baseline first, then the real model, and WHY
6. SERVING            (~8 min)   architecture, latency budget, batch vs online
7. EVAL & ROLLOUT     (~4 min)   offline eval, shadow, canary, A/B test
8. MONITOR & ITERATE  (~3 min)   drift, feedback loops, retraining, failure modes
```

## Step 1 — Clarify
```
Business goal        what does success look like in money/users/time?
Users & scale        how many users, items, requests/sec, peak vs average?
Latency              real-time (<100 ms)? near-real-time? batch nightly?
Constraints          privacy, regulation, explainability, fairness, budget
Current state        is there a heuristic/rules system today? (your baseline!)
Failure cost         what does a false positive cost vs a false negative?
```
Never start designing before asking about failure cost — it decides the metric, the threshold, and sometimes whether to use ML at all.

## Step 2 — Frame as ML
State the input, the output, and the task type explicitly: *"Given a user and a candidate post, predict P(engagement) — a binary classification used for ranking."* Then ask **where the labels come from**: explicit (ratings, human review), implicit (clicks, dwell, purchases), or derived later (chargebacks, churn). Implicit labels are plentiful but **biased by what the system already showed** (see Part 5.4).

**When NOT to use ML:** the rule is simple and stable; there are no labels and no way to get them; mistakes are unacceptable and must be explainable; the data volume is tiny; or a heuristic already meets the goal. Saying this out loud when it's true is a strong signal.

## Step 3 — Metrics (three layers)
```
BUSINESS   revenue, retention, time-to-resolution, fraud losses     (what matters)
ONLINE     CTR, conversion, dwell, session length, complaint rate   (A/B-testable)
OFFLINE    AUC, log loss, NDCG@k, recall@k, precision@k, RMSE       (fast to iterate)
GUARDRAIL  latency p99, error rate, diversity, fairness, cost       (must not break)
```
The hard, interview-worthy part is the **links**: does a better offline metric actually move the online metric? Offline/online mismatch is common — offline data comes from the old policy's exposures, metrics ignore position bias, and ranking metrics ignore calibration. Always propose validating the link with an A/B test.

## Step 4 — Data & features
Cover sources, label generation and delay, class balance, **point-in-time correctness**, freshness (batch vs streaming features), and privacy. Name feature families: user, item, context (time, device, location), user×item cross features, and real-time counters.

## Step 5 — Model
**Always start with a baseline**: a heuristic, popularity, or logistic regression. It gives a reference point, a fallback, and often most of the value. Then justify the real model by the data (tabular → gradient boosting; text/images → pretrained encoders; large-scale retrieval → two-tower). Mention the loss and why.

## Step 6 — Serving
Draw the request path. Give a **latency budget** broken down by stage. Decide batch (precompute nightly) vs online (at request time) vs hybrid. Address caching, fallbacks when the model is down, and cost.

## Step 7 — Evaluation & rollout
Offline eval on a **time-based** split → shadow deployment → canary → A/B test with a pre-registered primary metric, sample size, and guardrails.

## Step 8 — Monitoring & iteration
Data quality, input drift, prediction drift, delayed-label performance, feedback loops, retraining cadence, and the on-call runbook: what happens when it breaks at 3 a.m.?

---

# PART 1 — CHEAT SHEET

```
LATENCY BUDGET (typical real-time recommendation/ranking request, ~200 ms total)
  network + gateway        10-20 ms
  feature fetch (online)   5-15 ms     (KV store, parallel lookups)
  candidate retrieval      10-30 ms    (ANN over millions)
  ranking model            20-50 ms    (hundreds of candidates, batched)
  re-rank / business rules 5 ms
  slack for p99            ~50%

NUMBERS WORTH KNOWING (orders of magnitude)
  seconds per day          86,400          ~ 10^5
  1M requests/day          ~12 QPS average
  peak-to-average          2-5x (diurnal)
  main-memory read         ~100 ns
  SSD random read          ~100 us
  same-region network RTT  ~0.5-1 ms
  cross-region RTT         ~50-150 ms
  fp16 parameter           2 bytes -> 7B params = 14 GB
  Adam training memory     ~16 bytes/param (weights+grads+2 moments)

QUEUEING
  Little's law     L = lambda * W                       (any stable system)
  M/M/1            W = 1/(mu - lambda) -> explodes as utilization -> 1
  plan for         60-70% utilization; size for PEAK, not average

A/B TESTING
  sample size      n ~ 16 * p(1-p) / delta^2   per arm (alpha .05, power .8)
  never peek       or use sequential tests (alpha spending, mSPRT)
  unit             randomize by USER, not request (consistency, independence)

DRIFT
  PSI < 0.1 stable | 0.1-0.25 investigate | > 0.25 significant
  covariate drift  P(x) changes          -> detectable WITHOUT labels
  concept drift    P(y|x) changes        -> needs labels (often delayed)
  label drift      P(y) changes          -> base-rate shift, recalibrate
```

**Measured failure modes from the companion code (the numbers to quote):**

| Failure mode | What happened |
|---|---|
| Label leakage (naive join) | offline AUC **0.788** → serving **0.618**, worse than the honest model's 0.675 |
| Training-serving skew (units) | AUC 0.675 → 0.648 (looks fine) but % flagged **20.6% → 0.0%** |
| No batching past capacity | p50 latency **5.9 s** at 200 QPS vs 15.6 ms with dynamic batching |
| Peeking at an A/A test | false-positive rate **6.3% → 24.7%** |
| Naive canary on a good model | **12%** false rollbacks (2.3% when guarded) |
| Concept drift | PSI stays ~0.05 (invisible) while AUC falls **0.87 → 0.37** |
| Threshold 0.5 for 50:1 costs | **4.1×** the cost of the optimal threshold (0.024) |
| No exploration | finds **1 of 5** best items, earns half the achievable CTR |

---

# PART 2 — THE DATA LAYER

## 2.1 Data validation
Most "the model got worse" incidents are actually "the data changed" — renamed columns, switched units, new categories, a spike in nulls. Learn a schema from trusted data and check every batch **before** it reaches training or serving:
```
schema    expected columns and types
nulls     null rate within tolerance
range     numeric values within learned bounds
domain    categorical values seen before
drift     distribution tests (PSI, KS) — see Part 5
```
Tools: TFX Data Validation, Great Expectations, Deequ. The companion code catches a new country code, a units change (dollars → cents), and a jump from 0% to 31.5% nulls — **none of which would crash a model**; all would silently degrade it.

## 2.2 Label leakage
**Definition:** a feature contains information that would not be available at prediction time. Classic sources:
- **Temporal leakage** — features computed over a window that extends past the prediction time (the companion demo).
- **Target leakage** — a feature that is a consequence of the label (a "refund_issued" field when predicting fraud).
- **Train/test contamination** — the same user, session, or near-duplicate row in both splits.
- **Preprocessing leakage** — scaling, imputation, or target encoding fit on the full dataset before splitting.

**Detection:** suspiciously high offline metrics; one feature dominating importance; performance that collapses on a strictly later time split; and for every feature, the question *"could I have known this at the moment of prediction?"*

**The lesson from the companion code:** the leaky model had the **best offline metric and the worst production performance**. Leakage doesn't just inflate your numbers — it makes you choose the wrong model.

## 2.3 Point-in-time correctness and the feature store
For a training example with label time `t`, every feature must be computed **as of `t`** — never later. A naive join ("latest value per user") silently gives historical rows features computed from their future.

A **feature store** exists to own each feature definition once and serve it two ways:
```
OFFLINE store   historical values -> point-in-time joins -> training sets
ONLINE store    latest values in a low-latency KV store -> serving
```
One definition, two materializations — which eliminates both leakage (point-in-time joins) and skew (one code path). Examples: Feast, Tecton, internal platforms (Michelangelo, Zipline).

## 2.4 Training-serving skew
The same feature computed differently in training and serving. Common causes: separate code paths (Python batch job vs Java service), unit mismatches, different window boundaries, different handling of missing values, time-zone bugs, or a feature that is fresher offline than it can ever be online.

**Why it's so dangerous — measured:** in the companion code, logging spend in cents instead of dollars moves AUC only 0.675 → 0.648, because a monotone rescaling preserves ranking. A dashboard tracking AUC stays green. Meanwhile the mean predicted probability collapses 0.343 → 0.007 and the share of users flagged for retention outreach drops from **20.6% to 0.0%**.

**Defences:** a shared feature pipeline; logging the **features actually used at serving time** and training on those logs; comparing online and offline feature distributions; and monitoring prediction distributions, not just ranking metrics.

## 2.5 Labels in practice
```
delayed labels     fraud chargebacks arrive in 30-90 days; churn in weeks
                   -> evaluate with a lag; use proxy labels for fast signal
noisy labels       crowdsourced annotation -> agreement metrics, adjudication
biased labels      only observed for items the system showed (selection bias)
weak supervision   heuristic labelling functions combined (Snorkel)
active learning    label the examples the model is least certain about
human-in-the-loop  review queue for low-confidence predictions -> new labels
```

## 2.6 Reproducibility and the model registry
Version **everything** that produced a model: code, data snapshot, feature definitions, hyperparameters, seed, environment. A content hash of that configuration identifies the model uniquely. A registry tracks stages (staging → production → archived), gates promotion on evaluation, and keeps the previous production version so **rollback is a pointer flip, not a retrain**. Lineage (data → model → predictions) is what makes incidents debuggable and audits possible.

---

# PART 3 — THE SERVING LAYER

## 3.1 Batch vs online vs hybrid
```
BATCH (offline)   precompute predictions on a schedule, store, look up at request
                  + cheap, simple, no latency risk    - stale, can't use request context
                  e.g. nightly "recommended for you" emails, churn scores
ONLINE            compute at request time
                  + fresh, uses real-time context      - latency/cost, needs online features
                  e.g. search ranking, fraud at checkout
HYBRID            precompute expensive parts (embeddings, candidates), finish
                  online with cheap context-aware scoring
                  -> the most common production pattern
STREAMING         features updated continuously from event streams (Kafka/Flink)
```

## 3.2 Latency, queueing, and capacity
**Little's law** `L = λW` holds for any stable system. For an M/M/1 queue, `W = 1/(μ − λ)`, which **explodes** as utilization approaches 1. Measured in the companion code (service rate 100/s):

| Utilization | Mean latency |
|---|---|
| 50% | 20 ms |
| 90% | 100 ms |
| 99% | 1,000 ms |

Consequences: plan capacity for **peak** load at **60–70% utilization**; watch **p99**, not the mean (tail latency degrades first, and at fan-out every request waits for the slowest dependency); autoscale on queue depth or latency, not just CPU.

## 3.3 Dynamic batching
Inference has a fixed per-call cost (kernel launches, reading weights from memory, RPC overhead) plus a per-item cost. Batching amortizes the fixed part. A dynamic batcher launches when the batch is full **or** a timeout expires. Measured in the companion code (8 ms fixed + 1 ms/item):

| Load | No batching p50 | Batching (≤16, 2 ms) p50 |
|---|---|---|
| 100 QPS | 35.6 ms | 12.2 ms |
| 200 QPS | **5,952 ms** | 15.6 ms |
| 400 QPS | **9,733 ms** | 20.8 ms |

Without batching one replica caps at 111 QPS and the queue grows without bound. The timeout is a real trade-off: at low load a 10 ms wait doubles p50 (9 → 19 ms) to trim p99, while a 2 ms wait gets most of the p99 gain cheaply. For LLMs the equivalent is **continuous batching**, which adds and removes sequences from a running batch at every decode step.

## 3.4 Caching
Traffic follows a power law, so caching predictions (or embeddings, or features) pays off hugely. Measured: with Zipf-distributed keys, a cache holding **1%** of the keys serves **53%** of requests; 20% of keys serves 82%. Design questions to raise: the cache **key** must include every input that affects the prediction **and the model version**; the **TTL** must be shorter than the staleness tolerance; and plan for cache stampedes on expiry.

## 3.5 Scaling models and inputs
```
high-cardinality IDs   hashing trick (fixed memory, collisions) or learned
                       embeddings; measured: 5,000 IDs into 16K buckets -> 26.7%
                       collide, into 262K -> 2.0%
huge embedding tables  shard across parameter servers
model too slow         distillation, quantization, pruning, smaller architecture,
                       caching, precomputation, early-exit cascades
model too big          quantization, tensor/pipeline parallelism, offloading
many models            multi-model serving, adapters (LoRA) over one base
```

## 3.6 Reliability
**Always design a fallback**: if the model times out or errors, serve a cached prediction, a simpler model, or a popularity/rules default — never an error page. Use timeouts and circuit breakers on every dependency, graceful degradation (drop expensive features first), and idempotent retries.

---

# PART 4 — RETRIEVAL AND RANKING (the pattern behind half of all questions)

## 4.1 The funnel
```
CATALOGUE (millions-billions)
   | candidate generation  cheap, RECALL-oriented   ANN over two-tower embeddings,
   v                                                 co-visitation, popular, recent
CANDIDATES (hundreds-thousands)
   | ranking               expensive, PRECISION     GBDT or deep model with rich
   v                                                 user x item cross features
TOP-N (tens)
   | re-ranking / policy   diversity, freshness, dedup, business rules, fairness
   v
PAGE
```
The funnel exists because of the latency budget: the heavy model can't score the catalogue, and the light model alone isn't accurate. Search, feeds, ads, and recommendations all use this shape.

## 4.2 Two-tower retrieval
A user tower and an item tower produce embeddings trained so that `u · v` is high for engaged pairs (contrastive loss with in-batch negatives). Item embeddings are precomputed and indexed; the user embedding is computed at request time; retrieval is nearest-neighbour search. Watch for: **popularity bias** in in-batch negatives (correct with logQ correction), stale item embeddings, and cold-start items (use content features in the item tower).

## 4.3 Approximate nearest neighbours
Exact search is O(N·d) per query. ANN trades a little recall for large speed-ups. Measured with LSH in the companion code (2,000 items):

| Bits | Tables | Recall@10 | Items scored |
|---|---|---|---|
| 10 | 8 | 0.682 | 138 |
| 8 | 8 | 0.785 | 238 |
| 6 | 8 | **0.905** | 440 |

More bits → smaller buckets (faster, lower recall); more tables → more chances to find each neighbour (slower, higher recall). Production indexes: **HNSW** (graph-based, best recall/latency, memory-heavy), **IVF-PQ** (clustering + product quantization, memory-light), ScaNN. Always tune **recall@k vs latency**.

## 4.4 Ranking and re-ranking
The ranker optimizes a (often multi-task) objective — e.g. `P(click)`, `P(like)`, `P(long dwell)` — combined into one score with tuned weights. Losses: pointwise (log loss), pairwise (BPR, RankNet), listwise (LambdaMART, softmax). Correct for **position bias** (items at the top get clicked more regardless of relevance) using position as a training feature that is fixed at serving, or inverse propensity weighting.

**Diversity via MMR:** `next = argmax λ·rel(i) − (1−λ)·max_{j∈S} sim(i,j)`. Measured: MMR raised distinct genres per page **1.69 → 2.22 (+31%)** for **−1.5%** immediate engagement; both personalized policies beat popularity (0.714 vs 0.634).

---

# PART 5 — EXPERIMENTATION, MONITORING, AND DECISIONS

## 5.1 The rollout ladder
```
offline eval      time-based split, sliced metrics (by country, device, new users)
shadow            copy of live traffic, predictions logged, never served
                  -> zero risk; checks latency, errors, prediction distributions,
                     and where the new model DISAGREES with production
canary            small, growing % of real traffic, auto-rollback on regression
                  -> catches serving bugs, skew, crashes on unseen inputs
A/B test          randomized comparison on the business metric
gradual ramp      1% -> 5% -> 25% -> 50% -> 100%, with holdback for long-term effects
```

## 5.2 A/B testing
```
z = (p_B - p_A) / sqrt( p(1-p)(1/n_A + 1/n_B) )
n per arm ~ (z_{a/2} + z_b)^2 [p1(1-p1) + p2(1-p2)] / (p2 - p1)^2
```
Measured in the companion code: 10.0% vs 10.8% conversion on 10,000 users each gives **p = 0.064** with a 95% CI that includes zero — an 8% relative lift that isn't significant. And the sample-size table: on a 2% baseline, detecting a 20% lift needs 21K users/arm, a 5% lift 315K, a **1% lift 7.7 million**. Halving the detectable effect quadruples the sample.

**Pitfalls interviewers expect you to name:**
```
peeking             repeated looks inflate false positives (measured: 6.3% -> 24.7%)
                    -> fixed horizon or sequential testing
wrong unit          randomize by user, not request (consistency + independence)
network effects     users influence each other (marketplaces, social) -> SUTVA
                    violated -> cluster or switchback randomization
novelty/primacy     early behaviour isn't long-term behaviour -> run long enough,
                    keep a long-term holdback
multiple metrics    testing 20 metrics -> expect one false win -> pre-register a
                    primary metric; Bonferroni/BH for the rest
sample ratio mismatch  50/50 split came out 48/52 -> assignment is broken; stop
Simpson's paradox   aggregate result reverses within segments
underpowered tests  "no significant difference" is not "no difference"
```
**Variance reduction** (CUPED): regress out each user's pre-experiment metric to shrink variance and sample size, often by 30–50%.

**Interleaving** for ranking: show users a merged list from both rankers and attribute clicks — far more sensitive than a standard A/B test for comparing rankers.

## 5.3 Canary deployment
Measured over 300 simulated deploys in the companion code:

| Policy | False rollback | Catch rate | Traffic exposed to a bad model |
|---|---|---|---|
| naive (judge every stage, α=.05) | **12.0%** | 100% | 14.2% |
| guarded (≥500 requests, α=.01) | **2.3%** | 100% | 32.2% |

Judging a 1% canary on a few dozen requests is mostly noise — the peeking problem again — so good deploys get rolled back. Waiting for data fixes that at the cost of exposing more users to a genuinely bad model. **Resolution:** hard failures (crashes, error rates, latency SLO breaches) roll back instantly at any traffic level; only statistical quality regressions wait for enough data.

## 5.4 Monitoring — what to watch
```
SYSTEM        latency p50/p95/p99, throughput, error rate, CPU/GPU/memory, cost
DATA          schema violations, null rates, feature ranges, freshness/staleness
INPUT DRIFT   per-feature PSI / KS / JS vs the training reference
PREDICTION    distribution of scores, % positive, class balance of predictions
PERFORMANCE   actual metrics once labels arrive (often delayed) + proxy metrics
BUSINESS      the online metric the model exists to move
```
**Drift types:**
```
covariate / data drift   P(x) changes              detectable without labels
concept drift            P(y|x) changes            needs labels
label / prior drift      P(y) changes              base-rate shift -> recalibrate
```
Measured in the companion code:
- **Covariate drift** sets off PSI (≈2.1) but AUC **never drops** — the model was well specified. A drift alert is a reason to look, not proof of damage.
- **Concept drift** is **invisible** to input monitoring (PSI stays ≈0.05) while the static model's AUC collapses from 0.87 to 0.37.
- A trivial 0.03 shift on 40,000 rows gives KS **p = 0.0008** — "significant" — with PSI 0.0011. At production scale everything is statistically drifted. **Alert on effect size, not p-values.**

## 5.5 Retraining strategy
```
scheduled     retrain every day/week on a sliding window -> simple, predictable
triggered     retrain when monitored performance or drift crosses a threshold
online        continuous incremental updates (ads CTR, news) -> fastest
              adaptation, risk of instability and feedback amplification
```
Measured under concept drift: performance-triggered retraining scored mean AUC **0.811 with 1 retrain**; periodic retraining scored 0.795 with **11 retrains** (it keeps blending stale pre-drift data into its window). Always validate a retrained model against the current one before promoting it — automated retraining without a gate eventually ships a bad model.

## 5.6 Thresholds and calibration
**Choose the threshold by cost, not 0.5.** For calibrated probabilities the Bayes-optimal rule is to flag when `p > c_FP / (c_FP + c_FN)`. Measured (fraud-like, a miss costs 50× a false alarm): threshold 0.5 → cost 5,054; threshold 0.024 → cost 1,241 — **4.1× cheaper**; theory says 1/51 ≈ 0.020.

**Calibration** matters whenever the probability's *value* is used — expected-value decisions (bid = p(click) × value), cost thresholds, combining models, showing risk to humans. Measured: a miscalibrated model with ECE 0.282 drops to **0.011 with Platt scaling** and 0.013 with isotonic regression, with AUC unchanged. Fit calibrators on **held-out** data. Recalibrate after base-rate shifts and after any change to the sampling used for training (e.g. negative downsampling in ads).

## 5.7 Feedback loops
A model that learns from the data its own decisions generated can lock in its mistakes: recommenders only learn about items they show; fraud models only see labels for transactions they let through; predictive policing concentrates patrols where it already predicted crime.

Measured in the companion code (a recommender estimating CTR from its own impressions):

| Exploration | Realized CTR | Best-5 items discovered |
|---|---|---|
| 0% | 0.119 | **20%** |
| 5% | **0.237** | 100% |
| 15% | 0.224 | 100% |

(Optimal achievable CTR: 0.247.) With no exploration the system locks onto items that got lucky early and earns half the achievable CTR. **Remedies:** reserve traffic for exploration (ε-greedy, Thompson sampling); log the **propensity** of every impression; correct training data with inverse propensity weighting; use off-policy evaluation before launching new policies; keep a random holdout for unbiased labels.

---

# PART 6 — RESPONSIBLE AND SECURE ML

```
fairness       measure metrics per group (demographic parity, equal opportunity,
               calibration within groups) — these criteria can be mutually
               incompatible, so choose deliberately with stakeholders
privacy        data minimization, retention limits, anonymization, differential
               privacy, federated learning for on-device data, right to deletion
               (which makes retraining and lineage mandatory)
explainability SHAP/feature attributions for regulated decisions (credit,
               insurance, hiring); reason codes for adverse actions
security       adversarial inputs, data poisoning, model extraction, prompt
               injection for LLM systems; rate limiting and input validation
human review   route low-confidence or high-stakes decisions to people; the
               review decisions become new labels
```

---

# PART 7 — CASE STUDIES

Each follows the framework. In an interview, spend the most time where the case is **distinctive** — the "key challenge" line tells you where that is.

## 7.1 News feed / social feed ranking
```
Goal       long-term engagement and satisfaction, not just clicks
Frame      multi-task: P(click), P(like), P(comment), P(share), P(hide),
           P(long dwell) -> weighted value score
Metrics    online: sessions/user, time spent, retention, hide/report rate;
           offline: per-task AUC / log loss, NDCG
Data       impression logs with engagement; heavy position bias; 1B+ items
Features   user history embeddings, author affinity, content embeddings,
           freshness, social graph signals, real-time counters
Model      candidates from followed accounts + two-tower ANN + trending ->
           multi-task deep ranker -> diversity and integrity re-rank
Serving    precompute candidates, online ranking of ~500 items in <100 ms
Key challenge  clickbait: optimizing clicks alone degrades satisfaction.
           Combine positive and negative signals, use surveys as labels,
           and keep a long-term holdback to measure retention effects.
```

## 7.2 Search ranking (e-commerce or web)
```
Frame      learning to rank: given (query, document) predict relevance
Metrics    offline NDCG@10, MRR; online CTR, zero-result rate, conversion,
           time-to-first-click, reformulation rate (a sign of failure)
Data       click logs (biased by position) + human relevance judgements
Model      query understanding (spelling, intent, entities) -> retrieval
           (BM25 + dense, hybrid) -> learning-to-rank (LambdaMART or a
           cross-encoder on the top ~100)
Key challenge  position bias in click labels -> randomization experiments
           or click models to de-bias; head vs tail queries behave
           differently (tail queries need semantic retrieval)
```

## 7.3 Ads click-through-rate prediction
```
Frame      P(click | user, ad, context); auction ranks by bid x P(click)
Metrics    log loss and CALIBRATION (the probability is multiplied by money),
           AUC; online revenue, advertiser ROI, user experience guardrails
Data       billions of impressions/day, ~1% CTR -> heavy negative downsampling,
           which REQUIRES recalibration afterwards
Features   sparse high-cardinality IDs (hashing, embeddings), crosses
Model      logistic regression with crosses -> factorization machines ->
           deep & cross / wide & deep; ONLINE learning for freshness
Key challenge  calibration and freshness. A 1% calibration error is a 1%
           billing error across the whole marketplace.
```

## 7.4 Fraud detection (payments)
```
Frame      P(fraud | transaction) at authorization time; decision =
           approve / challenge (2FA) / decline
Metrics    precision-recall at the operating point, $ fraud caught vs $ of
           good transactions declined; PR-AUC (heavy imbalance, ~0.1%)
Data       labels are chargebacks arriving 30-90 days later; fraudsters adapt
           (adversarial concept drift); only APPROVED transactions get labels
Features   velocity counters (txns in last 1h/24h per card/device/IP),
           amount vs user history, geo distance, device fingerprint, graph
           features (shared devices/cards across accounts)
Model      rules for known patterns + gradient boosting; graph models for
           rings; <50-100 ms at checkout
Key challenge  cost-sensitive thresholds (see 5.6), delayed and biased labels
           (a random approved holdout for unbiased labels), and adversarial
           drift -> frequent retraining and human review queues.
```

## 7.5 ETA / delivery-time prediction
```
Frame      regression: predict arrival time
Metrics    MAE, and ASYMMETRIC cost (late is worse than early) -> quantile
           loss; report P50/P90 intervals, calibration of intervals
Features   route, distance, real-time traffic, weather, time of day, restaurant
           prep time, courier history
Model      gradient boosting or a sequence/graph model over route segments;
           re-predict as the trip progresses
Key challenge  the prediction changes behaviour (quoted time affects
           orders), and real-time features must be fresh within seconds.
```

## 7.6 Content moderation
```
Frame      multi-label classification of posts (hate, violence, spam, nudity)
           across text, images, video
Metrics    precision/recall per policy at each action threshold; prevalence
           of violating content seen by users; appeal overturn rate
Data       human review labels (expensive, subjective) -> inter-rater agreement;
           evolving policies; many languages
Model      multimodal pretrained encoders fine-tuned per policy; LLM-based
           classifiers for nuanced cases
Serving    tiered: cheap model on everything -> expensive model on flagged ->
           human review on uncertain / high-severity
Key challenge  high stakes both ways (over-removal harms speech, under-removal
           harms users), adversarial evasion, and cultural/language coverage.
```

## 7.7 Recommendation for a new-item-heavy catalogue (e.g. video, marketplace)
```
Key challenge  COLD START
  new users   onboarding preferences, popularity by segment, contextual bandit
  new items   content-based embeddings in the item tower, exploration budget
              for new items, boost-and-decay freshness scoring
Also         feedback loops (5.7): log propensities, reserve exploration traffic
```

## 7.8 Spam / abuse detection at signup
```
Frame      P(abusive account) at signup and over the first days
Features   IP/device reputation, email domain age, signup velocity, behavioural
           sequences, graph links to known bad accounts
Key challenge  adversaries probe the boundary: never expose scores, add friction
           (CAPTCHA) instead of hard blocks when uncertain, retrain frequently,
           and use graph-based detection for coordinated rings.
```

## 7.9 Customer-support assistant (LLM + RAG)
```
Goal       resolve tickets faster without wrong answers
Frame      retrieve relevant help articles and past tickets, generate a grounded
           answer with citations; escalate to a human when unsure
Metrics    offline: retrieval recall@k, answer faithfulness, correctness on a
           golden set (human-graded + LLM-as-judge calibrated to humans);
           online: resolution rate, escalation rate, CSAT, handle time
Data       help-centre docs (chunked, versioned), resolved tickets (PII removed)
Serving    hybrid retrieval + reranker -> LLM with a strict grounding prompt ->
           guardrails (PII filter, policy checks) -> human handoff
Cost       tokens x requests; cache frequent answers; route easy intents to a
           smaller model
Key challenge  hallucination and prompt injection (treat retrieved text as data,
           never instructions); keeping the index fresh as policies change.
           See NLP notes Part 9.
```

## 7.10 Visual search ("find similar products from a photo")
```
Frame      embed the query image; nearest-neighbour search over product images
Model      pretrained vision encoder fine-tuned with a contrastive/triplet loss
           on (query photo, matching product) pairs; object detection to crop
           the relevant item first
Serving    precompute product embeddings; ANN index (HNSW/IVF-PQ); re-rank by
           price/availability
Metrics    recall@k on a labelled set; online click and add-to-cart rate
Key challenge  domain gap between user photos (bad lighting, clutter) and
           studio catalogue images -> augmentation, and training pairs mined
           from real user queries.
```

---

# PART 8 — BACK-OF-ENVELOPE ESTIMATION

State assumptions, show arithmetic, round aggressively. Worked in the companion code:

```
10M DAU x 20 requests/day          = 200M requests/day
average QPS  = 200M / 86,400       ~ 2,300
peak QPS     = 3x average          ~ 6,900
one GPU replica: batch 8 in 20 ms  = 400 QPS raw, x 0.6 utilization = 240 QPS
replicas     = 6,900 / 240         ~ 29
7B model in fp16                   = 14 GB (before KV cache)
cost         = 29 x $4/h x 720 h   ~ $84K / month
after 4-bit quantization (0.5 B/param, ~30% faster): 21 replicas, 3.5 GB,
~$60K / month -> saves ~$23K / month
```
Other quick estimates: storage for 1B users × 256-float embeddings × 4 bytes = 1 TB; an ANN index with PQ compression can cut that 10–30×; logging 200M predictions/day at 1 KB each = 200 GB/day.

---

# PART 9 — RAPID-FIRE Q&A

**Q: Your model's offline AUC improved but the A/B test shows no gain. Why?** Offline/online mismatch: the offline metric doesn't track the business metric; the evaluation data came from the old policy's exposures (selection bias); leakage inflated offline results; position bias; the test was underpowered; or the improvement is in a segment that doesn't matter.

**Q: How do you detect leakage?** Metrics that look too good, one dominant feature, performance collapsing on a strictly later time split, and auditing every feature for "available at prediction time?" Use point-in-time joins.

**Q: What is training-serving skew and how do you prevent it?** The same feature computed differently offline and online. Prevent with one shared feature definition (a feature store), training on logged serving features, and comparing online vs offline feature distributions.

**Q: Your model's performance dropped in production. Walk me through debugging.** Check whether anything changed: deploys, upstream data schemas, feature pipelines, traffic mix. Check data quality and null rates, then input drift (PSI per feature), then prediction distribution, then performance by slice once labels arrive. Distinguish data problems from concept drift. Roll back if needed while investigating.

**Q: Batch or real-time predictions?** Batch when predictions don't depend on request context and can tolerate staleness (cheaper, simpler). Real-time when they need fresh context (search query, cart contents, transaction details). Hybrid is common: precompute the expensive parts.

**Q: How do you handle cold start?** New users: onboarding, popularity by segment, contextual bandits. New items: content features in the model, exploration budget, freshness boosts.

**Q: How do you choose a retraining frequency?** Measure how fast performance decays (train on month N, test on N+1, N+2...). Retrain faster than it decays, gate every retrain on validation, and trigger extra retrains on drift or performance alerts.

**Q: Why not just use accuracy?** Class imbalance makes it meaningless, and it ignores the asymmetric cost of errors. Use precision/recall at an operating point chosen by cost, PR-AUC, or a business-cost metric.

**Q: How do you reduce serving latency?** Profile the latency budget first. Then: caching, precomputation, batching, a smaller/distilled/quantized model, fewer or cheaper features, parallel feature fetches, a cascade (cheap model first, expensive only when needed), and hardware acceleration.

**Q: What is a feedback loop and how do you break it?** The model's decisions shape its future training data, reinforcing its own biases. Break it with exploration traffic, propensity logging and inverse propensity weighting, random holdouts for unbiased labels, and off-policy evaluation.

**Q: How long should an A/B test run?** Until the pre-computed sample size is reached — and at least one or two full weekly cycles to cover day-of-week effects and novelty. Never stop early because it looks significant.

**Q: How would you monitor a model whose labels arrive 60 days later?** Monitor inputs (drift), predictions (distribution, % positive), and proxy labels that arrive faster; evaluate on true labels with a lag; keep a small random holdout.

**Q: What's the difference between a canary and an A/B test?** A canary protects against operational failure — bugs, crashes, latency, skew — with automatic rollback. An A/B test measures whether the change improves the business metric. You usually canary first, then A/B test.

**Q: How do you design for failure?** Fallbacks (cached predictions, a simpler model, rules), timeouts and circuit breakers on dependencies, graceful degradation, instant rollback via the registry, and alerts on both system and model health.

---

# PART 10 — COVERAGE INDEX

| Concept | Implementation in `mlsys_from_scratch.py` | Demo |
|---|---|---|
| Schema / null / range / domain validation | `DataValidator` | 1 |
| Feature store, point-in-time join | `FeatureStore` | 2 |
| Label leakage (temporal) | `make_churn_events`, naive vs PIT features | 2 |
| Training-serving skew | `skewed_online_features` | 2 |
| Model registry, gating, rollback | `ModelRegistry` | 3 |
| M/M/1 queueing, Little's law | `mm1_stats` | 4 |
| Dynamic batching (discrete-event sim) | `simulate_batching` | 4 |
| LRU prediction cache, Zipf traffic | `LRUCache`, `zipf_keys` | 4 |
| Hashing trick | `hash_feature` | 4 |
| ANN via random-hyperplane LSH | `LSHIndex` | 5 |
| Two-stage recommender funnel | `TwoStageRecommender` | 5 |
| MMR diversity re-ranking | `mmr_rerank` | 5 |
| Two-proportion z-test, CI | `two_proportion_ztest` | 6 |
| Sample-size calculation | `sample_size_per_arm` | 6 |
| Peeking / optional stopping | `simulate_peeking` | 6 |
| Canary with auto-rollback | `canary_deploy` | 6 |
| Shadow deployment | `shadow_compare` | 6 |
| PSI, Kolmogorov-Smirnov | `psi`, `ks_statistic` | 7 |
| Covariate vs concept drift | `drifting_stream` | 7 |
| Static / periodic / triggered retraining | `retraining_policies` | 7 |
| Cost-based thresholds | `best_threshold_by_cost` | 8 |
| ECE, Platt, isotonic (PAV) | `expected_calibration_error`, `PlattScaler`, `IsotonicCalibrator` | 8 |
| Feedback loops and exploration | `simulate_feedback_loop` | 8 |
| Capacity and cost planning | `capacity_plan` | 9 |

**Covered in the notes, not implemented** — be ready to explain: streaming feature pipelines (Kafka/Flink), HNSW and IVF-PQ internals, multi-task rankers, position-bias click models, interleaving, CUPED, switchback and cluster randomization, off-policy evaluation estimators, fairness criteria, differential privacy, federated learning.

---

# PART 11 — 7-DAY PLAN

```
Day 1  The framework (Part 0) until you can recite it. Practise clarifying
       questions on three prompts without designing anything.
Day 2  Data layer: leakage, point-in-time joins, skew, validation. Run demos 1-3
       and explain each result in one sentence.
Day 3  Serving: batch vs online, latency budgets, queueing, batching, caching,
       fallbacks. Run demo 4. Do three back-of-envelope estimates (demo 9).
Day 4  Retrieval & ranking funnel, two-tower, ANN, position bias, diversity.
       Run demo 5. Case studies 7.1, 7.2, 7.7.
Day 5  Experimentation: A/B math, sample size, peeking, canary, shadow, pitfalls.
       Run demo 6.
Day 6  Monitoring, drift, retraining, calibration, thresholds, feedback loops.
       Run demos 7-8. Case studies 7.3, 7.4.
Day 7  Full mock interviews: pick 3 case studies, 45 minutes each, out loud,
       following the framework. Then the rapid-fire Q&A.
```

**Self-test — without notes, can you:**
1. Recite the 8-step framework and the time split?
2. Name four sources of label leakage and how to detect each?
3. Explain why training-serving skew can leave AUC unchanged while breaking the product?
4. Explain why latency explodes near full utilization, and what utilization to plan for?
5. Explain what dynamic batching trades off, and when the timeout is pure cost?
6. Draw the retrieval → ranking → re-ranking funnel and justify each stage?
7. Compute an A/B sample size and explain why peeking is wrong?
8. Distinguish covariate from concept drift, and say which needs labels?
9. Pick a classification threshold from error costs?
10. Explain a feedback loop and three ways to break it?
11. Do a capacity estimate from DAU to replicas and monthly cost?
12. Walk through two full case studies end to end in 45 minutes each?

---
*Companion code: `mlsys_from_scratch.py`. Related: `ML_Interview_Notes.md` (metrics, validation), `NLP_LLM_Interview_Notes.md` (RAG, LLM serving), `RL_Interview_Notes.md` (bandits, exploration).*
