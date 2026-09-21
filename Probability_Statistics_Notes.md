# PROBABILITY & STATISTICS — INTERVIEW STUDY NOTES

[ML](ML_Interview_Notes.md) · [DL](DL_Interview_Notes.md) · [RL](RL_Interview_Notes.md) · [NLP & LLMs](NLP_LLM_Interview_Notes.md) · [ML System Design](ML_System_Design_Notes.md) · **Probability & Stats**

> Probability, estimation, intervals, testing, regression inference, Bayesian methods, experiment design, and causal inference. Deepens the short overview in the ML notes (Part 1.3–1.4).
> **Companion code:** [`stats_from_scratch.py`](stats_from_scratch.py) — pure Python, standard library only. Every distribution function is validated against standard tables, and every interval and test is checked by simulation.
> **Numbers quoted from the code** come from seeded simulations. Each is labelled with its demo function, seed, and number of repetitions: read them as illustrations of a mechanism, and remember that simulated rates carry Monte Carlo error (roughly ±1 percentage point at a few thousand repetitions).
> **Math:** key equations are rendered (GitHub `math` blocks); ASCII code blocks are kept as compact reference.
> **Questions and puzzles** are collapsible — answer first, then expand.

## Contents

  - [THE FIVE TRAPS THIS TOPIC TESTS](#the-five-traps-this-topic-tests)
- [PART 0 — CHEAT SHEET](#part-0--cheat-sheet)
- [PART 1 — PROBABILITY](#part-1--probability)
  - [1.1 Conditioning and independence](#11-conditioning-and-independence)
  - [1.2 Bayes' rule and base rates](#12-bayes-rule-and-base-rates)
  - [1.3 Expectation and its tricks](#13-expectation-and-its-tricks)
  - [1.4 Classic probability, and why the protocol matters](#14-classic-probability-and-why-the-protocol-matters)
  - [1.5 Markov chains](#15-markov-chains)
- [PART 2 — DISTRIBUTIONS AND LIMIT THEOREMS](#part-2--distributions-and-limit-theorems)
  - [2.1 The distributions to know](#21-the-distributions-to-know)
  - [2.2 How they connect](#22-how-they-connect)
  - [2.3 The law of large numbers and the central limit theorem — with conditions](#23-the-law-of-large-numbers-and-the-central-limit-theorem--with-conditions)
- [PART 3 — ESTIMATION](#part-3--estimation)
  - [3.1 Bias, variance, MSE, consistency](#31-bias-variance-mse-consistency)
  - [3.2 Maximum likelihood](#32-maximum-likelihood)
  - [3.3 Unbiased is not the same as best](#33-unbiased-is-not-the-same-as-best)
  - [3.4 The bootstrap](#34-the-bootstrap)
- [PART 4 — CONFIDENCE INTERVALS](#part-4--confidence-intervals)
  - [4.1 What a confidence interval means — and what it doesn't](#41-what-a-confidence-interval-means--and-what-it-doesnt)
  - [4.2 z vs t](#42-z-vs-t)
  - [4.3 Intervals for proportions: Wald vs Wilson](#43-intervals-for-proportions-wald-vs-wilson)
  - [4.4 Bootstrap intervals and skewed data](#44-bootstrap-intervals-and-skewed-data)
- [PART 5 — HYPOTHESIS TESTING](#part-5--hypothesis-testing)
  - [5.1 The framework](#51-the-framework)
  - [5.2 What a p-value is NOT](#52-what-a-p-value-is-not)
  - [5.3 Choosing a test](#53-choosing-a-test)
  - [5.4 Assumptions matter — measured](#54-assumptions-matter--measured)
  - [5.5 Power and sample size](#55-power-and-sample-size)
  - [5.6 Permutation tests](#56-permutation-tests)
- [PART 6 — MULTIPLE TESTING](#part-6--multiple-testing)
  - [6.1 FWER vs FDR](#61-fwer-vs-fdr)
  - [6.2 Measured](#62-measured)
  - [6.3 How many significant results are false?](#63-how-many-significant-results-are-false)
- [PART 7 — REGRESSION INFERENCE](#part-7--regression-inference)
  - [7.1 OLS assumptions — and what each one buys you](#71-ols-assumptions--and-what-each-one-buys-you)
  - [7.2 Standard errors: classic vs robust](#72-standard-errors-classic-vs-robust)
  - [7.3 Multicollinearity](#73-multicollinearity)
  - [7.4 Omitted variable bias](#74-omitted-variable-bias)
  - [7.5 Regression to the mean](#75-regression-to-the-mean)
  - [7.6 Interpreting coefficients](#76-interpreting-coefficients)
- [PART 8 — BAYESIAN INFERENCE](#part-8--bayesian-inference)
  - [8.1 Priors, posteriors, conjugacy](#81-priors-posteriors-conjugacy)
  - [8.2 Credible vs confidence intervals](#82-credible-vs-confidence-intervals)
  - [8.3 Bayesian A/B comparison](#83-bayesian-ab-comparison)
- [PART 9 — EXPERIMENT DESIGN](#part-9--experiment-design)
  - [9.1 Why randomization works](#91-why-randomization-works)
  - [9.2 Getting more precision from the same sample](#92-getting-more-precision-from-the-same-sample)
  - [9.3 Common design failures](#93-common-design-failures)
- [PART 10 — CAUSAL INFERENCE](#part-10--causal-inference)
  - [10.1 Potential outcomes](#101-potential-outcomes)
  - [10.2 Confounders, mediators, colliders](#102-confounders-mediators-colliders)
  - [10.3 Simpson's paradox](#103-simpsons-paradox)
  - [10.4 Adjusting for measured confounders](#104-adjusting-for-measured-confounders)
  - [10.5 Quasi-experiments](#105-quasi-experiments)
- [PART 11 — PROBABILITY PUZZLES](#part-11--probability-puzzles)
- [PART 12 — RAPID-FIRE Q&A](#part-12--rapid-fire-qa)
- [PART 13 — COVERAGE INDEX](#part-13--coverage-index)
- [PART 14 — 10-DAY PLAN](#part-14--10-day-plan)
- [REFERENCES](#references)

---

## THE FIVE TRAPS THIS TOPIC TESTS

Statistics interviews rarely test arithmetic. They test whether you know **when a claim holds**. Almost every wrong answer is one of these:

1. **Reversing a conditional.** `P(data | H0)` is not `P(H0 | data)`; `P(positive | sick)` is not `P(sick | positive)`.
2. **Ignoring how the data were gathered.** The same observed fact can imply different probabilities under different protocols (Monty Hall, the two-children puzzle, selection bias).
3. **Using a result outside its conditions.** The CLT needs finite variance; the pooled t-test needs equal variances; the Wald interval needs `np` large; classic regression SEs need constant error variance.
4. **Confusing a property of the procedure with a property of the result.** A 95% confidence interval is not "95% likely to contain the truth"; a test's 5% false-positive rate says little about how many *significant* results are false.
5. **Mistaking association for causation.** Confounding can flip the sign of an effect; aggregation can reverse a comparison.

The companion code exists to make each trap visible with numbers.

---

# PART 0 — CHEAT SHEET

```
PROBABILITY
  P(A|B) = P(A∩B)/P(B)            Bayes  P(A|B) = P(B|A)P(A)/P(B)
  total probability  P(B) = sum_i P(B|A_i) P(A_i)
  linearity  E[X+Y] = E[X] + E[Y]   ALWAYS, even for dependent X, Y
  Var(X+Y) = Var X + Var Y + 2 Cov(X,Y)
  total expectation  E[X] = E[ E[X|Y] ]
  total variance     Var X = E[ Var(X|Y) ] + Var( E[X|Y] )

ESTIMATION
  MSE = bias^2 + variance       sample variance: divide by n-1 (unbiased)
  SE of a mean = sigma/sqrt(n)  SE of a proportion = sqrt(p(1-p)/n)
  MLE: asymptotically unbiased, normal, efficient — under regularity conditions

INTERVALS (95%)
  mean, sigma unknown   xbar +/- t_{n-1, .975} s/sqrt(n)
  proportion            prefer WILSON to Wald (Wald fails near 0 or 1)
  anything else         bootstrap (percentile, or BCa for small samples)

TESTS — pick by data type and design
  2 group means            Welch t (default; no equal-variance assumption)
  same units before/after  paired t
  heavy tails / ordinal    Mann-Whitney U (rank-based)
  2 categorical variables  chi-square (expected counts >= ~5) or Fisher exact
  3+ group means           ANOVA (Welch ANOVA if variances differ)
  any statistic            permutation test

POWER
  n per group ~ 2 (z_{1-a/2} + z_{1-b})^2 sigma^2 / delta^2
  alpha .05, power .8  ->  n ~ 16 sigma^2 / delta^2 per group
  effect 0.5 sd -> ~64 per group ; 0.2 sd -> ~400 per group

MULTIPLE TESTING
  Bonferroni   p <= alpha/m            controls FWER (any false positive)
  BH           largest k: p_(k) <= k q/m   controls FDR (share of false positives)

CAUSAL
  confounder   causes both T and Y   -> adjust for it
  mediator     on the path T -> M -> Y -> do NOT adjust (if you want the total effect)
  collider     caused by both T and Y -> do NOT adjust (creates spurious association)
```

**Measured in the companion code — the numbers to quote** (all from [`stats_from_scratch.py`](stats_from_scratch.py); seeds and repetition counts in the sections below):

| Trap | What the simulation shows |
|---|---|
| Base rates | 1% prevalence, 99%-sensitive, 95%-specific test → only **16.7%** of positives are sick |
| z interval at small n | "95%" interval covers **87.5%** at n = 5 (t interval: 94.8%) |
| Wald interval, rare event | "95%" interval covers **64.0%** at p = 0.05, n = 20 (Wilson: 92.2%) |
| Pooled t, unequal variances | false-positive rate **28.5%** instead of 5% (Welch: 5.5%) |
| t-test, heavy tails | power **6.2%** vs Mann-Whitney **49.4%** on shifted Cauchy data |
| No multiple-testing correction | **39%** of "discoveries" false; a false positive in every study |
| Heteroscedastic errors | classic 95% CI for a slope covers **87.3%** (robust: 93.5%) |
| Confounding | naive effect estimate **−1.13** when the truth is **+1** |
| CLT without finite variance | averaging 200 Cauchy draws: spread unchanged from one draw |

---

# PART 1 — PROBABILITY

## 1.1 Conditioning and independence

```math
P(A\mid B) = \frac{P(A\cap B)}{P(B)},\qquad
P(B) = \sum_i P(B\mid A_i)\,P(A_i),\qquad
P(A_i\mid B) = \frac{P(B\mid A_i)\,P(A_i)}{\sum_j P(B\mid A_j)\,P(A_j)}
```

- **Independence** `P(A∩B) = P(A)P(B)` is a property of the probability model, not of the events' meanings. Pairwise independence does not imply mutual independence.
- **Conditional independence** does not imply independence, and vice versa. Two coin flips are independent; conditioning on "exactly one head" makes them dependent. This is the mechanism behind **collider bias** (Part 10.2).
- **Mutually exclusive ≠ independent.** Disjoint events with non-zero probability are strongly *dependent*: if one happens, the other cannot.

## 1.2 Bayes' rule and base rates

Measured in `demo_probability()`, [`stats_from_scratch.py`](stats_from_scratch.py) (exact calculation, confirmed by simulating 200,000 people, seed 0): with prevalence 1%, sensitivity 99%, and specificity 95%, **P(sick | positive) = 0.167** (simulated 0.168).

```math
P(\text{sick}\mid +) = \frac{0.99 \times 0.01}{0.99\times 0.01 + 0.05\times 0.99} \approx 0.167
```

The 5% false-positive rate applied to 99% healthy people produces five times as many false positives as there are true positives. **Base-rate neglect** is the most common reasoning error in interviews and in practice: fraud alerts, medical screening, content moderation, and anomaly detection all live in this regime. The fix is not a better test alone but a better *prior*: screen a higher-risk population, or follow up positives with a second independent test.

## 1.3 Expectation and its tricks

```math
\mathbb E\Big[\sum_i X_i\Big] = \sum_i \mathbb E[X_i]\quad\text{(always)},\qquad
\mathbb E[X] = \mathbb E\big[\mathbb E[X\mid Y]\big],\qquad
\operatorname{Var}(X) = \mathbb E\big[\operatorname{Var}(X\mid Y)\big] + \operatorname{Var}\big(\mathbb E[X\mid Y]\big)
```

**The indicator trick** — write a count as a sum of 0/1 indicators and take expectations one at a time. Linearity holds even when the indicators are **dependent**, which is what makes this so powerful.

- *Fixed points of a random permutation:* `X = Σ 1[π(i) = i]`, each with probability `1/n`, so **E[X] = 1 for every n**. Simulated in `demo_probability()` (20,000 permutations each, seed 0): 0.997, 0.987, 1.000 for n = 3, 10, 50.
- *Coupon collector:* with `i` of `n` coupons collected, the wait for a new one is geometric with mean `n/(n−i)`, so `E[draws] = n·H_n`. For n = 20: exact **71.95**, simulated 71.93 (5,000 runs).

**Tail-sum formula** for non-negative integer variables: `E[X] = Σ_{k≥1} P(X ≥ k)`. It gives `E[max of two dice] = 161/36` in two lines (Part 11).

## 1.4 Classic probability, and why the protocol matters

**Monty Hall.** Switching wins 2/3 **because the host knows where the car is and always opens a goat door**. If the host opened a random door that merely happened to show a goat, switching would win only 1/2. Simulated (100,000 games, seed 0): stay 0.335, switch 0.665.

**Birthday problem** (assumes uniform, independent birthdays — real birthdays are neither, which slightly *raises* the collision probability):

```math
P(\text{shared birthday among } k) = 1 - \prod_{i=0}^{k-1}\frac{365-i}{365}
```

23 people already exceed 50% (exact 0.5073, simulated 0.5049 over 40,000 groups): collisions scale with the number of **pairs**, 253 among 23 people, not with the number of people. The same arithmetic governs hash collisions and duplicate detection.

**Gambler's ruin** — betting one unit with win probability `p`, starting at `s`, stopping at 0 or `N`:

```math
P(\text{reach } N) = \begin{cases} s/N & p = \tfrac12\\[4pt] \dfrac{1-r^{s}}{1-r^{N}},\quad r = \dfrac{1-p}{p} & p \ne \tfrac12\end{cases}
```

From 10 to a target of 20: 0.500 at `p = 0.50`, but **0.231 at `p = 0.47`** (simulated 0.502 and 0.236, 20,000 runs). A small edge compounds over a long path.

## 1.5 Markov chains

```math
\pi = \pi P,\qquad \sum_i \pi_i = 1
```

The stationary distribution is the long-run fraction of time in each state. It **exists and is unique for an irreducible finite chain**; the chain *converges* to it from any start when the chain is also **aperiodic**. Detailed balance `π_i P_ij = π_j P_ji` is sufficient (not necessary) for stationarity — it is the property MCMC samplers are built to satisfy. In `demo_probability()` a 3-state chain gives power-iteration `π = (0.6429, 0.2857, 0.0714)` and 200,000 simulated steps give `(0.6422, 0.2866, 0.0711)`.

Applications: PageRank (stationary distribution of a random surfer with teleportation, which enforces irreducibility and aperiodicity), MCMC, HMMs, queueing models.

---

# PART 2 — DISTRIBUTIONS AND LIMIT THEOREMS

## 2.1 The distributions to know

| Distribution | Models | Mean | Variance | Note |
|---|---|---|---|---|
| Bernoulli(p) | one yes/no trial | p | p(1−p) | variance maximal at p = ½ |
| Binomial(n,p) | successes in n independent trials | np | np(1−p) | independence + constant p |
| Geometric(p) | trials until first success | 1/p | (1−p)/p² | memoryless (discrete) |
| Poisson(λ) | counts of rare independent events | λ | λ | mean = variance; overdispersion ⇒ use negative binomial |
| Uniform(a,b) | "no information" on an interval | (a+b)/2 | (b−a)²/12 | |
| Exponential(λ) | waiting time between Poisson events | 1/λ | 1/λ² | memoryless (continuous) |
| Normal(μ,σ²) | sums of many small effects | μ | σ² | ~68 / 95 / 99.7% within 1 / 2 / 3 σ |
| Beta(α,β) | a probability, uncertain | α/(α+β) | — | conjugate prior for binomial p |
| Gamma(k,θ) | sum of k exponentials | kθ | kθ² | conjugate prior for Poisson rate |
| Student t(ν) | standardized mean, σ estimated | 0 (ν>1) | ν/(ν−2) (ν>2) | heavier tails; → normal as ν → ∞ |
| χ²(k) | sum of k squared standard normals | k | 2k | variances, goodness of fit |
| Cauchy | ratio of two standard normals | **undefined** | **undefined** | = t with 1 df; breaks the CLT |

## 2.2 How they connect

```
Binomial(n, p), n large & p small, np = lambda  ->  Poisson(lambda)
Binomial(n, p), np(1-p) large                   ->  Normal(np, np(1-p))
sum of k Exponential(lambda)                    =   Gamma(k, 1/lambda)
Z^2, Z standard normal                          =   chi-square(1)
Z / sqrt(chi2_k / k), independent               =   t(k)
(chi2_a / a) / (chi2_b / b), independent        =   F(a, b)       t(k)^2 = F(1, k)
Poisson process: counts ~ Poisson, gaps ~ Exponential
```

**Sampling:** if `U ~ Uniform(0,1)` then `F⁻¹(U)` has CDF `F` (**inverse-CDF sampling**; exponential: `−ln(1−U)/λ`). **Box-Muller** turns two uniforms into normals. **Rejection sampling** draws from a simple envelope and accepts with probability proportional to the target density. All three are implemented in the companion file.

## 2.3 The law of large numbers and the central limit theorem — with conditions

```math
\bar X_n \xrightarrow{\ \text{a.s.}\ } \mu \ \ \text{(LLN; needs } \mathbb E\lvert X\rvert<\infty\text{)},\qquad
\sqrt n\,\frac{\bar X_n-\mu}{\sigma}\ \xrightarrow{\ d\ }\ \mathcal N(0,1)\ \ \text{(CLT; i.i.d., needs } \sigma^2<\infty\text{)}
```

The CLT is a statement about the *limit*. **How large n must be depends on the data.** "n ≥ 30" is a folk rule, not a theorem.

Measured in `demo_limit_theorems()`, [`stats_from_scratch.py`](stats_from_scratch.py) (3,000 simulated sample means per row, seed 0):

| n | Exponential: skewness of the mean (theory 2/√n) | Exponential: IQR of the mean | **Cauchy**: IQR of the mean |
|---|---|---|---|
| 1 | 2.107 (2.000) | 1.108 | 2.002 |
| 10 | 0.717 (0.632) | 0.406 | 1.980 |
| 200 | 0.026 (0.141) | 0.094 | 2.049 |

(Simulated skewness is itself noisy by roughly ±0.1.) For Exponential data the CLT works but slowly — at n = 30 the theoretical skewness of the mean is still 0.37. For Cauchy data **averaging does nothing**: the mean of n standard Cauchy draws is again standard Cauchy, whose IQR is 2. Heavy-tailed real data (incomes, file sizes, latencies, insurance claims) can behave closer to the Cauchy column than the Exponential one at practical sample sizes — which is why medians, trimmed means, and rank-based tests exist.

---

# PART 3 — ESTIMATION

## 3.1 Bias, variance, MSE, consistency

```math
\operatorname{MSE}(\hat\theta) = \mathbb E\big[(\hat\theta-\theta)^2\big] = \underbrace{\big(\mathbb E[\hat\theta]-\theta\big)^2}_{\text{bias}^2} + \underbrace{\operatorname{Var}(\hat\theta)}_{\text{variance}}
```

- **Unbiased:** `E[θ̂] = θ`. **Consistent:** `θ̂ → θ` in probability as `n → ∞`. Neither implies the other (an estimator can be biased but consistent, e.g. the divide-by-n variance; or unbiased but inconsistent, e.g. using only the first observation).
- **Standard error** is the standard deviation of the estimator's sampling distribution — not the standard deviation of the data.

## 3.2 Maximum likelihood

```math
\hat\theta_{\text{MLE}} = \arg\max_\theta \sum_i \log p(x_i\mid\theta),\qquad
I(\theta) = -\mathbb E\!\left[\frac{\partial^2}{\partial\theta^2}\log p(X\mid\theta)\right],\qquad
\operatorname{Var}(\hat\theta) \ \ge\ \frac{1}{n\,I(\theta)}\ \ \text{(Cramér–Rao, unbiased } \hat\theta\text{)}
```

**Properties — under regularity conditions** (the true parameter is in the interior of the parameter space, the support does not depend on θ, the model is correctly specified, the likelihood is smooth enough): the MLE is consistent, asymptotically normal, and asymptotically efficient (attains the Cramér–Rao bound). It is also **invariant**: the MLE of `g(θ)` is `g(θ̂)`.

**When those conditions fail, so do the guarantees.** For Uniform(0, θ) the support depends on θ; the MLE is the sample maximum, which is biased low (the German tank problem, Part 11). Under **misspecification**, the MLE converges to the parameter that minimizes KL divergence to the truth — a well-defined target, but not "the truth".

## 3.3 Unbiased is not the same as best

Measured in `demo_estimation()`, [`stats_from_scratch.py`](stats_from_scratch.py):

**Bessel's correction** (n = 5, σ² = 1, 40,000 samples, seed 0): dividing by `n` averages **0.797** (theory 0.800); dividing by `n − 1` averages **0.996** (theory 1). But the resulting `s` still averages **0.938**, not 1: an unbiased variance does not give an unbiased standard deviation, because the square root is concave (Jensen).

**Shrinkage** — estimate the mean as `c · x̄` (n = 5, σ = 2, 20,000 samples per row, seed 0):

| c | MSE when true mean = 0.5 | MSE when true mean = 3.0 |
|---|---|---|
| 1.0 (unbiased) | 0.804 | **0.804** |
| 0.8 | 0.525 | 0.873 |
| 0.4 | **0.218** | 3.367 |

Shrinking trades bias for variance. It **wins** when the true value is small relative to the noise and **loses badly** when it isn't. That conditional benefit is the logic of ridge regression, empirical Bayes, and the James–Stein result (for three or more normal means with known variance, estimated jointly, shrinking toward a common point dominates the sample means in total squared error).

## 3.4 The bootstrap

Resample the data **with replacement** B times, recompute the statistic, and use the spread of the replicates. In `demo_estimation()` (one sample of 40 Exponential values, 2,000 resamples) the bootstrap SE of the mean is 0.398 against the formula's 0.399, and it delivers an SE for the **median** (0.701), which has no simple formula.

**When it fails:** statistics driven by extremes (the sample maximum), very small samples, heavy tails with infinite variance, and **dependent data** (time series — use a block bootstrap; clustered data — resample clusters). It also assumes the sample is representative of the population: it cannot correct selection bias.

---

# PART 4 — CONFIDENCE INTERVALS

## 4.1 What a confidence interval means — and what it doesn't

```math
\bar x \pm t_{n-1,\,1-\alpha/2}\,\frac{s}{\sqrt n}
```

A 95% confidence interval comes from a **procedure** that, over repeated samples, produces intervals containing the true parameter 95% of the time. Once you have computed one interval from one dataset, the true value is either in it or not; the 95% describes the method.

**Wrong:** "There is a 95% probability that μ is between 4.1 and 6.3." (That is a *Bayesian credible interval* statement, and it requires a prior — Part 8.)
**Right:** "This interval came from a method that captures μ in 95% of samples."
**Also wrong:** "95% of the data lie in the interval" (that's a prediction/tolerance interval idea) and "a 95% CI that excludes 0 means a 95% chance the effect is real".

A **prediction interval** for a new observation is much wider than a confidence interval for the mean — it includes the observation's own noise, not just uncertainty about μ.

## 4.2 z vs t

Measured in `demo_intervals()`, [`stats_from_scratch.py`](stats_from_scratch.py) — mean of Normal(10, 3²) data, 3,000 intervals per row, seed 0:

| n | z-interval coverage (sample s plugged in) | t-interval coverage | t width / z width |
|---|---|---|---|
| 5 | **87.5%** | 94.8% | 1.42× |
| 10 | 91.6% | 94.8% | 1.16× |
| 30 | 93.9% | 94.8% | 1.05× |

Using the normal quantile with an *estimated* σ under-covers at small n. The t distribution pays for the uncertainty in `s`. The t-interval is **exact for normal data at any n**; for non-normal data it is approximate, and with skewed data at small n it under-covers too (measured below).

## 4.3 Intervals for proportions: Wald vs Wilson

```math
\text{Wald: }\hat p \pm z\sqrt{\frac{\hat p(1-\hat p)}{n}},\qquad
\text{Wilson: }\frac{\hat p + \frac{z^2}{2n}}{1+\frac{z^2}{n}} \pm \frac{z}{1+\frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}
```

Measured in `demo_intervals()` (4,000 simulated samples per row, seed 2):

| true p | n | Wald coverage | Wilson coverage |
|---|---|---|---|
| 0.50 | 20 | 95.9% | 95.9% |
| 0.05 | 20 | **64.0%** | 92.2% |
| 0.05 | 100 | 87.0% | 96.8% |
| 0.01 | 100 | **63.0%** | 92.3% |

The textbook Wald interval collapses for rare events (with zero successes it has **zero width**). Coverage of discrete intervals also oscillates with n, so no interval hits exactly 95% everywhere; Wilson is the standard recommendation (Brown, Cai & DasGupta 2001). This matters for conversion rates, error rates, defect rates — anything small. Quick rule for zero events: the **rule of three** gives an approximate 95% upper bound of `3/n`.

## 4.4 Bootstrap intervals and skewed data

Measured in `demo_intervals()` — Exponential data, n = 15 (seed 5): the t-interval covers **92.0%** (2,000 samples) and the bootstrap percentile interval **89.8%** (only 400 samples, so ±~1.5 points of Monte Carlo error). Neither reaches 95%: the t-interval assumes normality, and the plain percentile bootstrap is known to under-cover in small samples. **BCa** (bias-corrected and accelerated) and **bootstrap-t** intervals correct for skewness and do better. The general lesson: a nominal confidence level is a claim that holds under assumptions — check coverage by simulation when the assumptions are doubtful.

---

# PART 5 — HYPOTHESIS TESTING

## 5.1 The framework

```
H0 null hypothesis, H1 alternative
test statistic T, computed from the data
p-value  = P( T at least as extreme as observed | H0 true )
reject H0 if p <= alpha (chosen BEFORE seeing data)

                 H0 true              H0 false
reject H0        Type I error (alpha) correct (power = 1 - beta)
keep H0          correct              Type II error (beta)
```

## 5.2 What a p-value is NOT

A p-value is `P(data this extreme | H0)`. It is **not**:

1. The probability that H0 is true (see Part 6.3 — that depends on the prior and power).
2. The probability the result is due to chance.
3. A measure of effect size — a tiny, useless effect is highly significant with enough data; an important effect can be non-significant in a small study.
4. Evidence for H0 when large. "Not significant" means "not enough evidence against H0", not "no effect". An underpowered study that finds nothing is weak evidence of absence.
5. Comparable across studies as a strength-of-evidence scale without accounting for sample size and design.

The ASA Statement on p-values (Wasserstein & Lazar 2016) is the standard reference for these points. Report **effect sizes with confidence intervals**, not just p-values.

**Under H0, a valid p-value is uniformly distributed** — which is exactly why a well-calibrated test rejects at rate α. That gives a direct check: simulate data where H0 is true and count rejections.

## 5.3 Choosing a test

| Question | Test | Key assumptions |
|---|---|---|
| One mean vs a value | one-sample t | independence; approx. normal mean (CLT) |
| Two independent means | **Welch t** | independence; approx. normal means; *not* equal variances |
| Two independent means, equal variances | Student (pooled) t | as above **plus equal variances** — risky, see 5.4 |
| Same units measured twice | paired t | differences approx. normal |
| Two groups, heavy tails / ordinal | Mann-Whitney U | independence; tests `P(A > B) ≠ ½` (a *median* test only if shapes match) |
| 3+ means | one-way ANOVA (or Welch ANOVA) | as t-test; follow up with corrected pairwise tests |
| Two categorical variables | chi-square | expected counts ≳ 5; otherwise Fisher's exact |
| Paired binary outcomes | McNemar | discordant pairs only |
| Two proportions | two-proportion z (or chi-square) | large counts |
| Any statistic, few assumptions | permutation test | exchangeability under H0 |

## 5.4 Assumptions matter — measured

Measured in `demo_tests()`, [`stats_from_scratch.py`](stats_from_scratch.py):

**Calibration when H0 is true** (3,000 tests each; the rate should be 5%):

| Situation | False-positive rate |
|---|---|
| Welch t, equal variances, n = 20/20 | 5.3% |
| **Student pooled t**, sd 4 vs 1, n = 10 vs 40 | **28.5%** |
| Welch t, same unequal-variance data | 5.5% |

The pooled test breaks when variances differ **and** the smaller group has the larger variance. Welch loses almost nothing when variances are equal, so it is the sensible default.

**Pairing** (15 subjects, true improvement +2, large between-person variation, seed 7): paired t **p = 0.0019**; the same data analysed as independent groups, **p = 0.48**. Pairing removes between-subject variance; ignoring it discards the design's main strength.

**Heavy tails** (shifted Cauchy, n = 30/30, 1,000 tests): Welch t detects the shift **6.2%** of the time, Mann-Whitney **49.4%**. Outliers inflate the t-test's variance estimate and destroy its power; ranks are unaffected by how extreme a value is.

## 5.5 Power and sample size

```math
\text{power} \approx \Phi\!\left(\frac{\lvert\delta\rvert}{\sigma\sqrt{2/n}} - z_{1-\alpha/2}\right),\qquad
n_{\text{per group}} = \frac{2\,(z_{1-\alpha/2}+z_{1-\beta})^2\,\sigma^2}{\delta^2}
```

Measured in `demo_tests()` — Welch t, effect 0.5 sd, 1,000 simulated tests per row:

| n per group | simulated power | normal approximation |
|---|---|---|
| 10 | 16.3% | 20.0% |
| 20 | 34.1% | 35.2% |
| 40 | 61.1% | 60.9% |
| 64 | 82.1% | 80.7% |
| 100 | 94.5% | 94.2% |

The normal approximation gives **63** per group for 80% power; t-based calculations give the textbook **64** (estimating σ costs a little power at small n — visible in the n = 10 row).

Power depends on effect size, noise, n, and α. **Decide the effect size worth detecting before collecting data** — the minimum detectable effect is a product decision, not a statistical one. Post-hoc "observed power" computed from the observed effect adds no information beyond the p-value.

## 5.6 Permutation tests

Under H0 the group labels are exchangeable, so shuffle them and recompute the statistic; the p-value is the fraction of shuffles at least as extreme as observed (with a +1 correction so it is never exactly 0). Works for any statistic — medians, ratios, AUC differences — with almost no distributional assumptions. In `demo_tests()` (n = 25/25, seed 9), a 2,000-shuffle permutation test gives **p = 0.0065**, matching Welch's p = 0.0065 on the same normal data. **Exchangeability is the assumption**: it fails for paired data (shuffle within pairs instead) or when groups differ in variance and you only care about means.

---

# PART 6 — MULTIPLE TESTING

## 6.1 FWER vs FDR

```math
\text{Bonferroni: reject } p_i \le \frac{\alpha}{m}\qquad
\text{BH: } k^\ast = \max\Big\{k : p_{(k)} \le \frac{k}{m}\,q\Big\},\ \text{reject } p_{(1)},\dots,p_{(k^\ast)}
```

- **FWER** (family-wise error rate) = P(at least one false rejection). Bonferroni controls it under **any** dependence.
- **FDR** (false discovery rate) = E[false rejections / total rejections]. Benjamini-Hochberg controls it at `q·m₀/m ≤ q` for **independent or positively dependent** tests; under arbitrary dependence use Benjamini-Yekutieli.
- Use FWER when **any** false positive is costly (a confirmatory clinical endpoint). Use FDR when you are **screening** many candidates and will follow up (genomics, feature screening, many A/B metrics).

## 6.2 Measured

Measured in `demo_multiple_testing()`, [`stats_from_scratch.py`](stats_from_scratch.py) — 200 hypotheses per study, 10% truly non-null (effect 0.8 sd, n = 20/20), **1,000 simulated studies**:

| Rule | FWER | FDR | Power |
|---|---|---|---|
| p < 0.05, no correction | **100%** | **39.0%** | 69.6% |
| Bonferroni | 3.9% | 1.9% | 8.7% |
| Benjamini-Hochberg | 20.9% | **4.5%** | 19.1% |

Without correction every study contains false positives and almost 4 in 10 "discoveries" are false. Bonferroni is safe but finds under 9% of the real effects. BH sits exactly at its theoretical FDR bound (q × fraction null = 0.05 × 0.9 = 4.5%) with more than double Bonferroni's power. (A 60-study run during development gave BH an FDR of 5.7% — Monte Carlo noise, resolved by using 1,000 studies.)

## 6.3 How many significant results are false?

```math
P(H_0\mid p<\alpha) = \frac{\alpha\,(1-\pi)}{\alpha\,(1-\pi) + (1-\beta)\,\pi}
```

where π is the fraction of tested hypotheses that are real effects and `1 − β` is power:

| Share of real effects π | Power | P(H0 true \| significant) |
|---|---|---|
| 50% | 80% | 5.9% |
| 10% | 80% | 36.0% |
| 10% | 30% | 60.0% |
| 1% | 30% | **94.3%** |

In a field that tests long shots with small samples, most significant results are false even with honest analysis (Ioannidis 2005). Add **p-hacking** — trying several analyses, outcome definitions, subgroups, or stopping rules and reporting the one that "worked" (the *garden of forking paths*) — and it gets worse. Defences: pre-registration, fixed analysis plans, correction for the number of looks and comparisons, replication, and reporting effect sizes with intervals.

---

# PART 7 — REGRESSION INFERENCE

## 7.1 OLS assumptions — and what each one buys you

```math
\hat\beta = (X^\top X)^{-1}X^\top y,\qquad
\widehat{\operatorname{Var}}(\hat\beta)_{\text{classic}} = \hat\sigma^2 (X^\top X)^{-1},\qquad
\widehat{\operatorname{Var}}(\hat\beta)_{\text{HC}} = (X^\top X)^{-1}\Big(\sum_i \hat e_i^{\,2}\,x_i x_i^\top\Big)(X^\top X)^{-1}
```

| Assumption | If it holds | If it fails |
|---|---|---|
| Linearity in parameters | β estimates the linear relation | systematic bias; add terms/transform |
| Exogeneity `E[ε\|X] = 0` | **unbiased** β | **biased** β — omitted variables, reverse causation, measurement error in X |
| No perfect collinearity | β identifiable | β undefined; near-collinearity inflates SEs |
| Homoscedasticity | classic SEs valid | β still unbiased, **classic SEs wrong** → use robust SEs |
| Independent errors | classic SEs valid | SEs too small → cluster-robust or time-series SEs |
| Normal errors | exact small-sample t/F tests | only needed for small-sample exactness; CLT covers large n |

Under the first five (the **Gauss–Markov** conditions), OLS is the best *linear* unbiased estimator — normality is **not** required for that.

## 7.2 Standard errors: classic vs robust

Measured in `demo_regression()`, [`stats_from_scratch.py`](stats_from_scratch.py) — 95% CI coverage for a slope, n = 60, 1,500 simulated datasets, seed 0:

| Errors | Classic SE | Robust HC1 SE |
|---|---|---|
| homoscedastic | 94.3% | 93.3% |
| heteroscedastic (noise grows with x) | **87.3%** | 93.5% |

With heteroscedastic errors the slope is still unbiased but the classic SE is too small, so intervals under-cover and p-values are too optimistic. Robust (sandwich) SEs fix it without modelling the variance. They are **not free**: when errors really are homoscedastic, HC1 slightly under-covers in small samples (93.3% here); **HC3** is the usual small-sample choice. With clustered data (users with many rows, students in schools), use **cluster-robust** SEs — ignoring clustering is one of the most common ways to get falsely tiny SEs.

## 7.3 Multicollinearity

```math
\operatorname{VIF}_j = \frac{1}{1-R_j^2},\qquad \operatorname{SE}(\hat\beta_j)\ \propto\ \sqrt{\operatorname{VIF}_j}
```

Measured in `demo_regression()` (x2 = x1 + small noise, x3 independent, n = 300, seed 1): VIFs **121.4, 121.4, 1.0**; coefficients x1 = +1.49 (SE 0.61), x2 = +0.61 (SE 0.60), x3 = +0.95 (SE 0.06), truth 1, 1, 1. The model cannot separate x1 from x2, so their individual coefficients are unstable and their SEs about 10× larger — but their **sum** is well estimated and **predictions are unaffected**. Collinearity is a problem for *interpreting* coefficients, not for prediction.

## 7.4 Omitted variable bias

```math
\text{true: } y = \beta_0 + \beta_1 x + \beta_2 z + \varepsilon,\ \ \text{fit } y \sim x:\qquad
\mathbb E[\hat\beta_1] = \beta_1 + \beta_2\,\frac{\operatorname{Cov}(x,z)}{\operatorname{Var}(x)}
```

Measured in `demo_regression()` (true effect of x = 2, n = 2,000, seed 0): regressing on x alone gives **4.100** (the formula predicts 4.114); including z gives **1.992**. More data does not help — the short regression converges to the wrong number. This is confounding expressed as regression (Part 10).

## 7.5 Regression to the mean

Measured in `demo_regression()` (test reliability 0.5, 5,000 people, seed 0): the top 10% on test 1 average **+1.76** and **+0.83** on a retest; the bottom 10% go from **−1.74** to **−0.88** — halfway back, with no intervention. Part of an extreme score is luck that doesn't repeat. Any "we treated the worst performers and they improved" claim needs a control group.

## 7.6 Interpreting coefficients

```
linear             beta_j = change in y per unit of x_j, holding the OTHER variables fixed
log(y) ~ x         100 * beta ~ % change in y per unit x (for small beta)
log(y) ~ log(x)    beta = elasticity: % change in y per 1% change in x
logistic           beta = change in log-odds; exp(beta) = odds ratio
interaction x*z    effect of x depends on z; main effects are effects at z = 0
standardized x     beta per standard deviation — comparable across features
```
"Holding the others fixed" is a statement about the **model**, and becomes a causal statement only under the assumptions of Part 10. R² measures fit, not correctness: a high R² model can be causally wrong, and a low R² model can estimate an effect precisely.

---

# PART 8 — BAYESIAN INFERENCE

## 8.1 Priors, posteriors, conjugacy

```math
p(\theta\mid x) \propto p(x\mid\theta)\,p(\theta),\qquad
\text{Beta}(a,b)\ \text{prior} + k \text{ successes in } n \ \Rightarrow\ \text{Beta}(a+k,\ b+n-k)
```

A **conjugate** prior gives a posterior in the same family, so updating is just arithmetic. A Beta(a, b) prior behaves like `a + b − 2` pseudo-observations; the flat Beta(1, 1) gives posterior mean `(k+1)/(n+2)` (Laplace's rule of succession). Other conjugate pairs: Gamma–Poisson, Normal–Normal (known variance), Dirichlet–Multinomial.

**Priors matter when data are scarce and wash out as data grow.** Measured in `demo_bayesian()`, [`stats_from_scratch.py`](stats_from_scratch.py) (exact posteriors):

| Data | Posterior mean, flat Beta(1,1) prior | Posterior mean, Beta(20,20) prior |
|---|---|---|
| 3 / 10 | 0.333 | 0.460 |
| 30 / 100 | 0.304 | 0.357 |
| 300 / 1000 | 0.300 | 0.308 |

## 8.2 Credible vs confidence intervals

For 3 successes in 10 trials, the flat-prior **95% credible interval** is [0.109, 0.610] and the **95% Wilson confidence interval** is [0.108, 0.603] (from `demo_bayesian()`). Numerically close here; different in meaning:

- **Credible interval:** given the model *and the prior*, P(θ in interval | data) = 0.95. A direct probability statement about θ.
- **Confidence interval:** the procedure covers θ in 95% of repeated samples. No probability statement about θ from one dataset.

They can diverge substantially with informative priors, small samples, or parameters near a boundary.

## 8.3 Bayesian A/B comparison

Measured in `demo_bayesian()` (1,000/10,000 vs 1,080/10,000 conversions, flat priors, 40,000 posterior draws, seed 0): **P(B > A | data) = 0.967**; expected lift +0.0080 with 95% interval [−0.0006, +0.0166]. For the same data the frequentist two-sided p-value is **0.064** (one-sided 0.032).

"97% likely better" and "not significant at 5%" describe **the same data**; they answer different questions. With a flat prior and large samples, `P(B > A) ≈ 1 − one-sided p`. Bayesian framing is often easier to communicate and pairs naturally with expected-loss decision rules — but it is **not immune to peeking**: checking the posterior repeatedly and stopping when it crosses a threshold still changes the operating characteristics, and results depend on the prior.

---

# PART 9 — EXPERIMENT DESIGN

## 9.1 Why randomization works

Random assignment makes treatment independent of **every** pre-treatment variable — measured or not — so any systematic difference in outcomes is attributable to the treatment. Nothing else guarantees this. It does **not** guarantee balance in any single small experiment; it guarantees correct inference *on average* over randomizations, which is exactly what p-values and intervals account for.

## 9.2 Getting more precision from the same sample

```
blocking / stratified randomization   randomize within groups (country, device)
                                      so they are balanced by design
paired / crossover designs            each unit is its own control (5.4: p 0.48 -> 0.0019)
covariate adjustment                  regress the outcome on pre-treatment covariates
CUPED                                 adjust by a pre-experiment version of the metric
                                      (see ML System Design notes, §5.2)
```
Only adjust for **pre-treatment** variables — adjusting for something the treatment affects biases the estimate (Part 10.2).

## 9.3 Common design failures

```
peeking / optional stopping      inflates false positives (ML System Design §5.2: 6.3% -> 24.7%)
sample ratio mismatch            50/50 split observed as 48/52 -> assignment is broken
wrong randomization unit         request-level randomization of a user-level experience
interference / network effects   units affect each other (marketplaces, social) -> SUTVA fails
novelty and primacy effects      short-run behaviour != long-run behaviour
too many metrics                 multiple testing (Part 6)
underpowered design              "no significant difference" is weak evidence (Part 5.5)
survivorship / attrition         analysing only units that stayed breaks randomization
```

---

# PART 10 — CAUSAL INFERENCE

## 10.1 Potential outcomes

```math
\text{ATE} = \mathbb E\big[Y(1) - Y(0)\big],\qquad
\underbrace{\mathbb E[Y\mid T=1]-\mathbb E[Y\mid T=0]}_{\text{observed difference}} = \text{ATT} + \underbrace{\mathbb E[Y(0)\mid T=1]-\mathbb E[Y(0)\mid T=0]}_{\text{selection bias}}
```

Each unit has two potential outcomes but we observe only one — the **fundamental problem of causal inference**. The naive comparison equals the causal effect on the treated **plus selection bias**; randomization sets the selection-bias term to zero in expectation.

## 10.2 Confounders, mediators, colliders

```
confounder   Z -> T,  Z -> Y         adjust for Z (it opens a backdoor path)
mediator     T -> M -> Y             don't adjust if you want the TOTAL effect of T
collider     T -> C <- Y             don't adjust — conditioning on C CREATES a
                                     spurious T-Y association
```
**Backdoor criterion:** adjusting for a set of variables that blocks every backdoor path from T to Y (and contains no descendant of T) identifies the causal effect. **Collider bias** in practice: studying only hospitalized patients, only funded startups, or only users who churned can manufacture correlations that don't exist in the population (Berkson's paradox).

## 10.3 Simpson's paradox

Measured in `demo_causal()`, [`stats_from_scratch.py`](stats_from_scratch.py) — kidney-stone treatment data (Charig et al. 1986), success rates:

| | small stones | large stones | all patients |
|---|---|---|---|
| Treatment A | **93.1%** | **73.0%** | 78.0% |
| Treatment B | 86.7% | 68.8% | **82.6%** |

A is better within **each** stone size, B looks better **overall** — because A was given mostly to the harder, large-stone cases. Stone size is a confounder, so the stratified comparison is the right one here. **The numbers alone cannot tell you which comparison to trust**; that requires knowing the causal structure. If the stratifying variable were a *consequence* of treatment, aggregating would be correct instead.

## 10.4 Adjusting for measured confounders

Measured in `demo_causal()` — severity `z` drives both treatment and outcome, **true effect = +1.00** (n = 4,000, seed 0):

| Estimator | Estimate |
|---|---|
| naive difference in means | **−1.13** |
| regression `y ~ t + z` | +0.98 |
| stratification on z (10 bins) | +0.92 |
| inverse propensity weighting | +1.07 |

```math
\hat\tau_{\text{IPW}} = \frac{\sum_i T_i Y_i/\hat e(x_i)}{\sum_i T_i/\hat e(x_i)} - \frac{\sum_i (1-T_i) Y_i/(1-\hat e(x_i))}{\sum_i (1-T_i)/(1-\hat e(x_i))},\qquad \hat e(x) = \hat P(T=1\mid x)
```

The naive estimate has the **wrong sign**. All three adjustments recover roughly +1 — stratification least well, because 10 coarse bins leave residual confounding within each bin. They share the same **untestable assumptions**:

- **No unmeasured confounding** (conditional ignorability): every common cause of T and Y is measured.
- **Overlap (positivity):** every unit had a real chance of either treatment at its covariate values. Extreme propensities produce huge IPW weights (clipped to [0.01, 0.99] in the code).
- **Correct model** for the outcome (regression) or for the propensity (IPW). *Doubly robust* estimators combine both and stay consistent if either one is right.

## 10.5 Quasi-experiments

```math
\text{DiD} = \big(\bar Y^{\,T}_{\text{after}} - \bar Y^{\,T}_{\text{before}}\big) - \big(\bar Y^{\,C}_{\text{after}} - \bar Y^{\,C}_{\text{before}}\big)
```

Measured in `demo_causal()` — **true effect +2.00**, 40 simulations of 200 treated + 200 control units (mean and SD across simulations):

| Comparison | Estimate | Contaminated by |
|---|---|---|
| treated, before vs after | +3.49 (SD 0.06) | the common time trend |
| treated vs control, after | +5.02 (SD 0.13) | the fixed group gap |
| difference-in-differences | **+1.99** (SD 0.07) | — |

DiD removes both **if parallel trends hold**: without treatment, both groups would have moved together. Check pre-period trends; it can never be proven. (A single run with seed 0 gives 1.83 — the most extreme of the 40 — which is why results are reported across simulations.)

```
Instrumental variables  a variable Z that shifts T but affects Y ONLY through T
                        (exclusion restriction — untestable) and is as good as
                        random. Estimates the effect for 'compliers'. Weak
                        instruments give badly biased estimates.
Regression discontinuity  treatment assigned by a threshold on a running variable
                        (a test score, an age cutoff); compare units just above
                        and below. Valid locally at the cutoff, if units cannot
                        precisely manipulate their side of it.
Synthetic control       build a weighted combination of untreated units that
                        tracks the treated unit before the intervention.
```

---

# PART 11 — PROBABILITY PUZZLES

Each answer below is checked by simulation in `demo_puzzles()`, [`stats_from_scratch.py`](stats_from_scratch.py). Try each before expanding.

<details>
<summary><b>A family has two children and at least one is a boy. What is P(both are boys)? What if instead you meet one of the two children and it is a boy?</b></summary>

**1/3** in the first case: of the equally likely BB, BG, GB, only BB has two boys among the three families with at least one boy. **1/2** in the second: meeting a specific child who is a boy says nothing about the other child. The fact learned is the same ("a boy"); the protocol that produced it differs, and so does the answer. Simulated (200,000 families): 0.3344 and 0.5001.

</details>

<details>
<summary><b>Expected number of fair-coin flips until you see HH? Until HT?</b></summary>

**6 for HH, 4 for HT.** Solve with states: for HH, let `a` = expected flips from scratch and `b` = from "last flip was H": `a = 1 + ½b + ½a`, `b = 1 + ½·0 + ½a`, giving `a = 6`. For HT, a failed attempt (H then H) leaves you still one flip away, rather than resetting you, so it is faster. Simulated (40,000 runs): 5.967 and 3.995.

</details>

<details>
<summary><b>Break a stick at two uniformly random points. P(the three pieces form a triangle)?</b></summary>

**1/4.** A triangle needs every piece shorter than half the stick; in the unit square of the two break points, the region satisfying all three inequalities has area 1/4. Simulated (200,000 sticks): 0.2484.

</details>

<details>
<summary><b>Expected value of the maximum of two fair dice?</b></summary>

**161/36 ≈ 4.472.** Use the tail sum: `P(max ≥ k) = 1 − ((k−1)/6)²`, so `E[max] = Σ_{k=1}^{6} (1 − (k−1)²/36) = 6 − 55/36 = 161/36`. Simulated (200,000 rolls): 4.4662.

</details>

<details>
<summary><b>You see serial numbers from k = 5 randomly captured tanks; the largest is m. Estimate the total number N.</b></summary>

The sample maximum is biased low: `E[m] = k(N+1)/(k+1)`. The minimum-variance unbiased estimator adds back the average gap between observed serials: **`N̂ = m(1 + 1/k) − 1`**. With N = 300 (20,000 simulations): the sample max averages 250.2 (exact 250.8); the corrected estimator averages 299.3 (exact 300).

</details>

<details>
<summary><b>Sample k items uniformly from a stream of unknown length, in one pass and O(k) memory.</b></summary>

**Reservoir sampling (Algorithm R):** keep the first k items; for the i-th item (i > k), draw j uniformly from 1..i and replace reservoir slot j if j ≤ k. By induction every item ends in the reservoir with probability exactly k/n. Simulated (k = 5 of 20, 40,000 runs): every item's inclusion frequency lies in [0.2447, 0.2546], target 0.25.

</details>

<details>
<summary><b>You interview n candidates in random order and must accept or reject each on the spot. How do you maximize the chance of hiring the single best?</b></summary>

**Secretary problem:** reject the first ≈ n/e candidates, then accept the first who beats everyone seen so far. Success probability → **1/e ≈ 0.368** as n grows. Simulated (n = 100, 20,000 runs): 0.372.

</details>

<details>
<summary><b>Buses arrive as a Poisson process at 3 per hour. You've already waited 15 minutes. P(no bus in the next 20 minutes)?</b></summary>

**e⁻¹ ≈ 0.368** — the 15 minutes already waited are irrelevant, because exponential waiting times are **memoryless**: `P(no event in 20 min) = e^{−3 × 20/60}`. Simulated (100,000 draws): 0.3670.

</details>

<details>
<summary><b>A test is 99% sensitive and 95% specific for a condition with 1% prevalence. You test positive. How worried should you be?</b></summary>

**About 17%** likely to have it (Part 1.2): the false positives from the 99% healthy majority outnumber the true positives about 5 to 1. A second, independent positive test would raise it substantially — posterior becomes the new prior.

</details>

---

# PART 12 — RAPID-FIRE Q&A

<details>
<summary><b>Q: What is a p-value?</b></summary>

The probability, assuming the null hypothesis is true, of a test statistic at least as extreme as the one observed. It is not the probability that the null is true, and not the probability the result is due to chance.

</details>

<details>
<summary><b>Q: Interpret a 95% confidence interval.</b></summary>

The interval comes from a procedure that contains the true parameter in 95% of repeated samples. It is not a 95% probability statement about the parameter given this one dataset — that's a Bayesian credible interval, which requires a prior.

</details>

<details>
<summary><b>Q: When does the central limit theorem not help you?</b></summary>

When the variance is infinite (Cauchy, some power laws), when observations are strongly dependent, or when n is too small for skewed or heavy-tailed data. "n ≥ 30" is a folk rule: for exponential data the mean still has skewness 0.37 at n = 30.

</details>

<details>
<summary><b>Q: Why divide by n − 1 in the sample variance?</b></summary>

Deviations are measured from the sample mean, which is fitted to the data and so sits closer to the points than the true mean; dividing by n underestimates σ² by a factor (n−1)/n. Dividing by n − 1 makes the variance unbiased — but s is still biased low for σ.

</details>

<details>
<summary><b>Q: Student's t or Welch's t?</b></summary>

Welch by default. The pooled Student test assumes equal variances; when variances and group sizes differ it can badly inflate false positives (28.5% instead of 5% in the companion simulation). Welch loses almost nothing when variances happen to be equal.

</details>

<details>
<summary><b>Q: When would you use a non-parametric test?</b></summary>

With heavy tails or outliers, ordinal data, or small samples from clearly non-normal distributions. Mann-Whitney had 49% power against the t-test's 6% on shifted Cauchy data. Remember it tests whether one group tends to be larger, not strictly a difference in medians.

</details>

<details>
<summary><b>Q: What is statistical power, and what affects it?</b></summary>

The probability of rejecting H0 when a specified effect is real. It rises with effect size, sample size, and α, and falls with noise. A 0.5-sd effect needs about 64 per group for 80% power at α = 0.05.

</details>

<details>
<summary><b>Q: You ran 20 tests and one was significant at 0.05. What do you conclude?</b></summary>

Very little — with 20 independent true nulls you'd expect about one false positive by chance. Correct for multiple testing (Bonferroni for FWER, BH for FDR), or treat it as a hypothesis to confirm on fresh data.

</details>

<details>
<summary><b>Q: FWER vs FDR?</b></summary>

FWER is the probability of any false rejection (Bonferroni, for confirmatory claims). FDR is the expected share of rejections that are false (Benjamini-Hochberg, for screening many candidates). FDR control has far more power when many effects are real.

</details>

<details>
<summary><b>Q: What does heteroscedasticity do to a regression?</b></summary>

Coefficients stay unbiased, but classic standard errors are wrong — typically too small — so intervals under-cover (87% instead of 95% in the companion simulation). Use heteroscedasticity-robust (sandwich) standard errors, HC3 in small samples, or cluster-robust ones for grouped data.

</details>

<details>
<summary><b>Q: Is multicollinearity a problem?</b></summary>

For interpreting individual coefficients, yes — they become unstable with inflated standard errors (VIF). For prediction, not really: the combined effect and the fitted values stay well estimated.

</details>

<details>
<summary><b>Q: What is omitted variable bias?</b></summary>

Leaving out a variable that affects the outcome and is correlated with an included regressor shifts that regressor's coefficient by β_omitted × Cov(x, z)/Var(x). It does not shrink with more data.

</details>

<details>
<summary><b>Q: Explain Simpson's paradox.</b></summary>

A comparison can reverse when data are aggregated across groups that differ both in outcome rates and in group composition — e.g. a treatment better for both small and large kidney stones looks worse overall because it was given mostly to hard cases. Which comparison is right depends on the causal structure, not the data.

</details>

<details>
<summary><b>Q: Correlation vs causation — how do you get causal evidence?</b></summary>

Randomize if you can. Otherwise, adjust for measured confounders (regression, matching, IPW) under the no-unmeasured-confounding and overlap assumptions, or exploit a natural experiment (difference-in-differences, instrumental variables, regression discontinuity), each with its own untestable assumption.

</details>

<details>
<summary><b>Q: Should you control for every variable you have?</b></summary>

No. Control for confounders (common causes of treatment and outcome). Don't control for mediators if you want the total effect, and never for colliders — conditioning on a common consequence creates spurious associations.

</details>

<details>
<summary><b>Q: What is regression to the mean, and why does it matter?</b></summary>

Units selected for being extreme on a noisy measure tend to be less extreme when remeasured, with no intervention. Any before/after study on units selected for extreme values needs a control group.

</details>

<details>
<summary><b>Q: Frequentist vs Bayesian — which should you use?</b></summary>

Frequentist methods control long-run error rates without a prior; Bayesian methods give direct probability statements about parameters and handle small samples and prior knowledge naturally, at the cost of choosing a prior. With lots of data and weak priors they usually agree numerically.

</details>

<details>
<summary><b>Q: A/B test result: p = 0.06. The PM wants to ship. What do you say?</b></summary>

Report the estimated effect with its confidence interval rather than the binary verdict; check whether the interval includes effects too small to matter; confirm the test was run to its planned sample size without peeking; and weigh the cost of a wrong ship against the cost of waiting. In the companion data, p = 0.064 two-sided corresponds to a 97% posterior probability that B is better under a flat prior — the evidence is suggestive, not conclusive.

</details>

<details>
<summary><b>Q: What is the Wald interval's problem for proportions?</b></summary>

It plugs p̂ into the standard error, so near 0 or 1 or at small n it is far too narrow — at zero successes it has zero width. The companion simulation shows 64% coverage for a nominal 95% at p = 0.05, n = 20. Use the Wilson interval.

</details>

<details>
<summary><b>Q: How do you check whether a statistical method works for your data?</b></summary>

Simulate: generate data resembling yours where the truth is known, apply the method many times, and measure coverage, false-positive rate, bias, or power. That is how every claim in the companion code is verified.

</details>

---

# PART 13 — COVERAGE INDEX

| Concept | Implementation in [`stats_from_scratch.py`](stats_from_scratch.py) | Demo |
|---|---|---|
| Normal, t, χ², F, binomial CDFs & quantiles | `normal_cdf`, `t_cdf`, `t_ppf`, `chi2_cdf`, `f_cdf`, `binom_cdf` | 0 |
| Incomplete beta and gamma functions | `reg_inc_beta`, `reg_inc_gamma_lower` | 0 |
| Inverse-CDF, Box-Muller, Poisson, Cauchy sampling | `sample_*` | used throughout |
| Bayes and base rates | `bayes_test`, `simulate_bayes_test` | 1 |
| Monty Hall, birthday, coupon collector, gambler's ruin | `simulate_*`, `*_exact` | 1 |
| Linearity of expectation (fixed points) | `fixed_points_expectation` | 1 |
| Markov chain stationary distribution | `stationary_distribution`, `simulate_chain` | 1 |
| LLN, CLT, and the Cauchy counterexample | `clt_check`, `running_mean_path` | 2 |
| Bias of variance estimators; E[s] < σ | `variance_estimator_bias` | 3 |
| Shrinkage and MSE | `shrinkage_mse` | 3 |
| Bootstrap standard errors | `bootstrap_se` | 3 |
| z vs t intervals, coverage by simulation | `ci_mean_z`, `ci_mean_t`, `coverage` | 4 |
| Wald vs Wilson proportion intervals | `ci_prop_wald`, `ci_prop_wilson` | 4 |
| Bootstrap percentile interval | `ci_bootstrap_percentile` | 4 |
| One-sample, Welch, pooled, paired t-tests | `ttest_*` | 5 |
| Chi-square independence | `chi2_independence` | 5 |
| Mann-Whitney U | `mann_whitney_u` | 5 |
| Permutation test | `permutation_test` | 5 |
| Test calibration under H0 | `false_positive_rate` | 5 |
| Power: simulated and analytic, sample size | `power_sim`, `power_two_sample_analytic`, `n_per_group` | 5 |
| Bonferroni, Benjamini-Hochberg | `bonferroni`, `benjamini_hochberg`, `simulate_multiple_testing` | 6 |
| P(H0 \| significant) | `prob_h0_given_significant` | 6 |
| OLS with classic and HC1 robust SEs | `ols`, `slope_ci_coverage` | 7 |
| VIF | `vif` | 7 |
| Omitted variable bias | `omitted_variable_bias` | 7 |
| Regression to the mean | `regression_to_the_mean` | 7 |
| Beta-Binomial posterior, credible interval | `beta_posterior`, `beta_ppf` | 8 |
| Bayesian A/B | `bayesian_ab` | 8 |
| Simpson's paradox | `simpsons_paradox` | 9 |
| Regression adjustment, stratification, IPW | `estimate_effects` | 9 |
| Difference-in-differences | `difference_in_differences` | 9 |
| Puzzles: two children, HH/HT, stick, dice max, German tank, reservoir, secretary, Poisson | `puzzle_*`, `reservoir_sample` | 10 |

**Covered in these notes, not implemented** — be ready to explain: ANOVA and Welch ANOVA, McNemar and Fisher's exact test, BCa and bootstrap-t intervals, HC3 and cluster-robust SEs, Benjamini-Yekutieli, doubly robust estimation, instrumental variables, regression discontinuity, synthetic control, MCMC.

---

# PART 14 — 10-DAY PLAN

```
Day 1   Conditional probability, Bayes, base rates. Run demo 1. Work the puzzles.
Day 2   Expectation tricks (linearity, indicators, tail sums), Markov chains.
Day 3   Distributions and their relationships; LLN/CLT and their conditions.
        Run demo 2 and explain the Cauchy column.
Day 4   Estimation: bias, variance, MSE, MLE and its conditions, bootstrap. Demo 3.
Day 5   Confidence intervals: meaning, z vs t, Wald vs Wilson. Demo 4.
Day 6   Hypothesis tests: p-values and their misreadings, choosing a test,
        assumptions, power. Demo 5.
Day 7   Multiple testing, P(H0 | significant), p-hacking. Demo 6.
Day 8   Regression inference: assumptions table, robust SEs, VIF, OVB,
        regression to the mean, interpreting coefficients. Demo 7.
Day 9   Bayesian inference; experiment design. Demo 8.
Day 10  Causal inference: potential outcomes, DAG roles, Simpson, adjustment,
        DiD/IV/RDD. Demo 9. Then the rapid-fire questions, closed-book.
```

**Self-test — from memory, can you:**
1. Compute P(sick | positive) for a rare condition and explain why it's low?
2. State the conditions for the LLN and CLT, and give a distribution that breaks the CLT?
3. Explain why E[s] < σ even though s² is unbiased?
4. State the MLE's asymptotic properties and the conditions they need?
5. Interpret a 95% confidence interval correctly, and contrast it with a credible interval?
6. Explain why the Wald interval fails for rare events and name the fix?
7. Choose a test for five different data situations and state each test's assumptions?
8. Say when the pooled t-test fails and why Welch is the default?
9. Compute a sample size for a two-group comparison?
10. Explain FWER vs FDR, and compute P(H0 | significant) from prior and power?
11. Say what heteroscedasticity does and doesn't damage in OLS?
12. Classify a variable as confounder, mediator, or collider and say whether to adjust for it?

---

# REFERENCES

Primary sources for the claims above. Where these notes simplify, the source is authoritative.

**Textbooks**
- Wasserman. *All of Statistics*. Springer, 2004.
- Casella & Berger. *Statistical Inference*, 2nd ed. Duxbury, 2002.
- Blitzstein & Hwang. *Introduction to Probability*, 2nd ed. CRC Press, 2019 (free online).
- Efron & Tibshirani. *An Introduction to the Bootstrap*. Chapman & Hall, 1993.
- Gelman et al. *Bayesian Data Analysis*, 3rd ed. CRC Press, 2013.
- Angrist & Pischke. *Mostly Harmless Econometrics*. Princeton University Press, 2009.
- Hernán & Robins. *Causal Inference: What If*. Chapman & Hall/CRC, 2020 (free online).
- Pearl, Glymour & Jewell. *Causal Inference in Statistics: A Primer*. Wiley, 2016.
- Kohavi, Tang & Xu. *Trustworthy Online Controlled Experiments*. Cambridge University Press, 2020.

**Papers**
- Brown, Cai & DasGupta. Interval estimation for a binomial proportion. *Statistical Science*, 2001.
- Welch. The generalization of 'Student's' problem when several different population variances are involved. *Biometrika*, 1947.
- Benjamini & Hochberg. Controlling the false discovery rate. *JRSS B*, 1995. · Benjamini & Yekutieli. FDR control under dependency. *Annals of Statistics*, 2001.
- Ioannidis. Why most published research findings are false. *PLoS Medicine*, 2005.
- Wasserstein & Lazar. The ASA statement on p-values: context, process, and purpose. *The American Statistician*, 2016.
- Hoenig & Heisey. The abuse of power: the pervasive fallacy of power calculations for data analysis. *The American Statistician*, 2001.
- White. A heteroskedasticity-consistent covariance matrix estimator. *Econometrica*, 1980. · Long & Ervin. Using heteroscedasticity consistent standard errors in the linear regression model. *The American Statistician*, 2000.
- White. Maximum likelihood estimation of misspecified models. *Econometrica*, 1982.
- James & Stein. Estimation with quadratic loss. *Proc. Fourth Berkeley Symposium*, 1961.
- Charig et al. Comparison of treatment of renal calculi by open surgery, PCNL, and ESWL. *BMJ*, 1986.
- Rosenbaum & Rubin. The central role of the propensity score in observational studies. *Biometrika*, 1983.
- Vitter. Random sampling with a reservoir. *ACM TOMS*, 1985.

---

[ML](ML_Interview_Notes.md) · [DL](DL_Interview_Notes.md) · [RL](RL_Interview_Notes.md) · [NLP & LLMs](NLP_LLM_Interview_Notes.md) · [ML System Design](ML_System_Design_Notes.md) · **Probability & Stats**
