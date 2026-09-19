# MACHINE LEARNING — COMPLETE INTERVIEW NOTES

> Everything from the math foundations up to modern architectures.
> Math is written in ASCII inside code blocks so it renders anywhere.
> Companion code: `ml_from_scratch.py` and `ml_from_scratch_part2.py`.

---

## HOW TO USE THIS

| If you have... | Read |
|---|---|
| 1 week | Part 0 cheat sheet, Part 2 (bias-variance, regularization, metrics), Part 3 core 8 algorithms, Part 9 rapid-fire |
| 1 month | Everything except Parts 7–8 |
| Doing research/DL roles | Add Parts 6–7 heavily |

**The 5 questions behind almost every ML interview question:**
1. What is the model (functional form)?
2. What is the loss, and *why that loss*?
3. How is it optimized (closed form / gradient / greedy / EM)?
4. What assumptions does it make, and what breaks it?
5. What's the bias-variance / complexity trade-off?

---

# PART 0 — THE ONE-PAGE CHEAT SHEET

```
LOSSES
  MSE            L = (1/n) sum (y - yhat)^2                  regression, sensitive to outliers
  MAE            L = (1/n) sum |y - yhat|                    robust, non-smooth at 0
  Huber          quadratic if |e|<=d else linear             robust + smooth
  Binary CE      L = -[y log p + (1-y) log(1-p)]             binary classification
  Categorical CE L = -sum_k y_k log p_k                      multiclass
  Hinge          L = max(0, 1 - y*f(x))                      SVM, y in {-1,+1}
  Exponential    L = exp(-y*f(x))                            AdaBoost
  KL divergence  KL(P||Q) = sum P log(P/Q)                   distribution matching

GRADIENTS THAT SHOW UP EVERYWHERE
  Linear reg     dL/dw = (2/n) X^T (Xw - y)
  Logistic reg   dL/dw = (1/n) X^T (sigmoid(Xw) - y)
  Softmax + CE   dL/dz = p - y_onehot
  Sigmoid        s'(z) = s(z)(1 - s(z))
  Tanh           tanh'(z) = 1 - tanh(z)^2
  ReLU           1 if z>0 else 0

CLOSED FORMS
  OLS            w = (X^T X)^-1 X^T y
  Ridge          w = (X^T X + lam I)^-1 X^T y
  PCA            eigenvectors of (1/(n-1)) X_c^T X_c
  LDA            argmax_c  x^T S^-1 mu_c - 0.5 mu_c^T S^-1 mu_c + log P(c)

COMPLEXITY (n samples, d features, k neighbours/clusters, T trees, D depth)
  OLS normal eq       train O(n d^2 + d^3)     predict O(d)
  Logistic (GD)       O(n d) per epoch         predict O(d)
  KNN                 train O(1)               predict O(n d)
  Decision tree       O(n d log n)             predict O(D)
  Random forest       O(T n d log n)           predict O(T D)
  SVM (kernel)        O(n^2 d) to O(n^3)       predict O(n_sv d)
  K-means             O(i n k d)               predict O(k d)
  PCA                 O(n d^2 + d^3)           transform O(d k)
  Naive Bayes         O(n d)                   predict O(c d)

THE BIG TRADE-OFF
  E[(y - fhat)^2] = Bias^2 + Variance + Irreducible noise
  Underfit -> high bias  -> more features, more capacity, less regularization
  Overfit  -> high var   -> more data, fewer features, more regularization, ensembling
```

---

# PART 1 — MATHEMATICAL FOUNDATIONS

## 1.1 Linear Algebra

### Vectors and norms
```
dot product      u . v = sum_i u_i v_i = ||u|| ||v|| cos(theta)
L1 norm          ||v||_1 = sum |v_i|                  -> Lasso, sparsity
L2 norm          ||v||_2 = sqrt(sum v_i^2)            -> Ridge, Euclidean distance
Lp norm          ||v||_p = (sum |v_i|^p)^(1/p)
L-inf            max |v_i|
Cosine sim       u.v / (||u|| ||v||)                  -> text/embedding similarity
```
**Interview:** *Why does L1 give sparsity and L2 doesn't?* The L1 ball has corners **on the axes**. The optimum of a convex loss constrained to that ball very often lands on a corner, where some coordinates are exactly 0. The L2 ball is smooth and round — the contact point almost never has an exact zero. Analytically, the L1 subgradient is a constant ±λ, so it keeps pushing a small weight all the way to 0; the L2 gradient is 2λw, which shrinks proportionally and never reaches 0.

### Matrix essentials
```
Rank        # of linearly independent rows/cols. Full rank => invertible (square).
Trace       tr(A) = sum A_ii = sum of eigenvalues
Determinant product of eigenvalues; |det| = volume scaling factor; det=0 => singular
Inverse     A A^-1 = I. Exists iff det != 0.
Transpose   (AB)^T = B^T A^T ,  (A^-1)^T = (A^T)^-1
Symmetric   A = A^T  => real eigenvalues, orthogonal eigenvectors
Orthogonal  Q^T Q = I => Q^-1 = Q^T, preserves lengths & angles (a rotation/reflection)
```

### Eigenvalues and eigenvectors
```
A v = lambda v      v is a direction that A only STRETCHES, never rotates.
Characteristic eq:  det(A - lambda I) = 0
Spectral theorem (A symmetric):  A = V L V^T, V orthogonal, L diagonal
```
Used by: PCA (eigenvectors of covariance), spectral clustering (eigenvectors of the graph Laplacian), PageRank (dominant eigenvector), Hessian analysis (eigenvalues tell you curvature and whether a critical point is a min/max/saddle).

### SVD — the most important decomposition in ML
```
A = U S V^T       A is (n x d), any matrix, always exists
  U (n x n) orthogonal      left singular vectors  (eigenvectors of A A^T)
  S (n x d) diagonal >= 0   singular values        (sqrt of eigenvalues of A^T A)
  V (d x d) orthogonal      right singular vectors (eigenvectors of A^T A)

Rank-k truncation A_k = U_k S_k V_k^T is the BEST rank-k approximation in
Frobenius norm (Eckart-Young theorem).
```
Used by: PCA (equivalent formulation, numerically better than forming X^T X), LSA/topic models, recommender systems, pseudo-inverse `A+ = V S^-1 U^T`, image compression, matrix completion.

**Interview:** *PCA via covariance eigendecomposition vs via SVD?* Same answer, different conditioning. Forming `X^T X` squares the condition number, so tiny singular values lose precision. SVD on the centred `X` avoids that and also works when d >> n.

### Positive (semi-)definite matrices
```
A is PSD  <=>  x^T A x >= 0 for all x  <=>  all eigenvalues >= 0
A is PD   <=>  x^T A x >  0 for x != 0 <=>  all eigenvalues >  0
```
Covariance matrices, kernel/Gram matrices, and `X^T X` are always PSD. A Hessian that is PSD everywhere ⇒ the function is convex. Mercer's condition for a valid kernel = the Gram matrix must be PSD.

### Quadratic forms and least squares geometry
```
f(w) = ||Xw - y||^2 = w^T X^T X w - 2 y^T X w + y^T y
grad = 2 X^T (Xw - y) = 0   =>   X^T X w = X^T y     (normal equations)
```
Geometric reading: `Xw` lives in the column space of X. Minimizing the residual means the residual `(y - Xw)` must be **orthogonal to every column of X**: `X^T(y - Xw) = 0` — the same equation. The hat matrix `H = X(X^T X)^-1 X^T` is the orthogonal projector onto col(X).

### Matrix calculus (memorize these)
```
d/dx (a^T x)        = a
d/dx (x^T A x)      = (A + A^T) x   = 2Ax if A symmetric
d/dx ||x||^2        = 2x
d/dW (W x)          = x^T   (row-wise)
d/dx ||Ax - b||^2   = 2 A^T (Ax - b)
d/dA log det(A)     = (A^-1)^T
d/dA tr(AB)         = B^T
```

---

## 1.2 Calculus & Optimization Theory

### Derivatives you need
```
Gradient   grad f = [df/dx_1, ..., df/dx_d]^T          direction of steepest ASCENT
Jacobian   J_ij = df_i/dx_j                            for vector-valued f
Hessian    H_ij = d^2 f / (dx_i dx_j)                  curvature; symmetric
Chain rule dz/dx = dz/dy * dy/dx                       <- ALL of backprop is this
```

### Taylor expansion (the basis of every optimizer)
```
f(x + p) ~= f(x) + grad^T p + 0.5 p^T H p

Drop the H term  -> gradient descent:  p = -alpha * grad
Keep the H term  -> Newton's method:   p = -H^-1 grad     (quadratic convergence,
                                                            but O(d^3) per step)
Approximate H    -> BFGS / L-BFGS      (superlinear, O(d^2) or O(md) memory)
```

### Convexity
```
f convex  <=>  f(t x + (1-t) y) <= t f(x) + (1-t) f(y)  for t in [0,1]
          <=>  Hessian is PSD everywhere (if twice differentiable)
```
Convex ⇒ **any local minimum is global**. Convex ML losses: linear regression, ridge, lasso, logistic regression, SVM (primal & dual), softmax regression. Non-convex: neural networks, k-means (in the joint assignment+centroid variable), GMM/EM, matrix factorization, decision-tree structure search.

### Lagrange multipliers & KKT (needed for the SVM dual)
```
Problem:  min f(x)  s.t.  g_i(x) <= 0,  h_j(x) = 0
Lagrangian: L(x, a, b) = f(x) + sum_i a_i g_i(x) + sum_j b_j h_j(x)

KKT conditions at the optimum:
  1. Stationarity          grad_x L = 0
  2. Primal feasibility    g_i(x) <= 0,  h_j(x) = 0
  3. Dual feasibility      a_i >= 0
  4. Complementary slack   a_i g_i(x) = 0   <-- the important one
```
Condition 4 is exactly why SVMs have **support vectors**: for any point that is not on the margin, `g_i < 0`, which forces `a_i = 0`. Only the points sitting on the margin get a non-zero multiplier and enter the decision function.

### Gradient descent family
```
Batch GD    w <- w - alpha * (1/n) sum_i grad_i        stable, slow, O(n) per step
SGD         w <- w - alpha * grad_i                    noisy, fast, escapes saddles
Mini-batch  w <- w - alpha * (1/B) sum_{i in B} grad_i the practical default

Momentum        v <- beta v + grad ;  w <- w - alpha v
                (damps oscillation across steep, narrow valleys)
Nesterov        evaluate gradient at the LOOK-AHEAD point w - alpha*beta*v
AdaGrad         G += grad^2 ; w -= alpha*grad/sqrt(G+eps)   (LR decays, can stall)
RMSProp         G = rho G + (1-rho) grad^2 ; same update    (fixes AdaGrad stalling)
Adam            m = b1 m + (1-b1) g          (1st moment, momentum)
                v = b2 v + (1-b2) g^2        (2nd moment, per-parameter scaling)
                mhat = m/(1-b1^t), vhat = v/(1-b2^t)   <- bias correction, since
                                                          m,v start at 0
                w -= alpha * mhat / (sqrt(vhat) + eps)
```
**Interview:** *Why bias correction in Adam?* `m` and `v` are initialized at zero, so early estimates are biased toward zero; dividing by `(1 - beta^t)` rescales them to be unbiased. Without it the first steps are far too small.

**Interview:** *Why does SGD often generalize better than full-batch?* The gradient noise acts as an implicit regularizer, discourages sharp minima, and helps escape saddle points — which dominate over local minima in high-dimensional non-convex landscapes.

### Learning-rate schedules
```
Step decay      alpha_t = alpha_0 * gamma^floor(t/s)
Exponential     alpha_t = alpha_0 * e^(-kt)
Cosine          alpha_t = alpha_min + 0.5(alpha_0 - alpha_min)(1 + cos(pi t/T))
Warmup          linearly ramp up for the first few thousand steps (transformers)
1/t             alpha_t = alpha_0 / (1 + kt)          required for SGD convergence
```
Robbins-Monro conditions for SGD convergence: `sum alpha_t = inf` and `sum alpha_t^2 < inf`.

---

## 1.3 Probability

### Foundations
```
Bayes' rule       P(A|B) = P(B|A) P(A) / P(B)
                  posterior = likelihood * prior / evidence
Chain rule        P(A,B,C) = P(A) P(B|A) P(C|A,B)
Total probability P(B) = sum_i P(B|A_i) P(A_i)
Independence      P(A,B) = P(A)P(B)
Conditional ind.  P(A,B|C) = P(A|C) P(B|C)      <- the "naive" in Naive Bayes
```

### Expectation, variance, covariance
```
E[X]            = sum x p(x)  or  integral x f(x) dx
Linearity       E[aX + bY] = aE[X] + bE[Y]           ALWAYS true, even if dependent
Var(X)          = E[X^2] - E[X]^2
Var(aX+b)       = a^2 Var(X)
Var(X+Y)        = Var(X) + Var(Y) + 2Cov(X,Y)
Cov(X,Y)        = E[XY] - E[X]E[Y]
Corr(X,Y)       = Cov(X,Y) / (sd(X) sd(Y))   in [-1, 1]
Law of total exp    E[X] = E[E[X|Y]]
Law of total var    Var(X) = E[Var(X|Y)] + Var(E[X|Y])
```

### Distributions worth knowing cold
| Distribution | PMF/PDF | Mean | Var | Where it appears |
|---|---|---|---|---|
| Bernoulli(p) | `p^x (1-p)^(1-x)` | p | p(1-p) | binary labels, logistic regression |
| Binomial(n,p) | `C(n,x) p^x (1-p)^(n-x)` | np | np(1-p) | # successes, A/B tests |
| Categorical(π) | `prod pi_k^{x_k}` | π | — | softmax output |
| Poisson(λ) | `λ^x e^-λ / x!` | λ | λ | counts, rare events |
| Geometric(p) | `(1-p)^(x-1) p` | 1/p | (1-p)/p² | trials until success |
| Uniform(a,b) | `1/(b-a)` | (a+b)/2 | (b-a)²/12 | init, sampling |
| Normal(μ,σ²) | `1/sqrt(2πσ²) exp(-(x-μ)²/2σ²)` | μ | σ² | noise, CLT, everything |
| Exponential(λ) | `λe^(-λx)` | 1/λ | 1/λ² | waiting times, memoryless |
| Beta(α,β) | `x^(α-1)(1-x)^(β-1)/B(α,β)` | α/(α+β) | — | conjugate prior for p |
| Gamma(α,β) | — | α/β | α/β² | conjugate prior for precision |
| Multivariate Normal | `1/sqrt((2π)^d\|Σ\|) exp(-0.5 (x-μ)^T Σ^-1 (x-μ))` | μ | Σ | GMM, LDA, GP |

**Mahalanobis distance** `sqrt((x-μ)^T Σ^-1 (x-μ))` — the exponent of the MVN. It's Euclidean distance after whitening by the covariance, so it accounts for correlated, differently-scaled features. Used in anomaly detection and QDA.

### Conjugate priors (Bayesian shortcut)
```
Beta prior  + Binomial likelihood  -> Beta posterior     Beta(a + k, b + n - k)
Dirichlet   + Multinomial          -> Dirichlet
Normal      + Normal (known var)   -> Normal
Gamma       + Poisson              -> Gamma
```

### Limit theorems
```
LLN   sample mean -> true mean as n -> inf
CLT   (Xbar - mu)/(sigma/sqrt(n)) -> N(0,1) regardless of the original distribution
```
CLT is why standard errors are `sigma/sqrt(n)`, why bootstrap works, and why Gaussian noise assumptions are so often defensible.

### MLE vs MAP
```
MLE    theta* = argmax  P(D | theta)          = argmax sum log P(x_i | theta)
MAP    theta* = argmax  P(D | theta) P(theta)

Key equivalences (memorize — extremely common interview question):
  MLE with Gaussian noise           == minimizing MSE
  MLE with Bernoulli likelihood     == minimizing cross-entropy
  MAP with Gaussian prior N(0, t^2) == L2 / Ridge regularization, lambda = sigma^2/t^2
  MAP with Laplace prior            == L1 / Lasso regularization
```
Derivation of the Ridge equivalence:
```
log P(w|D) = log P(D|w) + log P(w)
           = -1/(2 sig^2) sum (y_i - w.x_i)^2  -  1/(2 t^2) ||w||^2  + const
maximize   == minimize  sum (y - w.x)^2 + (sig^2/t^2) ||w||^2
```

---

## 1.4 Statistics

### Estimators
```
Bias(that)     = E[that] - theta
Var(that)      = E[(that - E[that])^2]
MSE(that)      = Bias^2 + Var            <- the bias-variance identity for estimators
Consistent     that -> theta as n -> inf
Efficient      attains the Cramer-Rao lower bound
```
**Interview:** *Why divide by (n−1) for sample variance?* Bessel's correction. Using the sample mean instead of the true mean removes one degree of freedom and makes the naive estimator biased low by a factor `(n-1)/n`; dividing by `n-1` makes it unbiased.

### Hypothesis testing
```
H0 null, H1 alternative
p-value  = P(observing data at least this extreme | H0 true)
           NOT the probability that H0 is true.
Type I   = false positive, rate alpha (you reject a true H0)
Type II  = false negative, rate beta
Power    = 1 - beta  (probability of detecting a real effect)
```
Common tests: z-test (known σ, large n), t-test (unknown σ), paired t-test, chi-square (categorical independence / goodness of fit), ANOVA / F-test (3+ group means), Mann-Whitney U (non-parametric), Kolmogorov-Smirnov (distribution comparison / drift detection).

### Multiple testing
```
Bonferroni       alpha' = alpha / m            conservative, controls FWER
Benjamini-Hochberg  rank p-values, find largest k with p_(k) <= (k/m) alpha
                                                controls FDR, much more powerful
```

### A/B testing (extremely common in applied interviews)
```
Sample size per arm (two-proportion test):
  n = 2 * (z_{1-a/2} + z_{1-b})^2 * pbar(1-pbar) / delta^2

Checklist: randomization unit, novelty effect, network interference/SUTVA,
peeking (use sequential tests or fix n in advance), Simpson's paradox,
guardrail metrics, minimum detectable effect chosen BEFORE launch.
```

### Bootstrap
Resample n points **with replacement** B times, recompute the statistic, use the empirical distribution for confidence intervals. Needs no distributional assumption. It is exactly the resampling scheme behind bagging.

---

## 1.5 Information Theory

```
Entropy            H(X)   = -sum p(x) log p(x)          expected surprise, in bits
Joint entropy      H(X,Y) = -sum p(x,y) log p(x,y)
Conditional        H(Y|X) = H(X,Y) - H(X)
Mutual information I(X;Y) = H(Y) - H(Y|X) = KL(P(x,y) || P(x)P(y))
                          >= 0, and = 0 iff X and Y are independent
Cross-entropy      H(P,Q) = -sum p log q
KL divergence      KL(P||Q) = sum p log(p/q) = H(P,Q) - H(P)   >= 0, NOT symmetric
JS divergence      0.5 KL(P||M) + 0.5 KL(Q||M), M = (P+Q)/2    symmetric, bounded
```
**Why minimizing cross-entropy = maximum likelihood:** `H(P,Q) = H(P) + KL(P||Q)`. The true distribution's entropy `H(P)` is a constant w.r.t. your parameters, so minimizing cross-entropy is exactly minimizing `KL(P||Q)` — pushing your model distribution onto the data distribution.

**Interview:** *Forward vs reverse KL?* `KL(P||Q)` (forward, used in MLE) is *mean-seeking*: it is infinite wherever P has mass and Q doesn't, so Q must cover all of P. `KL(Q||P)` (reverse, used in variational inference) is *mode-seeking*: Q can safely ignore parts of P and will collapse onto one mode.

**Information gain** in a decision tree is exactly mutual information between the feature split and the label: `IG = H(parent) - H(parent | split)`.

---

# PART 2 — CORE ML THEORY

## 2.1 The Bias-Variance Decomposition (derive it, don't just quote it)

Assume `y = f(x) + eps`, with `E[eps] = 0`, `Var(eps) = sigma^2`. For a model `fhat` trained on a random dataset:

```
E[(y - fhat(x))^2]
  = E[(f + eps - fhat)^2]
  = E[(f - fhat)^2] + E[eps^2]                    (cross term vanishes, E[eps]=0)
  = (f - E[fhat])^2 + E[(fhat - E[fhat])^2] + sigma^2
  =      Bias^2     +        Variance       + Irreducible error
```

| | Bias | Variance |
|---|---|---|
| Meaning | error from wrong assumptions | sensitivity to the particular training set |
| Symptom | high train AND test error | low train, high test error |
| High in | linear models, shallow trees, high k in KNN, strong regularization | deep trees, k=1 KNN, high-degree polynomials, small n |
| Fix | more features/capacity, less regularization, boosting | more data, fewer features, more regularization, bagging |

**Interview:** *Does bagging reduce bias or variance?* Variance. For B estimators with variance σ² and pairwise correlation ρ, the average has variance `ρσ² + (1-ρ)σ²/B`. Averaging kills the second term but not the first — which is precisely why random forests *also* subsample features: to lower ρ. Boosting is the mirror image: it reduces **bias** by sequentially fitting the residual, at the cost of variance.

**Double descent:** the classical U-shaped test-error curve breaks in the over-parameterized regime. Past the interpolation threshold (params ≈ n), test error *falls again*. Modern DL lives on the right side of that curve.

---

## 2.2 Regularization

```
L2 / Ridge    J + lambda ||w||^2       shrinks all weights smoothly, handles
                                       collinearity, MAP w/ Gaussian prior
L1 / Lasso    J + lambda ||w||_1       drives weights exactly to 0 -> feature
                                       selection, MAP w/ Laplace prior
Elastic Net   J + l1 ||w||_1 + l2 ||w||^2
                                       Lasso picks one of a correlated group
                                       arbitrarily; the L2 term makes it keep
                                       the whole group ("grouping effect")
```

Other regularizers: early stopping (equivalent to L2 for linear models under GD), dropout, data augmentation, batch/layer norm (partly), weight decay, label smoothing, max-norm constraints, ensembling, adding noise to inputs (equivalent to Tikhonov regularization).

**Never regularize the intercept.** Doing so makes the model depend on where you happened to centre y.

**Interview:** *Why standardize before regularizing?* The penalty is applied uniformly to all coefficients, but coefficient magnitude depends on the feature's units. A feature measured in millimetres gets a coefficient 1000× larger than the same feature in metres, so it would absorb almost all the penalty.

---

## 2.3 Validation, Leakage, and Model Selection

```
Hold-out         simple, high variance on small data
k-fold CV        each point used for validation exactly once; k=5 or 10 standard
Stratified k-fold  preserves class ratios -> mandatory for imbalanced data
Leave-one-out    k=n; nearly unbiased but high variance and expensive
Nested CV        outer loop = performance estimate, inner loop = hyperparameter
                 tuning. Required whenever you tune, or your estimate is optimistic.
Time-series CV   ALWAYS forward-chaining: train [1..t], validate [t+1..t+h].
                 Random k-fold on time series leaks the future.
Group k-fold     keep all rows of the same entity (patient, user) in one fold
```

**Data leakage — the #1 real-world killer.** Sources: fitting the scaler/imputer/encoder on the full dataset before splitting; target encoding without out-of-fold computation; features computed using post-outcome information; duplicated rows straddling the split; time leakage. Rule: **fit every transformation on the training fold only, then apply it to validation/test.**

**Hyperparameter search:** grid (exhaustive, exponential in dims), random (better when only a few hyperparameters matter — Bergstra & Bengio), Bayesian optimization/TPE (models the objective with a surrogate, picks the next point by an acquisition function like Expected Improvement), Hyperband/successive halving (budget allocation).

---

## 2.4 Evaluation Metrics

### Confusion matrix
```
                 predicted +   predicted -
   actual +          TP            FN         (Type II error)
   actual -          FP            TN
                (Type I error)

Accuracy    (TP+TN)/total          useless under imbalance
Precision   TP/(TP+FP)             "when I say yes, am I right?"    -> spam filter
Recall/TPR  TP/(TP+FN)             "did I catch them all?"          -> cancer screen
Specificity TN/(TN+FP)
FPR         FP/(FP+TN) = 1 - specificity
F1          2PR/(P+R)              harmonic mean; punishes imbalance between P and R
F-beta      (1+b^2)PR/(b^2 P + R)  b>1 weights recall more
MCC         balanced even for very skewed classes; the safest single number
Cohen kappa agreement corrected for chance
```

### Curves
```
ROC curve   TPR vs FPR across thresholds.  AUC = P(score(random pos) > score(random neg))
            = the Mann-Whitney U statistic. Insensitive to class balance —
            which is a virtue AND a trap.
PR curve    Precision vs Recall. PREFER this under heavy imbalance, because FPR
            barely moves when TN is enormous, making ROC look deceptively good.
```

### Regression metrics
```
MSE / RMSE   penalizes large errors quadratically; RMSE is in the target's units
MAE          robust to outliers, non-differentiable at 0
MAPE         scale-free but explodes when y ~ 0, and is asymmetric
R^2          1 - SS_res/SS_tot ; fraction of variance explained; can go negative
Adjusted R^2 1 - (1-R^2)(n-1)/(n-d-1) ; penalizes useless features
```

### Ranking / recommendation
```
Precision@k, Recall@k, MAP@k
NDCG@k = DCG@k / IDCG@k ,   DCG = sum_i rel_i / log2(i+1)
MRR    = mean of 1/rank of the first relevant item
Hit rate, coverage, diversity, novelty
```

### Calibration
A model can rank perfectly (AUC 0.99) yet output meaningless probabilities. Check with a reliability diagram or Brier score `(1/n) sum (p - y)^2`. Fix with **Platt scaling** (fit a logistic on the scores) or **isotonic regression** (non-parametric, needs more data). Tree ensembles and SVMs are typically poorly calibrated; logistic regression is well-calibrated by construction.

---

## 2.5 Imbalanced Data

```
Data level     random undersampling, random oversampling, SMOTE (interpolate
               between a minority point and one of its k minority neighbours),
               ADASYN, Tomek links / edited nearest neighbours for cleaning
Algorithm      class_weight = n/(k * n_c), scale_pos_weight in boosting,
               focal loss FL = -(1-p)^gamma log(p)  (down-weights easy examples)
Threshold      keep the model, move the decision threshold to maximize F1 or
               expected business cost. Usually the cheapest and best fix.
Metric         use PR-AUC, F1, MCC — never raw accuracy
```
Important: apply SMOTE **inside the CV fold, on training data only**. Oversampling before splitting leaks synthetic copies of validation points into training.

---

## 2.6 Feature Engineering

```
Numeric      scaling (standard / min-max / robust via median+IQR), log or
             Box-Cox transforms for skew, binning, clipping outliers,
             polynomial and interaction terms, ratios and differences
Categorical  one-hot (low cardinality), ordinal (only if genuinely ordered),
             target/mean encoding with out-of-fold smoothing, frequency
             encoding, hashing trick (high cardinality), embeddings (NN)
Datetime     hour/day/month/weekend flags, cyclical encoding sin(2*pi*t/T)
             and cos(2*pi*t/T) so that 23:00 and 00:00 are adjacent
Text         bag-of-words, n-grams, TF-IDF, embeddings
Missing      MCAR/MAR/MNAR distinction; mean/median/mode, KNN imputation,
             iterative (MICE), or an explicit "is_missing" indicator column —
             missingness itself is often predictive. Trees can handle NaN natively.
Outliers     z-score, IQR (1.5x rule), isolation forest, Mahalanobis distance
```

**Curse of dimensionality:** as d grows, volume concentrates in the shell of the space, all pairwise distances converge to the same value, and the data needed for a fixed density grows exponentially. Consequences: KNN and kernel methods degrade, distance-based clustering becomes meaningless, and everything overfits. Remedies: feature selection, PCA/embeddings, regularization, models with strong inductive bias.

**Feature selection:**
```
Filter    correlation, chi-square, mutual information, ANOVA F     fast, model-free
Wrapper   forward/backward stepwise, recursive feature elimination  slow, model-aware
Embedded  Lasso, tree importance, permutation importance           free with training
```
Prefer **permutation importance** or SHAP over impurity-based tree importance: impurity importance is biased toward high-cardinality and continuous features.

---

## 2.7 Generative vs Discriminative

```
Discriminative  models P(y|x) directly   logistic regression, SVM, trees, NN
Generative      models P(x|y) and P(y)   Naive Bayes, GMM, LDA, HMM, VAE, diffusion
                then applies Bayes' rule
```
Ng & Jordan's result: the generative pair (Naive Bayes) has a **higher asymptotic error** but **converges faster** — it reaches its plateau at O(log d) samples versus O(d) for logistic regression. So Naive Bayes tends to win on small data, logistic regression on large data. Generative models can also sample new data and handle missing features naturally.

---

## 2.8 Interpretability

```
Global   linear coefficients, tree structure, feature importance, PDP, ALE
Local    LIME (fit a local sparse linear surrogate around one point),
         SHAP (Shapley values from cooperative game theory — the unique
         attribution satisfying efficiency, symmetry, dummy, additivity)
Counterfactual  "what minimal change to x flips the prediction?"
```
Shapley value for feature i: average marginal contribution over all orderings of features —
```
phi_i = sum_{S subset F\{i}}  |S|!(|F|-|S|-1)!/|F|!  [ f(S U {i}) - f(S) ]
```

---

# PART 3 — SUPERVISED ALGORITHMS

Each entry: **model → loss → optimization → assumptions → trade-offs → interview traps.**

## 3.1 Linear Regression

```
Model    yhat = w^T x + b
Loss     J = (1/n) ||Xw - y||^2
Solve    closed form  w = (X^T X)^-1 X^T y
         or GD        w <- w - alpha (2/n) X^T (Xw - y)
```
**Gauss-Markov assumptions** (for OLS to be the Best Linear Unbiased Estimator): linearity in parameters, no perfect multicollinearity, exogeneity `E[eps|X]=0`, homoscedasticity, uncorrelated errors. Normality of errors is *not* needed for BLUE — only for exact t/F inference.

**Diagnostics:** residual plot (should be structureless), Q-Q plot (normality), Durbin-Watson (autocorrelation), VIF (multicollinearity; `VIF_j = 1/(1-R_j^2)`, flag > 5–10), Cook's distance (influential points).

**Interview traps**
- *When is `X^T X` non-invertible?* Perfect collinearity, or d > n. Fixes: drop features, Ridge (always invertible), or pseudo-inverse via SVD.
- *GD vs normal equation?* Normal equation is exact and hyperparameter-free but O(d³); GD is O(nd) per epoch, scales to huge d, works online, but needs a learning rate and feature scaling.
- *Why not use accuracy-style metrics?* Continuous target — use RMSE/MAE/R².

## 3.2 Ridge, Lasso, Elastic Net
```
Ridge        min ||Xw-y||^2 + lam ||w||^2     w = (X^TX + lam I)^-1 X^T y
Lasso        min ||Xw-y||^2 + lam ||w||_1     no closed form -> coordinate descent
                 soft-threshold S(a,k) = sign(a) max(|a|-k, 0)
Elastic Net  min ||Xw-y||^2 + l1||w||_1 + l2||w||^2
```
Ridge has a beautiful shrinkage form in the SVD basis: each principal direction's coefficient is scaled by `d_j^2/(d_j^2 + lam)` — directions with **low variance get shrunk most**. That is exactly the desired behaviour, since low-variance directions are the least reliably estimated.

## 3.3 Logistic Regression
```
Model  p = sigmoid(w^T x + b),  logit(p) = log(p/(1-p)) = w^T x + b
Loss   J = -(1/n) sum [ y log p + (1-y) log(1-p) ]     (convex!)
Grad   dJ/dw = (1/n) X^T (p - y)
Hessian H = (1/n) X^T S X,  S = diag(p_i(1-p_i))  -> PSD -> convex -> IRLS/Newton works
```
**Interpretation:** `w_j` is the change in **log-odds** per unit of `x_j`; `exp(w_j)` is the **odds ratio**.

**Interview traps**
- *Why not MSE?* Non-convex through the sigmoid, and its gradient contains `sigma'(z)`, which vanishes when the model is confidently wrong — learning stalls exactly where you need it most. Cross-entropy's gradient `(p - y)` is proportional to the error itself.
- *Perfect separation?* The MLE diverges — weights blow up to ±∞. Fix with L2 regularization.
- *Is it a linear model?* Yes — linear in the log-odds; the decision boundary `w^T x + b = 0` is a hyperplane.
- *Multiclass?* Softmax (multinomial), or one-vs-rest / one-vs-one.

## 3.4 Naive Bayes
```
argmax_c  log P(c) + sum_j log P(x_j | c)
Gaussian NB     P(x_j|c) = N(mu_jc, var_jc)         continuous features
Multinomial NB  P(w|c) = (count + a)/(total + a*V)  counts, text
Bernoulli NB    P(x_j|c) = p^x (1-p)^(1-x)          binary presence/absence
```
Laplace smoothing (α) is mandatory: one unseen feature value would otherwise zero out the entire product.

**Interview:** *Why does it work despite the false independence assumption?* Classification only needs the **argmax** to be right, not the probabilities. Correlated features distort the magnitude of the posterior but often not its ordering. NB probabilities are therefore badly calibrated (extremely over-confident) even when accuracy is fine.

## 3.5 K-Nearest Neighbours
```
Classify: majority vote over the k nearest; Regress: mean of the k nearest
Distance: Euclidean, Manhattan, Minkowski, cosine, Hamming (categorical)
```
Non-parametric, zero training cost, O(nd) prediction. Requires scaling. Small k → high variance; large k → high bias; use odd k for binary. Speed-ups: KD-tree (good for d ≲ 20), ball tree, LSH / HNSW approximate search for high d.

## 3.6 Support Vector Machines
```
PRIMAL (soft margin)
  min_w  0.5||w||^2 + C sum_i xi_i    s.t. y_i(w.x_i+b) >= 1 - xi_i,  xi_i >= 0
Equivalent unconstrained form:
  min  (lam/2)||w||^2 + (1/n) sum max(0, 1 - y_i (w.x_i + b))     [hinge loss]

DUAL
  max_a  sum a_i - 0.5 sum_i sum_j a_i a_j y_i y_j K(x_i, x_j)
  s.t.   0 <= a_i <= C,   sum a_i y_i = 0
  f(x) = sum_i a_i y_i K(x_i, x) + b
```
The margin is `2/||w||`, so maximizing margin = minimizing `||w||²`. **C** trades margin width against violations: large C → narrow margin, low bias, high variance.

**Kernels** (must satisfy Mercer's condition — the Gram matrix must be PSD):
```
Linear      K = u.v
Polynomial  K = (gamma u.v + r)^d
RBF/Gauss   K = exp(-gamma ||u-v||^2)      infinite-dimensional feature space
Sigmoid     K = tanh(gamma u.v + r)
```
**Interview traps**
- *Why the dual?* It exposes only inner products, which is what enables the kernel trick, and the constraint set is simpler. It also scales with n rather than d, so it's the right choice when d ≫ n.
- *What is a support vector?* A point with `a_i > 0` — one on or inside the margin. By KKT complementary slackness, all others have `a_i = 0` and can be deleted without changing the model.
- *Effect of gamma in RBF?* Large gamma → narrow kernel → each point influences only its immediate neighbourhood → very wiggly boundary → overfitting.
- *SVM vs logistic regression?* Hinge loss is exactly zero past the margin, so only boundary points matter → sparse, robust to far-away points. Log loss is never zero, so all points contribute → gives probabilities. SVM needs Platt scaling for probabilities.

**SVR (regression):** ε-insensitive loss `max(0, |y - f(x)| - eps)` — a tube of width 2ε inside which errors cost nothing.

## 3.7 Decision Trees (CART)
```
Split criterion: maximize  Gain = I(parent) - sum_child (n_c/n) I(child)
  Gini     1 - sum p_c^2            (faster, no log)
  Entropy  -sum p_c log2 p_c        (slightly more balanced trees)
  MSE      variance of the node     (regression)
Leaf value: majority class / mean target
```
Gini and entropy rarely disagree; Gini is the default because it avoids logarithms.

**Pruning:** pre-pruning (max_depth, min_samples_leaf, min_impurity_decrease) or post-pruning via **cost-complexity**: `R_alpha(T) = R(T) + alpha |leaves(T)|`, sweep α and choose by CV.

**Pros:** no scaling needed, handles mixed types and non-linear interactions, interpretable, fast prediction, native missing-value handling (surrogate splits).
**Cons:** high variance (a small data change reshuffles the whole tree), greedy so no global optimum, axis-aligned splits struggle with diagonal boundaries, biased toward high-cardinality features, cannot extrapolate beyond the training range.

## 3.8 Ensembles

### Bagging & Random Forest
```
Bootstrap n samples with replacement (63.2% unique; P(never picked) = (1-1/n)^n -> 1/e)
Random Forest = bagging + m random features per split (sqrt(d) clf, d/3 reg)
Aggregate by vote/mean.  Ensemble variance: rho*sig^2 + (1-rho)sig^2/B
OOB error: free validation using the ~37% left-out samples per tree
```
**Extra Trees** go one step further: thresholds are chosen *at random* rather than optimally, lowering variance and training cost at the cost of a little bias.

### Boosting
```
AdaBoost      eps_t = weighted error;  alpha_t = 0.5 ln((1-eps)/eps)
              w_i <- w_i exp(-alpha_t y_i h_t(x_i)) / Z
              == stagewise minimization of exponential loss
Gradient Boost  fit each new tree to the NEGATIVE GRADIENT of the loss w.r.t.
              current predictions (pseudo-residuals). Squared loss -> residual;
              logistic loss -> (y - sigmoid(F)).
              F_m = F_{m-1} + nu * h_m ,  nu = learning rate (shrinkage)
XGBoost       second-order (Newton) boosting. With g_i, h_i the 1st/2nd
              derivatives, the optimal leaf weight and split gain are
                 w_j* = -G_j/(H_j + lam)
                 Gain = 0.5[ G_L^2/(H_L+lam) + G_R^2/(H_R+lam)
                            - (G_L+G_R)^2/(H_L+H_R+lam) ] - gamma
              plus L1/L2 on leaf weights, column subsampling, sparsity-aware
              default directions, and a weighted-quantile sketch for split finding.
LightGBM      leaf-wise (best-first) growth + histogram binning + GOSS + EFB. Fastest.
CatBoost      ordered boosting (fixes target leakage in target statistics) and
              ordered target encoding for categoricals.
```
**Interview:** *Bagging vs boosting?* Bagging trains independent (parallelizable) deep learners to reduce **variance**; boosting trains sequential shallow learners on the previous errors to reduce **bias**. Boosting is more accurate but more prone to overfitting noisy labels and cannot be parallelized across trees.

### Stacking
Train level-0 models, generate **out-of-fold** predictions, feed those as features to a level-1 meta-learner (usually a regularized linear model). Using in-fold predictions leaks and destroys the whole thing.

## 3.9 Discriminant Analysis
```
LDA   shared covariance -> quadratic terms cancel -> LINEAR boundary
      delta_c(x) = x^T S^-1 mu_c - 0.5 mu_c^T S^-1 mu_c + log pi_c
QDA   per-class covariance -> QUADRATIC boundary, many more parameters
```
LDA is also a supervised dimensionality reducer: maximize the Fisher criterion `J(w) = (w^T S_B w)/(w^T S_W w)`, solved by the top eigenvectors of `S_W^-1 S_B`. It yields at most **C−1** components.

## 3.10 Model Selection Cheat Table

| Situation | Reach for |
|---|---|
| Tabular data, need accuracy | Gradient boosting (XGB/LGBM) |
| Need interpretability/regulated domain | Logistic or linear regression, single tree |
| n small, d huge (text, genomics) | Linear SVM, logistic + L1, Naive Bayes |
| Non-linear, n moderate (< ~50k) | Kernel SVM, random forest |
| Images / audio / text sequences | CNN / Transformer |
| Streaming or online data | SGD-based linear models |
| Probability quality matters | Logistic regression, or calibrate |
| Many correlated features | Ridge / Elastic Net / PCA first |
| Very fast baseline needed | Naive Bayes, logistic regression |

---

# PART 4 — UNSUPERVISED LEARNING

## 4.1 Clustering

### K-Means
```
Objective  J = sum_k sum_{x in C_k} ||x - mu_k||^2      (inertia / WCSS)
E-step     assign each point to the nearest centroid
M-step     mu_k = mean of its members  (the mean minimizes squared distance)
Converges  J decreases monotonically, finite assignments -> guaranteed to stop,
           but only at a LOCAL optimum
K-means++  seed 1 uniformly, then each next centre w.p. proportional to D(x)^2
           -> O(log k) expected approximation guarantee
```
Assumes spherical, similarly sized, similarly dense clusters, and uses Euclidean distance, so scaling is mandatory. Choosing k: elbow of inertia, silhouette, gap statistic, BIC (via GMM), or domain knowledge.

**Variants:** k-medoids/PAM (uses actual data points as centres, robust to outliers, works with any distance), k-modes (categorical), mini-batch k-means (scales to millions), fuzzy c-means (soft memberships).

### DBSCAN
```
core point   >= min_samples neighbours within eps
border       within eps of a core point but not core
noise        neither                                   (labelled -1)
```
Finds arbitrarily shaped clusters, needs no k, labels outliers explicitly. Struggles when clusters have very different densities (one global eps) and in high dimensions. **HDBSCAN** fixes the varying-density problem by building a hierarchy over ε.

Choosing eps: plot the sorted distance to the k-th nearest neighbour and pick the knee.

### Hierarchical (agglomerative)
```
single   min pairwise dist  -> chaining, elongated clusters
complete max pairwise dist  -> compact, equal-diameter clusters
average  mean pairwise dist -> compromise
Ward     merge the pair that minimizes the increase in total within-cluster
         variance -> the most k-means-like, usually the best default
```
Output is a dendrogram, so you can cut at any k after the fact. Cost O(n³) naive, O(n² log n) with priority queues → doesn't scale past ~10⁴ points.

### Gaussian Mixture Models (EM)
```
p(x) = sum_k pi_k N(x | mu_k, Sigma_k)

E-step  gamma_ik = pi_k N(x_i|mu_k,Sig_k) / sum_j pi_j N(x_i|mu_j,Sig_j)
M-step  N_k = sum_i gamma_ik
        pi_k = N_k/n
        mu_k = (1/N_k) sum_i gamma_ik x_i
        Sig_k = (1/N_k) sum_i gamma_ik (x_i-mu_k)(x_i-mu_k)^T
```
EM maximizes a lower bound (the ELBO) on the log-likelihood, so the likelihood never decreases — but it converges only to a local optimum. **GMM is soft k-means**: fix `Sigma = sigma^2 I` and let `sigma -> 0` and the responsibilities become hard 0/1 assignments, recovering k-means exactly. Ellipsoidal covariances let GMM handle elongated, correlated clusters that k-means cannot. Choose k with BIC/AIC. Add a ridge to Σ or a component can collapse onto a single point and send the likelihood to infinity.

### Spectral clustering
```
1. Build a similarity graph W (e.g. RBF affinity or kNN graph)
2. Degree matrix D; Laplacian L = D - W  (or normalized L_sym = I - D^-1/2 W D^-1/2)
3. Take the eigenvectors of the k smallest eigenvalues -> embedding
4. Run k-means in that embedding
```
The multiplicity of eigenvalue 0 equals the number of connected components. Excellent for non-convex clusters; cost is dominated by the eigendecomposition, O(n³).

### Mean Shift
Iteratively move each point toward the weighted mean of the points inside a bandwidth-h kernel window — a gradient ascent on the kernel density estimate. Modes of the density become clusters. No k required; bandwidth is the sole (critical) hyperparameter.

### Evaluating clusters
```
Internal   silhouette s = (b-a)/max(a,b),  Davies-Bouldin (lower better),
           Calinski-Harabasz (higher better), inertia
External   Adjusted Rand Index (chance-corrected), Normalized Mutual
           Information, Fowlkes-Mallows, purity     (need ground-truth labels)
```

---

## 4.2 Dimensionality Reduction

### PCA
```
1. Centre X (and standardize if units differ)
2. C = (1/(n-1)) X_c^T X_c
3. Eigendecompose C = V L V^T  (or SVD of X_c: X_c = U S V^T, lambda_i = s_i^2/(n-1))
4. Project Z = X_c V_k
Explained variance ratio = lambda_k / sum(lambda)
```
Two equivalent characterizations: **maximum variance** projection, and **minimum squared reconstruction error** (Eckart-Young). Components are orthogonal and ordered. It is a *linear*, *unsupervised* method — it can happily discard the direction that carries the label signal if that direction has low variance.

**Kernel PCA** applies the kernel trick to the centred Gram matrix, allowing non-linear manifolds.

### Other reducers
```
LDA        supervised; maximizes between/within class scatter; <= C-1 comps
ICA        finds statistically INDEPENDENT (not merely uncorrelated) sources
           by maximizing non-Gaussianity (kurtosis/negentropy) -> blind source
           separation, cocktail-party problem
t-SNE      converts distances to probabilities; minimizes KL(P||Q) where Q uses a
           heavy-tailed Student-t in 2-D to fight the "crowding problem".
           VISUALIZATION ONLY: non-parametric (no transform for new points),
           non-convex, cluster SIZES and inter-cluster DISTANCES are meaningless,
           perplexity ~ effective # of neighbours (5-50).
UMAP       fuzzy-topological graph + cross-entropy layout. Faster than t-SNE,
           preserves more global structure, and can transform new points.
Autoencoder  non-linear compression via a bottleneck; a linear autoencoder with
           MSE loss spans exactly the same subspace as PCA.
Matrix Fact. NMF (non-negative -> parts-based, interpretable topics), SVD/LSA
Random proj. Johnson-Lindenstrauss: n points can be embedded in O(log n / eps^2)
           dimensions with distances preserved to within (1 +/- eps)
```

**Interview:** *PCA vs t-SNE?* PCA is linear, deterministic, invertible, preserves global variance structure, and gives a reusable projection matrix. t-SNE is non-linear, stochastic, preserves local neighbourhoods only, and exists to make pictures — never feed t-SNE output into a downstream model as features.

---

## 4.3 Anomaly Detection

```
Statistical      z-score, IQR, Mahalanobis distance, Grubbs test
Density          KDE, GMM likelihood threshold, Local Outlier Factor (LOF —
                 compares a point's local density to its neighbours')
Distance         kNN distance to the k-th neighbour
Isolation Forest random splits; anomalies are ISOLATED in fewer splits.
                 score s(x) = 2^(-E[h(x)]/c(n)),  c(n) ~ 2 ln(n-1) - 2(n-1)/n
                 Score near 1 = anomaly, near 0.5 = normal. O(n log n), no distances.
One-class SVM    learn a boundary enclosing the normal data (nu = outlier fraction)
Reconstruction   autoencoder/PCA error; anomalies reconstruct badly
```
Evaluate with PR-AUC or precision@k — accuracy is meaningless when anomalies are 0.1% of the data.

---

## 4.4 Association Rule Mining

```
support(A)      = fraction of transactions containing A
confidence(A->B)= support(A U B) / support(A)         = P(B|A)
lift(A->B)      = confidence / support(B)             > 1 means positively associated
conviction      = (1 - support(B)) / (1 - confidence)
```
**Apriori** uses the downward-closure property (every subset of a frequent itemset is frequent) to prune candidates level by level. **FP-Growth** builds a compressed prefix tree and avoids candidate generation — much faster.

---

# PART 5 — DEEP LEARNING

## 5.1 The Feedforward Network and Backpropagation

```
FORWARD          z^l = W^l a^(l-1) + b^l ,   a^l = f(z^l) ,   a^0 = x

BACKWARD (chain rule, right to left — each layer reuses the next layer's delta)
  output layer   delta^L = dL/da^L  *  f'(z^L)
  hidden layer   delta^l = (W^(l+1))^T delta^(l+1)  *  f'(z^l)
  gradients      dL/dW^l = delta^l (a^(l-1))^T
                 dL/db^l = delta^l
```
**The magic cancellation:** with a softmax output and cross-entropy loss, the softmax Jacobian and the log's derivative cancel exactly, leaving `delta^L = a^L - y`. The same happens for sigmoid+BCE and for linear+MSE. This is why those three pairings are the standard ones — clean gradients, no vanishing at the output.

Backprop is **reverse-mode automatic differentiation**: cost is O(1) forward passes for the gradient w.r.t. *all* parameters, which is why it beats finite differences (O(#params) evaluations) by many orders of magnitude.

## 5.2 Activations
| Function | Formula | Derivative | Notes |
|---|---|---|---|
| Sigmoid | `1/(1+e^-z)` | `s(1-s)` | saturates, max grad 0.25 → vanishing; output layer only |
| Tanh | `(e^z-e^-z)/(e^z+e^-z)` | `1-tanh²` | zero-centred, still saturates |
| ReLU | `max(0,z)` | `1 if z>0 else 0` | fast, sparse, no saturation for z>0; dying-ReLU risk |
| Leaky ReLU | `max(αz,z)` | `1 or α` | fixes dying ReLU |
| ELU / SELU | `z or α(e^z−1)` | — | smooth, self-normalizing (SELU) |
| GELU | `z·Φ(z)` | — | transformer default |
| Swish/SiLU | `z·sigmoid(z)` | — | smooth, often beats ReLU |
| Softmax | `e^{z_k}/Σe^{z_j}` | see above | output layer, multiclass |

**Why non-linearity at all?** A stack of linear layers collapses to a single linear layer: `W₂(W₁x) = (W₂W₁)x`. Depth would buy nothing.

## 5.3 Vanishing / Exploding Gradients
Gradients multiply across layers: `prod_l W^l f'(z^l)`. If the factors are < 1 the product decays exponentially (vanishing); if > 1 it blows up (exploding).

```
Fixes:  ReLU-family activations, careful init, residual/skip connections,
        batch/layer normalization, gradient clipping (explosion),
        LSTM/GRU gates (sequences), shorter effective depth
```

**Initialization:**
```
Xavier/Glorot   Var(W) = 2/(fan_in + fan_out)     tanh, sigmoid
He              Var(W) = 2/fan_in                 ReLU (accounts for half the
                                                   units being dead)
Zero init is FATAL: all units in a layer compute the same thing and receive the
same gradient forever (symmetry is never broken).
```

## 5.4 Normalization
```
BatchNorm  normalize over the BATCH, per feature:  xhat = (x-mu_B)/sqrt(var_B+eps)
           then scale & shift: y = gamma*xhat + beta (learnable, so the network
           can undo it). Train uses batch stats; inference uses running averages.
           Benefits: faster training, higher LRs, mild regularization.
           Weak with small batches; awkward for RNNs.
LayerNorm  normalize over the FEATURES of each sample -> batch-size independent
           -> the standard in transformers and RNNs.
GroupNorm  compromise for small-batch vision. InstanceNorm: per-sample per-channel.
```

## 5.5 Regularization in DL
```
Dropout     drop units w.p. p at train time; scale by 1/(1-p) (inverted dropout)
            so inference needs no change. Approximates an ensemble of 2^n
            subnetworks. Do NOT use dropout and batchnorm carelessly together.
Weight decay  L2 on weights (AdamW decouples it from the adaptive scaling)
Early stopping  monitor validation loss with patience
Data augmentation  flips, crops, colour jitter, mixup, cutmix, SpecAugment
Label smoothing  y = (1-e)*onehot + e/K   -> prevents over-confidence
```

## 5.6 CNNs
```
Conv output size    O = floor((W - K + 2P)/S) + 1
Params per conv     (K_h * K_w * C_in + 1) * C_out       <- independent of image size
Receptive field     grows with depth, kernel size, stride, dilation
Pooling             max/avg; downsample, add small translation invariance
1x1 conv            channel mixing / dimensionality reduction, cheap
```
Key inductive biases: **local connectivity** (nearby pixels are related), **parameter sharing** (a feature detector is useful everywhere → translation equivariance), **hierarchy** (edges → textures → parts → objects).

Architectures: LeNet → AlexNet (ReLU, dropout) → VGG (3×3 stacks) → **ResNet** (residual `y = F(x) + x`, so the gradient has an identity path and 100+ layers become trainable) → Inception (multi-scale) → DenseNet → EfficientNet (compound scaling) → ConvNeXt.

## 5.7 RNNs, LSTM, GRU
```
Vanilla RNN   h_t = tanh(W_h h_(t-1) + W_x x_t + b)
              BPTT multiplies W_h repeatedly -> vanishing/exploding gradients

LSTM          f_t = sig(W_f [h_(t-1), x_t] + b_f)     forget gate
              i_t = sig(W_i [...])                     input gate
              o_t = sig(W_o [...])                     output gate
              g_t = tanh(W_g [...])                    candidate
              c_t = f_t * c_(t-1) + i_t * g_t          CELL STATE (additive!)
              h_t = o_t * tanh(c_t)

GRU           z_t = sig(...)  update gate
              r_t = sig(...)  reset gate
              h_t = (1-z_t)*h_(t-1) + z_t*htilde       fewer params, often as good
```
The cell state's **additive** update is the whole point: gradients flow through `c_t` without repeated matrix multiplication, so long-range dependencies survive.

## 5.8 Attention and Transformers
```
Scaled dot-product attention
    Attention(Q,K,V) = softmax( Q K^T / sqrt(d_k) ) V

Why divide by sqrt(d_k)? For random Q,K with unit-variance entries, the dot
product has variance d_k. Without the scaling, large d_k pushes softmax into
a saturated regime where gradients vanish.

Multi-head:  concat over h heads, each with its own W_Q, W_K, W_V, then W_O.
             Lets different heads attend to different relation types.

Block = MultiHeadAttn -> Add&Norm -> FeedForward(4x) -> Add&Norm
Positional encoding needed because attention is permutation-invariant:
        sinusoidal, learned, or rotary (RoPE)
Masked self-attention in the decoder prevents attending to future tokens.

Complexity: O(n^2 d) in sequence length -> the motivation for FlashAttention,
sparse/linear attention, sliding windows, and state-space models (Mamba).
```
**Interview:** *Why did transformers replace RNNs?* Full parallelism across the sequence during training (RNNs are inherently sequential), constant path length between any two tokens (so no vanishing-gradient over distance), and better scaling behaviour.

## 5.9 Generative Models
```
Autoencoder   encoder -> bottleneck -> decoder, reconstruction loss
VAE           ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
              reparameterization trick z = mu + sigma * eps  makes it differentiable
GAN           min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]
              failure modes: mode collapse, non-convergence -> WGAN, spectral norm
Diffusion     forward: gradually add Gaussian noise; reverse: learn to denoise.
              Train by predicting the noise eps added at a random timestep.
Autoregressive  p(x) = prod p(x_t | x_<t)   -> GPT-style LMs
```

## 5.10 Practical Training Checklist
```
1. Overfit a single batch first — if you can't, there's a bug.
2. Scale/normalize inputs. Check the label distribution.
3. Start with a known-good architecture and LR; use an LR range test.
4. Monitor train vs val curves:
     both high      -> underfitting: more capacity / train longer / lower reg
     train low, val high -> overfitting: more data / augmentation / more reg
     loss = NaN     -> LR too high, exploding gradients, log(0), bad init
     val loss noisy -> batch too small, LR too high
5. Use mixed precision, gradient accumulation for large effective batches.
6. Fix all seeds; log everything.
```

---

# PART 6 — SPECIALIZED DOMAINS

## 6.1 Time Series

```
Components   trend + seasonality + cyclic + residual
             additive Y = T + S + e   |   multiplicative Y = T * S * e

STATIONARITY (mean, variance, autocovariance constant over time) is required by
ARIMA. Test with ADF (H0: unit root, non-stationary) or KPSS (H0: stationary).
Achieve it by differencing, log transform, or de-trending.

AR(p)     y_t = c + sum_i phi_i y_(t-i) + e_t
MA(q)     y_t = c + e_t + sum_j th_j e_(t-j)
ARMA(p,q) both;  ARIMA(p,d,q) = ARMA on the d-times differenced series
SARIMA(p,d,q)(P,D,Q)_m adds seasonal terms
Choose p,q from PACF (cuts off at p for AR) and ACF (cuts off at q for MA), or AIC/BIC.

Exponential smoothing  yhat_(t+1) = a*y_t + (1-a)*yhat_t
Holt-Winters           adds trend + seasonality
Modern practice        gradient boosting on lag/rolling/calendar features,
                       Prophet, N-BEATS, DeepAR, temporal fusion transformers
```
**Validation must be forward-chaining.** Random k-fold leaks the future into the past. Features must use only information available at prediction time (watch out for rolling windows that peek ahead).

Metrics: MAE, RMSE, MAPE, sMAPE, MASE (scaled against a naive forecast — the only one comparable across series).

## 6.2 Recommender Systems

```
Content-based       item features + user profile; no cold-start for new items,
                    but no serendipity, and needs good features
Collaborative filtering
  user-based        similarity between users (cosine / Pearson)
  item-based        similarity between items — more stable, industry standard
Matrix factorization  R ~= P Q^T,  minimize
      sum_(u,i observed) (r_ui - p_u.q_i - b_u - b_i - mu)^2 + lam(||p||^2+||q||^2)
      solved by SGD or ALS (ALS parallelizes and handles implicit feedback)
Implicit feedback   treat clicks as positive with confidence weights (BPR, WARP)
Two-tower / neural CF  learned user and item embeddings + MLP or dot product
Hybrid + re-rank    candidate generation (fast, recall-oriented) then ranking
                    (slow, precision-oriented) — the standard production shape
```
**Cold start:** new user → onboarding questions, demographics, popularity fallback; new item → content features; use a bandit for exploration.
**Metrics:** offline Precision@k, Recall@k, NDCG, MAP, coverage, diversity; online CTR, dwell time, retention via A/B test.

## 6.3 NLP Fundamentals

```
Preprocessing  tokenize, lowercase, stopwords, stemming (crude, fast) vs
               lemmatization (dictionary-based, correct), subword tokenization
               (BPE, WordPiece, SentencePiece) — handles OOV by construction

BoW            counts; loses order
TF-IDF         tf(t,d) * log(N / df(t))       down-weights ubiquitous words
n-grams        capture local order at the cost of a combinatorial vocabulary

Word2Vec       skip-gram maximizes  sum log P(context | centre)
               with negative sampling: log sig(v_c.v_w) + sum_neg log sig(-v_n.v_w)
               (avoids the O(V) softmax denominator)
               CBOW predicts the centre from the context; faster, worse on rare words
GloVe          factorizes the global co-occurrence matrix
FastText       subword n-grams -> handles morphology and OOV
Contextual     ELMo, BERT (masked LM + next-sentence), GPT (causal LM)
               -> the same word gets different vectors in different contexts
```
Evaluation: perplexity `exp(-1/N sum log p)` for LMs, BLEU (translation, precision of n-grams), ROUGE (summarization, recall), F1/EM (QA).

## 6.4 Reinforcement Learning

```
MDP  (S, A, P, R, gamma)
Return          G_t = sum_k gamma^k r_(t+k+1)
State value     V^pi(s)   = E[G_t | s_t = s]
Action value    Q^pi(s,a) = E[G_t | s_t=s, a_t=a]

Bellman expectation   V(s) = sum_a pi(a|s) sum_s' P(s'|s,a)[R + gamma V(s')]
Bellman optimality    Q*(s,a) = E[ R + gamma max_a' Q*(s',a') ]

Q-learning (off-policy, model-free, tabular)
    Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]
SARSA (on-policy)
    Q(s,a) <- Q(s,a) + alpha [ r + gamma Q(s',a') - Q(s,a) ]

Exploration vs exploitation: epsilon-greedy, UCB (a + sqrt(2 ln t / n_a)),
Thompson sampling (sample from the posterior — usually the best in practice)

Policy gradient (REINFORCE)   grad J = E[ grad log pi(a|s) * G_t ]
Actor-Critic / A2C            subtract a learned baseline V(s) -> lower variance
PPO                           clipped surrogate objective, the workhorse
DQN                           NN Q-function + replay buffer + target network
RLHF                          reward model from human preferences, then PPO/DPO
```

## 6.5 Bayesian Methods

```
Bayesian linear regression
    prior  w ~ N(0, tau^2 I)  ,  likelihood y ~ N(Xw, sigma^2 I)
    posterior  w | D ~ N(mu_n, S_n)
        S_n  = (X^T X/sigma^2 + I/tau^2)^-1
        mu_n = S_n X^T y / sigma^2          <- the Ridge solution is the MAP/mean
    Predictive variance grows away from the training data -> honest uncertainty

Gaussian Process
    f ~ GP(m(x), k(x,x'))
    posterior mean  k_*^T (K + sigma^2 I)^-1 y
    posterior var   k_** - k_*^T (K + sigma^2 I)^-1 k_*
    Non-parametric, exact uncertainty, O(n^3) -> great for small-n Bayesian
    optimization, hopeless for big data without sparse approximations.

Variational inference   maximize the ELBO = E_q[log p(x,z)] - KL(q||p)
MCMC                    Metropolis-Hastings, Gibbs, HMC/NUTS — asymptotically
                        exact but slow
```

## 6.6 Causal Inference (increasingly asked)

```
Correlation != causation. Confounder Z causes both X and Y.
Potential outcomes:  ATE = E[Y(1) - Y(0)]
Randomization solves confounding by design; observational data needs assumptions:
  ignorability/unconfoundedness, positivity, SUTVA
Methods: propensity score matching/weighting (IPW), difference-in-differences,
         instrumental variables, regression discontinuity, double ML, uplift models
Simpson's paradox: an association can reverse when you condition on a subgroup.
```

## 6.7 MLOps & System Design

```
Pipeline   ingest -> validate -> feature store -> train -> evaluate -> register
           -> deploy (shadow / canary / A-B) -> monitor -> retrain
Serving    batch (offline scoring) vs online (low latency) vs streaming
Drift      data drift  P(x) changes      -> KS test, PSI, KL on feature dists
           concept drift P(y|x) changes  -> monitor live metrics, delayed labels
Training-serving skew  same feature code path for both; a feature store helps
Latency    quantization, distillation, pruning, ONNX/TensorRT, caching, ANN index
Testing    unit tests on transforms, data schema validation, model behavioural
           tests (invariance, directional expectation), backtesting
```
**Design-question framework:** clarify the objective and constraints → define the ML problem and label → data sources and leakage risks → offline metric AND online metric → baseline first → features → model → serving architecture and latency budget → monitoring and retraining cadence → failure modes and fallbacks.

---

# PART 7 — RAPID-FIRE Q&A

**Q: Bias-variance in one line?** Bias is error from wrong assumptions; variance is error from sensitivity to the training sample; total error is Bias² + Var + noise.

**Q: Why does L1 produce sparsity?** Constant-magnitude subgradient ±λ pushes small weights exactly to zero; the L1 ball's corners lie on the axes.

**Q: Generative vs discriminative?** Generative models P(x|y)P(y) and can sample; discriminative models P(y|x) directly and usually classifies better with enough data.

**Q: Why cross-entropy over MSE for classification?** Convex in the parameters, gradient `(p−y)` doesn't vanish when confidently wrong, and it's the MLE under a Bernoulli/categorical likelihood.

**Q: How do you handle missing data?** Determine the mechanism (MCAR/MAR/MNAR), then choose: drop (if trivial and MCAR), impute (median/KNN/MICE), add a missingness indicator, or use a model that handles NaN natively (LightGBM/XGBoost).

**Q: Curse of dimensionality?** Data sparsity grows exponentially with d, distances concentrate, so neighbourhood-based methods break down.

**Q: Precision vs recall — pick one?** Depends on error cost: recall for cancer screening or fraud detection (missing one is catastrophic), precision for spam filtering or ad targeting (false alarms are costly).

**Q: ROC-AUC vs PR-AUC?** Under heavy imbalance FPR barely moves because TN is huge, so ROC looks optimistic — use PR-AUC.

**Q: Why is random forest better than a single tree?** Averaging decorrelated trees kills variance without raising bias; feature subsampling is what decorrelates them.

**Q: XGBoost vs Random Forest?** RF: independent deep trees, parallel, variance reduction, hard to overfit, few hyperparameters. XGB: sequential shallow trees fitting residuals, bias reduction, usually more accurate, more tuning, more overfit risk on noisy labels.

**Q: How do you pick k in k-means?** Elbow of inertia, silhouette score, gap statistic, BIC via GMM, or downstream/business utility.

**Q: What if features are correlated?** Linear model coefficients become unstable (high variance, VIF). Use Ridge/Elastic Net, drop features, or PCA. Trees are unaffected in accuracy but their feature importances get split arbitrarily among the correlated group.

**Q: Explain the kernel trick.** Replace inner products with `K(u,v)`, which equals an inner product in a higher-dimensional space you never construct — you get non-linear boundaries at the cost of a linear-model optimization.

**Q: Why standardize?** Distance-based methods (KNN, k-means, SVM, PCA) are unit-sensitive; gradient descent converges faster on isotropic loss surfaces; regularization penalties assume comparable scales.

**Q: Parametric vs non-parametric?** Parametric fixes the number of parameters ahead of time (linear/logistic regression); non-parametric grows capacity with data (KNN, trees, GP, kernel SVM).

**Q: What is the vanishing gradient problem?** Repeated multiplication by factors < 1 across layers shrinks gradients exponentially; fix with ReLU, residual connections, normalization, LSTM gates, and good init.

**Q: What does the learning curve tell you?** If train and validation both plateau high → underfitting (get a bigger model). If there's a large persistent gap → overfitting (get more data or regularize). If validation is still falling → train longer.

**Q: Your model has 99% accuracy — are you happy?** Not until I see the class balance. On a 99:1 problem the constant predictor achieves that. Show me PR-AUC, recall on the minority class, and the confusion matrix.

**Q: Train and test both perform well but production fails. Why?** Distribution shift, training-serving skew, leakage in offline features, a temporally invalid split, or feedback loops from the model's own actions.

**Q: How would you detect leakage?** Suspiciously high offline metrics, a single feature dominating importance, performance that collapses on a strictly-later time split, and auditing each feature for "could I have known this before the label existed?"

**Q: EM in two sentences?** E-step computes soft assignments (posterior over latent variables) given the current parameters; M-step maximizes the expected complete-data log-likelihood to update parameters. It monotonically increases a lower bound on the likelihood, converging to a local optimum.

**Q: Why does dropout work?** It prevents co-adaptation of units and approximates averaging an exponential ensemble of subnetworks, acting as a strong regularizer.

**Q: How do you choose the threshold for a classifier?** Not 0.5 by default — sweep it to optimize the metric that reflects the actual cost asymmetry (F1, expected cost, or precision at a fixed recall target).

**Q: What is the difference between a parameter and a hyperparameter?** Parameters are learned from data by the optimizer (weights); hyperparameters are set before training and chosen by validation (learning rate, depth, λ).

---

# PART 8 — 2-WEEK STUDY PLAN

```
Day 1-2   Part 1: linear algebra, probability, MLE/MAP. Derive OLS both ways.
Day 3     Part 2: bias-variance derivation, regularization, all metrics.
Day 4     Linear + logistic regression: derive gradients from scratch on paper.
Day 5     Trees, gini/entropy, pruning. Implement a tree without looking.
Day 6     Ensembles: bagging math, AdaBoost alpha derivation, XGBoost gain formula.
Day 7     SVM: primal -> Lagrangian -> dual -> KKT -> support vectors -> kernels.
Day 8     Naive Bayes, KNN, LDA/QDA. Generative vs discriminative.
Day 9     Unsupervised: k-means convergence, GMM/EM, PCA via both routes.
Day 10    Deep learning: derive backprop for a 2-layer net by hand.
Day 11    CNN/RNN/Transformer math. Attention scaling. Normalization.
Day 12    Your specialization: NLP / time series / recsys / RL.
Day 13    ML system design + MLOps. Practice 3 full design questions out loud.
Day 14    Rapid-fire Q&A, run the companion code, explain every algorithm aloud
          in under 2 minutes each.
```

**Final advice:** in an interview, always structure the answer as *intuition → math → assumptions → trade-offs*. Saying "I'd start with a simple baseline and a metric that matches the business cost" before jumping to a model is worth more than naming the fanciest architecture.

---
*Companion code: `ml_from_scratch.py` (core algorithms) and `ml_from_scratch_part2.py` (everything else), both pure Python with the derivations in the docstrings.*

# APPENDIX — ALGORITHM COVERAGE INDEX

Every algorithm in these notes, and where its working implementation lives.
`P1` = `ml_from_scratch.py`, `P2` = `ml_from_scratch_part2.py`.

## Supervised — regression
| Algorithm | Code | Key idea to state in an interview |
|---|---|---|
| Linear Regression (GD) | P1 | convex MSE, gradient `(2/n)Xᵀ(Xw−y)` |
| Linear Regression (normal eq) | P1 | `(XᵀX)⁻¹Xᵀy`, O(d³), exact |
| Ridge | P1 | L2, MAP w/ Gaussian prior, always invertible |
| Lasso | P1 | L1 via coordinate descent + soft-thresholding → sparsity |
| Elastic Net | P2 | L1+L2, grouping effect on correlated features |
| Polynomial regression | P1 | linear model on expanded features |
| Huber regression | P2 | quadratic near 0, linear tails → robust |
| SVR | P2 | ε-insensitive tube, sparse support vectors |
| Bayesian linear regression | P2 | full posterior; Ridge = its MAP |
| Gaussian Process | P2 | prior over functions, exact uncertainty, O(n³) |
| KNN regression | P1 | distance-weighted local average |
| Regression tree | P1 | variance-reduction splits |
| Random forest regressor | P1 | bagging + feature subsampling |
| Gradient boosting regressor | P1 | fit trees to negative gradients |
| Bagging / stacking | P2 | OOF meta-features are mandatory |

## Supervised — classification
| Algorithm | Code | Key idea |
|---|---|---|
| Logistic regression | P1 | sigmoid + cross-entropy, gradient `(p−y)` |
| Softmax regression | P1 | multiclass CE, `δ = p − y_onehot` |
| Perceptron | P1 | mistake-driven, converges iff separable |
| Linear SVM | P1 | hinge loss + margin maximization |
| Kernel SVM (SMO) | P1 | dual + kernel trick, KKT → support vectors |
| Gaussian Naive Bayes | P1 | conditional independence, log-sum of densities |
| Multinomial NB | P1 | counts + Laplace smoothing, text |
| Bernoulli NB | P2 | penalizes absence too |
| LDA | P1 | shared Σ → linear boundary |
| QDA | P2 | per-class Σ → quadratic boundary, more variance |
| KNN classifier | P1 | majority vote, curse of dimensionality |
| Decision tree (CART) | P1 | gini/entropy gain, greedy, pruning |
| Decision stump | P1 | weighted weak learner for boosting |
| Random forest | P1 | ρσ² + (1−ρ)σ²/B variance formula |
| Extra Trees | P2 | random thresholds → lower ρ, faster |
| AdaBoost | P1 | `α = ½ln((1−ε)/ε)`, exponential loss |
| Gradient boosting classifier | P1 | residual = `y − sigmoid(F)` |
| XGBoost-style (Newton) | P2 | `w* = −G/(H+λ)`, exact split-gain formula |
| Isolation Forest | P2 | anomalies isolate in fewer splits |
| MLP + backprop | P1 | chain rule, He/Xavier init, momentum SGD |

## Unsupervised
| Algorithm | Code | Key idea |
|---|---|---|
| K-Means (+ k-means++) | P1 | Lloyd's alternation, monotone decrease of inertia |
| DBSCAN | P1 | density reachability, explicit noise label |
| Agglomerative | P1 | linkage criteria, dendrogram |
| Mean Shift | P2 | gradient ascent on a KDE, no k |
| Spectral clustering | P2 | eigenvectors of the graph Laplacian |
| GMM / EM | P1 | soft responsibilities, ELBO never decreases |
| PCA | P1 | eigenvectors of covariance = max variance |
| Truncated SVD / LSA | P2 | power iteration + deflation, no centring |
| t-SNE | P2 | KL(P‖Q), Student-t fixes crowding |
| Apriori | P2 | downward closure pruning, report lift |

## Sequence, recommendation, RL
| Algorithm | Code | Key idea |
|---|---|---|
| HMM: forward-backward | P2 | O(TN²) DP instead of O(Nᵀ) |
| HMM: Viterbi | P2 | same recursion with max + backpointers |
| HMM: Baum-Welch | P2 | EM; local optima → use restarts |
| Kalman filter | P2 | gain K balances model vs sensor trust |
| ARIMA(p,d,0) | P2 | differencing buys stationarity |
| Matrix factorization | P2 | biases + latent factors, observed entries only |
| Item-based CF | P2 | item similarities are more stable than user ones |
| Q-Learning | P2 | off-policy TD, `max_a'` in the target |
| Bandits (ε/UCB/Thompson) | P2 | optimism vs probability matching |
| Word2Vec skip-gram | P2 | negative sampling replaces the O(V) softmax |
| TF-IDF | P2 | inverse-document-frequency weighting |

## Utilities
| Tool | Code | Note |
|---|---|---|
| train/test split, stratified | P1 | preserve class ratios |
| k-fold CV, cross_val_score | P1 | nested CV when tuning |
| Standardizer, MinMax | P1 | fit on train only |
| Polynomial features, one-hot | P1 | |
| All metrics + ROC-AUC + silhouette | P1 | AUC via Mann-Whitney |
| Optimizers (SGD→Adam) | P2 | bias correction explained |
| SMOTE | P2 | inside the fold, never before the split |
| Mutual-information selection | P2 | captures non-monotonic dependence |
| Platt scaling | P2 | calibrate SVM/tree scores |
| Permutation importance | P2 | beats impurity importance |

## Covered in the notes, deliberately not implemented
Implementing these from scratch teaches little that the above doesn't, but you should be able to **explain** them:

- **CNNs, RNN/LSTM/GRU, Transformers** — equations in Part 5. Interview focus: parameter counts, receptive fields, the gating equations, the `√d_k` scaling, why attention beat recurrence.
- **VAE / GAN / diffusion** — objectives in Part 5.9.
- **LightGBM / CatBoost** — the deltas from XGBoost (leaf-wise growth, histograms, GOSS/EFB, ordered boosting) are in Part 3.8.
- **UMAP, HDBSCAN, LOF, one-class SVM** — Parts 4.2 and 4.3.
- **FP-Growth, ALS, BPR** — Parts 4.4 and 6.2.
- **MCMC, variational inference** — Part 6.5.
- **Causal inference methods** — Part 6.6.

---

## FINAL SELF-TEST

Close the notes. Can you, from memory:

1. Derive the OLS solution two ways (calculus and geometric projection)?
2. Derive the logistic regression gradient and say why MSE fails there?
3. Derive the bias-variance decomposition?
4. Go from the SVM primal to the dual and explain what KKT gives you?
5. Write AdaBoost's α and explain why it has that form?
6. Write the XGBoost split-gain formula and say where it comes from?
7. Write the E-step and M-step for a GMM, and reduce it to k-means?
8. Derive backprop for a 2-layer network and show `δ_L = a − y` for softmax+CE?
9. Explain why L1 gives sparsity, three different ways?
10. State the MLE ↔ loss-function equivalences (Gaussian→MSE, Bernoulli→CE, Gaussian prior→L2, Laplace prior→L1)?
11. Explain the scaled dot-product attention formula including the `√d_k`?
12. Name five ways data leakage sneaks into a pipeline?

If yes to all twelve, you're prepared.
