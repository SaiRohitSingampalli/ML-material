"""
================================================================================
 ML FROM SCRATCH — PART 2: everything the first file didn't cover
================================================================================
 Pure Python (math + random only). Companion to `ml_from_scratch.py`, whose
 helpers this file re-implements standalone so it can run on its own.

 CONTENTS
 --------
 A. Optimizers ............ SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam
 B. Regression ............ ElasticNet, BayesianLinearRegression, SVR,
                            GaussianProcessRegressor, HuberRegressor
 C. Classification ........ BernoulliNB, QDA, XGBoostStyleBooster (Newton
                            boosting with the exact gain formula)
 D. Ensembles ............. ExtraTrees, StackingEnsemble, IsolationForest
 E. Clustering ............ MeanShift, SpectralClustering
 F. Manifold .............. TSNE (simplified), TruncatedSVD
 G. Recommenders .......... MatrixFactorization (SGD), ItemBasedCF
 H. Sequences ............. HiddenMarkovModel (forward-backward, Viterbi,
                            Baum-Welch), KalmanFilter, ARModel
 I. NLP ................... TfidfVectorizer, Word2VecSkipGram
 J. RL .................... QLearning, MultiArmedBandit (eps-greedy/UCB/Thompson)
 K. Mining ................ Apriori association rules
 L. Utilities ............. SMOTE, mutual-information feature selection,
                            PlattScaling, permutation importance
================================================================================
"""

import math
import random

random.seed(7)
EPS = 1e-12


# ------------------------------------------------------------------ helpers
def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def transpose(A):
    return [list(c) for c in zip(*A)]


def zeros(n, m=None):
    return [0.0] * n if m is None else [[0.0] * m for _ in range(n)]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def matmul(A, B):
    p = len(B[0])
    out = [[0.0] * p for _ in A]
    for i, row in enumerate(A):
        for k, a in enumerate(row):
            if a:
                for j in range(p):
                    out[i][j] += a * B[k][j]
    return out


def matvec(A, v):
    return [dot(r, v) for r in A]


def solve(A, b):
    n = len(A)
    M = [r[:] + [b[i]] for i, r in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < EPS:
            raise ValueError("singular")
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def inverse(A):
    n = len(A)
    I = identity(n)
    M = [A[i][:] + I[i] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < EPS:
            raise ValueError("singular")
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c]:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [r[n:] for r in M]


def sigmoid(z):
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def mean(v):
    return sum(v) / len(v)


def r2_score(y, p):
    m = mean(y)
    return 1 - sum((a - b) ** 2 for a, b in zip(y, p)) / max(
        sum((a - m) ** 2 for a in y), EPS)


def accuracy(y, p):
    return sum(1 for a, b in zip(y, p) if a == b) / len(y)


def rmse(y, p):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y, p)) / len(y))


# =============================================================================
# A. OPTIMIZERS
# =============================================================================
class Optimizer:
    """
    THE OPTIMIZER FAMILY, side by side.
    -----------------------------------
    All of them answer one question: given grad g at step t, how far and in
    what direction do we move?

      SGD        w -= lr * g
      Momentum   v = beta*v + g ;            w -= lr*v
                 (accumulates a velocity -> damps zig-zag in narrow valleys)
      Nesterov   evaluate g at the look-ahead point w - lr*beta*v (a correction
                 term, so it "sees" where momentum is about to take it)
      AdaGrad    G += g^2 ;                  w -= lr*g/(sqrt(G)+eps)
                 (per-parameter LR; great for sparse features, but G only grows,
                  so the effective LR decays to 0 and learning stalls)
      RMSProp    G = rho*G + (1-rho)*g^2 ;   w -= lr*g/(sqrt(G)+eps)
                 (exponential average instead of a sum -> never stalls)
      Adam       m = b1*m + (1-b1)*g         1st moment (direction / momentum)
                 v = b2*v + (1-b2)*g^2       2nd moment (per-parameter scale)
                 mhat = m/(1-b1^t) ; vhat = v/(1-b2^t)   <- BIAS CORRECTION,
                 needed because m and v are initialized at 0 and are therefore
                 biased toward 0 for the first steps.
                 w -= lr * mhat/(sqrt(vhat)+eps)
    """

    def __init__(self, kind="adam", lr=0.05, beta=0.9, beta2=0.999, rho=0.9,
                 eps=1e-8):
        self.kind, self.lr, self.beta, self.beta2 = kind, lr, beta, beta2
        self.rho, self.eps = rho, eps
        self.t = 0
        self.m = self.v = None

    def step(self, w, g):
        if self.m is None:
            self.m, self.v = [0.0] * len(w), [0.0] * len(w)
        self.t += 1
        k = self.kind
        out = []
        for i in range(len(w)):
            gi = g[i]
            if k == "sgd":
                out.append(w[i] - self.lr * gi)
            elif k == "momentum":
                self.m[i] = self.beta * self.m[i] + gi
                out.append(w[i] - self.lr * self.m[i])
            elif k == "nesterov":
                prev = self.m[i]
                self.m[i] = self.beta * self.m[i] + gi
                out.append(w[i] - self.lr * (gi + self.beta *
                                             (self.m[i] - prev) + self.beta * self.m[i]))
            elif k == "adagrad":
                self.v[i] += gi * gi
                out.append(w[i] - self.lr * gi / (math.sqrt(self.v[i]) + self.eps))
            elif k == "rmsprop":
                self.v[i] = self.rho * self.v[i] + (1 - self.rho) * gi * gi
                out.append(w[i] - self.lr * gi / (math.sqrt(self.v[i]) + self.eps))
            else:  # adam
                self.m[i] = self.beta * self.m[i] + (1 - self.beta) * gi
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * gi * gi
                mhat = self.m[i] / (1 - self.beta ** self.t)
                vhat = self.v[i] / (1 - self.beta2 ** self.t)
                out.append(w[i] - self.lr * mhat / (math.sqrt(vhat) + self.eps))
        return out


# =============================================================================
# B. REGRESSION
# =============================================================================
class ElasticNet:
    """
    ELASTIC NET = L1 + L2, by coordinate descent
    --------------------------------------------
        J = (1/2n)||y - Xw||^2 + a*l1*||w||_1 + 0.5*a*(1-l1)*||w||^2

    Coordinate update with soft-thresholding S(x,k)=sign(x)max(|x|-k,0):

        w_j = S(rho_j/n, a*l1) / (z_j/n + a*(1-l1))

    WHY BOTH PENALTIES: pure Lasso, faced with a group of highly correlated
    features, arbitrarily keeps ONE and zeroes the rest — unstable across
    resamples. The L2 term restores the "grouping effect" so correlated
    features are shrunk together and kept together. Also, Lasso can select at
    most n features when d > n; Elastic Net can select more.
    """

    def __init__(self, alpha=0.1, l1_ratio=0.5, epochs=300, tol=1e-8):
        self.alpha, self.l1_ratio, self.epochs, self.tol = alpha, l1_ratio, epochs, tol

    @staticmethod
    def _soft(x, k):
        return x - k if x > k else (x + k if x < -k else 0.0)

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.w, self.b = [0.0] * d, mean(y)
        z = [sum(r[j] ** 2 for r in X) for j in range(d)]
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1 - self.l1_ratio)
        for _ in range(self.epochs):
            pred = self.predict(X)
            biggest = 0.0
            for j in range(d):
                if z[j] < EPS:
                    continue
                rho = sum(X[i][j] * (y[i] - pred[i] + self.w[j] * X[i][j])
                          for i in range(n))
                new = self._soft(rho / n, l1) / (z[j] / n + l2)
                if new != self.w[j]:
                    delta = new - self.w[j]
                    for i in range(n):
                        pred[i] += delta * X[i][j]
                    biggest = max(biggest, abs(delta))
                    self.w[j] = new
            if biggest < self.tol:
                break
        return self

    def predict(self, X):
        return [dot(self.w, r) + self.b for r in X]


class HuberRegressor:
    """
    HUBER LOSS — robust regression
    ------------------------------
        L(e) = 0.5 e^2            if |e| <= delta
             = delta(|e| - 0.5d)  otherwise
        dL/de = e                 if |e| <= delta
              = delta * sign(e)   otherwise

    Quadratic near zero (efficient, smooth) but LINEAR in the tails, so a wild
    outlier contributes a bounded gradient instead of dominating the fit the
    way it would under MSE.
    """

    def __init__(self, delta=1.35, lr=0.01, epochs=2000):
        self.delta, self.lr, self.epochs = delta, lr, epochs

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.w, self.b = [0.0] * d, 0.0
        for _ in range(self.epochs):
            gw, gb = [0.0] * d, 0.0
            for xi, ti in zip(X, y):
                e = dot(self.w, xi) + self.b - ti
                g = e if abs(e) <= self.delta else self.delta * (1 if e > 0 else -1)
                for j in range(d):
                    gw[j] += g * xi[j]
                gb += g
            self.w = [w - self.lr * g / n for w, g in zip(self.w, gw)]
            self.b -= self.lr * gb / n
        return self

    def predict(self, X):
        return [dot(self.w, r) + self.b for r in X]


class BayesianLinearRegression:
    """
    BAYESIAN LINEAR REGRESSION
    --------------------------
    prior       w ~ N(0, tau^2 I)
    likelihood  y ~ N(Xw, sigma^2 I)
    posterior   w | D ~ N(mu_n, S_n)   (Gaussian is conjugate to itself)

        S_n  = ( X^T X / sigma^2 + I / tau^2 )^-1
        mu_n = S_n X^T y / sigma^2

    Note mu_n is EXACTLY the Ridge solution with lambda = sigma^2/tau^2 — Ridge
    is the MAP estimate under a Gaussian prior. The payoff over Ridge is the
    predictive distribution:

        p(y* | x*) = N( mu_n . x* ,  sigma^2 + x*^T S_n x* )

    The variance term GROWS as x* moves away from the training data, so the
    model reports honest uncertainty instead of confidently extrapolating.
    """

    def __init__(self, sigma2=1.0, tau2=10.0):
        self.sigma2, self.tau2 = sigma2, tau2

    def fit(self, X, y):
        Xb = [[1.0] + r for r in X]
        d = len(Xb[0])
        Xt = transpose(Xb)
        A = matmul(Xt, Xb)
        for i in range(d):
            for j in range(d):
                A[i][j] /= self.sigma2
            A[i][i] += 1.0 / self.tau2
        self.S = inverse(A)
        Xty = matvec(Xt, y)
        self.mu = matvec(self.S, [v / self.sigma2 for v in Xty])
        return self

    def predict(self, X, return_std=False):
        Xb = [[1.0] + r for r in X]
        m = [dot(self.mu, r) for r in Xb]
        if not return_std:
            return m
        sd = [math.sqrt(self.sigma2 + dot(r, matvec(self.S, r))) for r in Xb]
        return m, sd


class SVR:
    """
    SUPPORT VECTOR REGRESSION (linear, epsilon-insensitive)
    -------------------------------------------------------
        L_eps(e) = max(0, |e| - eps)
        J = 0.5||w||^2 + C * (1/n) sum L_eps(y_i - f(x_i))

    Errors smaller than eps cost NOTHING — the model only cares about points
    that fall outside a tube of width 2*eps around the fit. Those boundary
    points are the support vectors; everything inside the tube can be deleted
    without changing the solution. eps controls sparsity, C controls the
    tolerance for points outside the tube.
    """

    def __init__(self, C=1.0, eps=0.1, lr=0.01, epochs=1000):
        self.C, self.eps, self.lr, self.epochs = C, eps, lr, epochs

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.w, self.b = [0.0] * d, 0.0
        for _ in range(self.epochs):
            gw = list(self.w)                       # gradient of 0.5||w||^2
            gb = 0.0
            for xi, ti in zip(X, y):
                e = dot(self.w, xi) + self.b - ti
                if abs(e) > self.eps:               # outside the tube
                    s = 1.0 if e > 0 else -1.0
                    for j in range(d):
                        gw[j] += self.C * s * xi[j] / n
                    gb += self.C * s / n
            self.w = [w - self.lr * g for w, g in zip(self.w, gw)]
            self.b -= self.lr * gb
        return self

    def predict(self, X):
        return [dot(self.w, r) + self.b for r in X]


class GaussianProcessRegressor:
    """
    GAUSSIAN PROCESS REGRESSION
    ---------------------------
    A prior over FUNCTIONS: f ~ GP(0, k(x,x')). Any finite set of points is
    jointly Gaussian, so conditioning gives a closed-form posterior:

        mean(x*) = k_*^T (K + sigma_n^2 I)^-1 y
        var(x*)  = k(x*,x*) - k_*^T (K + sigma_n^2 I)^-1 k_*

    RBF kernel  k(u,v) = sig_f^2 exp(-||u-v||^2 / (2 l^2))
        l    = length-scale: how fast the function wiggles
        sig_f = output scale: how far it swings
        sig_n = noise level: how much of y is measurement error

    Non-parametric and gives exact uncertainty, but the matrix inverse makes it
    O(n^3) train / O(n^2) predict — so it shines for small n (Bayesian
    optimization, hyperparameter search) and is impractical for big data
    without sparse/inducing-point approximations.
    """

    def __init__(self, length_scale=1.0, sigma_f=1.0, sigma_n=0.1):
        self.l, self.sf, self.sn = length_scale, sigma_f, sigma_n

    def _k(self, u, v):
        return self.sf ** 2 * math.exp(-sum((a - b) ** 2 for a, b in zip(u, v))
                                       / (2 * self.l ** 2))

    def fit(self, X, y):
        self.X, self.y = X, y
        n = len(X)
        K = [[self._k(X[i], X[j]) + (self.sn ** 2 if i == j else 0.0)
              for j in range(n)] for i in range(n)]
        self.Kinv = inverse(K)
        self.alpha = matvec(self.Kinv, y)
        return self

    def predict(self, X, return_std=False):
        means, stds = [], []
        for x in X:
            ks = [self._k(x, xi) for xi in self.X]
            means.append(dot(ks, self.alpha))
            if return_std:
                v = self._k(x, x) - dot(ks, matvec(self.Kinv, ks))
                stds.append(math.sqrt(max(v, 0.0)))
        return (means, stds) if return_std else means


# =============================================================================
# C. CLASSIFICATION
# =============================================================================
class BernoulliNaiveBayes:
    """
    BERNOULLI NAIVE BAYES — binary feature presence/absence.
    Unlike Multinomial NB, it explicitly penalizes the ABSENCE of a feature:

        P(x|c) = prod_j [ p_jc^x_j * (1 - p_jc)^(1-x_j) ]
        p_jc   = (count_jc + a) / (n_c + 2a)          Laplace smoothing

    Better than Multinomial NB for short documents where non-occurrence is
    informative.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = sorted(set(y))
        d = len(X[0])
        self.logprior, self.p = {}, {}
        for c in self.classes:
            rows = [X[i] for i in range(len(X)) if y[i] == c]
            self.logprior[c] = math.log(len(rows) / len(X))
            self.p[c] = [(sum(r[j] for r in rows) + self.alpha) /
                         (len(rows) + 2 * self.alpha) for j in range(d)]
        return self

    def predict(self, X):
        out = []
        for x in X:
            best, bc = -float("inf"), None
            for c in self.classes:
                s = self.logprior[c]
                for j, v in enumerate(x):
                    pj = self.p[c][j]
                    s += math.log(pj) if v else math.log(1 - pj)
                if s > best:
                    best, bc = s, c
            out.append(bc)
        return out


class QDA:
    """
    QUADRATIC DISCRIMINANT ANALYSIS
    -------------------------------
    Same generative setup as LDA, but each class keeps its OWN covariance, so
    the quadratic terms no longer cancel:

        delta_c(x) = -0.5 log|Sig_c| - 0.5 (x-mu_c)^T Sig_c^-1 (x-mu_c) + log pi_c

    The boundary becomes a quadric (ellipse/hyperbola) rather than a hyperplane.
    Cost: d(d+1)/2 parameters PER CLASS, so it needs far more data than LDA and
    fails when a class has fewer samples than features (singular Sigma_c).
    LDA vs QDA is a textbook bias-variance trade-off.
    """

    def __init__(self, reg=1e-4):
        self.reg = reg

    def fit(self, X, y):
        self.classes = sorted(set(y))
        d = len(X[0])
        self.mu, self.Sinv, self.logdet, self.logprior = {}, {}, {}, {}
        for c in self.classes:
            rows = [X[i] for i in range(len(X)) if y[i] == c]
            m = [sum(r[j] for r in rows) / len(rows) for j in range(d)]
            self.mu[c] = m
            self.logprior[c] = math.log(len(rows) / len(X))
            S = zeros(d, d)
            for r in rows:
                dv = [r[j] - m[j] for j in range(d)]
                for a in range(d):
                    for b in range(d):
                        S[a][b] += dv[a] * dv[b]
            for a in range(d):
                for b in range(d):
                    S[a][b] /= max(len(rows) - 1, 1)
                S[a][a] += self.reg
            self.Sinv[c] = inverse(S)
            # log|S| from the LU pivots
            M = [r[:] for r in S]
            ld = 0.0
            for cc in range(d):
                p = max(range(cc, d), key=lambda r: abs(M[r][cc]))
                M[cc], M[p] = M[p], M[cc]
                ld += math.log(abs(M[cc][cc]) + EPS)
                for r in range(cc + 1, d):
                    f = M[r][cc] / M[cc][cc]
                    for k in range(cc, d):
                        M[r][k] -= f * M[cc][k]
            self.logdet[c] = ld
        return self

    def predict(self, X):
        out = []
        for x in X:
            best, bc = -float("inf"), None
            for c in self.classes:
                dv = [a - b for a, b in zip(x, self.mu[c])]
                q = dot(dv, matvec(self.Sinv[c], dv))
                s = -0.5 * self.logdet[c] - 0.5 * q + self.logprior[c]
                if s > best:
                    best, bc = s, c
            out.append(bc)
        return out


# ---------------------------------------------------------------- boosting core
class _RegTree:
    """Depth-limited regression tree used as the base learner for boosting."""

    def __init__(self, max_depth=3, min_samples=2):
        self.max_depth, self.min_samples = max_depth, min_samples

    def fit(self, X, y):
        self.root = self._build(list(range(len(X))), X, y, 0)
        return self

    def _build(self, idx, X, y, depth):
        vals = [y[i] for i in idx]
        if depth >= self.max_depth or len(idx) < self.min_samples:
            return ("leaf", sum(vals) / len(vals))
        best = (0.0, None, None)
        base = self._sse(vals)
        for f in range(len(X[0])):
            order = sorted(idx, key=lambda i: X[i][f])
            left = []
            for k in range(len(order) - 1):
                left.append(y[order[k]])
                if X[order[k]][f] == X[order[k + 1]][f]:
                    continue
                right = [y[i] for i in order[k + 1:]]
                gain = base - self._sse(left) - self._sse(right)
                if gain > best[0]:
                    best = (gain, f, (X[order[k]][f] + X[order[k + 1]][f]) / 2)
        if best[1] is None:
            return ("leaf", sum(vals) / len(vals))
        f, t = best[1], best[2]
        li = [i for i in idx if X[i][f] <= t]
        ri = [i for i in idx if X[i][f] > t]
        if not li or not ri:
            return ("leaf", sum(vals) / len(vals))
        return ("split", f, t, self._build(li, X, y, depth + 1),
                self._build(ri, X, y, depth + 1))

    @staticmethod
    def _sse(v):
        if not v:
            return 0.0
        m = sum(v) / len(v)
        return sum((x - m) ** 2 for x in v)

    def _one(self, node, x):
        while node[0] == "split":
            node = node[3] if x[node[1]] <= node[2] else node[4]
        return node[1]

    def predict(self, X):
        return [self._one(self.root, x) for x in X]


class XGBoostStyleBooster:
    """
    NEWTON (SECOND-ORDER) GRADIENT BOOSTING — the XGBoost formulation
    -----------------------------------------------------------------
    Take a 2nd-order Taylor expansion of the loss around the current prediction:

        Obj ~= sum_i [ g_i f(x_i) + 0.5 h_i f(x_i)^2 ] + Omega(f)
        Omega(f) = gamma * T + 0.5 * lambda * sum_j w_j^2      (T = # leaves)

    where g_i = dL/dF_i and h_i = d2L/dF_i^2. For a fixed tree structure the
    objective is quadratic in each leaf weight, so the optimum is exact:

        w_j*   = -G_j / (H_j + lambda)              G_j = sum g, H_j = sum h
        Obj*   = -0.5 sum_j G_j^2/(H_j + lambda) + gamma*T

    which gives the famous SPLIT GAIN:

        Gain = 0.5[ G_L^2/(H_L+lam) + G_R^2/(H_R+lam)
                   - (G_L+G_R)^2/(H_L+H_R+lam) ] - gamma

    A split is only taken if Gain > 0 — gamma is therefore built-in pruning.
    For logistic loss:  g = p - y,  h = p(1-p).
    For squared loss:   g = F - y,  h = 1  (which recovers plain gradient boosting).

    This is the real reason XGBoost outperformed classic GBM: it uses curvature
    (h) to size each leaf, and folds regularization directly into the split
    criterion instead of bolting it on afterwards.
    """

    def __init__(self, n_estimators=50, lr=0.3, max_depth=3,
                 lam=1.0, gamma=0.0, objective="logistic"):
        self.n, self.lr, self.max_depth = n_estimators, lr, max_depth
        self.lam, self.gamma, self.objective = lam, gamma, objective

    def _grad_hess(self, y, F):
        if self.objective == "logistic":
            p = [sigmoid(f) for f in F]
            return [pi - yi for pi, yi in zip(p, y)], [pi * (1 - pi) for pi in p]
        return [f - yi for f, yi in zip(F, y)], [1.0] * len(y)

    # ---- tree grown directly on (g, h) using the exact gain formula
    def _build(self, idx, X, g, h, depth):
        G, H = sum(g[i] for i in idx), sum(h[i] for i in idx)
        leaf = ("leaf", -G / (H + self.lam))
        if depth >= self.max_depth or len(idx) < 2:
            return leaf
        best = (self.gamma, None, None)
        for f in range(len(X[0])):
            order = sorted(idx, key=lambda i: X[i][f])
            GL = HL = 0.0
            for k in range(len(order) - 1):
                GL += g[order[k]]
                HL += h[order[k]]
                if X[order[k]][f] == X[order[k + 1]][f]:
                    continue
                GR, HR = G - GL, H - HL
                gain = 0.5 * (GL * GL / (HL + self.lam) + GR * GR / (HR + self.lam)
                              - G * G / (H + self.lam)) - self.gamma
                if gain > best[0]:
                    best = (gain, f, (X[order[k]][f] + X[order[k + 1]][f]) / 2)
        if best[1] is None:
            return leaf
        f, t = best[1], best[2]
        li = [i for i in idx if X[i][f] <= t]
        ri = [i for i in idx if X[i][f] > t]
        if not li or not ri:
            return leaf
        return ("split", f, t, self._build(li, X, g, h, depth + 1),
                self._build(ri, X, g, h, depth + 1))

    @staticmethod
    def _walk(node, x):
        while node[0] == "split":
            node = node[3] if x[node[1]] <= node[2] else node[4]
        return node[1]

    def fit(self, X, y):
        if self.objective == "logistic":
            p = min(max(mean(y), 1e-6), 1 - 1e-6)
            self.F0 = math.log(p / (1 - p))
        else:
            self.F0 = mean(y)
        F = [self.F0] * len(y)
        self.trees = []
        for _ in range(self.n):
            g, h = self._grad_hess(y, F)
            tree = self._build(list(range(len(X))), X, g, h, 0)
            self.trees.append(tree)
            for i, x in enumerate(X):
                F[i] += self.lr * self._walk(tree, x)
        return self

    def _raw(self, X):
        out = [self.F0] * len(X)
        for t in self.trees:
            for i, x in enumerate(X):
                out[i] += self.lr * self._walk(t, x)
        return out

    def predict_proba(self, X):
        return [sigmoid(v) for v in self._raw(X)]

    def predict(self, X):
        if self.objective == "logistic":
            return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]
        return self._raw(X)


# =============================================================================
# D. MORE ENSEMBLES
# =============================================================================
class ExtraTrees:
    """
    EXTREMELY RANDOMIZED TREES
    --------------------------
    Random forest picks the BEST threshold among candidates; Extra Trees draws
    ONE threshold uniformly at random per feature and keeps the best of those.
    It also (classically) uses the whole dataset instead of bootstrap samples.

    Effect: even lower correlation between trees -> lower ensemble variance,
    slightly higher bias, and much faster training (no sorting/scanning for the
    optimal cut point).
    """

    def __init__(self, n_estimators=30, max_depth=8, n_features=None,
                 task="clf", seed=0):
        self.n, self.max_depth, self.nf = n_estimators, max_depth, n_features
        self.task, self.seed = task, seed

    def _build(self, X, y, depth, rng):
        if depth >= self.max_depth or len(set(y)) == 1 or len(y) < 2:
            return ("leaf", self._leaf(y))
        d = len(X[0])
        feats = rng.sample(range(d), min(self.nf or max(1, int(math.sqrt(d))), d))
        best = (None, None, -float("inf"))
        for f in feats:
            lo = min(r[f] for r in X)
            hi = max(r[f] for r in X)
            if hi - lo < EPS:
                continue
            t = rng.uniform(lo, hi)                   # RANDOM cut point
            li = [i for i in range(len(X)) if X[i][f] <= t]
            ri = [i for i in range(len(X)) if X[i][f] > t]
            if not li or not ri:
                continue
            score = -(len(li) * self._imp([y[i] for i in li])
                      + len(ri) * self._imp([y[i] for i in ri])) / len(y)
            if score > best[2]:
                best = (f, t, score)
        if best[0] is None:
            return ("leaf", self._leaf(y))
        f, t = best[0], best[1]
        li = [i for i in range(len(X)) if X[i][f] <= t]
        ri = [i for i in range(len(X)) if X[i][f] > t]
        return ("split", f, t,
                self._build([X[i] for i in li], [y[i] for i in li], depth + 1, rng),
                self._build([X[i] for i in ri], [y[i] for i in ri], depth + 1, rng))

    def _imp(self, y):
        if self.task == "reg":
            if not y:
                return 0.0
            m = mean(y)
            return sum((v - m) ** 2 for v in y) / len(y)
        cnt = {}
        for t in y:
            cnt[t] = cnt.get(t, 0) + 1
        return 1 - sum((c / len(y)) ** 2 for c in cnt.values())

    def _leaf(self, y):
        if self.task == "reg":
            return mean(y)
        cnt = {}
        for t in y:
            cnt[t] = cnt.get(t, 0) + 1
        return max(cnt, key=cnt.get)

    def fit(self, X, y):
        rng = random.Random(self.seed)
        self.trees = [self._build(X, y, 0, rng) for _ in range(self.n)]
        return self

    @staticmethod
    def _walk(node, x):
        while node[0] == "split":
            node = node[3] if x[node[1]] <= node[2] else node[4]
        return node[1]

    def predict(self, X):
        out = []
        for x in X:
            votes = [self._walk(t, x) for t in self.trees]
            if self.task == "reg":
                out.append(mean(votes))
            else:
                cnt = {}
                for v in votes:
                    cnt[v] = cnt.get(v, 0) + 1
                out.append(max(cnt, key=cnt.get))
        return out


class StackingEnsemble:
    """
    STACKED GENERALIZATION
    ----------------------
    1. Split the training set into k folds.
    2. For each base model, produce OUT-OF-FOLD predictions for every training
       row (train on k-1 folds, predict the held-out one).
    3. Train a meta-learner on those OOF predictions as features.
    4. Refit base models on all the data for test-time prediction.

    THE CRITICAL DETAIL is step 2. If you feed the meta-learner IN-FOLD
    predictions, a model that memorizes the training set looks perfect, the
    meta-learner assigns it all the weight, and the whole ensemble collapses at
    test time. Out-of-fold predictions are what make stacking honest.
    """

    def __init__(self, base_factories, meta_factory, k=5, seed=0):
        self.base_factories, self.meta_factory = base_factories, meta_factory
        self.k, self.seed = k, seed

    def fit(self, X, y):
        n = len(X)
        idx = list(range(n))
        random.Random(self.seed).shuffle(idx)
        folds = [idx[i::self.k] for i in range(self.k)]
        meta_X = [[0.0] * len(self.base_factories) for _ in range(n)]
        for f in range(self.k):
            val = folds[f]
            tr = [i for g in range(self.k) if g != f for i in folds[g]]
            for m, factory in enumerate(self.base_factories):
                mdl = factory()
                mdl.fit([X[i] for i in tr], [y[i] for i in tr])
                for i, p in zip(val, mdl.predict([X[i] for i in val])):
                    meta_X[i][m] = float(p)
        self.bases = []
        for factory in self.base_factories:
            mdl = factory()
            mdl.fit(X, y)
            self.bases.append(mdl)
        self.meta = self.meta_factory()
        self.meta.fit(meta_X, y)
        return self

    def predict(self, X):
        cols = [[float(v) for v in b.predict(X)] for b in self.bases]
        meta_X = [[cols[m][i] for m in range(len(self.bases))] for i in range(len(X))]
        return self.meta.predict(meta_X)


class IsolationForest:
    """
    ISOLATION FOREST — anomaly detection by how EASILY a point is separated
    ----------------------------------------------------------------------
    Build random trees by picking a random feature and a random split value.
    Anomalies live in sparse regions, so they get isolated near the ROOT, while
    normal points require many splits. Average path length h(x) is the signal:

        c(n)  = 2 H(n-1) - 2(n-1)/n  ~= 2(ln(n-1) + 0.5772) - 2(n-1)/n
                (expected path length of an unsuccessful BST search — the
                 normalizer that makes scores comparable across sample sizes)
        s(x)  = 2^( -E[h(x)] / c(n) )

    s -> 1  strongly anomalous ;  s -> 0.5  normal ;  s -> 0 very dense region.

    Unlike LOF or Mahalanobis this computes NO distances, so it is O(n log n)
    and scales well to high dimensions.
    """

    def __init__(self, n_estimators=100, sample_size=256, seed=0):
        self.n, self.sample_size, self.seed = n_estimators, sample_size, seed

    @staticmethod
    def _c(n):
        if n <= 1:
            return 1e-9
        return 2 * (math.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

    def _build(self, X, depth, limit, rng):
        if depth >= limit or len(X) <= 1:
            return ("leaf", len(X))
        d = len(X[0])
        f = rng.randrange(d)
        lo, hi = min(r[f] for r in X), max(r[f] for r in X)
        if hi - lo < EPS:
            return ("leaf", len(X))
        t = rng.uniform(lo, hi)
        left = [r for r in X if r[f] <= t]
        right = [r for r in X if r[f] > t]
        if not left or not right:
            return ("leaf", len(X))
        return ("split", f, t, self._build(left, depth + 1, limit, rng),
                self._build(right, depth + 1, limit, rng))

    def fit(self, X):
        rng = random.Random(self.seed)
        self.psi = min(self.sample_size, len(X))
        limit = math.ceil(math.log2(max(self.psi, 2)))
        self.trees = []
        for _ in range(self.n):
            sample = [X[rng.randrange(len(X))] for _ in range(self.psi)]
            self.trees.append(self._build(sample, 0, limit, rng))
        return self

    def _path(self, node, x, depth=0):
        while node[0] == "split":
            node = node[3] if x[node[1]] <= node[2] else node[4]
            depth += 1
        return depth + self._c(node[1])      # + correction for the unsplit leaf

    def score_samples(self, X):
        cn = self._c(self.psi)
        return [2 ** (-mean([self._path(t, x) for t in self.trees]) / cn) for x in X]

    def predict(self, X, contamination=0.1):
        s = self.score_samples(X)
        thr = sorted(s, reverse=True)[max(int(len(s) * contamination) - 1, 0)]
        return [1 if v >= thr else 0 for v in s]      # 1 = anomaly


# =============================================================================
# E. MORE CLUSTERING
# =============================================================================
class MeanShift:
    """
    MEAN SHIFT — mode seeking on a kernel density estimate
    ------------------------------------------------------
    Repeatedly move each point to the weighted mean of its neighbourhood:

        m(x) = sum_i K(x_i - x) x_i / sum_i K(x_i - x)     (flat or Gaussian K)

    The shift vector m(x) - x points along the GRADIENT of the KDE, so every
    point climbs to a density mode. Points converging to the same mode form a
    cluster. No k needed — but the bandwidth h is critical and O(n^2) per
    iteration is the price.
    """

    def __init__(self, bandwidth=2.0, max_iter=100, tol=1e-3):
        self.h, self.max_iter, self.tol = bandwidth, max_iter, tol

    def fit(self, X):
        shifted = [list(p) for p in X]
        for _ in range(self.max_iter):
            moved = 0.0
            for i, p in enumerate(shifted):
                num = [0.0] * len(p)
                den = 0.0
                for q in X:
                    d2 = sum((a - b) ** 2 for a, b in zip(p, q))
                    w = math.exp(-d2 / (2 * self.h ** 2))     # Gaussian kernel
                    den += w
                    for j in range(len(p)):
                        num[j] += w * q[j]
                new = [v / den for v in num]
                moved = max(moved, euclidean(new, p))
                shifted[i] = new
            if moved < self.tol:
                break
        # merge points that converged to the same mode
        modes, labels = [], []
        for p in shifted:
            for k, m in enumerate(modes):
                if euclidean(p, m) < self.h / 2:
                    labels.append(k)
                    break
            else:
                modes.append(p)
                labels.append(len(modes) - 1)
        self.cluster_centers_, self.labels_ = modes, labels
        return self


class SpectralClustering:
    """
    SPECTRAL CLUSTERING — clustering as graph partitioning
    ------------------------------------------------------
    1. Affinity        W_ij = exp(-||x_i - x_j||^2 / (2 sigma^2))
    2. Degree          D = diag(sum_j W_ij)
    3. Laplacian       L = D - W        (or L_sym = I - D^-1/2 W D^-1/2)
    4. Take the eigenvectors of the k SMALLEST eigenvalues -> spectral embedding
    5. Run k-means in that embedding

    Why it works: minimizing the normalized cut is NP-hard, but its continuous
    relaxation is exactly the eigenvector problem above (Rayleigh quotient).
    The multiplicity of eigenvalue 0 equals the number of connected components,
    so the bottom eigenvectors encode the graph's natural partitions. It handles
    non-convex, interleaved shapes that defeat k-means. Cost: O(n^3).
    """

    def __init__(self, n_clusters=2, sigma=1.0, n_neighbors=10, seed=0):
        self.k, self.sigma, self.nn, self.seed = n_clusters, sigma, n_neighbors, seed

    def fit(self, X):
        n = len(X)
        W = [[math.exp(-sum((a - b) ** 2 for a, b in zip(X[i], X[j]))
                       / (2 * self.sigma ** 2)) if i != j else 0.0
              for j in range(n)] for i in range(n)]
        if self.nn:
            # Sparsify to a symmetric kNN graph. A fully-connected affinity lets
            # weak long-range edges bridge the two moons; keeping only each
            # point's nearest neighbours preserves the manifold structure and is
            # what makes spectral clustering work on non-convex shapes.
            keep = [set(sorted(range(n), key=lambda j: -W[i][j])[:self.nn])
                    for i in range(n)]
            W = [[W[i][j] if (j in keep[i] or i in keep[j]) else 0.0
                  for j in range(n)] for i in range(n)]
        deg = [sum(row) + EPS for row in W]
        # normalized Laplacian L_sym = I - D^-1/2 W D^-1/2
        L = [[(1.0 if i == j else 0.0) - W[i][j] / math.sqrt(deg[i] * deg[j])
              for j in range(n)] for i in range(n)]
        vals, vecs = _jacobi(L, sweeps=30)
        order = sorted(range(n), key=lambda i: vals[i])[:self.k]  # SMALLEST
        emb = [[vecs[r][i] for i in order] for r in range(n)]
        emb = [[v / (math.sqrt(sum(x * x for x in row)) + EPS) for v in row]
               for row in emb]                                    # row-normalize
        self.labels_ = _kmeans(emb, self.k, seed=self.seed)
        return self


def _jacobi(A, sweeps=30, tol=1e-10):
    """
    CYCLIC JACOBI eigendecomposition for symmetric matrices.

    Each rotation J(p,q,theta) zeroes one off-diagonal entry:
        tan(2 theta) = 2 a_pq / (a_qq - a_pp)
    A <- J^T A J drives A toward diagonal; V <- V J accumulates eigenvectors.

    Sweeping over every (p,q) pair in order costs O(n^2) per sweep and converges
    in roughly 6-10 sweeps, versus scanning for the largest off-diagonal element
    before each rotation, which is O(n^2) PER ROTATION and O(n^4) overall.
    That distinction matters a lot for spectral clustering, where n is the
    number of data points.
    """
    n = len(A)
    a = [r[:] for r in A]
    V = identity(n)
    for _ in range(sweeps):
        off = math.sqrt(sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off < tol:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-14:
                    continue
                th = 0.5 * math.atan2(2 * a[p][q], a[q][q] - a[p][p])
                c, s_ = math.cos(th), math.sin(th)
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s_ * akq, s_ * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s_ * aqk, s_ * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p], V[k][q] = c * vkp - s_ * vkq, s_ * vkp + c * vkq
    return [a[i][i] for i in range(n)], V


def _kmeans(X, k, iters=100, seed=0):
    """Minimal k-means used by SpectralClustering and TSNE demos."""
    rng = random.Random(seed)
    centers = [list(X[rng.randrange(len(X))]) for _ in range(k)]
    labels = [0] * len(X)
    for _ in range(iters):
        new = [min(range(k), key=lambda c: sum((a - b) ** 2
                                               for a, b in zip(x, centers[c])))
               for x in X]
        if new == labels:
            break
        labels = new
        for c in range(k):
            pts = [X[i] for i in range(len(X)) if labels[i] == c]
            if pts:
                centers[c] = [sum(p[j] for p in pts) / len(pts)
                              for j in range(len(X[0]))]
    return labels


# =============================================================================
# F. MANIFOLD LEARNING
# =============================================================================
class TSNE:
    """
    t-SNE (simplified, exact O(n^2) version)
    ----------------------------------------
    HIGH-D: convert distances to conditional probabilities with a Gaussian whose
    bandwidth is tuned per point by binary search so that the neighbourhood
    entropy matches log(perplexity):

        p_j|i = exp(-||xi-xj||^2 / 2 sig_i^2) / sum_k exp(...)
        p_ij  = (p_j|i + p_i|j) / 2n            (symmetrized)

    LOW-D: use a STUDENT-t with 1 degree of freedom (heavy tails):

        q_ij = (1 + ||yi-yj||^2)^-1 / sum_kl (1 + ||yk-yl||^2)^-1

    Cost: C = KL(P || Q), gradient

        dC/dy_i = 4 sum_j (p_ij - q_ij)(y_i - y_j)(1 + ||y_i-y_j||^2)^-1

    WHY THE STUDENT-t: the "crowding problem". The volume available at radius r
    grows as r^d, so moderately-distant points in high-d cannot all fit at the
    same relative distance in 2-D. The heavy tail lets dissimilar points sit far
    apart without paying a large penalty.

    CAVEATS: visualization only. No transform for new points, stochastic,
    cluster SIZES and BETWEEN-cluster distances are not meaningful, and
    perplexity (~5-50) changes the picture substantially.
    """

    def __init__(self, n_components=2, perplexity=15.0, lr=5.0,
                 n_iter=500, seed=0):
        self.d, self.perp, self.lr, self.n_iter = n_components, perplexity, lr, n_iter
        self.seed = seed

    def _p_matrix(self, X):
        n = len(X)
        D = [[sum((a - b) ** 2 for a, b in zip(X[i], X[j])) for j in range(n)]
             for i in range(n)]
        P = zeros(n, n)
        logU = math.log(self.perp)
        for i in range(n):
            lo, hi, beta = 1e-20, 1e20, 1.0
            for _ in range(50):                       # binary search on beta
                p = [math.exp(-D[i][j] * beta) if j != i else 0.0 for j in range(n)]
                s = sum(p) + EPS
                p = [v / s for v in p]
                H = -sum(v * math.log(v + EPS) for v in p)
                if abs(H - logU) < 1e-5:
                    break
                if H > logU:
                    lo, beta = beta, beta * 2 if hi == 1e20 else (beta + hi) / 2
                else:
                    hi, beta = beta, beta / 2 if lo == 1e-20 else (beta + lo) / 2
            P[i] = p
        return [[(P[i][j] + P[j][i]) / (2 * n) for j in range(n)] for i in range(n)]

    def fit_transform(self, X):
        rng = random.Random(self.seed)
        n = len(X)
        P = self._p_matrix(X)
        P = [[max(4.0 * v, 1e-12) for v in row] for row in P]   # early exaggeration
        Y = [[rng.gauss(0, 1e-2) for _ in range(self.d)] for _ in range(n)]
        vel = zeros(n, self.d)
        for it in range(self.n_iter):
            if it == 100:
                P = [[v / 4.0 for v in row] for row in P]
            num = [[1.0 / (1.0 + sum((Y[i][k] - Y[j][k]) ** 2 for k in range(self.d)))
                    if i != j else 0.0 for j in range(n)] for i in range(n)]
            Z = sum(sum(r) for r in num) + EPS
            mom = 0.5 if it < 100 else 0.8
            for i in range(n):
                grad = [0.0] * self.d
                for j in range(n):
                    if i == j:
                        continue
                    m = (P[i][j] - num[i][j] / Z) * num[i][j]
                    for k in range(self.d):
                        grad[k] += 4 * m * (Y[i][k] - Y[j][k])
                for k in range(self.d):
                    g = max(-5.0, min(5.0, grad[k]))    # clip: the plain
                    #                    gradient (no per-parameter gains) can
                    #                    diverge with a large learning rate
                    vel[i][k] = mom * vel[i][k] - self.lr * g
                    Y[i][k] += vel[i][k]
            cm = [sum(Y[i][k] for i in range(n)) / n for k in range(self.d)]
            Y = [[Y[i][k] - cm[k] for k in range(self.d)] for i in range(n)]
        return Y


class TruncatedSVD:
    """
    TRUNCATED SVD / LSA via POWER ITERATION with deflation
    ------------------------------------------------------
    Power iteration finds the dominant eigenvector of a symmetric matrix:
    repeatedly v <- Av/||Av||. Any starting vector's component along the top
    eigenvector is amplified by (lam1/lam2)^t relative to the rest.

    Apply it to X^T X to get right singular vectors, then DEFLATE
    (A <- A - lam v v^T) and repeat for the next component.

    Unlike PCA this does NOT centre the data, which is what lets it run on huge
    sparse matrices (term-document, user-item) — centring would destroy the
    sparsity. That's Latent Semantic Analysis.
    """

    def __init__(self, n_components=2, n_iter=200, seed=0):
        self.k, self.n_iter, self.seed = n_components, n_iter, seed

    def fit(self, X):
        d = len(X[0])
        Xt = transpose(X)
        C = matmul(Xt, X)
        rng = random.Random(self.seed)
        self.components_, self.singular_values_ = [], []
        for _ in range(self.k):
            v = [rng.gauss(0, 1) for _ in range(d)]
            nrm = math.sqrt(sum(a * a for a in v))
            v = [a / nrm for a in v]
            for _ in range(self.n_iter):
                w = matvec(C, v)
                nrm = math.sqrt(sum(a * a for a in w)) + EPS
                v = [a / nrm for a in w]
            lam = dot(v, matvec(C, v))
            self.components_.append(v)
            self.singular_values_.append(math.sqrt(max(lam, 0.0)))
            for i in range(d):                                  # deflation
                for j in range(d):
                    C[i][j] -= lam * v[i] * v[j]
        return self

    def transform(self, X):
        return [[dot(c, r) for c in self.components_] for r in X]


# =============================================================================
# G. RECOMMENDER SYSTEMS
# =============================================================================
class MatrixFactorization:
    """
    LATENT-FACTOR RECOMMENDER (Funk SVD / the Netflix Prize model)
    --------------------------------------------------------------
        rhat_ui = mu + b_u + b_i + p_u . q_i

    Minimize over OBSERVED ratings only (the matrix is ~99% missing, so this is
    matrix COMPLETION, not decomposition):

        min sum_(u,i in R) (r_ui - rhat_ui)^2
            + lam(||p_u||^2 + ||q_i||^2 + b_u^2 + b_i^2)

    SGD updates, with e = r - rhat:
        b_u += lr(e - lam b_u)      p_u += lr(e q_i - lam p_u)
        b_i += lr(e - lam b_i)      q_i += lr(e p_u - lam q_i)

    The bias terms matter enormously: they absorb "this user rates everything
    high" and "this movie is broadly liked", letting the latent factors model
    genuine TASTE interaction instead of global offsets.

    Cold start: a brand-new user or item has no ratings, so p_u/q_i stay at
    their priors -> fall back to popularity or content features.
    """

    def __init__(self, n_factors=8, lr=0.01, reg=0.05, epochs=200, seed=0):
        self.k, self.lr, self.reg, self.epochs, self.seed = \
            n_factors, lr, reg, epochs, seed

    def fit(self, ratings):
        """ratings: list of (user_id, item_id, rating)."""
        rng = random.Random(self.seed)
        users = sorted({u for u, _, _ in ratings})
        items = sorted({i for _, i, _ in ratings})
        self.ui = {u: k for k, u in enumerate(users)}
        self.ii = {i: k for k, i in enumerate(items)}
        self.mu = mean([r for _, _, r in ratings])
        nu, ni = len(users), len(items)
        self.P = [[rng.gauss(0, 0.1) for _ in range(self.k)] for _ in range(nu)]
        self.Q = [[rng.gauss(0, 0.1) for _ in range(self.k)] for _ in range(ni)]
        self.bu, self.bi = [0.0] * nu, [0.0] * ni
        data = list(ratings)
        for _ in range(self.epochs):
            rng.shuffle(data)
            for u, i, r in data:
                a, b = self.ui[u], self.ii[i]
                pred = self.mu + self.bu[a] + self.bi[b] + dot(self.P[a], self.Q[b])
                e = r - pred
                self.bu[a] += self.lr * (e - self.reg * self.bu[a])
                self.bi[b] += self.lr * (e - self.reg * self.bi[b])
                for f in range(self.k):
                    pf, qf = self.P[a][f], self.Q[b][f]
                    self.P[a][f] += self.lr * (e * qf - self.reg * pf)
                    self.Q[b][f] += self.lr * (e * pf - self.reg * qf)
        return self

    def predict(self, u, i):
        if u not in self.ui or i not in self.ii:
            return self.mu                                  # cold-start fallback
        a, b = self.ui[u], self.ii[i]
        return self.mu + self.bu[a] + self.bi[b] + dot(self.P[a], self.Q[b])

    def recommend(self, u, n=5, exclude=()):
        scored = [(i, self.predict(u, i)) for i in self.ii if i not in exclude]
        return sorted(scored, key=lambda t: -t[1])[:n]


class ItemBasedCF:
    """
    ITEM-BASED COLLABORATIVE FILTERING
    ----------------------------------
        sim(i,j)   = cosine (or adjusted-cosine / Pearson) over co-rating users
        rhat(u,i)  = sum_j sim(i,j) r_uj / sum_j |sim(i,j)|    over j rated by u

    Item-item is the industry default over user-user because item similarities
    are far more STABLE over time (a movie's neighbours barely change; a user's
    taste and rating history change constantly), and there are usually fewer
    items than users, so the similarity matrix is smaller and precomputable.
    """

    def __init__(self, k=10):
        self.k = k

    def fit(self, ratings):
        self.by_item, self.by_user = {}, {}
        for u, i, r in ratings:
            self.by_item.setdefault(i, {})[u] = r
            self.by_user.setdefault(u, {})[i] = r
        self.sim = {}
        items = list(self.by_item)
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                i, j = items[a], items[b]
                common = set(self.by_item[i]) & set(self.by_item[j])
                if len(common) < 2:
                    continue
                vi = [self.by_item[i][u] for u in common]
                vj = [self.by_item[j][u] for u in common]
                den = math.sqrt(sum(x * x for x in vi)) * math.sqrt(sum(x * x for x in vj))
                s = dot(vi, vj) / den if den > EPS else 0.0
                self.sim[(i, j)] = self.sim[(j, i)] = s
        return self

    def predict(self, u, i):
        if u not in self.by_user:
            return mean([r for it in self.by_item.values() for r in it.values()])
        sims = [(self.sim.get((i, j), 0.0), r) for j, r in self.by_user[u].items()
                if j != i]
        sims = sorted(sims, key=lambda t: -abs(t[0]))[:self.k]
        den = sum(abs(s) for s, _ in sims)
        if den < EPS:
            return mean(list(self.by_user[u].values()))
        return sum(s * r for s, r in sims) / den


# =============================================================================
# H. SEQUENCE MODELS
# =============================================================================
class HiddenMarkovModel:
    """
    HIDDEN MARKOV MODEL — three classic problems, three algorithms
    ---------------------------------------------------------------
    Parameters: pi (initial), A (transition), B (emission).

    1. EVALUATION  P(O | model)  ->  FORWARD algorithm
         alpha_1(i) = pi_i B_i(o_1)
         alpha_t(j) = [ sum_i alpha_(t-1)(i) A_ij ] B_j(o_t)
         P(O) = sum_i alpha_T(i)
       Cost O(T N^2) instead of the naive O(N^T) — dynamic programming over the
       fact that the future is independent of the past given the current state.

    2. DECODING  argmax_states P(S | O)  ->  VITERBI
         delta_t(j) = max_i [ delta_(t-1)(i) A_ij ] B_j(o_t)
       Same recursion with max instead of sum, plus backpointers.

    3. LEARNING  ->  BAUM-WELCH (EM)
         E-step: gamma_t(i)  = alpha_t(i) beta_t(i) / P(O)
                 xi_t(i,j)   = alpha_t(i) A_ij B_j(o_(t+1)) beta_(t+1)(j) / P(O)
         M-step: A_ij = sum_t xi_t(i,j) / sum_t gamma_t(i)
                 B_j(k) = sum_{t: o_t=k} gamma_t(j) / sum_t gamma_t(j)

    Assumptions: Markov property (next state depends only on current) and
    output independence. HMMs power classic speech recognition, POS tagging,
    and gene finding; CRFs and RNNs are the discriminative/neural successors.
    """

    def __init__(self, n_states, n_symbols, seed=0):
        self.N, self.M = n_states, n_symbols
        rng = random.Random(seed)

        def rowstoch(r, c):
            m = [[rng.random() + 0.5 for _ in range(c)] for _ in range(r)]
            return [[v / sum(row) for v in row] for row in m]
        self.A = rowstoch(self.N, self.N)
        self.B = rowstoch(self.N, self.M)
        p = [rng.random() + 0.5 for _ in range(self.N)]
        self.pi = [v / sum(p) for v in p]

    def _forward(self, O):
        T = len(O)
        alpha = zeros(T, self.N)
        scale = [0.0] * T
        for i in range(self.N):
            alpha[0][i] = self.pi[i] * self.B[i][O[0]]
        scale[0] = sum(alpha[0]) + EPS
        alpha[0] = [v / scale[0] for v in alpha[0]]
        for t in range(1, T):
            for j in range(self.N):
                alpha[t][j] = sum(alpha[t - 1][i] * self.A[i][j]
                                  for i in range(self.N)) * self.B[j][O[t]]
            scale[t] = sum(alpha[t]) + EPS
            alpha[t] = [v / scale[t] for v in alpha[t]]
        return alpha, scale

    def _backward(self, O, scale):
        T = len(O)
        beta = zeros(T, self.N)
        beta[T - 1] = [1.0 / scale[T - 1]] * self.N
        for t in range(T - 2, -1, -1):
            for i in range(self.N):
                beta[t][i] = sum(self.A[i][j] * self.B[j][O[t + 1]] * beta[t + 1][j]
                                 for j in range(self.N)) / scale[t]
        return beta

    def log_likelihood(self, O):
        _, scale = self._forward(O)
        return sum(math.log(s) for s in scale)

    def viterbi(self, O):
        T = len(O)
        delta = zeros(T, self.N)
        psi = [[0] * self.N for _ in range(T)]
        for i in range(self.N):
            delta[0][i] = math.log(self.pi[i] + EPS) + math.log(self.B[i][O[0]] + EPS)
        for t in range(1, T):
            for j in range(self.N):
                best = max(range(self.N),
                           key=lambda i: delta[t - 1][i] + math.log(self.A[i][j] + EPS))
                psi[t][j] = best
                delta[t][j] = delta[t - 1][best] + math.log(self.A[best][j] + EPS) \
                    + math.log(self.B[j][O[t]] + EPS)
        path = [max(range(self.N), key=lambda i: delta[T - 1][i])]
        for t in range(T - 1, 0, -1):
            path.append(psi[t][path[-1]])
        return path[::-1]

    def fit(self, O, n_iter=50, n_restarts=5, seed=0):
        """
        Baum-Welch is EM, so it only finds a LOCAL optimum — different random
        initializations of (pi, A, B) routinely converge to log-likelihoods that
        differ by tens of nats. Standard practice, implemented here: run several
        restarts and keep the parameters with the highest likelihood.
        """
        best = None
        rng = random.Random(seed)
        for r in range(n_restarts):
            if r > 0:                                   # re-randomize and retry
                def rowstoch(a, b):
                    mt = [[rng.random() + 0.5 for _ in range(b)] for _ in range(a)]
                    return [[v / sum(row) for v in row] for row in mt]
                self.A = rowstoch(self.N, self.N)
                self.B = rowstoch(self.N, self.M)
                p = [rng.random() + 0.5 for _ in range(self.N)]
                self.pi = [v / sum(p) for v in p]
            self._em(O, n_iter)
            ll = self.log_likelihood(O)
            if best is None or ll > best[0]:
                best = (ll, [r[:] for r in self.A], [r[:] for r in self.B],
                        self.pi[:])
        self.log_likelihood_, self.A, self.B, self.pi = best
        return self

    def _em(self, O, n_iter):
        T = len(O)
        for _ in range(n_iter):
            alpha, scale = self._forward(O)
            beta = self._backward(O, scale)
            gamma = [[alpha[t][i] * beta[t][i] for i in range(self.N)]
                     for t in range(T)]
            gamma = [[v / (sum(g) + EPS) for v in g] for g in gamma]
            xi = []
            for t in range(T - 1):
                m = [[alpha[t][i] * self.A[i][j] * self.B[j][O[t + 1]] * beta[t + 1][j]
                      for j in range(self.N)] for i in range(self.N)]
                s = sum(sum(r) for r in m) + EPS
                xi.append([[v / s for v in r] for r in m])
            self.pi = gamma[0][:]
            for i in range(self.N):
                den = sum(sum(x[i]) for x in xi) + EPS
                self.A[i] = [sum(x[i][j] for x in xi) / den for j in range(self.N)]
                gd = sum(gamma[t][i] for t in range(T)) + EPS
                self.B[i] = [sum(gamma[t][i] for t in range(T) if O[t] == k) / gd
                             for k in range(self.M)]
        return self


class KalmanFilter:
    """
    KALMAN FILTER — optimal recursive estimation for linear-Gaussian systems
    ------------------------------------------------------------------------
    state    x_t = F x_(t-1) + w,   w ~ N(0, Q)
    measure  z_t = H x_t + v,       v ~ N(0, R)

    PREDICT
        xhat = F x ;   P = F P F^T + Q
    UPDATE
        y = z - H xhat                      innovation (surprise)
        S = H P H^T + R                     innovation covariance
        K = P H^T S^-1                      KALMAN GAIN
        x = xhat + K y ;  P = (I - K H) P

    The gain K is the whole story: it's the ratio of how uncertain you are (P)
    to total uncertainty (P + measurement noise R). Noisy sensor -> small K ->
    trust the model; confident sensor -> large K -> trust the measurement.
    This is exactly recursive Bayesian updating of a Gaussian posterior, which
    is why the Kalman filter is optimal (minimum MSE) in the linear-Gaussian case.
    Non-linear extensions: EKF (linearize), UKF (sigma points), particle filter.
    """

    def __init__(self, F, H, Q, R, x0, P0):
        self.F, self.H, self.Q, self.R = F, H, Q, R
        self.x, self.P = x0, P0

    def step(self, z):
        # predict
        x = matvec(self.F, self.x)
        P = matmul(matmul(self.F, self.P), transpose(self.F))
        P = [[P[i][j] + self.Q[i][j] for j in range(len(P))] for i in range(len(P))]
        # update
        y = [a - b for a, b in zip(z, matvec(self.H, x))]
        S = matmul(matmul(self.H, P), transpose(self.H))
        S = [[S[i][j] + self.R[i][j] for j in range(len(S))] for i in range(len(S))]
        K = matmul(matmul(P, transpose(self.H)), inverse(S))
        self.x = [a + b for a, b in zip(x, matvec(K, y))]
        KH = matmul(K, self.H)
        n = len(P)
        self.P = [[sum((identity(n)[i][k] - KH[i][k]) * P[k][j] for k in range(n))
                   for j in range(n)] for i in range(n)]
        return self.x


class ARIMA:
    """
    ARIMA(p, d, 0) — autoregression on the d-times differenced series
    -----------------------------------------------------------------
        AR(p):  y_t = c + phi_1 y_(t-1) + ... + phi_p y_(t-p) + e_t

    Fit by least squares on the lag design matrix (equivalently the Yule-Walker
    equations, which express phi in terms of the autocorrelations).

    WHY THE "I": AR assumes STATIONARITY — constant mean, variance, and
    autocovariance. A trending series violates it, and the fitted phi then
    extrapolate catastrophically over long horizons (see the demo: RMSE 12 on
    the raw trended series vs 0.5 after differencing). Differencing
    z_t = y_t - y_(t-1) removes a linear trend; d=2 removes a quadratic one.
    Test stationarity with ADF (H0: unit root) or KPSS (H0: stationary), and
    choose p from where the PACF cuts off, or by minimizing AIC/BIC.

    Forecasts are produced on the differenced scale and then INTEGRATED back
    (cumulative sums) to the original scale — that's the inverse of differencing.
    """

    def __init__(self, p=2, d=0):
        self.p, self.d = p, d

    @staticmethod
    def _diff(s):
        return [s[i] - s[i - 1] for i in range(1, len(s))]

    def fit(self, series):
        self.original = list(series)
        self.tails = []                     # last value before each differencing
        z = list(series)
        for _ in range(self.d):
            self.tails.append(z[-1])
            z = self._diff(z)
        p = self.p
        X = [[1.0] + [z[t - k - 1] for k in range(p)] for t in range(p, len(z))]
        y = z[p:]
        Xt = transpose(X)
        self.coef = solve(matmul(Xt, X), matvec(Xt, y))
        self.z = z
        return self

    def forecast(self, steps=5):
        hist = list(self.z)
        out = []
        for _ in range(steps):
            x = [1.0] + [hist[-k - 1] for k in range(self.p)]
            v = dot(self.coef, x)
            out.append(v)
            hist.append(v)
        # integrate back up through each differencing level
        for last in reversed(self.tails):
            acc, restored = last, []
            for v in out:
                acc += v
                restored.append(acc)
            out = restored
        return out


# =============================================================================
# I. NLP
# =============================================================================
class TfidfVectorizer:
    """
    TF-IDF
    ------
        tf(t,d)  = count(t,d) / len(d)          (or raw count, or log(1+count))
        idf(t)   = log( (1+N) / (1+df(t)) ) + 1  (smoothed, sklearn-style)
        tfidf    = tf * idf, then L2-normalize each document vector

    The idf factor down-weights terms that appear in many documents ("the",
    "and") because a term that occurs everywhere carries no discriminative
    information — this is essentially an inverse-entropy weighting. L2
    normalization removes document-length effects so cosine similarity is
    meaningful.
    """

    def __init__(self, min_df=1, lowercase=True):
        self.min_df, self.lowercase = min_df, lowercase

    @staticmethod
    def _tok(doc, lower):
        d = doc.lower() if lower else doc
        return [w for w in "".join(c if c.isalnum() else " " for c in d).split() if w]

    def fit(self, docs):
        toks = [self._tok(d, self.lowercase) for d in docs]
        df = {}
        for t in toks:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        self.vocab = sorted(w for w, c in df.items() if c >= self.min_df)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        N = len(docs)
        self.idf = [math.log((1 + N) / (1 + df[w])) + 1 for w in self.vocab]
        return self

    def transform(self, docs):
        out = []
        for d in docs:
            toks = self._tok(d, self.lowercase)
            vec = [0.0] * len(self.vocab)
            for w in toks:
                if w in self.index:
                    vec[self.index[w]] += 1
            if toks:
                vec = [v / len(toks) * self.idf[i] for i, v in enumerate(vec)]
            nrm = math.sqrt(sum(v * v for v in vec))
            out.append([v / nrm for v in vec] if nrm > EPS else vec)
        return out

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)


class Word2VecSkipGram:
    """
    WORD2VEC — SKIP-GRAM WITH NEGATIVE SAMPLING
    -------------------------------------------
    Objective for a (centre, context) pair, with k sampled negatives:

        L = log sigma(v_c . u_o) + sum_{n=1..k} log sigma(-v_c . u_n)

    Gradients (v = input/centre vector, u = output/context vector):
        for the positive:  dL/dv += (1 - sig(v.u)) u ;  dL/du += (1 - sig(v.u)) v
        for a negative:    dL/dv -= sig(v.u_n) u_n  ;   dL/du_n -= sig(v.u_n) v

    WHY NEGATIVE SAMPLING: the full softmax denominator sums over the entire
    vocabulary (10^5-10^6 terms) on every single update. Negative sampling
    replaces it with k (5-20) binary logistic problems — "is this a real
    context word or a random one?" — turning an O(V) update into O(k).
    Negatives are drawn from the unigram distribution raised to the 3/4 power,
    which upsamples rare words relative to their frequency.

    The distributional hypothesis: words in similar contexts get similar
    vectors, which is where the famous king - man + woman ~ queen geometry
    comes from.
    """

    def __init__(self, dim=16, window=2, negatives=5, lr=0.05, epochs=50, seed=0):
        self.dim, self.window, self.neg = dim, window, negatives
        self.lr, self.epochs, self.seed = lr, epochs, seed

    def fit(self, sentences):
        rng = random.Random(self.seed)
        freq = {}
        for s in sentences:
            for w in s:
                freq[w] = freq.get(w, 0) + 1
        self.vocab = sorted(freq)
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        # negative-sampling distribution ~ freq^0.75
        weights = [freq[w] ** 0.75 for w in self.vocab]
        total = sum(weights)
        self.cum = []
        acc = 0.0
        for w in weights:
            acc += w / total
            self.cum.append(acc)
        self.Win = [[rng.uniform(-0.5, 0.5) / self.dim for _ in range(self.dim)]
                    for _ in range(V)]
        self.Wout = [[0.0] * self.dim for _ in range(V)]

        def sample_neg():
            r = rng.random()
            lo, hi = 0, len(self.cum) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if self.cum[mid] < r:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        pairs = []
        for s in sentences:
            ids = [self.idx[w] for w in s]
            for i, c in enumerate(ids):
                for j in range(max(0, i - self.window),
                               min(len(ids), i + self.window + 1)):
                    if i != j:
                        pairs.append((c, ids[j]))
        for _ in range(self.epochs):
            rng.shuffle(pairs)
            for c, o in pairs:
                v = self.Win[c]
                targets = [(o, 1.0)] + [(sample_neg(), 0.0) for _ in range(self.neg)]
                grad_v = [0.0] * self.dim
                for t, label in targets:
                    u = self.Wout[t]
                    g = (label - sigmoid(dot(v, u))) * self.lr
                    for k in range(self.dim):
                        grad_v[k] += g * u[k]
                        u[k] += g * v[k]
                for k in range(self.dim):
                    v[k] += grad_v[k]
        return self

    def similar(self, word, n=3):
        if word not in self.idx:
            return []
        v = self.Win[self.idx[word]]
        def cos(a, b):
            na = math.sqrt(sum(x * x for x in a)) + EPS
            nb = math.sqrt(sum(x * x for x in b)) + EPS
            return dot(a, b) / (na * nb)
        scored = [(w, cos(v, self.Win[i])) for w, i in self.idx.items() if w != word]
        return sorted(scored, key=lambda t: -t[1])[:n]


# =============================================================================
# J. REINFORCEMENT LEARNING
# =============================================================================
class QLearning:
    """
    Q-LEARNING — off-policy temporal-difference control
    ---------------------------------------------------
        Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]
                                   |------- TD target -------|
                                   |------ TD error (delta) ------|

    OFF-POLICY: the update uses max_a' Q(s',a') — the greedy action — regardless
    of what the behaviour policy actually did. That's why it can learn the
    optimal policy while exploring randomly. (SARSA substitutes Q(s',a') for the
    action actually taken, making it on-policy and more conservative near
    cliffs.)

    Exploration: epsilon-greedy, usually with epsilon decayed over time.
    gamma trades immediate vs future reward; alpha is the learning rate.
    Converges to Q* if every (s,a) is visited infinitely often and alpha
    satisfies the Robbins-Monro conditions.
    """

    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 eps=1.0, eps_decay=0.995, eps_min=0.01, seed=0):
        self.Q = [[0.0] * n_actions for _ in range(n_states)]
        self.nA = n_actions
        self.alpha, self.gamma = alpha, gamma
        self.eps, self.decay, self.eps_min = eps, eps_decay, eps_min
        self.rng = random.Random(seed)

    def act(self, s, greedy=False):
        if not greedy and self.rng.random() < self.eps:
            return self.rng.randrange(self.nA)
        return max(range(self.nA), key=lambda a: self.Q[s][a])

    def update(self, s, a, r, s2, done):
        target = r if done else r + self.gamma * max(self.Q[s2])
        self.Q[s][a] += self.alpha * (target - self.Q[s][a])

    def end_episode(self):
        self.eps = max(self.eps_min, self.eps * self.decay)


class MultiArmedBandit:
    """
    THE EXPLORATION-EXPLOITATION TRADE-OFF, in its purest form
    ----------------------------------------------------------
    eps-greedy   explore uniformly w.p. eps         simple, never stops exploring
    UCB1         pick argmax  mu_a + sqrt(2 ln t / n_a)
                 The bonus term is a confidence radius that SHRINKS as an arm is
                 pulled more -> "optimism in the face of uncertainty". Regret
                 is O(log t), which is optimal up to constants.
    Thompson     sample theta_a ~ Beta(successes+1, failures+1) for each arm and
                 pull the argmax. Bayesian, probability-matching, and usually
                 the best empirical performer; also handles delayed feedback
                 gracefully.
    """

    def __init__(self, n_arms, strategy="thompson", eps=0.1, seed=0):
        self.n = n_arms
        self.strategy, self.eps = strategy, eps
        self.counts = [0] * n_arms
        self.wins = [0] * n_arms
        self.values = [0.0] * n_arms
        self.t = 0
        self.rng = random.Random(seed)

    def select(self):
        self.t += 1
        if self.strategy == "eps":
            if self.rng.random() < self.eps:
                return self.rng.randrange(self.n)
            return max(range(self.n), key=lambda a: self.values[a])
        if self.strategy == "ucb":
            for a in range(self.n):
                if self.counts[a] == 0:
                    return a
            return max(range(self.n),
                       key=lambda a: self.values[a]
                       + math.sqrt(2 * math.log(self.t) / self.counts[a]))
        # thompson: sample from Beta(1+wins, 1+losses) via two Gammas
        best, ba = -1.0, 0
        for a in range(self.n):
            s = self.rng.gammavariate(1 + self.wins[a], 1)
            f = self.rng.gammavariate(1 + self.counts[a] - self.wins[a], 1)
            theta = s / (s + f)
            if theta > best:
                best, ba = theta, a
        return ba

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.wins[arm] += reward
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


# =============================================================================
# K. ASSOCIATION RULE MINING
# =============================================================================
def apriori(transactions, min_support=0.3, min_confidence=0.6):
    """
    APRIORI
    -------
        support(A)       = P(A)
        confidence(A->B) = support(A U B)/support(A) = P(B|A)
        lift(A->B)       = confidence / support(B)   > 1 => positive association

    The APRIORI PRINCIPLE (downward closure): every subset of a frequent itemset
    is itself frequent. Contrapositive: if an itemset is infrequent, every
    superset is infrequent and can be pruned without checking. That single fact
    collapses a 2^d search space into something tractable.

    Confidence alone is misleading: if B occurs in 90% of baskets, a rule with
    confidence 0.9 tells you nothing. LIFT corrects for B's base rate — always
    report it.
    """
    n = len(transactions)
    tsets = [set(t) for t in transactions]

    def sup(items):
        return sum(1 for t in tsets if items <= t) / n

    items = sorted({i for t in transactions for i in t})
    current = [frozenset([i]) for i in items if sup({i}) >= min_support]
    frequent = {c: sup(set(c)) for c in current}
    k = 2
    while current:
        cands = set()
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                u = current[i] | current[j]
                if len(u) == k:
                    # prune: all (k-1)-subsets must be frequent
                    if all(frozenset(u - {x}) in frequent for x in u):
                        cands.add(u)
        current = []
        for c in cands:
            s = sup(set(c))
            if s >= min_support:
                frequent[c] = s
                current.append(c)
        k += 1

    rules = []
    for itemset, s in frequent.items():
        if len(itemset) < 2:
            continue
        for x in itemset:
            ante, cons = itemset - {x}, frozenset([x])
            conf = s / frequent[ante]
            if conf >= min_confidence:
                lift = conf / frequent[cons]
                rules.append((set(ante), set(cons), round(s, 3),
                              round(conf, 3), round(lift, 3)))
    return frequent, sorted(rules, key=lambda r: -r[4])


# =============================================================================
# L. UTILITIES
# =============================================================================
def smote(X, y, minority_class=1, k=5, n_new=None, seed=0):
    """
    SMOTE — Synthetic Minority Over-sampling TEchnique
    --------------------------------------------------
        pick a minority point x, pick one of its k minority neighbours x_nn,
        new = x + rand(0,1) * (x_nn - x)

    Creates points ALONG THE LINE SEGMENTS between real minority examples,
    rather than duplicating them. Random oversampling duplicates rows, which
    lets the model memorize them and shrinks the effective decision region;
    SMOTE broadens it instead.

    CRITICAL: run SMOTE inside each CV fold, on training data ONLY. Oversampling
    before splitting leaks synthetic copies of validation points into training
    and produces spectacular, meaningless scores.
    """
    rng = random.Random(seed)
    minority = [X[i] for i in range(len(X)) if y[i] == minority_class]
    majority_n = sum(1 for t in y if t != minority_class)
    n_new = n_new if n_new is not None else max(0, majority_n - len(minority))
    newX, newY = [], []
    for _ in range(n_new):
        i = rng.randrange(len(minority))
        x = minority[i]
        nb = sorted(range(len(minority)),
                    key=lambda j: euclidean(x, minority[j]))[1:k + 1]
        if not nb:
            continue
        xn = minority[rng.choice(nb)]
        g = rng.random()
        newX.append([a + g * (b - a) for a, b in zip(x, xn)])
        newY.append(minority_class)
    return X + newX, list(y) + newY


def mutual_info_selection(X, y, bins=8, top_k=3):
    """
    FILTER FEATURE SELECTION BY MUTUAL INFORMATION
    ----------------------------------------------
        I(X;Y) = sum_x sum_y p(x,y) log( p(x,y) / (p(x)p(y)) )

    Continuous features are discretized into equal-width bins. Unlike Pearson
    correlation, MI captures ARBITRARY (including non-monotonic) dependence and
    is zero if and only if the variables are independent. It is a filter method:
    model-agnostic and fast, but blind to feature interactions and redundancy —
    two perfectly correlated informative features both score highly even though
    one is enough.
    """
    n, d = len(X), len(X[0])
    scores = []
    for j in range(d):
        col = [r[j] for r in X]
        lo, hi = min(col), max(col)
        width = (hi - lo) / bins + EPS
        disc = [min(int((v - lo) / width), bins - 1) for v in col]
        joint, px, py = {}, {}, {}
        for a, b in zip(disc, y):
            joint[(a, b)] = joint.get((a, b), 0) + 1
            px[a] = px.get(a, 0) + 1
            py[b] = py.get(b, 0) + 1
        mi = 0.0
        for (a, b), c in joint.items():
            pab = c / n
            mi += pab * math.log(pab / ((px[a] / n) * (py[b] / n)) + EPS)
        scores.append((j, mi))
    ranked = sorted(scores, key=lambda t: -t[1])
    return ranked[:top_k], ranked


class PlattScaling:
    """
    PLATT SCALING — turning SVM/tree scores into calibrated probabilities
    ---------------------------------------------------------------------
        P(y=1 | f) = sigmoid(A * f + B)

    Fit A and B by minimizing log loss on a HELD-OUT set (fitting on the
    training scores badly overfits, since training scores are unrepresentatively
    separated). Isotonic regression is the non-parametric alternative: more
    flexible, but needs considerably more calibration data.
    """

    def __init__(self, lr=0.05, epochs=1000):
        self.lr, self.epochs = lr, epochs
        self.A, self.B = -1.0, 0.0

    def fit(self, scores, y):
        n = len(scores)
        for _ in range(self.epochs):
            gA = gB = 0.0
            for f, t in zip(scores, y):
                p = sigmoid(self.A * f + self.B)
                gA += (p - t) * f
                gB += (p - t)
            self.A -= self.lr * gA / n
            self.B -= self.lr * gB / n
        return self

    def predict_proba(self, scores):
        return [sigmoid(self.A * f + self.B) for f in scores]


def permutation_importance(model, X, y, metric, n_repeats=5, seed=0):
    """
    PERMUTATION IMPORTANCE
    ----------------------
    Shuffle one feature column and measure how much the metric degrades:

        importance_j = baseline_score - score(X with column j permuted)

    Model-agnostic, computed on HELD-OUT data, and it measures what the model
    actually relies on. Prefer it to tree impurity importance, which is biased
    toward high-cardinality and continuous features and is computed on training
    data. Caveat: with correlated features, permuting one leaves the information
    available through its partner, so both look unimportant — check correlations
    first or use grouped permutation.
    """
    rng = random.Random(seed)
    base = metric(y, model.predict(X))
    d = len(X[0])
    out = []
    for j in range(d):
        drops = []
        for _ in range(n_repeats):
            Xp = [r[:] for r in X]
            col = [r[j] for r in Xp]
            rng.shuffle(col)
            for i in range(len(Xp)):
                Xp[i][j] = col[i]
            drops.append(base - metric(y, model.predict(Xp)))
        out.append((j, sum(drops) / len(drops)))
    return sorted(out, key=lambda t: -t[1])


# =============================================================================
# M. DEMO
# =============================================================================
def _hdr(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def demo():
    rng = random.Random(3)

    # ---------------------------------------------------------------- data
    Xr = [[rng.uniform(-5, 5), rng.uniform(-5, 5), rng.uniform(-5, 5)]
          for _ in range(240)]
    yr = [3 + 2 * a - 1.5 * b + 0.5 * c + rng.gauss(0, 1.0) for a, b, c in Xr]
    Xrtr, Xrte, yrtr, yrte = Xr[:190], Xr[190:], yr[:190], yr[190:]

    Xc, yc = [], []
    for i in range(300):
        t = math.pi * rng.random()
        if i % 2 == 0:
            Xc.append([math.cos(t) + rng.gauss(0, .18), math.sin(t) + rng.gauss(0, .18)])
            yc.append(0)
        else:
            Xc.append([1 - math.cos(t) + rng.gauss(0, .18),
                       .5 - math.sin(t) + rng.gauss(0, .18)])
            yc.append(1)
    Xctr, Xcte, yctr, ycte = Xc[:230], Xc[230:], yc[:230], yc[230:]

    # ---------------------------------------------------------------- A
    _hdr("A. OPTIMIZERS — minimizing the Rosenbrock-ish bowl f(w)=w0^2+10*w1^2")
    for kind in ["sgd", "momentum", "adagrad", "rmsprop", "adam"]:
        w = [5.0, 5.0]
        opt = Optimizer(kind, lr=0.1)
        for _ in range(200):
            g = [2 * w[0], 20 * w[1]]
            w = opt.step(w, g)
        print(f"  {kind:<10} final w = [{w[0]: .6f}, {w[1]: .6f}]  "
              f"loss {w[0]**2 + 10*w[1]**2:.3e}")

    # ---------------------------------------------------------------- B
    _hdr("B. REGRESSION")
    for name, m in [("ElasticNet(a=.1,l1=.5)", ElasticNet(0.1, 0.5)),
                    ("Huber (robust)", HuberRegressor()),
                    ("SVR (eps=0.3)", SVR(C=1.0, eps=0.3, lr=0.02, epochs=1500)),
                    ("BayesianLinearRegression", BayesianLinearRegression(1.0, 10.0))]:
        m.fit(Xrtr, yrtr)
        print(f"  {name:<28} test R2 {r2_score(yrte, m.predict(Xrte)):.4f} | "
              f"RMSE {rmse(yrte, m.predict(Xrte)):.4f}")

    blr = BayesianLinearRegression(1.0, 10.0).fit(Xrtr, yrtr)
    mu, sd = blr.predict([[0, 0, 0], [50, 50, 50]], return_std=True)
    print(f"      predictive std at a typical point : {sd[0]:.3f}")
    print(f"      predictive std FAR outside the data: {sd[1]:.3f}   "
          f"<- uncertainty grows, as it should")

    gp = GaussianProcessRegressor(2.0, 3.0, 0.3).fit(Xrtr[:60], yrtr[:60])
    gm, gs = gp.predict(Xrte[:5], return_std=True)
    print(f"  GaussianProcess  first 5 preds {[round(v,2) for v in gm]}")
    print(f"                   with std      {[round(v,2) for v in gs]}")

    # ---------------------------------------------------------------- C
    _hdr("C. CLASSIFICATION (two-moons, non-linear)")
    xgb = XGBoostStyleBooster(60, 0.3, 3, lam=1.0, gamma=0.0).fit(Xctr, yctr)
    print(f"  XGBoostStyleBooster (Newton)   test acc "
          f"{accuracy(ycte, xgb.predict(Xcte)):.4f}")
    qda = QDA().fit(Xctr, yctr)
    print(f"  QDA (quadratic boundary)       test acc "
          f"{accuracy(ycte, qda.predict(Xcte)):.4f}")
    Xb = [[1 if v > 0 else 0 for v in r] for r in Xctr]
    Xbte = [[1 if v > 0 else 0 for v in r] for r in Xcte]
    bnb = BernoulliNaiveBayes().fit(Xb, yctr)
    print(f"  BernoulliNB (binarized feats)  test acc "
          f"{accuracy(ycte, bnb.predict(Xbte)):.4f}")

    # ---------------------------------------------------------------- D
    _hdr("D. ENSEMBLES")
    et = ExtraTrees(40, 8, task="clf", seed=1).fit(Xctr, yctr)
    print(f"  ExtraTrees                     test acc "
          f"{accuracy(ycte, et.predict(Xcte)):.4f}")

    stack = StackingEnsemble(
        base_factories=[lambda: ExtraTrees(15, 6, task="reg", seed=2),
                        lambda: ElasticNet(0.05, 0.5)],
        meta_factory=lambda: ElasticNet(0.001, 0.0), k=4).fit(Xrtr, yrtr)
    print(f"  Stacking (ExtraTrees + Elastic) test R2 "
          f"{r2_score(yrte, stack.predict(Xrte)):.4f}   (out-of-fold meta features)")

    Xa = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(200)]
    Xa += [[rng.uniform(6, 9), rng.uniform(6, 9)] for _ in range(10)]   # anomalies
    iso = IsolationForest(100, 128, seed=0).fit(Xa)
    flags = iso.predict(Xa, contamination=0.05)
    caught = sum(flags[200:])
    print(f"  IsolationForest                caught {caught}/10 planted anomalies "
          f"in the top 5% of scores")

    # ---------------------------------------------------------------- E
    _hdr("E. CLUSTERING")
    Xbl = []
    for cx, cy in [(0, 0), (8, 8), (0, 8)]:
        Xbl += [[cx + rng.gauss(0, .8), cy + rng.gauss(0, .8)] for _ in range(50)]
    ms = MeanShift(bandwidth=2.0).fit(Xbl)
    print(f"  MeanShift          found {len(ms.cluster_centers_)} modes "
          f"(true = 3), no k supplied")
    mrng = random.Random(5)
    Xm2, ym2 = [], []
    for i in range(120):
        t = math.pi * mrng.random()
        if i % 2 == 0:
            Xm2.append([math.cos(t) + mrng.gauss(0, .07),
                        math.sin(t) + mrng.gauss(0, .07)])
            ym2.append(0)
        else:
            Xm2.append([1 - math.cos(t) + mrng.gauss(0, .07),
                        .5 - math.sin(t) + mrng.gauss(0, .07)])
            ym2.append(1)
    a1 = accuracy(ym2, SpectralClustering(2, .3, n_neighbors=8, seed=0).fit(Xm2).labels_)
    a2 = accuracy(ym2, SpectralClustering(2, .3, n_neighbors=0, seed=0).fit(Xm2).labels_)
    a3 = accuracy(ym2, _kmeans(Xm2, 2, seed=0))
    print(f"  On two moons (non-convex clusters):")
    print(f"     Spectral, kNN affinity graph   {max(a1, 1-a1):.3f}")
    print(f"     Spectral, fully connected      {max(a2, 1-a2):.3f}"
          f"   <- weak long-range edges bridge the moons")
    print(f"     plain k-means                  {max(a3, 1-a3):.3f}"
          f"   <- can only cut with a straight line")

    # ---------------------------------------------------------------- F
    _hdr("F. MANIFOLD LEARNING")
    trng = random.Random(3)
    sub, truth = [], []
    for c, (cx, cy) in enumerate([(0, 0), (8, 8), (0, 8)]):
        sub += [[cx + trng.gauss(0, .8), cy + trng.gauss(0, .8)] for _ in range(30)]
        truth += [c] * 30
    Y = TSNE(2, perplexity=10, lr=5.0, n_iter=500, seed=0).fit_transform(sub)
    # t-SNE optimizes NEIGHBOURHOODS, so measure neighbourhood preservation
    pres = 0.0
    for i in range(len(Y)):
        nb = sorted(range(len(Y)), key=lambda j: euclidean(Y[i], Y[j]))[1:6]
        pres += sum(1 for j in nb if truth[j] == truth[i]) / 5
    lab = _kmeans(Y, 3, seed=1)
    pur = sum(max([truth[i] for i in range(len(lab)) if lab[i] == c].count(t)
                  for t in set(truth)) for c in set(lab)) / len(lab)
    print(f"  t-SNE of 3 blobs: 5-NN neighbourhood preservation "
          f"{pres/len(Y):.3f}, k-means purity in 2-D {pur:.3f}")
    sv = TruncatedSVD(2, seed=0).fit(Xbl)
    print(f"  TruncatedSVD singular values: "
          f"{[round(v, 2) for v in sv.singular_values_]}")

    # ---------------------------------------------------------------- G
    _hdr("G. RECOMMENDERS")
    ratings = []
    true_p = {u: [rng.gauss(0, 1) for _ in range(2)] for u in range(30)}
    true_q = {i: [rng.gauss(0, 1) for _ in range(2)] for i in range(20)}
    for u in range(30):
        for i in range(20):
            if rng.random() < 0.5:
                r = 3 + dot(true_p[u], true_q[i]) + rng.gauss(0, .3)
                ratings.append((u, i, max(1, min(5, r))))
    split = int(len(ratings) * .8)
    tr, te = ratings[:split], ratings[split:]
    mf = MatrixFactorization(3, 0.02, 0.05, 150, seed=0).fit(tr)
    pred = [mf.predict(u, i) for u, i, _ in te]
    print(f"  MatrixFactorization  test RMSE {rmse([r for _,_,r in te], pred):.4f}")
    print(f"  top-3 for user 0: {[(i, round(s,2)) for i, s in mf.recommend(0, 3)]}")
    cf = ItemBasedCF(8).fit(tr)
    predcf = [cf.predict(u, i) for u, i, _ in te]
    print(f"  ItemBasedCF          test RMSE {rmse([r for _,_,r in te], predcf):.4f}")

    # ---------------------------------------------------------------- H
    _hdr("H. SEQUENCE MODELS")
    # generate from a known HMM: state 0 emits mostly 0/1, state 1 mostly 2/3
    trueA = [[0.9, 0.1], [0.15, 0.85]]
    trueB = [[0.5, 0.4, 0.05, 0.05], [0.05, 0.05, 0.5, 0.4]]
    s, obs, states = 0, [], []
    for _ in range(600):
        r = rng.random()
        obs.append(0 if r < trueB[s][0] else 1 if r < sum(trueB[s][:2])
                   else 2 if r < sum(trueB[s][:3]) else 3)
        states.append(s)
        s = 0 if rng.random() < trueA[s][0] else 1
    hmm = HiddenMarkovModel(2, 4, seed=1)
    before = hmm.log_likelihood(obs)
    hmm.fit(obs, n_iter=40, n_restarts=5, seed=2)
    after = hmm.log_likelihood(obs)
    path = hmm.viterbi(obs)
    acc = accuracy(states, path)
    print(f"  HMM Baum-Welch log-likelihood: {before:.2f} -> {after:.2f} "
          f"(5 restarts, best kept)")
    print(f"  Viterbi state recovery: {max(acc, 1-acc):.4f} "
          f"(labels may be permuted)")

    kf = KalmanFilter(F=[[1, 1], [0, 1]], H=[[1, 0]],
                      Q=[[0.001, 0], [0, 0.001]], R=[[0.5]],
                      x0=[0.0, 0.0], P0=[[1.0, 0], [0, 1.0]])
    truth_pos, est, noisy = [], [], []
    pos, vel = 0.0, 0.5
    for _ in range(60):
        pos += vel
        truth_pos.append(pos)
        z = pos + rng.gauss(0, 0.7)
        noisy.append(z)
        est.append(kf.step([z])[0])
    print(f"  Kalman filter  RMSE of raw sensor {rmse(truth_pos, noisy):.4f} "
          f"-> filtered {rmse(truth_pos, est):.4f}")

    srng = random.Random(3)
    series = [10 + 5 * math.sin(t / 6) + 0.3 * t + srng.gauss(0, .3)
              for t in range(140)]
    raw = ARIMA(p=10, d=0).fit(series[:110]).forecast(15)
    dif = ARIMA(p=10, d=1).fit(series[:110]).forecast(15)
    print(f"  ARIMA(10,0,0) on the raw TRENDED series : 15-step RMSE "
          f"{rmse(series[110:125], raw):.4f}")
    print(f"  ARIMA(10,1,0) after one differencing    : 15-step RMSE "
          f"{rmse(series[110:125], dif):.4f}   <- stationarity matters")

    # ---------------------------------------------------------------- I
    _hdr("I. NLP")
    docs = ["the cat sat on the mat", "the dog sat on the log",
            "cats and dogs are pets", "machine learning is fun",
            "learning machine models is fun"]
    tf = TfidfVectorizer().fit(docs)
    V = tf.transform(docs)
    def cos(a, b):
        return dot(a, b)
    print(f"  TF-IDF vocab size {len(tf.vocab)}")
    print(f"  cos(doc0, doc1) = {cos(V[0], V[1]):.3f}  (both about sitting)")
    print(f"  cos(doc0, doc3) = {cos(V[0], V[3]):.3f}  (unrelated)")
    print(f"  cos(doc3, doc4) = {cos(V[3], V[4]):.3f}  (both about ML)")

    sents = [["king", "queen", "royal", "palace"],
             ["queen", "king", "royal", "crown"],
             ["cat", "dog", "pet", "animal"],
             ["dog", "cat", "animal", "pet"]] * 25
    w2v = Word2VecSkipGram(dim=12, window=2, negatives=4, epochs=25, seed=1).fit(sents)
    print(f"  word2vec neighbours of 'king': "
          f"{[(w, round(s,2)) for w, s in w2v.similar('king', 3)]}")
    print(f"  word2vec neighbours of 'cat' : "
          f"{[(w, round(s,2)) for w, s in w2v.similar('cat', 3)]}")

    # ---------------------------------------------------------------- J
    _hdr("J. REINFORCEMENT LEARNING")
    # 1-D corridor: 8 states, reach the right end for reward 1
    N = 8
    ql = QLearning(N, 2, alpha=0.2, gamma=0.95, seed=0)
    for ep in range(600):
        s = 0
        for _ in range(60):
            a = ql.act(s)
            s2 = max(0, s - 1) if a == 0 else min(N - 1, s + 1)
            done = s2 == N - 1
            ql.update(s, a, 1.0 if done else -0.01, s2, done)
            s = s2
            if done:
                break
        ql.end_episode()
    policy = "".join("R" if ql.act(s, greedy=True) == 1 else "L" for s in range(N))
    print(f"  Q-learning learned policy (L/R per state): {policy}")
    print(f"  V(s) = max_a Q(s,a): "
          f"{[round(max(ql.Q[s]), 3) for s in range(N)]}   <- rises toward the goal")

    true_rates = [0.2, 0.5, 0.75]
    for strat in ["eps", "ucb", "thompson"]:
        b = MultiArmedBandit(3, strat, 0.1, seed=0)
        total = 0
        for _ in range(1500):
            a = b.select()
            r = 1 if random.Random(b.t * 7 + a).random() < true_rates[a] else 0
            b.update(a, r)
            total += r
        best_arm_pulls = b.counts[2] / sum(b.counts)
        print(f"  Bandit {strat:<9} reward {total:4d}/1500 | "
              f"fraction on the best arm {best_arm_pulls:.3f}")

    # ---------------------------------------------------------------- K
    _hdr("K. ASSOCIATION RULES (Apriori)")
    baskets = [["bread", "milk"], ["bread", "diaper", "beer", "eggs"],
               ["milk", "diaper", "beer", "cola"], ["bread", "milk", "diaper", "beer"],
               ["bread", "milk", "diaper", "cola"], ["milk", "diaper", "beer"],
               ["bread", "beer", "diaper"], ["bread", "milk"]]
    freq, rules = apriori(baskets, 0.3, 0.6)
    print(f"  {len(freq)} frequent itemsets at support >= 0.3")
    print("  top rules by lift:")
    for a, c, s, cf_, lf in rules[:5]:
        print(f"     {sorted(a)} -> {sorted(c)}   supp {s}  conf {cf_}  lift {lf}")

    # ---------------------------------------------------------------- L
    _hdr("L. UTILITIES")
    Xi = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(190)] + \
         [[rng.gauss(2, .5), rng.gauss(2, .5)] for _ in range(10)]
    yi = [0] * 190 + [1] * 10
    Xs, ys = smote(Xi, yi, 1, k=3, seed=0)
    print(f"  SMOTE: class balance {yi.count(1)}/{len(yi)} -> "
          f"{ys.count(1)}/{len(ys)}")

    Xm = [[r[0], r[1], rng.gauss(0, 1)] for r in Xctr]      # feature 2 = pure noise
    top, ranked = mutual_info_selection(Xm, yctr, bins=8, top_k=2)
    print(f"  Mutual information per feature (2 = injected noise): "
          f"{[(j, round(v, 4)) for j, v in ranked]}")

    scores = [dot([1.0, -1.0], r) for r in Xcte]
    ps = PlattScaling().fit(scores, ycte)
    probs = ps.predict_proba(scores)
    print(f"  Platt scaling: raw scores in [{min(scores):.2f}, {max(scores):.2f}] "
          f"-> probs in [{min(probs):.3f}, {max(probs):.3f}]")

    imp = permutation_importance(et, Xcte, ycte, accuracy, n_repeats=5)
    print(f"  Permutation importance (ExtraTrees on moons): "
          f"{[(j, round(v, 4)) for j, v in imp]}")

    print("\n" + "=" * 72)
    print("Part 2 complete — all algorithms run with zero external libraries.")
    print("=" * 72)


if __name__ == "__main__":
    print("=" * 72)
    print(" ML FROM SCRATCH — PART 2: the algorithms the first file didn't cover")
    print("=" * 72)
    demo()
