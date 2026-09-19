"""
================================================================================
 MACHINE LEARNING ALGORITHMS FROM SCRATCH — pure Python, zero libraries
================================================================================
 Only `math` and `random` from the standard library. No NumPy, no scikit-learn.
 Every algorithm carries its derivation in the class docstring.

 CONTENTS
 --------
 0. Linear algebra utilities .......... transpose, matmul, solve, inverse,
                                        determinant, Jacobi eigendecomposition
 1. Data utilities .................... train_test_split, KFold, Standardizer,
                                        MinMaxScaler, one-hot, polynomial features
 2. Metrics ........................... MSE, RMSE, MAE, R2, accuracy, precision,
                                        recall, F1, confusion matrix, ROC-AUC,
                                        silhouette
 3. Linear models ..................... LinearRegression (GD + normal equation),
                                        RidgeRegression, LassoRegression (coord.
                                        descent), LogisticRegression, Perceptron,
                                        LinearSVM (hinge subgradient),
                                        KernelSVM (simplified SMO)
 4. Instance based .................... KNNClassifier, KNNRegressor
 5. Probabilistic ..................... GaussianNaiveBayes, MultinomialNaiveBayes,
                                        LinearDiscriminantAnalysis
 6. Trees ............................. DecisionTree (CART: gini/entropy/MSE)
 7. Ensembles ......................... RandomForest, AdaBoost (SAMME),
                                        GradientBoosting (reg + binary clf),
                                        BaggingRegressor
 8. Neural networks ................... MLP (backpropagation, ReLU/tanh/sigmoid,
                                        softmax + cross-entropy, SGD/momentum)
 9. Unsupervised ...................... KMeans, KMeans++ init, DBSCAN,
                                        AgglomerativeClustering, PCA,
                                        GaussianMixture (EM)

 CONVENTIONS
 -----------
 X  : list of rows, each row a list of floats            shape (n, d)
 y  : list of floats (regression) or ints (classification)
 All models expose .fit(X, y) and .predict(X); classifiers also .predict_proba.
================================================================================
"""

import math
import random

random.seed(42)

EPS = 1e-12


# =============================================================================
# 0. LINEAR ALGEBRA
# =============================================================================
def zeros(n, m=None):
    return [0.0] * n if m is None else [[0.0] * m for _ in range(n)]


def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(A):
    return [list(col) for col in zip(*A)]


def matmul(A, B):
    """(n x m) @ (m x p) -> (n x p).  Cost O(n*m*p)."""
    mB, p = len(B), len(B[0])
    out = [[0.0] * p for _ in range(len(A))]
    for i, row in enumerate(A):
        oi = out[i]
        for k in range(mB):
            a = row[k]
            if a == 0.0:
                continue
            Bk = B[k]
            for j in range(p):
                oi[j] += a * Bk[j]
    return out


def matvec(A, v):
    return [sum(a * x for a, x in zip(row, v)) for row in A]


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def vec_add(u, v):
    return [a + b for a, b in zip(u, v)]


def vec_sub(u, v):
    return [a - b for a, b in zip(u, v)]


def scale(v, c):
    return [c * a for a in v]


def norm(v):
    return math.sqrt(sum(a * a for a in v))


def solve(A, b):
    """
    Solve A x = b by Gaussian elimination with partial pivoting.
    Partial pivoting keeps |multipliers| <= 1, which bounds round-off growth.
    Cost O(n^3).
    """
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < EPS:
            raise ValueError("Singular matrix (collinear features).")
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / pv
            if f:
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def inverse(A):
    """Gauss-Jordan: augment [A | I], reduce to [I | A^-1]."""
    n = len(A)
    M = [A[i][:] + identity(n)[i] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < EPS:
            raise ValueError("Singular matrix.")
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [row[n:] for row in M]


def determinant(A):
    """det = product of pivots, sign-flipped once per row swap."""
    n = len(A)
    M = [row[:] for row in A]
    det, sign = 1.0, 1.0
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[piv][c]) < EPS:
            return 0.0
        if piv != c:
            M[c], M[piv] = M[piv], M[c]
            sign = -sign
        det *= M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n):
                M[r][k] -= f * M[c][k]
    return sign * det


def jacobi_eigen(A, iters=100, tol=1e-10):
    """
    Eigendecomposition of a SYMMETRIC matrix by cyclic Jacobi rotations.

    Repeatedly zero the largest off-diagonal element a_pq with a rotation
    J(p, q, theta) where   tan(2*theta) = 2*a_pq / (a_qq - a_pp).
    A <- J^T A J drives A toward a diagonal matrix of eigenvalues, while
    V <- V J accumulates the eigenvectors (columns of V).

    Returns (eigenvalues, eigenvectors_as_columns) sorted descending.
    """
    n = len(A)
    a = [row[:] for row in A]
    V = identity(n)
    for _ in range(iters):
        p, q, off = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(a[i][j]) > off:
                    off, p, q = abs(a[i][j]), i, j
        if off < tol:
            break
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        theta = 0.5 * math.atan2(2 * apq, aqq - app)
        c, s = math.cos(theta), math.sin(theta)
        for k in range(n):
            akp, akq = a[k][p], a[k][q]
            a[k][p] = c * akp - s * akq
            a[k][q] = s * akp + c * akq
        for k in range(n):
            apk, aqk = a[p][k], a[q][k]
            a[p][k] = c * apk - s * aqk
            a[q][k] = s * apk + c * aqk
        for k in range(n):
            vkp, vkq = V[k][p], V[k][q]
            V[k][p] = c * vkp - s * vkq
            V[k][q] = s * vkp + c * vkq
    eigvals = [a[i][i] for i in range(n)]
    order = sorted(range(n), key=lambda i: -eigvals[i])
    vals = [eigvals[i] for i in order]
    vecs = [[V[r][i] for i in order] for r in range(n)]   # eigenvectors in columns
    return vals, vecs


# =============================================================================
# 1. DATA UTILITIES
# =============================================================================
def train_test_split(X, y, test_size=0.2, seed=0, stratify=False):
    """Shuffle then cut. stratify=True keeps class proportions in both halves."""
    rng = random.Random(seed)
    if not stratify:
        idx = list(range(len(X)))
        rng.shuffle(idx)
        cut = int(len(X) * (1 - test_size))
        tr, te = idx[:cut], idx[cut:]
    else:
        buckets = {}
        for i, label in enumerate(y):
            buckets.setdefault(label, []).append(i)
        tr, te = [], []
        for label, ids in buckets.items():
            rng.shuffle(ids)
            cut = int(len(ids) * (1 - test_size))
            tr += ids[:cut]
            te += ids[cut:]
        rng.shuffle(tr)
        rng.shuffle(te)
    return ([X[i] for i in tr], [X[i] for i in te],
            [y[i] for i in tr], [y[i] for i in te])


def k_fold_indices(n, k=5, seed=0):
    """Yield (train_idx, val_idx) for k-fold cross validation."""
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    for i in range(k):
        val = folds[i]
        train = [j for f in range(k) if f != i for j in folds[f]]
        yield train, val


def cross_val_score(model_factory, X, y, k=5, metric=None, seed=0):
    """Generic k-fold CV. model_factory() must return a fresh untrained model."""
    scores = []
    for tr, va in k_fold_indices(len(X), k, seed):
        m = model_factory()
        m.fit([X[i] for i in tr], [y[i] for i in tr])
        pred = m.predict([X[i] for i in va])
        scores.append(metric([y[i] for i in va], pred))
    return scores


class Standardizer:
    """z = (x - mu) / sigma. Fit on TRAIN only to avoid leaking test statistics."""

    def fit(self, X):
        n, d = len(X), len(X[0])
        self.mean = [sum(r[j] for r in X) / n for j in range(d)]
        self.std = []
        for j in range(d):
            var = sum((r[j] - self.mean[j]) ** 2 for r in X) / n
            self.std.append(math.sqrt(var) if var > EPS else 1.0)
        return self

    def transform(self, X):
        return [[(r[j] - self.mean[j]) / self.std[j] for j in range(len(r))] for r in X]

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    """x' = (x - min) / (max - min), mapped to [0, 1]."""

    def fit(self, X):
        d = len(X[0])
        self.lo = [min(r[j] for r in X) for j in range(d)]
        self.hi = [max(r[j] for r in X) for j in range(d)]
        return self

    def transform(self, X):
        return [[(r[j] - self.lo[j]) / max(self.hi[j] - self.lo[j], EPS)
                 for j in range(len(r))] for r in X]

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def one_hot(y, n_classes=None):
    k = n_classes or (max(y) + 1)
    return [[1.0 if j == label else 0.0 for j in range(k)] for label in y]


def polynomial_features(X, degree=2):
    """Adds squares and pairwise products so a LINEAR model can fit curves."""
    out = []
    for row in X:
        d = len(row)
        new = list(row)
        for deg in range(2, degree + 1):
            for j in range(d):
                new.append(row[j] ** deg)
        for i in range(d):
            for j in range(i + 1, d):
                new.append(row[i] * row[j])
        out.append(new)
    return out


# =============================================================================
# 2. METRICS
# =============================================================================
def mse(y, p):
    return sum((a - b) ** 2 for a, b in zip(y, p)) / len(y)


def rmse(y, p):
    return math.sqrt(mse(y, p))


def mae(y, p):
    return sum(abs(a - b) for a, b in zip(y, p)) / len(y)


def r2_score(y, p):
    """R^2 = 1 - SS_res/SS_tot. 1.0 is perfect, 0.0 equals predicting the mean."""
    m = sum(y) / len(y)
    ss_res = sum((a - b) ** 2 for a, b in zip(y, p))
    ss_tot = sum((a - m) ** 2 for a in y)
    return 1 - ss_res / max(ss_tot, EPS)


def accuracy(y, p):
    return sum(1 for a, b in zip(y, p) if a == b) / len(y)


def confusion_matrix(y, p, labels=None):
    labels = labels or sorted(set(y) | set(p))
    idx = {l: i for i, l in enumerate(labels)}
    M = zeros(len(labels), len(labels))
    for a, b in zip(y, p):
        M[idx[a]][idx[b]] += 1
    return M, labels


def precision_recall_f1(y, p, positive=1):
    """
    precision = TP / (TP + FP)   "of those I flagged, how many were right"
    recall    = TP / (TP + FN)   "of the real positives, how many did I catch"
    F1        = harmonic mean = 2PR / (P + R)
    """
    tp = sum(1 for a, b in zip(y, p) if a == positive and b == positive)
    fp = sum(1 for a, b in zip(y, p) if a != positive and b == positive)
    fn = sum(1 for a, b in zip(y, p) if a == positive and b != positive)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, EPS)
    return prec, rec, f1


def roc_auc(y, scores, positive=1):
    """
    AUC via the Mann-Whitney U statistic: the probability that a random
    positive is scored above a random negative. Computed from rank sums.
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    pos = [i for i, t in enumerate(y) if t == positive]
    neg = [i for i, t in enumerate(y) if t != positive]
    if not pos or not neg:
        return float("nan")
    rank_sum = sum(ranks[i] for i in pos)
    u = rank_sum - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def log_loss(y, prob, eps=1e-15):
    """-1/n * sum[ y*log(p) + (1-y)*log(1-p) ] — the logistic loss."""
    s = 0.0
    for t, p in zip(y, prob):
        p = min(max(p, eps), 1 - eps)
        s += t * math.log(p) + (1 - t) * math.log(1 - p)
    return -s / len(y)


def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def manhattan(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def silhouette_score(X, labels):
    """
    s(i) = (b - a) / max(a, b)
      a = mean distance to own cluster, b = mean distance to nearest other cluster.
    Ranges [-1, 1]; higher means tighter, better-separated clusters.
    """
    clusters = {}
    for i, l in enumerate(labels):
        clusters.setdefault(l, []).append(i)
    if len(clusters) < 2:
        return 0.0
    total = 0.0
    for i, li in enumerate(labels):
        own = [j for j in clusters[li] if j != i]
        a = sum(euclidean(X[i], X[j]) for j in own) / len(own) if own else 0.0
        b = min(sum(euclidean(X[i], X[j]) for j in ids) / len(ids)
                for l, ids in clusters.items() if l != li)
        total += (b - a) / max(a, b, EPS)
    return total / len(X)


# =============================================================================
# 3. LINEAR MODELS
# =============================================================================
class LinearRegression:
    """
    ORDINARY LEAST SQUARES
    ----------------------
    Model      : y_hat = b + w . x
    Loss       : J = (1/n) * sum (y_hat - y)^2                       [MSE]
    Gradients  : dJ/dw_j = (2/n) * sum (y_hat_i - y_i) * x_ij
                 dJ/db   = (2/n) * sum (y_hat_i - y_i)
    Update     : w <- w - alpha * dJ/dw

    Why the closed form works: J is convex and quadratic in theta, so setting
    the gradient to zero gives the global optimum in one shot:
                 X^T X theta = X^T y        (the "normal equations")
    Gradient descent is preferred when d is large (inverting is O(d^3)) or
    when the data streams in. Standardize features first so all directions of
    the loss surface curve at a similar rate.
    """

    def __init__(self, lr=0.01, epochs=2000, tol=1e-11):
        self.lr, self.epochs, self.tol = lr, epochs, tol
        self.history = []

    def predict(self, X):
        return [dot(self.w, r) + self.b for r in X]

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.w, self.b = [0.0] * d, 0.0
        prev = float("inf")
        for _ in range(self.epochs):
            err = [p - t for p, t in zip(self.predict(X), y)]
            loss = sum(e * e for e in err) / n
            self.history.append(loss)
            gw = [0.0] * d
            for i, r in enumerate(X):
                e = err[i]
                for j in range(d):
                    gw[j] += e * r[j]
            self.w = [w - self.lr * (2 / n) * g for w, g in zip(self.w, gw)]
            self.b -= self.lr * (2 / n) * sum(err)
            if abs(prev - loss) < self.tol:
                break
            prev = loss
        return self

    def fit_normal_equation(self, X, y):
        Xb = [[1.0] + r for r in X]
        Xt = transpose(Xb)
        theta = solve(matmul(Xt, Xb), matvec(Xt, y))
        self.b, self.w = theta[0], theta[1:]
        return self


class RidgeRegression:
    """
    RIDGE (L2 regularization)
    -------------------------
    J = sum (y_hat - y)^2 + lambda * ||w||^2

    Closed form: theta = (X^T X + lambda * I')^-1 X^T y, where I' has a 0 in the
    bias position (we never shrink the intercept).

    Effect: shrinks correlated coefficients toward each other, trading a little
    bias for a large drop in variance. It also makes X^T X invertible even when
    features are perfectly collinear or d > n.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        Xb = [[1.0] + r for r in X]
        d = len(Xb[0])
        Xt = transpose(Xb)
        A = matmul(Xt, Xb)
        for j in range(1, d):                    # skip index 0 = bias
            A[j][j] += self.alpha
        theta = solve(A, matvec(Xt, y))
        self.b, self.w = theta[0], theta[1:]
        return self

    def predict(self, X):
        return [dot(self.w, r) + self.b for r in X]


class LassoRegression:
    """
    LASSO (L1 regularization) via COORDINATE DESCENT
    ------------------------------------------------
    J = (1/2n) sum (y_hat - y)^2 + lambda * ||w||_1

    |w| is not differentiable at 0, so we optimize one coordinate at a time.
    Holding all other weights fixed, the minimizer of coordinate j is the
    soft-threshold of its least-squares update:

        rho_j = sum_i x_ij * (y_i - y_hat_i + w_j * x_ij)
        w_j   = S(rho_j / n, lambda) / (z_j / n),   z_j = sum_i x_ij^2
        S(a, k) = sign(a) * max(|a| - k, 0)          [soft-thresholding]

    Because S sets small coefficients exactly to 0, Lasso performs feature
    selection — this is the key behavioural difference from Ridge.
    """

    def __init__(self, alpha=0.1, epochs=500, tol=1e-8):
        self.alpha, self.epochs, self.tol = alpha, epochs, tol

    @staticmethod
    def _soft(a, k):
        if a > k:
            return a - k
        if a < -k:
            return a + k
        return 0.0

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.w, self.b = [0.0] * d, sum(y) / n
        z = [sum(r[j] ** 2 for r in X) for j in range(d)]
        for _ in range(self.epochs):
            max_change = 0.0
            pred = self.predict(X)
            for j in range(d):
                if z[j] < EPS:
                    continue
                rho = sum(X[i][j] * (y[i] - pred[i] + self.w[j] * X[i][j])
                          for i in range(n))
                new = self._soft(rho / n, self.alpha) / (z[j] / n)
                if new != self.w[j]:
                    delta = new - self.w[j]
                    for i in range(n):            # keep predictions in sync
                        pred[i] += delta * X[i][j]
                    max_change = max(max_change, abs(delta))
                    self.w[j] = new
            resid_mean = sum(y[i] - (pred[i] - self.b) for i in range(n)) / n
            self.b, pred = resid_mean, [p + (resid_mean - self.b) for p in pred]
            if max_change < self.tol:
                break
        return self

    def predict(self, X):
        return [dot(self.w, r) + self.b for r in X]


def sigmoid(z):
    """1 / (1 + e^-z), written in the numerically stable two-branch form."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class LogisticRegression:
    """
    LOGISTIC REGRESSION (binary)
    ----------------------------
    Model : p = sigma(z),  z = b + w . x,  sigma(z) = 1/(1+e^-z)
    We model log-odds as linear:  log(p / (1-p)) = b + w . x

    Loss (negative log-likelihood / binary cross-entropy):
        J = -(1/n) sum [ y log p + (1-y) log(1-p) ]

    The beautiful part: after substituting sigma, the derivative collapses to
    exactly the same shape as linear regression —
        dJ/dw_j = (1/n) sum (p_i - y_i) x_ij
        dJ/db   = (1/n) sum (p_i - y_i)
    because sigma'(z) = sigma(z)(1 - sigma(z)) cancels the log's denominator.

    Squared error is NOT used here: it is non-convex through the sigmoid and
    has vanishing gradients when predictions are confidently wrong.
    Optional L2 term adds lambda*w_j to the gradient.
    """

    def __init__(self, lr=0.1, epochs=1000, l2=0.0):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.history = []

    def predict_proba(self, X):
        return [sigmoid(dot(self.w, r) + self.b) for r in X]

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.w, self.b = [0.0] * d, 0.0
        for _ in range(self.epochs):
            p = self.predict_proba(X)
            err = [pi - ti for pi, ti in zip(p, y)]
            self.history.append(log_loss(y, p))
            gw = [0.0] * d
            for i, r in enumerate(X):
                e = err[i]
                for j in range(d):
                    gw[j] += e * r[j]
            self.w = [w - self.lr * (g / n + self.l2 * w)
                      for w, g in zip(self.w, gw)]
            self.b -= self.lr * sum(err) / n
        return self


class SoftmaxRegression:
    """
    MULTINOMIAL LOGISTIC REGRESSION (softmax)
    -----------------------------------------
    For K classes:  z_k = b_k + w_k . x,  p_k = e^{z_k} / sum_j e^{z_j}
    Loss: categorical cross-entropy  J = -(1/n) sum_i log p_{i, y_i}
    Gradient: dJ/dw_k = (1/n) sum_i (p_ik - 1{y_i = k}) x_i

    Subtracting max(z) before exponentiating prevents overflow and leaves the
    softmax unchanged (the constant cancels in numerator and denominator).
    """

    def __init__(self, lr=0.1, epochs=500, l2=0.0):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def _softmax(self, z):
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return [v / s for v in e]

    def predict_proba(self, X):
        return [self._softmax([dot(self.W[k], r) + self.b[k]
                               for k in range(self.K)]) for r in X]

    def predict(self, X):
        return [max(range(self.K), key=lambda k: p[k]) for p in self.predict_proba(X)]

    def fit(self, X, y):
        n, d = len(X), len(X[0])
        self.K = max(y) + 1
        self.W = zeros(self.K, d)
        self.b = [0.0] * self.K
        for _ in range(self.epochs):
            P = self.predict_proba(X)
            gW = zeros(self.K, d)
            gb = [0.0] * self.K
            for i, r in enumerate(X):
                for k in range(self.K):
                    err = P[i][k] - (1.0 if y[i] == k else 0.0)
                    gb[k] += err
                    for j in range(d):
                        gW[k][j] += err * r[j]
            for k in range(self.K):
                self.W[k] = [w - self.lr * (g / n + self.l2 * w)
                             for w, g in zip(self.W[k], gW[k])]
                self.b[k] -= self.lr * gb[k] / n
        return self


class Perceptron:
    """
    ROSENBLATT'S PERCEPTRON (1958) — the ancestor of neural networks
    ---------------------------------------------------------------
    Prediction: y_hat = sign(w . x + b), labels in {-1, +1}
    Rule: only update on a MISTAKE —
        if y_i (w . x_i + b) <= 0:  w <- w + eta*y_i*x_i ;  b <- b + eta*y_i

    Convergence theorem: if the data is linearly separable with margin gamma
    and ||x|| <= R, the algorithm makes at most (R/gamma)^2 mistakes — a bound
    independent of n and d. On non-separable data it never settles, which is
    precisely what the hinge-loss SVM below fixes.
    """

    def __init__(self, lr=1.0, epochs=100):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        """y in {0, 1} or {-1, +1}; converted internally to {-1, +1}."""
        yy = [1 if t > 0 else -1 for t in y]
        d = len(X[0])
        self.w, self.b = [0.0] * d, 0.0
        for _ in range(self.epochs):
            errors = 0
            for xi, ti in zip(X, yy):
                if ti * (dot(self.w, xi) + self.b) <= 0:
                    self.w = [w + self.lr * ti * x for w, x in zip(self.w, xi)]
                    self.b += self.lr * ti
                    errors += 1
            if errors == 0:                     # perfectly separated, stop
                break
        return self

    def predict(self, X):
        return [1 if dot(self.w, r) + self.b > 0 else 0 for r in X]


class LinearSVM:
    """
    SOFT-MARGIN LINEAR SVM (primal, subgradient descent)
    ----------------------------------------------------
    Objective:  J = (lambda/2)||w||^2 + (1/n) sum max(0, 1 - y_i (w.x_i + b))

    The hinge term is zero once a point sits on the correct side by a margin
    of at least 1, so only margin violators push on the boundary — those are
    the support vectors. Maximizing the geometric margin 2/||w|| is the same
    as minimizing ||w||^2, which is where the first term comes from.

    Subgradient:
        if y_i(w.x_i + b) >= 1:  dJ = lambda*w
        else:                    dJ = lambda*w - y_i*x_i ,  db = -y_i
    """

    def __init__(self, lr=0.01, epochs=1000, lambda_=0.01):
        self.lr, self.epochs, self.lambda_ = lr, epochs, lambda_

    def fit(self, X, y):
        yy = [1 if t > 0 else -1 for t in y]
        n, d = len(X), len(X[0])
        self.w, self.b = [0.0] * d, 0.0
        for ep in range(self.epochs):
            lr = self.lr / (1 + ep * 0.001)      # decaying step size
            gw = [self.lambda_ * w for w in self.w]
            gb = 0.0
            for xi, ti in zip(X, yy):
                if ti * (dot(self.w, xi) + self.b) < 1:
                    for j in range(d):
                        gw[j] -= ti * xi[j] / n
                    gb -= ti / n
            self.w = [w - lr * g for w, g in zip(self.w, gw)]
            self.b -= lr * gb
        return self

    def decision_function(self, X):
        return [dot(self.w, r) + self.b for r in X]

    def predict(self, X):
        return [1 if v > 0 else 0 for v in self.decision_function(X)]


class KernelSVM:
    """
    KERNEL SVM (dual form, simplified SMO)
    --------------------------------------
    Dual problem:
        max_a  sum a_i - 1/2 sum_i sum_j a_i a_j y_i y_j K(x_i, x_j)
        s.t.   0 <= a_i <= C   and   sum_i a_i y_i = 0
    Decision: f(x) = sum_i a_i y_i K(x_i, x) + b

    The KERNEL TRICK: K(u,v) is an inner product in a higher-dimensional space
    we never explicitly build. RBF K(u,v) = exp(-gamma ||u-v||^2) corresponds
    to an infinite-dimensional feature map, which is how a "linear" separator
    becomes a curved boundary in the original space.

    SMO optimizes two multipliers at a time (the equality constraint means you
    cannot move just one), clipping them to the box [L, H] so both stay in
    [0, C]. Points with a_i > 0 are the support vectors.
    """

    def __init__(self, C=1.0, kernel="rbf", gamma=0.5, degree=3,
                 tol=1e-3, max_passes=10, seed=0):
        self.C, self.kernel, self.gamma = C, kernel, gamma
        self.degree, self.tol, self.max_passes = degree, tol, max_passes
        self.rng = random.Random(seed)

    def _K(self, u, v):
        if self.kernel == "linear":
            return dot(u, v)
        if self.kernel == "poly":
            return (1.0 + dot(u, v)) ** self.degree
        sq = sum((a - b) ** 2 for a, b in zip(u, v))
        return math.exp(-self.gamma * sq)

    def fit(self, X, y):
        yy = [1.0 if t > 0 else -1.0 for t in y]
        n = len(X)
        self.X, self.y = X, yy
        self.a = [0.0] * n
        self.b = 0.0
        Kmat = [[self._K(X[i], X[j]) for j in range(n)] for i in range(n)]

        passes = 0
        while passes < self.max_passes:
            changed = 0
            for i in range(n):
                Ei = sum(self.a[k] * yy[k] * Kmat[k][i] for k in range(n)) + self.b - yy[i]
                if (yy[i] * Ei < -self.tol and self.a[i] < self.C) or \
                   (yy[i] * Ei > self.tol and self.a[i] > 0):
                    j = self.rng.choice([k for k in range(n) if k != i])
                    Ej = sum(self.a[k] * yy[k] * Kmat[k][j] for k in range(n)) + self.b - yy[j]
                    ai_old, aj_old = self.a[i], self.a[j]

                    if yy[i] != yy[j]:
                        L, H = max(0, aj_old - ai_old), min(self.C, self.C + aj_old - ai_old)
                    else:
                        L, H = max(0, ai_old + aj_old - self.C), min(self.C, ai_old + aj_old)
                    if L >= H:
                        continue

                    eta = 2 * Kmat[i][j] - Kmat[i][i] - Kmat[j][j]
                    if eta >= 0:
                        continue

                    aj = aj_old - yy[j] * (Ei - Ej) / eta
                    aj = min(H, max(L, aj))                       # clip to box
                    if abs(aj - aj_old) < 1e-8:
                        continue
                    ai = ai_old + yy[i] * yy[j] * (aj_old - aj)   # keep sum a_i y_i = 0
                    self.a[i], self.a[j] = ai, aj

                    b1 = self.b - Ei - yy[i] * (ai - ai_old) * Kmat[i][i] \
                        - yy[j] * (aj - aj_old) * Kmat[i][j]
                    b2 = self.b - Ej - yy[i] * (ai - ai_old) * Kmat[i][j] \
                        - yy[j] * (aj - aj_old) * Kmat[j][j]
                    if 0 < ai < self.C:
                        self.b = b1
                    elif 0 < aj < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    changed += 1
            passes = passes + 1 if changed == 0 else 0
        self.sv = [i for i in range(n) if self.a[i] > 1e-8]
        return self

    def decision_function(self, X):
        return [sum(self.a[i] * self.y[i] * self._K(self.X[i], r) for i in self.sv) + self.b
                for r in X]

    def predict(self, X):
        return [1 if v > 0 else 0 for v in self.decision_function(X)]


# =============================================================================
# 4. INSTANCE-BASED LEARNING
# =============================================================================
class KNNClassifier:
    """
    k-NEAREST NEIGHBOURS (classification)
    -------------------------------------
    No training phase at all — the data IS the model ("lazy learning").
    Predict: find the k closest training points under a distance metric,
    then take a (optionally distance-weighted) majority vote.

        d(u,v) = sqrt(sum (u_j - v_j)^2)          Euclidean
        weight_i = 1 / (d_i + eps)                 closer neighbours count more

    Bias-variance: small k -> jagged boundary, low bias, high variance;
    large k -> smooth boundary, higher bias. Distances are meaningless across
    features on different scales, so ALWAYS standardize first. Cost is O(n*d)
    per query, and in high dimensions all points become nearly equidistant
    (the curse of dimensionality).
    """

    def __init__(self, k=5, metric=euclidean, weighted=False):
        self.k, self.metric, self.weighted = k, metric, weighted

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def _neighbours(self, x):
        d = sorted(((self.metric(x, xi), yi) for xi, yi in zip(self.X, self.y)),
                   key=lambda t: t[0])
        return d[:self.k]

    def predict(self, X):
        out = []
        for x in X:
            votes = {}
            for dist, label in self._neighbours(x):
                w = 1.0 / (dist + EPS) if self.weighted else 1.0
                votes[label] = votes.get(label, 0.0) + w
            out.append(max(votes, key=votes.get))
        return out


class KNNRegressor:
    """Same idea, but average the neighbours' targets instead of voting."""

    def __init__(self, k=5, metric=euclidean, weighted=True):
        self.k, self.metric, self.weighted = k, metric, weighted

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def predict(self, X):
        out = []
        for x in X:
            nb = sorted(((self.metric(x, xi), yi) for xi, yi in zip(self.X, self.y)),
                        key=lambda t: t[0])[:self.k]
            if self.weighted:
                ws = [1.0 / (d + EPS) for d, _ in nb]
                out.append(sum(w * v for w, (_, v) in zip(ws, nb)) / sum(ws))
            else:
                out.append(sum(v for _, v in nb) / len(nb))
        return out


# =============================================================================
# 5. PROBABILISTIC MODELS
# =============================================================================
class GaussianNaiveBayes:
    """
    GAUSSIAN NAIVE BAYES
    --------------------
    Bayes' rule:      P(c | x) = P(x | c) P(c) / P(x)
    "Naive" assumption: features are conditionally independent given the class,
                      P(x | c) = prod_j P(x_j | c)
    Each P(x_j | c) is modelled as a normal density:

        P(x_j | c) = 1/sqrt(2*pi*var_jc) * exp( -(x_j - mu_jc)^2 / (2*var_jc) )

    We compare log-posteriors (P(x) is a constant across classes, so drop it):

        log P(c | x)  ∝  log P(c) + sum_j log P(x_j | c)

    Working in logs turns a product of tiny numbers into a stable sum.
    The independence assumption is nearly always false, yet the argmax is often
    still correct — which is why NB remains a strong, near-instant baseline.
    """

    def __init__(self, var_smoothing=1e-9):
        self.vs = var_smoothing

    def fit(self, X, y):
        self.classes = sorted(set(y))
        n, d = len(X), len(X[0])
        self.prior, self.mu, self.var = {}, {}, {}
        for c in self.classes:
            rows = [X[i] for i in range(n) if y[i] == c]
            self.prior[c] = len(rows) / n
            self.mu[c] = [sum(r[j] for r in rows) / len(rows) for j in range(d)]
            self.var[c] = [sum((r[j] - self.mu[c][j]) ** 2 for r in rows) / len(rows)
                           + self.vs for j in range(d)]
        return self

    def _joint_log(self, x):
        out = {}
        for c in self.classes:
            ll = math.log(self.prior[c])
            for j, v in enumerate(x):
                var = self.var[c][j]
                ll += -0.5 * math.log(2 * math.pi * var) \
                      - (v - self.mu[c][j]) ** 2 / (2 * var)
            out[c] = ll
        return out

    def predict(self, X):
        return [max(self._joint_log(x).items(), key=lambda kv: kv[1])[0] for x in X]

    def predict_proba(self, X):
        """Normalize log-scores with the log-sum-exp trick."""
        out = []
        for x in X:
            lj = self._joint_log(x)
            m = max(lj.values())
            e = {c: math.exp(v - m) for c, v in lj.items()}
            s = sum(e.values())
            out.append({c: v / s for c, v in e.items()})
        return out


class MultinomialNaiveBayes:
    """
    MULTINOMIAL NAIVE BAYES — the text-classification workhorse.
    Features are counts (e.g. word frequencies).

        P(word_j | c) = (count_jc + alpha) / (total_c + alpha * d)

    alpha is LAPLACE / additive SMOOTHING: without it, a single word never seen
    in class c would drive P(x|c) to exactly 0 and veto every other piece of
    evidence.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes = sorted(set(y))
        d = len(X[0])
        self.prior, self.loglik = {}, {}
        for c in self.classes:
            rows = [X[i] for i in range(len(X)) if y[i] == c]
            self.prior[c] = math.log(len(rows) / len(X))
            counts = [sum(r[j] for r in rows) + self.alpha for j in range(d)]
            total = sum(counts)
            self.loglik[c] = [math.log(v / total) for v in counts]
        return self

    def predict(self, X):
        out = []
        for x in X:
            best, best_c = -float("inf"), None
            for c in self.classes:
                s = self.prior[c] + sum(x[j] * self.loglik[c][j] for j in range(len(x)))
                if s > best:
                    best, best_c = s, c
            out.append(best_c)
        return out


class LinearDiscriminantAnalysis:
    """
    LDA (Fisher)
    ------------
    Like Gaussian NB, but keeps the FULL covariance and assumes every class
    shares the SAME covariance matrix Sigma (pooled within-class scatter).
    That shared Sigma makes the quadratic terms cancel, leaving a LINEAR
    discriminant:

        delta_c(x) = x^T Sigma^-1 mu_c - 1/2 mu_c^T Sigma^-1 mu_c + log P(c)

    Predict argmax_c delta_c(x). LDA also doubles as supervised dimensionality
    reduction: project onto the top eigenvectors of Sigma_within^-1 Sigma_between.
    """

    def __init__(self, reg=1e-6):
        self.reg = reg

    def fit(self, X, y):
        self.classes = sorted(set(y))
        n, d = len(X), len(X[0])
        self.mu, self.prior = {}, {}
        S = zeros(d, d)
        for c in self.classes:
            rows = [X[i] for i in range(n) if y[i] == c]
            self.prior[c] = len(rows) / n
            mu = [sum(r[j] for r in rows) / len(rows) for j in range(d)]
            self.mu[c] = mu
            for r in rows:                                   # within-class scatter
                dv = [r[j] - mu[j] for j in range(d)]
                for a in range(d):
                    for b in range(d):
                        S[a][b] += dv[a] * dv[b]
        for a in range(d):
            for b in range(d):
                S[a][b] /= (n - len(self.classes))
            S[a][a] += self.reg
        self.Sinv = inverse(S)
        return self

    def predict(self, X):
        out = []
        for x in X:
            best, bc = -float("inf"), None
            for c in self.classes:
                m = self.mu[c]
                Sm = matvec(self.Sinv, m)
                delta = dot(x, Sm) - 0.5 * dot(m, Sm) + math.log(self.prior[c])
                if delta > best:
                    best, bc = delta, c
            out.append(bc)
        return out


# =============================================================================
# 6. DECISION TREES (CART)
# =============================================================================
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature, self.threshold = feature, threshold
        self.left, self.right, self.value = left, right, value

    @property
    def is_leaf(self):
        return self.value is not None


def gini(y, weights=None):
    """
    Gini impurity = 1 - sum_c p_c^2
    = probability that two random draws from the node disagree. 0 = pure.
    """
    if weights is None:
        counts = {}
        for t in y:
            counts[t] = counts.get(t, 0) + 1
        n = len(y)
        return 1.0 - sum((c / n) ** 2 for c in counts.values())
    tot = sum(weights)
    counts = {}
    for t, w in zip(y, weights):
        counts[t] = counts.get(t, 0.0) + w
    return 1.0 - sum((c / tot) ** 2 for c in counts.values())


def entropy(y):
    """H = -sum_c p_c log2 p_c — expected bits needed to encode the label."""
    counts = {}
    for t in y:
        counts[t] = counts.get(t, 0) + 1
    n = len(y)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c)


def variance_criterion(y):
    """For regression: node impurity = variance of the targets."""
    if not y:
        return 0.0
    m = sum(y) / len(y)
    return sum((v - m) ** 2 for v in y) / len(y)


class DecisionTree:
    """
    CART — Classification And Regression Trees
    ------------------------------------------
    Greedily pick the (feature, threshold) split that maximizes INFORMATION GAIN:

        Gain = I(parent) - [ n_L/n * I(left) + n_R/n * I(right) ]

    where I is Gini impurity or entropy (classification) or variance (regression).
    Recurse on each child; stop at max_depth, min_samples_split, a pure node, or
    when no split yields positive gain. Leaves store the majority class or the
    mean target.

    Notes:
      * Finding the globally optimal tree is NP-hard — greedy is the practical
        compromise, which is why trees are unstable and benefit hugely from
        ensembling (see RandomForest / GradientBoosting below).
      * Trees need no feature scaling: splits depend only on ORDER, not units.
      * Candidate thresholds are midpoints between consecutive unique values.
    """

    def __init__(self, max_depth=8, min_samples_split=2, min_samples_leaf=1,
                 criterion="gini", n_features=None, seed=0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion            # 'gini' | 'entropy' | 'mse'
        self.n_features = n_features          # random subset per split (forests)
        self.rng = random.Random(seed)

    # ---- impurity dispatch
    def _impurity(self, y):
        if self.criterion == "gini":
            return gini(y)
        if self.criterion == "entropy":
            return entropy(y)
        return variance_criterion(y)

    def _leaf_value(self, y):
        if self.criterion == "mse":
            return sum(y) / len(y)
        counts = {}
        for t in y:
            counts[t] = counts.get(t, 0) + 1
        return max(counts, key=counts.get)

    def _best_split(self, X, y, feats):
        parent = self._impurity(y)
        n = len(y)
        best = (0.0, None, None)
        for f in feats:
            pairs = sorted(zip((r[f] for r in X), y))
            values = [p[0] for p in pairs]
            labels = [p[1] for p in pairs]
            for i in range(1, n):
                if values[i] == values[i - 1]:
                    continue
                if i < self.min_samples_leaf or n - i < self.min_samples_leaf:
                    continue
                left, right = labels[:i], labels[i:]
                gain = parent - (i / n) * self._impurity(left) \
                              - ((n - i) / n) * self._impurity(right)
                if gain > best[0]:
                    best = (gain, f, (values[i] + values[i - 1]) / 2)
        return best

    def _build(self, X, y, depth):
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or len(set(y)) == 1):
            return _Node(value=self._leaf_value(y))
        d = len(X[0])
        feats = (self.rng.sample(range(d), min(self.n_features, d))
                 if self.n_features else range(d))
        gain, f, thr = self._best_split(X, y, feats)
        if f is None or gain <= 1e-12:
            return _Node(value=self._leaf_value(y))
        li = [i for i in range(len(X)) if X[i][f] <= thr]
        ri = [i for i in range(len(X)) if X[i][f] > thr]
        if not li or not ri:
            return _Node(value=self._leaf_value(y))
        left = self._build([X[i] for i in li], [y[i] for i in li], depth + 1)
        right = self._build([X[i] for i in ri], [y[i] for i in ri], depth + 1)
        return _Node(f, thr, left, right)

    def fit(self, X, y):
        self.root = self._build(X, y, 0)
        return self

    def _walk(self, node, x):
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict(self, X):
        return [self._walk(self.root, x) for x in X]

    def print_tree(self, node=None, indent="", feature_names=None):
        node = node or self.root
        if node.is_leaf:
            v = node.value
            print(f"{indent}-> {v:.4f}" if isinstance(v, float) else f"{indent}-> class {v}")
            return
        name = feature_names[node.feature] if feature_names else f"x{node.feature}"
        print(f"{indent}[{name} <= {node.threshold:.4f}]")
        self.print_tree(node.left, indent + "  ", feature_names)
        self.print_tree(node.right, indent + "  ", feature_names)


class DecisionStump:
    """
    A depth-1 tree that supports SAMPLE WEIGHTS — the canonical weak learner
    for AdaBoost. Minimizes weighted classification error over all
    (feature, threshold, polarity) triples. Labels are {-1, +1}.
    """

    def fit(self, X, y, w):
        n, d = len(X), len(X[0])
        best_err, self.f, self.thr, self.pol = float("inf"), 0, 0.0, 1
        for f in range(d):
            vals = sorted(set(r[f] for r in X))
            thresholds = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)] or vals
            for t in thresholds:
                for pol in (1, -1):
                    pred = [pol if X[i][f] <= t else -pol for i in range(n)]
                    err = sum(w[i] for i in range(n) if pred[i] != y[i])
                    if err < best_err:
                        best_err, self.f, self.thr, self.pol = err, f, t, pol
        return best_err

    def predict(self, X):
        return [self.pol if r[self.f] <= self.thr else -self.pol for r in X]


# =============================================================================
# 7. ENSEMBLE METHODS
# =============================================================================
class RandomForest:
    """
    RANDOM FOREST = bagging + random feature subspaces
    --------------------------------------------------
    1. BOOTSTRAP: each tree trains on n samples drawn WITH replacement
       (~63.2% unique rows; the left-out 36.8% are the "out-of-bag" set,
        since P(a given row is never picked) = (1 - 1/n)^n -> 1/e).
    2. FEATURE SUBSAMPLING: at every split consider only m random features
       (m ≈ sqrt(d) for classification, d/3 for regression).
    3. AGGREGATE: majority vote / mean.

    Why it works: for B estimators with pairwise correlation rho and variance
    sigma^2, the ensemble variance is

        rho*sigma^2 + (1-rho)/B * sigma^2

    Averaging kills the second term; step 2 exists purely to shrink rho, which
    is what plain bagging cannot do. Bias stays roughly that of one deep tree,
    so trees are grown deep and left unpruned.
    """

    def __init__(self, n_estimators=25, max_depth=8, min_samples_split=2,
                 criterion="gini", n_features=None, seed=0):
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.min_samples_split, self.criterion = min_samples_split, criterion
        self.n_features, self.seed = n_features, seed

    def fit(self, X, y):
        rng = random.Random(self.seed)
        n, d = len(X), len(X[0])
        m = self.n_features or (max(1, int(math.sqrt(d)))
                                if self.criterion != "mse" else max(1, d // 3))
        self.trees = []
        for b in range(self.n_estimators):
            idx = [rng.randrange(n) for _ in range(n)]        # bootstrap sample
            t = DecisionTree(self.max_depth, self.min_samples_split,
                             criterion=self.criterion, n_features=m,
                             seed=rng.randrange(10 ** 6))
            t.fit([X[i] for i in idx], [y[i] for i in idx])
            self.trees.append(t)
        return self

    def predict(self, X):
        preds = [t.predict(X) for t in self.trees]
        out = []
        for i in range(len(X)):
            col = [p[i] for p in preds]
            if self.criterion == "mse":
                out.append(sum(col) / len(col))
            else:
                counts = {}
                for v in col:
                    counts[v] = counts.get(v, 0) + 1
                out.append(max(counts, key=counts.get))
        return out


class AdaBoost:
    """
    ADABOOST (Freund & Schapire) — adaptive boosting with decision stumps
    --------------------------------------------------------------------
    Maintain a weight distribution over samples, initially w_i = 1/n. For each
    round t:
        1. Fit a weak learner minimizing the WEIGHTED error
               eps_t = sum_i w_i * 1{h_t(x_i) != y_i}
        2. Its vote strength is
               alpha_t = 1/2 * ln((1 - eps_t) / eps_t)
           (derived by minimizing the exponential loss sum exp(-y_i F(x_i));
            eps -> 0 gives alpha -> inf, eps = 0.5 gives alpha = 0 — a coin flip
            earns no say.)
        3. Reweight and renormalize:
               w_i <- w_i * exp(-alpha_t * y_i * h_t(x_i)) / Z_t
           Misclassified points get heavier, so the next learner focuses there.

    Final model: H(x) = sign( sum_t alpha_t h_t(x) ).
    Training error falls exponentially while test error often keeps improving
    even after training error hits zero — boosting keeps widening the margin.
    """

    def __init__(self, n_estimators=20):
        self.n_estimators = n_estimators

    def fit(self, X, y):
        yy = [1 if t > 0 else -1 for t in y]
        n = len(X)
        w = [1.0 / n] * n
        self.learners, self.alphas = [], []
        for _ in range(self.n_estimators):
            stump = DecisionStump()
            err = stump.fit(X, yy, w)
            err = min(max(err, 1e-10), 1 - 1e-10)
            alpha = 0.5 * math.log((1 - err) / err)
            pred = stump.predict(X)
            w = [wi * math.exp(-alpha * yi * pi) for wi, yi, pi in zip(w, yy, pred)]
            z = sum(w)
            w = [wi / z for wi in w]
            self.learners.append(stump)
            self.alphas.append(alpha)
            if err < 1e-10:
                break
        return self

    def decision_function(self, X):
        agg = [0.0] * len(X)
        for a, h in zip(self.alphas, self.learners):
            for i, p in enumerate(h.predict(X)):
                agg[i] += a * p
        return agg

    def predict(self, X):
        return [1 if v > 0 else 0 for v in self.decision_function(X)]


class GradientBoostingRegressor:
    """
    GRADIENT BOOSTING (regression, squared loss)
    --------------------------------------------
    Boosting as GRADIENT DESCENT IN FUNCTION SPACE. We build an additive model

        F_m(x) = F_{m-1}(x) + nu * h_m(x)

    where h_m is fit to the NEGATIVE GRADIENT of the loss w.r.t. the current
    predictions ("pseudo-residuals"). For L = 1/2 (y - F)^2:

        -dL/dF = y - F      -> the pseudo-residual is simply the residual

    So each new shallow tree learns what the ensemble still gets wrong.
    nu (learning rate) shrinks each step; small nu + more trees generalizes
    better than large nu + few trees. F_0 is initialized to the mean of y.
    Swap in a different loss and only the residual formula changes.
    """

    def __init__(self, n_estimators=60, lr=0.1, max_depth=3, seed=0):
        self.n_estimators, self.lr, self.max_depth, self.seed = \
            n_estimators, lr, max_depth, seed

    def fit(self, X, y):
        self.F0 = sum(y) / len(y)
        F = [self.F0] * len(y)
        self.trees = []
        for m in range(self.n_estimators):
            residual = [yi - fi for yi, fi in zip(y, F)]      # negative gradient
            t = DecisionTree(self.max_depth, criterion="mse", seed=self.seed + m)
            t.fit(X, residual)
            upd = t.predict(X)
            F = [f + self.lr * u for f, u in zip(F, upd)]
            self.trees.append(t)
        return self

    def predict(self, X):
        out = [self.F0] * len(X)
        for t in self.trees:
            for i, v in enumerate(t.predict(X)):
                out[i] += self.lr * v
        return out


class GradientBoostingClassifier:
    """
    GRADIENT BOOSTING (binary, logistic loss)
    -----------------------------------------
    Work in LOG-ODDS space: F(x) = log(p / (1-p)), p = sigmoid(F).
    Loss L = -[y log p + (1-y) log(1-p)]; its negative gradient is again clean:

        -dL/dF = y - sigmoid(F)

    Initialize F_0 = log(p_bar / (1 - p_bar)) with p_bar the base rate, fit each
    tree to (y - p), and add nu * tree. Predict with sigmoid(F_M(x)).
    """

    def __init__(self, n_estimators=60, lr=0.1, max_depth=3, seed=0):
        self.n_estimators, self.lr, self.max_depth, self.seed = \
            n_estimators, lr, max_depth, seed

    def fit(self, X, y):
        p = min(max(sum(y) / len(y), 1e-6), 1 - 1e-6)
        self.F0 = math.log(p / (1 - p))
        F = [self.F0] * len(y)
        self.trees = []
        for m in range(self.n_estimators):
            resid = [yi - sigmoid(f) for yi, f in zip(y, F)]
            t = DecisionTree(self.max_depth, criterion="mse", seed=self.seed + m)
            t.fit(X, resid)
            upd = t.predict(X)
            F = [f + self.lr * u for f, u in zip(F, upd)]
            self.trees.append(t)
        return self

    def _raw(self, X):
        out = [self.F0] * len(X)
        for t in self.trees:
            for i, v in enumerate(t.predict(X)):
                out[i] += self.lr * v
        return out

    def predict_proba(self, X):
        return [sigmoid(v) for v in self._raw(X)]

    def predict(self, X):
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]


# =============================================================================
# 8. NEURAL NETWORK
# =============================================================================
class MLP:
    """
    MULTILAYER PERCEPTRON with BACKPROPAGATION
    ------------------------------------------
    FORWARD PASS, layer l:
        z^l = W^l a^{l-1} + b^l ,   a^l = f(z^l) ,   a^0 = x

    BACKWARD PASS — backprop is just the chain rule applied right-to-left,
    reusing each layer's error term instead of recomputing it:

        output layer:  delta^L = dL/da^L * f'(z^L)
        hidden layer:  delta^l = (W^{l+1})^T delta^{l+1} * f'(z^l)
        gradients   :  dL/dW^l = delta^l (a^{l-1})^T ,   dL/db^l = delta^l

    SPECIAL CASE: with softmax output + cross-entropy loss, the Jacobian of
    softmax and the derivative of log cancel exactly, leaving

        delta^L = a^L - y_onehot

    which is why that pairing is standard (and numerically stable).

    INITIALIZATION matters: He (ReLU) scales weights by sqrt(2/fan_in), Xavier
    (tanh/sigmoid) by sqrt(1/fan_in), keeping activation variance stable across
    depth so signals neither explode nor vanish.

    Includes mini-batch SGD with momentum:  v <- beta*v + g ;  W <- W - lr*v
    """

    def __init__(self, layers, activation="relu", output="softmax",
                 lr=0.05, epochs=200, batch_size=32, momentum=0.9, seed=0):
        self.layers, self.activation, self.output = layers, activation, output
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.momentum = momentum
        self.rng = random.Random(seed)
        self._init_weights()
        self.history = []

    def _init_weights(self):
        self.W, self.b, self.vW, self.vb = [], [], [], []
        for i in range(len(self.layers) - 1):
            fan_in, fan_out = self.layers[i], self.layers[i + 1]
            s = math.sqrt(2.0 / fan_in) if self.activation == "relu" \
                else math.sqrt(1.0 / fan_in)
            self.W.append([[self.rng.gauss(0, s) for _ in range(fan_in)]
                           for _ in range(fan_out)])
            self.b.append([0.0] * fan_out)
            self.vW.append(zeros(fan_out, fan_in))
            self.vb.append([0.0] * fan_out)

    def _act(self, z):
        if self.activation == "relu":
            return [v if v > 0 else 0.0 for v in z]
        if self.activation == "tanh":
            return [math.tanh(v) for v in z]
        return [sigmoid(v) for v in z]

    def _act_grad(self, a):
        """Derivatives expressed in terms of the ACTIVATION (cheaper)."""
        if self.activation == "relu":
            return [1.0 if v > 0 else 0.0 for v in a]
        if self.activation == "tanh":
            return [1 - v * v for v in a]
        return [v * (1 - v) for v in a]

    @staticmethod
    def _softmax(z):
        m = max(z)
        e = [math.exp(v - m) for v in z]
        s = sum(e)
        return [v / s for v in e]

    def _forward(self, x):
        acts = [x]
        a = x
        L = len(self.W)
        for l in range(L):
            z = [dot(self.W[l][k], a) + self.b[l][k] for k in range(len(self.b[l]))]
            if l == L - 1:
                a = self._softmax(z) if self.output == "softmax" else \
                    ([sigmoid(v) for v in z] if self.output == "sigmoid" else z)
            else:
                a = self._act(z)
            acts.append(a)
        return acts

    def _backward(self, acts, target):
        L = len(self.W)
        gW = [zeros(len(self.b[l]), len(self.W[l][0])) for l in range(L)]
        gb = [[0.0] * len(self.b[l]) for l in range(L)]
        # output delta: softmax+CE, sigmoid+BCE and linear+MSE all reduce to (a - y)
        delta = [a - t for a, t in zip(acts[-1], target)]
        for l in range(L - 1, -1, -1):
            prev = acts[l]
            for k in range(len(delta)):
                gb[l][k] += delta[k]
                dk = delta[k]
                row = gW[l][k]
                for j in range(len(prev)):
                    row[j] += dk * prev[j]
            if l > 0:
                back = [sum(self.W[l][k][j] * delta[k] for k in range(len(delta)))
                        for j in range(len(prev))]
                delta = [b * g for b, g in zip(back, self._act_grad(prev))]
        return gW, gb

    def fit(self, X, Y):
        """Y: one-hot rows for softmax, [[v]] rows for regression/sigmoid."""
        n = len(X)
        idx = list(range(n))
        for ep in range(self.epochs):
            self.rng.shuffle(idx)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = idx[start:start + self.batch_size]
                accW = [zeros(len(self.b[l]), len(self.W[l][0]))
                        for l in range(len(self.W))]
                accb = [[0.0] * len(self.b[l]) for l in range(len(self.W))]
                for i in batch:
                    acts = self._forward(X[i])
                    if self.output == "softmax":
                        epoch_loss -= math.log(max(
                            acts[-1][max(range(len(Y[i])), key=lambda k: Y[i][k])], 1e-15))
                    else:
                        epoch_loss += sum((a - t) ** 2 for a, t in zip(acts[-1], Y[i]))
                    gW, gb = self._backward(acts, Y[i])
                    for l in range(len(self.W)):
                        for k in range(len(gb[l])):
                            accb[l][k] += gb[l][k]
                            for j in range(len(gW[l][k])):
                                accW[l][k][j] += gW[l][k][j]
                m = len(batch)
                for l in range(len(self.W)):
                    for k in range(len(self.b[l])):
                        self.vb[l][k] = self.momentum * self.vb[l][k] + accb[l][k] / m
                        self.b[l][k] -= self.lr * self.vb[l][k]
                        for j in range(len(self.W[l][k])):
                            self.vW[l][k][j] = self.momentum * self.vW[l][k][j] \
                                + accW[l][k][j] / m
                            self.W[l][k][j] -= self.lr * self.vW[l][k][j]
            self.history.append(epoch_loss / n)
        return self

    def predict_raw(self, X):
        return [self._forward(x)[-1] for x in X]

    def predict(self, X):
        out = self.predict_raw(X)
        if self.output == "softmax":
            return [max(range(len(o)), key=lambda k: o[k]) for o in out]
        if self.output == "sigmoid":
            return [1 if o[0] >= 0.5 else 0 for o in out]
        return [o[0] for o in out]


# =============================================================================
# 9. UNSUPERVISED LEARNING
# =============================================================================
class KMeans:
    """
    K-MEANS (Lloyd's algorithm)
    ---------------------------
    Minimize within-cluster sum of squares (inertia):

        J = sum_k sum_{x in C_k} ||x - mu_k||^2

    Alternate two steps, each of which can only DECREASE J:
      E-step (assign) : c_i = argmin_k ||x_i - mu_k||^2
      M-step (update) : mu_k = mean of points assigned to k
                        (the mean is exactly the minimizer of squared distance)
    J is bounded below and there are finitely many assignments, so it converges
    — but only to a LOCAL optimum, hence multiple restarts (n_init).

    K-MEANS++ INITIALIZATION: pick the first centre uniformly, then each next
    centre with probability proportional to D(x)^2, its squared distance to the
    nearest chosen centre. This spreads seeds out and gives an O(log k)
    approximation guarantee in expectation.

    Assumes roughly spherical, similarly sized clusters; k must be chosen in
    advance (use the elbow of inertia or silhouette score).
    """

    def __init__(self, k=3, max_iter=300, n_init=10, tol=1e-8, seed=0):
        self.k, self.max_iter, self.n_init, self.tol = k, max_iter, n_init, tol
        self.seed = seed

    def _init_pp(self, X, rng):
        centers = [list(X[rng.randrange(len(X))])]
        for _ in range(self.k - 1):
            d2 = [min(sum((a - b) ** 2 for a, b in zip(x, c)) for c in centers)
                  for x in X]
            total = sum(d2)
            if total < EPS:
                centers.append(list(X[rng.randrange(len(X))]))
                continue
            r = rng.random() * total
            acc = 0.0
            for i, v in enumerate(d2):
                acc += v
                if acc >= r:
                    centers.append(list(X[i]))
                    break
        return centers

    def _run(self, X, rng):
        centers = self._init_pp(X, rng)
        labels = [0] * len(X)
        for _ in range(self.max_iter):
            # E-step
            new_labels = [min(range(self.k),
                              key=lambda k: sum((a - b) ** 2
                                                for a, b in zip(x, centers[k])))
                          for x in X]
            # M-step
            new_centers = []
            for k in range(self.k):
                pts = [X[i] for i in range(len(X)) if new_labels[i] == k]
                if not pts:                                   # empty cluster
                    new_centers.append(list(X[rng.randrange(len(X))]))
                else:
                    new_centers.append([sum(p[j] for p in pts) / len(pts)
                                        for j in range(len(X[0]))])
            shift = sum(euclidean(a, b) for a, b in zip(centers, new_centers))
            centers, labels = new_centers, new_labels
            if shift < self.tol:
                break
        inertia = sum(sum((a - b) ** 2 for a, b in zip(X[i], centers[labels[i]]))
                      for i in range(len(X)))
        return centers, labels, inertia

    def fit(self, X, y=None):
        rng = random.Random(self.seed)
        best = None
        for _ in range(self.n_init):
            c, l, j = self._run(X, rng)
            if best is None or j < best[2]:
                best = (c, l, j)
        self.centers, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        return [min(range(self.k),
                    key=lambda k: sum((a - b) ** 2 for a, b in zip(x, self.centers[k])))
                for x in X]


class DBSCAN:
    """
    DBSCAN — Density-Based Spatial Clustering of Applications with Noise
    --------------------------------------------------------------------
    Two parameters: eps (neighbourhood radius) and min_samples.
      * CORE point      : has >= min_samples neighbours within eps
      * BORDER point    : within eps of a core point, but not core itself
      * NOISE           : neither (labelled -1)
    Clusters grow by DENSITY-REACHABILITY: start at an unvisited core point and
    repeatedly absorb the eps-neighbourhoods of every core point discovered.

    Advantages over k-means: finds arbitrarily shaped clusters, needs no k, and
    labels outliers explicitly. Weakness: one global eps struggles when clusters
    have very different densities.
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps, self.min_samples = eps, min_samples

    def fit(self, X, y=None):
        n = len(X)
        labels = [None] * n
        neigh = [[j for j in range(n) if euclidean(X[i], X[j]) <= self.eps]
                 for i in range(n)]
        cid = 0
        for i in range(n):
            if labels[i] is not None:
                continue
            if len(neigh[i]) < self.min_samples:
                labels[i] = -1                                # tentative noise
                continue
            labels[i] = cid
            queue = list(neigh[i])
            while queue:
                j = queue.pop()
                if labels[j] == -1:
                    labels[j] = cid                           # border point
                if labels[j] is not None:
                    continue
                labels[j] = cid
                if len(neigh[j]) >= self.min_samples:         # core -> expand
                    queue.extend(neigh[j])
            cid += 1
        self.labels_ = labels
        self.n_clusters_ = cid
        return self


class AgglomerativeClustering:
    """
    HIERARCHICAL AGGLOMERATIVE CLUSTERING (bottom-up)
    -------------------------------------------------
    Start with every point as its own cluster, then repeatedly merge the two
    closest clusters until n_clusters remain. LINKAGE defines "closest":

        single   : min distance between members   (chains, finds elongated shapes)
        complete : max distance between members   (compact, equal-diameter blobs)
        average  : mean pairwise distance         (a compromise)

    The merge order forms a dendrogram, so you can cut at any height instead of
    committing to k up front. Cost here is O(n^3) with the naive scan.
    """

    def __init__(self, n_clusters=3, linkage="average"):
        self.n_clusters, self.linkage = n_clusters, linkage

    def _dist(self, A, B, X):
        ds = [euclidean(X[i], X[j]) for i in A for j in B]
        if self.linkage == "single":
            return min(ds)
        if self.linkage == "complete":
            return max(ds)
        return sum(ds) / len(ds)

    def fit(self, X, y=None):
        clusters = [[i] for i in range(len(X))]
        self.merges = []
        while len(clusters) > self.n_clusters:
            best, pair = float("inf"), (0, 1)
            for a in range(len(clusters)):
                for b in range(a + 1, len(clusters)):
                    d = self._dist(clusters[a], clusters[b], X)
                    if d < best:
                        best, pair = d, (a, b)
            a, b = pair
            self.merges.append((len(clusters[a]), len(clusters[b]), best))
            clusters[a] = clusters[a] + clusters[b]
            clusters.pop(b)
        labels = [0] * len(X)
        for cid, members in enumerate(clusters):
            for i in members:
                labels[i] = cid
        self.labels_ = labels
        return self


class PCA:
    """
    PRINCIPAL COMPONENT ANALYSIS
    ----------------------------
    Find orthogonal directions of MAXIMUM VARIANCE.
      1. Centre the data: X_c = X - mean
      2. Covariance matrix  C = (1/(n-1)) X_c^T X_c   (symmetric, d x d)
      3. Eigendecompose C = V L V^T (Jacobi rotations above). Eigenvector v_k
         is the k-th principal direction; eigenvalue lambda_k is the variance
         captured along it.
      4. Project: Z = X_c V_k  (keep the top k eigenvectors)

    Equivalently, PCA minimizes squared reconstruction error — the best rank-k
    linear approximation of the data (Eckart-Young). Explained variance ratio
    is lambda_k / sum(lambda). Because it maximizes VARIANCE, features must be
    standardized first or whichever feature has the biggest units will dominate.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X, y=None):
        n, d = len(X), len(X[0])
        self.mean = [sum(r[j] for r in X) / n for j in range(d)]
        Xc = [[r[j] - self.mean[j] for j in range(d)] for r in X]
        C = zeros(d, d)
        for r in Xc:
            for a in range(d):
                for b in range(d):
                    C[a][b] += r[a] * r[b]
        for a in range(d):
            for b in range(d):
                C[a][b] /= (n - 1)
        vals, vecs = jacobi_eigen(C)
        self.explained_variance_ = vals[:self.n_components]
        total = sum(abs(v) for v in vals)
        self.explained_variance_ratio_ = [v / total for v in self.explained_variance_]
        self.components_ = [[vecs[r][k] for r in range(d)]          # rows = PCs
                            for k in range(self.n_components)]
        return self

    def transform(self, X):
        Xc = [[r[j] - self.mean[j] for j in range(len(r))] for r in X]
        return [[dot(pc, r) for pc in self.components_] for r in Xc]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        """Reconstruct in the original space: X ≈ Z V_k^T + mean."""
        return [[sum(z[k] * self.components_[k][j] for k in range(len(z))) + self.mean[j]
                 for j in range(len(self.mean))] for z in Z]


class GaussianMixture:
    """
    GAUSSIAN MIXTURE MODEL fitted by EXPECTATION-MAXIMIZATION
    ---------------------------------------------------------
    Density:  p(x) = sum_k pi_k * N(x | mu_k, Sigma_k),   sum_k pi_k = 1

    E-STEP — responsibilities (soft assignments, the posterior over components):
        gamma_ik = pi_k N(x_i | mu_k, Sigma_k) / sum_j pi_j N(x_i | mu_j, Sigma_j)

    M-STEP — weighted MLE updates, with N_k = sum_i gamma_ik:
        pi_k    = N_k / n
        mu_k    = (1/N_k) sum_i gamma_ik x_i
        Sigma_k = (1/N_k) sum_i gamma_ik (x_i - mu_k)(x_i - mu_k)^T

    EM never decreases the log-likelihood (it maximizes a lower bound / ELBO),
    but converges only to a local optimum. GMM is the soft, elliptical
    generalization of k-means: fix Sigma = sigma^2 I and let sigma -> 0 and the
    responsibilities become hard 0/1 assignments — exactly k-means.
    A small ridge on Sigma prevents a component collapsing onto one point
    (which would send the likelihood to infinity).
    """

    def __init__(self, n_components=3, max_iter=100, tol=1e-6, reg=1e-6, seed=0):
        self.k, self.max_iter, self.tol, self.reg = n_components, max_iter, tol, reg
        self.seed = seed

    def _gauss_pdf(self, x, mu, Sinv, det, d):
        dv = [a - b for a, b in zip(x, mu)]
        q = dot(dv, matvec(Sinv, dv))                       # Mahalanobis distance^2
        return math.exp(-0.5 * q) / math.sqrt(((2 * math.pi) ** d) * max(det, EPS))

    def fit(self, X, y=None):
        n, d = len(X), len(X[0])
        km = KMeans(self.k, n_init=3, seed=self.seed).fit(X)   # sensible init
        self.mu = [c[:] for c in km.centers]
        self.pi = [1.0 / self.k] * self.k
        var = [sum((r[j] - km.centers[0][j]) ** 2 for r in X) / n for j in range(d)]
        self.Sigma = [[[var[a] if a == b else 0.0 for b in range(d)] for a in range(d)]
                      for _ in range(self.k)]

        prev_ll = -float("inf")
        for _ in range(self.max_iter):
            Sinv = [inverse([[s[a][b] + (self.reg if a == b else 0.0)
                              for b in range(d)] for a in range(d)]) for s in self.Sigma]
            dets = [determinant([[s[a][b] + (self.reg if a == b else 0.0)
                                  for b in range(d)] for a in range(d)])
                    for s in self.Sigma]

            # ---- E-step
            gamma, ll = [], 0.0
            for x in X:
                probs = [self.pi[k] * self._gauss_pdf(x, self.mu[k], Sinv[k], dets[k], d)
                         for k in range(self.k)]
                s = sum(probs) + EPS
                ll += math.log(s)
                gamma.append([p / s for p in probs])

            # ---- M-step
            for k in range(self.k):
                Nk = sum(g[k] for g in gamma) + EPS
                self.pi[k] = Nk / n
                self.mu[k] = [sum(gamma[i][k] * X[i][j] for i in range(n)) / Nk
                              for j in range(d)]
                S = zeros(d, d)
                for i in range(n):
                    dv = [X[i][j] - self.mu[k][j] for j in range(d)]
                    g = gamma[i][k]
                    for a in range(d):
                        for b in range(d):
                            S[a][b] += g * dv[a] * dv[b]
                self.Sigma[k] = [[S[a][b] / Nk for b in range(d)] for a in range(d)]

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        self.log_likelihood_ = prev_ll
        self.labels_ = self.predict(X)
        return self

    def predict_proba(self, X):
        d = len(X[0])
        Sinv = [inverse([[s[a][b] + (self.reg if a == b else 0.0)
                          for b in range(d)] for a in range(d)]) for s in self.Sigma]
        dets = [determinant([[s[a][b] + (self.reg if a == b else 0.0)
                              for b in range(d)] for a in range(d)]) for s in self.Sigma]
        out = []
        for x in X:
            p = [self.pi[k] * self._gauss_pdf(x, self.mu[k], Sinv[k], dets[k], d)
                 for k in range(self.k)]
            s = sum(p) + EPS
            out.append([v / s for v in p])
        return out

    def predict(self, X):
        return [max(range(self.k), key=lambda k: p[k]) for p in self.predict_proba(X)]


# =============================================================================
# 10. SYNTHETIC DATASETS
# =============================================================================
def make_regression(n=400, d=3, noise=2.0, seed=1):
    """y = b + w.x + gaussian noise, with known ground-truth coefficients."""
    rng = random.Random(seed)
    w = [2.0, -1.5, 0.8][:d] + [rng.uniform(-2, 2) for _ in range(max(0, d - 3))]
    b = 3.5
    X, y = [], []
    for _ in range(n):
        row = [rng.uniform(-5, 5) for _ in range(d)]
        y.append(b + dot(w, row) + rng.gauss(0, noise))
        X.append(row)
    return X, y, w, b


def make_blobs(n=300, centers=3, d=2, spread=1.0, seed=2):
    """Isotropic Gaussian blobs — the friendly case for k-means / GMM.
    Centres are rejection-sampled so they stay well separated."""
    rng = random.Random(seed)
    cs = []
    while len(cs) < centers:
        cand = [rng.uniform(-10, 10) for _ in range(d)]
        if all(euclidean(cand, c) > 7 * spread for c in cs):
            cs.append(cand)
    X, y = [], []
    for i in range(n):
        c = i % centers
        X.append([rng.gauss(cs[c][j], spread) for j in range(d)])
        y.append(c)
    return X, y


def make_moons(n=300, noise=0.15, seed=3):
    """Two interleaving half-circles — NOT linearly separable. Kernel/tree food."""
    rng = random.Random(seed)
    X, y = [], []
    for i in range(n):
        t = math.pi * rng.random()
        if i % 2 == 0:
            X.append([math.cos(t) + rng.gauss(0, noise),
                      math.sin(t) + rng.gauss(0, noise)])
            y.append(0)
        else:
            X.append([1 - math.cos(t) + rng.gauss(0, noise),
                      0.5 - math.sin(t) + rng.gauss(0, noise)])
            y.append(1)
    return X, y


def make_classification(n=400, d=4, seed=4):
    """Linearly separable-ish binary data with label noise."""
    rng = random.Random(seed)
    w = [rng.uniform(-2, 2) for _ in range(d)]
    X, y = [], []
    for _ in range(n):
        row = [rng.gauss(0, 1) for _ in range(d)]
        z = dot(w, row)
        p = sigmoid(z)
        X.append(row)
        y.append(1 if rng.random() < p else 0)
    return X, y


# =============================================================================
# 11. DEMO
# =============================================================================
def _hdr(t):
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def _reg_report(name, model, Xtr, ytr, Xte, yte):
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    print(f"  {name:<34} train R2 {r2_score(ytr, ptr):6.4f} | "
          f"test R2 {r2_score(yte, pte):6.4f} | test RMSE {rmse(yte, pte):7.4f}")


def _clf_report(name, model, Xtr, ytr, Xte, yte):
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    p, r, f1 = precision_recall_f1(yte, pte)
    print(f"  {name:<34} train acc {accuracy(ytr, ptr):6.4f} | "
          f"test acc {accuracy(yte, pte):6.4f} | P {p:.3f} R {r:.3f} F1 {f1:.3f}")


def demo_regression():
    _hdr("REGRESSION  —  y = 3.5 + 2*x1 - 1.5*x2 + 0.8*x3 + N(0, 2)")
    X, y, tw, tb = make_regression(400, 3)
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.2, seed=1)
    sc = Standardizer().fit(Xtr)
    Str, Ste = sc.transform(Xtr), sc.transform(Xte)

    gd = LinearRegression(lr=0.05, epochs=4000).fit(Str, ytr)
    _reg_report("LinearRegression (grad descent)", gd, Str, ytr, Ste, yte)

    ne = LinearRegression().fit_normal_equation(Xtr, ytr)
    _reg_report("LinearRegression (normal eq)", ne, Xtr, ytr, Xte, yte)
    print(f"      true w = {tw}, b = {tb}")
    print(f"      est. w = {[round(v, 3) for v in ne.w]}, b = {round(ne.b, 3)}")

    _reg_report("Ridge (alpha=1.0)", RidgeRegression(1.0).fit(Xtr, ytr),
                Xtr, ytr, Xte, yte)
    la = LassoRegression(alpha=0.5, epochs=300).fit(Str, ytr)
    _reg_report("Lasso (alpha=0.5)", la, Str, ytr, Ste, yte)
    print(f"      lasso weights (0 = feature dropped): {[round(v, 3) for v in la.w]}")

    _reg_report("KNNRegressor (k=7, distance wt)",
                KNNRegressor(7).fit(Str, ytr), Str, ytr, Ste, yte)
    _reg_report("DecisionTree (depth 5, MSE)",
                DecisionTree(5, criterion="mse").fit(Xtr, ytr), Xtr, ytr, Xte, yte)
    _reg_report("RandomForest (30 trees)",
                RandomForest(30, 8, criterion="mse").fit(Xtr, ytr), Xtr, ytr, Xte, yte)
    _reg_report("GradientBoosting (80 x depth3)",
                GradientBoostingRegressor(80, 0.1, 3).fit(Xtr, ytr), Xtr, ytr, Xte, yte)

    cv = cross_val_score(lambda: RidgeRegression(1.0), X, y, k=5, metric=r2_score)
    print(f"\n  5-fold CV R2 for Ridge: {[round(s, 4) for s in cv]}")
    print(f"  mean {sum(cv)/len(cv):.4f}")


def demo_linear_classification():
    _hdr("BINARY CLASSIFICATION  —  linearly separable-ish data")
    X, y = make_classification(400, 4)
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.25, seed=2, stratify=True)
    sc = Standardizer().fit(Xtr)
    Str, Ste = sc.transform(Xtr), sc.transform(Xte)

    lr = LogisticRegression(lr=0.5, epochs=800).fit(Str, ytr)
    _clf_report("LogisticRegression", lr, Str, ytr, Ste, yte)
    print(f"      test log-loss {log_loss(yte, lr.predict_proba(Ste)):.4f} | "
          f"ROC-AUC {roc_auc(yte, lr.predict_proba(Ste)):.4f}")

    _clf_report("Perceptron", Perceptron(1.0, 60).fit(Str, ytr), Str, ytr, Ste, yte)
    _clf_report("LinearSVM (hinge)",
                LinearSVM(0.05, 800, 0.01).fit(Str, ytr), Str, ytr, Ste, yte)
    _clf_report("GaussianNaiveBayes",
                GaussianNaiveBayes().fit(Str, ytr), Str, ytr, Ste, yte)
    _clf_report("LDA", LinearDiscriminantAnalysis().fit(Str, ytr), Str, ytr, Ste, yte)
    _clf_report("KNN (k=9)", KNNClassifier(9).fit(Str, ytr), Str, ytr, Ste, yte)
    _clf_report("DecisionTree (depth 4, gini)",
                DecisionTree(4).fit(Str, ytr), Str, ytr, Ste, yte)
    _clf_report("RandomForest (40 trees)",
                RandomForest(40, 6).fit(Str, ytr), Str, ytr, Ste, yte)
    _clf_report("AdaBoost (30 stumps)",
                AdaBoost(30).fit(Str, ytr), Str, ytr, Ste, yte)
    _clf_report("GradientBoosting (60 x depth2)",
                GradientBoostingClassifier(60, 0.1, 2).fit(Str, ytr), Str, ytr, Ste, yte)

    M, labels = confusion_matrix(yte, lr.predict(Ste))
    print("\n  Confusion matrix for LogisticRegression (rows=true, cols=pred):")
    print(f"      labels {labels}")
    for row in M:
        print("      " + "  ".join(f"{int(v):4d}" for v in row))


def demo_nonlinear():
    _hdr("NON-LINEAR CLASSIFICATION  —  two interleaving moons")
    X, y = make_moons(300, 0.15)
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.25, seed=3, stratify=True)
    sc = Standardizer().fit(Xtr)
    Str, Ste = sc.transform(Xtr), sc.transform(Xte)

    print("  (a linear model should struggle here; kernels/trees/nets should not)")
    _clf_report("LogisticRegression (linear)",
                LogisticRegression(0.5, 600).fit(Str, ytr), Str, ytr, Ste, yte)
    _clf_report("LogisticRegression + poly deg2",
                LogisticRegression(0.5, 800).fit(polynomial_features(Str, 2), ytr),
                polynomial_features(Str, 2), ytr, polynomial_features(Ste, 2), yte)
    _clf_report("KernelSVM (RBF, C=1)",
                KernelSVM(1.0, "rbf", 1.0, max_passes=5).fit(Str, ytr),
                Str, ytr, Ste, yte)
    _clf_report("KNN (k=5)", KNNClassifier(5).fit(Str, ytr), Str, ytr, Ste, yte)
    _clf_report("RandomForest (50 trees)",
                RandomForest(50, 8).fit(Str, ytr), Str, ytr, Ste, yte)
    _clf_report("GradientBoosting (100 x depth3)",
                GradientBoostingClassifier(100, 0.1, 3).fit(Str, ytr), Str, ytr, Ste, yte)

    net = MLP([2, 16, 8, 2], "relu", "softmax", lr=0.1, epochs=120, seed=1)
    net.fit(Str, one_hot(ytr, 2))
    _clf_report("MLP [2-16-8-2] relu+softmax", net, Str, ytr, Ste, yte)
    print(f"      cross-entropy: {net.history[0]:.4f} -> {net.history[-1]:.4f}")


def demo_multiclass():
    _hdr("MULTICLASS CLASSIFICATION  —  4 Gaussian blobs in 3-D")
    X, y = make_blobs(400, centers=4, d=3, spread=2.2, seed=7)
    Xtr, Xte, ytr, yte = train_test_split(X, y, 0.25, seed=5, stratify=True)
    sc = Standardizer().fit(Xtr)
    Str, Ste = sc.transform(Xtr), sc.transform(Xte)

    for name, m in [
        ("SoftmaxRegression", SoftmaxRegression(0.5, 400)),
        ("GaussianNaiveBayes", GaussianNaiveBayes()),
        ("LDA", LinearDiscriminantAnalysis()),
        ("KNN (k=7)", KNNClassifier(7)),
        ("DecisionTree (entropy, d=6)", DecisionTree(6, criterion="entropy")),
        ("RandomForest (40 trees)", RandomForest(40, 8)),
        ("MLP [3-24-4]", MLP([3, 24, 4], "tanh", "softmax", lr=0.1, epochs=120, seed=2)),
    ]:
        if isinstance(m, MLP):
            m.fit(Str, one_hot(ytr, 4))
        else:
            m.fit(Str, ytr)
        print(f"  {name:<34} train acc {accuracy(ytr, m.predict(Str)):6.4f} | "
              f"test acc {accuracy(yte, m.predict(Ste)):6.4f}")


def demo_unsupervised():
    _hdr("UNSUPERVISED LEARNING")
    X, ytrue = make_blobs(240, centers=3, d=2, spread=1.1, seed=11)

    km = KMeans(3, n_init=10, seed=0).fit(X)
    print(f"  KMeans        inertia {km.inertia_:9.3f} | "
          f"silhouette {silhouette_score(X, km.labels_):.4f}")
    print(f"      centers: {[[round(v,2) for v in c] for c in km.centers]}")

    print("\n  Elbow scan (inertia should drop sharply then flatten at true k=3):")
    for k in range(1, 7):
        m = KMeans(k, n_init=5, seed=0).fit(X)
        print(f"      k={k}  inertia {m.inertia_:9.3f}")

    db = DBSCAN(eps=1.5, min_samples=5).fit(X)
    noise = sum(1 for l in db.labels_ if l == -1)
    print(f"\n  DBSCAN        clusters {db.n_clusters_} | noise points {noise}")

    ag = AgglomerativeClustering(3, "average").fit(X)
    print(f"  Agglomerative silhouette {silhouette_score(X, ag.labels_):.4f}")

    gm = GaussianMixture(3, seed=0).fit(X)
    print(f"  GMM (EM)      log-likelihood {gm.log_likelihood_:9.3f} | "
          f"weights {[round(p,3) for p in gm.pi]}")

    # PCA on correlated 4-D data
    _hdr("PCA  —  4-D data whose true structure is 2-D")
    rng = random.Random(9)
    Xp = []
    for _ in range(300):
        a, b = rng.gauss(0, 3), rng.gauss(0, 1)
        Xp.append([a + rng.gauss(0, .1), 2 * a + rng.gauss(0, .1),
                   b + rng.gauss(0, .1), -b + rng.gauss(0, .1)])
    Xs = Standardizer().fit_transform(Xp)
    pca = PCA(2).fit(Xs)
    print(f"  explained variance ratio: "
          f"{[round(v, 4) for v in pca.explained_variance_ratio_]}")
    print(f"  total captured by 2 of 4 components: "
          f"{sum(pca.explained_variance_ratio_):.4f}")
    Z = pca.transform(Xs)
    rec = pca.inverse_transform(Z)
    err = sum(euclidean(a, b) for a, b in zip(Xs, rec)) / len(Xs)
    print(f"  mean reconstruction error: {err:.4f}")
    print(f"  PC1 loadings: {[round(v, 3) for v in pca.components_[0]]}")


def demo_tree_structure():
    _hdr("A DECISION TREE, PRINTED")
    X, y = make_moons(200, noise=0.2, seed=21)
    t = DecisionTree(max_depth=3).fit(X, y)
    print("  Trained on the 'moons' data; each line is an axis-aligned split.\n")
    t.print_tree(feature_names=["x", "y"])
    print(f"\n  training accuracy of this depth-3 tree: {accuracy(y, t.predict(X)):.4f}")


if __name__ == "__main__":
    print("=" * 72)
    print(" MACHINE LEARNING FROM SCRATCH — pure Python, no external libraries")
    print("=" * 72)
    demo_regression()
    demo_linear_classification()
    demo_nonlinear()
    demo_multiclass()
    demo_unsupervised()
    demo_tree_structure()
    print("\n" + "=" * 72)
    print("All algorithms trained and evaluated with zero external libraries.")
    print("=" * 72)
