"""
================================================================================
 CALCULUS FOR ML FROM SCRATCH — pure Python, standard library only
================================================================================
 Only `math`, `cmath` and `random`. Every derivative rule, identity, and
 convergence rate below is CHECKED numerically: matrix-calculus identities
 against finite differences, convergence orders by measuring slopes, optimizer
 speeds against the condition number that theory says controls them.

 CONTENTS
 --------
 1. Numerical derivatives ... forward and central differences, the step-size
                              trade-off (truncation vs rounding), complex-step
 2. Automatic differentiation forward mode (dual numbers), reverse mode (a
                              tape), and why reverse mode wins for gradients
 3. Matrix calculus ......... gradients of x'Ax, ||Ax-b||^2, tr(AX), log det X,
                              softmax, sigmoid and logistic loss — all verified
 4. Taylor expansion ........ approximation order measured from error slopes
 5. Optimization ............ gradient descent vs condition number, step-size
                              stability (2/L), momentum, Newton's quadratic
                              convergence, saddle points
 6. Constrained optimization  Lagrange multipliers as shadow prices, KKT and
                              complementary slackness, projected gradient
 7. Integration ............. trapezoid and Simpson orders, Monte Carlo 1/sqrt(n),
                              high-dimensional volumes, change of variables
 8. Numerical pitfalls ...... log-sum-exp, cancellation, variance formulas,
                              summation error
 9. Demos
================================================================================
"""

import cmath
import math
import random

EPS = 2.220446049250313e-16


# =============================================================================
# 1. NUMERICAL DERIVATIVES
# =============================================================================
def forward_diff(f, x, h):
    """(f(x+h) - f(x)) / h.  Truncation error ~ h |f''|/2 (first order in h)."""
    return (f(x + h) - f(x)) / h


def central_diff(f, x, h):
    """(f(x+h) - f(x-h)) / 2h.  The even Taylor terms cancel, so the truncation
    error is ~ h^2 |f'''|/6 (second order)."""
    return (f(x + h) - f(x - h)) / (2 * h)


def complex_step(f, x, h=1e-20):
    """
    COMPLEX-STEP DERIVATIVE:  f'(x) ~ Im f(x + i h) / h.
    From the Taylor series f(x+ih) = f(x) + i h f'(x) - h^2 f''(x)/2 - ...,
    the imaginary part is h f'(x) + O(h^3). There is NO subtraction of nearly
    equal numbers, so h can be absurdly small (1e-20) and the result is
    accurate to machine precision. Requires f to be real-analytic and written
    with complex-safe operations (no abs(), no comparisons on the input).
    """
    return f(complex(x, h)).imag / h


def optimal_steps():
    """Balance truncation against rounding (rounding error ~ eps |f| / h):
         forward : h ~ sqrt(eps)   ~ 1.5e-8  -> best error ~ 1e-8
         central : h ~ eps^(1/3)   ~ 6e-6    -> best error ~ 1e-11"""
    return math.sqrt(EPS), EPS ** (1 / 3)


# =============================================================================
# 2. AUTOMATIC DIFFERENTIATION
# =============================================================================
OPS = {"count": 0}


class Dual:
    """
    FORWARD-MODE AD with dual numbers a + b*epsilon, epsilon^2 = 0.
    Carrying (value, derivative) through every operation computes ONE
    directional derivative per pass. The gradient of f: R^n -> R therefore
    needs n passes — cheap for few inputs, expensive for many (a neural
    network has millions).
    """
    __slots__ = ("v", "d")

    def __init__(self, v, d=0.0):
        self.v, self.d = v, d
        OPS["count"] += 1

    def __add__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.v + o.v, self.d + o.d)

    __radd__ = __add__

    def __sub__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.v - o.v, self.d - o.d)

    def __mul__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.v * o.v, self.d * o.v + self.v * o.d)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = o if isinstance(o, Dual) else Dual(o)
        return Dual(self.v / o.v, (self.d * o.v - self.v * o.d) / (o.v * o.v))


def d_sin(x):
    return Dual(math.sin(x.v), math.cos(x.v) * x.d) if isinstance(x, Dual) else math.sin(x)


def d_exp(x):
    return Dual(math.exp(x.v), math.exp(x.v) * x.d) if isinstance(x, Dual) else math.exp(x)


class Var:
    """
    REVERSE-MODE AD (backpropagation). The forward pass records each operation
    and its local partial derivatives on a graph; one backward pass pushes
    d(output)/d(node) from the output to every input by the chain rule. The
    gradient of f: R^n -> R costs a SMALL CONSTANT multiple of one function
    evaluation, independent of n (Baur-Strassen / the 'cheap gradient'
    principle). The price is memory: every intermediate value must be stored.
    """
    __slots__ = ("v", "g", "parents")

    def __init__(self, v, parents=()):
        self.v, self.g, self.parents = v, 0.0, parents
        OPS["count"] += 1

    def __add__(self, o):
        o = o if isinstance(o, Var) else Var(o)
        return Var(self.v + o.v, ((self, 1.0), (o, 1.0)))

    __radd__ = __add__

    def __sub__(self, o):
        o = o if isinstance(o, Var) else Var(o)
        return Var(self.v - o.v, ((self, 1.0), (o, -1.0)))

    def __mul__(self, o):
        o = o if isinstance(o, Var) else Var(o)
        return Var(self.v * o.v, ((self, o.v), (o, self.v)))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = o if isinstance(o, Var) else Var(o)
        return Var(self.v / o.v, ((self, 1 / o.v), (o, -self.v / (o.v * o.v))))

    def backward(self):
        order, seen = [], set()
        stack = [(self, False)]
        while stack:                                   # iterative topological sort
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p, _ in node.parents:
                stack.append((p, False))
        self.g = 1.0
        for node in reversed(order):
            for p, local in node.parents:
                p.g += local * node.g
                OPS["count"] += 1


def r_sin(x):
    return Var(math.sin(x.v), ((x, math.cos(x.v)),)) if isinstance(x, Var) else math.sin(x)


def r_exp(x):
    return Var(math.exp(x.v), ((x, math.exp(x.v)),)) if isinstance(x, Var) else math.exp(x)


def chain_objective(xs, sin=math.sin):
    """f(x) = sum_i x_i * sin(x_{i-1}) + x_0^2 — a scalar function of n inputs."""
    total = xs[0] * xs[0]
    for i in range(1, len(xs)):
        total = total + xs[i] * sin(xs[i - 1])
    return total


def chain_gradient_exact(xs):
    n = len(xs)
    g = [0.0] * n
    g[0] = 2 * xs[0]
    for i in range(1, n):
        g[i] += math.sin(xs[i - 1])
        g[i - 1] += xs[i] * math.cos(xs[i - 1])
    return g


def gradient_forward_mode(xs):
    """n forward passes, each seeding one input's derivative with 1."""
    OPS["count"] = 0
    g = []
    for j in range(len(xs)):
        duals = [Dual(x, 1.0 if i == j else 0.0) for i, x in enumerate(xs)]
        g.append(chain_objective(duals, sin=d_sin).d)
    return g, OPS["count"]


def gradient_reverse_mode(xs):
    """One forward pass (recording) + one backward pass."""
    OPS["count"] = 0
    vs = [Var(x) for x in xs]
    out = chain_objective(vs, sin=r_sin)
    out.backward()
    return [v.g for v in vs], OPS["count"]


def function_cost(xs):
    """Operation count of ONE plain evaluation (same counting convention)."""
    OPS["count"] = 0
    chain_objective([Dual(x) for x in xs], sin=d_sin)
    return OPS["count"]


# =============================================================================
# 3. MATRIX CALCULUS — identities checked against finite differences
# =============================================================================
def _mv(A, x):
    return [sum(a * b for a, b in zip(r, x)) for r in A]


def _mm(A, B):
    Bt = list(zip(*B))
    return [[sum(a * b for a, b in zip(r, c)) for c in Bt] for r in A]


def _T(A):
    return [list(c) for c in zip(*A)]


def _logdet(X):
    """log|det X| via LU with partial pivoting."""
    n = len(X)
    M = [r[:] for r in X]
    total = 0.0
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        total += math.log(abs(M[c][c]))
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n):
                M[r][k] -= f * M[c][k]
    return total


def _inv(X):
    n = len(X)
    M = [X[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r != c:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [r[n:] for r in M]


def numeric_grad_vec(f, x, h=1e-6):
    return [(f(x[:i] + [x[i] + h] + x[i + 1:]) - f(x[:i] + [x[i] - h] + x[i + 1:])) / (2 * h)
            for i in range(len(x))]


def numeric_grad_mat(f, X, h=1e-6):
    G = []
    for i in range(len(X)):
        row = []
        for j in range(len(X[0])):
            Xp = [r[:] for r in X]
            Xm = [r[:] for r in X]
            Xp[i][j] += h
            Xm[i][j] -= h
            row.append((f(Xp) - f(Xm)) / (2 * h))
        G.append(row)
    return G


def _rel(a, b):
    fa = [v for r in a for v in (r if isinstance(r, list) else [r])]
    fb = [v for r in b for v in (r if isinstance(r, list) else [r])]
    num = math.sqrt(sum((p - q) ** 2 for p, q in zip(fa, fb)))
    den = math.sqrt(sum(q * q for q in fb)) or 1.0
    return num / den


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def check_matrix_identities(seed=0):
    """
    Each identity is compared with a central-difference gradient; the result
    is the relative error. ~1e-9 or smaller means the formula is right.
    """
    rng = random.Random(seed)
    n = 4
    A = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
    x = [rng.gauss(0, 1) for _ in range(n)]
    b = [rng.gauss(0, 1) for _ in range(n)]
    out = []

    # d/dx x^T A x = (A + A^T) x
    f = lambda v: sum(vi * wi for vi, wi in zip(v, _mv(A, v)))
    exact = [p + q for p, q in zip(_mv(A, x), _mv(_T(A), x))]
    out.append(("d/dx  x'Ax = (A + A')x", _rel(numeric_grad_vec(f, x), exact)))

    # d/dx ||Ax - b||^2 = 2 A^T (Ax - b)
    f = lambda v: sum((p - q) ** 2 for p, q in zip(_mv(A, v), b))
    r = [p - q for p, q in zip(_mv(A, x), b)]
    exact = [2 * v for v in _mv(_T(A), r)]
    out.append(("d/dx  ||Ax-b||^2 = 2A'(Ax-b)", _rel(numeric_grad_vec(f, x), exact)))

    # d/dX tr(A X) = A^T
    X = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(n)]
    f = lambda M: sum(_mm(A, M)[i][i] for i in range(n))
    out.append(("d/dX  tr(AX) = A'", _rel(numeric_grad_mat(f, X), _T(A))))

    # d/dX log|det X| = X^{-T}
    Xs = [[X[i][j] + (3.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
    out.append(("d/dX  log|det X| = X^{-T}",
                _rel(numeric_grad_mat(_logdet, Xs), _T(_inv(Xs)))))

    # d/dX ||X||_F^2 = 2X
    f = lambda M: sum(v * v for row in M for v in row)
    out.append(("d/dX  ||X||_F^2 = 2X", _rel(numeric_grad_mat(f, X),
                                            [[2 * v for v in r] for r in X])))

    # softmax Jacobian: diag(p) - p p^T
    z = [rng.gauss(0, 1) for _ in range(n)]
    p = softmax(z)
    J_exact = [[(p[i] if i == j else 0.0) - p[i] * p[j] for j in range(n)] for i in range(n)]
    J_num = [numeric_grad_vec(lambda v, i=i: softmax(v)[i], z) for i in range(n)]
    out.append(("softmax Jacobian = diag(p) - pp'", _rel(J_num, J_exact)))

    # cross-entropy on softmax: d/dz [-log p_y] = p - onehot(y)
    y = 2
    f = lambda v: -math.log(softmax(v)[y])
    exact = [pi - (1.0 if i == y else 0.0) for i, pi in enumerate(p)]
    out.append(("d/dz  -log softmax(z)_y = p - e_y", _rel(numeric_grad_vec(f, z), exact)))

    # logistic regression loss: d/dw = (sigma(w.x) - y) x
    w = [rng.gauss(0, 1) for _ in range(n)]
    xi, yi = [rng.gauss(0, 1) for _ in range(n)], 1.0

    def logloss(v):
        s = 1 / (1 + math.exp(-sum(a * c for a, c in zip(v, xi))))
        return -(yi * math.log(s) + (1 - yi) * math.log(1 - s))
    s = 1 / (1 + math.exp(-sum(a * c for a, c in zip(w, xi))))
    out.append(("d/dw  logistic loss = (sigma - y)x",
                _rel(numeric_grad_vec(logloss, w), [(s - yi) * c for c in xi])))

    # a common WRONG identity, to show the check has teeth
    f = lambda v: sum(vi * wi for vi, wi in zip(v, _mv(A, v)))
    wrong = [2 * v for v in _mv(A, x)]            # correct only if A is symmetric
    out.append(("WRONG: d/dx x'Ax = 2Ax (A not symmetric)",
                _rel(numeric_grad_vec(f, x), wrong)))
    return out


# =============================================================================
# 4. TAYLOR EXPANSION
# =============================================================================
def taylor_orders(f, d1, d2, d3, x0=0.7, hs=(1e-1, 5e-2, 2.5e-2, 1.25e-2)):
    """
    Error of the k-th order Taylor approximation should scale like h^(k+1).
    Measure the slope of log(error) vs log(h) between successive h.
    """
    fx = f(x0)
    rows = []
    for h in hs:
        e0 = abs(f(x0 + h) - fx)
        e1 = abs(f(x0 + h) - (fx + d1 * h))
        e2 = abs(f(x0 + h) - (fx + d1 * h + d2 * h * h / 2))
        rows.append((h, e0, e1, e2))
    slopes = []
    for k in (1, 2, 3):
        s = [math.log(rows[i][k] / rows[i + 1][k]) / math.log(rows[i][0] / rows[i + 1][0])
             for i in range(len(rows) - 1)]
        slopes.append(sum(s) / len(s))
    return rows, slopes


# =============================================================================
# 5. OPTIMIZATION
# =============================================================================
def gd_quadratic(eigs, lr, tol=1e-6, max_iter=10 ** 6, momentum=0.0, x0=None):
    """
    Minimize f(x) = 1/2 sum_i lambda_i x_i^2 (a quadratic in its eigenbasis —
    any convex quadratic looks like this after rotation). Gradient descent
    multiplies each coordinate by (1 - lr * lambda_i) per step, so with the
    best fixed step (lr = 1/L, L = lambda_max) the slowest coordinate shrinks
    by (1 - 1/kappa): iterations grow LINEARLY in kappa = L / mu.

    Heavy-ball momentum  x <- x - lr grad + beta (x - x_prev)  with tuned
    (lr, beta) contracts at (sqrt(kappa) - 1)/(sqrt(kappa) + 1): iterations
    grow like sqrt(kappa). Returns iterations until ||x|| < tol, or None if the
    iterate diverges.
    """
    x = list(x0) if x0 else [1.0] * len(eigs)
    prev = x[:]
    for it in range(1, max_iter + 1):
        g = [l * xi for l, xi in zip(eigs, x)]
        new = [xi - lr * gi + momentum * (xi - pi) for xi, gi, pi in zip(x, g, prev)]
        prev, x = x, new
        nrm = math.sqrt(sum(v * v for v in x))
        if nrm < tol:
            return it
        if nrm > 1e12 or nrm != nrm:
            return None
    return max_iter


def heavy_ball_params(L, mu):
    """Optimal heavy-ball parameters for a quadratic (Polyak)."""
    lr = 4 / (math.sqrt(L) + math.sqrt(mu)) ** 2
    beta = ((math.sqrt(L / mu) - 1) / (math.sqrt(L / mu) + 1)) ** 2
    return lr, beta


def newton_1d(fp, fpp, x0, iters=6):
    """Newton's method for minimization, x <- x - f'(x)/f''(x). Near a
    minimum with f'' > 0 the error SQUARES each step (quadratic convergence):
    the number of correct digits roughly doubles."""
    xs = [x0]
    for _ in range(iters):
        x = xs[-1]
        xs.append(x - fp(x) / fpp(x))
    return xs


def saddle_escape(y0, lr=0.1, noise=0.0, max_iter=20000, seed=0):
    """
    f(x, y) = x^2/2 + y^4/4 - y^2/2 : a saddle at the origin (Hessian eigenvalues
    +1 and -1), minima at (0, +/-1). Starting ON the stable manifold (y = 0)
    plain gradient descent converges to the saddle and stays; starting a
    distance y0 off it, it escapes in ~ log(1/y0)/log(1 + lr) steps. Noise
    (as in SGD) knocks it off. Returns steps until |y| > 0.5, or None.
    """
    rng = random.Random(seed)
    x, y = 1.0, y0
    for it in range(1, max_iter + 1):
        gx, gy = x, y ** 3 - y
        x -= lr * gx
        y -= lr * gy
        if noise:
            y += noise * rng.gauss(0, 1)
        if abs(y) > 0.5:
            return it
    return None


# =============================================================================
# 6. CONSTRAINED OPTIMIZATION
# =============================================================================
def lagrange_product(c):
    """
    maximize f = x*y subject to x + y = c.
    Lagrangian L = xy - lam (x + y - c):  y = lam, x = lam, x + y = c
        => x = y = c/2,  lam = c/2,  f* = c^2/4.
    The multiplier is the SHADOW PRICE of the constraint: df*/dc = c/2 = lam.
    """
    return c / 2, c / 2, c * c / 4, c / 2


def projected_gradient(target, a, bnd, lr=0.1, iters=2000):
    """
    minimize ||x - target||^2 subject to a.x <= bnd, by gradient steps followed
    by projection onto the half-space. Returns (x, lambda) where lambda is
    recovered from stationarity: 2(x - target) + lam * a = 0.
    """
    x = [0.0] * len(target)
    na2 = sum(v * v for v in a)
    for _ in range(iters):
        g = [2 * (xi - ti) for xi, ti in zip(x, target)]
        x = [xi - lr * gi for xi, gi in zip(x, g)]
        viol = sum(ai * xi for ai, xi in zip(a, x)) - bnd
        if viol > 0:                                   # project back onto a.x = bnd
            x = [xi - viol * ai / na2 for xi, ai in zip(x, a)]
    g = [2 * (xi - ti) for xi, ti in zip(x, target)]
    lam = -sum(gi * ai for gi, ai in zip(g, a)) / na2
    return x, max(lam, 0.0)


# =============================================================================
# 7. INTEGRATION
# =============================================================================
def trapezoid(f, a, b, n):
    h = (b - a) / n
    return h * (0.5 * f(a) + sum(f(a + i * h) for i in range(1, n)) + 0.5 * f(b))


def simpson(f, a, b, n):
    if n % 2:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b) + sum((4 if i % 2 else 2) * f(a + i * h) for i in range(1, n))
    return s * h / 3


def mc_integral(f, a, b, n, rng):
    return (b - a) * sum(f(a + (b - a) * rng.random()) for _ in range(n)) / n


def unit_ball_volume(d):
    return math.pi ** (d / 2) / math.gamma(d / 2 + 1)


def mc_ball_fraction(d, n, rng):
    """Fraction of the cube [-1, 1]^d inside the unit ball — the MC estimate of
    volume(ball)/2^d, whose standard error is sqrt(p(1-p)/n) in ANY dimension."""
    hits = 0
    for _ in range(n):
        if sum(rng.uniform(-1, 1) ** 2 for _ in range(d)) <= 1:
            hits += 1
    return hits / n


def change_of_variables_check(A, n=200000, seed=0):
    """
    Push the uniform distribution on the unit square through y = A x. The image
    is a parallelogram of area |det A|, so the pushed-forward density is
    1/|det A| there. Check: estimate the area of the image by Monte Carlo
    (sample a bounding box, test membership by inverting A) and compare with
    |det A|. The same Jacobian factor appears in normalizing flows:
        log p_Y(y) = log p_X(x) - log |det J|.
    """
    rng = random.Random(seed)
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    inv = [[A[1][1] / det, -A[0][1] / det], [-A[1][0] / det, A[0][0] / det]]
    corners = [(0, 0), (1, 0), (0, 1), (1, 1)]
    ys = [(A[0][0] * u + A[0][1] * v, A[1][0] * u + A[1][1] * v) for u, v in corners]
    lox, hix = min(p[0] for p in ys), max(p[0] for p in ys)
    loy, hiy = min(p[1] for p in ys), max(p[1] for p in ys)
    hits = 0
    for _ in range(n):
        y1, y2 = rng.uniform(lox, hix), rng.uniform(loy, hiy)
        x1 = inv[0][0] * y1 + inv[0][1] * y2
        x2 = inv[1][0] * y1 + inv[1][1] * y2
        hits += 0 <= x1 <= 1 and 0 <= x2 <= 1
    return hits / n * (hix - lox) * (hiy - loy), abs(det)


# =============================================================================
# 8. NUMERICAL PITFALLS
# =============================================================================
def logsumexp(z):
    """log sum exp(z_i) = m + log sum exp(z_i - m), m = max z. The shift is exact
    algebra and prevents overflow (exp(1000) = inf) and total underflow."""
    m = max(z)
    return m + math.log(sum(math.exp(v - m) for v in z))


def naive_logsumexp(z):
    try:
        return math.log(sum(math.exp(v) for v in z))
    except OverflowError:
        return float("inf")


def welford_variance(xs):
    """One-pass, numerically stable variance (Welford 1962)."""
    n, mean, m2 = 0, 0.0, 0.0
    for x in xs:
        n += 1
        d = x - mean
        mean += d / n
        m2 += d * (x - mean)
    return m2 / (n - 1)


def naive_variance(xs):
    """E[x^2] - E[x]^2 : subtracts two huge, nearly equal numbers when the mean
    is large relative to the spread — catastrophic cancellation."""
    n = len(xs)
    s, s2 = sum(xs), sum(x * x for x in xs)
    return (s2 - s * s / n) / (n - 1)


def naive_sum(xs):
    """Plain left-to-right accumulation. Each addition rounds; the errors can
    accumulate roughly in proportion to the number of terms.
    NOTE: Python 3.12+'s built-in sum() already uses compensated (Neumaier)
    summation for floats, so it is NOT a naive baseline — this loop is."""
    s = 0.0
    for x in xs:
        s += x
    return s


def kahan_sum(xs):
    """Compensated summation: carries the rounding error of each addition
    forward, so the error no longer grows with the number of terms."""
    s, c = 0.0, 0.0
    for x in xs:
        y = x - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


# =============================================================================
# 9. DEMOS
# =============================================================================
def _hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def _test_f(x):
    if isinstance(x, complex):
        return cmath.exp(x) * cmath.sin(x) / (1 + x * x)
    return math.exp(x) * math.sin(x) / (1 + x * x)


def _test_fp(x):
    u, du = math.exp(x) * math.sin(x), math.exp(x) * (math.sin(x) + math.cos(x))
    v, dv = 1 + x * x, 2 * x
    return (du * v - u * dv) / (v * v)


def demo_numerical_derivatives():
    _hdr("1. NUMERICAL DERIVATIVES — smaller h is NOT always better")
    x0 = 0.8
    ex = _test_fp(x0)
    print("  f(x) = e^x sin(x) / (1 + x^2) at x = 0.8; relative error vs the exact")
    print("  derivative (quotient rule):")
    print(f"  {'h':>8}{'forward':>12}{'central':>12}{'complex-step':>15}")
    for h in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14]:
        fe = abs(forward_diff(_test_f, x0, h) - ex) / abs(ex)
        ce = abs(central_diff(_test_f, x0, h) - ex) / abs(ex)
        cs = abs(complex_step(_test_f, x0, h) - ex) / abs(ex)
        print(f"  {h:>8.0e}{fe:>12.1e}{ce:>12.1e}{cs:>15.1e}")
    hf, hc = optimal_steps()
    print(f"  theory: best forward step ~ sqrt(eps) = {hf:.1e}, "
          f"best central step ~ eps^(1/3) = {hc:.1e}")
    print("  -> truncation error falls as h shrinks, but ROUNDING error grows like")
    print("     eps/h because f(x+h) - f(x) subtracts nearly equal numbers. The")
    print("     total is U-shaped: forward differences bottom out near 1e-8 and")
    print("     central near 1e-10, then get WORSE. The complex step subtracts")
    print("     nothing, so it reaches machine precision for any tiny h.")


def demo_autodiff():
    _hdr("2. AUTOMATIC DIFFERENTIATION — forward vs reverse mode")
    rng = random.Random(0)
    print("  Gradient of f(x) = x_0^2 + sum_i x_i sin(x_(i-1)), f: R^n -> R.")
    print("  Cost counted in primitive operations (seed 0):")
    print(f"  {'n':>6}{'one f eval':>12}{'forward mode':>15}{'reverse mode':>15}"
          f"{'fwd / f':>9}{'rev / f':>9}{'max error':>11}")
    for n in [10, 100, 1000]:
        xs = [rng.gauss(0, 1) for _ in range(n)]
        gf, cf = gradient_forward_mode(xs)
        gr, cr = gradient_reverse_mode(xs)
        c0 = function_cost(xs)
        ge = chain_gradient_exact(xs)
        err = max(max(abs(a - b) for a, b in zip(gf, ge)),
                  max(abs(a - b) for a, b in zip(gr, ge)))
        print(f"  {n:>6}{c0:>12,}{cf:>15,}{cr:>15,}{cf / c0:>9.1f}{cr / c0:>9.2f}"
              f"{err:>11.1e}")
    print("  -> forward mode computes one directional derivative per pass, so the")
    print("     full gradient costs n passes. Reverse mode gets every partial")
    print("     derivative from ONE backward pass at a constant multiple (~2.25x")
    print("     here) of the function cost. That constant is why backpropagation")
    print("     can train models with billions of parameters. (Forward mode wins")
    print("     the other way round: few inputs, many outputs, or Jacobian-vector")
    print("     products. Reverse mode's price is storing every intermediate.)")


def demo_matrix_calculus():
    _hdr("3. MATRIX CALCULUS — identities checked, not memorized")
    print("  Each formula vs a central-difference gradient (random 4 x 4 data,")
    print("  seed 0). Relative error ~1e-10 means the identity is right:")
    for name, e in check_matrix_identities():
        print(f"    {name:<44}{e:>10.1e}")
    print("  -> the last line is a common mistake: 2Ax is the gradient of x'Ax")
    print("     ONLY when A is symmetric. A numerical gradient check catches this")
    print("     kind of error in seconds — do it for every hand-derived gradient.")


def demo_taylor():
    _hdr("4. TAYLOR EXPANSION — approximation order, measured")
    rows, slopes = taylor_orders(math.exp, math.exp(0.7), math.exp(0.7), math.exp(0.7))
    print("  exp(x) around x0 = 0.7, error of the k-th order Taylor polynomial:")
    print(f"  {'h':>9}{'order 0':>12}{'order 1':>12}{'order 2':>12}")
    for h, e0, e1, e2 in rows:
        print(f"  {h:>9.4f}{e0:>12.2e}{e1:>12.2e}{e2:>12.2e}")
    print(f"  measured slopes of log(error) vs log(h): "
          f"{', '.join(f'{s:.3f}' for s in slopes)}  (theory 1, 2, 3)")
    print("  -> a k-th order expansion has error O(h^(k+1)). Gradient descent uses")
    print("     the first-order model, Newton's method the second-order one.")


def demo_optimization():
    _hdr("5. OPTIMIZATION — condition number decides the speed")
    print("  f(x) = 1/2 (mu x1^2 + L x2^2), L = 1, kappa = L/mu; iterations until")
    print("  ||x|| < 1e-6 from (1, 1):")
    print(f"  {'kappa':>8}{'GD (lr=1/L)':>13}{'GD theory':>11}{'heavy ball':>12}"
          f"{'HB rate formula':>17}{'Newton':>8}")
    for k in [10, 100, 1000, 10000]:
        eigs = [1.0 / k, 1.0]
        gd = gd_quadratic(eigs, 1.0)
        lrh, beta = heavy_ball_params(1.0, 1.0 / k)
        hb = gd_quadratic(eigs, lrh, momentum=beta)
        gdt = math.ceil(math.log(1e-6) / math.log(1 - 1 / k))
        hbt = math.ceil(math.log(1e-6) / math.log((math.sqrt(k) - 1) / (math.sqrt(k) + 1)))
        print(f"  {k:>8}{gd:>13,}{gdt:>11,}{hb:>12,}{hbt:>17,}{1:>8}")
    print("  -> gradient descent needs ~kappa * ln(1/tol) steps: exactly the")
    print("     formula. Tuned momentum scales like sqrt(kappa) (x3.3 per 10x kappa,")
    print("     sqrt(10) = 3.16) but needs ~1.5x the simple rate formula, because at")
    print("     the optimal tuning its iteration matrix has a repeated eigenvalue,")
    print("     adding a k * rho^k term. Newton rescales by the Hessian and solves")
    print("     any quadratic in ONE step, at O(d^3) cost per step.\n")

    print("  Step-size stability on f(x) = x^2/2 (L = 1):")
    for f in [1.9, 1.99, 2.01, 2.1]:
        r = gd_quadratic([1.0], f, max_iter=5000)
        print(f"    lr = {f:.2f}/L -> {'diverges' if r is None else f'converges in {r} steps'}")
    print("  -> each step multiplies the error by (1 - lr*L); the step must satisfy")
    print("     lr < 2/L. Near the limit it converges but oscillates slowly.\n")

    xs = newton_1d(lambda x: math.exp(x) - 2, lambda x: math.exp(x), 1.0)
    print("  Newton's method on f(x) = e^x - 2x (minimum at ln 2), errors per step:")
    print("    " + ", ".join(f"{abs(x - math.log(2)):.1e}" for x in xs[:6]))
    print("  -> quadratic convergence: the exponent roughly doubles each step,")
    print("     once the iterate is close enough (it is only LOCALLY guaranteed).\n")

    print("  Saddle point f = x^2/2 + y^4/4 - y^2/2 (Hessian eigenvalues +1, -1 at")
    print("  the origin), gradient descent lr = 0.1, steps until |y| > 0.5:")
    for y0, noise, label in [(0.0, 0.0, "start exactly at y = 0"),
                             (1e-8, 0.0, "start at y = 1e-8"),
                             (1e-4, 0.0, "start at y = 1e-4"),
                             (0.0, 1e-6, "y = 0 plus noise sd 1e-6")]:
        r = saddle_escape(y0, noise=noise)
        print(f"    {label:<28}{'never escapes' if r is None else f'{r} steps'}")
    print(f"  theory for y0 = 1e-8: ln(0.5/1e-8)/ln(1.1) = "
          f"{math.log(0.5 / 1e-8) / math.log(1.1):.0f} steps")
    print("  -> a saddle has zero gradient, so exact gradient descent started on")
    print("     its stable manifold never leaves. Any perturbation grows")
    print("     exponentially along the negative-curvature direction, which is why")
    print("     saddles SLOW noisy optimizers down rather than trap them.")


def demo_constrained():
    _hdr("6. CONSTRAINED OPTIMIZATION — Lagrange multipliers and KKT")
    x, y, fs, lam = lagrange_product(10.0)
    h = 1e-3
    shadow = (lagrange_product(10 + h)[2] - lagrange_product(10 - h)[2]) / (2 * h)
    print("  maximize xy subject to x + y = c, at c = 10:")
    print(f"    solution x = y = {x}, f* = {fs}, multiplier lambda = {lam}")
    print(f"    numerical d f*/d c = {shadow:.6f}")
    print("  -> the multiplier equals the rate at which the optimum improves as the")
    print("     constraint is relaxed: its SHADOW PRICE.\n")

    xa, la = projected_gradient([2.0, 1.0], [1.0, 1.0], 2.0)
    xi, li = projected_gradient([0.5, 0.5], [1.0, 1.0], 2.0)

    def opt(b):
        z, _ = projected_gradient([2.0, 1.0], [1.0, 1.0], b)
        return (z[0] - 2) ** 2 + (z[1] - 1) ** 2
    dfdb = (opt(2 + h) - opt(2 - h)) / (2 * h)
    print("  minimize ||x - t||^2 subject to x1 + x2 <= 2 (projected gradient):")
    print(f"    t = (2, 1): x* = ({xa[0]:.4f}, {xa[1]:.4f}), lambda = {la:.4f}, "
          f"constraint ACTIVE (x1 + x2 = {xa[0] + xa[1]:.4f})")
    print(f"    t = (0.5, 0.5): x* = ({xi[0]:.4f}, {xi[1]:.4f}), lambda = {li:.1e}, "
          f"constraint INACTIVE (x1 + x2 = {xi[0] + xi[1]:.4f})")
    print(f"    shadow price check (active case): d f*/d(bound) = {dfdb:.4f} = -lambda")
    print("  -> COMPLEMENTARY SLACKNESS: lambda * (constraint slack) = 0. Either the")
    print("     constraint binds and lambda can be positive, or it is slack and")
    print("     lambda = 0. This is exactly why most SVM multipliers are zero and")
    print("     only the support vectors matter.")


def demo_integration():
    _hdr("7. INTEGRATION — convergence orders and dimension")
    print("  integral of sin(x) over [0, pi] = 2:")
    print(f"  {'n':>5}{'trapezoid error':>17}{'Simpson error':>15}")
    prev = None
    for n in [4, 8, 16, 32]:
        te = abs(trapezoid(math.sin, 0, math.pi, n) - 2)
        se = abs(simpson(math.sin, 0, math.pi, n) - 2)
        print(f"  {n:>5}{te:>17.2e}{se:>15.2e}")
    print("  -> doubling n cuts trapezoid error ~4x (order h^2) and Simpson ~16x")
    print("     (order h^4), for smooth integrands.\n")

    rng = random.Random(0)
    print("  Monte Carlo, same integral; RMS error over 20 repetitions (seed 0):")
    for n in [100, 1000, 10000, 100000]:
        errs = [abs(mc_integral(math.sin, 0, math.pi, n, rng) - 2) for _ in range(20)]
        rms = math.sqrt(sum(e * e for e in errs) / 20)
        print(f"    n = {n:>7,}: RMS error {rms:.2e}")
    print("  -> error ~ 1/sqrt(n): 100x more samples buy only 10x accuracy, far")
    print("     slower than Simpson in 1-D. MC's advantage is elsewhere: its rate")
    print("     does not depend on dimension, while a grid with k points per axis")
    print("     needs k^d points.\n")

    print("  Fraction of the cube [-1,1]^d inside the unit ball (20,000 MC points each):")
    print(f"  {'d':>4}{'MC estimate':>13}{'exact':>12}")
    for d in [2, 5, 10, 20]:
        print(f"  {d:>4}{mc_ball_fraction(d, 20000, rng):>13.5f}"
              f"{unit_ball_volume(d) / 2 ** d:>12.2e}")
    print("  -> in 20 dimensions the ball fills 2.5e-8 of the cube: 20,000 samples")
    print("     see none of it. MC error is dimension-free in ABSOLUTE terms, but")
    print("     rare events need enormous n for any RELATIVE accuracy — the curse of")
    print("     dimensionality, and why importance sampling exists.\n")

    A = [[2.0, 1.0], [0.5, 1.5]]
    area, det = change_of_variables_check(A)
    print(f"  Change of variables: the unit square mapped by A = {A}")
    print(f"    Monte Carlo area of the image = {area:.4f}; |det A| = {det:.4f}")
    print("  -> a linear map scales volumes by |det A|, so densities scale by")
    print("     1/|det A|. Non-linear maps do the same locally with |det Jacobian|:")
    print("     the foundation of normalizing flows and of the reparameterization")
    print("     trick's density bookkeeping.")


def demo_numerics():
    _hdr("8. NUMERICAL PITFALLS — correct formulas, wrong answers")
    z = [1000.0, 1000.5, 999.0]
    print(f"  log-sum-exp of {z}:")
    print(f"    naive log(sum(exp(z))) = {naive_logsumexp(z)}")
    print(f"    shifted by max(z)      = {logsumexp(z):.6f}")
    print("  -> exp(1000) overflows. Subtracting the max is exact algebra and is")
    print("     how every softmax and cross-entropy implementation works.\n")

    x = 1e-8
    print(f"  1 - cos(x) at x = 1e-8 (true value ~ x^2/2 = {x * x / 2:.3e}):")
    print(f"    1 - cos(x)      = {1 - math.cos(x):.3e}")
    print(f"    2 sin^2(x/2)    = {2 * math.sin(x / 2) ** 2:.3e}")
    print("  -> cos(1e-8) rounds to exactly 1.0, so the subtraction returns 0:")
    print("     CATASTROPHIC CANCELLATION. Rewrite to avoid subtracting nearly")
    print("     equal numbers (the same reason math.expm1 and math.log1p exist).\n")

    rng = random.Random(1)
    data = [1e9 + rng.gauss(0, 1) for _ in range(10000)]
    print("  Variance of 10,000 values ~ N(1e9, 1) (true variance 1, seed 1):")
    print(f"    E[x^2] - E[x]^2 formula : {naive_variance(data):.4f}")
    print(f"    Welford one-pass        : {welford_variance(data):.4f}")
    print("  -> the textbook identity subtracts two numbers ~1e18 that agree in")
    print("     almost every digit. Centre first, or use Welford's update.\n")

    xs = [0.1] * 1_000_000
    print("  Adding 0.1 one million times (exact answer 100,000):")
    print(f"    plain loop           : error {naive_sum(xs) - 100000:.2e}")
    print(f"    Kahan compensated    : error {kahan_sum(xs) - 100000:.2e}")
    print(f"    math.fsum (correctly rounded): error {math.fsum(xs) - 100000:.2e}")
    print("  -> rounding errors accumulate with the number of terms; compensated")
    print("     summation carries them forward. (Python 3.12+'s built-in sum() is")
    print("     already compensated for floats, so it would show 0 here.)")


if __name__ == "__main__":
    print("=" * 78)
    print(" CALCULUS FOR ML FROM SCRATCH — every rule and rate checked numerically")
    print("=" * 78)
    demo_numerical_derivatives()
    demo_autodiff()
    demo_matrix_calculus()
    demo_taylor()
    demo_optimization()
    demo_constrained()
    demo_integration()
    demo_numerics()
    print("\n" + "=" * 78)
    print("All results use fixed seeds; rerunning reproduces them exactly.")
    print("=" * 78)
