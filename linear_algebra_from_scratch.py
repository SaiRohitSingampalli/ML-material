"""
================================================================================
 LINEAR ALGEBRA FOR ML FROM SCRATCH — pure Python, standard library only
================================================================================
 Only `math` and `random`. Every decomposition is implemented here, then
 MEASURED: not just "does it give the right answer" but "how much accuracy does
 it lose, and when". In exact arithmetic most of these methods are
 interchangeable. In floating point they are not, and interviews probe exactly
 that difference — why you never invert X'X, why pivoting matters, why SVD is
 computed on X rather than on X'X.

 Floating point here is IEEE double precision: machine epsilon ~ 2.2e-16. A
 useful rule: solving a problem with condition number kappa by a stable method
 loses about log10(kappa) of the ~16 available decimal digits.

 CONTENTS
 --------
 0. Core ................. vectors, matrices, norms, random orthogonal
                           matrices, matrices with a CHOSEN condition number
 1. Subspaces ............ row reduction, rank, null space, rank-nullity, the
                           four fundamental subspaces, projections
 2. Linear systems ....... Gaussian elimination with and without pivoting, LU,
                           Cholesky (and as a positive-definiteness test)
 3. Orthogonalization .... classical Gram-Schmidt, modified Gram-Schmidt,
                           Householder QR
 4. Least squares ........ normal equations vs QR, and their accuracy vs kappa
 5. Eigenvalues .......... power iteration and the eigengap, symmetric Jacobi
                           eigendecomposition, the spectral theorem
 6. SVD .................. one-sided Jacobi SVD, SVD via X'X and what it
                           loses, Eckart-Young, pseudo-inverse, PCA
 7. Conditioning ......... condition numbers, perturbation amplification,
                           Hilbert matrices, why the determinant is a bad
                           measure of near-singularity
 8. Demos ................ every claim above, measured
================================================================================
"""

import math
import random

EPS = 2.220446049250313e-16          # machine epsilon for IEEE doubles


# =============================================================================
# 0. CORE
# =============================================================================
def zeros(n, m):
    return [[0.0] * m for _ in range(n)]


def eye(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def transpose(A):
    return [list(c) for c in zip(*A)]


def matmul(A, B):
    Bt = transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def matvec(A, x):
    return [sum(a * b for a, b in zip(row, x)) for row in A]


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def norm2(v):
    """Euclidean norm, scaled to avoid overflow/underflow for extreme values."""
    s = max((abs(x) for x in v), default=0.0)
    if s == 0.0:
        return 0.0
    return s * math.sqrt(sum((x / s) ** 2 for x in v))


def sub(u, v):
    return [a - b for a, b in zip(u, v)]


def add(u, v):
    return [a + b for a, b in zip(u, v)]


def scale(v, c):
    return [c * a for a in v]


def fro(A):
    return math.sqrt(sum(x * x for row in A for x in row))


def mat_sub(A, B):
    return [[a - b for a, b in zip(r, s)] for r, s in zip(A, B)]


def orthogonality_loss(Q):
    """|| Q^T Q - I ||_F : zero for perfectly orthonormal columns."""
    return fro(mat_sub(matmul(transpose(Q), Q), eye(len(Q[0]))))


def random_matrix(n, m, rng):
    return [[rng.gauss(0, 1) for _ in range(m)] for _ in range(n)]


def random_orthogonal(n, rng):
    """Q from the Householder QR of a Gaussian matrix (uniformly random up to
    signs). Householder is used because it produces a Q that is orthogonal to
    machine precision — see Part 3 for why Gram-Schmidt would not."""
    Q, _ = householder_qr(random_matrix(n, n, rng))
    return Q


def matrix_with_singular_values(m, n, svals, seed=0):
    """
    A = U diag(s) V^T with U (m x n) orthonormal columns and V (n x n)
    orthogonal, so the singular values — and hence the condition number
    kappa = s_max / s_min — are exactly what we choose. This is how every
    accuracy experiment below controls conditioning.
    """
    rng = random.Random(seed)
    U = [row[:n] for row in random_orthogonal(m, rng)]
    V = random_orthogonal(n, rng)
    US = [[U[i][k] * svals[k] for k in range(n)] for i in range(m)]
    return matmul(US, transpose(V))


def log_spaced(hi, lo, n):
    return [hi * (lo / hi) ** (k / (n - 1)) for k in range(n)]


# =============================================================================
# 1. SUBSPACES
# =============================================================================
def rref(A, tol=1e-10):
    """
    Reduced row echelon form with partial pivoting and a TOLERANCE.
    Returns (R, pivot_columns). In floating point, 'is this pivot zero?' is a
    judgment call: numerical rank depends on the tolerance. Singular values
    (Part 6) give a more reliable rank: count sigma_i > tol * sigma_max.
    """
    R = [row[:] for row in A]
    n, m = len(R), len(R[0])
    pivots, r = [], 0
    for c in range(m):
        if r == n:
            break
        p = max(range(r, n), key=lambda i: abs(R[i][c]))
        if abs(R[p][c]) < tol:
            continue
        R[r], R[p] = R[p], R[r]
        pv = R[r][c]
        R[r] = [x / pv for x in R[r]]
        for i in range(n):
            if i != r and R[i][c] != 0.0:
                f = R[i][c]
                R[i] = [a - f * b for a, b in zip(R[i], R[r])]
        pivots.append(c)
        r += 1
    return R, pivots


def rank(A, tol=1e-10):
    return len(rref(A, tol)[1])


def null_space(A, tol=1e-10):
    """
    Basis for N(A) = { x : Ax = 0 }. One basis vector per FREE column: set that
    free variable to 1, the other free variables to 0, and solve for the pivot
    variables from the RREF.
    """
    R, piv = rref(A, tol)
    m = len(A[0])
    free = [c for c in range(m) if c not in piv]
    basis = []
    for f in free:
        x = [0.0] * m
        x[f] = 1.0
        for row, pc in enumerate(piv):
            x[pc] = -R[row][f]
        basis.append(x)
    return basis


def projection_matrix(A):
    """
    Orthogonal projector onto the column space of A (full column rank):
        P = A (A^T A)^-1 A^T
    P is symmetric and idempotent (P^2 = P); I - P projects onto the
    orthogonal complement N(A^T). The formula is fine for understanding; for
    computation use QR: P = Q Q^T, which never forms A^T A.
    """
    Q, _ = householder_qr(A)
    k = len(A[0])
    Qk = [row[:k] for row in Q]
    return matmul(Qk, transpose(Qk))


# =============================================================================
# 2. LINEAR SYSTEMS
# =============================================================================
def gaussian_solve(A, b, pivot=True):
    """
    Gaussian elimination, with or without PARTIAL PIVOTING (swap in the row
    with the largest-magnitude entry in the current column).

    Without pivoting, a tiny pivot creates huge multipliers; the subtraction
    then wipes out the information in the other rows (catastrophic
    cancellation), and the answer can be completely wrong even when the matrix
    is perfectly well conditioned. Partial pivoting keeps every multiplier
    <= 1 in magnitude. It is stable in practice (the worst case — exponential
    growth — exists but essentially never occurs in real problems).
    """
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        if pivot:
            p = max(range(c, n), key=lambda r: abs(M[r][c]))
            M[c], M[p] = M[p], M[c]
        if M[c][c] == 0.0:
            raise ZeroDivisionError("zero pivot")
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def lu_decompose(A):
    """
    PA = LU with partial pivoting: L unit lower triangular, U upper triangular,
    P a permutation. Factor once in O(n^3), then solve for each new right-hand
    side in O(n^2) by forward and back substitution — which is why you factor
    rather than invert. Returns (perm, L, U).
    """
    n = len(A)
    U = [row[:] for row in A]
    L = eye(n)
    perm = list(range(n))
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(U[r][c]))
        U[c], U[p] = U[p], U[c]
        perm[c], perm[p] = perm[p], perm[c]
        for k in range(c):
            L[c][k], L[p][k] = L[p][k], L[c][k]
        for r in range(c + 1, n):
            f = U[r][c] / U[c][c]
            L[r][c] = f
            for k in range(c, n):
                U[r][k] -= f * U[c][k]
    return perm, L, U


def cholesky(A):
    """
    A = L L^T for SYMMETRIC POSITIVE DEFINITE A. About half the cost of LU and
    needs no pivoting. It also doubles as the standard, cheapest test for
    positive definiteness: the factorization succeeds iff A is PD (up to
    rounding). Used for Gaussian log-likelihoods (log det = 2 sum log L_ii),
    sampling from N(mu, Sigma) as mu + L z, and solving normal equations.
    Raises ValueError if A is not positive definite.
    """
    n = len(A)
    L = zeros(n, n)
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0:
                    raise ValueError(f"not positive definite (pivot {s:.3g} at {i})")
                L[i][i] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def solve_lower(L, b):
    x = [0.0] * len(b)
    for i in range(len(b)):
        x[i] = (b[i] - sum(L[i][k] * x[k] for k in range(i))) / L[i][i]
    return x


def solve_upper(U, b):
    n = len(b)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(U[i][k] * x[k] for k in range(i + 1, n))) / U[i][i]
    return x


# =============================================================================
# 3. ORTHOGONALIZATION
# =============================================================================
def classical_gram_schmidt(A):
    """
    CGS: q_j = normalize( a_j - sum_i (q_i . a_j) q_i ), projecting the
    ORIGINAL column a_j onto every earlier q_i. Mathematically correct; in
    floating point the q's drift away from orthogonality roughly like
    kappa^2 * eps, and for ill-conditioned A they can end up far from
    orthogonal.
    """
    m, n = len(A), len(A[0])
    cols = transpose(A)
    Q, R = [], zeros(n, n)
    for j in range(n):
        v = cols[j][:]
        for i in range(j):
            R[i][j] = dot(Q[i], cols[j])
        for i in range(j):
            v = sub(v, scale(Q[i], R[i][j]))
        R[j][j] = norm2(v)
        Q.append(scale(v, 1 / R[j][j]))
    return transpose(Q), R


def modified_gram_schmidt(A):
    """
    MGS: the same arithmetic count, but each projection is subtracted from the
    CURRENT, partially orthogonalized vector rather than from the original
    column. That single reordering makes the loss of orthogonality grow like
    kappa * eps instead of kappa^2 * eps.
    """
    m, n = len(A), len(A[0])
    V = transpose(A)
    Q, R = [], zeros(n, n)
    for j in range(n):
        v = V[j][:]
        for i in range(j):
            R[i][j] = dot(Q[i], v)
            v = sub(v, scale(Q[i], R[i][j]))
        R[j][j] = norm2(v)
        Q.append(scale(v, 1 / R[j][j]))
    return transpose(Q), R


def householder_qr(A):
    """
    HOUSEHOLDER QR: zero out each column below the diagonal with a reflection
    H = I - 2 v v^T / (v^T v). Reflections are orthogonal by construction, so
    the computed Q is orthogonal to machine precision REGARDLESS of how ill
    conditioned A is. This is what LAPACK uses for QR and least squares.

    Returns Q (m x m) and R (m x n).
    """
    m, n = len(A), len(A[0])
    R = [row[:] for row in A]
    Q = eye(m)
    for k in range(min(m - 1, n)):
        x = [R[i][k] for i in range(k, m)]
        nx = norm2(x)
        if nx == 0.0:
            continue
        v = x[:]
        v[0] += math.copysign(nx, x[0])          # sign choice avoids cancellation
        vv = dot(v, v)
        if vv == 0.0:
            continue
        for j in range(k, n):
            s = 2 * sum(v[i - k] * R[i][j] for i in range(k, m)) / vv
            for i in range(k, m):
                R[i][j] -= s * v[i - k]
        for r in range(m):
            s = 2 * sum(Q[r][i] * v[i - k] for i in range(k, m)) / vv
            for i in range(k, m):
                Q[r][i] -= s * v[i - k]
    return Q, R


# =============================================================================
# 4. LEAST SQUARES
# =============================================================================
def lstsq_normal_equations(A, b):
    """
    Solve  A^T A x = A^T b  (via Cholesky). Fast and simple — but forming A^T A
    SQUARES the condition number: kappa(A^T A) = kappa(A)^2. Accuracy degrades
    like kappa^2 * eps, so at kappa ~ 1e8 every digit is gone.
    """
    At = transpose(A)
    L = cholesky(matmul(At, A))
    return solve_upper(transpose(L), solve_lower(L, matvec(At, b)))


def lstsq_qr(A, b):
    """
    Solve via A = QR:  R x = Q^T b. Never forms A^T A, so it works with
    kappa(A), not kappa(A)^2. The standard method; SVD is the choice when A may
    be rank deficient.
    """
    n = len(A[0])
    Q, R = householder_qr(A)
    qtb = matvec(transpose(Q), b)
    return solve_upper([row[:n] for row in R[:n]], qtb[:n])


def rel_err(x, x_true):
    return norm2(sub(x, x_true)) / norm2(x_true)


# =============================================================================
# 5. EIGENVALUES
# =============================================================================
def power_iteration(A, tol=1e-10, max_iter=100000, seed=0, v_true=None):
    """
    Repeatedly apply A and normalize. The component along the dominant
    eigenvector grows like |lambda_1|^k, the next like |lambda_2|^k, so the
    error shrinks like |lambda_2 / lambda_1|^k. Convergence speed is set by the
    EIGENGAP: a ratio of 0.5 needs ~33 iterations for 1e-10, a ratio of 0.99
    needs thousands. Returns (eigenvalue, eigenvector, iterations).
    """
    rng = random.Random(seed)
    v = [rng.gauss(0, 1) for _ in range(len(A))]
    v = scale(v, 1 / norm2(v))
    for it in range(1, max_iter + 1):
        w = matvec(A, v)
        w = scale(w, 1 / norm2(w))
        if dot(w, v) < 0:
            w = scale(w, -1)
        # eigenvectors are only defined up to sign, so compare against +/- truth
        if v_true is not None:
            err = min(norm2(sub(w, v_true)), norm2(add(w, v_true)))
        else:
            err = norm2(sub(w, v))
        v = w
        if err < tol:
            break
    lam = dot(v, matvec(A, v))                  # Rayleigh quotient
    return lam, v, it


def symmetric_eigen(A, sweeps=50, tol=1e-15):
    """
    Cyclic JACOBI eigenvalue method for SYMMETRIC matrices. Each rotation
    zeroes one off-diagonal pair; sweeping repeatedly drives A to diagonal.
    Returns (eigenvalues descending, eigenvectors as COLUMNS of V).

    THE SPECTRAL THEOREM guarantees what this computes: a real symmetric matrix
    has real eigenvalues and an ORTHONORMAL basis of eigenvectors,
    A = V diag(lambda) V^T. Jacobi is slower than QR-algorithm methods but
    simple and very accurate.
    """
    n = len(A)
    a = [row[:] for row in A]
    V = eye(n)
    for _ in range(sweeps):
        off = math.sqrt(sum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j))
        if off < tol * max(1.0, fro(a)):
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if a[p][q] == 0.0:
                    continue
                th = 0.5 * math.atan2(2 * a[p][q], a[q][q] - a[p][p])
                c, s = math.cos(th), math.sin(th)
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p], a[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k], a[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p], V[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    order = sorted(range(n), key=lambda i: -a[i][i])
    vals = [a[i][i] for i in order]
    vecs = [[V[r][i] for i in order] for r in range(n)]
    return vals, vecs


def symmetric_with_eigenvalues(vals, seed=0):
    rng = random.Random(seed)
    Q = random_orthogonal(len(vals), rng)
    QL = [[Q[i][k] * vals[k] for k in range(len(vals))] for i in range(len(vals))]
    return matmul(QL, transpose(Q)), Q


# =============================================================================
# 6. SVD
# =============================================================================
def svd_jacobi(A, sweeps=60, tol=1e-15):
    """
    ONE-SIDED JACOBI SVD (Hestenes). Apply rotations to pairs of COLUMNS of A
    until all columns are mutually orthogonal: then A V = U Sigma, the column
    norms are the singular values. It works on A directly — never on A^T A —
    so it keeps high RELATIVE accuracy even for tiny singular values.
    Returns (U, singular values descending, V) with A = U diag(s) V^T.
    """
    m, n = len(A), len(A[0])
    U = [row[:] for row in A]
    V = eye(n)
    for _ in range(sweeps):
        rotated = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha = sum(U[i][p] ** 2 for i in range(m))
                beta = sum(U[i][q] ** 2 for i in range(m))
                gamma = sum(U[i][p] * U[i][q] for i in range(m))
                if abs(gamma) <= tol * math.sqrt(alpha * beta) or gamma == 0.0:
                    continue
                rotated = True
                zeta = (beta - alpha) / (2 * gamma)
                t = math.copysign(1.0, zeta) / (abs(zeta) + math.sqrt(1 + zeta * zeta))
                c = 1 / math.sqrt(1 + t * t)
                s = c * t
                for i in range(m):
                    up, uq = U[i][p], U[i][q]
                    U[i][p], U[i][q] = c * up - s * uq, s * up + c * uq
                for i in range(n):
                    vp, vq = V[i][p], V[i][q]
                    V[i][p], V[i][q] = c * vp - s * vq, s * vp + c * vq
        if not rotated:
            break
    svals = [norm2([U[i][j] for i in range(m)]) for j in range(n)]
    order = sorted(range(n), key=lambda j: -svals[j])
    s = [svals[j] for j in order]
    Uo = [[(U[i][j] / svals[j]) if svals[j] > 0 else 0.0 for j in order] for i in range(m)]
    Vo = [[V[i][j] for j in order] for i in range(n)]
    return Uo, s, Vo


def singular_values_via_gram(A):
    """sigma_i = sqrt(eigenvalues of A^T A). Mathematically identical, but A^T A
    has eigenvalues sigma^2, and anything below ~eps * sigma_max^2 is lost in
    rounding — so singular values below ~sqrt(eps) * sigma_max ~ 1e-8 * sigma_max
    come back as noise."""
    vals, _ = symmetric_eigen(matmul(transpose(A), A))
    return [math.sqrt(max(v, 0.0)) for v in vals]


def low_rank_approx(U, s, V, k):
    m, n = len(U), len(V)
    return [[sum(U[i][r] * s[r] * V[j][r] for r in range(k)) for j in range(n)]
            for i in range(m)]


def pinv(A, rtol=1e-12):
    """
    MOORE-PENROSE PSEUDO-INVERSE  A+ = V diag(1/sigma_i) U^T, inverting only
    singular values above a tolerance (the rest are treated as exactly 0 —
    that truncation is what makes it stable). x = A+ b is the least-squares
    solution of MINIMUM NORM: for underdetermined systems, the unique solution
    with no component in the null space.
    """
    U, s, V = svd_jacobi(A)
    cut = rtol * s[0]
    m, n = len(A), len(A[0])
    return [[sum(V[i][r] * (1 / s[r]) * U[j][r] for r in range(len(s)) if s[r] > cut)
             for j in range(m)] for i in range(n)]


# =============================================================================
# 7. CONDITIONING
# =============================================================================
def cond2(A):
    """kappa_2(A) = sigma_max / sigma_min — how much relative errors in the data
    can be amplified in the solution of Ax = b."""
    _, s, _ = svd_jacobi(A)
    return s[0] / s[-1] if s[-1] > 0 else float("inf")


def hilbert(n):
    """H_ij = 1/(i + j + 1): the classic, innocent-looking, catastrophically ill-
    conditioned matrix (it arises from least-squares fitting of polynomials
    with a monomial basis on [0, 1])."""
    return [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]


def determinant(A):
    perm, L, U = lu_decompose(A)
    sign = 1.0
    p = perm[:]
    for i in range(len(p)):                       # parity of the permutation
        while p[i] != i:
            j = p[i]
            p[i], p[j] = p[j], p[i]
            sign = -sign
    d = sign
    for i in range(len(U)):
        d *= U[i][i]
    return d


# =============================================================================
# 8. DEMOS
# =============================================================================
def _hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def demo_subspaces():
    _hdr("1. SUBSPACES — rank, null space, and the four fundamental subspaces")
    # rank-2 matrix, 4 x 5: rows 3 and 4 are combinations of rows 1 and 2
    r1 = [1.0, 2.0, 0.0, 1.0, 3.0]
    r2 = [0.0, 1.0, 1.0, 2.0, 1.0]
    A = [r1, r2, add(r1, r2), sub(scale(r1, 2), r2)]
    rk = rank(A)
    N = null_space(A)
    Nt = null_space(transpose(A))
    print("  A is 4 x 5; rows 3 and 4 are combinations of rows 1 and 2.")
    print(f"    rank(A) = {rk}")
    print(f"    dim N(A)   = {len(N)}   -> rank + nullity = {rk} + {len(N)} = "
          f"{rk + len(N)} = number of columns (rank-nullity theorem)")
    print(f"    dim N(A^T) = {len(Nt)}   -> rank + {len(Nt)} = number of rows")
    worst_null = max(norm2(matvec(A, x)) for x in N)
    worst_orth = max(abs(dot(row, x)) for row in A for x in N)
    print(f"    max ||A x|| over the null-space basis     : {worst_null:.1e}")
    print(f"    max |row . x| (row space vs null space)   : {worst_orth:.1e}")
    print("  -> the null space is orthogonal to every row: N(A) is the orthogonal")
    print("     complement of the row space. Likewise N(A^T) is orthogonal to the")
    print("     column space. These four subspaces are the geometry of Ax = b.\n")

    rng = random.Random(5)
    B = random_matrix(8, 3, rng)
    P = projection_matrix(B)
    P2 = matmul(P, P)
    y = [rng.gauss(0, 1) for _ in range(8)]
    Py = matvec(P, y)
    resid = sub(y, Py)
    print("  Projection onto the column space of a random 8 x 3 matrix:")
    print(f"    ||P^2 - P||_F = {fro(mat_sub(P2, P)):.1e}   (idempotent: projecting twice"
          " changes nothing)")
    print(f"    ||P - P^T||_F = {fro(mat_sub(P, transpose(P))):.1e}   (symmetric: an"
          " ORTHOGONAL projector)")
    print(f"    max |B^T (y - Py)| = {max(abs(v) for v in matvec(transpose(B), resid)):.1e}"
          "   (residual orthogonal to every column)")
    print("  -> that last line IS least squares: the best fit leaves a residual")
    print("     orthogonal to the column space (the normal equations).")


def demo_systems():
    _hdr("2. LINEAR SYSTEMS — pivoting, factorization, positive definiteness")
    A = [[1e-17, 1.0], [1.0, 1.0]]
    b = [1.0, 2.0]
    print("  A = [[1e-17, 1], [1, 1]], b = [1, 2]; true solution ~ [1, 1]")
    print(f"    condition number of A      : {cond2(A):.3f}  (perfectly well conditioned)")
    print(f"    elimination WITHOUT pivoting: x = {gaussian_solve(A, b, pivot=False)}")
    print(f"    elimination WITH pivoting   : x = {gaussian_solve(A, b, pivot=True)}")
    print("  -> the problem is easy; the ALGORITHM failed. Dividing by the tiny")
    print("     pivot makes a multiplier of 1e17, and 1 - 1e17 rounds to -1e17,")
    print("     destroying the information in row 2. An unstable algorithm can ruin")
    print("     a well-conditioned problem; partial pivoting keeps multipliers <= 1.\n")

    rng = random.Random(6)
    M = random_matrix(6, 6, rng)
    perm, L, U = lu_decompose(M)
    PM = [M[i] for i in perm]
    print(f"  LU with partial pivoting on a random 6 x 6: ||PA - LU||_F = "
          f"{fro(mat_sub(PM, matmul(L, U))):.1e}")
    print(f"    largest |L_ij| = {max(abs(L[i][j]) for i in range(6) for j in range(i)):.3f}"
          "  (pivoting keeps every multiplier <= 1)")
    print("  -> factor once in O(n^3), then solve each new right-hand side in")
    print("     O(n^2). Forming the inverse costs more and is less accurate.\n")

    G = random_matrix(10, 4, rng)
    S = matmul(transpose(G), G)                      # Gram matrix: PSD, here PD
    Lc = cholesky(S)
    print(f"  Cholesky of a Gram matrix G^T G: ||S - L L^T||_F = "
          f"{fro(mat_sub(S, matmul(Lc, transpose(Lc)))):.1e}")
    ind = [[2.0, 1.0, 0.0], [1.0, -1.0, 1.0], [0.0, 1.0, 3.0]]
    vals, _ = symmetric_eigen(ind)
    try:
        cholesky(ind)
        res = "succeeded"
    except ValueError as e:
        res = f"FAILED — {e}"
    print(f"  Cholesky of a symmetric matrix with eigenvalues "
          f"{[round(v, 3) for v in vals]}: {res}")
    print("  -> Cholesky succeeds exactly when the matrix is positive definite,")
    print("     which makes it the cheapest PD test (much cheaper than eigenvalues).")


def demo_qr():
    _hdr("3. ORTHOGONALIZATION — the same math, very different accuracy")
    print("  Loss of orthogonality ||Q^T Q - I|| for 20 x 10 matrices with chosen")
    print("  condition numbers (seed 2):")
    print(f"  {'kappa':>9}{'classical GS':>15}{'modified GS':>14}{'Householder':>14}"
          f"{'kappa^2 eps':>13}{'kappa eps':>11}")
    for k in [1e2, 1e5, 1e8, 1e11]:
        A = matrix_with_singular_values(20, 10, log_spaced(1, 1 / k, 10), seed=2)
        cgs = orthogonality_loss(classical_gram_schmidt(A)[0])
        mgs = orthogonality_loss(modified_gram_schmidt(A)[0])
        hh = orthogonality_loss(householder_qr(A)[0])
        print(f"  {k:>9.0e}{cgs:>15.1e}{mgs:>14.1e}{hh:>14.1e}"
              f"{k * k * EPS:>13.1e}{k * EPS:>11.1e}")
    print("  -> classical Gram-Schmidt degrades roughly like kappa^2 * eps; at")
    print("     kappa = 1e11 its 'orthonormal' vectors are not orthogonal at all.")
    print("     Modified GS — the same arithmetic in a different ORDER — degrades")
    print("     like kappa * eps. Householder reflections are orthogonal by")
    print("     construction and stay at machine precision.")


def demo_least_squares():
    _hdr("4. LEAST SQUARES — never form A^T A (unless you know kappa is small)")
    rng = random.Random(3)
    print("  Consistent 30 x 8 systems b = A x_true with chosen kappa(A) (seed 3):")
    print(f"  {'kappa(A)':>10}{'kappa(A^T A)':>14}{'normal eqns error':>19}"
          f"{'QR error':>11}")
    for k in [1e2, 1e4, 1e6, 1e7, 1e8]:
        A = matrix_with_singular_values(30, 8, log_spaced(1, 1 / k, 8), seed=3)
        xt = [rng.gauss(0, 1) for _ in range(8)]
        b = matvec(A, xt)
        try:
            ne = f"{rel_err(lstsq_normal_equations(A, b), xt):.1e}"
        except ValueError:
            ne = "Cholesky fails"
        qr = rel_err(lstsq_qr(A, b), xt)
        print(f"  {k:>10.0e}{k * k:>14.0e}{ne:>19}{qr:>11.1e}")
    print("  -> the normal equations square the condition number, so their error")
    print("     grows like kappa^2 * eps: at kappa = 1e8 only ~1 correct digit")
    print("     remains, while QR still has ~9. (For consistent systems QR's error")
    print("     scales like kappa * eps; with large residuals a kappa^2 term")
    print("     appears for every method — it is then inherent in the problem.)")


def demo_eigen():
    _hdr("5. EIGENVALUES — the spectral theorem and why the eigengap matters")
    S, Q = symmetric_with_eigenvalues([3.0, 1.5, -0.5, 0.2], seed=7)
    vals, V = symmetric_eigen(S)
    VL = [[V[i][k] * vals[k] for k in range(4)] for i in range(4)]
    print("  Symmetric 4 x 4 with eigenvalues 3, 1.5, -0.5, 0.2 (seed 7), Jacobi:")
    print(f"    computed eigenvalues : {[round(v, 12) for v in vals]}")
    print(f"    ||V^T V - I||_F      : {orthogonality_loss(V):.1e}   (orthonormal eigenvectors)")
    print(f"    ||A - V L V^T||_F    : {fro(mat_sub(S, matmul(VL, transpose(V)))):.1e}")
    print("  -> real eigenvalues and an orthonormal eigenbasis: the spectral")
    print("     theorem for symmetric matrices. Covariance matrices, Hessians and")
    print("     kernel matrices are symmetric, so this is the case ML meets most.\n")

    print("  Power iteration: iterations to reach eigenvector error 1e-8 (seed 0):")
    print(f"  {'|lambda2/lambda1|':>18}{'iterations':>12}{'theory log(1e-8)/log(ratio)':>30}")
    for r in [0.5, 0.9, 0.99]:
        S2, Q2 = symmetric_with_eigenvalues([1.0, r, r * 0.5, 0.1, 0.05], seed=4)
        vt = [Q2[i][0] for i in range(5)]
        lam, v, it = power_iteration(S2, 1e-8, v_true=vt)
        print(f"  {r:>18}{it:>12}{math.ceil(math.log(1e-8) / math.log(r)):>30}")
    print("  -> the error shrinks like (lambda2/lambda1)^k. A small eigengap makes")
    print("     power iteration crawl — the same reason PageRank adds teleportation")
    print("     (it bounds the second eigenvalue) and why Krylov methods such as")
    print("     Lanczos/Arnoldi replace it in practice.")


def demo_svd():
    _hdr("6. SVD — the decomposition that works on everything")
    rng = random.Random(8)
    s_true = [1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    A = matrix_with_singular_values(12, 6, s_true, seed=8)
    _, s_j, _ = svd_jacobi(A)
    s_g = singular_values_via_gram(A)
    print("  Singular values of a 12 x 6 matrix built with known values (seed 8):")
    print(f"  {'true':>10}{'one-sided Jacobi':>18}{'sqrt(eig(A^T A))':>19}")
    for t, a, b in zip(s_true, s_j, s_g):
        print(f"  {t:>10.0e}{a:>18.6e}{b:>19.6e}")
    print("  -> forming A^T A squares the values: 1e-10 becomes 1e-20, far below")
    print("     rounding error on the largest entry (~1e-16), so everything under")
    print("     ~1e-8 comes back as noise. SVD algorithms work on A itself.\n")

    B = matrix_with_singular_values(15, 8, [9, 6, 4, 2, 1, 0.5, 0.2, 0.1], seed=9)
    U, s, V = svd_jacobi(B)
    print("  Eckart-Young: the best rank-k approximation keeps the top k singular")
    print("  values; its error is exactly sqrt(sum of the discarded sigma_i^2).")
    print(f"  {'k':>3}{'||A - A_k||_F':>16}{'formula':>11}{'best of 200 random rank-k':>28}")
    for k in [1, 2, 4]:
        Ak = low_rank_approx(U, s, V, k)
        err = fro(mat_sub(B, Ak))
        formula = math.sqrt(sum(x * x for x in s[k:]))
        best_rand = float("inf")
        for _ in range(200):                          # random rank-k competitors
            Qr, _ = householder_qr(random_matrix(8, k, rng))
            Qk = [row[:k] for row in Qr]
            Pk = matmul(Qk, transpose(Qk))
            best_rand = min(best_rand, fro(mat_sub(B, matmul(B, Pk))))
        print(f"  {k:>3}{err:>16.6f}{formula:>11.6f}{best_rand:>28.6f}")
    print("  -> the truncated SVD matches the formula exactly and beats every random")
    print("     rank-k projection. This is PCA, LSA, and low-rank compression.\n")

    C = [[1.0, 2.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0, 2.0], [1.0, 0.0, 1.0, 1.0, 0.0]]
    d = [4.0, 3.0, 2.0]
    Cp = pinv(C)
    x = matvec(Cp, d)
    N = null_space(C)
    x_other = add(x, scale(N[0], 0.7))
    print("  Underdetermined system: 3 equations, 5 unknowns (infinitely many solutions)")
    print(f"    pinv solution           : residual {norm2(sub(matvec(C, x), d)):.1e}, "
          f"||x|| = {norm2(x):.4f}")
    print(f"    pinv + null-space vector: residual "
          f"{norm2(sub(matvec(C, x_other), d)):.1e}, ||x|| = {norm2(x_other):.4f}")
    print("  -> both solve the system; the pseudo-inverse picks the MINIMUM-NORM")
    print("     solution, which has no component in the null space. (Gradient descent")
    print("     from zero on least squares converges to this same solution — an")
    print("     example of implicit regularization.)")


def demo_conditioning():
    _hdr("7. CONDITIONING — sensitivity of the problem, not the algorithm")
    print("  Hilbert matrices H_ij = 1/(i+j+1):")
    print(f"  {'n':>4}{'kappa computed':>17}{'published':>12}")
    for n, ref in [(4, 1.551e4), (6, 1.495e7), (8, 1.526e10), (10, 1.602e13)]:
        print(f"  {n:>4}{cond2(hilbert(n)):>17.4e}{ref:>12.3e}")
    print("  -> an innocent-looking 10 x 10 matrix loses ~13 of 16 digits.\n")

    A = matrix_with_singular_values(6, 6, [1, 0.5, 0.2, 0.1, 1e-3, 1e-6], seed=10)
    U, s, V = svd_jacobi(A)
    kappa = s[0] / s[-1]
    rng = random.Random(10)
    xt = [rng.gauss(0, 1) for _ in range(6)]
    b = matvec(A, xt)
    x0 = gaussian_solve(A, b)
    rel_b = 1e-10
    worst_dir = [U[i][0] for i in range(6)]          # b along u_1, perturb along u_n
    bw = [U[i][0] for i in range(6)]
    xw = gaussian_solve(A, bw)
    db = scale([U[i][5] for i in range(6)], rel_b * norm2(bw))
    xw2 = gaussian_solve(A, add(bw, db))
    amp_worst = (norm2(sub(xw2, xw)) / norm2(xw)) / rel_b
    amps = []
    for _ in range(20):
        r = [rng.gauss(0, 1) for _ in range(6)]
        db = scale(r, rel_b * norm2(b) / norm2(r))
        x1 = gaussian_solve(A, add(b, db))
        amps.append((norm2(sub(x1, x0)) / norm2(x0)) / rel_b)
    print(f"  Perturbing b by a relative 1e-10 in a system with kappa = {kappa:.2e}:")
    print(f"    worst case (b along u_1, perturbation along u_n): amplification "
          f"{amp_worst:.2e}")
    print(f"    20 random perturbations: amplification {min(amps):.1e} to {max(amps):.1e}")
    print("  -> the bound ||dx||/||x|| <= kappa ||db||/||b|| is attained in the worst")
    print("     direction. Random perturbations are amplified less — but conditioning")
    print("     is a property of the PROBLEM; no algorithm can beat it.\n")

    I100 = [[0.1 if i == j else 0.0 for j in range(100)] for i in range(100)]
    T = [[1.0, 1e8], [0.0, 1.0]]
    print("  The determinant is NOT a measure of near-singularity:")
    print(f"    0.1 * I (100 x 100): det = {determinant(I100):.1e}, "
          f"kappa = {cond2(I100):.1f}")
    print(f"    [[1, 1e8], [0, 1]]   : det = {determinant(T):.1f},   kappa = {cond2(T):.1e}")
    print("  -> a tiny determinant can belong to a perfect matrix, and det = 1 to a")
    print("     nearly singular one. The determinant scales with n and with units;")
    print("     the condition number (ratio of singular values) does not.")


def demo_pca():
    _hdr("8. PCA — covariance eigenvectors = right singular vectors")
    rng = random.Random(11)
    X = []
    for _ in range(300):
        a, b = rng.gauss(0, 3), rng.gauss(0, 1)
        X.append([a + 0.1 * rng.gauss(0, 1), 2 * a + 0.1 * rng.gauss(0, 1),
                  b + 0.1 * rng.gauss(0, 1), -b + 0.1 * rng.gauss(0, 1)])
    mu = [sum(r[j] for r in X) / 300 for j in range(4)]
    Xc = [[r[j] - mu[j] for j in range(4)] for r in X]
    C = [[sum(r[i] * r[j] for r in Xc) / 299 for j in range(4)] for i in range(4)]
    ev, evec = symmetric_eigen(C)
    _, s, V = svd_jacobi(Xc)
    ev_svd = [x * x / 299 for x in s]
    agree = max(abs(abs(dot([evec[i][0] for i in range(4)], [V[i][0] for i in range(4)])) - 1),
                abs(abs(dot([evec[i][1] for i in range(4)], [V[i][1] for i in range(4)])) - 1))
    print("  300 points in 4-D generated from 2 latent factors (seed 11):")
    print(f"    covariance eigenvalues     : {[round(v, 4) for v in ev]}")
    print(f"    sigma_i^2 / (n-1) from SVD : {[round(v, 4) for v in ev_svd]}")
    print(f"    principal directions agree up to sign: max |1 - |cos|| = {agree:.1e}")
    tot = sum(ev)
    print(f"    variance explained by 2 components: {(ev[0] + ev[1]) / tot:.4f}")
    print("  -> identical answers. In practice use the SVD of the centred data: it")
    print("     avoids forming X^T X (Part 6) and works when features outnumber rows.")


if __name__ == "__main__":
    print("=" * 78)
    print(" LINEAR ALGEBRA FOR ML FROM SCRATCH — accuracy measured, not assumed")
    print("=" * 78)
    demo_subspaces()
    demo_systems()
    demo_qr()
    demo_least_squares()
    demo_eigen()
    demo_svd()
    demo_conditioning()
    demo_pca()
    print("\n" + "=" * 78)
    print("All results use fixed seeds; rerunning reproduces them exactly.")
    print("=" * 78)
