# LINEAR ALGEBRA FOR ML — INTERVIEW STUDY NOTES

[ML](ML_Interview_Notes.md) · [DL](DL_Interview_Notes.md) · [RL](RL_Interview_Notes.md) · [NLP & LLMs](NLP_LLM_Interview_Notes.md) · [ML System Design](ML_System_Design_Notes.md) · [Probability & Stats](Probability_Statistics_Notes.md) · **Linear Algebra** · [Calculus](Calculus_Notes.md)

> Vectors, subspaces, decompositions, eigenvalues, SVD, and — the part most courses skip — how accurate each method is in floating point. Deepens the condensed review in the ML notes (Part 1.1).
> **Companion code:** [`linear_algebra_from_scratch.py`](linear_algebra_from_scratch.py) — pure Python, standard library only, runs in about a second. Every decomposition is implemented and its accuracy **measured** on matrices built with a chosen condition number.
> **Numbers quoted from the code** are labelled with their demo function and seed. Floating-point results depend on the exact matrices, so read them as orders of magnitude that illustrate a mechanism.
> **Math:** key equations are rendered (GitHub `math` blocks); ASCII code blocks are kept as compact reference.
> **Rapid-fire questions** are collapsible — answer first, then expand.

## Contents

  - [THE FIVE IDEAS THIS TOPIC TESTS](#the-five-ideas-this-topic-tests)
- [PART 0 — CHEAT SHEET](#part-0--cheat-sheet)
- [PART 1 — VECTORS, NORMS, INNER PRODUCTS](#part-1--vectors-norms-inner-products)
- [PART 2 — MATRICES AS LINEAR MAPS](#part-2--matrices-as-linear-maps)
- [PART 3 — THE FOUR FUNDAMENTAL SUBSPACES](#part-3--the-four-fundamental-subspaces)
- [PART 4 — SOLVING LINEAR SYSTEMS](#part-4--solving-linear-systems)
  - [4.1 Elimination and pivoting](#41-elimination-and-pivoting)
  - [4.2 LU and Cholesky](#42-lu-and-cholesky)
  - [4.3 Why you (almost) never compute an inverse](#43-why-you-almost-never-compute-an-inverse)
- [PART 5 — ORTHOGONALITY AND QR](#part-5--orthogonality-and-qr)
- [PART 6 — LEAST SQUARES](#part-6--least-squares)
- [PART 7 — EIGENVALUES AND EIGENVECTORS](#part-7--eigenvalues-and-eigenvectors)
  - [7.1 Definitions and facts](#71-definitions-and-facts)
  - [7.2 The spectral theorem](#72-the-spectral-theorem)
  - [7.3 Power iteration and the eigengap](#73-power-iteration-and-the-eigengap)
- [PART 8 — POSITIVE SEMIDEFINITE MATRICES AND QUADRATIC FORMS](#part-8--positive-semidefinite-matrices-and-quadratic-forms)
- [PART 9 — THE SINGULAR VALUE DECOMPOSITION](#part-9--the-singular-value-decomposition)
  - [9.1 Definition and geometry](#91-definition-and-geometry)
  - [9.2 Don't compute it from AᵀA](#92-dont-compute-it-from-aᵀa)
  - [9.3 Eckart–Young: optimal low-rank approximation](#93-eckartyoung-optimal-low-rank-approximation)
  - [9.4 Pseudo-inverse and minimum-norm solutions](#94-pseudo-inverse-and-minimum-norm-solutions)
  - [9.5 PCA via SVD](#95-pca-via-svd)
- [PART 10 — CONDITIONING AND NUMERICAL STABILITY](#part-10--conditioning-and-numerical-stability)
  - [10.1 Condition number](#101-condition-number)
  - [10.2 Conditioning vs stability](#102-conditioning-vs-stability)
  - [10.3 The determinant trap](#103-the-determinant-trap)
- [PART 11 — WHERE LINEAR ALGEBRA SHOWS UP IN ML](#part-11--where-linear-algebra-shows-up-in-ml)
- [PART 12 — RAPID-FIRE Q&A](#part-12--rapid-fire-qa)
- [PART 13 — COVERAGE INDEX](#part-13--coverage-index)
- [PART 14 — 7-DAY PLAN](#part-14--7-day-plan)
- [REFERENCES](#references)

---

## THE FIVE IDEAS THIS TOPIC TESTS

1. **A matrix is a linear map.** Multiplication is composition; columns are where the basis vectors land; rank is the dimension of what the map can reach.
2. **Four subspaces explain `Ax = b`.** Whether a solution exists, whether it is unique, and what least squares does are all statements about the column space, null space, and their orthogonal complements.
3. **Choose the decomposition by the structure.** LU for general square systems, Cholesky for symmetric positive definite, QR for least squares, eigendecomposition for symmetric matrices, SVD for everything else and for rank questions.
4. **Conditioning is a property of the problem; stability is a property of the algorithm.** An ill-conditioned problem cannot be solved accurately by anyone. A well-conditioned problem can still be ruined by an unstable algorithm.
5. **Never invert, never square.** Solving is cheaper and more accurate than inverting, and forming `AᵀA` squares the condition number.

---

# PART 0 — CHEAT SHEET

```
INVERTIBILITY (square A) — all equivalent
  A^-1 exists  <=>  det A != 0  <=>  rank A = n  <=>  N(A) = {0}
  <=>  columns independent  <=>  0 is not an eigenvalue  <=>  all sigma_i > 0

DETERMINANT & TRACE
  det(AB) = det A det B     det(A^T) = det A     det(A^-1) = 1/det A
  det A = product of eigenvalues      tr A = sum of eigenvalues
  |det A| = volume scaling of the map (NOT a measure of near-singularity)

DECOMPOSITIONS (cost for n x n, or m x n with m >= n)
  PA = LU              general square systems           ~ 2n^3/3 flops
  A = L L^T            symmetric positive definite       ~ n^3/3
  A = QR               least squares, orthogonal bases   ~ 2mn^2 - 2n^3/3 (Householder)
  A = V diag(l) V^T    symmetric (spectral theorem)      iterative, O(n^3)
  A = U diag(s) V^T    ANY matrix (SVD)                  iterative, O(mn^2), larger constant

CONDITIONING
  kappa(A) = sigma_max / sigma_min      kappa(A^T A) = kappa(A)^2
  ||dx||/||x|| <= kappa ||db||/||b||    digits lost ~ log10(kappa)

PSD (symmetric A) — all equivalent
  x^T A x >= 0 for all x  <=>  all eigenvalues >= 0  <=>  A = B^T B for some B
  (positive DEFINITE: > 0, strictly; Cholesky succeeds)

PROJECTION onto col(A), full column rank
  P = A (A^T A)^-1 A^T = Q Q^T      P^2 = P,  P^T = P
```

**Measured in the companion code — the numbers to quote** (from [`linear_algebra_from_scratch.py`](linear_algebra_from_scratch.py); seeds in each section):

| Claim | What the code shows |
|---|---|
| Pivoting matters even when the problem is easy | κ = 2.6, yet elimination without pivoting returns `[0, 1]` instead of `[1, 1]` |
| Gram-Schmidt order matters | at κ = 10⁸: classical GS orthogonality error **4×10⁻²**, modified **2×10⁻⁹**, Householder **1.4×10⁻¹⁵** |
| Normal equations square κ | at κ(A) = 10⁸: normal-equation error **0.13**, QR error **4×10⁻⁹** |
| SVD via `AᵀA` loses small values | true σ = 10⁻¹⁰ comes back as **7.8×10⁻⁹** from `AᵀA`, **1.0×10⁻¹⁰** from one-sided Jacobi |
| Eigengap sets power-iteration speed | ratio 0.5 → **28** iterations, 0.99 → **1,931** (theory 27, 1,833) |
| The determinant is not a singularity measure | `0.1·I₁₀₀` has det **10⁻¹⁰⁰** and κ = 1; `[[1,10⁸],[0,1]]` has det 1 and κ = **10¹⁶** |
| Eckart–Young is exact | rank-k SVD error equals `√Σσ²_tail` to 6 digits; best of 200 random rank-k projections is always worse |

---

# PART 1 — VECTORS, NORMS, INNER PRODUCTS

```math
\langle u,v\rangle = u^\top v = \lVert u\rVert\,\lVert v\rVert\cos\theta,\qquad
\lvert u^\top v\rvert \le \lVert u\rVert\,\lVert v\rVert\ \ \text{(Cauchy–Schwarz)},\qquad
\operatorname{proj}_v u = \frac{u^\top v}{v^\top v}\,v
```

```
L1   sum |v_i|           sparsity (Lasso), robust to outliers
L2   sqrt(sum v_i^2)     Euclidean distance, Ridge, the default
Linf max |v_i|           worst-case error bounds, adversarial perturbations
cosine similarity  u.v / (|u||v|) — direction only; used for embeddings
```
- **Orthogonal** vectors have `uᵀv = 0`; an **orthonormal** set is orthogonal with unit length. Orthonormal bases make coordinates trivial: the coordinate along `qᵢ` is just `qᵢᵀx`.
- **Norms in high dimensions:** random Gaussian vectors are nearly orthogonal and nearly equal in length — why cosine similarity of unrelated embeddings clusters near 0, and why distance-based methods degrade (see the ML notes on the curse of dimensionality).
- The L1-vs-L2 geometry behind sparsity is in the ML notes, Part 1.1.

---

# PART 2 — MATRICES AS LINEAR MAPS

```math
Ax = x_1 a_1 + x_2 a_2 + \dots + x_n a_n\qquad\text{(column picture: a combination of the columns)}
```

- **Column picture:** `Ax` is a linear combination of A's columns, so `Ax = b` asks whether `b` is in the column space. **Row picture:** each equation is a hyperplane; the solution is their intersection.
- **Matrix multiplication is composition:** `(AB)x = A(Bx)`. That is why it is associative but not commutative, and why `(AB)ᵀ = BᵀAᵀ` and `(AB)⁻¹ = B⁻¹A⁻¹` reverse the order.
- **Rank** = number of independent columns = number of independent rows = dimension of the column space. `rank(AB) ≤ min(rank A, rank B)` — the reason a low-rank factorization such as LoRA's `BA` can only represent low-rank updates.
- **Determinant** = signed volume scaling: `|det A|` is the factor by which A scales areas/volumes; the sign records orientation. `det = 0` means the map collapses some dimension.
- **Special matrices:** orthogonal (`QᵀQ = I`, preserves lengths and angles — rotations/reflections), symmetric (`A = Aᵀ`), diagonal (scales axes), triangular (solve by substitution), permutation (reorders), projection (`P² = P`).

---

# PART 3 — THE FOUR FUNDAMENTAL SUBSPACES

```math
\begin{aligned}
&\text{for } A \in \mathbb R^{m\times n} \text{ of rank } r:\\
&\operatorname{col}(A) \subseteq \mathbb R^m\ (\dim r),\quad N(A^\top) \subseteq \mathbb R^m\ (\dim m-r),\quad \operatorname{col}(A)\perp N(A^\top)\\
&\operatorname{row}(A) \subseteq \mathbb R^n\ (\dim r),\quad N(A) \subseteq \mathbb R^n\ (\dim n-r),\quad \operatorname{row}(A)\perp N(A)
\end{aligned}
```

**Rank–nullity:** `rank(A) + dim N(A) = n`. Measured in `demo_subspaces()`, [`linear_algebra_from_scratch.py`](linear_algebra_from_scratch.py): a 4×5 matrix whose rows 3–4 are combinations of rows 1–2 has rank 2, a 3-dimensional null space (2 + 3 = 5 columns) and a 2-dimensional left null space (2 + 2 = 4 rows); every null-space vector is exactly orthogonal to every row.

**What the subspaces say about `Ax = b`:**
```
solution exists         <=>  b in col(A)
solution unique         <=>  N(A) = {0}  (full column rank)
general solution        =   one particular solution + anything in N(A)
no exact solution       ->  least squares: project b onto col(A)  (Part 6)
many solutions          ->  minimum-norm solution has no N(A) component (Part 9)
```

**Numerical rank:** in floating point, "is this pivot zero?" needs a tolerance. The reliable way to measure rank is to count singular values above `tol × σ_max` (Part 9).

---

# PART 4 — SOLVING LINEAR SYSTEMS

## 4.1 Elimination and pivoting

Gaussian elimination reduces `A` to upper-triangular form, then back-substitutes. **Partial pivoting** — swapping in the row with the largest entry in the current column — keeps every multiplier ≤ 1 in magnitude.

Measured in `demo_systems()`, [`linear_algebra_from_scratch.py`](linear_algebra_from_scratch.py): for `A = [[10⁻¹⁷, 1], [1, 1]]`, `b = [1, 2]` (true solution ≈ `[1, 1]`, condition number **2.6**):

| Method | Result |
|---|---|
| without pivoting | `x = [0.0, 1.0]` — completely wrong |
| with partial pivoting | `x = [1.0, 1.0]` |

The problem is easy; the **algorithm** failed. Dividing by the tiny pivot creates a multiplier of 10¹⁷, and `1 − 10¹⁷` rounds to `−10¹⁷`, erasing row 2's information. This is the cleanest illustration of the **conditioning vs stability** distinction (Part 10). Partial pivoting is stable in practice; its theoretical worst case (exponential element growth) exists but essentially never appears in real problems.

## 4.2 LU and Cholesky

```math
PA = LU,\qquad A = LL^\top\ \ (A \text{ symmetric positive definite})
```

- **Factor once, solve many times:** `O(n³)` to factor, then `O(n²)` per right-hand side by forward and back substitution. In `demo_systems()` a random 6×6 gives `‖PA − LU‖ = 5×10⁻¹⁶` with every multiplier ≤ 0.98.
- **Cholesky** costs about half of LU and needs no pivoting for SPD matrices. It is also the **cheapest test for positive definiteness**: it succeeds exactly when the matrix is PD. In the demo it factors a Gram matrix to machine precision and fails with a negative pivot on a symmetric matrix with eigenvalues (3.29, 2.22, −1.51).
- **Uses in ML:** Gaussian log-likelihoods (`log det Σ = 2 Σ log Lᵢᵢ`), sampling `x = μ + Lz` from `N(μ, Σ)`, Gaussian processes, and solving `(XᵀX + λI)w = Xᵀy` for ridge regression.

## 4.3 Why you (almost) never compute an inverse

To solve `Ax = b`, factor and substitute. Forming `A⁻¹` and multiplying costs more (about 3× the flops of LU) and is generally less accurate. `A⁻¹` is rarely needed as a matrix; when you see `A⁻¹B` in a formula, read it as "solve `AX = B`". Exceptions: tiny matrices, or when the inverse's entries themselves are the output (e.g., a precision matrix you need to inspect).

---

# PART 5 — ORTHOGONALITY AND QR

```math
Q^\top Q = I \ \Rightarrow\ \lVert Qx\rVert_2 = \lVert x\rVert_2,\qquad A = QR,\qquad
H = I - 2\frac{vv^\top}{v^\top v}\ \ \text{(Householder reflection)}
```

Orthogonal transformations preserve lengths, so they never amplify rounding errors — which is why stable algorithms are built from them.

Three ways to compute the same QR, measured in `demo_qr()`, [`linear_algebra_from_scratch.py`](linear_algebra_from_scratch.py) — orthogonality error `‖QᵀQ − I‖` for 20×10 matrices with chosen condition number (seed 2):

| κ | classical Gram-Schmidt | modified Gram-Schmidt | Householder | κ²·ε | κ·ε |
|---|---|---|---|---|---|
| 10² | 7.4×10⁻¹⁴ | 4.1×10⁻¹⁵ | 1.6×10⁻¹⁵ | 2.2×10⁻¹² | 2.2×10⁻¹⁴ |
| 10⁵ | 1.5×10⁻⁷ | 1.7×10⁻¹² | 1.5×10⁻¹⁵ | 2.2×10⁻⁶ | 2.2×10⁻¹¹ |
| 10⁸ | **4.0×10⁻²** | 2.1×10⁻⁹ | 1.4×10⁻¹⁵ | 2.2 | 2.2×10⁻⁸ |
| 10¹¹ | **2.9** | 8.1×10⁻⁷ | 1.4×10⁻¹⁵ | 2.2×10⁶ | 2.2×10⁻⁵ |

- **Classical GS** projects each original column onto all earlier `q`s; its loss of orthogonality grows roughly like κ²·ε — at κ = 10¹¹ the "orthonormal" vectors aren't orthogonal at all.
- **Modified GS** does the same arithmetic in a different **order** (subtracting each projection from the partially updated vector) and degrades like κ·ε.
- **Householder** applies exact reflections and stays at machine precision regardless of κ. It is what LAPACK uses. (Re-orthogonalizing — running Gram-Schmidt twice — also rescues GS, and is used in Krylov methods.)

---

# PART 6 — LEAST SQUARES

```math
\min_x\lVert Ax-b\rVert_2^2 \ \Rightarrow\ A^\top(b - A\hat x) = 0\ \Rightarrow\ A^\top A\,\hat x = A^\top b,
\qquad \text{via QR: } R\hat x = Q^\top b
```

**Geometry:** `Ax̂` is the orthogonal projection of `b` onto the column space, so the residual is orthogonal to every column (demo: `max|Bᵀ(y − Py)| = 1.5×10⁻¹⁵`). The normal equations are just that orthogonality condition written out.

**Normal equations vs QR**, measured in `demo_least_squares()`, [`linear_algebra_from_scratch.py`](linear_algebra_from_scratch.py) — consistent 30×8 systems `b = A x_true` with chosen κ (seed 3), relative error in `x`:

| κ(A) | κ(AᵀA) | normal equations | QR |
|---|---|---|---|
| 10² | 10⁴ | 7.0×10⁻¹⁴ | 3.4×10⁻¹⁵ |
| 10⁴ | 10⁸ | 8.0×10⁻¹⁰ | 2.0×10⁻¹³ |
| 10⁶ | 10¹² | 1.2×10⁻⁵ | 6.7×10⁻¹² |
| 10⁸ | 10¹⁶ | **1.3×10⁻¹** | 3.8×10⁻⁹ |

Forming `AᵀA` squares the condition number, so the normal equations' error grows like κ²·ε: at κ = 10⁸ about one correct digit remains, while QR keeps about nine. Caveats worth stating in an interview:
- The normal equations are fine — and fastest — when κ is modest (standardized features, not too collinear), which is common.
- For **consistent** systems QR's error scales like κ·ε. With a **large residual** a κ² term appears for every method; it is then inherent in the problem, not the algorithm.
- For **rank-deficient or nearly rank-deficient** `A`, use the SVD / pseudo-inverse (Part 9) or regularize: ridge regression solves `(AᵀA + λI)x = Aᵀb`, which caps the condition number at `(σ²_max + λ)/(σ²_min + λ)`.

---

# PART 7 — EIGENVALUES AND EIGENVECTORS

## 7.1 Definitions and facts

```math
Av = \lambda v,\qquad \det(A-\lambda I) = 0,\qquad \sum_i\lambda_i = \operatorname{tr}A,\qquad \prod_i\lambda_i = \det A
```

- An eigenvector is a direction the map only **stretches** (by λ), never rotates.
- `A` is **diagonalizable**, `A = VΛV⁻¹`, iff it has n linearly independent eigenvectors; n distinct eigenvalues suffice. Some matrices are not (e.g. `[[1,1],[0,1]]`, a shear).
- Real non-symmetric matrices can have **complex** eigenvalues (a rotation has none that are real).
- Powers are easy in the eigenbasis: `Aᵏ = VΛᵏV⁻¹`. Long-run behaviour of repeated maps — Markov chains, RNN hidden-state recursions, gradient descent on a quadratic — is governed by the largest `|λ|`.

## 7.2 The spectral theorem

```math
A = A^\top \in \mathbb R^{n\times n}\ \Rightarrow\ A = V\Lambda V^\top,\quad V^\top V = I,\quad \Lambda \text{ real diagonal}
```

A real **symmetric** matrix has real eigenvalues and an **orthonormal** eigenbasis. Covariance matrices, Hessians, Gram/kernel matrices, and graph Laplacians are all symmetric — which is why this case dominates ML. Measured in `demo_eigen()` (seed 7): Jacobi recovers eigenvalues 3, 1.5, 0.2, −0.5 exactly, with `‖VᵀV − I‖ = 8×10⁻¹⁶` and `‖A − VΛVᵀ‖ = 2×10⁻¹⁵`.

**Rayleigh quotient** `R(x) = xᵀAx / xᵀx` lies between `λ_min` and `λ_max`, attaining them at the eigenvectors — the variational characterization behind PCA ("maximize variance `uᵀΣu` subject to `‖u‖ = 1`").

## 7.3 Power iteration and the eigengap

Repeatedly apply `A` and normalize; the error shrinks like `|λ₂/λ₁|ᵏ`. Measured in `demo_eigen()` (seed 0), iterations to reach eigenvector error 10⁻⁸:

| \|λ₂/λ₁\| | iterations | theory `log(10⁻⁸)/log(ratio)` |
|---|---|---|
| 0.5 | 28 | 27 |
| 0.9 | 185 | 175 |
| 0.99 | **1,931** | 1,833 |

A small **eigengap** makes power iteration crawl. PageRank's teleportation term guarantees the second eigenvalue is at most the damping factor (0.85), which bounds the gap and hence the convergence time. Production codes use Krylov methods (Lanczos for symmetric, Arnoldi otherwise) or the QR algorithm instead.

---

# PART 8 — POSITIVE SEMIDEFINITE MATRICES AND QUADRATIC FORMS

```math
A \succeq 0 \iff x^\top A x \ge 0\ \ \forall x \iff \lambda_i(A) \ge 0\ \ \forall i \iff A = B^\top B \text{ for some } B
```

(for symmetric A; **positive definite** means strictly `> 0` for `x ≠ 0`, all `λᵢ > 0`, Cholesky succeeds).

- **Always PSD:** covariance matrices, Gram matrices `XᵀX`, valid kernel matrices (Mercer's condition *is* PSD-ness), `BᵀB` for any B.
- **Geometry:** `{x : xᵀAx ≤ 1}` for PD A is an ellipsoid whose axes are the eigenvectors, with semi-axis lengths `1/√λᵢ`. The Mahalanobis distance `√((x−μ)ᵀΣ⁻¹(x−μ))` measures distance in units of this ellipsoid.
- **Optimization:** a twice-differentiable function is convex iff its Hessian is PSD everywhere; at a critical point, a PD Hessian means a strict local minimum, an indefinite one a saddle (see the [Calculus notes](Calculus_Notes.md), Parts 7–8).
- **Testing PD in practice:** attempt Cholesky (cheapest). Checking the sign of the determinant is **not** sufficient, and checking leading principal minors (Sylvester's criterion) is numerically poor.

---

# PART 9 — THE SINGULAR VALUE DECOMPOSITION

## 9.1 Definition and geometry

```math
A = U\Sigma V^\top,\quad U^\top U = I,\ V^\top V = I,\ \Sigma = \operatorname{diag}(\sigma_1\ge\sigma_2\ge\dots\ge 0),\qquad
A^\top A = V\Sigma^2V^\top,\quad AA^\top = U\Sigma^2U^\top
```

**Every** matrix has an SVD. Geometrically, any linear map is a rotation (`Vᵀ`), then an axis-aligned stretch (`Σ`), then another rotation (`U`). The singular values are the stretch factors; `rank = #{σᵢ > 0}`; `‖A‖₂ = σ₁`; `‖A‖_F = √Σσᵢ²`.

## 9.2 Don't compute it from AᵀA

Measured in `demo_svd()`, [`linear_algebra_from_scratch.py`](linear_algebra_from_scratch.py) — a 12×6 matrix built with known singular values (seed 8):

| true σ | one-sided Jacobi on A | √eig(AᵀA) |
|---|---|---|
| 10⁻⁴ | 1.000000×10⁻⁴ | 1.000000×10⁻⁴ |
| 10⁻⁶ | 1.000000×10⁻⁶ | 9.999920×10⁻⁷ |
| 10⁻⁸ | 1.000000×10⁻⁸ | **8.40×10⁻⁹** |
| 10⁻¹⁰ | 9.999999×10⁻¹¹ | **7.85×10⁻⁹** |

`AᵀA` has eigenvalues σ²; 10⁻²⁰ is far below rounding error on the largest entry, so singular values below about `√ε·σ₁ ≈ 10⁻⁸σ₁` come back as noise. SVD algorithms (Golub–Kahan bidiagonalization, one-sided Jacobi) work on `A` itself.

## 9.3 Eckart–Young: optimal low-rank approximation

```math
A_k = \sum_{i=1}^{k}\sigma_i u_i v_i^\top = \arg\min_{\operatorname{rank}(B)\le k}\lVert A-B\rVert_F,\qquad
\lVert A-A_k\rVert_F = \sqrt{\textstyle\sum_{i>k}\sigma_i^2},\qquad \lVert A-A_k\rVert_2 = \sigma_{k+1}
```

Measured in `demo_svd()` (15×8 matrix, seed 9):

| k | ‖A − A_k‖_F | formula | best of 200 random rank-k projections |
|---|---|---|---|
| 1 | 7.569676 | 7.569676 | 8.730132 |
| 2 | 4.615192 | 4.615192 | 7.392752 |
| 4 | 1.140175 | 1.140175 | 5.139714 |

The basis of PCA, latent semantic analysis, low-rank compression, matrix completion, and the intuition behind LoRA (fine-tuning updates are assumed to be approximately low rank).

## 9.4 Pseudo-inverse and minimum-norm solutions

```math
A^+ = V\Sigma^+U^\top,\qquad \Sigma^+_{ii} = \begin{cases}1/\sigma_i & \sigma_i > \text{tol}\\ 0 & \text{otherwise}\end{cases}
```

`x = A⁺b` is the least-squares solution of **minimum norm**: when there are many solutions, it is the one with no component in the null space. Measured in `demo_svd()` — 3 equations, 5 unknowns: the pseudo-inverse solution has residual 4×10⁻¹⁶ and norm **1.6571**; adding a null-space vector still solves the system but raises the norm to 1.9303. Gradient descent on least squares started from zero converges to this same minimum-norm solution — a simple example of **implicit regularization**. The tolerance on small σ is what makes the pseudo-inverse stable; inverting tiny singular values amplifies noise enormously (truncated SVD is itself a regularizer, akin to ridge).

## 9.5 PCA via SVD

For centred data `X_c = UΣVᵀ`, the principal directions are the columns of V and the explained variances are `σᵢ²/(n−1)` — identical to the covariance eigendecomposition. Measured in `demo_pca()` (300 points, 4-D, 2 latent factors, seed 11): both routes give variances (45.7078, 1.951, 0.0118, 0.0085) and principal directions agreeing to 2×10⁻¹⁶; two components explain 99.96% of variance. Prefer the SVD route: it never forms `XᵀX` (Part 9.2) and works when features outnumber samples.

---

# PART 10 — CONDITIONING AND NUMERICAL STABILITY

## 10.1 Condition number

```math
\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}},\qquad
\frac{\lVert\delta x\rVert}{\lVert x\rVert} \le \kappa(A)\,\frac{\lVert\delta b\rVert}{\lVert b\rVert}\quad\text{for } A(x+\delta x) = b+\delta b
```

κ measures how much the **problem** amplifies relative errors in the data. Measured in `demo_conditioning()`, [`linear_algebra_from_scratch.py`](linear_algebra_from_scratch.py) (κ = 10⁶, seed 10): a 10⁻¹⁰ relative perturbation of b along the worst direction (b along `u₁`, perturbation along `uₙ`) is amplified by exactly **1.00×10⁶**; twenty random perturbations were amplified by 7×10³ to 3.7×10⁵. The bound is attained in the worst direction; typical directions are milder, but you can't choose which errors your data has.

**Rule of thumb:** a stable algorithm loses about `log₁₀ κ` of the ~16 significant decimal digits of double precision.

**Hilbert matrices** `Hᵢⱼ = 1/(i+j+1)` — measured κ: n = 4: **1.5514×10⁴**; 6: 1.4951×10⁷; 8: 1.5258×10¹⁰; 10: **1.6025×10¹³**, matching published values to four digits. They arise from fitting polynomials in the monomial basis on [0, 1] — why polynomial regression uses orthogonal bases or scaled features.

## 10.2 Conditioning vs stability

```
CONDITIONING   property of the PROBLEM: how sensitive the exact answer is to
               the data. Nothing can fix it except reformulating the problem
               (better features, regularization, more informative data).
STABILITY      property of the ALGORITHM: whether it adds more error than the
               conditioning makes unavoidable. A BACKWARD-STABLE algorithm
               gives the exact answer to a slightly perturbed problem, so its
               error is ~ kappa * eps. Householder QR, LU with partial
               pivoting (in practice), and Cholesky are backward stable.
```
The pivoting demo (Part 4.1) is an unstable algorithm on a well-conditioned problem; Hilbert matrices are an ill-conditioned problem that ruins even stable algorithms.

## 10.3 The determinant trap

Measured in `demo_conditioning()`:

| Matrix | det | κ |
|---|---|---|
| `0.1 · I` (100×100) | **1.0×10⁻¹⁰⁰** | **1.0** |
| `[[1, 10⁸], [0, 1]]` | **1.0** | **1.0×10¹⁶** |

A tiny determinant can belong to a perfectly conditioned matrix and a determinant of 1 to a nearly singular one: the determinant scales with dimension and units (`det(cA) = cⁿ det A`), while κ is scale-invariant. **Use singular values or κ to judge near-singularity, never det.**

---

# PART 11 — WHERE LINEAR ALGEBRA SHOWS UP IN ML

| Concept | Where it appears |
|---|---|
| Projection, normal equations | linear regression; residuals orthogonal to features |
| QR / Cholesky | solving least squares and ridge; Gaussian likelihoods; GPs |
| Eigendecomposition | PCA, spectral clustering (graph Laplacian), PageRank, Markov chains, Hessian curvature |
| SVD / low rank | PCA, LSA, recommender matrix factorization, LoRA, compression, pseudo-inverse |
| PSD matrices | covariance, kernels (Mercer), Hessians of convex losses, Mahalanobis distance |
| Condition number | feature scaling, collinearity (VIF), optimizer speed (κ of the Hessian — see Calculus Part 8), ridge as conditioning fix |
| Orthogonal matrices | orthogonal initialization, rotations in RoPE, whitening, Householder layers |
| Spectral norm σ₁ | Lipschitz bounds, spectral normalization in GANs, exploding/vanishing gradients in RNNs (repeated multiplication by `W_h`) |
| Matrix products | every neural-network layer; attention `softmax(QKᵀ/√d)V`; the cost model of deep learning |

---

# PART 12 — RAPID-FIRE Q&A

<details>
<summary><b>Q: What does the rank of a matrix tell you?</b></summary>

The dimension of its column space (= row space): how many independent directions the map can produce. It determines whether `Ax = b` has solutions for every b (full row rank) and whether solutions are unique (full column rank).

</details>

<details>
<summary><b>Q: When does Ax = b have no solution, one, or infinitely many?</b></summary>

None if b is not in the column space; exactly one if b is in the column space and the null space is {0}; infinitely many if b is in the column space and the null space is non-trivial (any null-space vector can be added).

</details>

<details>
<summary><b>Q: Why shouldn't you compute a matrix inverse to solve a system?</b></summary>

Factoring (LU/Cholesky/QR) and substituting is cheaper and more accurate, and the factorization can be reused for new right-hand sides. The inverse is rarely needed as a matrix.

</details>

<details>
<summary><b>Q: Why use QR instead of the normal equations for least squares?</b></summary>

Forming AᵀA squares the condition number, so the normal equations lose roughly twice as many digits. In the companion code at κ = 10⁸ they give 13% error versus 4×10⁻⁹ for QR. The normal equations are fine when κ is modest.

</details>

<details>
<summary><b>Q: What is partial pivoting for?</b></summary>

Keeping elimination multipliers ≤ 1 so rounding errors are not amplified. Without it, a tiny pivot can destroy the answer even for a well-conditioned system (the companion demo gets `[0, 1]` instead of `[1, 1]` at κ = 2.6).

</details>

<details>
<summary><b>Q: State the spectral theorem and why it matters in ML.</b></summary>

A real symmetric matrix has real eigenvalues and an orthonormal eigenbasis, A = VΛVᵀ. Covariances, Hessians, kernel matrices, and graph Laplacians are symmetric, so PCA, curvature analysis, and spectral methods all rely on it.

</details>

<details>
<summary><b>Q: Eigendecomposition vs SVD?</b></summary>

Eigendecomposition exists only for (diagonalizable) square matrices and may be complex; the SVD exists for every matrix, with real non-negative singular values and orthonormal U and V. For a symmetric PSD matrix they coincide.

</details>

<details>
<summary><b>Q: How are singular values related to eigenvalues?</b></summary>

σᵢ(A) = √λᵢ(AᵀA); the right singular vectors are eigenvectors of AᵀA and the left ones of AAᵀ. But computing them that way loses accuracy for small σ (the companion code turns 10⁻¹⁰ into 7.8×10⁻⁹).

</details>

<details>
<summary><b>Q: What is the best rank-k approximation of a matrix?</b></summary>

The truncated SVD keeping the top k singular values (Eckart–Young), with Frobenius error √(Σ_{i>k} σᵢ²) and spectral error σ_{k+1}.

</details>

<details>
<summary><b>Q: How do you check whether a symmetric matrix is positive definite?</b></summary>

Try a Cholesky factorization — it succeeds iff the matrix is PD, and it's the cheapest test. Eigenvalues also work but cost more. A positive determinant alone is not enough.

</details>

<details>
<summary><b>Q: What does the condition number measure?</b></summary>

The worst-case amplification of relative errors from the data to the solution: ‖δx‖/‖x‖ ≤ κ‖δb‖/‖b‖, with κ = σ_max/σ_min. Roughly log₁₀κ decimal digits are lost.

</details>

<details>
<summary><b>Q: Is a matrix with a tiny determinant nearly singular?</b></summary>

Not necessarily. 0.1·I in 100 dimensions has determinant 10⁻¹⁰⁰ and condition number 1. Judge near-singularity by the smallest singular value relative to the largest.

</details>

<details>
<summary><b>Q: What is the pseudo-inverse and when do you use it?</b></summary>

A⁺ = VΣ⁺Uᵀ, inverting only singular values above a tolerance. It gives the minimum-norm least-squares solution, and handles rank-deficient and underdetermined problems.

</details>

<details>
<summary><b>Q: Why is classical Gram-Schmidt a bad idea numerically?</b></summary>

Its loss of orthogonality grows roughly like κ²·ε; at κ = 10¹¹ the companion code's Q is not orthogonal at all. Modified Gram-Schmidt (κ·ε) or Householder QR (machine precision) should be used.

</details>

<details>
<summary><b>Q: How does PCA use linear algebra?</b></summary>

Principal components are the eigenvectors of the covariance matrix — equivalently, the right singular vectors of the centred data — ordered by eigenvalue (explained variance). Compute via the SVD of the centred data for accuracy.

</details>

<details>
<summary><b>Q: What determines how fast power iteration converges?</b></summary>

The ratio |λ₂/λ₁|: the error shrinks like that ratio to the k-th power. At 0.99 the companion code needs ~1,900 iterations versus ~28 at 0.5.

</details>

<details>
<summary><b>Q: Why does ridge regression help with collinear features, in linear-algebra terms?</b></summary>

It solves (AᵀA + λI)x = Aᵀb, adding λ to every eigenvalue of AᵀA, which bounds the condition number and damps the directions with small singular values that would otherwise amplify noise.

</details>

---

# PART 13 — COVERAGE INDEX

| Concept | Implementation in [`linear_algebra_from_scratch.py`](linear_algebra_from_scratch.py) | Demo |
|---|---|---|
| Matrices with a chosen condition number | `matrix_with_singular_values`, `random_orthogonal` | used throughout |
| RREF, rank, null space | `rref`, `rank`, `null_space` | 1 |
| Four fundamental subspaces, rank-nullity | `demo_subspaces` | 1 |
| Orthogonal projection | `projection_matrix` | 1 |
| Gaussian elimination, with/without pivoting | `gaussian_solve` | 2 |
| LU with partial pivoting | `lu_decompose` | 2 |
| Cholesky and PD testing | `cholesky` | 2 |
| Classical GS, modified GS, Householder QR | `classical_gram_schmidt`, `modified_gram_schmidt`, `householder_qr` | 3 |
| Least squares: normal equations vs QR | `lstsq_normal_equations`, `lstsq_qr` | 4 |
| Symmetric eigendecomposition (Jacobi) | `symmetric_eigen` | 5 |
| Power iteration and eigengap | `power_iteration` | 5 |
| One-sided Jacobi SVD | `svd_jacobi` | 6 |
| SVD via AᵀA and its precision loss | `singular_values_via_gram` | 6 |
| Eckart–Young low-rank approximation | `low_rank_approx` | 6 |
| Pseudo-inverse, minimum-norm solution | `pinv` | 6 |
| Condition number, Hilbert matrices | `cond2`, `hilbert` | 7 |
| Perturbation amplification | `demo_conditioning` | 7 |
| Determinant via LU | `determinant` | 7 |
| PCA via covariance vs SVD | `demo_pca` | 8 |

**Covered in these notes, not implemented** — be ready to explain: Jordan form (non-diagonalizable matrices), the QR algorithm, Lanczos/Arnoldi, Golub–Kahan bidiagonalization, sparse and iterative solvers (conjugate gradient), Kronecker products, tensor decompositions.

---

# PART 14 — 7-DAY PLAN

```
Day 1  Vectors, norms, inner products; matrices as maps; rank. Part 1-2.
Day 2  Four fundamental subspaces, rank-nullity, solvability. Run demo 1.
Day 3  Elimination, pivoting, LU, Cholesky; why not invert. Run demo 2.
Day 4  Orthogonality, Gram-Schmidt vs Householder; least squares via QR
       vs normal equations. Run demos 3-4.
Day 5  Eigenvalues, spectral theorem, power iteration, PSD matrices. Demo 5.
Day 6  SVD: geometry, Eckart-Young, pseudo-inverse, PCA. Demos 6 and 8.
Day 7  Conditioning vs stability, determinant trap; the ML map (Part 11);
       rapid-fire questions closed-book. Demo 7.
```

**Self-test — from memory, can you:**
1. List six equivalent conditions for a square matrix to be invertible?
2. Name the four fundamental subspaces, their dimensions, and which pairs are orthogonal?
3. Explain why pivoting matters with a 2×2 example?
4. Say which decomposition to use for a general square system, an SPD system, least squares, and a rank-deficient problem?
5. Explain why the normal equations lose accuracy and quantify it?
6. State the spectral theorem and give three ML matrices it applies to?
7. Explain what controls power iteration's convergence speed?
8. Give four equivalent characterizations of a PSD matrix?
9. State Eckart–Young, including both error formulas?
10. Explain the minimum-norm property of the pseudo-inverse?
11. Distinguish conditioning from stability, with one example of each failing?
12. Explain why the determinant is a poor measure of near-singularity?

---

# REFERENCES

Primary sources for the claims above. Where these notes simplify, the source is authoritative.

- Strang. *Introduction to Linear Algebra*, 6th ed. Wellesley-Cambridge Press, 2023 — the four fundamental subspaces.
- Strang. *Linear Algebra and Learning from Data*. Wellesley-Cambridge Press, 2019.
- Trefethen & Bau. *Numerical Linear Algebra*. SIAM, 1997 — conditioning, stability, Gram-Schmidt vs Householder.
- Golub & Van Loan. *Matrix Computations*, 4th ed. Johns Hopkins University Press, 2013.
- Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM, 2002.
- Axler. *Linear Algebra Done Right*, 4th ed. Springer, 2024 (free online).
- Deisenroth, Faisal & Ong. *Mathematics for Machine Learning*. Cambridge University Press, 2020 (free online).
- Eckart & Young. The approximation of one matrix by another of lower rank. *Psychometrika*, 1936.
- Björck. Solving linear least squares problems by Gram-Schmidt orthogonalization. *BIT*, 1967.
- Demmel & Veselić. Jacobi's method is more accurate than QR. *SIAM J. Matrix Anal. Appl.*, 1992.
- Haveliwala & Kamvar. The second eigenvalue of the Google matrix. Stanford Technical Report, 2003.

---

[ML](ML_Interview_Notes.md) · [DL](DL_Interview_Notes.md) · [RL](RL_Interview_Notes.md) · [NLP & LLMs](NLP_LLM_Interview_Notes.md) · [ML System Design](ML_System_Design_Notes.md) · [Probability & Stats](Probability_Statistics_Notes.md) · **Linear Algebra** · [Calculus](Calculus_Notes.md)
