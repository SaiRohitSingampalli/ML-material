# CALCULUS FOR ML — INTERVIEW STUDY NOTES

[ML](ML_Interview_Notes.md) · [DL](DL_Interview_Notes.md) · [RL](RL_Interview_Notes.md) · [NLP & LLMs](NLP_LLM_Interview_Notes.md) · [ML System Design](ML_System_Design_Notes.md) · [Probability & Stats](Probability_Statistics_Notes.md) · [Linear Algebra](Linear_Algebra_Notes.md) · **Calculus**

> Derivatives, gradients, Jacobians and Hessians, the chain rule and automatic differentiation, matrix calculus, Taylor expansion, convexity, unconstrained and constrained optimization, integration, and floating-point pitfalls. Deepens the condensed review in the ML notes (Part 1.2).
> **Companion code:** [`calculus_from_scratch.py`](calculus_from_scratch.py) — pure Python, standard library only, runs in about a second. Every identity is checked against finite differences and every convergence rate is measured.
> **Numbers quoted from the code** are labelled with their demo function and seed; floating-point results are illustrations of a mechanism, and their exact digits depend on the test function.
> **Math:** key equations are rendered (GitHub `math` blocks); ASCII code blocks are kept as compact reference.
> **Rapid-fire questions** are collapsible — answer first, then expand.

## Contents

  - [THE FIVE IDEAS THIS TOPIC TESTS](#the-five-ideas-this-topic-tests)
- [PART 0 — CHEAT SHEET](#part-0--cheat-sheet)
- [PART 1 — DERIVATIVES AND LINEAR APPROXIMATION](#part-1--derivatives-and-linear-approximation)
- [PART 2 — GRADIENTS, JACOBIANS, HESSIANS](#part-2--gradients-jacobians-hessians)
- [PART 3 — THE CHAIN RULE AND AUTOMATIC DIFFERENTIATION](#part-3--the-chain-rule-and-automatic-differentiation)
  - [3.1 The multivariate chain rule](#31-the-multivariate-chain-rule)
  - [3.2 Forward vs reverse mode](#32-forward-vs-reverse-mode)
- [PART 4 — MATRIX CALCULUS](#part-4--matrix-calculus)
  - [4.1 Conventions and technique](#41-conventions-and-technique)
  - [4.2 Identities — checked](#42-identities--checked)
- [PART 5 — NUMERICAL DIFFERENTIATION AND GRADIENT CHECKING](#part-5--numerical-differentiation-and-gradient-checking)
- [PART 6 — TAYLOR EXPANSION](#part-6--taylor-expansion)
- [PART 7 — CONVEXITY](#part-7--convexity)
- [PART 8 — UNCONSTRAINED OPTIMIZATION](#part-8--unconstrained-optimization)
  - [8.1 Optimality conditions](#81-optimality-conditions)
  - [8.2 Gradient descent and the condition number](#82-gradient-descent-and-the-condition-number)
  - [8.3 Step size](#83-step-size)
  - [8.4 Newton's method](#84-newtons-method)
  - [8.5 Saddle points](#85-saddle-points)
- [PART 9 — CONSTRAINED OPTIMIZATION](#part-9--constrained-optimization)
  - [9.1 Lagrange multipliers](#91-lagrange-multipliers)
  - [9.2 KKT conditions (inequality constraints)](#92-kkt-conditions-inequality-constraints)
  - [9.3 Duality and practical methods](#93-duality-and-practical-methods)
- [PART 10 — INTEGRATION](#part-10--integration)
  - [10.1 Quadrature vs Monte Carlo](#101-quadrature-vs-monte-carlo)
  - [10.2 The curse of dimensionality, measured](#102-the-curse-of-dimensionality-measured)
  - [10.3 Change of variables](#103-change-of-variables)
  - [10.4 Differentiating an expectation](#104-differentiating-an-expectation)
- [PART 11 — NUMERICAL PITFALLS](#part-11--numerical-pitfalls)
- [PART 12 — RAPID-FIRE Q&A](#part-12--rapid-fire-qa)
- [PART 13 — COVERAGE INDEX](#part-13--coverage-index)
- [PART 14 — 7-DAY PLAN](#part-14--7-day-plan)
- [REFERENCES](#references)

---

## THE FIVE IDEAS THIS TOPIC TESTS

1. **A derivative is the best local linear approximation.** Gradients, Jacobians, and Taylor expansions are all ways of saying "near this point, the function is approximately linear (or quadratic)".
2. **The chain rule is all of backpropagation.** Reverse-mode automatic differentiation is the chain rule evaluated in the order that makes gradients cheap.
3. **Curvature governs optimization.** The Hessian's condition number decides how fast gradient descent converges, how large a step can be, and whether a critical point is a minimum or a saddle.
4. **Constraints become multipliers.** Lagrange multipliers and KKT conditions turn constrained problems into unconstrained ones — and the multipliers have meaning (prices, support vectors).
5. **Floating point is not real arithmetic.** Correct formulas can give wrong answers; finite differences, variances, and softmax all need care.

---

# PART 0 — CHEAT SHEET

```
DERIVATIVES
  d/dx x^n = n x^(n-1)    d/dx e^x = e^x    d/dx ln x = 1/x
  d/dx sigmoid(x) = s(1-s)   d/dx tanh x = 1 - tanh^2 x   d/dx softplus = sigmoid
  product (fg)' = f'g + fg'   quotient (f/g)' = (f'g - fg')/g^2
  chain   (f(g(x)))' = f'(g(x)) g'(x)

MULTIVARIABLE  (f: R^n -> R, F: R^n -> R^m)
  gradient  grad f in R^n          direction of steepest ascent, normal to level sets
  Jacobian  J_F in R^(m x n)       J_ij = dF_i/dx_j ; chain rule: J_(F.G) = J_F J_G
  Hessian   H in R^(n x n)         H_ij = d^2 f/dx_i dx_j ; symmetric if C^2

MATRIX CALCULUS (denominator layout: gradient has the shape of the variable)
  d/dx a'x = a              d/dx x'Ax = (A + A')x   (= 2Ax only if A symmetric)
  d/dx ||Ax - b||^2 = 2A'(Ax - b)
  d/dX tr(AX) = A'          d/dX log|det X| = X^{-T}    d/dX ||X||_F^2 = 2X
  softmax Jacobian = diag(p) - pp'      d/dz [-log softmax(z)_y] = p - e_y

TAYLOR
  f(x+h) = f(x) + f'(x)h + f''(x)h^2/2 + O(h^3)
  f(x+d) ~ f(x) + grad'd + 1/2 d'Hd   (the model behind GD and Newton)

OPTIMIZATION (L-smooth, mu-strongly convex, kappa = L/mu)
  GD, lr = 1/L          error x (1 - 1/kappa) per step   ->  ~kappa ln(1/tol) steps
  stability             need lr < 2/L
  heavy ball / Nesterov ~sqrt(kappa) ln(1/tol) steps
  Newton                quadratic convergence (locally); 1 step on a quadratic

CONSTRAINED
  Lagrangian  L = f + sum lam_i g_i + sum nu_j h_j
  KKT: stationarity, primal feasibility, lam >= 0, lam_i g_i = 0 (compl. slackness)

NUMERICAL DIFFERENTIATION
  forward diff  error ~ h + eps/h  -> best h ~ 1e-8, error ~ 1e-8
  central diff  error ~ h^2 + eps/h -> best h ~ 6e-6, error ~ 1e-10/1e-11
  complex step  Im f(x+ih)/h        -> machine precision for tiny h
```

**Measured in the companion code — the numbers to quote** (from [`calculus_from_scratch.py`](calculus_from_scratch.py)):

| Claim | What the code shows |
|---|---|
| Smaller h is not always better | forward-difference error **2.3×10⁻⁸** at h = 10⁻⁸, rising to **8×10⁻³** at h = 10⁻¹⁴ |
| Reverse mode makes gradients cheap | gradient of an n-input function: forward mode **n×** the function cost, reverse mode **2.25×** for n = 10 to 1,000 |
| A common matrix-calculus error | "∂(xᵀAx)/∂x = 2Ax" for non-symmetric A is off by **92%**; the correct `(A+Aᵀ)x` matches to 7×10⁻¹¹ |
| GD speed ∝ condition number | iterations to 10⁻⁶: **132, 1,375, 13,809, 138,149** for κ = 10 … 10⁴ (formula matches exactly) |
| Step size limit 2/L | lr = 1.99/L converges, lr = 2.01/L diverges |
| Newton converges quadratically | errors 4×10⁻², 9×10⁻⁴, 4×10⁻⁷, 8×10⁻¹⁴ |
| Multiplier = shadow price | numerical `∂f*/∂c = 5.000000 = λ` |
| Catastrophic cancellation | `1 − cos(10⁻⁸)` computes as **0**; `2 sin²(x/2)` gives the correct 5×10⁻¹⁷ |

---

# PART 1 — DERIVATIVES AND LINEAR APPROXIMATION

```math
f'(x) = \lim_{h\to 0}\frac{f(x+h)-f(x)}{h},\qquad f(x+h) = f(x) + f'(x)\,h + o(h)
```

- The derivative is the slope of the **best linear approximation** at a point — the definition that generalizes to gradients and Jacobians.
- **Differentiable ⇒ continuous**, not conversely (`|x|` at 0).
- **Non-differentiable points in ML:** ReLU at 0, `|x|` (L1 penalty), hinge loss at the margin, max-pooling ties. Optimizers use a **subgradient** — any slope `g` with `f(y) ≥ f(x) + g(y−x)` for all y (for convex f). The subdifferential of `|x|` at 0 is the whole interval [−1, 1], which is why L1 can hold a weight exactly at zero. Frameworks simply pick a value (ReLU′(0) = 0).
- **Activation derivatives to know:** `σ′ = σ(1−σ)` (at most 0.25 — a source of vanishing gradients), `tanh′ = 1 − tanh²` (at most 1), `ReLU′ ∈ {0, 1}`, softplus′ = sigmoid.

---

# PART 2 — GRADIENTS, JACOBIANS, HESSIANS

```math
\nabla f(x) = \begin{bmatrix}\frac{\partial f}{\partial x_1}\\ \vdots\\ \frac{\partial f}{\partial x_n}\end{bmatrix},\qquad
D_u f(x) = \nabla f(x)^\top u,\qquad
J_{ij} = \frac{\partial F_i}{\partial x_j},\qquad
H_{ij} = \frac{\partial^2 f}{\partial x_i\,\partial x_j}
```

- The **directional derivative** `∇fᵀu` is maximized over unit `u` by `u = ∇f/‖∇f‖` (Cauchy–Schwarz): the gradient is the direction of **steepest ascent**, and gradient descent moves along `−∇f`.
- The gradient is **perpendicular to level sets** (contours of constant f) — the geometric fact behind Lagrange multipliers (Part 9).
- The **Jacobian** of `F: ℝⁿ → ℝᵐ` is the m×n matrix of first derivatives; it is the linear map that best approximates F locally. Its determinant measures local volume change (Part 10).
- The **Hessian** is symmetric when the second partials are continuous (Schwarz/Clairaut). Its eigenvalues are the curvatures along its eigenvector directions: all positive → local bowl; mixed signs → saddle.
- Size matters: for a model with n parameters the gradient has n entries but the Hessian has n², which is why second-order methods are rarely used directly for large networks (Hessian-vector products, `Hv`, can be computed at the cost of about two gradients without forming H).

---

# PART 3 — THE CHAIN RULE AND AUTOMATIC DIFFERENTIATION

## 3.1 The multivariate chain rule

```math
\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_j}\,\frac{\partial y_j}{\partial x_i},\qquad
J_{f\circ g}(x) = J_f\big(g(x)\big)\,J_g(x)
```

The derivative of a composition is the **product of Jacobians**. A deep network is a long composition `f_L ∘ … ∘ f_1`, so its gradient is a product of L Jacobians — which is also why gradients can vanish or explode (repeated products of matrices with norms below or above 1; see the [DL notes](DL_Interview_Notes.md)).

## 3.2 Forward vs reverse mode

The product `J_L ⋯ J_2 J_1` can be evaluated right-to-left (**forward mode**: push a tangent vector through with the function) or left-to-right (**reverse mode**: pull a gradient back from the output).

Measured in `demo_autodiff()`, [`calculus_from_scratch.py`](calculus_from_scratch.py) — the full gradient of `f(x) = x₀² + Σ xᵢ sin(xᵢ₋₁)`, cost in primitive operations (seed 0):

| n | one function evaluation | forward mode | reverse mode | forward ÷ f | reverse ÷ f |
|---|---|---|---|---|---|
| 10 | 38 | 380 | 85 | 10.0 | 2.24 |
| 100 | 398 | 39,800 | 895 | 100.0 | 2.25 |
| 1,000 | 3,998 | 3,998,000 | 8,995 | **1000.0** | **2.25** |

Both are exact (error 0). Forward mode yields one directional derivative per pass, so a full gradient costs n passes. Reverse mode gets every partial derivative from one backward pass at a **constant multiple** of the function cost — the "cheap gradient principle" (Baur–Strassen), and the reason backpropagation can train billion-parameter models.

```
use FORWARD mode   few inputs, many outputs; Jacobian-vector products (Jv)
use REVERSE mode   many inputs, one scalar output (losses!); vector-Jacobian
                   products (v'J)
reverse mode cost  must STORE every intermediate value for the backward pass
                   -> memory, and gradient checkpointing to trade compute for it
```

Automatic differentiation is **not** symbolic differentiation (no expression swell) and **not** numerical differentiation (no truncation error): it applies exact derivative rules to the actual sequence of operations executed.

---

# PART 4 — MATRIX CALCULUS

## 4.1 Conventions and technique

Two layout conventions exist. These notes use **denominator layout**: the gradient of a scalar with respect to a vector or matrix has the **same shape** as the variable. Most ML papers and code follow this; shape-checking catches most errors.

The reliable derivation technique is **differentials**: expand `f(X + dX)` to first order and read off the term linear in `dX`, using `tr(AB) = tr(BA)`.

```math
d(x^\top A x) = dx^\top A x + x^\top A\,dx = x^\top (A + A^\top)\,dx \ \Rightarrow\ \nabla_x (x^\top A x) = (A + A^\top)\,x
```

## 4.2 Identities — checked

Measured in `demo_matrix_calculus()`, [`calculus_from_scratch.py`](calculus_from_scratch.py) — each formula compared with a central-difference gradient on random 4×4 data (seed 0):

| Identity | Relative error |
|---|---|
| ∂(xᵀAx)/∂x = (A + Aᵀ)x | 6.5×10⁻¹¹ |
| ∂‖Ax − b‖²/∂x = 2Aᵀ(Ax − b) | 1.4×10⁻¹⁰ |
| ∂tr(AX)/∂X = Aᵀ | 8.4×10⁻¹¹ |
| ∂log\|det X\|/∂X = X⁻ᵀ | 2.7×10⁻¹⁰ |
| ∂‖X‖²_F/∂X = 2X | 2.6×10⁻¹⁰ |
| softmax Jacobian = diag(p) − ppᵀ | 1.4×10⁻¹⁰ |
| ∂[−log softmax(z)_y]/∂z = p − e_y | 1.9×10⁻¹⁰ |
| ∂(logistic loss)/∂w = (σ(wᵀx) − y)x | 8.4×10⁻¹⁰ |
| **WRONG:** ∂(xᵀAx)/∂x = 2Ax, A non-symmetric | **9.2×10⁻¹** |

```math
\frac{\partial\,\mathrm{softmax}(z)_i}{\partial z_j} = p_i(\delta_{ij} - p_j),\qquad
\frac{\partial}{\partial X}\log\lvert\det X\rvert = X^{-\top}
```

The `p − e_y` result is the famous cancellation behind softmax + cross-entropy (see the DL notes, Part 1.2). `log det` appears in Gaussian likelihoods and normalizing flows.

---

# PART 5 — NUMERICAL DIFFERENTIATION AND GRADIENT CHECKING

```math
\text{forward: } \frac{f(x+h)-f(x)}{h} = f'(x) + \underbrace{\tfrac{h}{2}f''(\xi)}_{\text{truncation}},\quad
\text{error} \approx \frac{h}{2}\lvert f''\rvert + \frac{\varepsilon\lvert f\rvert}{h}\ \Rightarrow\ h^\ast\approx\sqrt{\varepsilon}
```

Measured in `demo_numerical_derivatives()`, [`calculus_from_scratch.py`](calculus_from_scratch.py) — `f(x) = eˣ sin x / (1 + x²)` at x = 0.8, relative error vs the exact derivative:

| h | forward | central | complex-step |
|---|---|---|---|
| 10⁻² | 6.1×10⁻³ | 2.2×10⁻⁶ | 2.2×10⁻⁶ |
| 10⁻⁴ | 6.1×10⁻⁵ | 2.2×10⁻¹⁰ | 2.2×10⁻¹⁰ |
| 10⁻⁶ | 6.1×10⁻⁷ | **9.4×10⁻¹¹** | 2.2×10⁻¹⁴ |
| 10⁻⁸ | **2.3×10⁻⁸** | 2.3×10⁻⁸ | 0 |
| 10⁻¹⁰ | 1.1×10⁻⁶ | 5.7×10⁻⁷ | 1.1×10⁻¹⁶ |
| 10⁻¹⁴ | 8.0×10⁻³ | 8.0×10⁻³ | 2.3×10⁻¹⁶ |

The error is **U-shaped**: truncation error falls as h shrinks, but rounding error grows like ε/h because `f(x+h) − f(x)` subtracts nearly equal numbers. Theory: best forward step ≈ √ε ≈ 1.5×10⁻⁸, best central step ≈ ε^(1/3) ≈ 6×10⁻⁶, matching the minima above. The **complex-step** derivative `Im f(x+ih)/h` involves no subtraction and reaches machine precision for any tiny h — but it needs a real-analytic function written with complex-safe operations.

**Gradient checking a model in practice:** use central differences with h ≈ 10⁻⁵–10⁻⁶ in double precision; compare with the relative error `|a − n| / max(|a| + |n|, tiny)` (≲10⁻⁷ is good); freeze all randomness (dropout, sampling) first; check a few random coordinates rather than all; beware kinks (ReLU at 0) where the numerical gradient straddles a non-differentiable point.

---

# PART 6 — TAYLOR EXPANSION

```math
f(x+\delta) = f(x) + \nabla f(x)^\top\delta + \tfrac12\,\delta^\top H(x)\,\delta + O(\lVert\delta\rVert^3)
```

Measured in `demo_taylor()`, [`calculus_from_scratch.py`](calculus_from_scratch.py) — `exp(x)` around 0.7: the slopes of log(error) against log(h) for the 0th-, 1st-, and 2nd-order approximations are **1.021, 2.014, 3.011** (theory 1, 2, 3). A k-th order expansion has error `O(h^{k+1})` — **for a sufficiently smooth function and small enough h**; the expansion is local.

Where Taylor expansion is used in ML:
```
gradient descent   minimizes the 1st-order model plus a step-size penalty
Newton's method    minimizes the 2nd-order model exactly
trust regions      trust the quadratic model only within a radius
Laplace approx.    Gaussian fitted at a posterior mode using the Hessian
delta method       Var(g(X)) ~ g'(mu)^2 Var(X) — standard errors of transforms
XGBoost            2nd-order expansion of the loss per boosting step
GELU approximation tanh-based polynomial fit (DL notes)
```

---

# PART 7 — CONVEXITY

```math
f(\theta x + (1-\theta)y) \le \theta f(x) + (1-\theta)f(y),\qquad
f(y) \ge f(x) + \nabla f(x)^\top (y-x),\qquad
\nabla^2 f(x) \succeq 0
```

The three characterizations — chord below the graph; tangent plane below the graph; PSD Hessian — are equivalent for twice-differentiable f on a convex domain.

- **Why it matters:** every local minimum of a convex function is global, and first-order conditions (`∇f = 0`) are sufficient. Strictly convex ⇒ at most one minimizer.
- **Strong convexity** (`∇²f ⪰ μI`, μ > 0) and **L-smoothness** (`∇²f ⪯ LI`, i.e. Lipschitz gradient) bound curvature from both sides; their ratio κ = L/μ is the condition number that controls optimization speed (Part 8).
- **Convexity-preserving operations:** non-negative weighted sums, composition with an affine map, pointwise maximum, and certain compositions (e.g. convex non-decreasing ∘ convex).
- **Convex in ML:** linear and logistic regression, ridge, lasso, SVMs, softmax regression, log-sum-exp. **Not convex:** neural networks, k-means, matrix factorization, mixture-model likelihoods — so their optimizers carry no global guarantee.

---

# PART 8 — UNCONSTRAINED OPTIMIZATION

## 8.1 Optimality conditions

```
first order    grad f(x*) = 0                  (necessary for an interior minimum)
second order   H(x*) PSD                        (necessary)
               H(x*) positive definite          (sufficient for a strict local min)
               H(x*) indefinite                 -> saddle point
```

## 8.2 Gradient descent and the condition number

```math
x_{k+1} = x_k - \eta\,\nabla f(x_k),\qquad
\lVert x_k - x^\ast\rVert \le \Big(1-\frac{1}{\kappa}\Big)^k\lVert x_0-x^\ast\rVert\ \ \ (\eta = 1/L,\ \kappa = L/\mu)
```

Measured in `demo_optimization()`, [`calculus_from_scratch.py`](calculus_from_scratch.py) — `f = ½(μx₁² + Lx₂²)`, L = 1, iterations until ‖x‖ < 10⁻⁶:

| κ | GD (lr = 1/L) | GD formula | heavy-ball momentum | momentum rate formula | Newton |
|---|---|---|---|---|---|
| 10 | 132 | 132 | 27 | 22 | 1 |
| 100 | 1,375 | 1,375 | 95 | 69 | 1 |
| 1,000 | 13,809 | 13,809 | 321 | 219 | 1 |
| 10,000 | **138,149** | 138,149 | **1,074** | 691 | 1 |

- Gradient descent needs about **κ·ln(1/tol)** steps — exactly the formula. The step size is limited by the *steepest* direction (L), so progress along the *flattest* direction (μ) crawls. This is why feature scaling, normalization layers, and preconditioning (Adam's per-parameter scaling is a diagonal approximation) matter.
- Tuned momentum scales like **√κ** (×3.3 per 10× κ, vs √10 = 3.16) but needs ~1.5× the simple rate formula: at the optimal tuning its iteration matrix has a repeated eigenvalue, adding a `k·ρᵏ` term. (Nesterov acceleration attains the √κ rate for general smooth strongly convex functions; heavy-ball's guarantee is for quadratics.)
- **Newton** rescales by the inverse Hessian, making every direction equally easy: one step on any quadratic, at `O(d³)` cost per step. Quasi-Newton methods (BFGS, L-BFGS) approximate `H⁻¹` from gradient differences.
- Without strong convexity (μ = 0), gradient descent converges at `O(1/k)` in function value and accelerated methods at `O(1/k²)`.

## 8.3 Step size

For `f = x²/2` (L = 1), measured: lr = 1.90/L and 1.99/L converge (slowly, oscillating), **lr = 2.01/L diverges**. Each step multiplies the error by `(1 − ηL)`, which exceeds 1 in magnitude once `η > 2/L`. For general L-smooth functions, `η ≤ 1/L` guarantees monotone decrease (the descent lemma). Learning-rate warmup and schedules exist because L is unknown and changes during training.

## 8.4 Newton's method

```math
x_{k+1} = x_k - H(x_k)^{-1}\nabla f(x_k),\qquad \lVert x_{k+1}-x^\ast\rVert \le C\,\lVert x_k - x^\ast\rVert^2 \ \ \text{(near } x^\ast\text{)}
```

Measured on `f(x) = eˣ − 2x` (minimum at ln 2), errors per step: 3.1×10⁻¹, 4.3×10⁻², 9.0×10⁻⁴, 4.0×10⁻⁷, 8.0×10⁻¹⁴ — the exponent roughly doubles. The guarantee is **local**: it needs a positive-definite Hessian near the minimum (with a Lipschitz Hessian), and far away Newton can diverge or head for a maximum or saddle — hence damped Newton with line search, or trust regions.

## 8.5 Saddle points

Measured on `f = x²/2 + y⁴/4 − y²/2` (Hessian eigenvalues +1, −1 at the origin), gradient descent with lr = 0.1, steps until `|y| > 0.5`:

| Start | Steps |
|---|---|
| exactly y = 0 | **never escapes** |
| y = 10⁻⁸ | 188 (theory ln(0.5/10⁻⁸)/ln 1.1 ≈ 186) |
| y = 10⁻⁴ | 91 |
| y = 0 plus noise (sd 10⁻⁶ per step) | 134 |

A saddle has zero gradient, so exact gradient descent started on its stable manifold stays there. Any perturbation grows exponentially along the negative-curvature direction — so saddles **slow down** noisy optimizers rather than trap them (with random initialization, gradient descent almost surely avoids strict saddles; Lee et al. 2016). In high dimensions saddles are far more common than bad local minima, so plateaus in training curves are often saddle regions.

---

# PART 9 — CONSTRAINED OPTIMIZATION

## 9.1 Lagrange multipliers

```math
\min_x f(x)\ \text{s.t.}\ h(x) = 0 \quad\Rightarrow\quad \nabla f(x^\ast) = -\nu\,\nabla h(x^\ast),\qquad \mathcal L(x,\nu) = f(x) + \nu\,h(x)
```

At a constrained optimum the gradient of f is **parallel** to the constraint's gradient: you cannot improve f by moving along the constraint surface. The multiplier has a meaning — the **shadow price**, the rate at which the optimal value changes as the constraint is relaxed.

Measured in `demo_constrained()`, [`calculus_from_scratch.py`](calculus_from_scratch.py): maximize `xy` subject to `x + y = c` at c = 10 gives `x = y = 5`, `f* = 25`, **λ = 5**, and the numerical derivative `∂f*/∂c = 5.000000` — the multiplier.

## 9.2 KKT conditions (inequality constraints)

```math
\begin{aligned}
&\min_x f(x)\ \ \text{s.t.}\ \ g_i(x)\le 0,\ \ h_j(x) = 0\\
&\nabla f + \textstyle\sum_i\lambda_i\nabla g_i + \sum_j\nu_j\nabla h_j = 0,\quad g_i\le 0,\quad h_j = 0,\quad \lambda_i\ge 0,\quad \lambda_i\,g_i = 0
\end{aligned}
```

For convex problems satisfying a constraint qualification (e.g. **Slater's condition**: a strictly feasible point exists), the KKT conditions are necessary and sufficient for optimality.

Measured in `demo_constrained()` — minimize `‖x − t‖²` subject to `x₁ + x₂ ≤ 2` by projected gradient descent:

| Target t | Solution x* | λ | Constraint |
|---|---|---|---|
| (2, 1) | (1.5, 0.5) | **1.0** | active (x₁ + x₂ = 2) |
| (0.5, 0.5) | (0.5, 0.5) | **0** | inactive (x₁ + x₂ = 1) |

and in the active case `∂f*/∂(bound) = −1.0 = −λ`. **Complementary slackness** `λᵢgᵢ = 0`: either a constraint binds and its multiplier can be positive, or it is slack and its multiplier is zero. This is exactly why most SVM dual variables are zero and only the **support vectors** — the points on or inside the margin — have non-zero multipliers.

## 9.3 Duality and practical methods

```
Lagrange dual   g(lam, nu) = min_x L(x, lam, nu) — always concave; its max is a
                LOWER BOUND on the primal (weak duality); equal for convex
                problems under Slater (strong duality). The SVM dual is where
                the kernel trick appears.
projected GD    step, then project back onto the feasible set (cheap for boxes,
                balls, simplices, half-spaces)
penalty methods add mu * violation^2 to the objective; increase mu over time
barrier methods add -mu * log(-g(x)) (interior point)
```

---

# PART 10 — INTEGRATION

## 10.1 Quadrature vs Monte Carlo

```math
\int_a^b f(x)\,dx:\quad \text{trapezoid error } O(h^2),\quad \text{Simpson error } O(h^4),\qquad
\text{Monte Carlo: } \frac{b-a}{n}\sum_{i=1}^n f(U_i),\ \ \text{error } O\!\left(\frac{\sigma}{\sqrt n}\right)
```

Measured in `demo_integration()`, [`calculus_from_scratch.py`](calculus_from_scratch.py) — `∫₀^π sin x dx = 2`:

| n | trapezoid error | Simpson error |
|---|---|---|
| 4 | 1.04×10⁻¹ | 4.56×10⁻³ |
| 8 | 2.58×10⁻² | 2.69×10⁻⁴ |
| 16 | 6.43×10⁻³ | 1.66×10⁻⁵ |
| 32 | 1.61×10⁻³ | 1.03×10⁻⁶ |

Doubling n cuts the trapezoid error ~4× and Simpson's ~16× — **for smooth integrands** (the orders need two and four continuous derivatives respectively). Monte Carlo on the same integral (RMS error over 20 repetitions, seed 0): 8.1×10⁻² at n = 100, 7.4×10⁻³ at 10⁴, 2.7×10⁻³ at 10⁵ — the 1/√n rate: 100× more samples for 10× accuracy.

**Monte Carlo's advantage is dimension:** its rate does not depend on d (given finite variance), while a grid with k points per axis needs kᵈ points. That is why expectations in ML — losses, ELBOs, policy gradients — are estimated by sampling.

## 10.2 The curse of dimensionality, measured

Fraction of the cube `[−1,1]ᵈ` inside the unit ball, 20,000 Monte Carlo points each (seed 0):

| d | Monte Carlo | exact |
|---|---|---|
| 2 | 0.78595 | 0.785 |
| 5 | 0.16440 | 0.164 |
| 10 | 0.00240 | 0.00249 |
| 20 | **0** | **2.5×10⁻⁸** |

In 20 dimensions the ball fills 2.5×10⁻⁸ of the cube and 20,000 samples see none of it. Monte Carlo error is dimension-free in **absolute** terms, but estimating a **rare** event to useful **relative** accuracy needs enormous n — the motivation for importance sampling. It also illustrates why almost all the volume of a high-dimensional cube is in its corners, and why nearest-neighbour distances concentrate.

## 10.3 Change of variables

```math
p_Y(y) = p_X(x)\,\big\lvert\det J_{g^{-1}}(y)\big\rvert = \frac{p_X(x)}{\lvert\det J_g(x)\rvert},\qquad y = g(x)
```

Measured in `demo_integration()`: mapping the unit square through `A = [[2, 1], [0.5, 1.5]]` gives a Monte Carlo image area of **2.4946** against `|det A| = 2.5`. A map scales volumes by `|det J|`, so densities scale by `1/|det J|`. This is the foundation of **normalizing flows** (`log p_Y(y) = log p_X(x) − log|det J|`, with architectures designed so the determinant is cheap), of sampling transforms (Box–Muller, inverse CDF), and of the density bookkeeping around the reparameterization trick.

## 10.4 Differentiating an expectation

```math
\nabla_\theta\,\mathbb E_{x\sim p_\theta}[f(x)] = \mathbb E_{x\sim p_\theta}\big[f(x)\,\nabla_\theta\log p_\theta(x)\big]
\qquad\text{vs}\qquad
\nabla_\theta\,\mathbb E_{\varepsilon}\big[f(g_\theta(\varepsilon))\big] = \mathbb E_{\varepsilon}\big[\nabla_\theta f(g_\theta(\varepsilon))\big]
```

Moving a gradient inside an integral (the Leibniz rule) needs regularity conditions (e.g. dominated convergence). The **score-function** estimator (left) works for any f, even non-differentiable — it is REINFORCE (see the [RL notes](RL_Interview_Notes.md)). The **reparameterization** estimator (right) needs a differentiable f and a sampler written as a deterministic function of noise; it usually has much lower variance — it is how VAEs are trained.

---

# PART 11 — NUMERICAL PITFALLS

```
IEEE double   ~16 significant digits, eps ~ 2.2e-16, max ~ 1.8e308,
              smallest normal ~ 2.2e-308
float32       ~7 digits, eps ~ 1.2e-7, max ~ 3.4e38  (exp(89) overflows!)
bfloat16      float32's range, only ~3 digits of precision
```

Measured in `demo_numerics()`, [`calculus_from_scratch.py`](calculus_from_scratch.py):

| Computation | Naive | Stable |
|---|---|---|
| `log Σ exp(z)`, z = (1000, 1000.5, 999) | **inf** (overflow) | 1001.104131 (shift by max) |
| `1 − cos(10⁻⁸)`, true ≈ 5×10⁻¹⁷ | **0** | 5.000×10⁻¹⁷ via `2 sin²(x/2)` |
| variance of N(10⁹, 1), n = 10,000 (seed 1) | **209.74** via `E[x²] − E[x]²` | 0.9846 via Welford |
| 0.1 added 10⁶ times | error 1.3×10⁻⁶ (plain loop) | 0 (Kahan, `math.fsum`) |

- **Catastrophic cancellation:** subtracting nearly equal numbers destroys the significant digits they share. Rewrite the formula (`2 sin²(x/2)`, `expm1`, `log1p`, centring data before computing variances).
- **Log-sum-exp** (`m + log Σ exp(z − m)`) is exact algebra and is how every softmax and cross-entropy implementation avoids overflow; work with **log-probabilities** throughout (products of many probabilities underflow).
- **Summation:** rounding errors accumulate with the number of terms; compensated (Kahan) summation carries them forward. Note that **Python 3.12+'s built-in `sum()` already uses compensated summation for floats** — a naive baseline must be an explicit loop.
- **Mixed precision training** keeps a float32 master copy of the weights and scales the loss so float16 gradients don't underflow — the same issues at a larger scale.

---

# PART 12 — RAPID-FIRE Q&A

<details>
<summary><b>Q: What is a gradient, geometrically?</b></summary>

The vector of partial derivatives; it points in the direction of steepest ascent, its length is the rate of increase in that direction, and it is perpendicular to the level sets of the function.

</details>

<details>
<summary><b>Q: What's the difference between a Jacobian and a Hessian?</b></summary>

The Jacobian collects first derivatives of a vector-valued function (m×n). The Hessian collects second derivatives of a scalar function (n×n, symmetric when the second partials are continuous) — it is the Jacobian of the gradient.

</details>

<details>
<summary><b>Q: Why is reverse-mode AD used for training neural networks?</b></summary>

Training needs the gradient of one scalar loss with respect to millions of parameters. Reverse mode computes all of them in one backward pass at a constant multiple of the forward cost (2.25× in the companion code for n from 10 to 1,000), while forward mode would need one pass per parameter.

</details>

<details>
<summary><b>Q: What is the gradient of xᵀAx?</b></summary>

(A + Aᵀ)x, which equals 2Ax only when A is symmetric. The companion code shows the "2Ax" shortcut is 92% wrong for a non-symmetric A.

</details>

<details>
<summary><b>Q: What is the gradient of cross-entropy with respect to softmax logits?</b></summary>

p − e_y: the predicted probabilities minus the one-hot target. The softmax Jacobian diag(p) − ppᵀ and the derivative of the log cancel.

</details>

<details>
<summary><b>Q: How do you verify a hand-derived gradient?</b></summary>

Compare it with central finite differences (h ≈ 10⁻⁵–10⁻⁶, double precision) using a relative error, on a few random coordinates, with randomness frozen and away from kinks.

</details>

<details>
<summary><b>Q: Why doesn't a smaller finite-difference step always help?</b></summary>

Truncation error shrinks with h, but rounding error grows like ε/h because the numerator subtracts nearly equal numbers. The total is U-shaped, with the best forward step near √ε ≈ 10⁻⁸.

</details>

<details>
<summary><b>Q: How do you tell whether a critical point is a minimum, maximum, or saddle?</b></summary>

Look at the Hessian's eigenvalues: all positive → local minimum; all negative → local maximum; mixed signs → saddle; zero eigenvalues → the second-order test is inconclusive.

</details>

<details>
<summary><b>Q: Why is gradient descent slow on ill-conditioned problems?</b></summary>

The step size must stay below 2/L to be stable in the steepest direction, so progress along the flattest direction (curvature μ) is tiny: about κ = L/μ times ln(1/tol) iterations. The companion code measures exactly 138,149 steps at κ = 10⁴.

</details>

<details>
<summary><b>Q: What does momentum buy you?</b></summary>

On ill-conditioned convex problems, tuned momentum reduces the dependence on κ to √κ (1,074 vs 138,149 steps at κ = 10⁴ in the companion code). It accumulates velocity along consistent directions and damps oscillations across steep ones.

</details>

<details>
<summary><b>Q: Why don't we use Newton's method to train neural networks?</b></summary>

The Hessian has n² entries and inverting it costs O(n³) — impossible for millions of parameters. It also isn't positive definite in non-convex regions, so pure Newton steps can move toward saddles or maxima. Approximations (L-BFGS, K-FAC, Hessian-vector products, Adam's diagonal scaling) capture some of the benefit.

</details>

<details>
<summary><b>Q: What does convexity guarantee?</b></summary>

Every local minimum is global, and the first-order condition ∇f = 0 is sufficient. With strong convexity and smoothness, gradient descent converges linearly at a rate set by the condition number.

</details>

<details>
<summary><b>Q: What is a Lagrange multiplier, intuitively?</b></summary>

The price of the constraint: how fast the optimal objective changes as the constraint is relaxed. In the companion code ∂f*/∂c equals λ = 5 exactly.

</details>

<details>
<summary><b>Q: Explain complementary slackness and connect it to SVMs.</b></summary>

For each inequality constraint, λᵢgᵢ(x*) = 0: either the constraint is active or its multiplier is zero. In the SVM dual, points strictly outside the margin have zero multipliers, so only the support vectors determine the solution.

</details>

<details>
<summary><b>Q: When would you use Monte Carlo integration instead of quadrature?</b></summary>

In more than a few dimensions: quadrature needs kᵈ points while Monte Carlo's error is O(σ/√n) regardless of d. In one dimension, quadrature (Simpson's O(h⁴)) is far more efficient.

</details>

<details>
<summary><b>Q: Where does the Jacobian determinant appear in ML?</b></summary>

In change-of-variables for densities — normalizing flows compute log p(y) = log p(x) − log|det J| — and in sampling transforms such as Box–Muller.

</details>

<details>
<summary><b>Q: Score-function vs reparameterization gradients?</b></summary>

Both give unbiased estimates of the gradient of an expectation. The score-function (REINFORCE) estimator works for non-differentiable objectives but has high variance; the reparameterization estimator needs a differentiable path from noise to sample and usually has much lower variance.

</details>

<details>
<summary><b>Q: Why does softmax subtract the maximum logit?</b></summary>

To prevent exp overflow: softmax(z) = softmax(z − m) exactly, and after the shift the largest exponent is exp(0) = 1. Log-sum-exp uses the same identity.

</details>

---

# PART 13 — COVERAGE INDEX

| Concept | Implementation in [`calculus_from_scratch.py`](calculus_from_scratch.py) | Demo |
|---|---|---|
| Forward, central differences; optimal step | `forward_diff`, `central_diff`, `optimal_steps` | 1 |
| Complex-step derivative | `complex_step` | 1 |
| Forward-mode AD (dual numbers) | `Dual`, `gradient_forward_mode` | 2 |
| Reverse-mode AD (tape) | `Var`, `gradient_reverse_mode` | 2 |
| Cost comparison by operation count | `function_cost`, `OPS` | 2 |
| Matrix-calculus identities vs finite differences | `check_matrix_identities` | 3 |
| Softmax Jacobian, cross-entropy gradient | `check_matrix_identities` | 3 |
| Taylor approximation orders | `taylor_orders` | 4 |
| Gradient descent vs condition number | `gd_quadratic` | 5 |
| Heavy-ball momentum | `heavy_ball_params`, `gd_quadratic(momentum=...)` | 5 |
| Step-size stability | `gd_quadratic` | 5 |
| Newton's method, quadratic convergence | `newton_1d` | 5 |
| Saddle-point escape | `saddle_escape` | 5 |
| Lagrange multiplier as shadow price | `lagrange_product` | 6 |
| KKT, complementary slackness, projected gradient | `projected_gradient` | 6 |
| Trapezoid and Simpson rules | `trapezoid`, `simpson` | 7 |
| Monte Carlo integration and 1/√n | `mc_integral` | 7 |
| High-dimensional volumes | `unit_ball_volume`, `mc_ball_fraction` | 7 |
| Change of variables / Jacobian determinant | `change_of_variables_check` | 7 |
| Log-sum-exp | `logsumexp`, `naive_logsumexp` | 8 |
| Stable variance, compensated summation | `welford_variance`, `naive_variance`, `kahan_sum`, `naive_sum` | 8 |

**Covered in these notes, not implemented** — be ready to explain: subgradients in general, Hessian-vector products, quasi-Newton (BFGS/L-BFGS), Nesterov acceleration, duality in depth, interior-point methods, importance sampling, the Leibniz integral rule's conditions, mixed-precision training.

---

# PART 14 — 7-DAY PLAN

```
Day 1  Derivatives as linear approximation; subgradients; activation derivatives.
Day 2  Gradients, Jacobians, Hessians; chain rule; forward vs reverse AD. Demo 2.
Day 3  Matrix calculus by differentials; derive the identities table yourself,
       then check with demo 3. Finite differences and gradient checking: demo 1.
Day 4  Taylor expansion; convexity (three definitions), strong convexity and
       smoothness. Demo 4.
Day 5  Unconstrained optimization: optimality conditions, GD rate vs kappa,
       step-size limit, momentum, Newton, saddles. Demo 5.
Day 6  Lagrange multipliers, KKT, complementary slackness, duality, SVM link.
       Demo 6.
Day 7  Integration: quadrature vs Monte Carlo, dimension, change of variables,
       gradients of expectations; numerical pitfalls. Demos 7-8. Then the
       rapid-fire questions closed-book.
```

**Self-test — from memory, can you:**
1. Explain why the gradient points in the direction of steepest ascent?
2. Write the chain rule as a product of Jacobians and explain vanishing/exploding gradients with it?
3. Explain why reverse mode is cheaper than forward mode for a scalar loss?
4. Derive ∇(xᵀAx), ∇‖Ax − b‖², and the softmax cross-entropy gradient?
5. Explain the U-shaped finite-difference error and the optimal step?
6. Give three equivalent definitions of convexity?
7. State GD's convergence rate in terms of κ, and the step-size limit?
8. Explain quadratic convergence and why Newton is only locally guaranteed?
9. Classify a critical point from its Hessian, and explain why saddles slow training?
10. Solve a small Lagrange problem and interpret the multiplier?
11. State the KKT conditions and connect complementary slackness to support vectors?
12. Explain when Monte Carlo beats quadrature, and how change of variables works?

---

# REFERENCES

Primary sources for the claims above. Where these notes simplify, the source is authoritative.

- Deisenroth, Faisal & Ong. *Mathematics for Machine Learning*. Cambridge University Press, 2020 (free online) — vector calculus and optimization chapters.
- Boyd & Vandenberghe. *Convex Optimization*. Cambridge University Press, 2004 (free online) — convexity, duality, KKT.
- Nocedal & Wright. *Numerical Optimization*, 2nd ed. Springer, 2006 — GD rates, Newton, quasi-Newton, line search, trust regions.
- Nesterov. *Lectures on Convex Optimization*, 2nd ed. Springer, 2018 — accelerated methods and lower bounds.
- Petersen & Pedersen. *The Matrix Cookbook*. 2012 (free online) — matrix-calculus identities.
- Magnus & Neudecker. *Matrix Differential Calculus with Applications in Statistics and Econometrics*, 3rd ed. Wiley, 2019 — the differential method.
- Griewank & Walther. *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, 2nd ed. SIAM, 2008.
- Baydin et al. Automatic differentiation in machine learning: a survey. *JMLR*, 2018.
- Baur & Strassen. The complexity of partial derivatives. *Theoretical Computer Science*, 1983.
- Squire & Trapp. Using complex variables to estimate derivatives of real functions. *SIAM Review*, 1998.
- Polyak. Some methods of speeding up the convergence of iteration methods. *USSR Comp. Math. and Math. Physics*, 1964.
- Lee et al. Gradient descent only converges to minimizers. *COLT*, 2016.
- Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM, 2002 — cancellation, summation.
- Welford. Note on a method for calculating corrected sums of squares and products. *Technometrics*, 1962.
- Kahan. Further remarks on reducing truncation errors. *Communications of the ACM*, 1965.
- Rezende & Mohamed. Variational inference with normalizing flows. *ICML*, 2015.

---

[ML](ML_Interview_Notes.md) · [DL](DL_Interview_Notes.md) · [RL](RL_Interview_Notes.md) · [NLP & LLMs](NLP_LLM_Interview_Notes.md) · [ML System Design](ML_System_Design_Notes.md) · [Probability & Stats](Probability_Statistics_Notes.md) · [Linear Algebra](Linear_Algebra_Notes.md) · **Calculus**
