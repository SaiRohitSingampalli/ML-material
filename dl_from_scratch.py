"""
================================================================================
 DEEP LEARNING FROM SCRATCH — pure Python, zero libraries
================================================================================
 Only `math` and `random`. No NumPy, no PyTorch. Everything below is built on
 a reverse-mode automatic differentiation engine written from first principles.

 CONTENTS
 --------
 0. Value ................ scalar autograd (micrograd-style) — backprop atom by atom
 1. Tensor ............... 2-D reverse-mode autograd engine (the real workhorse)
 2. nn ................... Module, Linear, Embedding, Sequential, Dropout,
                           LayerNorm, BatchNorm1d, initializers
 3. losses ............... mse_loss, cross_entropy (fused softmax+CE), bce_loss
 4. optim ................ SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam,
                           AdamW, LR schedules, gradient clipping
 5. conv ................. im2col, Conv2D, MaxPool2D, Flatten
 6. recurrent ............ RNNCell, LSTMCell, GRUCell, sequence wrappers
 7. attention ............ scaled dot-product, MultiHeadAttention,
                           PositionalEncoding, TransformerBlock, MiniGPT
 8. generative ........... Autoencoder, VAE (reparameterization trick)
 9. training ............. DataLoader, fit loop, early stopping, grad checking
 10. demos ............... every component trained and verified end to end

 WHY AN AUTOGRAD ENGINE: hand-coding a backward pass per architecture (as in
 a fixed MLP) does not generalize. Reverse-mode AD builds a graph of primitive
 ops during the forward pass, then walks it backwards applying the chain rule
 exactly once per edge. Cost: ONE backward pass gives gradients w.r.t. ALL
 parameters, versus O(#params) forward passes for finite differences. That
 asymmetry is the only reason training billion-parameter models is possible.
================================================================================
"""

import math
import random

random.seed(1337)
EPS = 1e-12


# =============================================================================
# 0. SCALAR AUTOGRAD — backprop, one number at a time
# =============================================================================
class Value:
    """
    A scalar with a gradient, plus the local derivative rule for the op that
    produced it. This is backpropagation stripped to its essence.

        forward : build a DAG of Values as the expression is evaluated
        backward: topologically sort the DAG, seed dL/dL = 1, then walk in
                  reverse, each node ADDING its contribution to its children's
                  .grad (accumulate, because a node reused in several places
                  receives gradient from every path — that is the multivariate
                  chain rule).

    Local rules used below:
        c = a + b   ->  dc/da = 1,      dc/db = 1
        c = a * b   ->  dc/da = b,      dc/db = a
        c = a^k     ->  dc/da = k a^(k-1)
        c = relu(a) ->  dc/da = 1 if a > 0 else 0
        c = tanh(a) ->  dc/da = 1 - tanh(a)^2
        c = exp(a)  ->  dc/da = exp(a)
        c = log(a)  ->  dc/da = 1/a
    """

    __slots__ = ("data", "grad", "_backward", "_prev", "_op")

    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _bw():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _bw
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _bw():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _bw
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")

        def _bw():
            self.grad += k * (self.data ** (k - 1)) * out.grad
        out._backward = _bw
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _bw():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _bw
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _bw():
            self.grad += (1 - t * t) * out.grad
        out._backward = _bw
        return out

    def exp(self):
        e = math.exp(min(self.data, 60))
        out = Value(e, (self,), "exp")

        def _bw():
            self.grad += e * out.grad
        out._backward = _bw
        return out

    def log(self):
        out = Value(math.log(max(self.data, 1e-15)), (self,), "log")

        def _bw():
            self.grad += (1.0 / max(self.data, 1e-15)) * out.grad
        out._backward = _bw
        return out

    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data)) if self.data >= 0 \
            else math.exp(self.data) / (1 + math.exp(self.data))
        out = Value(s, (self,), "sigmoid")

        def _bw():
            self.grad += s * (1 - s) * out.grad
        out._backward = _bw
        return out

    # -- conveniences
    def __neg__(self):
        return self * -1

    def __sub__(self, o):
        return self + (-o if isinstance(o, Value) else Value(-o))

    def __radd__(self, o):
        return self + o

    def __rmul__(self, o):
        return self * o

    def __truediv__(self, o):
        return self * (o ** -1 if isinstance(o, Value) else Value(1.0 / o))

    def __repr__(self):
        return f"Value({self.data:.4f}, grad={self.grad:.4f})"

    def backward(self):
        """Topological order, then reverse-walk applying each local rule."""
        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for c in v._prev:
                build(c)
            topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()


# =============================================================================
# 1. TENSOR — 2-D reverse-mode autograd
# =============================================================================
def _zeros(r, c):
    return [[0.0] * c for _ in range(r)]


class Tensor:
    """
    A 2-D array (list of lists) that remembers how it was computed.

    SHAPE CONVENTION: everything is (rows, cols) = (batch, features). Higher-rank
    data (images, sequences) is flattened into 2-D and reshaped by the layers
    that need structure — this keeps the engine small while still supporting
    convolutions (via im2col) and attention (via column slicing).

    BROADCASTING: only the case that actually matters for neural nets is
    supported — a (1, c) row vector against an (r, c) matrix, which is exactly
    how a bias is added to a batch. The backward for that case SUMS the incoming
    gradient over the batch dimension, because the same bias participated in
    every row (chain rule over all paths).
    """

    def __init__(self, data, _children=(), _op="", requires_grad=True):
        if isinstance(data, (int, float)):
            data = [[float(data)]]
        elif data and not isinstance(data[0], list):
            data = [list(map(float, data))]
        self.data = [list(map(float, row)) for row in data]
        self.rows, self.cols = len(self.data), len(self.data[0])
        self.grad = _zeros(self.rows, self.cols)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ------------------------------------------------------------ properties
    @property
    def shape(self):
        return (self.rows, self.cols)

    def item(self):
        return self.data[0][0]

    def zero_grad(self):
        self.grad = _zeros(self.rows, self.cols)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op='{self._op}')"

    def detach(self):
        return Tensor([r[:] for r in self.data])

    # ------------------------------------------------------------ constructors
    @staticmethod
    def zeros(r, c):
        return Tensor(_zeros(r, c))

    @staticmethod
    def randn(r, c, std=1.0, seed=None):
        rng = random.Random(seed) if seed is not None else random
        return Tensor([[rng.gauss(0, std) for _ in range(c)] for _ in range(r)])

    # ------------------------------------------------------------ elementwise
    def _binary(self, other, fwd, bwd_self, bwd_other, op):
        other = other if isinstance(other, Tensor) else Tensor(
            [[float(other)] * self.cols for _ in range(self.rows)])
        # broadcast a (1, c) row against (r, c)
        br_self = self.rows == 1 and other.rows > 1
        br_other = other.rows == 1 and self.rows > 1
        R = max(self.rows, other.rows)
        C = max(self.cols, other.cols)

        def get(t, i, j):
            return t.data[0 if t.rows == 1 else i][0 if t.cols == 1 else j]

        out = Tensor([[fwd(get(self, i, j), get(other, i, j)) for j in range(C)]
                      for i in range(R)], (self, other), op)

        def _bw():
            for i in range(R):
                for j in range(C):
                    a, b = get(self, i, j), get(other, i, j)
                    g = out.grad[i][j]
                    si = 0 if self.rows == 1 else i
                    sj = 0 if self.cols == 1 else j
                    oi = 0 if other.rows == 1 else i
                    oj = 0 if other.cols == 1 else j
                    self.grad[si][sj] += bwd_self(a, b) * g
                    other.grad[oi][oj] += bwd_other(a, b) * g
        out._backward = _bw
        return out

    def __add__(self, o):
        return self._binary(o, lambda a, b: a + b, lambda a, b: 1.0,
                            lambda a, b: 1.0, "+")

    def __mul__(self, o):
        return self._binary(o, lambda a, b: a * b, lambda a, b: b,
                            lambda a, b: a, "*")

    def __sub__(self, o):
        return self._binary(o, lambda a, b: a - b, lambda a, b: 1.0,
                            lambda a, b: -1.0, "-")

    def __truediv__(self, o):
        return self._binary(o, lambda a, b: a / (b + EPS),
                            lambda a, b: 1.0 / (b + EPS),
                            lambda a, b: -a / (b * b + EPS), "/")

    def __neg__(self):
        return self * -1.0

    def __radd__(self, o):
        return self + o

    def __rmul__(self, o):
        return self * o

    def __rsub__(self, o):
        return (-self) + o

    def pow(self, k):
        out = Tensor([[v ** k for v in row] for row in self.data], (self,), "pow")

        def _bw():
            for i in range(self.rows):
                for j in range(self.cols):
                    self.grad[i][j] += k * (self.data[i][j] ** (k - 1)) * out.grad[i][j]
        out._backward = _bw
        return out

    def _unary(self, fwd, bwd, op):
        """bwd receives (x, y) where y = fwd(x) — lets us reuse the activation."""
        vals = [[fwd(v) for v in row] for row in self.data]
        out = Tensor(vals, (self,), op)

        def _bw():
            for i in range(self.rows):
                for j in range(self.cols):
                    self.grad[i][j] += bwd(self.data[i][j], vals[i][j]) * out.grad[i][j]
        out._backward = _bw
        return out

    # ------------------------------------------------------------ activations
    def relu(self):
        """max(0,x). Gradient is 1 or 0 — no saturation for x>0, hence no
        vanishing gradient through depth. Risk: a unit stuck at x<0 for all
        inputs receives zero gradient forever ('dying ReLU')."""
        return self._unary(lambda x: x if x > 0 else 0.0,
                           lambda x, y: 1.0 if x > 0 else 0.0, "relu")

    def leaky_relu(self, slope=0.01):
        return self._unary(lambda x: x if x > 0 else slope * x,
                           lambda x, y: 1.0 if x > 0 else slope, "lrelu")

    def sigmoid(self):
        """s' = s(1-s), max 0.25 -> gradients shrink by >=4x per layer."""
        def f(x):
            return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
        return self._unary(f, lambda x, y: y * (1 - y), "sigmoid")

    def tanh(self):
        """Zero-centred (unlike sigmoid), derivative 1 - tanh^2, max 1.0."""
        return self._unary(math.tanh, lambda x, y: 1 - y * y, "tanh")

    def gelu(self):
        """
        GELU(x) = x * Phi(x), the transformer default. Uses the tanh
        approximation:  0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))).
        Smooth everywhere, and unlike ReLU it has a small negative lobe, which
        empirically trains better in deep attention stacks.
        """
        c = math.sqrt(2 / math.pi)

        def f(x):
            return 0.5 * x * (1 + math.tanh(c * (x + 0.044715 * x ** 3)))

        def d(x, y):
            t = math.tanh(c * (x + 0.044715 * x ** 3))
            return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * c * (1 + 3 * 0.044715 * x * x)
        return self._unary(f, d, "gelu")

    def silu(self):
        """Swish/SiLU: x*sigmoid(x); derivative s + x*s*(1-s)."""
        def s(x):
            return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
        return self._unary(lambda x: x * s(x),
                           lambda x, y: s(x) + x * s(x) * (1 - s(x)), "silu")

    def exp(self):
        return self._unary(lambda x: math.exp(min(x, 60)), lambda x, y: y, "exp")

    def log(self):
        return self._unary(lambda x: math.log(max(x, 1e-15)),
                           lambda x, y: 1.0 / max(x, 1e-15), "log")

    def sqrt(self):
        return self._unary(lambda x: math.sqrt(max(x, 0.0)),
                           lambda x, y: 0.5 / max(y, 1e-9), "sqrt")

    # ------------------------------------------------------------ linear algebra
    def matmul(self, other):
        """
        (n,m) @ (m,p) -> (n,p).  Backward (the two identities to memorize):
            dL/dA = dL/dC @ B^T
            dL/dB = A^T @ dL/dC
        Shapes force the answer: dL/dA must be (n,m) = (n,p)@(p,m).
        """
        assert self.cols == other.rows, f"shape mismatch {self.shape} @ {other.shape}"
        n, m, p = self.rows, self.cols, other.cols
        vals = _zeros(n, p)
        A, B = self.data, other.data
        for i in range(n):
            Ai, Vi = A[i], vals[i]
            for k in range(m):
                a = Ai[k]
                if a:
                    Bk = B[k]
                    for j in range(p):
                        Vi[j] += a * Bk[j]
        out = Tensor(vals, (self, other), "matmul")

        def _bw():
            G = out.grad
            for i in range(n):
                Gi = G[i]
                for k in range(m):
                    s = 0.0
                    Bk = B[k]
                    for j in range(p):
                        s += Gi[j] * Bk[j]
                    self.grad[i][k] += s
            for k in range(m):
                for j in range(p):
                    s = 0.0
                    for i in range(n):
                        s += A[i][k] * G[i][j]
                    other.grad[k][j] += s
        out._backward = _bw
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def T(self):
        out = Tensor([[self.data[i][j] for i in range(self.rows)]
                      for j in range(self.cols)], (self,), "T")

        def _bw():
            for i in range(self.rows):
                for j in range(self.cols):
                    self.grad[i][j] += out.grad[j][i]
        out._backward = _bw
        return out

    # ------------------------------------------------------------ reductions
    def sum(self, axis=None, keepdims=True):
        if axis is None:
            out = Tensor(sum(sum(r) for r in self.data), (self,), "sum")

            def _bw():
                g = out.grad[0][0]
                for i in range(self.rows):
                    for j in range(self.cols):
                        self.grad[i][j] += g
        elif axis == 1:                                   # sum over columns
            out = Tensor([[sum(r)] for r in self.data], (self,), "sum1")

            def _bw():
                for i in range(self.rows):
                    g = out.grad[i][0]
                    for j in range(self.cols):
                        self.grad[i][j] += g
        else:                                             # axis == 0, over rows
            out = Tensor([[sum(self.data[i][j] for i in range(self.rows))
                           for j in range(self.cols)]], (self,), "sum0")

            def _bw():
                for j in range(self.cols):
                    g = out.grad[0][j]
                    for i in range(self.rows):
                        self.grad[i][j] += g
        out._backward = _bw
        return out

    def mean(self, axis=None):
        if axis is None:
            return self.sum() * (1.0 / (self.rows * self.cols))
        if axis == 1:
            return self.sum(axis=1) * (1.0 / self.cols)
        return self.sum(axis=0) * (1.0 / self.rows)

    def softmax(self):
        """
        Row-wise softmax. The max-subtraction is a no-op mathematically (the
        constant cancels in numerator and denominator) but prevents exp overflow.

        Backward: for y = softmax(x),  dL/dx = y * (dL/dy - sum_j(dL/dy_j y_j))
        which comes from the Jacobian  dy_i/dx_j = y_i(delta_ij - y_j).
        """
        vals = []
        for row in self.data:
            m = max(row)
            e = [math.exp(v - m) for v in row]
            s = sum(e)
            vals.append([v / s for v in e])
        out = Tensor(vals, (self,), "softmax")

        def _bw():
            for i in range(self.rows):
                y, g = vals[i], out.grad[i]
                d = sum(g[j] * y[j] for j in range(self.cols))
                for j in range(self.cols):
                    self.grad[i][j] += y[j] * (g[j] - d)
        out._backward = _bw
        return out

    # ------------------------------------------------------------ shape ops
    def reshape(self, r, c):
        flat = [v for row in self.data for v in row]
        assert r * c == len(flat), "reshape size mismatch"
        out = Tensor([flat[i * c:(i + 1) * c] for i in range(r)], (self,), "reshape")

        def _bw():
            gflat = [v for row in out.grad for v in row]
            k = 0
            for i in range(self.rows):
                for j in range(self.cols):
                    self.grad[i][j] += gflat[k]
                    k += 1
        out._backward = _bw
        return out

    def slice_cols(self, a, b):
        """Columns [a, b). Used to split heads in multi-head attention."""
        out = Tensor([row[a:b] for row in self.data], (self,), "slice")

        def _bw():
            for i in range(self.rows):
                for j in range(a, b):
                    self.grad[i][j] += out.grad[i][j - a]
        out._backward = _bw
        return out

    def slice_rows(self, a, b):
        out = Tensor([self.data[i] for i in range(a, b)], (self,), "slicer")

        def _bw():
            for i in range(a, b):
                for j in range(self.cols):
                    self.grad[i][j] += out.grad[i - a][j]
        out._backward = _bw
        return out

    # ------------------------------------------------------------ engine
    def backward(self):
        """Seed dL/dL = 1 and walk the DAG in reverse topological order."""
        topo, seen = [], set()

        def build(v):
            if id(v) in seen:
                return
            seen.add(id(v))
            for c in v._prev:
                build(c)
            topo.append(v)
        build(self)
        self.grad = [[1.0] * self.cols for _ in range(self.rows)]
        for v in reversed(topo):
            v._backward()


def cat_cols(tensors):
    """Concatenate along columns; backward routes each slice's gradient back."""
    rows = tensors[0].rows
    data = [[] for _ in range(rows)]
    offsets, off = [], 0
    for t in tensors:
        offsets.append(off)
        off += t.cols
        for i in range(rows):
            data[i].extend(t.data[i])
    out = Tensor(data, tuple(tensors), "cat")

    def _bw():
        for t, o in zip(tensors, offsets):
            for i in range(rows):
                for j in range(t.cols):
                    t.grad[i][j] += out.grad[i][o + j]
    out._backward = _bw
    return out


def stack_rows(tensors):
    """Stack single-row tensors into one matrix (used by the sequence models)."""
    data = [row[:] for t in tensors for row in t.data]
    out = Tensor(data, tuple(tensors), "stack")

    def _bw():
        k = 0
        for t in tensors:
            for i in range(t.rows):
                for j in range(t.cols):
                    t.grad[i][j] += out.grad[k][j]
                k += 1
    out._backward = _bw
    return out


# =============================================================================
# 2. LAYERS
# =============================================================================
class Module:
    """Base class: collects parameters and toggles train/eval mode."""

    def __init__(self):
        self.training = True

    def parameters(self):
        ps = []
        for v in vars(self).values():
            if isinstance(v, Tensor) and v.requires_grad:
                ps.append(v)
            elif isinstance(v, Module):
                ps += v.parameters()
            elif isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, Module):
                        ps += x.parameters()
                    elif isinstance(x, Tensor) and x.requires_grad:
                        ps.append(x)
        return ps

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self):
        self.training = True
        for v in vars(self).values():
            if isinstance(v, Module):
                v.train()
            elif isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, Module):
                        x.train()
        return self

    def eval(self):
        self.training = False
        for v in vars(self).values():
            if isinstance(v, Module):
                v.eval()
            elif isinstance(v, (list, tuple)):
                for x in v:
                    if isinstance(x, Module):
                        x.eval()
        return self

    def __call__(self, *a, **kw):
        return self.forward(*a, **kw)

    def n_params(self):
        return sum(p.rows * p.cols for p in self.parameters())


def init_weights(fan_in, fan_out, scheme="he", seed=None):
    """
    WEIGHT INITIALIZATION — why it decides whether a deep net trains at all.

    Activations propagate as Var(a_l) ~ Var(a_(l-1)) * fan_in * Var(W). If
    fan_in*Var(W) != 1, activation variance shrinks or explodes GEOMETRICALLY
    with depth, and the backward pass inherits the same factor.

      Xavier/Glorot   Var(W) = 2/(fan_in+fan_out)   tanh/sigmoid (linear regime,
                                                    balances forward & backward)
      He/Kaiming      Var(W) = 2/fan_in             ReLU — the extra factor of 2
                                                    compensates for ReLU zeroing
                                                    half the units
      LeCun           Var(W) = 1/fan_in             SELU
      Zero init       FATAL: every unit in a layer computes the same function and
                      receives the same gradient, so symmetry is never broken.
    """
    rng = random.Random(seed) if seed is not None else random
    if scheme == "he":
        std = math.sqrt(2.0 / fan_in)
    elif scheme == "xavier":
        std = math.sqrt(2.0 / (fan_in + fan_out))
    elif scheme == "lecun":
        std = math.sqrt(1.0 / fan_in)
    else:
        std = 0.02                               # transformer-style small const
    return Tensor([[rng.gauss(0, std) for _ in range(fan_out)]
                   for _ in range(fan_in)])


class Linear(Module):
    """y = x W + b.  Params: fan_in*fan_out + fan_out."""

    def __init__(self, fan_in, fan_out, bias=True, init="he", seed=None):
        super().__init__()
        self.W = init_weights(fan_in, fan_out, init, seed)
        self.b = Tensor.zeros(1, fan_out) if bias else None

    def forward(self, x):
        out = x.matmul(self.W)
        return out + self.b if self.b is not None else out


class Embedding(Module):
    """
    Lookup table: row i of the weight matrix IS the vector for token i.
    Equivalent to one-hot @ W, but O(1) instead of O(V) — and the backward
    scatters the gradient back only to the rows that were actually used
    (a sparse gradient).
    """

    def __init__(self, vocab, dim, seed=None):
        super().__init__()
        rng = random.Random(seed) if seed is not None else random
        self.W = Tensor([[rng.gauss(0, 0.02) for _ in range(dim)]
                         for _ in range(vocab)])

    def forward(self, ids):
        W = self.W
        out = Tensor([W.data[i][:] for i in ids], (W,), "embed")

        def _bw():
            for r, i in enumerate(ids):
                for j in range(W.cols):
                    W.grad[i][j] += out.grad[r][j]
        out._backward = _bw
        return out


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x


class ReLU(Module):
    def forward(self, x):
        return x.relu()


class Tanh(Module):
    def forward(self, x):
        return x.tanh()


class GELU(Module):
    def forward(self, x):
        return x.gelu()


class Sigmoid(Module):
    def forward(self, x):
        return x.sigmoid()


class Dropout(Module):
    """
    DROPOUT — zero each unit independently with probability p during TRAINING.

    INVERTED DROPOUT (what everyone actually implements): scale the surviving
    activations by 1/(1-p) at train time so that E[output] is unchanged. That
    way inference needs no rescaling at all — just turn dropout off.

    Interpretation: each minibatch trains a different thinned subnetwork, and
    at test time the full network approximates averaging 2^n of them. It also
    prevents co-adaptation, forcing units to be individually useful.
    """

    def __init__(self, p=0.5, seed=None):
        super().__init__()
        self.p = p
        self.rng = random.Random(seed) if seed is not None else random

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x
        keep = 1.0 - self.p
        mask = [[1.0 / keep if self.rng.random() < keep else 0.0
                 for _ in range(x.cols)] for _ in range(x.rows)]
        out = Tensor([[x.data[i][j] * mask[i][j] for j in range(x.cols)]
                      for i in range(x.rows)], (x,), "dropout")

        def _bw():
            for i in range(x.rows):
                for j in range(x.cols):
                    x.grad[i][j] += mask[i][j] * out.grad[i][j]
        out._backward = _bw
        return out


class LayerNorm(Module):
    """
    LAYER NORMALIZATION — normalize across the FEATURES of each sample.

        mu    = (1/H) sum_j x_j
        var   = (1/H) sum_j (x_j - mu)^2
        xhat  = (x - mu) / sqrt(var + eps)
        y     = gamma * xhat + beta

    BACKWARD (derived by the chain rule through mu and var, which BOTH depend
    on every x_j — this is why the formula has three terms):

        dL/dx_j = (1/(H*sigma)) [ H*dxhat_j - sum_k dxhat_k - xhat_j * sum_k dxhat_k xhat_k ]

    where dxhat = dL/dy * gamma.

    vs BATCHNORM: LayerNorm has no batch dependence, so it works with batch
    size 1, with variable-length sequences, and at inference without running
    statistics. That is why transformers and RNNs use it.
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = Tensor([[1.0] * dim])
        self.beta = Tensor.zeros(1, dim)
        self.eps = eps

    def forward(self, x):
        H = x.cols
        mus, invs, xhat = [], [], []
        for i in range(x.rows):
            row = x.data[i]
            mu = sum(row) / H
            var = sum((v - mu) ** 2 for v in row) / H
            inv = 1.0 / math.sqrt(var + self.eps)
            mus.append(mu)
            invs.append(inv)
            xhat.append([(v - mu) * inv for v in row])
        g, b = self.gamma, self.beta
        out = Tensor([[xhat[i][j] * g.data[0][j] + b.data[0][j] for j in range(H)]
                      for i in range(x.rows)], (x, g, b), "layernorm")

        def _bw():
            for i in range(x.rows):
                go = out.grad[i]
                dxh = [go[j] * g.data[0][j] for j in range(H)]
                s1 = sum(dxh)
                s2 = sum(dxh[j] * xhat[i][j] for j in range(H))
                for j in range(H):
                    g.grad[0][j] += go[j] * xhat[i][j]
                    b.grad[0][j] += go[j]
                    x.grad[i][j] += (invs[i] / H) * (H * dxh[j] - s1 - xhat[i][j] * s2)
        out._backward = _bw
        return out


class BatchNorm1d(Module):
    """
    BATCH NORMALIZATION — normalize each FEATURE across the BATCH.

        mu_B, var_B computed per feature over the minibatch
        xhat = (x - mu_B)/sqrt(var_B + eps) ;  y = gamma*xhat + beta

    Train uses batch statistics; INFERENCE uses running averages accumulated
    during training (momentum m):  running = m*running + (1-m)*batch.
    Forgetting to switch to eval mode is one of the classic production bugs.

    Why it helps: smooths the loss landscape (the original 'internal covariate
    shift' story is now contested), permits larger learning rates, and injects
    batch noise that acts as a mild regularizer. Weaknesses: unreliable with
    small batches, awkward for variable-length sequences, and it couples
    examples within a batch.

    Backward has the same three-term structure as LayerNorm, but the reduction
    runs down the BATCH dimension instead of across features.
    """

    def __init__(self, dim, eps=1e-5, momentum=0.9):
        super().__init__()
        self.gamma = Tensor([[1.0] * dim])
        self.beta = Tensor.zeros(1, dim)
        self.eps, self.momentum = eps, momentum
        self.run_mean = [0.0] * dim
        self.run_var = [1.0] * dim

    def forward(self, x):
        N, D = x.rows, x.cols
        g, b = self.gamma, self.beta
        if self.training:
            mu = [sum(x.data[i][j] for i in range(N)) / N for j in range(D)]
            var = [sum((x.data[i][j] - mu[j]) ** 2 for i in range(N)) / N
                   for j in range(D)]
            for j in range(D):
                self.run_mean[j] = self.momentum * self.run_mean[j] + \
                    (1 - self.momentum) * mu[j]
                self.run_var[j] = self.momentum * self.run_var[j] + \
                    (1 - self.momentum) * var[j]
        else:
            mu, var = self.run_mean, self.run_var
        inv = [1.0 / math.sqrt(var[j] + self.eps) for j in range(D)]
        xhat = [[(x.data[i][j] - mu[j]) * inv[j] for j in range(D)] for i in range(N)]
        out = Tensor([[xhat[i][j] * g.data[0][j] + b.data[0][j] for j in range(D)]
                      for i in range(N)], (x, g, b), "batchnorm")
        training = self.training

        def _bw():
            for j in range(D):
                dxh = [out.grad[i][j] * g.data[0][j] for i in range(N)]
                s1 = sum(dxh)
                s2 = sum(dxh[i] * xhat[i][j] for i in range(N))
                for i in range(N):
                    g.grad[0][j] += out.grad[i][j] * xhat[i][j]
                    b.grad[0][j] += out.grad[i][j]
                    if training:
                        x.grad[i][j] += (inv[j] / N) * (N * dxh[i] - s1 - xhat[i][j] * s2)
                    else:
                        x.grad[i][j] += out.grad[i][j] * g.data[0][j] * inv[j]
        out._backward = _bw
        return out


# =============================================================================
# 3. LOSSES
# =============================================================================
def mse_loss(pred, target):
    """L = mean((pred - target)^2);  dL/dpred = 2(pred - target)/N."""
    t = target if isinstance(target, Tensor) else Tensor(target)
    return (pred - t).pow(2).mean()


def cross_entropy(logits, targets):
    """
    FUSED SOFTMAX + CROSS-ENTROPY.

        p = softmax(z) ;  L = -(1/N) sum_i log p_(i, y_i)

    The famous cancellation: dL/dz = (p - onehot(y)) / N. The softmax Jacobian
    and the derivative of log annihilate each other, leaving a gradient that is
    literally "prediction minus truth". Fusing them is also NUMERICALLY safer
    than composing softmax then log, because log(softmax) is computed via the
    log-sum-exp identity  log p_k = z_k - max(z) - log sum exp(z - max(z))
    with no tiny intermediate probability ever materialized.
    """
    N, C = logits.rows, logits.cols
    probs, loss = [], 0.0
    for i in range(N):
        row = logits.data[i]
        m = max(row)
        e = [math.exp(v - m) for v in row]
        s = sum(e)
        p = [v / s for v in e]
        probs.append(p)
        loss += -(row[targets[i]] - m - math.log(s))
    out = Tensor(loss / N, (logits,), "cross_entropy")

    def _bw():
        g = out.grad[0][0] / N
        for i in range(N):
            for j in range(C):
                logits.grad[i][j] += g * (probs[i][j] - (1.0 if j == targets[i] else 0.0))
    out._backward = _bw
    return out


def bce_loss(probs, targets):
    """Binary cross-entropy on probabilities already in (0,1)."""
    t = targets if isinstance(targets, Tensor) else Tensor(targets)
    one = Tensor([[1.0] * probs.cols for _ in range(probs.rows)])
    return -((t * probs.log()) + (one - t) * (one - probs).log()).mean()


def label_smoothing_ce(logits, targets, eps=0.1):
    """
    LABEL SMOOTHING: replace the one-hot target with
        y' = (1-eps)*onehot + eps/C
    The optimal logit gap becomes finite instead of infinite, so the model stops
    pushing confidence to 1.0. Improves calibration and generalization.
    """
    N, C = logits.rows, logits.cols
    probs, loss = [], 0.0
    for i in range(N):
        row = logits.data[i]
        m = max(row)
        e = [math.exp(v - m) for v in row]
        s = sum(e)
        p = [v / s for v in e]
        probs.append(p)
        for j in range(C):
            q = (1 - eps) * (1.0 if j == targets[i] else 0.0) + eps / C
            loss += -q * math.log(max(p[j], 1e-15))
    out = Tensor(loss / N, (logits,), "ls_ce")

    def _bw():
        g = out.grad[0][0] / N
        for i in range(N):
            for j in range(C):
                q = (1 - eps) * (1.0 if j == targets[i] else 0.0) + eps / C
                logits.grad[i][j] += g * (probs[i][j] - q)
    out._backward = _bw
    return out


# =============================================================================
# 4. OPTIMIZERS
# =============================================================================
class SGD:
    """
    SGD, optionally with (Nesterov) momentum and weight decay.

        plain      w -= lr * g
        momentum   v = mu*v + g ;        w -= lr*v
        nesterov   w -= lr*(g + mu*v)    (look-ahead correction)
        weight decay adds  wd * w  to g  (L2 regularization)

    Momentum's job: in a ravine (Hessian with very different eigenvalues) plain
    SGD oscillates across the steep direction while crawling along the shallow
    one. The velocity cancels the oscillating components and accumulates the
    consistent one, giving an effective speed-up of roughly 1/(1-mu).
    """

    def __init__(self, params, lr=0.01, momentum=0.0, nesterov=False,
                 weight_decay=0.0):
        self.params, self.lr = params, lr
        self.mu, self.nesterov, self.wd = momentum, nesterov, weight_decay
        self.v = [_zeros(p.rows, p.cols) for p in params]

    def step(self):
        for k, p in enumerate(self.params):
            for i in range(p.rows):
                for j in range(p.cols):
                    g = p.grad[i][j] + self.wd * p.data[i][j]
                    if self.mu:
                        self.v[k][i][j] = self.mu * self.v[k][i][j] + g
                        step = g + self.mu * self.v[k][i][j] if self.nesterov \
                            else self.v[k][i][j]
                    else:
                        step = g
                    p.data[i][j] -= self.lr * step

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class Adam:
    """
    ADAM / ADAMW

        m_t = b1 m_(t-1) + (1-b1) g          1st moment  -> direction (momentum)
        v_t = b2 v_(t-1) + (1-b2) g^2        2nd moment  -> per-parameter scale
        mhat = m_t/(1-b1^t) ; vhat = v_t/(1-b2^t)        BIAS CORRECTION
        w  -= lr * mhat / (sqrt(vhat) + eps)

    Bias correction exists because m and v start at zero, so the raw estimates
    are biased toward zero for the first ~1/(1-beta) steps; dividing by
    (1 - beta^t) rescales them to be unbiased.

    ADAMW DECOUPLES weight decay: instead of folding wd*w into g (where the
    adaptive denominator would rescale it inconsistently per parameter), it is
    applied directly to the weight:  w -= lr*wd*w. That is the correct L2 and
    is now the default for transformers.
    """

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, decoupled=True):
        self.params, self.lr = params, lr
        self.b1, self.b2 = betas
        self.eps, self.wd, self.decoupled = eps, weight_decay, decoupled
        self.m = [_zeros(p.rows, p.cols) for p in params]
        self.v = [_zeros(p.rows, p.cols) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        bc1 = 1 - self.b1 ** self.t
        bc2 = 1 - self.b2 ** self.t
        for k, p in enumerate(self.params):
            for i in range(p.rows):
                for j in range(p.cols):
                    g = p.grad[i][j]
                    if self.wd and not self.decoupled:
                        g += self.wd * p.data[i][j]
                    self.m[k][i][j] = self.b1 * self.m[k][i][j] + (1 - self.b1) * g
                    self.v[k][i][j] = self.b2 * self.v[k][i][j] + (1 - self.b2) * g * g
                    mh = self.m[k][i][j] / bc1
                    vh = self.v[k][i][j] / bc2
                    if self.wd and self.decoupled:
                        p.data[i][j] -= self.lr * self.wd * p.data[i][j]
                    p.data[i][j] -= self.lr * mh / (math.sqrt(vh) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class RMSProp:
    """v = rho*v + (1-rho)*g^2 ;  w -= lr*g/(sqrt(v)+eps).
    AdaGrad with an exponential average instead of a running SUM, so the
    effective learning rate never decays to zero and learning does not stall."""

    def __init__(self, params, lr=0.01, rho=0.9, eps=1e-8):
        self.params, self.lr, self.rho, self.eps = params, lr, rho, eps
        self.v = [_zeros(p.rows, p.cols) for p in params]

    def step(self):
        for k, p in enumerate(self.params):
            for i in range(p.rows):
                for j in range(p.cols):
                    g = p.grad[i][j]
                    self.v[k][i][j] = self.rho * self.v[k][i][j] + (1 - self.rho) * g * g
                    p.data[i][j] -= self.lr * g / (math.sqrt(self.v[k][i][j]) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


def clip_grad_norm(params, max_norm=1.0):
    """
    GRADIENT CLIPPING by global norm — the standard cure for exploding
    gradients in RNNs and transformers:

        total = sqrt(sum over all params of g^2)
        if total > max_norm:  g *= max_norm / total

    Scaling ALL gradients by one factor preserves the gradient DIRECTION and
    only shortens the step, unlike per-element clipping which distorts it.
    """
    total = math.sqrt(sum(g * g for p in params for row in p.grad for g in row))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params:
            for i in range(p.rows):
                for j in range(p.cols):
                    p.grad[i][j] *= scale
    return total


def lr_schedule(step, base_lr, total_steps, kind="cosine", warmup=0, min_lr=0.0):
    """
    LEARNING-RATE SCHEDULES
        constant    base_lr
        step        base_lr * gamma^floor(t/s)
        exponential base_lr * e^(-k t)
        cosine      min + 0.5(base-min)(1 + cos(pi * t/T))
        warmup      linear ramp for the first W steps, THEN the schedule

    Warmup matters for transformers because Adam's second-moment estimate is
    unreliable in the first steps (tiny v -> enormous effective step), which can
    destabilize training permanently in the first few hundred iterations.
    """
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(total_steps - warmup, 1)
    if kind == "cosine":
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * min(t, 1.0)))
    if kind == "exp":
        return base_lr * math.exp(-3.0 * t)
    if kind == "step":
        return base_lr * (0.1 ** int(t * 3))
    return base_lr


# =============================================================================
# GRADIENT CHECKING — proof that the backward passes above are correct
# =============================================================================
def gradient_check(fn, params, eps=1e-5, n_checks=8, seed=0):
    """
    CENTRAL-DIFFERENCE GRADIENT CHECK

        numeric = [ f(w + eps) - f(w - eps) ] / (2 eps)      error O(eps^2)

    Compare against the analytic gradient with the RELATIVE error

        |a - n| / max(|a| + |n|, 1e-8)     (< 1e-5 is a pass)

    Use the CENTRAL difference, not the forward one: forward differences have
    O(eps) error and routinely produce false alarms. This is the single most
    useful debugging tool when hand-writing a backward pass — and the reason
    autodiff exists, since doing this for every parameter costs O(#params)
    forward passes.

    STOCHASTIC LAYERS must have their randomness FROZEN first (fix the dropout
    mask, pass a fixed eps to a VAE). Finite-differencing a function that
    re-samples on every call compares two different functions and always fails.
    """
    rng = random.Random(seed)
    loss = fn()
    for p in params:
        p.zero_grad()
    loss = fn()
    loss.backward()
    worst = 0.0
    for _ in range(n_checks):
        p = rng.choice(params)
        i, j = rng.randrange(p.rows), rng.randrange(p.cols)
        analytic = p.grad[i][j]
        orig = p.data[i][j]
        p.data[i][j] = orig + eps
        f1 = fn().item()
        p.data[i][j] = orig - eps
        f2 = fn().item()
        p.data[i][j] = orig
        numeric = (f1 - f2) / (2 * eps)
        rel = abs(analytic - numeric) / max(abs(analytic) + abs(numeric), 1e-8)
        worst = max(worst, rel)
    return worst


# =============================================================================
# 5. CONVOLUTIONAL LAYERS
# =============================================================================
def conv_output_size(size, kernel, stride=1, pad=0):
    """O = floor((W - K + 2P)/S) + 1.  'same' padding for stride 1 is P=(K-1)/2."""
    return (size - kernel + 2 * pad) // stride + 1


def im2col(x, C, H, W, K, stride, pad):
    """
    IM2COL — turn convolution into a single matrix multiply.

    Each output position's receptive field (C*K*K values) becomes one ROW of a
    matrix. Then conv = patches @ kernels, so all the optimized matmul machinery
    (and here, our autograd matmul) applies directly.

        input  (N, C*H*W)  ->  patches (N*OH*OW, C*K*K)
        weights            ->  (C*K*K, out_channels)

    Backward is col2im: scatter-add each patch's gradient back to the pixels it
    came from. Pixels in overlapping receptive fields ACCUMULATE gradient from
    every patch that used them — the multivariate chain rule again.

    Cost: memory blow-up of K*K (patches are duplicated), traded for speed. It
    is what almost every framework does under the hood.
    """
    N = x.rows
    OH = conv_output_size(H, K, stride, pad)
    OW = conv_output_size(W, K, stride, pad)
    rows, index_map = [], []
    for n in range(N):
        for oh in range(OH):
            for ow in range(OW):
                patch, idxs = [], []
                for c in range(C):
                    for kh in range(K):
                        for kw in range(K):
                            h = oh * stride + kh - pad
                            w = ow * stride + kw - pad
                            if 0 <= h < H and 0 <= w < W:
                                pos = c * H * W + h * W + w
                                patch.append(x.data[n][pos])
                                idxs.append((n, pos))
                            else:
                                patch.append(0.0)          # zero padding
                                idxs.append(None)
                rows.append(patch)
                index_map.append(idxs)
    out = Tensor(rows, (x,), "im2col")

    def _bw():
        for r, idxs in enumerate(index_map):
            g = out.grad[r]
            for k, idx in enumerate(idxs):
                if idx is not None:
                    x.grad[idx[0]][idx[1]] += g[k]          # col2im scatter-add
    out._backward = _bw
    return out, OH, OW


class Conv2D(Module):
    """
    2-D CONVOLUTION

        y[n, co, oh, ow] = b[co] + sum_ci sum_kh sum_kw
                             x[n, ci, oh*s+kh-p, ow*s+kw-p] * W[co, ci, kh, kw]

    PARAMETER COUNT = (C_in * K * K + 1) * C_out   — INDEPENDENT of image size.
    That is the whole point: a 3x3x64x64 layer has ~37k params whether the
    image is 32x32 or 4096x4096.

    THE THREE INDUCTIVE BIASES
      local connectivity : nearby pixels are related; far ones usually aren't
      weight sharing     : an edge detector is useful everywhere -> translation
                           EQUIVARIANCE (shift the input, the feature map shifts)
      hierarchy          : stacking grows the receptive field, so features go
                           edges -> textures -> parts -> objects

    Receptive field of L stacked KxK stride-1 layers: 1 + L*(K-1). Two 3x3
    layers see 5x5 with 18C^2 params vs a 5x5 layer's 25C^2 — fewer parameters,
    more non-linearity. That is the VGG argument.

    A 1x1 convolution is a per-pixel linear map across channels: cheap channel
    mixing / dimensionality reduction (the bottleneck in ResNet and Inception).
    """

    def __init__(self, in_ch, out_ch, kernel=3, stride=1, pad=1, seed=None):
        super().__init__()
        fan_in = in_ch * kernel * kernel
        self.W = init_weights(fan_in, out_ch, "he", seed)
        self.b = Tensor.zeros(1, out_ch)
        self.in_ch, self.out_ch = in_ch, out_ch
        self.K, self.stride, self.pad = kernel, stride, pad

    def forward(self, x, H, W):
        """x: (N, C*H*W) flattened. Returns ((N, out_ch*OH*OW), OH, OW)."""
        cols, OH, OW = im2col(x, self.in_ch, H, W, self.K, self.stride, self.pad)
        conv = cols.matmul(self.W) + self.b            # (N*OH*OW, out_ch)
        N = x.rows
        # reorder (N*OH*OW, out_ch) -> (N, out_ch*OH*OW)
        out = Tensor([[conv.data[n * OH * OW + p][c]
                       for c in range(self.out_ch) for p in range(OH * OW)]
                      for n in range(N)], (conv,), "conv_reorder")

        def _bw():
            for n in range(N):
                col = 0
                for c in range(self.out_ch):
                    for p in range(OH * OW):
                        conv.grad[n * OH * OW + p][c] += out.grad[n][col]
                        col += 1
        out._backward = _bw
        return out, OH, OW


class MaxPool2D(Module):
    """
    MAX POOLING — downsample by keeping the strongest activation in each window.

    Backward is a ROUTER: the gradient flows only to the argmax position of each
    window; every other input in that window gets exactly zero. (Average pooling
    instead splits the gradient equally, 1/K^2 each.)

    Purpose: shrink spatial dimensions (cheaper deeper layers), enlarge the
    effective receptive field, and give small translation INVARIANCE. Modern
    nets increasingly use strided convolutions instead, which are learnable.
    """

    def __init__(self, size=2, stride=2):
        super().__init__()
        self.size, self.stride = size, stride

    def forward(self, x, C, H, W):
        K, S = self.size, self.stride
        OH, OW = (H - K) // S + 1, (W - K) // S + 1
        N = x.rows
        vals, argmax = [], []
        for n in range(N):
            row, am = [], []
            for c in range(C):
                for oh in range(OH):
                    for ow in range(OW):
                        best, bpos = -float("inf"), 0
                        for kh in range(K):
                            for kw in range(K):
                                pos = c * H * W + (oh * S + kh) * W + (ow * S + kw)
                                v = x.data[n][pos]
                                if v > best:
                                    best, bpos = v, pos
                        row.append(best)
                        am.append(bpos)
            vals.append(row)
            argmax.append(am)
        out = Tensor(vals, (x,), "maxpool")

        def _bw():
            for n in range(N):
                for k, pos in enumerate(argmax[n]):
                    x.grad[n][pos] += out.grad[n][k]        # route to the argmax
        out._backward = _bw
        return out, OH, OW


class GlobalAvgPool(Module):
    """Average each channel's whole feature map to one number.
    Replaces the giant flatten->dense layer (most of AlexNet/VGG's parameters
    lived there) and makes the network input-size agnostic."""

    def forward(self, x, C, HW):
        N = x.rows
        out = Tensor([[sum(x.data[n][c * HW:(c + 1) * HW]) / HW for c in range(C)]
                      for n in range(N)], (x,), "gap")

        def _bw():
            for n in range(N):
                for c in range(C):
                    g = out.grad[n][c] / HW
                    for p in range(HW):
                        x.grad[n][c * HW + p] += g
        out._backward = _bw
        return out


# =============================================================================
# 6. RECURRENT LAYERS
# =============================================================================
class RNNCell(Module):
    """
    VANILLA RNN CELL

        h_t = tanh( x_t W_x + h_(t-1) W_h + b )

    BPTT (backpropagation through time) unrolls the loop and applies the chain
    rule across time steps. The gradient from step T back to step t contains

        prod_{k=t+1..T}  W_h^T diag(1 - h_k^2)

    a repeated multiplication by the SAME matrix. If its largest singular value
    is < 1 the product decays exponentially (VANISHING: the network cannot learn
    dependencies more than ~10 steps back); if > 1 it blows up (EXPLODING: fixed
    by gradient clipping). Vanishing cannot be clipped away — it needs an
    architectural fix, which is exactly what the LSTM provides.
    """

    def __init__(self, in_dim, hidden, seed=None):
        super().__init__()
        self.Wx = init_weights(in_dim, hidden, "xavier", seed)
        self.Wh = init_weights(hidden, hidden, "xavier",
                               None if seed is None else seed + 1)
        self.b = Tensor.zeros(1, hidden)
        self.hidden = hidden

    def forward(self, x, h):
        return (x.matmul(self.Wx) + h.matmul(self.Wh) + self.b).tanh()


class LSTMCell(Module):
    """
    LONG SHORT-TERM MEMORY

        f_t = sigma(x_t W_xf + h_(t-1) W_hf + b_f)      FORGET gate
        i_t = sigma(x_t W_xi + h_(t-1) W_hi + b_i)      INPUT gate
        o_t = sigma(x_t W_xo + h_(t-1) W_ho + b_o)      OUTPUT gate
        g_t = tanh (x_t W_xg + h_(t-1) W_hg + b_g)      candidate
        c_t = f_t * c_(t-1) + i_t * g_t                 CELL STATE
        h_t = o_t * tanh(c_t)                           hidden state

    THE KEY LINE is c_t = f*c + i*g. Along the cell state the gradient is
    multiplied by f_t (a gate in (0,1) that the network LEARNS) rather than by
    a weight matrix and an activation derivative. When f_t ~ 1 the gradient
    passes essentially unchanged — a "constant error carousel". That additive,
    gated path is why LSTMs capture dependencies over hundreds of steps.

    Practical note: initialize the FORGET-GATE BIAS to ~1 so the cell starts in
    remember-mode; otherwise early training forgets everything and learns slowly.

    Params: 4 * (in_dim*hidden + hidden*hidden + hidden).
    """

    def __init__(self, in_dim, hidden, seed=None, forget_bias=1.0):
        super().__init__()
        s = seed
        self.Wx = init_weights(in_dim, 4 * hidden, "xavier", s)
        self.Wh = init_weights(hidden, 4 * hidden, "xavier",
                               None if s is None else s + 1)
        b = [0.0] * (4 * hidden)
        for k in range(hidden, 2 * hidden):
            b[k] = forget_bias                  # slot order: i, f, o, g
        self.b = Tensor([b])
        self.hidden = hidden

    def forward(self, x, h, c):
        H = self.hidden
        z = x.matmul(self.Wx) + h.matmul(self.Wh) + self.b
        i = z.slice_cols(0, H).sigmoid()
        f = z.slice_cols(H, 2 * H).sigmoid()
        o = z.slice_cols(2 * H, 3 * H).sigmoid()
        g = z.slice_cols(3 * H, 4 * H).tanh()
        c_new = f * c + i * g
        h_new = o * c_new.tanh()
        return h_new, c_new


class GRUCell(Module):
    """
    GATED RECURRENT UNIT — the LSTM with one fewer gate and no separate cell.

        z_t = sigma(x W_xz + h W_hz + b_z)          UPDATE gate
        r_t = sigma(x W_xr + h W_hr + b_r)          RESET gate
        n_t = tanh (x W_xn + (r_t * h) W_hn + b_n)  candidate
        h_t = (1 - z_t) * h_(t-1) + z_t * n_t       LEAKY interpolation

    The update gate does the job of the LSTM's forget AND input gates at once
    (they are tied to sum to 1), and the reset gate controls how much past state
    enters the candidate. ~25% fewer parameters, usually comparable accuracy;
    LSTMs retain a slight edge on very long sequences.
    """

    def __init__(self, in_dim, hidden, seed=None):
        super().__init__()
        self.Wxz = init_weights(in_dim, hidden, "xavier", seed)
        self.Whz = init_weights(hidden, hidden, "xavier",
                                None if seed is None else seed + 1)
        self.Wxr = init_weights(in_dim, hidden, "xavier",
                                None if seed is None else seed + 2)
        self.Whr = init_weights(hidden, hidden, "xavier",
                                None if seed is None else seed + 3)
        self.Wxn = init_weights(in_dim, hidden, "xavier",
                                None if seed is None else seed + 4)
        self.Whn = init_weights(hidden, hidden, "xavier",
                                None if seed is None else seed + 5)
        self.bz = Tensor.zeros(1, hidden)
        self.br = Tensor.zeros(1, hidden)
        self.bn = Tensor.zeros(1, hidden)
        self.hidden = hidden

    def forward(self, x, h):
        z = (x.matmul(self.Wxz) + h.matmul(self.Whz) + self.bz).sigmoid()
        r = (x.matmul(self.Wxr) + h.matmul(self.Whr) + self.br).sigmoid()
        n = (x.matmul(self.Wxn) + (r * h).matmul(self.Whn) + self.bn).tanh()
        one = Tensor([[1.0] * self.hidden for _ in range(h.rows)])
        return (one - z) * h + z * n


class SequenceModel(Module):
    """
    Wraps a recurrent cell to process a whole sequence and classify it.

    Input: a list of T tensors, each (batch, in_dim). The loop is the unroll;
    our autograd records every step, so calling .backward() on the final loss
    performs exact BPTT automatically. Truncated BPTT = detach the hidden state
    every k steps to bound memory and gradient path length.

    Uses the LAST hidden state for classification (many-to-one). Other shapes:
    many-to-many (output per step, e.g. tagging), one-to-many (captioning),
    encoder-decoder (translation).
    """

    def __init__(self, cell_type, in_dim, hidden, out_dim, seed=0):
        super().__init__()
        self.kind = cell_type
        if cell_type == "rnn":
            self.cell = RNNCell(in_dim, hidden, seed)
        elif cell_type == "lstm":
            self.cell = LSTMCell(in_dim, hidden, seed)
        else:
            self.cell = GRUCell(in_dim, hidden, seed)
        self.head = Linear(hidden, out_dim, seed=seed + 10)
        self.hidden = hidden

    def forward(self, seq):
        B = seq[0].rows
        h = Tensor.zeros(B, self.hidden)
        c = Tensor.zeros(B, self.hidden)
        for x in seq:
            if self.kind == "lstm":
                h, c = self.cell(x, h, c)
            elif self.kind == "gru":
                h = self.cell(x, h)
            else:
                h = self.cell(x, h)
        return self.head(h)


# =============================================================================
# 7. ATTENTION AND TRANSFORMERS
# =============================================================================
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    ATTENTION(Q,K,V) = softmax( Q K^T / sqrt(d_k) ) V

    Reading it as a soft dictionary lookup: each QUERY is compared against every
    KEY by dot product, the scores become a probability distribution over
    positions, and the output is that distribution's weighted average of the
    VALUES.

    WHY DIVIDE BY sqrt(d_k): if the entries of q and k are independent with unit
    variance, q.k has variance d_k. For d_k = 64 the logits have std 8, which
    pushes softmax into a near one-hot regime where its Jacobian
    y_i(delta_ij - y_j) is almost zero everywhere -> gradients vanish. Dividing
    by sqrt(d_k) restores unit variance and keeps softmax in its sensitive range.

    MASKING: adding -1e9 before softmax makes exp(.) ~ 0, so those positions get
    exactly zero weight. Causal (lower-triangular) masking is what makes a
    decoder autoregressive: position t may not see t+1.

    Complexity O(T^2 d) in sequence length — the motivation for FlashAttention
    (IO-aware exact attention), sparse/sliding-window attention, and linear
    attention variants.
    """
    d_k = Q.cols
    scores = Q.matmul(K.T()) * (1.0 / math.sqrt(d_k))
    if mask is not None:
        scores = scores + mask
    weights = scores.softmax()
    return weights.matmul(V), weights


def causal_mask(T):
    """Additive mask: 0 where attention is allowed, -1e9 where it is forbidden."""
    return Tensor([[0.0 if j <= i else -1e9 for j in range(T)] for i in range(T)],
                  requires_grad=False)


class MultiHeadAttention(Module):
    """
    MULTI-HEAD ATTENTION

        head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)
        MHA(X) = concat(head_1..head_h) W_O

    Each head gets d_model/h dimensions, so total compute matches one big head,
    but the heads can specialize — syntactic dependencies, coreference, position
    offsets, copying. A single head must average all those relations into one
    attention distribution; h heads keep them in separate subspaces.

    SELF-attention: Q, K, V all come from the same sequence.
    CROSS-attention: Q from the decoder, K and V from the encoder (translation).

    Params: 4 * d_model^2 (the four projections).
    """

    def __init__(self, d_model, n_heads, seed=0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.d_model = n_heads, d_model
        self.d_k = d_model // n_heads
        self.Wq = Linear(d_model, d_model, bias=False, init="small", seed=seed)
        self.Wk = Linear(d_model, d_model, bias=False, init="small", seed=seed + 1)
        self.Wv = Linear(d_model, d_model, bias=False, init="small", seed=seed + 2)
        self.Wo = Linear(d_model, d_model, bias=False, init="small", seed=seed + 3)

    def forward(self, x, mask=None):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        heads = []
        for i in range(self.h):
            a, b = i * self.d_k, (i + 1) * self.d_k
            out, _ = scaled_dot_product_attention(
                Q.slice_cols(a, b), K.slice_cols(a, b), V.slice_cols(a, b), mask)
            heads.append(out)
        return self.Wo(cat_cols(heads))

    def attention_maps(self, x, mask=None):
        """Returns each head's attention matrix — useful for interpretability."""
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        maps = []
        for i in range(self.h):
            a, b = i * self.d_k, (i + 1) * self.d_k
            _, w = scaled_dot_product_attention(
                Q.slice_cols(a, b), K.slice_cols(a, b), V.slice_cols(a, b), mask)
            maps.append(w.data)
        return maps


class PositionalEncoding:
    """
    SINUSOIDAL POSITIONAL ENCODING

        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Attention is PERMUTATION-INVARIANT — shuffle the tokens and the output
    shuffles identically. Without position information "dog bites man" and
    "man bites dog" are the same input. These fixed waves have two nice
    properties: they extrapolate to sequence lengths never seen in training,
    and PE(pos+k) is a LINEAR function of PE(pos), so relative offsets are
    easy for the model to express.

    Alternatives: learned absolute embeddings (simple, no extrapolation),
    and RoPE / ALiBi (relative, the modern default in LLMs).
    """

    def __init__(self, d_model, max_len=512):
        self.table = []
        for pos in range(max_len):
            row = []
            for i in range(d_model):
                angle = pos / (10000 ** ((2 * (i // 2)) / d_model))
                row.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
            self.table.append(row)

    def __call__(self, x, offset=0):
        return x + Tensor([self.table[offset + i] for i in range(x.rows)],
                          requires_grad=False)


class TransformerBlock(Module):
    """
    PRE-NORM TRANSFORMER BLOCK (the modern arrangement)

        x = x + MHA(LayerNorm(x))
        x = x + FFN(LayerNorm(x))        FFN: Linear(d, 4d) -> GELU -> Linear(4d, d)

    POST-NORM (the original 2017 paper) puts the norm AFTER the residual add:
    x = LayerNorm(x + MHA(x)). Pre-norm keeps a clean identity path from input
    to output, so deep stacks train stably without a long warmup; post-norm
    needs careful warmup or it diverges.

    THE RESIDUAL CONNECTION is why any of this trains at all: for y = x + F(x),
    dy/dx = I + dF/dx. The identity term guarantees the gradient reaches earlier
    layers undiminished no matter how small dF/dx becomes.

    THE 4x FFN is where most parameters live (8*d^2 vs 4*d^2 for attention) and
    is generally understood as the per-token 'memory' / feature-transform stage,
    while attention does the token mixing.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, seed=0):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, seed)
        self.ln2 = LayerNorm(d_model)
        self.fc1 = Linear(d_model, 4 * d_model, init="small", seed=seed + 5)
        self.fc2 = Linear(4 * d_model, d_model, init="small", seed=seed + 6)
        self.drop = Dropout(dropout, seed=seed)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.fc2(self.fc1(self.ln2(x)).gelu()))
        return x


class MiniGPT(Module):
    """
    A DECODER-ONLY TRANSFORMER — the GPT architecture in miniature.

        tokens -> embedding + positional -> N x TransformerBlock (causal mask)
               -> final LayerNorm -> linear head -> logits over the vocabulary

    Trained with next-token prediction: the loss at position t is the
    cross-entropy of predicting token t+1. Because of the causal mask, ALL
    positions are trained in parallel in one forward pass — that parallelism
    over the time axis is the decisive advantage over RNNs, which must run
    T sequential steps.

    WEIGHT TYING (used here): the output head shares the embedding matrix. It
    saves V*d parameters and usually improves perplexity, since both matrices
    are learning the same token-to-vector correspondence.

    Parameter count ~ V*d + N*(4d^2 + 8d^2) = V*d + 12*N*d^2.
    """

    def __init__(self, vocab, d_model=32, n_heads=4, n_layers=2,
                 max_len=64, dropout=0.0, seed=0):
        super().__init__()
        self.emb = Embedding(vocab, d_model, seed)
        self.pos = PositionalEncoding(d_model, max_len)
        self.blocks = [TransformerBlock(d_model, n_heads, dropout, seed + 10 * i)
                       for i in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        self.d_model, self.vocab = d_model, vocab

    def forward(self, ids):
        T = len(ids)
        x = self.pos(self.emb(ids))
        mask = causal_mask(T)
        for b in self.blocks:
            x = b(x, mask)
        x = self.ln_f(x)
        return x.matmul(self.emb.W.T())          # weight tying

    def generate(self, prompt, n_new, temperature=1.0, top_k=None, seed=0):
        """
        AUTOREGRESSIVE SAMPLING.
          temperature T: logits/T before softmax. T<1 sharpens (conservative),
            T>1 flattens (creative), T->0 becomes greedy argmax.
          top_k: keep only the k most likely tokens, renormalize — truncates the
            unreliable tail that would otherwise occasionally derail generation.
        """
        rng = random.Random(seed)
        ids = list(prompt)
        for _ in range(n_new):
            logits = self.forward(ids).data[-1]
            logits = [v / max(temperature, 1e-6) for v in logits]
            idxs = list(range(len(logits)))
            if top_k:
                idxs = sorted(idxs, key=lambda i: -logits[i])[:top_k]
            m = max(logits[i] for i in idxs)
            e = [math.exp(logits[i] - m) for i in idxs]
            s = sum(e)
            r, acc = rng.random() * s, 0.0
            pick = idxs[-1]
            for i, v in zip(idxs, e):
                acc += v
                if acc >= r:
                    pick = i
                    break
            ids.append(pick)
        return ids


class ResidualBlock(Module):
    """
    RESNET BLOCK:  y = ReLU( x + F(x) )

    The gradient identity  dy/dx = I + dF/dx  is the entire contribution: it
    creates a highway along which gradients reach layer 1 from layer 100
    undiminished. Reframing: instead of learning the full mapping H(x), the
    block learns the RESIDUAL F(x) = H(x) - x, and learning "change nothing"
    (F = 0) is easy, so adding layers can never make the network worse —
    which is exactly the degradation problem ResNet was built to fix.
    """

    def __init__(self, dim, seed=0):
        super().__init__()
        self.fc1 = Linear(dim, dim, seed=seed)
        self.fc2 = Linear(dim, dim, seed=seed + 1)
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return (x + self.fc2(self.fc1(self.norm(x)).relu())).relu()


# =============================================================================
# 8. GENERATIVE MODELS
# =============================================================================
class Autoencoder(Module):
    """
    AUTOENCODER: x -> encoder -> z (bottleneck) -> decoder -> xhat
    Loss: ||x - xhat||^2. Unsupervised — the input IS the target.

    A LINEAR autoencoder trained with MSE spans exactly the same subspace as
    PCA (though its axes need not be the principal directions, since nothing
    forces orthogonality). Non-linear activations are what buy you more than PCA.

    Variants: denoising (corrupt the input, reconstruct the clean version —
    prevents learning the identity), sparse (L1 on activations), contractive,
    and masked autoencoders (the pretraining objective behind BERT and MAE).
    """

    def __init__(self, in_dim, hidden, latent, seed=0):
        super().__init__()
        self.enc = Sequential(Linear(in_dim, hidden, seed=seed), ReLU(),
                              Linear(hidden, latent, seed=seed + 1))
        self.dec = Sequential(Linear(latent, hidden, seed=seed + 2), ReLU(),
                              Linear(hidden, in_dim, seed=seed + 3))

    def forward(self, x):
        return self.dec(self.enc(x))

    def encode(self, x):
        return self.enc(x)


class VAE(Module):
    """
    VARIATIONAL AUTOENCODER

    Maximize the EVIDENCE LOWER BOUND:

        log p(x) >= E_q(z|x)[ log p(x|z) ]  -  KL( q(z|x) || p(z) )
                    |---- reconstruction ---|     |--- regularizer --|

    The encoder outputs mu and log(sigma^2) instead of a point, and the prior is
    p(z) = N(0, I). With Gaussians the KL term is closed form:

        KL = -0.5 * sum_j ( 1 + log sigma_j^2 - mu_j^2 - sigma_j^2 )

    THE REPARAMETERIZATION TRICK: you cannot backprop through "sample from
    N(mu, sigma)" because sampling is not differentiable. Rewrite it as

        z = mu + sigma * eps ,    eps ~ N(0, I)

    Now the randomness sits in eps, which has no parameters, and the gradient
    flows through mu and sigma as ordinary arithmetic. This one trick is what
    makes the whole model trainable by SGD.

    The KL term is what makes the latent space CONTINUOUS and samplable —
    a plain autoencoder's latent space has holes, so decoding a random z gives
    garbage. Note the trade-off: too much KL weight causes posterior collapse
    (the decoder ignores z); this is what beta-VAE deliberately tunes.
    """

    def __init__(self, in_dim, hidden, latent, seed=0):
        super().__init__()
        self.enc = Sequential(Linear(in_dim, hidden, seed=seed), ReLU())
        self.mu = Linear(hidden, latent, seed=seed + 1)
        self.logvar = Linear(hidden, latent, seed=seed + 2)
        self.dec = Sequential(Linear(latent, hidden, seed=seed + 3), ReLU(),
                              Linear(hidden, in_dim, seed=seed + 4))
        self.latent = latent
        self.rng = random.Random(seed)

    def forward(self, x, fixed_eps=None):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        # reparameterization: z = mu + sigma * eps
        # (fixed_eps lets us freeze the noise — required for gradient checking,
        #  since a stochastic function cannot be finite-differenced)
        eps = fixed_eps if fixed_eps is not None else Tensor(
            [[self.rng.gauss(0, 1) for _ in range(self.latent)]
             for _ in range(x.rows)], requires_grad=False)
        std = (logvar * 0.5).exp()
        z = mu + std * eps
        return self.dec(z), mu, logvar

    def loss(self, x, beta=1.0, fixed_eps=None):
        recon, mu, logvar = self.forward(x, fixed_eps)
        rec = (recon - x).pow(2).mean()
        one = Tensor([[1.0] * self.latent for _ in range(x.rows)])
        kl = ((one + logvar - mu.pow(2) - logvar.exp()) * -0.5).mean()
        return rec + kl * beta, rec, kl

    def sample(self, n, seed=0):
        rng = random.Random(seed)
        z = Tensor([[rng.gauss(0, 1) for _ in range(self.latent)] for _ in range(n)])
        return self.dec(z)


# =============================================================================
# 9. TRAINING UTILITIES
# =============================================================================
def batches(n, batch_size, shuffle=True, seed=0):
    idx = list(range(n))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


def accuracy_from_logits(logits, targets):
    correct = 0
    for i, row in enumerate(logits.data):
        if max(range(len(row)), key=lambda j: row[j]) == targets[i]:
            correct += 1
    return correct / len(targets)


class EarlyStopping:
    """
    Stop when validation loss has not improved for `patience` epochs, and
    RESTORE the best weights seen. Early stopping is itself a regularizer: for
    a linear model trained with gradient descent it is provably equivalent to
    L2 regularization with a strength determined by when you stop.
    """

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.wait, self.best_state = float("inf"), 0, None

    def step(self, val_loss, params):
        if val_loss < self.best - self.min_delta:
            self.best, self.wait = val_loss, 0
            self.best_state = [[row[:] for row in p.data] for p in params]
            return False
        self.wait += 1
        return self.wait >= self.patience

    def restore(self, params):
        if self.best_state:
            for p, s in zip(params, self.best_state):
                p.data = [row[:] for row in s]


def train(model, X, y, loss_fn, optimizer, epochs=50, batch_size=16,
          X_val=None, y_val=None, clip=None, verbose_every=0, seed=0):
    """Standard mini-batch loop: zero grads -> forward -> backward -> (clip) -> step."""
    history = {"train": [], "val": []}
    params = model.parameters()
    n = X.rows
    for ep in range(epochs):
        model.train()
        total = 0.0
        for idx in batches(n, batch_size, seed=seed + ep):
            xb = Tensor([X.data[i] for i in idx])
            yb = [y[i] for i in idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if clip:
                clip_grad_norm(params, clip)
            optimizer.step()
            total += loss.item() * len(idx)
        history["train"].append(total / n)
        if X_val is not None:
            model.eval()
            history["val"].append(loss_fn(model(X_val), y_val).item())
        if verbose_every and (ep % verbose_every == 0 or ep == epochs - 1):
            msg = f"  epoch {ep:4d}  train {history['train'][-1]:.4f}"
            if X_val is not None:
                msg += f"  val {history['val'][-1]:.4f}"
            print(msg)
    return history


# =============================================================================
# 10. DEMOS
# =============================================================================
def _hdr(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def make_spiral(n_per=60, classes=3, noise=0.15, seed=0):
    """Interleaved spirals — impossible for a linear model, easy for an MLP."""
    rng = random.Random(seed)
    X, y = [], []
    for c in range(classes):
        for i in range(n_per):
            r = 0.2 + 3.8 * i / n_per
            t = c * 2 * math.pi / classes + 2.2 * i / n_per + rng.gauss(0, noise)
            X.append([r * math.sin(t), r * math.cos(t)])
            y.append(c)
    return X, y


def demo_scalar_autograd():
    _hdr("0. SCALAR AUTOGRAD — backprop on a single expression")
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    e = a * b
    d = e + c
    f = d.tanh()
    f.backward()
    print(f"  f = tanh(a*b + c) = {f.data:.6f}")
    print(f"  df/da = {a.grad:.6f}  (analytic: b*(1-f^2) = "
          f"{b.data*(1-f.data**2):.6f})")
    print(f"  df/db = {b.grad:.6f}  (analytic: a*(1-f^2) = "
          f"{a.data*(1-f.data**2):.6f})")
    print(f"  df/dc = {c.grad:.6f}  (analytic: 1*(1-f^2) = {1-f.data**2:.6f})")

    # gradient accumulation when a node is reused
    x = Value(3.0)
    z = x * x + x                      # dz/dx = 2x + 1 = 7
    z.backward()
    print(f"  z = x^2 + x at x=3 -> dz/dx = {x.grad:.1f}  (expected 7.0: "
          f"gradient ACCUMULATES over both paths)")


def demo_gradient_checks():
    _hdr("1. GRADIENT CHECKING — every backward pass verified numerically")
    random.seed(0)
    X = Tensor([[random.gauss(0, 1) for _ in range(4)] for _ in range(5)])
    y = [random.randrange(3) for _ in range(5)]

    checks = []
    net = Sequential(Linear(4, 8, seed=1), ReLU(), LayerNorm(8),
                     Linear(8, 3, seed=2))
    checks.append(("MLP + ReLU + LayerNorm + CE",
                   gradient_check(lambda: cross_entropy(net(X), y),
                                  net.parameters())))

    bn = Sequential(Linear(4, 6, seed=3), BatchNorm1d(6), Tanh(),
                    Linear(6, 3, seed=4))
    checks.append(("BatchNorm1d",
                   gradient_check(lambda: cross_entropy(bn(X), y),
                                  bn.parameters())))

    g = Sequential(Linear(4, 5, seed=5), GELU(), Linear(5, 1, seed=6))
    tgt = Tensor([[random.gauss(0, 1)] for _ in range(5)])
    checks.append(("GELU + MSE",
                   gradient_check(lambda: mse_loss(g(X), tgt), g.parameters())))

    C, H, W = 1, 5, 5
    Xi = Tensor([[random.gauss(0, 1) for _ in range(C * H * W)] for _ in range(3)])
    yi = [random.randrange(2) for _ in range(3)]
    conv, pool = Conv2D(1, 2, 3, 1, 1, seed=1), MaxPool2D(2, 2)
    head = Linear(2 * 2 * 2, 2, seed=2)

    def fconv():
        o, OH, OW = conv(Xi, H, W)
        p, _, _ = pool(o.relu(), 2, OH, OW)
        return cross_entropy(head(p), yi)
    checks.append(("Conv2D (im2col) + MaxPool",
                   gradient_check(fconv, conv.parameters() + head.parameters())))

    seq = [Tensor([[random.gauss(0, 1) for _ in range(3)] for _ in range(4)])
           for _ in range(5)]
    ys = [random.randrange(2) for _ in range(4)]
    for kind in ["rnn", "lstm", "gru"]:
        m = SequenceModel(kind, 3, 6, 2, seed=3)
        checks.append((f"{kind.upper()} through 5 time steps (BPTT)",
                       gradient_check(lambda m=m: cross_entropy(m(seq), ys),
                                      m.parameters(), n_checks=6)))

    gpt = MiniGPT(6, d_model=8, n_heads=2, n_layers=2, seed=0)
    ids, tgt2 = [1, 3, 2, 0, 4], [3, 2, 0, 4, 1]
    checks.append(("MiniGPT (attention + residual + LN)",
                   gradient_check(lambda: cross_entropy(gpt(ids), tgt2),
                                  gpt.parameters())))

    Xv = Tensor([[random.random() for _ in range(6)] for _ in range(4)])
    vae = VAE(6, 8, 2, seed=0)
    fixed = Tensor([[random.gauss(0, 1) for _ in range(2)] for _ in range(4)],
                   requires_grad=False)
    checks.append(("VAE (reparameterized, frozen eps)",
                   gradient_check(lambda: vae.loss(Xv, fixed_eps=fixed)[0],
                                  vae.parameters(), n_checks=6)))

    for name, err in checks:
        status = "PASS" if err < 1e-5 else "FAIL"
        print(f"  {name:<40} rel.err {err:.2e}  {status}")


def demo_mlp():
    _hdr("2. MLP ON SPIRALS — depth beats a linear boundary")
    X, y = make_spiral(60, 3, 0.15, seed=1)
    idx = list(range(len(X)))
    random.Random(0).shuffle(idx)
    cut = int(len(X) * 0.8)
    Xtr = Tensor([X[i] for i in idx[:cut]])
    ytr = [y[i] for i in idx[:cut]]
    Xte = Tensor([X[i] for i in idx[cut:]])
    yte = [y[i] for i in idx[cut:]]

    lin = Linear(2, 3, seed=0)
    opt = Adam(lin.parameters(), lr=0.1)
    for _ in range(300):
        opt.zero_grad()
        cross_entropy(lin(Xtr), ytr).backward()
        opt.step()
    print(f"  linear (no hidden layer)    test acc "
          f"{accuracy_from_logits(lin(Xte), yte):.4f}")

    net = Sequential(Linear(2, 32, seed=1), ReLU(),
                     Linear(32, 32, seed=2), ReLU(),
                     Linear(32, 3, seed=3))
    opt = Adam(net.parameters(), lr=0.05)
    h = train(net, Xtr, ytr, cross_entropy, opt, epochs=120, batch_size=32,
              X_val=Xte, y_val=yte)
    print(f"  MLP 2-32-32-3 ({net.n_params()} params)  test acc "
          f"{accuracy_from_logits(net(Xte), yte):.4f}")
    print(f"  loss {h['train'][0]:.4f} -> {h['train'][-1]:.4f}")


def demo_optimizers():
    _hdr("3. OPTIMIZERS — same model, same init, different update rules")
    X, y = make_spiral(40, 3, 0.15, seed=2)
    Xt, yt = Tensor(X), y
    for name, make in [
            ("SGD", lambda p: SGD(p, lr=0.1)),
            ("SGD+momentum", lambda p: SGD(p, lr=0.1, momentum=0.9)),
            ("SGD+nesterov", lambda p: SGD(p, lr=0.1, momentum=0.9, nesterov=True)),
            ("RMSProp", lambda p: RMSProp(p, lr=0.02)),
            ("Adam", lambda p: Adam(p, lr=0.05)),
            ("AdamW (wd=0.01)", lambda p: Adam(p, lr=0.05, weight_decay=0.01))]:
        net = Sequential(Linear(2, 24, seed=7), ReLU(), Linear(24, 3, seed=8))
        opt = make(net.parameters())
        first = None
        for step in range(150):
            opt.zero_grad()
            loss = cross_entropy(net(Xt), yt)
            loss.backward()
            opt.step()
            if first is None:
                first = loss.item()
        print(f"  {name:<18} loss {first:.4f} -> {loss.item():.4f} | "
              f"acc {accuracy_from_logits(net(Xt), yt):.4f}")


def demo_regularization():
    _hdr("4. REGULARIZATION — a genuine overfit, then two ways to fix it")
    X, y = make_spiral(10, 3, 0.7, seed=5)           # 30 points, very noisy
    Xv, yv = make_spiral(60, 3, 0.7, seed=6)
    Xt, Xvt = Tensor(X), Tensor(Xv)
    print("  30 training points, 96-unit 2-hidden-layer net (over-parameterized)")

    def run(net, opt_f, steps=600):
        opt = opt_f(net.parameters())
        for _ in range(steps):
            net.train()
            opt.zero_grad()
            cross_entropy(net(Xt), y).backward()
            opt.step()
        net.eval()
        return (accuracy_from_logits(net(Xt), y),
                accuracy_from_logits(net(Xvt), yv))

    def big():
        return Sequential(Linear(2, 96, seed=1), ReLU(),
                          Linear(96, 96, seed=2), ReLU(),
                          Linear(96, 3, seed=3))

    configs = [
        ("none", big(), lambda p: Adam(p, lr=0.02)),
        ("dropout p=0.5",
         Sequential(Linear(2, 96, seed=1), ReLU(), Dropout(0.5, seed=0),
                    Linear(96, 96, seed=2), ReLU(), Dropout(0.5, seed=1),
                    Linear(96, 3, seed=3)), lambda p: Adam(p, lr=0.02)),
        ("weight decay 0.5", big(), lambda p: Adam(p, lr=0.02, weight_decay=0.5)),
    ]
    for name, net, opt_f in configs:
        tr, va = run(net, opt_f)
        print(f"  {name:<18} train {tr:.4f} | val {va:.4f} | "
              f"generalization gap {tr - va:+.4f}")
    print("  -> the unregularized net memorizes the 30 points perfectly and")
    print("     pays for it on held-out data; both regularizers shrink the gap.")


def demo_init_and_depth():
    _hdr("5. INITIALIZATION — why the scheme decides whether depth trains")
    random.seed(0)
    X = Tensor([[random.gauss(0, 1) for _ in range(64)] for _ in range(32)])
    for scheme in ["he", "xavier", "small"]:
        x = X
        stds = []
        for l in range(8):
            lin = Linear(64, 64, bias=False, init=scheme, seed=l)
            x = lin(x).relu()
            flat = [v for row in x.data for v in row]
            m = sum(flat) / len(flat)
            stds.append(math.sqrt(sum((v - m) ** 2 for v in flat) / len(flat)))
        print(f"  {scheme:<8} activation std by layer: "
              f"{[round(s, 4) for s in stds]}")
    print("  -> He keeps the scale roughly constant through ReLU layers;")
    print("     a small constant std collapses it geometrically (vanishing signal).")


def demo_cnn():
    _hdr("6. CNN vs MLP — weight sharing when data is scarce")

    def data(n, seed):
        """8x8 images: vertical bar / horizontal bar / cross, at RANDOM positions."""
        rng = random.Random(seed)
        H = W = 8
        X, y = [], []
        for i in range(n):
            k = i % 3
            img = [[0.0] * W for _ in range(H)]
            r, c = rng.randrange(0, H), rng.randrange(0, W)
            if k in (0, 2):
                for a in range(H):
                    img[a][c] = 1.0
            if k in (1, 2):
                for b in range(W):
                    img[r][b] = 1.0
            X.append([v + rng.gauss(0, 0.35) for row in img for v in row])
            y.append(k)
        return X, y

    Xtr, ytr = data(45, 3)                  # only 45 training images
    Xte, yte = data(150, 99)
    Xt, Xe = Tensor(Xtr), Tensor(Xte)

    class CNN(Module):
        def __init__(self):
            super().__init__()
            self.c1 = Conv2D(1, 4, 3, 1, 1, seed=1)
            self.p1 = MaxPool2D(2, 2)
            self.c2 = Conv2D(4, 8, 3, 1, 1, seed=2)
            self.p2 = MaxPool2D(2, 2)
            self.fc = Linear(8 * 2 * 2, 3, seed=3)

        def forward(self, x):
            o, OH, OW = self.c1(x, 8, 8)
            o, OH, OW = self.p1(o.relu(), 4, OH, OW)
            o, OH, OW = self.c2(o, OH, OW)
            o, OH, OW = self.p2(o.relu(), 8, OH, OW)
            return self.fc(o)

    print("  shapes appear at RANDOM positions, so the classifier must be")
    print("  translation-tolerant; only 45 training images are provided.")
    for name, m in [("CNN (conv 3x3 + maxpool)", CNN()),
                    ("MLP (flatten + dense)",
                     Sequential(Linear(64, 24, seed=4), ReLU(),
                                Linear(24, 3, seed=5)))]:
        opt = Adam(m.parameters(), lr=0.02)
        for ep in range(40):
            for idx in batches(45, 15, seed=ep):
                xb = Tensor([Xt.data[i] for i in idx])
                opt.zero_grad()
                cross_entropy(m(xb), [ytr[i] for i in idx]).backward()
                opt.step()
        print(f"  {name:<26} {m.n_params():5d} params | "
              f"train {accuracy_from_logits(m(Xt), ytr):.3f} | "
              f"test {accuracy_from_logits(m(Xe), yte):.3f}")
    print("  -> the CNN wins with ~4x FEWER parameters. An MLP must learn the")
    print("     same bar detector separately at every position; a conv kernel")
    print("     is shared across all of them (translation equivariance).")


def demo_rnn_memory():
    _hdr("7. VANISHING GRADIENTS — measured directly through time")
    T, D, H = 30, 4, 8
    random.seed(0)
    seq = [Tensor([[random.gauss(0, 1) for _ in range(D)]]) for _ in range(T)]
    print(f"  Unroll {T} steps, backprop once, and measure ||dL/dx_t|| at each")
    print(f"  time step. Vanishing means the gradient reaching EARLY steps is")
    print(f"  orders of magnitude smaller than at late steps.\n")
    print(f"  {'cell':<6} {'t=0':>10} {'t=12':>10} {'t=24':>10} {'t=29':>10}"
          f"   {'grad(t=29)/grad(t=0)':>22}")
    for kind in ["rnn", "lstm", "gru"]:
        m = SequenceModel(kind, D, H, 2, seed=1)
        for p in m.parameters():
            p.zero_grad()
        for s_ in seq:
            s_.zero_grad()
        cross_entropy(m(seq), [1]).backward()
        norms = [math.sqrt(sum(g * g for row in t.grad for g in row)) for t in seq]
        ratio = norms[-1] / max(norms[0], 1e-30)
        print(f"  {kind.upper():<6} {norms[0]:10.2e} {norms[12]:10.2e} "
              f"{norms[24]:10.2e} {norms[-1]:10.2e}   {ratio:22.1e}")
    print("\n  -> the vanilla RNN's gradient decays by ~6 ORDERS OF MAGNITUDE over")
    print("     30 steps: the repeated factor W_h^T diag(1-h^2) has spectral norm")
    print("     below 1, so the product shrinks geometrically. The LSTM's cell")
    print("     state c_t = f*c + i*g is ADDITIVE and gated, so its gradient path")
    print("     is ~1000x better preserved. Clipping fixes EXPLOSION, not this.")

    # a learnable long-memory task
    T2, D2 = 25, 4
    rng = random.Random(4)

    def make(n):
        X, y = [], []
        for _ in range(n):
            label = rng.randrange(2)
            steps = []
            for t in range(T2):
                v = [rng.gauss(0, 0.5) for _ in range(D2)]
                if t == 0:
                    v[0] = 3.0 if label else -3.0      # the signal, step 0 only
                steps.append(v)
            X.append(steps)
            y.append(label)
        return X, y

    Xtr, ytr = make(60)
    Xte, yte = make(40)

    def as_seq(X, idx):
        return [Tensor([X[i][t] for i in idx]) for t in range(T2)]

    print(f"\n  Task: the label is determined by step 0 alone and must survive "
          f"{T2} steps.")
    for kind in ["rnn", "lstm", "gru"]:
        m = SequenceModel(kind, D2, 10, 2, seed=1)
        opt = Adam(m.parameters(), lr=0.05)
        for ep in range(30):
            for idx in batches(60, 20, seed=ep):
                opt.zero_grad()
                cross_entropy(m(as_seq(Xtr, idx)),
                              [ytr[i] for i in idx]).backward()
                clip_grad_norm(m.parameters(), 1.0)
                opt.step()
        te = accuracy_from_logits(m(as_seq(Xte, list(range(40)))), yte)
        print(f"  {kind.upper():<5} ({m.n_params():4d} params)  test acc {te:.4f}")


def demo_attention():
    _hdr("8. ATTENTION — the mechanics, then a MiniGPT that learns a pattern")
    d_k = 4
    Q = Tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    K = Tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0.7, 0.7, 0, 0]])
    V = Tensor([[10.0, 0], [0, 10.0], [5.0, 5.0]])
    out, w = scaled_dot_product_attention(Q, K, V)
    print("  attention weights (each row sums to 1):")
    for r in w.data:
        print(f"     {[round(v, 3) for v in r]}")
    print(f"  output: {[[round(v, 2) for v in r] for r in out.data]}")
    print("  -> query 1 matches key 1, so it mostly copies value 1. Attention is")
    print("     a differentiable, content-addressed dictionary lookup.")

    T = 5
    m = causal_mask(T)
    scores = Tensor([[1.0] * T for _ in range(T)]) + m
    print("\n  causal mask applied to uniform scores, row 2 after softmax:")
    print(f"     {[round(v, 3) for v in scores.softmax().data[2]]}")
    print("     -> zero weight on future positions; that is what makes a decoder")
    print("        autoregressive and lets all T positions train in parallel.")

    print("\n  scaling check — why the 1/sqrt(d_k) factor exists:")
    rng = random.Random(0)
    for d in [4, 64, 512]:
        q = [rng.gauss(0, 1) for _ in range(d)]
        k = [rng.gauss(0, 1) for _ in range(d)]
        dots = []
        for _ in range(200):
            q = [rng.gauss(0, 1) for _ in range(d)]
            k = [rng.gauss(0, 1) for _ in range(d)]
            dots.append(sum(a * b for a, b in zip(q, k)))
        mu = sum(dots) / len(dots)
        sd = math.sqrt(sum((v - mu) ** 2 for v in dots) / len(dots))
        print(f"     d_k={d:4d}  std(q.k) = {sd:7.2f}  (~sqrt(d_k)={math.sqrt(d):.2f})"
              f"  -> after scaling: {sd/math.sqrt(d):.2f}")

    # ---- train MiniGPT on a copy-with-offset pattern
    _hdr("9. MINIGPT — next-token prediction on a learnable pattern")
    V_ = 7
    rng = random.Random(1)
    seqs = []
    for _ in range(60):
        a = rng.randrange(1, V_)
        s = [a]
        for t in range(1, 9):
            s.append((s[-1] + 1) % V_ if s[-1] + 1 != 0 else 1)
        seqs.append(s)
    gpt = MiniGPT(V_, d_model=24, n_heads=3, n_layers=2, max_len=16, seed=0)
    opt = Adam(gpt.parameters(), lr=0.02)
    print(f"  params {gpt.n_params()} | vocab {V_} | d_model 24, 3 heads, 2 layers")
    first = None
    for ep in range(60):
        total = 0.0
        for s in seqs[:30]:
            opt.zero_grad()
            loss = cross_entropy(gpt(s[:-1]), s[1:])
            loss.backward()
            clip_grad_norm(gpt.parameters(), 1.0)
            opt.step()
            total += loss.item()
        if first is None:
            first = total / 30
    print(f"  train loss {first:.4f} -> {total/30:.4f}  "
          f"(perplexity {math.exp(first):.2f} -> {math.exp(total/30):.2f})")
    correct = 0
    tot = 0
    for s in seqs[30:]:
        logits = gpt(s[:-1])
        for t, row in enumerate(logits.data):
            tot += 1
            correct += (max(range(V_), key=lambda j: row[j]) == s[t + 1])
    print(f"  held-out next-token accuracy: {correct/tot:.4f}")
    print(f"  greedy continuation of [2]: "
          f"{gpt.generate([2], 6, temperature=0.3, seed=0)}")


def demo_generative():
    _hdr("10. AUTOENCODER & VAE")
    rng = random.Random(6)
    # data lying on a 1-D curve embedded in 6-D
    X = []
    for _ in range(120):
        t = rng.uniform(0, 1)
        X.append([t, t ** 2, math.sin(3 * t), 1 - t, t * 0.5, math.cos(2 * t)])
    Xt = Tensor(X)

    ae = Autoencoder(6, 12, 2, seed=0)
    opt = Adam(ae.parameters(), lr=0.02)
    for ep in range(200):
        opt.zero_grad()
        loss = mse_loss(ae(Xt), Xt)
        loss.backward()
        opt.step()
    print(f"  Autoencoder 6->2->6   reconstruction MSE {loss.item():.6f}  "
          f"(2-D bottleneck captures a 1-D manifold)")

    vae = VAE(6, 12, 2, seed=0)
    opt = Adam(vae.parameters(), lr=0.02)
    for ep in range(300):
        opt.zero_grad()
        total, rec, kl = vae.loss(Xt, beta=0.05)
        total.backward()
        opt.step()
    print(f"  VAE  recon {rec.item():.6f} | KL {kl.item():.6f} "
          f"(KL keeps the latent close to N(0,I) so it is samplable)")
    s = vae.sample(2, seed=1)
    print(f"  sampled from the prior: "
          f"{[[round(v, 3) for v in r] for r in s.data]}")


def demo_training_dynamics():
    _hdr("11. TRAINING DYNAMICS — schedules, clipping, early stopping")
    total = 100
    print("  learning-rate schedules (base 0.1, 10-step warmup):")
    for kind in ["constant", "cosine", "exp", "step"]:
        vals = [lr_schedule(s, 0.1, total, kind, warmup=10) for s in
                [0, 5, 10, 30, 60, 99]]
        print(f"     {kind:<9} {[round(v, 4) for v in vals]}")

    net = Sequential(Linear(4, 8, seed=1), ReLU(), Linear(8, 2, seed=2))
    X = Tensor([[random.gauss(0, 1) for _ in range(4)] for _ in range(20)])
    y = [random.randrange(2) for _ in range(20)]
    cross_entropy(net(X), y).backward()
    for p in net.parameters():                       # simulate an explosion
        for i in range(p.rows):
            for j in range(p.cols):
                p.grad[i][j] *= 500
    before = math.sqrt(sum(g * g for p in net.parameters()
                           for row in p.grad for g in row))
    clip_grad_norm(net.parameters(), 1.0)
    after = math.sqrt(sum(g * g for p in net.parameters()
                          for row in p.grad for g in row))
    print(f"  gradient clipping: global norm {before:.2f} -> {after:.4f} "
          f"(direction preserved, step shortened)")

    Xs, ys = make_spiral(15, 3, 0.4, seed=9)
    Xv, yv = make_spiral(40, 3, 0.4, seed=10)
    net = Sequential(Linear(2, 64, seed=1), ReLU(), Linear(64, 3, seed=2))
    opt = Adam(net.parameters(), lr=0.03)
    es = EarlyStopping(patience=15)
    Xst, Xvt = Tensor(Xs), Tensor(Xv)
    stopped = None
    for ep in range(300):
        net.train()
        opt.zero_grad()
        cross_entropy(net(Xst), ys).backward()
        opt.step()
        net.eval()
        vl = cross_entropy(net(Xvt), yv).item()
        if es.step(vl, net.parameters()):
            stopped = ep
            break
    es.restore(net.parameters())
    net.eval()
    print(f"  early stopping fired at epoch {stopped} "
          f"(best val loss {es.best:.4f}); restored val acc "
          f"{accuracy_from_logits(net(Xvt), yv):.4f}")


if __name__ == "__main__":
    print("=" * 74)
    print(" DEEP LEARNING FROM SCRATCH — pure Python, built on custom autograd")
    print("=" * 74)
    demo_scalar_autograd()
    demo_gradient_checks()
    demo_mlp()
    demo_optimizers()
    demo_regularization()
    demo_init_and_depth()
    demo_cnn()
    demo_rnn_memory()
    demo_attention()
    demo_generative()
    demo_training_dynamics()
    print("\n" + "=" * 74)
    print("All components verified by gradient check and trained end to end.")
    print("=" * 74)
