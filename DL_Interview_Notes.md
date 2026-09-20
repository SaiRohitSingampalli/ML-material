# DEEP LEARNING — COMPLETE INTERVIEW NOTES

> Companion to `ML_Interview_Notes.md`. Math in ASCII code blocks so it renders anywhere.
> Working code: `dl_from_scratch.py` — a full autograd engine, every layer below, every backward pass verified by gradient check.

---

## THE 6 QUESTIONS BEHIND EVERY DL INTERVIEW QUESTION

1. What is the **forward** computation?
2. What is the **backward** computation, and does the gradient survive depth?
3. What **inductive bias** does this architecture encode, and does it match the data?
4. What is the **parameter count** and the **compute/memory complexity**?
5. What **fails** — and what is the specific symptom you'd see in the loss curve?
6. What is the **modern alternative**, and why did it win?

---

# PART 0 — CHEAT SHEET

```
ACTIVATIONS                      f(x)                    f'(x)
  sigmoid        1/(1+e^-x)                        s(1-s)        max 0.25
  tanh           (e^x-e^-x)/(e^x+e^-x)             1-t^2         max 1.0
  ReLU           max(0,x)                          1 or 0
  LeakyReLU      max(ax,x)                         1 or a
  GELU           x*Phi(x)                          smooth        transformer default
  SiLU/Swish     x*sigmoid(x)                      s+xs(1-s)
  softmax        e^zk / sum e^zj                   y_i(d_ij - y_j)

OUTPUT LAYER + LOSS PAIRINGS (all give delta_L = a - y)
  regression       linear   + MSE
  binary           sigmoid  + BCE
  multiclass       softmax  + categorical CE
  multilabel       sigmoid per class + BCE per class

INIT              Var(W)
  He / Kaiming    2/fan_in              ReLU family
  Xavier/Glorot   2/(fan_in+fan_out)    tanh, sigmoid
  LeCun           1/fan_in              SELU
  Transformer     ~0.02 constant std, scaled residual branches

KEY FORMULAS
  conv out size       O = floor((W - K + 2P)/S) + 1
  conv params         (C_in*K*K + 1) * C_out
  receptive field     1 + L*(K-1)  for L stacked KxK stride-1 layers
  attention           softmax(QK^T / sqrt(d_k)) V
  transformer params  ~ 12 * n_layers * d_model^2  (+ vocab*d_model)
  LSTM params         4*(in*h + h*h + h)

COMPLEXITY per layer
  Dense      O(B * d_in * d_out)
  Conv       O(B * C_in * C_out * K^2 * H_out * W_out)
  RNN        O(B * T * h * (in + h))   sequential in T
  Attention  O(B * T^2 * d)            parallel in T
```

**Symptom → cause table (memorize this one):**

| Symptom | Likely cause |
|---|---|
| Loss = NaN | LR too high, exploding gradients, log(0), division by zero, bad init |
| Loss flat from step 0 | LR too small, dead ReLUs, zero init, disconnected graph, frozen params |
| Train loss falls, val rises | Overfitting |
| Both stuck high | Underfitting: too little capacity, too much regularization, bad features |
| Train loss spikes periodically | LR too high, bad batch, missing grad clipping |
| Works in train, broken at eval | Forgot `.eval()` — BatchNorm/Dropout still in train mode |
| Val loss < train loss | Dropout active during training only (normal), or a leaky split |
| Fine on 1 GPU, broken on 8 | BatchNorm across small per-device batches, LR not scaled |

---

# PART 1 — BACKPROPAGATION

## 1.1 The forward and backward pass

```
FORWARD, layer l:
    z^l = W^l a^(l-1) + b^l
    a^l = f(z^l)                 a^0 = x

BACKWARD (chain rule, right to left):
    output:  delta^L = dL/da^L  *  f'(z^L)
    hidden:  delta^l = (W^(l+1))^T delta^(l+1)  *  f'(z^l)
    grads:   dL/dW^l = delta^l (a^(l-1))^T
             dL/db^l = delta^l
```

The two matrix-calculus identities that make this mechanical — derive everything else from them:
```
C = A B   ->   dL/dA = dL/dC B^T ,   dL/dB = A^T dL/dC
```
Shapes force the answer: `dL/dA` must be (n,m), and (n,p)@(p,m) is the only way to get it.

## 1.2 The cancellation you must be able to derive

For softmax + cross-entropy:
```
p_k = e^{z_k}/sum_j e^{z_j} ,  L = -log p_y

dL/dz_i = sum_k (dL/dp_k)(dp_k/dz_i)
        = -1/p_y * p_y(delta_iy - p_i)
        = p_i - 1{i = y}
```
So `delta_L = p - onehot(y)` — literally "prediction minus truth". The same happens for sigmoid+BCE and linear+MSE. This is why those three pairings are standard: the gradient is clean, bounded, and never vanishes at the output layer.

**Follow-up they will ask:** *what if you used MSE with a sigmoid output?* Then `dL/dz = (a-y)·a(1-a)`, and when the model is confidently wrong (a≈1, y=0) the factor `a(1-a)≈0` kills the gradient. The network learns slowest exactly where the error is largest.

## 1.3 Why autodiff, not finite differences

```
finite differences  numeric = [f(w+eps) - f(w-eps)] / (2 eps)
                    -> O(#params) forward passes. For 10^9 params: hopeless.
reverse-mode AD     one forward (build the graph) + one backward (walk it)
                    -> gradients w.r.t. ALL params in O(1) extra passes
```
Reverse mode is cheap when there are many inputs and ONE output (a scalar loss) — exactly the shape of ML training. Forward mode is the opposite and is used for Jacobian-vector products.

Central differences are still the gold standard for **debugging** a hand-written backward:
```
relative error = |analytic - numeric| / max(|analytic| + |numeric|, 1e-8)
< 1e-5 passes.  Freeze all randomness (dropout masks, VAE noise) first.
```

## 1.4 Gradient accumulation
A node used in several places receives gradient from **every** path (multivariate chain rule), so backward passes always `+=` into `.grad` rather than assigning. This is also why you must zero gradients between steps — and why "gradient accumulation" over several micro-batches simulates a large batch on small memory.

## 1.5 Computational graph memory
Activations from the forward pass must be kept to compute the backward pass: memory is O(depth × batch × activation size), usually far more than the parameters themselves. **Gradient checkpointing** trades compute for memory — store only every k-th activation and recompute the rest during backward, roughly √L memory for ~1.3× compute.

---

# PART 2 — ACTIVATIONS, INITIALIZATION, NORMALIZATION

## 2.1 Why non-linearity
A stack of linear layers collapses: `W₃(W₂(W₁x)) = (W₃W₂W₁)x`. Depth buys nothing without a non-linearity between layers.

## 2.2 Choosing an activation
```
sigmoid   output layer for binary only. Saturates both ends, not zero-centred
          (all gradients in a layer share a sign -> zig-zag updates).
tanh      zero-centred, still saturates. Used inside LSTM/GRU.
ReLU      the default for CNNs. Fast, sparse, no positive saturation.
          DYING RELU: a unit whose pre-activation is negative for every input
          gets zero gradient forever. Causes: too-high LR pushing a large
          negative bias, or bad init. Fix: LeakyReLU/ELU, lower LR, better init.
GELU      transformer default; smooth, small negative lobe.
Swish     x*sigmoid(x); often marginally better than ReLU in deep nets.
softmax   output only — it is a normalizer over classes, not a unit activation.
```

## 2.3 Initialization — the variance argument
Forward variance through a layer: `Var(a_l) ≈ fan_in · Var(W) · Var(a_(l-1))`. If `fan_in·Var(W) ≠ 1`, activations shrink or explode **geometrically with depth**, and the backward pass inherits the same factor.

```
Xavier   Var(W) = 2/(fan_in + fan_out)     balances forward AND backward variance
He       Var(W) = 2/fan_in                 the extra 2 compensates for ReLU
                                           zeroing half the units
```
**Zero init is fatal** — every unit in a layer computes the same function and receives an identical gradient, so symmetry is never broken. (Biases at zero are fine; weights are not.)

Measured in the companion code: with He init, activation std across 8 ReLU layers stays ≈0.8; with a small constant std it collapses 0.096 → 0.011 → 0.0012 → ~0. That is the vanishing signal problem, before training even begins.

## 2.4 BatchNorm
```
mu_B, var_B per feature over the minibatch
xhat = (x - mu_B)/sqrt(var_B + eps)
y    = gamma*xhat + beta          (learnable, so the net can undo normalization)

Train:     batch statistics
Inference: running averages accumulated during training
```
Backward has three terms, because `mu` and `var` each depend on **every** x in the batch:
```
dL/dx_i = (1/(N*sigma)) [ N*dxhat_i - sum_k dxhat_k - xhat_i * sum_k dxhat_k xhat_k ]
```
Benefits: allows higher learning rates, smooths the loss landscape, mild regularization from batch noise. Costs: unreliable with small batches, couples examples in a batch, awkward for RNNs and variable-length sequences, and train/inference behaviour differs (forgetting `.eval()` is a classic production bug).

**Interview:** *Do you need a bias in a layer followed by BatchNorm?* No — BN subtracts the mean, which cancels any constant bias; `beta` replaces it.

## 2.5 LayerNorm and friends
```
LayerNorm   normalize over the FEATURES of each sample  -> batch-independent
GroupNorm   split channels into G groups, normalize within each -> small-batch vision
InstanceNorm per-sample, per-channel -> style transfer
RMSNorm     drop the mean subtraction, divide by RMS only -> cheaper, used in LLaMA
```
**Why transformers use LayerNorm:** no batch dependence, so it works with batch size 1, variable sequence lengths, and at inference with no running statistics.

**Pre-norm vs post-norm:**
```
post-norm (2017)  x = LayerNorm(x + Sublayer(x))     needs warmup, can diverge deep
pre-norm (modern) x = x + Sublayer(LayerNorm(x))     clean identity path, trains deep
```

---

# PART 3 — OPTIMIZATION

## 3.1 The optimizer family
```
SGD         w -= lr * g
Momentum    v = mu*v + g ;  w -= lr*v
            damps oscillation across a ravine; effective speed-up ~ 1/(1-mu)
Nesterov    evaluate the gradient at the look-ahead point w - lr*mu*v
AdaGrad     G += g^2 ;             w -= lr*g/(sqrt(G)+eps)
            per-parameter LR, great for sparse features, but G only GROWS
            so the effective LR decays to zero and learning stalls
RMSProp     G = rho*G + (1-rho)g^2  -> exponential average, never stalls
Adam        m = b1 m + (1-b1) g          1st moment: direction
            v = b2 v + (1-b2) g^2        2nd moment: per-parameter scale
            mhat = m/(1-b1^t), vhat = v/(1-b2^t)      BIAS CORRECTION
            w -= lr * mhat/(sqrt(vhat)+eps)
AdamW       decouple weight decay: w -= lr*wd*w, applied OUTSIDE the adaptive
            denominator. This is the correct L2 and the transformer default.
Lion, Shampoo, Sophia, Muon — newer, second-order-ish or sign-based
```

**Why bias correction:** `m` and `v` start at 0, so early estimates are biased toward zero. Dividing by `(1−β^t)` makes them unbiased; without it the first ~1/(1−β) steps are far too small.

**Why AdamW ≠ Adam+L2:** folding `wd·w` into `g` means the adaptive denominator `sqrt(vhat)` rescales the decay differently for every parameter — parameters with large gradients get almost no decay. Decoupling restores uniform shrinkage.

**Interview:** *SGD or Adam?* Adam converges faster and needs less tuning, and is essential for transformers and sparse gradients. Well-tuned SGD+momentum still generalizes slightly better on vision benchmarks. A common recipe: Adam early, SGD for the final epochs.

## 3.2 Learning rate — the single most important hyperparameter
```
too high  loss diverges or oscillates, NaNs
too low   training crawls, may stall in a bad region
Find it   LR range test: sweep exponentially and plot loss vs LR; pick just
          below the minimum of the descending region.
```
Schedules:
```
step         lr * gamma^floor(t/s)
exponential  lr * e^(-kt)
cosine       lr_min + 0.5(lr-lr_min)(1+cos(pi t/T))       the modern default
warmup       linear ramp over the first W steps, THEN the schedule
one-cycle    ramp up then down; enables very high peak LRs
```
**Why warmup for transformers:** Adam's second-moment estimate is unreliable in the first steps (tiny `v` ⇒ enormous effective step), which can permanently destabilize a deep stack in the first few hundred iterations.

**LR and batch size:** larger batches give lower-variance gradients, so you can raise the LR. Linear scaling rule: multiply LR by k when you multiply batch size by k (with warmup). Beyond a critical batch size the returns vanish.

## 3.3 Exploding gradients and clipping
```
total_norm = sqrt(sum over all params of g^2)
if total_norm > max_norm:  g *= max_norm/total_norm
```
Scaling all gradients by one factor preserves the **direction** and only shortens the step; per-element clipping distorts the direction. Standard in RNNs and transformers (typ. max_norm = 1.0).

Clipping fixes **explosion**. It cannot fix **vanishing** — that needs architecture (gates, residuals, normalization).

## 3.4 The loss landscape
- High-dimensional non-convex losses are dominated by **saddle points**, not bad local minima; most local minima have similar loss.
- **Flat minima** are believed to generalize better than sharp ones; large-batch training tends toward sharper minima, which is one story for why small-batch SGD generalizes well.
- **Double descent**: past the interpolation threshold (params ≈ n), test error falls again — modern over-parameterized nets live on the right of the classical U-curve.
- **Lottery ticket hypothesis**: a dense net contains a sparse subnetwork that, trained from the same init, matches the full net's accuracy.

---

# PART 4 — REGULARIZATION AND GENERALIZATION

## 4.1 The toolbox
```
L2 / weight decay    penalize ||w||^2; AdamW decouples it
L1                   sparsity; rarely used in DL (unstructured sparsity
                     doesn't speed up dense hardware)
Dropout              zero units w.p. p; INVERTED dropout scales survivors by
                     1/(1-p) at train time so inference needs no change
Early stopping       provably equivalent to L2 for linear models under GD
Data augmentation    the most effective regularizer when applicable
Label smoothing      y' = (1-e)*onehot + e/C; caps confidence, improves calibration
Batch/Layer norm     partly regularizing (batch noise)
Stochastic depth     randomly skip whole residual blocks
Mixup / CutMix       train on convex combinations of inputs AND labels
Ensembling / SWA     average weights or predictions
Gradient noise       inject small Gaussian noise into gradients
```

## 4.2 Dropout in depth
```
train:  mask ~ Bernoulli(1-p), a = a * mask / (1-p)
eval:   identity
```
Interpretations: (a) an exponential ensemble of 2^n thinned subnetworks, approximately averaged at test time by the full network; (b) prevents **co-adaptation**, forcing units to be individually useful; (c) a form of noise injection ≈ adaptive L2.

Practical notes: typical p = 0.1–0.3 for transformers, 0.5 for old-style dense layers. Rarely used on convolutional layers (they are already parameter-efficient; use BatchNorm or DropBlock). Dropout + BatchNorm together can hurt — dropout shifts the variance BN estimated, so if you use both, put dropout **after** BN.

## 4.3 Data augmentation
```
vision    flip, crop, rotate, colour jitter, RandAugment, mixup, cutmix, cutout
audio     SpecAugment (time/frequency masking), noise, speed perturbation
text      synonym swap, back-translation, token dropout, EDA
general   adding input noise is provably ~ Tikhonov (L2) regularization
```
Augmentation encodes **invariances you know the task has**. Flipping is right for natural images and wrong for digit recognition (6 vs 9) and text.

## 4.4 Diagnosing with curves
```
train high, val high        underfit  -> bigger model, train longer, less reg
train low, val high         overfit   -> more data, augmentation, more reg
train and val still falling            -> train longer
val loss up, val ACCURACY up           -> the model is becoming over-confident
                                          on a few examples; consider label
                                          smoothing or temperature calibration
train loss < val loss always           -> normal with dropout (it is off at eval)
```

---

# PART 5 — CONVOLUTIONAL NETWORKS

## 5.1 The convolution
```
y[n,co,oh,ow] = b[co] + sum_ci sum_kh sum_kw
                  x[n, ci, oh*S+kh-P, ow*S+kw-P] * W[co, ci, kh, kw]

output size   O = floor((W - K + 2P)/S) + 1
'same' pad    P = (K-1)/2 for stride 1
params        (C_in*K*K + 1) * C_out         <- INDEPENDENT of image size
FLOPs         ~ 2 * C_in*C_out*K^2 * H_out*W_out per image
```

## 5.2 The three inductive biases
1. **Local connectivity** — nearby pixels are correlated; distant ones usually aren't.
2. **Weight sharing** — an edge detector is useful everywhere, giving translation **equivariance** (shift the input, the feature map shifts identically). Pooling then adds a little translation **invariance**.
3. **Hierarchy** — stacking grows the receptive field: edges → textures → parts → objects.

Measured in the companion code: on 8×8 shapes at random positions with only 45 training images, a 435-parameter CNN reaches 0.667 test accuracy while a 1,635-parameter MLP gets 0.460. The MLP must relearn the same bar detector at every position; the CNN shares one kernel across all of them.

## 5.3 Backward passes
```
conv backward   dL/dW = correlation of input patches with output gradient
                dL/dx = 'full' convolution of the gradient with the FLIPPED kernel
                (im2col turns both into matmuls; the reverse is col2im, a
                 scatter-ADD, because overlapping receptive fields share pixels)
maxpool         route the gradient ONLY to the argmax; everything else gets 0
avgpool         split the gradient equally, 1/K^2 to each input
```

## 5.4 Design patterns
```
3x3 stacks      two 3x3 layers see 5x5 with 18C^2 params vs 25C^2 for one 5x5,
                and add an extra non-linearity  (the VGG argument)
1x1 conv        per-pixel linear map across channels: cheap channel mixing and
                bottlenecks (ResNet, Inception)
stride vs pool  strided conv is a learnable downsample; pooling is fixed
dilated conv    inflate the receptive field without extra params or downsampling
depthwise sep.  depthwise + pointwise = K^2*C + C*C_out instead of K^2*C*C_out,
                an ~8-9x reduction (MobileNet, Xception)
global avg pool replaces the giant flatten->dense head; makes the net size-agnostic
transposed conv learnable upsampling (segmentation, generators); watch for
                checkerboard artifacts -> prefer upsample + conv
```

## 5.5 Architecture lineage — know the ONE idea each contributed
```
LeNet-5 (1998)     convolution + pooling works
AlexNet (2012)     ReLU, dropout, GPUs, augmentation -> ImageNet breakthrough
VGG (2014)         depth via uniform 3x3 stacks
Inception (2014)   multi-scale parallel branches + 1x1 bottlenecks
ResNet (2015)      RESIDUAL connections -> 100+ layers trainable
DenseNet (2016)    concatenate all previous feature maps
MobileNet (2017)   depthwise separable convs for edge devices
SENet (2017)       channel attention (squeeze-and-excite)
EfficientNet(2019) compound scaling of depth/width/resolution
ViT (2020)         image as 16x16 patches into a transformer; beats CNNs with
                   enough data (it has weaker inductive bias, so it needs more)
ConvNeXt (2022)    a CNN modernized with transformer design choices, competitive again
```

**ResNet is the one to be able to justify:** for `y = x + F(x)`, `dy/dx = I + dF/dx`. The identity term guarantees gradient reaches early layers no matter how small `dF/dx` gets. Reframing: the block learns the **residual** `F(x) = H(x) − x`, and learning "change nothing" (F = 0) is easy — so adding layers can never make the network worse, which solves the degradation problem.

## 5.6 Vision tasks beyond classification
```
detection       two-stage (R-CNN family: propose then classify) vs one-stage
                (YOLO, SSD, RetinaNet with focal loss); anchors, NMS, IoU, mAP
segmentation    semantic (per-pixel class) vs instance (per-object mask);
                U-Net's skip connections restore spatial detail lost to pooling
keypoints       heatmap regression
metric learning triplet loss max(0, d(a,p) - d(a,n) + margin), contrastive,
                InfoNCE -> SimCLR/CLIP-style self-supervision
```

---

# PART 6 — SEQUENCE MODELS

## 6.1 Vanilla RNN and BPTT
```
h_t = tanh(W_x x_t + W_h h_(t-1) + b)

BPTT: unroll and apply the chain rule across time. The gradient from step T
back to step t contains

    prod_{k=t+1..T}  W_h^T diag(1 - h_k^2)

the SAME matrix repeated. Largest singular value < 1 -> vanishing.
Largest singular value > 1 -> exploding.
```
Measured in the companion code (30 steps, one backward pass): the vanilla RNN's gradient at t=0 is ~7×10⁵ times smaller than at t=29; the LSTM's is only ~160× smaller — roughly a 4,000× better-preserved gradient path. On a task where the label is set at step 0 and must survive 25 steps, the RNN scores 0.45 (chance) and the LSTM 1.00.

**Clipping fixes explosion, not vanishing.** Vanishing is an architecture problem.

## 6.2 LSTM
```
f_t = sigma(W_f [h_(t-1), x_t] + b_f)      forget gate
i_t = sigma(W_i [...])                      input gate
o_t = sigma(W_o [...])                      output gate
g_t = tanh (W_g [...])                      candidate
c_t = f_t * c_(t-1) + i_t * g_t             CELL STATE  <- the key line
h_t = o_t * tanh(c_t)

params = 4 * (in*h + h*h + h)
```
Along `c_t` the gradient is multiplied by `f_t` — a **learned** gate in (0,1) — instead of by a weight matrix and an activation derivative. When `f_t ≈ 1` the gradient passes essentially unchanged (the "constant error carousel"). Initialize the forget-gate bias to ~1 so the cell starts in remember-mode.

## 6.3 GRU
```
z_t = sigma(...)                       update gate (ties LSTM's forget+input)
r_t = sigma(...)                       reset gate
n_t = tanh(W_xn x + W_hn (r_t * h))    candidate
h_t = (1 - z_t) * h_(t-1) + z_t * n_t
```
~25% fewer parameters than an LSTM, usually comparable accuracy; LSTMs keep a small edge on very long sequences.

## 6.4 Sequence architectures
```
many-to-one     sentiment classification (use the last hidden state, or pool)
many-to-many    tagging, per-step outputs
one-to-many     image captioning
encoder-decoder translation; the fixed-size context vector was the BOTTLENECK
                that attention was invented to remove (Bahdanau 2014)
bidirectional   run forwards and backwards, concatenate — only valid when the
                whole sequence is available (not for autoregressive generation)
truncated BPTT  detach the hidden state every k steps to bound memory
teacher forcing feed the TRUE previous token during training; fast to train but
                creates exposure bias (at inference the model sees its own
                mistakes) -> scheduled sampling, or just use large-scale pretraining
```

---

# PART 7 — ATTENTION AND TRANSFORMERS

## 7.1 Scaled dot-product attention
```
Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V

Q (T_q, d_k)   queries  — "what am I looking for?"
K (T_k, d_k)   keys     — "what do I contain?"
V (T_k, d_v)   values   — "what do I contribute?"
```
Read it as a **differentiable, content-addressed dictionary lookup**: score every query against every key, softmax the scores into a distribution over positions, return that distribution's weighted average of the values.

**Why divide by √d_k** — the question that gets asked most. If q and k have i.i.d. unit-variance entries, `q·k` has variance `d_k`. Measured in the companion code: std(q·k) is 1.70 at d_k=4, 8.41 at d_k=64, 23.77 at d_k=512 — tracking √d_k. Logits with std 8 push softmax into a near one-hot regime where its Jacobian `y_i(δ_ij − y_j)` is ≈ 0 everywhere, so **gradients vanish**. Dividing by √d_k restores unit variance.

**Masking:** add `−1e9` before the softmax so `exp(·) ≈ 0`. A lower-triangular causal mask makes a decoder autoregressive — position t cannot see t+1 — which is what lets **all T positions be trained in parallel in one forward pass**. Padding masks do the same for variable-length batches.

## 7.2 Multi-head attention
```
head_i = Attention(X W_Q^i, X W_K^i, X W_V^i)       each of width d_model/h
MHA(X) = concat(head_1 ... head_h) W_O

params = 4 * d_model^2
```
Total compute matches one big head, but heads can **specialize** — syntax, coreference, positional offsets, copying. One head must average all relations into a single distribution; h heads keep them in separate subspaces.

```
self-attention   Q, K, V from the same sequence
cross-attention  Q from the decoder, K and V from the encoder (translation)
MQA / GQA        share K,V across heads -> much smaller KV cache at inference
```

## 7.3 Positional encoding
Attention is **permutation-invariant**: shuffle the tokens and the output shuffles identically. Without position information, "dog bites man" = "man bites dog".
```
sinusoidal  PE(pos,2i) = sin(pos/10000^(2i/d)), PE(pos,2i+1) = cos(...)
            extrapolates to unseen lengths; PE(pos+k) is a LINEAR function of
            PE(pos), so relative offsets are easy to express
learned     a trainable embedding per position; simple, no extrapolation
RoPE        rotate Q and K by a position-dependent angle -> relative positions
            appear naturally in the dot product. The LLM default.
ALiBi       add a linear distance penalty to attention scores; extrapolates well
```

## 7.4 The transformer block
```
PRE-NORM (modern):
    x = x + MHA(LayerNorm(x))
    x = x + FFN(LayerNorm(x))
FFN = Linear(d, 4d) -> GELU -> Linear(4d, d)
```
Parameter budget per block: attention `4d²`, FFN `8d²`, so **the FFN holds two-thirds of the parameters**. Intuition: attention MIXES information across tokens; the FFN transforms each token independently and acts as the per-token memory/feature store.

```
total params ~ vocab*d + 12 * n_layers * d^2
```

**Encoder-only** (BERT): bidirectional, masked-LM pretraining, for classification/NER/retrieval.
**Decoder-only** (GPT): causal, next-token prediction, for generation. Now dominant.
**Encoder-decoder** (T5, original transformer): for translation and seq2seq.

## 7.5 Complexity and the efficiency zoo
```
attention   O(T^2 * d) time and O(T^2) memory in sequence length
FFN         O(T * d^2)
-> attention dominates for long sequences, FFN for short ones
```
```
FlashAttention   exact, but IO-aware: tile the computation to stay in SRAM and
                 never materialize the T x T matrix -> big speed/memory win
sparse/local     sliding window, strided, BigBird — O(T sqrt(T)) or O(T)
linear attention Performer, Linformer — kernel or low-rank approximation
state space      Mamba/S4 — O(T) recurrence with a parallel scan
KV cache         at inference, cache past K,V so each new token costs O(T d)
                 instead of re-running the whole prefix
```

## 7.6 Why transformers beat RNNs
1. **Parallelism over time** during training (RNNs are inherently sequential in T).
2. **Constant path length** between any two positions — no gradient decay with distance.
3. **Scaling behaviour** — performance keeps improving predictably with parameters, data, and compute (scaling laws), which is not true of RNNs.

Trade-off: quadratic cost in sequence length, and weaker inductive bias, so transformers need more data than CNNs/RNNs at small scale.

## 7.7 Large language models — the practical layer
```
Tokenization   BPE / WordPiece / SentencePiece — subwords eliminate OOV and
               balance vocabulary size against sequence length
Pretraining    next-token prediction on a huge corpus; loss = cross-entropy;
               perplexity = exp(loss)
Scaling laws   loss falls as a power law in params/data/compute; Chinchilla
               showed most models were UNDER-trained — scale data with params
Fine-tuning    full FT, or PEFT: LoRA (W + BA with rank r << d, training ~0.1%
               of params), adapters, prefix tuning, QLoRA (4-bit base + LoRA)
Alignment      SFT on demonstrations -> reward model from human preferences ->
               PPO (RLHF), or DPO which optimizes the preference objective
               directly with no separate reward model or RL loop
Inference      temperature (logits/T), top-k, top-p/nucleus, beam search,
               repetition penalty; speculative decoding for speed
Context        RoPE scaling, position interpolation, sliding windows
RAG            retrieve relevant chunks and put them in the prompt; cheaper and
               more current than fine-tuning for knowledge injection
Known failure  hallucination, sycophancy, prompt injection, stale knowledge,
               quadratic cost, no guaranteed reasoning
```
**Interview:** *Fine-tune or RAG?* RAG for knowledge that changes or must be cited; fine-tuning for behaviour, format, and style. They compose — RAG for facts, fine-tuning for how to use them.

---

# PART 8 — GENERATIVE MODELS

## 8.1 Autoencoders
```
x -> encoder -> z (bottleneck) -> decoder -> xhat ,  loss = ||x - xhat||^2
```
A **linear** autoencoder trained with MSE spans the same subspace as PCA (its axes need not be orthogonal). Non-linearity is what buys more than PCA.
```
denoising    corrupt the input, reconstruct the clean version — prevents
             learning the identity function
sparse       L1 penalty on activations
masked       mask tokens/patches and reconstruct — the objective behind BERT and MAE
```

## 8.2 VAE
```
ELBO:  log p(x) >= E_q(z|x)[log p(x|z)]  -  KL( q(z|x) || p(z) )
                   |--- reconstruction ---|   |--- regularizer ---|

Gaussian KL (closed form):
    KL = -0.5 * sum_j ( 1 + log sigma_j^2 - mu_j^2 - sigma_j^2 )

REPARAMETERIZATION TRICK:  z = mu + sigma * eps ,  eps ~ N(0, I)
```
You cannot backprop through "sample from N(μ,σ)" — sampling is not differentiable. Moving the randomness into a parameter-free `eps` makes μ and σ ordinary differentiable arithmetic. That single trick is what makes the model trainable by SGD.

The KL term is what makes the latent space **continuous and samplable**; a plain autoencoder's latent space has holes, so decoding a random z gives garbage. Trade-off: too much KL weight causes **posterior collapse** (the decoder ignores z) — which is exactly what β-VAE tunes deliberately for disentanglement.

## 8.3 GANs
```
min_G max_D  E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```
A two-player game: D learns to tell real from fake, G learns to fool D. At the optimum D ≡ 1/2 and the objective reduces to minimizing the **Jensen-Shannon divergence** between the real and generated distributions.
```
failure modes  mode collapse (G produces one output), non-convergence
               (the game oscillates), vanishing D gradient when D wins too fast
fixes          non-saturating loss, WGAN + gradient penalty (Wasserstein
               distance gives usable gradients everywhere), spectral norm,
               two-timescale updates, progressive growing, StyleGAN tricks
```

## 8.4 Diffusion models
```
forward   q(x_t | x_(t-1)) = N(sqrt(1-b_t) x_(t-1), b_t I)   gradually add noise
          closed form:  x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps
reverse   train a network eps_theta(x_t, t) to PREDICT the noise added
loss      simple MSE:  E || eps - eps_theta(x_t, t) ||^2
sample    iteratively denoise from pure noise; DDIM makes it deterministic
          and far fewer steps
guidance  classifier-free: train with and without conditioning, then
          extrapolate  eps = eps_uncond + w(eps_cond - eps_uncond)
```
Why diffusion overtook GANs for images: a **stable** regression objective (no adversarial game, no mode collapse), better mode coverage, and easy conditioning. Cost: sampling needs many network evaluations.

## 8.5 The generative landscape compared
| Family | Objective | Samples | Likelihood | Weakness |
|---|---|---|---|---|
| Autoregressive | exact NLL | high quality | exact | slow sequential sampling |
| VAE | ELBO (lower bound) | blurry | approximate | posterior collapse |
| GAN | adversarial | sharp | none | unstable, mode collapse |
| Diffusion | denoising MSE | state of the art | approximate | many sampling steps |
| Flows | exact NLL | good | exact | architectural constraints (invertibility) |

---

# PART 9 — REPRESENTATION LEARNING & MULTIMODAL

## 9.1 Transfer learning
```
feature extraction  freeze the backbone, train a new head      small data
fine-tuning         unfreeze some/all layers, small LR          more data
discriminative LRs  lower LR for early layers (generic features), higher for late
gradual unfreezing  unfreeze from the top down over epochs
catastrophic forgetting  fine-tuning erases pretrained knowledge -> low LR,
                    few epochs, LoRA, or replay/EWC
```
Early layers learn generic features (edges, Gabor filters, syntax) and transfer broadly; late layers are task-specific.

## 9.2 Self-supervised learning
```
contrastive   InfoNCE: pull augmented views of the same image together, push
              other images apart. SimCLR (big batches), MoCo (momentum queue).
non-contrastive  BYOL, SimSiam — no negatives, avoid collapse via a predictor
              head and a stop-gradient. Barlow Twins uses a decorrelation loss.
masked        BERT (mask tokens), MAE (mask 75% of image patches), data2vec
next-token    the GPT objective — the most scalable pretext task known
```
The point: labels are expensive, raw data is not. Pretext tasks manufacture supervision from the data's own structure.

## 9.3 Embeddings and retrieval
```
similarity   cosine for normalized vectors; dot product when magnitude matters
losses       triplet max(0, d(a,p) - d(a,n) + m), InfoNCE, ArcFace
hard mining  random negatives become uninformative quickly; mine semi-hard ones
ANN index    HNSW (graph, best recall/latency), IVF-PQ (quantized, memory-light),
             ScaNN. Exact search is O(n d) and fine up to ~10^5.
```

## 9.4 Multimodal
```
CLIP        two encoders (image, text), contrastive over a batch with a learned
            temperature -> shared embedding space, zero-shot classification by
            comparing an image against text prompts
ViT         image -> 16x16 patches -> linear projection -> transformer. Weaker
            inductive bias than a CNN, so it needs more data (or distillation)
Whisper     encoder-decoder on log-mel spectrograms
VLMs        project image patch embeddings into the LLM's token space
```

---

# PART 10 — EFFICIENCY AND PRODUCTION

## 10.1 Making models smaller and faster
```
quantization    FP32 -> FP16/BF16 (training), INT8/INT4 (inference).
                PTQ (post-training, needs a calibration set) vs QAT (simulate
                quantization during training, better accuracy, costs a retrain).
                GPTQ/AWQ for LLM weight-only quantization.
pruning         unstructured (sparse weights; needs sparse kernels to pay off)
                vs structured (drop whole channels/heads; always pays off)
distillation    train a student on the teacher's SOFT targets with temperature T;
                soft labels carry 'dark knowledge' — the relative probabilities
                of the wrong classes. Loss = a*CE(hard) + (1-a)*T^2*KL(soft).
low-rank        LoRA and friends: W + BA with rank r << d
early exit      classify at an intermediate layer when confident
```

## 10.2 Training at scale
```
data parallel     replicate the model, split the batch, all-reduce the gradients
model parallel    split layers across devices (pipeline) or within a layer (tensor)
ZeRO / FSDP       shard optimizer state, gradients, and parameters across devices
mixed precision   BF16/FP16 compute with an FP32 master copy; loss scaling for FP16
                  to keep small gradients from underflowing
grad accumulation simulate a big batch on small memory
grad checkpointing recompute activations in backward: ~sqrt(L) memory for ~1.3x compute
```
Memory for training ≈ params + gradients + optimizer state + activations. With Adam in FP32 that is roughly **16 bytes per parameter** before activations (4 weights + 4 grads + 8 for m and v).

## 10.3 Deployment and monitoring
```
export        ONNX, TorchScript, TensorRT; fuse ops, static shapes
serving       batching (dynamic/continuous), caching, KV cache for LLMs
monitor       latency p50/p99, throughput, error rate; input drift (PSI, KS),
              prediction drift, and delayed ground-truth metrics
rollout       shadow -> canary -> A/B; always keep a rollback path
reproducibility  seed everything, pin versions, version data AND model, log configs
```

## 10.4 The debugging checklist (interviewers love this)
```
1. Overfit ONE batch first. If you can't drive the loss to ~0, there is a bug —
   stop tuning hyperparameters and find it.
2. Check the data: shapes, ranges, label alignment, class balance. Visualize
   actual samples coming out of the loader.
3. Check the loss at init: for C balanced classes it should be ~ln(C)
   (ln 10 = 2.30). If not, the head or the loss is wrong.
4. Verify gradients flow: print per-layer gradient norms. All zeros = a broken
   graph, a detach, or a frozen parameter.
5. Turn OFF regularization and augmentation while debugging.
6. LR range test before anything else.
7. Confirm train/eval mode switching (BatchNorm, Dropout).
8. Seed everything; compare against a trivial baseline.
```

---

# PART 11 — RAPID-FIRE Q&A

**Q: Why does depth help?** Composition. Deep nets represent some functions exponentially more compactly than shallow ones, and they learn a feature hierarchy instead of requiring hand-engineered features.

**Q: Why can't you initialize weights to zero?** All units in a layer compute the same function and receive identical gradients, so symmetry is never broken and the layer collapses to one unit.

**Q: Vanishing vs exploding gradients?** Both come from repeated multiplication across depth/time. Vanishing → early layers stop learning (fix: ReLU, residuals, normalization, gates). Exploding → NaNs and spikes (fix: gradient clipping, better init, lower LR).

**Q: BatchNorm vs LayerNorm?** BN normalizes each feature across the batch (batch-dependent, needs running stats, weak with small batches). LN normalizes each sample across features (batch-independent, works with variable lengths — hence transformers).

**Q: Why is the FFN in a transformer 4× wide?** It holds most of the model's parameters and capacity; attention mixes tokens while the FFN transforms each token. The 4× ratio is empirical but has held up remarkably well.

**Q: Why √d_k in attention?** The dot product's variance grows with d_k; without scaling, softmax saturates and its gradient vanishes.

**Q: What does the residual connection actually do?** `dy/dx = I + dF/dx` — the identity term gives gradients an unimpeded path to early layers, and the block only has to learn a residual, so extra depth can't hurt.

**Q: Why is Adam popular but SGD sometimes better?** Adam adapts per-parameter step sizes so it converges fast with little tuning, which is essential for sparse gradients and transformers. Well-tuned SGD+momentum often finds flatter minima that generalize slightly better on vision.

**Q: Dropout at inference?** Off. Inverted dropout already scaled activations by 1/(1−p) during training, so no test-time change is needed. (Keeping it on and averaging many passes = MC dropout, a cheap uncertainty estimate.)

**Q: What is teacher forcing and what's wrong with it?** Feeding the true previous token during training. Fast and stable, but creates exposure bias: at inference the model conditions on its own (possibly wrong) outputs, a distribution it never saw.

**Q: Why do transformers need positional encoding?** Attention is permutation-invariant; without positions, word order carries no information.

**Q: Your loss is NaN at step 300. Debug it.** Check for LR too high and exploding gradients (print the global grad norm), `log(0)` or division by zero in a custom loss, FP16 overflow (use loss scaling or BF16), bad data (inf/NaN inputs), and an unstable softmax/normalization without epsilon.

**Q: Training accuracy 99%, validation 60%. What now?** Overfitting: more data, stronger augmentation, dropout/weight decay, a smaller model, early stopping. First though, verify there's no leakage or a broken validation split — a 39-point gap is large enough to be suspicious.

**Q: How do you choose batch size?** As large as fits, subject to generalization. Larger → lower-variance gradients, better hardware utilization, but sharper minima and diminishing returns past a critical size. Scale LR roughly linearly with batch size, with warmup.

**Q: What is label smoothing for?** It replaces the one-hot target with a slightly softened one, making the optimal logit gap finite. Result: better calibration, less over-confidence, usually a small accuracy gain.

**Q: Why is cross-entropy preferred to accuracy as a loss?** Accuracy is piecewise constant, so its gradient is zero almost everywhere. Cross-entropy is a smooth, convex-in-the-logits surrogate.

**Q: What's the difference between parameters and FLOPs?** Parameters measure memory; FLOPs measure compute. A conv layer has few parameters but many FLOPs (they're reused across positions); an embedding table is the opposite.

**Q: How would you handle a 100:1 class imbalance in a neural net?** Class-weighted loss or focal loss, threshold tuning on the validation set, oversampling the minority within each batch, and evaluation by PR-AUC rather than accuracy.

**Q: When would you NOT use deep learning?** Small tabular datasets (gradient boosting wins), hard interpretability or regulatory requirements, very tight latency/compute budgets, or when a simple baseline already meets the target.

---

# PART 12 — COVERAGE INDEX

Every concept above, and where the runnable implementation lives in `dl_from_scratch.py`.

| Concept | Implementation | Verified by |
|---|---|---|
| Scalar backprop, graph, accumulation | `Value` | analytic comparison |
| Reverse-mode AD over matrices | `Tensor` | — |
| matmul backward (`dA = dC B^T`) | `Tensor.matmul` | gradient check |
| Softmax Jacobian | `Tensor.softmax` | gradient check |
| Fused softmax + CE (`δ = p − y`) | `cross_entropy` | gradient check |
| Label smoothing | `label_smoothing_ce` | — |
| ReLU / LeakyReLU / sigmoid / tanh / GELU / SiLU | `Tensor` methods | gradient check |
| He / Xavier / LeCun init | `init_weights` | depth experiment (demo 5) |
| Linear, Embedding, Sequential | `Linear`, `Embedding` | gradient check |
| Inverted dropout | `Dropout` | overfit experiment (demo 4) |
| LayerNorm + 3-term backward | `LayerNorm` | gradient check |
| BatchNorm + running stats | `BatchNorm1d` | gradient check |
| SGD / momentum / Nesterov | `SGD` | demo 3 |
| RMSProp | `RMSProp` | demo 3 |
| Adam + bias correction, AdamW | `Adam` | demo 3 |
| Gradient clipping by global norm | `clip_grad_norm` | demo 11 |
| LR schedules + warmup | `lr_schedule` | demo 11 |
| Early stopping with weight restore | `EarlyStopping` | demo 11 |
| im2col / col2im | `im2col` | gradient check |
| Conv2D, parameter counting | `Conv2D` | gradient check, demo 6 |
| MaxPool (argmax routing), GlobalAvgPool | `MaxPool2D`, `GlobalAvgPool` | gradient check |
| RNN cell + BPTT | `RNNCell` | gradient check, demo 7 |
| LSTM (cell state, forget-bias init) | `LSTMCell` | gradient check, demo 7 |
| GRU | `GRUCell` | gradient check, demo 7 |
| Vanishing gradients, measured | `demo_rnn_memory` | per-timestep grad norms |
| Scaled dot-product attention | `scaled_dot_product_attention` | demo 8 |
| Causal masking | `causal_mask` | demo 8 |
| Multi-head attention | `MultiHeadAttention` | gradient check |
| Sinusoidal positional encoding | `PositionalEncoding` | — |
| Pre-norm transformer block | `TransformerBlock` | gradient check |
| Decoder-only LM + weight tying | `MiniGPT` | gradient check, demo 9 |
| Temperature / top-k sampling | `MiniGPT.generate` | demo 9 |
| Residual connection | `ResidualBlock` | — |
| Autoencoder | `Autoencoder` | demo 10 |
| VAE + reparameterization + KL | `VAE` | gradient check, demo 10 |
| Numerical gradient checking | `gradient_check` | all of the above |

**Covered in these notes, not implemented** (explain, don't code): ResNet/Inception/EfficientNet/ViT specifics, object detection and segmentation heads, GANs and diffusion training loops, FlashAttention/RoPE/GQA, LoRA and quantization, RLHF/DPO, distributed training (ZeRO/FSDP), CLIP and multimodal encoders.

---

# PART 13 — 2-WEEK PLAN

```
Day 1   Backprop by hand: derive delta_L = a - y for softmax+CE. Run demo 0-1.
Day 2   Activations, init variance argument, normalization backward passes.
Day 3   Optimizers: derive Adam, explain bias correction and AdamW. Demo 3.
Day 4   Regularization + the symptom->cause table. Demo 4.
Day 5   CNNs: output size, param count, receptive field, im2col. Demo 6.
Day 6   CNN architectures — one sentence per model, ResNet in depth.
Day 7   RNN/LSTM/GRU equations from memory. Demo 7.
Day 8   Attention: derive it, explain sqrt(d_k), masking, multi-head. Demo 8.
Day 9   Transformer block, param counting, pre/post-norm, complexity. Demo 9.
Day 10  LLMs: tokenization, scaling laws, LoRA, RLHF/DPO, RAG, decoding.
Day 11  Generative: VAE ELBO + reparameterization, GAN objective, diffusion.
Day 12  Efficiency: quantization, distillation, pruning, distributed training.
Day 13  Debugging checklist + 3 system-design questions out loud.
Day 14  Rapid-fire Q&A. Run the whole file. Explain each component in 2 minutes.
```

**Self-test — from memory, can you:**
1. Derive backprop for a 2-layer net and show `δ_L = a − y`?
2. Explain why He init has a factor of 2 and Xavier doesn't?
3. Write the LayerNorm backward and say why it has three terms?
4. Write Adam including bias correction, and explain AdamW's difference?
5. Compute the output size and parameter count of a conv layer?
6. Write the LSTM equations and point to the line that solves vanishing gradients?
7. Write attention and justify every symbol including √d_k?
8. Count the parameters of a transformer block?
9. Derive the VAE ELBO and explain the reparameterization trick?
10. Debug a NaN loss, a flat loss, and a train/val gap — three different playbooks?

---
*Companion code: `dl_from_scratch.py`. Prior notes: `ML_Interview_Notes.md`, `ml_from_scratch.py`, `ml_from_scratch_part2.py`.*
