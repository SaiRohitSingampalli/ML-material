"""
================================================================================
 REINFORCEMENT LEARNING FROM SCRATCH — pure Python, zero libraries
================================================================================
 Only `math` and `random`. No NumPy, no Gym, no PyTorch. Environments, value
 functions, policy gradients and the neural networks are all built here.

 CONTENTS
 --------
 0. Infrastructure ....... MLP with manual backprop, Adam, replay buffer,
                           schedules, running normalizer
 1. Environments ......... GridWorld, CliffWalking, WindyGrid, FrozenLake,
                           CartPole (real physics), Chain, MultiArmedBandit
 2. Dynamic programming .. policy evaluation, policy iteration, value iteration
 3. Prediction ........... Monte Carlo (first/every visit), TD(0), n-step TD,
                           TD(lambda) with eligibility traces
 4. Tabular control ...... MC control, SARSA, Expected SARSA, Q-learning,
                           Double Q-learning, Dyna-Q (model-based planning)
 5. Approximation ........ tile coding, linear semi-gradient SARSA, DQN
                           (replay buffer + target network), Double DQN
 6. Policy gradient ...... REINFORCE, REINFORCE + baseline, Actor-Critic,
                           A2C with GAE, PPO (clipped surrogate)
 7. Bandits .............. epsilon-greedy, UCB1, Thompson, gradient bandit
 8. Demos ................ every algorithm trained and compared

 THE ONE IDEA BEHIND ALL OF IT
 -----------------------------
 Supervised learning is given (x, y). RL is given only a scalar reward that may
 arrive many steps after the action that caused it. Everything below is a way of
 solving that credit-assignment problem:
     - DP            : we know the dynamics; solve the Bellman equations exactly
     - Monte Carlo   : wait for the episode to end, use the actual return
     - TD            : BOOTSTRAP — update a guess toward a better guess
     - Policy grad   : differentiate the expected return w.r.t. the policy directly
================================================================================
"""

import math
import random

random.seed(0)
EPS = 1e-12


# =============================================================================
# 0. INFRASTRUCTURE
# =============================================================================
class MLP:
    """
    A small multilayer perceptron with a hand-written backward pass.

    The backward entry point takes dL/d(output) rather than a loss, because RL
    needs several different "losses" on the same machinery:

        DQN            dL/dQ   = 2(Q(s,a) - target) for the taken action, else 0
        REINFORCE      dL/dlogits = (pi - onehot(a)) * (-advantage)
        value critic   dL/dV   = 2(V(s) - target)

    All three reduce to "push this output up or down by this much", which is the
    only interface a policy/value network needs.

    Layers: z = a_prev W + b ; a = f(z). Backward is the usual
        dL/dW = a_prev^T delta ,  dL/db = sum(delta) ,  delta_prev = delta W^T * f'
    """

    def __init__(self, sizes, activation="tanh", out_activation=None, seed=0):
        rng = random.Random(seed)
        self.sizes, self.activation, self.out_act = sizes, activation, out_activation
        self.W, self.b = [], []
        for i in range(len(sizes) - 1):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            # He for relu, Xavier otherwise
            std = math.sqrt(2.0 / fan_in) if activation == "relu" \
                else math.sqrt(1.0 / fan_in)
            self.W.append([[rng.gauss(0, std) for _ in range(fan_out)]
                           for _ in range(fan_in)])
            self.b.append([0.0] * fan_out)

    # ---- activations
    def _f(self, z):
        if self.activation == "relu":
            return [v if v > 0 else 0.0 for v in z]
        return [math.tanh(v) for v in z]

    def _df(self, a):
        if self.activation == "relu":
            return [1.0 if v > 0 else 0.0 for v in a]
        return [1 - v * v for v in a]

    def forward(self, X):
        """X: list of input vectors. Caches activations for the backward pass."""
        self.cache = [X]
        a = X
        L = len(self.W)
        for l in range(L):
            W, b = self.W[l], self.b[l]
            z = [[sum(a[n][i] * W[i][j] for i in range(len(W))) + b[j]
                  for j in range(len(b))] for n in range(len(a))]
            a = z if l == L - 1 else [self._f(row) for row in z]
            self.cache.append(a)
        return a

    def backward(self, dout):
        """dout: dL/d(output), same shape as the output. Returns grads."""
        L = len(self.W)
        gW = [[[0.0] * len(self.W[l][0]) for _ in range(len(self.W[l]))]
              for l in range(L)]
        gb = [[0.0] * len(self.b[l]) for l in range(L)]
        delta = dout
        for l in range(L - 1, -1, -1):
            prev = self.cache[l]
            for n in range(len(prev)):
                for j in range(len(gb[l])):
                    d = delta[n][j]
                    if d == 0.0:
                        continue
                    gb[l][j] += d
                    for i in range(len(prev[n])):
                        gW[l][i][j] += prev[n][i] * d
            if l > 0:
                back = [[sum(self.W[l][i][j] * delta[n][j]
                             for j in range(len(gb[l])))
                         for i in range(len(self.W[l]))]
                        for n in range(len(prev))]
                df = [self._df(row) for row in self.cache[l]]
                delta = [[back[n][i] * df[n][i] for i in range(len(back[n]))]
                         for n in range(len(back))]
        return gW, gb

    def params(self):
        return self.W, self.b

    def n_params(self):
        return sum(len(W) * len(W[0]) + len(b) for W, b in zip(self.W, self.b))

    def copy_from(self, other, tau=1.0):
        """Hard (tau=1) or soft (Polyak) update of the target network."""
        for l in range(len(self.W)):
            for i in range(len(self.W[l])):
                for j in range(len(self.W[l][i])):
                    self.W[l][i][j] = tau * other.W[l][i][j] + \
                        (1 - tau) * self.W[l][i][j]
            for j in range(len(self.b[l])):
                self.b[l][j] = tau * other.b[l][j] + (1 - tau) * self.b[l][j]


class AdamMLP:
    """Adam for the MLP above (see the DL notes for the derivation)."""

    def __init__(self, net, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.net, self.lr, self.b1, self.b2, self.eps = net, lr, b1, b2, eps
        self.mW = [[[0.0] * len(w[0]) for _ in range(len(w))] for w in net.W]
        self.vW = [[[0.0] * len(w[0]) for _ in range(len(w))] for w in net.W]
        self.mb = [[0.0] * len(b) for b in net.b]
        self.vb = [[0.0] * len(b) for b in net.b]
        self.t = 0

    def step(self, gW, gb, clip=None):
        if clip:
            total = math.sqrt(sum(g * g for l in gW for r in l for g in r)
                              + sum(g * g for l in gb for g in l))
            if total > clip:
                s = clip / (total + 1e-8)
                gW = [[[g * s for g in r] for r in l] for l in gW]
                gb = [[g * s for g in l] for l in gb]
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        net = self.net
        for l in range(len(net.W)):
            for i in range(len(net.W[l])):
                for j in range(len(net.W[l][i])):
                    g = gW[l][i][j]
                    self.mW[l][i][j] = self.b1 * self.mW[l][i][j] + (1 - self.b1) * g
                    self.vW[l][i][j] = self.b2 * self.vW[l][i][j] + (1 - self.b2) * g * g
                    net.W[l][i][j] -= self.lr * (self.mW[l][i][j] / c1) / \
                        (math.sqrt(self.vW[l][i][j] / c2) + self.eps)
            for j in range(len(net.b[l])):
                g = gb[l][j]
                self.mb[l][j] = self.b1 * self.mb[l][j] + (1 - self.b1) * g
                self.vb[l][j] = self.b2 * self.vb[l][j] + (1 - self.b2) * g * g
                net.b[l][j] -= self.lr * (self.mb[l][j] / c1) / \
                    (math.sqrt(self.vb[l][j] / c2) + self.eps)


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


class ReplayBuffer:
    """
    EXPERIENCE REPLAY — a ring buffer of (s, a, r, s', done) transitions.

    Two problems it solves, both fatal without it:
      1. CORRELATION. Consecutive transitions are highly correlated, violating
         the i.i.d. assumption SGD relies on; sampling uniformly from a large
         buffer decorrelates the minibatch.
      2. SAMPLE EFFICIENCY. Each transition is reused many times instead of
         being seen once and discarded.

    It only works for OFF-POLICY algorithms: the stored data was generated by an
    older policy, and Q-learning's max operator does not care whose policy
    produced the transition. On-policy methods (SARSA, REINFORCE, PPO) cannot
    reuse stale data without importance corrections.
    """

    def __init__(self, capacity=10000, seed=0):
        self.capacity = capacity
        self.buf = []
        self.pos = 0
        self.rng = random.Random(seed)

    def push(self, s, a, r, s2, done):
        item = (s, a, r, s2, done)
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, n):
        return [self.buf[self.rng.randrange(len(self.buf))]
                for _ in range(min(n, len(self.buf)))]

    def __len__(self):
        return len(self.buf)


def linear_decay(step, start, end, steps):
    """Epsilon schedule: explore early, exploit late."""
    if step >= steps:
        return end
    return start + (end - start) * step / steps


class RunningNorm:
    """Welford's online mean/variance — used to normalize observations or
    advantages without storing the data."""

    def __init__(self, dim):
        self.n = 0
        self.mean = [0.0] * dim
        self.M2 = [0.0] * dim

    def update(self, x):
        self.n += 1
        for i, v in enumerate(x):
            d = v - self.mean[i]
            self.mean[i] += d / self.n
            self.M2[i] += d * (v - self.mean[i])

    def normalize(self, x):
        if self.n < 2:
            return list(x)
        return [(v - self.mean[i]) / (math.sqrt(self.M2[i] / self.n) + 1e-8)
                for i, v in enumerate(x)]


# =============================================================================
# 1. ENVIRONMENTS
# =============================================================================
class GridWorld:
    """
    A classic tabular MDP.

        S = every non-wall cell (plus a terminal absorbing state)
        A = {up, right, down, left}
        P = deterministic, or 'slippery' with probability `slip` of a side-step
        R = step_cost everywhere, goal_reward at the goal, hazard_cost in pits
        gamma discounts future reward

    Why the step cost matters: with gamma = 1 and zero step cost, EVERY policy
    that eventually reaches the goal has the same return, so nothing prefers a
    short path. A small negative per-step reward (or gamma < 1) is what makes
    "get there fast" the optimal behaviour.
    """

    def __init__(self, layout=None, slip=0.0, step_cost=-0.04,
                 goal_reward=1.0, hazard_cost=-1.0, gamma=0.95, seed=0):
        self.layout = layout or [
            "....G",
            ".##.H",
            ".....",
            "S...."]
        self.H, self.W = len(self.layout), len(self.layout[0])
        self.slip, self.step_cost = slip, step_cost
        self.goal_reward, self.hazard_cost, self.gamma = goal_reward, hazard_cost, gamma
        self.rng = random.Random(seed)
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]      # U R D L
        self.action_names = ["U", "R", "D", "L"]
        self.states = [(r, c) for r in range(self.H) for c in range(self.W)
                       if self.layout[r][c] != "#"]
        self.sidx = {s: i for i, s in enumerate(self.states)}
        self.nS, self.nA = len(self.states), 4
        self.start = next(s for s in self.states
                          if self.layout[s[0]][s[1]] == "S")

    def is_terminal(self, s):
        return self.layout[s[0]][s[1]] in "GH"

    def _move(self, s, a):
        dr, dc = self.actions[a]
        r, c = s[0] + dr, s[1] + dc
        if 0 <= r < self.H and 0 <= c < self.W and self.layout[r][c] != "#":
            return (r, c)
        return s                                     # bump into a wall: stay put

    def transitions(self, s, a):
        """Returns [(prob, next_state, reward, done)] — the MODEL, used by DP."""
        if self.is_terminal(s):
            return [(1.0, s, 0.0, True)]
        outcomes = {}
        cands = [(1 - self.slip, a)] if self.slip == 0 else \
                [(1 - self.slip, a), (self.slip / 2, (a - 1) % 4),
                 (self.slip / 2, (a + 1) % 4)]
        for p, act in cands:
            s2 = self._move(s, act)
            outcomes[s2] = outcomes.get(s2, 0.0) + p
        res = []
        for s2, p in outcomes.items():
            ch = self.layout[s2[0]][s2[1]]
            r = self.goal_reward if ch == "G" else \
                self.hazard_cost if ch == "H" else self.step_cost
            res.append((p, s2, r, ch in "GH"))
        return res

    def reset(self):
        self.s = self.start
        return self.s

    def step(self, a):
        outs = self.transitions(self.s, a)
        x = self.rng.random()
        acc = 0.0
        for p, s2, r, done in outs:
            acc += p
            if x <= acc:
                self.s = s2
                return s2, r, done
        p, s2, r, done = outs[-1]
        self.s = s2
        return s2, r, done

    def render_policy(self, policy):
        rows = []
        for r in range(self.H):
            row = ""
            for c in range(self.W):
                ch = self.layout[r][c]
                if ch == "#":
                    row += " # "
                elif ch == "G":
                    row += " G "
                elif ch == "H":
                    row += " H "
                else:
                    row += " " + self.action_names[policy[(r, c)]] + " "
            rows.append(row)
        return "\n".join(rows)

    def render_values(self, V):
        rows = []
        for r in range(self.H):
            row = ""
            for c in range(self.W):
                if self.layout[r][c] == "#":
                    row += "   ##  "
                else:
                    row += f"{V[(r, c)]:7.2f}"
            rows.append(row)
        return "\n".join(rows)


def cliff_world():
    """
    THE CLIFF-WALKING TASK (Sutton & Barto 6.6) — the canonical demonstration of
    the difference between on-policy and off-policy control.

    A row of cliff cells sits between the start and the goal. The OPTIMAL path
    hugs the cliff edge. Q-learning learns exactly that (it evaluates the greedy
    policy), but while still exploring with epsilon it occasionally falls in, so
    its ONLINE return is worse. SARSA learns the safer route one row up, because
    it evaluates the policy it is actually following, epsilon and all.
    """
    layout = ["......",
              "......",
              "......",
              "SHHHHG"]
    return GridWorld(layout, slip=0.0, step_cost=-1.0, goal_reward=0.0,
                     hazard_cost=-100.0, gamma=1.0)


def frozen_lake():
    """Slippery grid: the intended action succeeds only 80% of the time."""
    layout = ["S...",
              ".H.H",
              "...H",
              "H..G"]
    return GridWorld(layout, slip=0.2, step_cost=0.0, goal_reward=1.0,
                     hazard_cost=0.0, gamma=0.99)


class CartPole:
    """
    CARTPOLE — continuous state, discrete actions. Real physics, no Gym.

        state  = (x, x_dot, theta, theta_dot)
        action = push left (0) or right (1), force = +/- 10 N
        reward = +1 per step survived
        done   = |theta| > 12 degrees, |x| > 2.4, or 500 steps

    Equations of motion (Barto/Sutton, semi-implicit Euler integration):

        temp      = (F + m_p*l*theta_dot^2*sin(th)) / m_total
        theta_acc = (g*sin(th) - cos(th)*temp) /
                    ( l * (4/3 - m_p*cos(th)^2/m_total) )
        x_acc     = temp - m_p*l*theta_acc*cos(th)/m_total

    This is the standard first benchmark for function approximation: the state
    space is continuous, so a lookup table is impossible without discretization.
    """

    def __init__(self, seed=0, max_steps=500):
        self.g, self.m_c, self.m_p, self.l = 9.8, 1.0, 0.1, 0.5
        self.force, self.tau = 10.0, 0.02
        self.theta_thr, self.x_thr = 12 * math.pi / 180, 2.4
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.nA, self.obs_dim = 2, 4

    def reset(self):
        self.state = [self.rng.uniform(-0.05, 0.05) for _ in range(4)]
        self.steps = 0
        return list(self.state)

    def step(self, a):
        x, x_dot, th, th_dot = self.state
        F = self.force if a == 1 else -self.force
        ct, st = math.cos(th), math.sin(th)
        total = self.m_c + self.m_p
        temp = (F + self.m_p * self.l * th_dot ** 2 * st) / total
        th_acc = (self.g * st - ct * temp) / \
            (self.l * (4.0 / 3.0 - self.m_p * ct ** 2 / total))
        x_acc = temp - self.m_p * self.l * th_acc * ct / total
        x += self.tau * x_dot
        x_dot += self.tau * x_acc
        th += self.tau * th_dot
        th_dot += self.tau * th_acc
        self.state = [x, x_dot, th, th_dot]
        self.steps += 1
        done = (abs(th) > self.theta_thr or abs(x) > self.x_thr
                or self.steps >= self.max_steps)
        return list(self.state), 1.0, done


class ChainMDP:
    """
    A 1-D chain — the minimal environment for studying PREDICTION.
    States 0..n-1, start in the middle, +1 reward at the right end, 0 at the
    left. The true value of each state is its probability of reaching the right
    end under a random walk, i.e. V(i) = i/(n-1). Having the exact answer lets
    us MEASURE the error of MC vs TD vs n-step, which is the point.
    """

    def __init__(self, n=7, seed=0):
        self.n = n
        self.rng = random.Random(seed)
        self.nA = 2

    def true_values(self):
        return {i: i / (self.n - 1) for i in range(1, self.n - 1)}

    def reset(self):
        self.s = self.n // 2
        return self.s

    def step(self, a=None):
        move = 1 if (a if a is not None else self.rng.randrange(2)) == 1 else -1
        self.s += move
        if self.s == self.n - 1:
            return self.s, 1.0, True
        if self.s == 0:
            return self.s, 0.0, True
        return self.s, 0.0, False


class BernoulliBandit:
    """
    k arms, arm i pays 1 with probability p_i. No states — pure
    exploration/exploitation, the simplest possible RL problem.

    WARNING worth internalizing: give the environment a DIFFERENT seed from the
    agent. Two `random.Random(1)` objects emit the identical number sequence, so
    if the agent samples an action with one and the environment samples the
    reward with the other, action and reward become correlated and results are
    silently garbage. This bit the gradient-bandit implementation here until the
    seeds were separated.
    """

    def __init__(self, probs, seed=0):
        self.probs = probs
        self.k = len(probs)
        self.rng = random.Random(seed)
        self.best = max(probs)

    def pull(self, a):
        return 1.0 if self.rng.random() < self.probs[a] else 0.0


# =============================================================================
# 2. DYNAMIC PROGRAMMING — when the model P(s'|s,a) is KNOWN
# =============================================================================
def policy_evaluation(env, policy, theta=1e-8, max_iter=1000):
    """
    ITERATIVE POLICY EVALUATION — solve the Bellman EXPECTATION equation

        V^pi(s) = sum_a pi(a|s) sum_s' P(s'|s,a) [ R + gamma V^pi(s') ]

    Sweep the update until it stops changing. The Bellman operator is a
    gamma-CONTRACTION in the max norm:

        ||T V - T U||_inf <= gamma ||V - U||_inf

    so by the Banach fixed-point theorem it has a unique fixed point and this
    converges geometrically at rate gamma, from ANY initialization.
    """
    V = {s: 0.0 for s in env.states}
    for it in range(max_iter):
        delta = 0.0
        for s in env.states:
            if env.is_terminal(s):
                continue
            v = V[s]
            total = 0.0
            for a, pa in enumerate(policy[s]):
                if pa == 0:
                    continue
                total += pa * sum(p * (r + env.gamma * V[s2])
                                  for p, s2, r, _ in env.transitions(s, a))
            V[s] = total
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            return V, it + 1
    return V, max_iter


def policy_iteration(env, theta=1e-8):
    """
    POLICY ITERATION = evaluate, then act greedily, repeat.

        1. V <- V^pi                                   (policy evaluation)
        2. pi'(s) <- argmax_a sum_s' P[R + gamma V(s')] (policy improvement)

    POLICY IMPROVEMENT THEOREM: if Q^pi(s, pi'(s)) >= V^pi(s) for all s, then
    V^pi' >= V^pi everywhere. Greedy improvement satisfies this by construction,
    so each round is at least as good as the last. There are finitely many
    deterministic policies, so it terminates at the optimum — usually in very
    few iterations (each one is expensive, being a full evaluation).
    """
    policy = {s: [1.0 / env.nA] * env.nA for s in env.states}
    for it in range(200):
        V, _ = policy_evaluation(env, policy, theta)
        stable = True
        for s in env.states:
            if env.is_terminal(s):
                continue
            old = max(range(env.nA), key=lambda a: policy[s][a])
            qs = [sum(p * (r + env.gamma * V[s2])
                      for p, s2, r, _ in env.transitions(s, a))
                  for a in range(env.nA)]
            best = max(range(env.nA), key=lambda a: qs[a])
            policy[s] = [1.0 if a == best else 0.0 for a in range(env.nA)]
            if best != old:
                stable = False
        if stable:
            return policy, V, it + 1
    return policy, V, 200


def value_iteration(env, theta=1e-10, max_iter=2000):
    """
    VALUE ITERATION — collapse evaluation and improvement into ONE update, the
    Bellman OPTIMALITY equation:

        V*(s) = max_a sum_s' P(s'|s,a) [ R + gamma V*(s') ]

    Equivalent to policy iteration with exactly one sweep of evaluation per
    improvement. Cheaper per iteration, more iterations. Also a gamma-contraction,
    so ||V_k - V*||_inf <= gamma^k ||V_0 - V*||_inf — geometric convergence.

    Complexity per sweep: O(|S|^2 |A|). The curse of dimensionality: |S| grows
    exponentially with the number of state variables, which is exactly why we
    need sampling (Parts 3-4) and function approximation (Part 5).
    """
    V = {s: 0.0 for s in env.states}
    for it in range(max_iter):
        delta = 0.0
        for s in env.states:
            if env.is_terminal(s):
                continue
            v = V[s]
            V[s] = max(sum(p * (r + env.gamma * V[s2])
                           for p, s2, r, _ in env.transitions(s, a))
                       for a in range(env.nA))
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    policy = {}
    for s in env.states:
        if env.is_terminal(s):
            policy[s] = 0
            continue
        policy[s] = max(range(env.nA),
                        key=lambda a: sum(p * (r + env.gamma * V[s2])
                                          for p, s2, r, _ in env.transitions(s, a)))
    return policy, V, it + 1


# =============================================================================
# 3. PREDICTION — estimate V^pi from SAMPLES (no model)
# =============================================================================
def mc_prediction(env, episodes=500, first_visit=True, alpha=None, seed=0):
    """
    MONTE CARLO PREDICTION

        G_t = r_(t+1) + gamma r_(t+2) + ... (the ACTUAL return, no bootstrapping)
        V(s) <- average of the returns observed after visiting s

    FIRST-VISIT averages only the first occurrence of s in each episode (its
    samples are i.i.d., so it is unbiased with a clean variance analysis);
    EVERY-VISIT uses all occurrences (biased for finite data, but consistent).

    Properties: UNBIASED, but HIGH VARIANCE (the return depends on every random
    choice for the rest of the episode). Requires EPISODIC tasks — you cannot
    update until the episode terminates.
    """
    rng = random.Random(seed)
    V = {s: 0.0 for s in range(1, env.n - 1)}
    counts = {s: 0 for s in V}
    for _ in range(episodes):
        s = env.reset()
        traj = []
        done = False
        while not done:
            a = rng.randrange(2)
            s2, r, done = env.step(a)
            traj.append((s, r))
            s = s2
        G = 0.0
        seen = set()
        for t in range(len(traj) - 1, -1, -1):
            st, r = traj[t]
            G = r + 1.0 * G                       # gamma = 1 on the chain
            if first_visit and st in [x[0] for x in traj[:t]]:
                continue
            if st in V:
                counts[st] += 1
                step = alpha if alpha else 1.0 / counts[st]
                V[st] += step * (G - V[st])
    return V


def td0_prediction(env, episodes=500, alpha=0.1, seed=0):
    """
    TD(0) — the central idea in RL: BOOTSTRAPPING.

        V(s_t) <- V(s_t) + alpha [ r + gamma V(s_(t+1)) - V(s_t) ]
                                    |------- TD target -------|
                                    |-------- TD error (delta) ------|

    Update a guess toward a better guess. Unlike MC you do not wait for the
    episode to end — you learn ONLINE, after every single step, and the method
    works on continuing (non-terminating) tasks.

    Trade-off vs MC: TD is BIASED (V(s_(t+1)) is currently wrong) but has much
    LOWER VARIANCE (one random reward and one random transition, instead of the
    whole remaining episode). In practice TD usually wins, and it is the basis
    for SARSA, Q-learning and every actor-critic method.
    """
    rng = random.Random(seed)
    V = {s: 0.0 for s in range(1, env.n - 1)}
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a = rng.randrange(2)
            s2, r, done = env.step(a)
            target = r + (0.0 if done else V.get(s2, 0.0))
            V[s] += alpha * (target - V[s])       # the TD error
            s = s2
    return V


def n_step_td(env, n=3, episodes=500, alpha=0.1, seed=0):
    """
    n-STEP TD — the whole spectrum between TD(0) and Monte Carlo.

        G_t:t+n = r_(t+1) + gamma r_(t+2) + ... + gamma^(n-1) r_(t+n)
                  + gamma^n V(s_(t+n))
        V(s_t) <- V(s_t) + alpha [ G_t:t+n - V(s_t) ]

    n = 1 is TD(0); n = infinity is Monte Carlo. Intermediate n is usually best:
    enough real reward to cut the bias, enough bootstrapping to cut the variance.
    This is the bias-variance trade-off appearing inside RL.
    """
    rng = random.Random(seed)
    V = {s: 0.0 for s in range(1, env.n - 1)}
    for _ in range(episodes):
        s = env.reset()
        states, rewards = [s], []
        T = 10 ** 9                       # 'infinity' until the episode ends
        t = 0
        while True:
            if t < T:
                s2, r, done = env.step(rng.randrange(2))
                states.append(s2)
                rewards.append(r)
                if done:
                    T = t + 1
            tau = t - n + 1
            if tau >= 0:
                G = sum(rewards[i] for i in range(tau, min(tau + n, T)))
                if tau + n < T:
                    G += V.get(states[tau + n], 0.0)
                if states[tau] in V:
                    V[states[tau]] += alpha * (G - V[states[tau]])
            if tau == T - 1:
                break
            t += 1
    return V


def td_lambda(env, lam=0.8, episodes=500, alpha=0.05, seed=0):
    """
    TD(lambda) WITH ELIGIBILITY TRACES — n-step averaging done online.

    The lambda-return geometrically averages ALL n-step returns:

        G_t^lambda = (1-lambda) sum_{n>=1} lambda^(n-1) G_t:t+n

    Computing that forward requires the whole episode. The BACKWARD view gets
    the same result incrementally with a trace per state:

        e(s) <- gamma*lambda*e(s) + 1{s = s_t}        (accumulating trace)
        delta = r + gamma V(s') - V(s)
        V(s) <- V(s) + alpha * delta * e(s)   for ALL s

    Interpretation: the trace is a short-term memory of how recently and how
    often each state was visited, i.e. how much CREDIT it deserves for the
    current TD error. lambda = 0 recovers TD(0); lambda = 1 recovers Monte Carlo.
    """
    rng = random.Random(seed)
    V = {s: 0.0 for s in range(1, env.n - 1)}
    for _ in range(episodes):
        s = env.reset()
        e = {k: 0.0 for k in V}
        done = False
        while not done:
            s2, r, done = env.step(rng.randrange(2))
            delta = r + (0.0 if done else V.get(s2, 0.0)) - V.get(s, 0.0)
            if s in e:
                e[s] += 1.0
            for k in V:
                V[k] += alpha * delta * e[k]
                e[k] *= lam                        # gamma = 1 here
            s = s2
    return V


# =============================================================================
# 4. TABULAR CONTROL
# =============================================================================
def epsilon_greedy(Q, s, nA, eps, rng):
    """
    THE EXPLORATION-EXPLOITATION TRADE-OFF, in its simplest form.
    With probability eps act uniformly at random; otherwise act greedily.

    GLIE (Greedy in the Limit with Infinite Exploration) is the condition for
    convergence: every (s,a) must be visited infinitely often AND eps must decay
    to 0. eps = 1/k or an exponential decay both satisfy it.
    """
    if rng.random() < eps:
        return rng.randrange(nA)
    qs = Q[s]
    best = max(qs)
    return rng.choice([a for a in range(nA) if qs[a] == best])


def mc_control(env, episodes=5000, eps_start=1.0, eps_end=0.05, gamma=None, seed=0):
    """
    MONTE CARLO CONTROL — GPI with sampled returns.
    Q(s,a) <- average return following (s,a), then act eps-greedily w.r.t. Q.
    No model needed, but must wait for episode termination, and high variance.
    """
    rng = random.Random(seed)
    g = gamma if gamma is not None else env.gamma
    Q = {s: [0.0] * env.nA for s in env.states}
    N = {s: [0] * env.nA for s in env.states}
    for ep in range(episodes):
        eps = linear_decay(ep, eps_start, eps_end, episodes * 0.8)
        s = env.reset()
        traj = []
        for _ in range(200):
            a = epsilon_greedy(Q, s, env.nA, eps, rng)
            s2, r, done = env.step(a)
            traj.append((s, a, r))
            s = s2
            if done:
                break
        G = 0.0
        for t in range(len(traj) - 1, -1, -1):
            st, at, r = traj[t]
            G = r + g * G
            N[st][at] += 1
            Q[st][at] += (G - Q[st][at]) / N[st][at]
    return Q


def sarsa(env, episodes=3000, alpha=0.5, eps_start=1.0, eps_end=0.05,
          gamma=None, seed=0, max_steps=200):
    """
    SARSA — ON-POLICY TD control. The name is the tuple it uses:
        (S, A, R, S', A')

        Q(s,a) <- Q(s,a) + alpha [ r + gamma Q(s', a') - Q(s,a) ]

    a' is the action ACTUALLY TAKEN next by the behaviour policy. So SARSA
    evaluates and improves the eps-greedy policy it is really following,
    including its random mistakes. On the cliff this makes it learn the SAFE
    path away from the edge — the optimal policy given that you sometimes act
    randomly. Converges to the optimal policy only as eps -> 0.
    """
    rng = random.Random(seed)
    g = gamma if gamma is not None else env.gamma
    Q = {s: [0.0] * env.nA for s in env.states}
    returns = []
    for ep in range(episodes):
        eps = linear_decay(ep, eps_start, eps_end, episodes * 0.8)
        s = env.reset()
        a = epsilon_greedy(Q, s, env.nA, eps, rng)
        total = 0.0
        for _ in range(max_steps):
            s2, r, done = env.step(a)
            total += r
            a2 = epsilon_greedy(Q, s2, env.nA, eps, rng)
            target = r if done else r + g * Q[s2][a2]
            Q[s][a] += alpha * (target - Q[s][a])
            s, a = s2, a2
            if done:
                break
        returns.append(total)
    return Q, returns


def expected_sarsa(env, episodes=3000, alpha=0.5, eps_start=1.0, eps_end=0.05,
                   gamma=None, seed=0, max_steps=200):
    """
    EXPECTED SARSA — replace the sampled Q(s',a') with its EXPECTATION under the
    policy:

        Q(s,a) <- Q(s,a) + alpha [ r + gamma sum_a' pi(a'|s') Q(s',a') - Q(s,a) ]

    Removes the variance due to sampling a', so it tolerates much larger alpha
    and is never worse than SARSA (at slightly more compute per step). Note
    that with a GREEDY target policy the expectation becomes a max — Expected
    SARSA generalizes Q-learning.
    """
    rng = random.Random(seed)
    g = gamma if gamma is not None else env.gamma
    Q = {s: [0.0] * env.nA for s in env.states}
    returns = []
    for ep in range(episodes):
        eps = linear_decay(ep, eps_start, eps_end, episodes * 0.8)
        s = env.reset()
        total = 0.0
        for _ in range(max_steps):
            a = epsilon_greedy(Q, s, env.nA, eps, rng)
            s2, r, done = env.step(a)
            total += r
            if done:
                target = r
            else:
                qs = Q[s2]
                best = max(qs)
                nbest = sum(1 for v in qs if v == best)
                exp_q = 0.0
                for i, v in enumerate(qs):
                    p = eps / env.nA + ((1 - eps) / nbest if v == best else 0.0)
                    exp_q += p * v
                target = r + g * exp_q
            Q[s][a] += alpha * (target - Q[s][a])
            s = s2
            if done:
                break
        returns.append(total)
    return Q, returns


def q_learning(env, episodes=3000, alpha=0.5, eps_start=1.0, eps_end=0.05,
               gamma=None, seed=0, max_steps=200, q_init=0.0):
    """
    Q-LEARNING — OFF-POLICY TD control (Watkins 1989).

        Q(s,a) <- Q(s,a) + alpha [ r + gamma max_a' Q(s',a') - Q(s,a) ]

    The target uses max_a', the GREEDY action, regardless of what the behaviour
    policy actually did. That decoupling is what makes it off-policy: it learns
    Q* while following any sufficiently exploratory policy, which in turn is
    what allows experience replay and learning from demonstrations.

    Converges to Q* with probability 1 given infinite visits to every (s,a) and
    Robbins-Monro step sizes (sum alpha = inf, sum alpha^2 < inf).

    On the cliff it learns the optimal edge-hugging path, but its ONLINE return
    during training is worse than SARSA's because eps-greedy exploration keeps
    pushing it off the cliff.

    q_init sets the OPTIMISTIC INITIALIZATION level. If Q starts ABOVE any
    achievable value, every untried action looks better than the tried ones, so
    the agent explores systematically even with eps = 0 — "optimism in the face
    of uncertainty" again, this time as an initialization rather than a bonus.
    It is a powerful trick in stationary episodic tasks and useless in
    non-stationary ones (the optimism is spent once and never renewed).
    """
    rng = random.Random(seed)
    g = gamma if gamma is not None else env.gamma
    Q = {s: [q_init] * env.nA for s in env.states}
    returns = []
    for ep in range(episodes):
        eps = linear_decay(ep, eps_start, eps_end, episodes * 0.8)
        s = env.reset()
        total = 0.0
        for _ in range(max_steps):
            a = epsilon_greedy(Q, s, env.nA, eps, rng)
            s2, r, done = env.step(a)
            total += r
            target = r if done else r + g * max(Q[s2])
            Q[s][a] += alpha * (target - Q[s][a])
            s = s2
            if done:
                break
        returns.append(total)
    return Q, returns


def double_q_learning(env, episodes=3000, alpha=0.5, eps_start=1.0, eps_end=0.05,
                      gamma=None, seed=0, max_steps=200):
    """
    DOUBLE Q-LEARNING — fixes MAXIMIZATION BIAS.

    The problem: E[max_a Q(s,a)] >= max_a E[Q(s,a)]. Taking a max over NOISY
    estimates systematically overestimates, because whichever action happens to
    have positive noise gets selected. In a stochastic environment this makes
    Q-learning persistently optimistic and can lock in a bad action.

    The fix: keep two independent tables and DECOUPLE selection from evaluation.

        a* = argmax_a Q_A(s',a)            select with A
        target = r + gamma * Q_B(s', a*)   evaluate with B

    Q_B's noise on a* is independent of the argmax, so the bias cancels. The
    same idea scaled up is Double DQN.
    """
    rng = random.Random(seed)
    g = gamma if gamma is not None else env.gamma
    QA = {s: [0.0] * env.nA for s in env.states}
    QB = {s: [0.0] * env.nA for s in env.states}
    returns = []
    for ep in range(episodes):
        eps = linear_decay(ep, eps_start, eps_end, episodes * 0.8)
        s = env.reset()
        total = 0.0
        for _ in range(max_steps):
            comb = {s: [QA[s][a] + QB[s][a] for a in range(env.nA)]}
            a = epsilon_greedy(comb, s, env.nA, eps, rng)
            s2, r, done = env.step(a)
            total += r
            if rng.random() < 0.5:
                astar = max(range(env.nA), key=lambda x: QA[s2][x])
                target = r if done else r + g * QB[s2][astar]
                QA[s][a] += alpha * (target - QA[s][a])
            else:
                bstar = max(range(env.nA), key=lambda x: QB[s2][x])
                target = r if done else r + g * QA[s2][bstar]
                QB[s][a] += alpha * (target - QB[s][a])
            s = s2
            if done:
                break
        returns.append(total)
    Q = {s: [(QA[s][a] + QB[s][a]) / 2 for a in range(env.nA)] for s in env.states}
    return Q, returns


def dyna_q(env, episodes=300, planning_steps=10, alpha=0.5, eps=0.1,
           gamma=None, seed=0, max_steps=200):
    """
    DYNA-Q — model-based planning fused with model-free learning.

    After every REAL step:
      1. do a normal Q-learning update on the real transition
      2. store it in a learned model: Model[(s,a)] = (r, s')
      3. do `planning_steps` extra Q-learning updates on transitions SAMPLED
         from that model (imagined experience)

    Planning and learning are the same update, applied to simulated vs real
    data. The payoff is dramatic SAMPLE EFFICIENCY: each real interaction is
    amortized over many backups, which matters when environment steps are
    expensive (robots, medicine). The risk is model bias — if the learned model
    is wrong, you confidently plan on fiction (Dyna-Q+ adds exploration bonuses
    for stale transitions to cope with a changing world).
    """
    rng = random.Random(seed)
    g = gamma if gamma is not None else env.gamma
    Q = {s: [0.0] * env.nA for s in env.states}
    model = {}
    steps_per_episode = []
    for ep in range(episodes):
        s = env.reset()
        n = 0
        for _ in range(max_steps):
            a = epsilon_greedy(Q, s, env.nA, eps, rng)
            s2, r, done = env.step(a)
            n += 1
            target = r if done else r + g * max(Q[s2])
            Q[s][a] += alpha * (target - Q[s][a])
            model[(s, a)] = (r, s2, done)
            keys = list(model)
            for _ in range(planning_steps):            # PLANNING
                ps, pa = keys[rng.randrange(len(keys))]
                pr, ps2, pdone = model[(ps, pa)]
                t = pr if pdone else pr + g * max(Q[ps2])
                Q[ps][pa] += alpha * (t - Q[ps][pa])
            s = s2
            if done:
                break
        steps_per_episode.append(n)
    return Q, steps_per_episode


def greedy_policy(Q, env):
    return {s: max(range(env.nA), key=lambda a: Q[s][a]) for s in env.states}


def evaluate_policy(env, policy, episodes=200, max_steps=200, seed=0):
    """Run the greedy policy and report the average undiscounted return."""
    env.rng = random.Random(seed)
    total = 0.0
    for _ in range(episodes):
        s = env.reset()
        for _ in range(max_steps):
            s, r, done = env.step(policy[s] if not callable(policy) else policy(s))
            total += r
            if done:
                break
    return total / episodes


# =============================================================================
# 5. FUNCTION APPROXIMATION — when the state space is too big for a table
# =============================================================================
class TileCoder:
    """
    TILE CODING — the classic linear feature construction for continuous states.

    Overlay `n_tilings` grids over the state space, each OFFSET slightly. A
    state activates exactly one tile per tiling, so the feature vector is
    binary and sparse with exactly n_tilings ones.

    Why it works well:
      * generalization is LOCAL and controllable — nearby states share most
        tiles, distant ones share none
      * the effective learning rate is alpha/n_tilings, and the features are
        binary, so linear updates are cheap and stable
      * it is a way to get non-linear value functions while keeping the
        LINEAR convergence guarantees
    """

    def __init__(self, lows, highs, bins=8, n_tilings=8, seed=0):
        rng = random.Random(seed)
        self.lows, self.highs, self.bins, self.n = lows, highs, bins, n_tilings
        self.dim = len(lows)
        self.widths = [(h - l) / bins for l, h in zip(lows, highs)]
        self.offsets = [[rng.random() * w for w in self.widths]
                        for _ in range(n_tilings)]
        self.size = n_tilings * (bins + 1) ** self.dim

    def features(self, s):
        idxs = []
        for t in range(self.n):
            code = 0
            for d in range(self.dim):
                v = (s[d] - self.lows[d] + self.offsets[t][d]) / self.widths[d]
                i = int(max(0, min(self.bins, v)))
                code = code * (self.bins + 1) + i
            idxs.append(t * (self.bins + 1) ** self.dim + code)
        return idxs                                   # sparse: list of active indices


def linear_sarsa(env, coder, episodes=200, alpha=0.1, eps=0.1, gamma=1.0,
                 seed=0, max_steps=500):
    """
    SEMI-GRADIENT SARSA with linear function approximation.

        Q(s,a) = w_a . x(s)
        w_a <- w_a + alpha [ r + gamma Q(s',a') - Q(s,a) ] * x(s)

    "SEMI-gradient" because we differentiate only Q(s,a), treating the TARGET
    r + gamma Q(s',a') as a constant even though it also depends on w. The true
    gradient (residual gradient) is slower and often worse in practice.

    THE DEADLY TRIAD: function approximation + bootstrapping + off-policy
    training can DIVERGE. On-policy semi-gradient SARSA (as here) is safe;
    off-policy Q-learning with approximation is the dangerous combination, which
    is why DQN needs target networks and replay to be stable.
    """
    rng = random.Random(seed)
    w = [[0.0] * coder.size for _ in range(env.nA)]

    def q(feats, a):
        # binary features: the value is just the SUM of the active tiles'
        # weights (no division — the alpha/n_tilings below does the scaling)
        return sum(w[a][i] for i in feats)

    def act(feats):
        if rng.random() < eps:
            return rng.randrange(env.nA)
        qs = [q(feats, a) for a in range(env.nA)]
        return max(range(env.nA), key=lambda a: qs[a])

    lengths = []
    for ep in range(episodes):
        s = env.reset()
        f = coder.features(s)
        a = act(f)
        for t in range(max_steps):
            s2, r, done = env.step(a)
            f2 = coder.features(s2)
            a2 = act(f2)
            target = r if done else r + gamma * q(f2, a2)
            delta = target - q(f, a)
            for i in f:
                w[a][i] += alpha * delta / coder.n
            f, a = f2, a2
            if done:
                lengths.append(t + 1)
                break
        else:
            lengths.append(max_steps)
    return w, lengths


def dqn(env, episodes=150, hidden=32, lr=1e-3, gamma=0.99, batch=32,
        buffer_size=10000, target_update=200, eps_start=1.0, eps_end=0.05,
        eps_steps=2000, train_every=2, double=False, seed=0, max_steps=500):
    """
    DEEP Q-NETWORK (Mnih et al. 2015)

        loss = ( r + gamma * max_a' Q_target(s',a')  -  Q(s,a) )^2

    Q-learning with a neural network diverges without two specific fixes:

      1. EXPERIENCE REPLAY — sample minibatches uniformly from a large buffer.
         Breaks the temporal correlation of consecutive transitions and reuses
         each transition many times.
      2. TARGET NETWORK — compute the bootstrap target with a FROZEN copy of the
         network, synced every C steps. Without it you are regressing toward a
         target that moves every time you update, which creates a feedback loop
         ("chasing your own tail") and oscillation or divergence.

    DOUBLE DQN adds the decoupling fix for maximization bias:

        vanilla: y = r + gamma * max_a' Q_target(s', a')
        double : a* = argmax_a' Q_online(s', a')
                 y  = r + gamma * Q_target(s', a*)

    Select with the online net, evaluate with the target net — the same trick
    as tabular Double Q-learning.

    Gradient of the loss w.r.t. the network output: 2(Q(s,a) - y) on the taken
    action's output unit, 0 on all others (Huber loss clips this to +/-1, which
    is what the original paper used for stability).
    """
    rng = random.Random(seed)
    online = MLP([env.obs_dim, hidden, hidden, env.nA], "relu", seed=seed)
    target = MLP([env.obs_dim, hidden, hidden, env.nA], "relu", seed=seed)
    target.copy_from(online)
    opt = AdamMLP(online, lr=lr)
    buf = ReplayBuffer(buffer_size, seed)
    step_count = 0
    returns = []
    for ep in range(episodes):
        s = env.reset()
        total = 0.0
        for _ in range(max_steps):
            eps = linear_decay(step_count, eps_start, eps_end, eps_steps)
            if rng.random() < eps:
                a = rng.randrange(env.nA)
            else:
                qs = online.forward([s])[0]
                a = max(range(env.nA), key=lambda i: qs[i])
            s2, r, done = env.step(a)
            buf.push(s, a, r, s2, done)
            total += r
            s = s2
            step_count += 1

            if len(buf) >= batch and step_count % train_every == 0:
                bt = buf.sample(batch)
                states = [b[0] for b in bt]
                next_states = [b[3] for b in bt]
                qn_target = target.forward(next_states)
                qn_online = online.forward(next_states) if double else None
                # the online forward on `states` must come LAST so that the
                # cached activations match the backward pass we are about to do
                qs_online = online.forward(states)
                dout = [[0.0] * env.nA for _ in range(len(bt))]
                for i, (_, ai, ri, _, di) in enumerate(bt):
                    if di:
                        y = ri
                    elif double:
                        astar = max(range(env.nA), key=lambda k: qn_online[i][k])
                        y = ri + gamma * qn_target[i][astar]
                    else:
                        y = ri + gamma * max(qn_target[i])
                    err = qs_online[i][ai] - y
                    err = max(-1.0, min(1.0, err))          # Huber clipping
                    dout[i][ai] = 2.0 * err / len(bt)
                gW, gb = online.backward(dout)
                opt.step(gW, gb, clip=10.0)

            if step_count % target_update == 0:
                target.copy_from(online)                     # hard sync
            if done:
                break
        returns.append(total)
    return online, returns


def greedy_eval_net(env, net, episodes=20, max_steps=500):
    total = 0.0
    for _ in range(episodes):
        s = env.reset()
        for _ in range(max_steps):
            qs = net.forward([s])[0]
            s, r, done = env.step(max(range(env.nA), key=lambda i: qs[i]))
            total += r
            if done:
                break
    return total / episodes


# =============================================================================
# 6. POLICY GRADIENT METHODS
# =============================================================================
def discounted_returns(rewards, gamma):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return out[::-1]


def reinforce(env, episodes=300, hidden=32, lr=5e-3, gamma=0.99,
              baseline=False, entropy_coef=0.0, seed=0, max_steps=500):
    """
    REINFORCE — the POLICY GRADIENT THEOREM made into an algorithm.

        J(theta) = E_pi[ G_0 ]
        grad J   = E_pi[ sum_t  grad log pi(a_t|s_t) * G_t ]

    The derivation rests on the LOG-DERIVATIVE (score function) trick:

        grad p = p * grad log p
        => grad E_p[f] = E_p[ f * grad log p ]

    which turns a gradient of an expectation into an expectation of a gradient,
    so it can be estimated from samples. Crucially it needs NO model of the
    environment and NO differentiable reward.

    Intuition: increase the log-probability of actions that led to high return,
    decrease it for low return. The magnitude of the return scales the step.

    BASELINE: subtracting any function b(s) that does not depend on the action
    leaves the gradient UNBIASED, because E[grad log pi * b(s)] = b(s) * grad(1) = 0,
    but it can massively reduce VARIANCE. The standard choice is b(s) = V(s),
    which makes the weight the ADVANTAGE A = G - V. Here we use the batch-mean
    return as a simple baseline.

    ENTROPY BONUS: add beta * H(pi) to the objective to keep the policy from
    collapsing to a deterministic (and possibly premature) choice too early.

    Gradient w.r.t. the logits for a softmax policy:
        dlogpi/dz = onehot(a) - pi
        so the ASCENT direction is (onehot(a) - pi) * A, and the descent
        gradient we hand to the optimizer is (pi - onehot(a)) * A.
    """
    rng = random.Random(seed)
    net = MLP([env.obs_dim, hidden, env.nA], "tanh", seed=seed)
    opt = AdamMLP(net, lr=lr)
    returns = []
    for ep in range(episodes):
        s = env.reset()
        states, actions, rewards = [], [], []
        for _ in range(max_steps):
            logits = net.forward([s])[0]
            p = softmax(logits)
            x, acc, a = rng.random(), 0.0, env.nA - 1
            for i, pi_ in enumerate(p):
                acc += pi_
                if x <= acc:
                    a = i
                    break
            s2, r, done = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = s2
            if done:
                break
        G = discounted_returns(rewards, gamma)
        if baseline:
            m = sum(G) / len(G)
            sd = math.sqrt(sum((g - m) ** 2 for g in G) / len(G)) + 1e-8
            G = [(g - m) / sd for g in G]           # normalized advantage
        logits = net.forward(states)
        dout = []
        n = len(states)
        for i in range(n):
            p = softmax(logits[i])
            row = [(p[k] - (1.0 if k == actions[i] else 0.0)) * G[i] / n
                   for k in range(env.nA)]
            if entropy_coef:                        # d(-H)/dz = p*(logp + H)
                H = -sum(pk * math.log(pk + 1e-12) for pk in p)
                for k in range(env.nA):
                    row[k] += entropy_coef * p[k] * (math.log(p[k] + 1e-12) + H) / n
            dout.append(row)
        gW, gb = net.backward(dout)
        opt.step(gW, gb, clip=5.0)
        returns.append(sum(rewards))
    return net, returns


def actor_critic(env, episodes=300, hidden=32, lr_a=5e-3, lr_c=1e-2, gamma=0.99,
                 lam=0.95, use_gae=True, entropy_coef=0.01, seed=0, max_steps=500):
    """
    ADVANTAGE ACTOR-CRITIC (A2C)

        actor  pi(a|s; theta)      updated by  grad log pi * A
        critic V(s; w)             updated by  regression onto the TD target

        one-step advantage:  A_t = r + gamma V(s') - V(s)          (the TD error!)

    The critic IS the learned baseline. REINFORCE waits for the full return
    (unbiased, high variance); the critic replaces it with a bootstrapped
    estimate (biased, low variance). Actor-critic is exactly the MC-vs-TD
    trade-off applied to policy gradients.

    GENERALIZED ADVANTAGE ESTIMATION (GAE) interpolates the whole spectrum:

        A_t^GAE = sum_{l>=0} (gamma*lambda)^l * delta_(t+l),
        delta_t = r_t + gamma V(s_(t+1)) - V(s_t)

    lambda = 0 gives the one-step TD advantage (low variance, high bias);
    lambda = 1 gives the Monte Carlo advantage (unbiased, high variance).
    lambda ~ 0.95 is the standard compromise, and it is computed backwards in
    one pass with A_t = delta_t + gamma*lambda*A_(t+1).
    """
    rng = random.Random(seed)
    actor = MLP([env.obs_dim, hidden, env.nA], "tanh", seed=seed)
    critic = MLP([env.obs_dim, hidden, 1], "tanh", seed=seed + 1)
    opt_a, opt_c = AdamMLP(actor, lr=lr_a), AdamMLP(critic, lr=lr_c)
    returns = []
    for ep in range(episodes):
        s = env.reset()
        states, actions, rewards, dones = [], [], [], []
        for _ in range(max_steps):
            p = softmax(actor.forward([s])[0])
            x, acc, a = rng.random(), 0.0, env.nA - 1
            for i, pi_ in enumerate(p):
                acc += pi_
                if x <= acc:
                    a = i
                    break
            s2, r, done = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            dones.append(done)
            s = s2
            if done:
                break
        vals = [v[0] for v in critic.forward(states)]
        last_v = 0.0 if dones[-1] else critic.forward([s])[0][0]
        n = len(states)
        adv = [0.0] * n
        if use_gae:
            gae = 0.0
            for t in range(n - 1, -1, -1):
                v_next = last_v if t == n - 1 else vals[t + 1]
                delta = rewards[t] + gamma * v_next * (0.0 if dones[t] else 1.0) - vals[t]
                gae = delta + gamma * lam * (0.0 if dones[t] else 1.0) * gae
                adv[t] = gae
        else:
            for t in range(n - 1, -1, -1):
                v_next = last_v if t == n - 1 else vals[t + 1]
                adv[t] = rewards[t] + gamma * v_next * (0.0 if dones[t] else 1.0) - vals[t]
        targets = [adv[t] + vals[t] for t in range(n)]
        m = sum(adv) / n
        sd = math.sqrt(sum((a_ - m) ** 2 for a_ in adv) / n) + 1e-8
        norm_adv = [(a_ - m) / sd for a_ in adv]

        logits = actor.forward(states)
        dout = []
        for i in range(n):
            p = softmax(logits[i])
            row = [(p[k] - (1.0 if k == actions[i] else 0.0)) * norm_adv[i] / n
                   for k in range(env.nA)]
            H = -sum(pk * math.log(pk + 1e-12) for pk in p)
            for k in range(env.nA):
                row[k] += entropy_coef * p[k] * (math.log(p[k] + 1e-12) + H) / n
            dout.append(row)
        gW, gb = actor.backward(dout)
        opt_a.step(gW, gb, clip=5.0)

        vpred = critic.forward(states)
        dv = [[2.0 * (vpred[i][0] - targets[i]) / n] for i in range(n)]
        gW, gb = critic.backward(dv)
        opt_c.step(gW, gb, clip=5.0)
        returns.append(sum(rewards))
    return actor, returns


def ppo(env, iterations=60, steps_per_iter=600, hidden=32, lr_a=3e-3, lr_c=1e-2,
        gamma=0.99, lam=0.95, clip_eps=0.2, epochs=4, minibatch=64,
        entropy_coef=0.01, seed=0, max_steps=500):
    """
    PROXIMAL POLICY OPTIMIZATION — the modern default (and the algorithm behind
    RLHF).

    THE PROBLEM IT SOLVES: vanilla policy gradient takes one gradient step per
    batch of data, because a large step can collapse the policy and there is no
    way back — the data for the new policy does not exist yet. TRPO fixed this
    with a hard KL trust region and second-order optimization. PPO gets ~the
    same effect with a first-order CLIPPED SURROGATE:

        ratio r_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
        L = E[ min( r_t A_t ,  clip(r_t, 1-eps, 1+eps) A_t ) ]

    Read the min carefully:
      * A > 0 (good action): the objective stops improving once r > 1+eps, so
        there is no incentive to push the probability arbitrarily high.
      * A < 0 (bad action): it stops once r < 1-eps.
      * The MIN makes it a pessimistic (lower) bound, so the update is
        conservative in both directions.
    The clip zeroes the gradient outside the trust region, which is what lets
    PPO safely do MULTIPLE epochs over the same batch — the source of its
    sample efficiency over REINFORCE.

    Full objective: L_clip - c1 * value_loss + c2 * entropy.
    """
    rng = random.Random(seed)
    actor = MLP([env.obs_dim, hidden, env.nA], "tanh", seed=seed)
    critic = MLP([env.obs_dim, hidden, 1], "tanh", seed=seed + 1)
    opt_a, opt_c = AdamMLP(actor, lr=lr_a), AdamMLP(critic, lr=lr_c)
    ep_returns = []
    s = env.reset()
    cur = 0.0
    for it in range(iterations):
        S, A, R, D, LOGP = [], [], [], [], []
        for _ in range(steps_per_iter):
            p = softmax(actor.forward([s])[0])
            x, acc, a = rng.random(), 0.0, env.nA - 1
            for i, pi_ in enumerate(p):
                acc += pi_
                if x <= acc:
                    a = i
                    break
            s2, r, done = env.step(a)
            S.append(s)
            A.append(a)
            R.append(r)
            D.append(done)
            LOGP.append(math.log(p[a] + 1e-12))
            cur += r
            s = s2
            if done:
                ep_returns.append(cur)
                cur = 0.0
                s = env.reset()
        vals = [v[0] for v in critic.forward(S)]
        last_v = critic.forward([s])[0][0]
        n = len(S)
        adv = [0.0] * n
        gae = 0.0
        for t in range(n - 1, -1, -1):
            v_next = last_v if t == n - 1 else vals[t + 1]
            nonterm = 0.0 if D[t] else 1.0
            delta = R[t] + gamma * v_next * nonterm - vals[t]
            gae = delta + gamma * lam * nonterm * gae
            adv[t] = gae
        ret = [adv[t] + vals[t] for t in range(n)]
        m = sum(adv) / n
        sd = math.sqrt(sum((a_ - m) ** 2 for a_ in adv) / n) + 1e-8
        adv = [(a_ - m) / sd for a_ in adv]

        idx = list(range(n))
        for _ in range(epochs):
            rng.shuffle(idx)
            for start in range(0, n, minibatch):
                mb = idx[start:start + minibatch]
                if not mb:
                    continue
                st = [S[i] for i in mb]
                logits = actor.forward(st)
                dout = []
                for j, i in enumerate(mb):
                    p = softmax(logits[j])
                    ratio = math.exp(math.log(p[A[i]] + 1e-12) - LOGP[i])
                    a_i = adv[i]
                    # gradient flows only INSIDE the trust region
                    if (a_i >= 0 and ratio > 1 + clip_eps) or \
                       (a_i < 0 and ratio < 1 - clip_eps):
                        coef = 0.0
                    else:
                        coef = ratio * a_i
                    row = [(p[k] - (1.0 if k == A[i] else 0.0)) * coef / len(mb)
                           for k in range(env.nA)]
                    H = -sum(pk * math.log(pk + 1e-12) for pk in p)
                    for k in range(env.nA):
                        row[k] += entropy_coef * p[k] * \
                            (math.log(p[k] + 1e-12) + H) / len(mb)
                    dout.append(row)
                gW, gb = actor.backward(dout)
                opt_a.step(gW, gb, clip=5.0)

                vpred = critic.forward(st)
                dv = [[2.0 * (vpred[j][0] - ret[i]) / len(mb)]
                      for j, i in enumerate(mb)]
                gW, gb = critic.backward(dv)
                opt_c.step(gW, gb, clip=5.0)
    return actor, ep_returns


def eval_policy_net(env, net, episodes=20, max_steps=500):
    total = 0.0
    for _ in range(episodes):
        s = env.reset()
        for _ in range(max_steps):
            p = softmax(net.forward([s])[0])
            a = max(range(env.nA), key=lambda i: p[i])
            s, r, done = env.step(a)
            total += r
            if done:
                break
    return total / episodes


# =============================================================================
# 7. BANDITS — RL with a single state
# =============================================================================
def run_bandit(bandit, strategy="thompson", steps=2000, eps=0.1, c=2.0,
               alpha=0.1, seed=0):
    """
    THE EXPLORATION STRATEGIES, side by side.

    eps-greedy   explore uniformly w.p. eps. Simple; never stops exploring, so
                 regret grows LINEARLY unless eps decays.
    UCB1         pick argmax [ mu_a + c*sqrt(ln t / n_a) ].
                 "Optimism in the face of uncertainty": the bonus is a
                 confidence radius that SHRINKS as an arm is pulled, so
                 under-explored arms get tried. Regret O(log t) — optimal up to
                 constants.
    Thompson     sample theta_a ~ Beta(1+successes, 1+failures), pull the argmax.
                 Bayesian probability matching: pull each arm with the
                 probability it is optimal. Usually the best in practice and
                 handles delayed/batched feedback gracefully.
    gradient     maintain preferences H_a, pi = softmax(H), and update
                 H_a += alpha (R - Rbar)(1{a=A} - pi_a). This is REINFORCE on a
                 one-state MDP — the simplest possible policy gradient.

    REGRET = t * mu* - sum of rewards received. It is the standard metric
    because it measures the COST of learning, not just final performance.
    """
    rng = random.Random(seed)
    k = bandit.k
    counts = [0] * k
    values = [0.0] * k
    wins = [0] * k
    H = [0.0] * k
    avg_reward = 0.0
    total, regret_curve = 0.0, []
    for t in range(1, steps + 1):
        if strategy == "eps":
            a = rng.randrange(k) if rng.random() < eps else \
                max(range(k), key=lambda i: values[i])
        elif strategy == "ucb":
            unseen = [i for i in range(k) if counts[i] == 0]
            if unseen:
                a = unseen[0]                      # try each arm once first
            else:
                a = max(range(k), key=lambda i: values[i] +
                        c * math.sqrt(math.log(t) / counts[i]))
        elif strategy == "gradient":
            p = softmax(H)
            x, acc, a = rng.random(), 0.0, k - 1
            for i, pi_ in enumerate(p):
                acc += pi_
                if x <= acc:
                    a = i
                    break
        else:                                          # thompson
            best, a = -1.0, 0
            for i in range(k):
                s_ = rng.gammavariate(1 + wins[i], 1)
                f_ = rng.gammavariate(1 + counts[i] - wins[i], 1)
                theta = s_ / (s_ + f_)
                if theta > best:
                    best, a = theta, i
        r = bandit.pull(a)
        counts[a] += 1
        wins[a] += int(r)
        values[a] += (r - values[a]) / counts[a]
        total += r
        if strategy == "gradient":
            avg_reward += (r - avg_reward) / t
            p = softmax(H)
            for i in range(k):
                H[i] += alpha * (r - avg_reward) * ((1.0 if i == a else 0.0) - p[i])
        regret_curve.append(t * bandit.best - total)
    return total, regret_curve, counts


# =============================================================================
# 8. DEMOS
# =============================================================================
def _hdr(t):
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def demo_dp():
    _hdr("1. DYNAMIC PROGRAMMING — the model is known, solve Bellman exactly")
    env = GridWorld()
    pi_pi, V_pi, n1 = policy_iteration(env)
    pi_vi, V_vi, n2 = value_iteration(env)
    agree = all(max(range(env.nA), key=lambda a: pi_pi[s][a]) == pi_vi[s]
                for s in env.states if not env.is_terminal(s))
    print(f"  policy iteration converged in {n1} policy improvements")
    print(f"  value iteration converged in {n2} sweeps")
    print(f"  the two agree on every state: {agree}")
    print("\n  optimal policy (U/R/D/L):")
    print("  " + env.render_policy(pi_vi).replace("\n", "\n  "))
    print("\n  optimal state values V*:")
    print("  " + env.render_values(V_vi).replace("\n", "\n  "))
    print("\n  Note how V* decreases with distance from the goal: each step costs")
    print("  -0.04 and future reward is discounted by gamma = 0.95. The policy")
    print("  routes AROUND the hazard H even though that path is longer.")


def demo_prediction():
    _hdr("2. PREDICTION — MC vs TD vs n-step vs TD(lambda)")
    ch = ChainMDP(7)
    true = ch.true_values()
    print(f"  Random walk on 7 states. TRUE values V(i) = i/6 are known, so we")
    print(f"  can measure the actual RMSE of each estimator.")
    print(f"  true V: {[round(true[s], 3) for s in sorted(true)]}")

    def rmse(V):
        return math.sqrt(sum((V[s] - true[s]) ** 2 for s in true) / len(true))

    print("\n  RMSE after 500 episodes, averaged over 15 seeds:")
    print(f"  {'alpha':<8}{'MC':>10}{'TD(0)':>10}{'n=3':>10}{'TD(0.8)':>10}")
    for a in [0.02, 0.05, 0.1, 0.2, 0.4]:
        r = []
        for fn in [lambda s: mc_prediction(ch, 500, alpha=a, seed=s),
                   lambda s: td0_prediction(ch, 500, alpha=a, seed=s),
                   lambda s: n_step_td(ch, 3, 500, alpha=a, seed=s),
                   lambda s: td_lambda(ch, 0.8, 500, alpha=a, seed=s)]:
            r.append(sum(rmse(fn(s)) for s in range(15)) / 15)
        print(f"  {a:<8}{r[0]:>10.4f}{r[1]:>10.4f}{r[2]:>10.4f}{r[3]:>10.4f}")
    print("\n  Read the table: TD(0)'s best result beats MC's best, and TD stays")
    print("  usable at large alpha where MC degrades badly. That is the variance")
    print("  difference — MC's target is the whole noisy return, TD's is one")
    print("  reward plus a bootstrapped estimate.")


def demo_cliff():
    _hdr("3. SARSA vs Q-LEARNING — the cliff, and what 'on-policy' means")
    cw = cliff_world()
    print("  Layout (S=start, H=cliff -100, G=goal), constant eps = 0.1:")
    for row in cw.layout:
        print("    " + row)
    results = {}
    for name, fn in [("SARSA (on-policy)", sarsa),
                     ("Expected SARSA", expected_sarsa),
                     ("Q-learning (off-policy)", q_learning),
                     ("Double Q-learning", double_q_learning)]:
        Q, ret = fn(cw, episodes=500, alpha=0.5, eps_start=0.1, eps_end=0.1,
                    seed=2, max_steps=100)
        online = sum(ret[-100:]) / 100
        greedy = evaluate_policy(cw, greedy_policy(Q, cw), 20, 100)
        results[name] = (online, greedy)
        print(f"  {name:<26} online return {online:8.2f} | "
              f"greedy policy return {greedy:7.2f}")
    print("\n  Q-learning's GREEDY policy is better (it finds the short path")
    print("  along the cliff edge) but its ONLINE return is worse, because")
    print("  eps-greedy exploration keeps knocking it off the cliff.")
    print("  SARSA backs up the action it actually takes, so it accounts for its")
    print("  own randomness and learns the safer route one row up.")
    print("  Moral: off-policy learns the optimal policy; on-policy learns the")
    print("  best policy GIVEN that you are still exploring.")


def demo_dyna():
    _hdr("4. DYNA-Q — planning with a learned model buys sample efficiency")
    maze = ["S.........",
            ".####.###.",
            ".#...#...#",
            ".#.#.#.#..",
            "...#...#.G"]

    def mk():
        return GridWorld(maze, slip=0.0, step_cost=-0.01, goal_reward=1.0,
                         gamma=0.95, seed=0)

    print("  A 10x5 maze. Cost is measured in REAL environment steps taken to")
    print("  learn — the currency that matters when interaction is expensive.\n")
    for pl in [0, 5, 30]:
        env = mk()
        Q, steps = dyna_q(env, episodes=30, planning_steps=pl, seed=3,
                          max_steps=300)
        ev = evaluate_policy(env, greedy_policy(Q, env), 20, 300)
        label = "plain Q-learning" if pl == 0 else f"Dyna-Q, {pl} planning steps"
        print(f"  {label:<28} real steps used: {sum(steps):5d} | "
              f"last5 episodes: {sum(steps[-5:])/5:5.1f} | greedy {ev:.3f}")
    print("\n  Same final policy, but 30 planning steps per real step cut the")
    print("  environment interaction by ~3.7x. Each real transition is replayed")
    print("  through the learned model many times, so value propagates backwards")
    print("  from the goal without having to walk the maze again.")
    print("  The risk: if the model is wrong, you confidently plan on fiction.")


def demo_exploration():
    _hdr("5. EXPLORATION — epsilon vs optimistic initialization")
    maze = ["S.........",
            ".####.###.",
            ".#...#...#",
            ".#.#.#.#..",
            "...#...#.G"]

    def mk():
        return GridWorld(maze, slip=0.0, step_cost=-0.01, goal_reward=1.0,
                         gamma=0.95, seed=0)

    print("  Q-learning on a 10x5 maze, 400 episodes. Rewards are negative until")
    print("  the goal, so Q initialized at 0 is already OPTIMISTIC; -1 is")
    print("  pessimistic (below any achievable value).\n")
    print(f"  {'Q init':<14}{'eps':<8}{'greedy return':>15}   {'':<4}")
    for q0, eps in [(-1.0, 0.0), (-1.0, 0.1), (-1.0, 0.3), (-1.0, 0.5),
                    (0.0, 0.0)]:
        env = mk()
        Q, _ = q_learning(env, episodes=400, alpha=0.5, eps_start=eps,
                          eps_end=eps, seed=5, max_steps=300, q_init=q0)
        ev = evaluate_policy(env, greedy_policy(Q, env), 20, 300)
        tag = "pessimistic" if q0 < 0 else "optimistic"
        mark = "found the goal" if ev > 0 else "never found the goal"
        print(f"  {tag:<14}{eps:<8}{ev:>15.3f}   {mark}")
    print("\n  Two lessons, and the second is the one people miss:")
    print("  1. Pure exploitation from a pessimistic init NEVER finds the goal —")
    print("     the agent commits to its first guess. Optimistic init alone")
    print("     solves it with eps = 0, because every untried action looks good.")
    print("  2. Epsilon-greedy barely helps here. Undirected random exploration")
    print("     has to random-walk through a maze to stumble on reward, and the")
    print("     probability of that decays exponentially with distance. This is")
    print("     the HARD-EXPLORATION problem (Montezuma's Revenge is the famous")
    print("     case), and it is why the field developed count-based bonuses,")
    print("     curiosity/prediction-error rewards, and Go-Explore.")


def demo_function_approx():
    _hdr("9. FUNCTION APPROXIMATION — CartPole has a CONTINUOUS state space")
    print("  4 continuous variables (x, x_dot, theta, theta_dot): a lookup table")
    print("  is impossible. Two ways to generalize across states:\n")

    cp = CartPole(seed=0, max_steps=200)
    coder = TileCoder([-2.4, -3, -0.21, -3], [2.4, 3, 0.21, 3],
                      bins=8, n_tilings=8, seed=0)
    w, lens = linear_sarsa(cp, coder, episodes=200, alpha=0.5, eps=0.05,
                           gamma=1.0, max_steps=200)
    print(f"  (a) TILE CODING + linear semi-gradient SARSA "
          f"({coder.n} tilings x {coder.bins}^4 tiles)")
    print(f"      episode length: first10 {sum(lens[:10])/10:.1f} -> "
          f"last20 {sum(lens[-20:])/20:.1f}  (max 200)")

    print(f"\n  (b) DQN — a neural Q-function with replay + target network")
    cp = CartPole(seed=0, max_steps=200)
    net, ret = dqn(cp, episodes=130, hidden=32, lr=1e-3, eps_steps=2000,
                   target_update=500, train_every=1, double=False, seed=1,
                   max_steps=200)
    print(f"      vanilla DQN  last20 {sum(ret[-20:])/20:6.1f} | "
          f"greedy {greedy_eval_net(cp, net, 10, 200):6.1f}")
    cp = CartPole(seed=0, max_steps=200)
    net2, ret2 = dqn(cp, episodes=130, hidden=32, lr=1e-3, eps_steps=2000,
                     target_update=500, train_every=1, double=True, seed=1,
                     max_steps=200)
    print(f"      Double DQN   last20 {sum(ret2[-20:])/20:6.1f} | "
          f"greedy {greedy_eval_net(cp, net2, 10, 200):6.1f}")
    print("\n  Double DQN wins by decoupling action SELECTION from EVALUATION,")
    print("  which removes the systematic overestimation of max_a Q(s,a).")
    print("  Note also how much noisier these runs are than the tabular ones —")
    print("  value-based deep RL is genuinely unstable, which is the whole")
    print("  reason replay buffers and target networks exist.")


def demo_policy_gradient():
    _hdr("7. POLICY GRADIENT — optimize the policy directly")
    cp = CartPole(seed=0, max_steps=200)

    print("  All runs on CartPole, max 200 steps (200 = solved).\n")
    _, r1 = reinforce(cp, episodes=200, lr=8e-3, baseline=False,
                      entropy_coef=0.01, seed=1, max_steps=200)
    cp = CartPole(seed=0, max_steps=200)
    net2, r2 = reinforce(cp, episodes=200, lr=8e-3, baseline=True,
                         entropy_coef=0.01, seed=1, max_steps=200)
    cp = CartPole(seed=0, max_steps=200)
    net3, r3 = actor_critic(cp, episodes=200, lr_a=8e-3, lr_c=2e-2,
                            use_gae=True, seed=1, max_steps=200)
    cp = CartPole(seed=0, max_steps=200)
    net4, r4 = ppo(cp, iterations=25, steps_per_iter=500, lr_a=5e-3,
                   seed=1, max_steps=200)

    print(f"  {'algorithm':<28}{'first 20 eps':>14}{'last 20 eps':>14}"
          f"{'greedy eval':>14}")
    cpe = CartPole(seed=0, max_steps=200)
    print(f"  {'REINFORCE (no baseline)':<28}{sum(r1[:20])/20:>14.1f}"
          f"{sum(r1[-20:])/20:>14.1f}{'-':>14}")
    print(f"  {'REINFORCE + baseline':<28}{sum(r2[:20])/20:>14.1f}"
          f"{sum(r2[-20:])/20:>14.1f}"
          f"{eval_policy_net(cpe, net2, 10, 200):>14.1f}")
    print(f"  {'Actor-Critic (A2C+GAE)':<28}{sum(r3[:20])/20:>14.1f}"
          f"{sum(r3[-20:])/20:>14.1f}"
          f"{eval_policy_net(cpe, net3, 10, 200):>14.1f}")
    print(f"  {'PPO (clipped surrogate)':<28}{sum(r4[:20])/20:>14.1f}"
          f"{sum(r4[-20:])/20:>14.1f}"
          f"{eval_policy_net(cpe, net4, 10, 200):>14.1f}")
    print("\n  The baseline costs nothing in bias (E[grad log pi * b(s)] = 0) and")
    print("  cuts variance sharply. The critic replaces the Monte Carlo return")
    print("  with a bootstrapped estimate — the MC-vs-TD trade-off again. PPO")
    print("  adds a trust region so the SAME batch can be reused for several")
    print("  epochs without the policy collapsing.")


def demo_ppo_clip():
    _hdr("8. THE PPO CLIP — what the objective actually does")
    eps = 0.2
    print("  L = min( r*A , clip(r, 1-eps, 1+eps)*A ),  eps = 0.2\n")
    print(f"  {'ratio r':>9}{'A=+1':>12}{'A=-1':>12}   interpretation")
    for r in [0.5, 0.8, 0.95, 1.0, 1.05, 1.2, 1.5]:
        lp = min(r * 1.0, max(1 - eps, min(1 + eps, r)) * 1.0)
        ln = min(r * -1.0, max(1 - eps, min(1 + eps, r)) * -1.0)
        note = ""
        if r > 1 + eps:
            note = "good action: gain CAPPED, no push further up"
        elif r < 1 - eps:
            note = "bad action: penalty CAPPED, no push further down"
        else:
            note = "inside the trust region: normal gradient"
        print(f"  {r:>9.2f}{lp:>12.2f}{ln:>12.2f}   {note}")
    print("\n  Outside the band the objective is flat, so its gradient is ZERO —")
    print("  the policy cannot run away from the data that was collected under")
    print("  pi_old. That is what makes multiple epochs on one batch safe.")


def demo_bandits():
    _hdr("6. BANDITS — exploration strategies and REGRET")
    probs = [0.2, 0.5, 0.75]
    print(f"  3 arms with payout probabilities {probs}; 1500 pulls.")
    print(f"  Regret = 1500 * {max(probs)} - (reward actually collected).\n")
    print(f"  {'strategy':<14}{'reward':>9}{'regret':>10}{'% pulls on best arm':>22}")
    for st in ["eps", "ucb", "thompson", "gradient"]:
        b = BernoulliBandit(probs, seed=99)     # env seed != agent seed!
        total, curve, counts = run_bandit(b, st, 1500, alpha=0.2, seed=1)
        print(f"  {st:<14}{total:>9.0f}{curve[-1]:>10.1f}"
              f"{100*counts[2]/sum(counts):>21.1f}%")
    print("\n  eps-greedy never stops exploring, so its regret grows LINEARLY.")
    print("  UCB1's bonus sqrt(2 ln t / n_a) shrinks as an arm is pulled, giving")
    print("  O(log t) regret. Thompson sampling matches the posterior")
    print("  probability that each arm is best and usually wins in practice.")
    print("  (Negative regret just means this run got luckier than the expected")
    print("  payout of always playing the best arm.)")


if __name__ == "__main__":
    print("=" * 74)
    print(" REINFORCEMENT LEARNING FROM SCRATCH — pure Python, no libraries")
    print("=" * 74)
    demo_dp()
    demo_prediction()
    demo_cliff()
    demo_dyna()
    demo_exploration()
    demo_bandits()
    demo_policy_gradient()
    demo_ppo_clip()
    demo_function_approx()
    print("\n" + "=" * 74)
    print("Every algorithm implemented and trained with zero external libraries.")
    print("=" * 74)
