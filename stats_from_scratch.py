"""
================================================================================
 PROBABILITY & STATISTICS FROM SCRATCH — pure Python, standard library only
================================================================================
 Only `math` and `random`. Every distribution function, test, and interval is
 implemented here and then CHECKED BY SIMULATION: an interval claiming 95%
 coverage is run thousands of times to see whether it really covers 95% of the
 time; a test claiming a 5% false-positive rate is run on data where the null
 is true to see how often it fires.

 That is the point of this file. Most statistics mistakes are not arithmetic
 errors; they are claims used outside the conditions that make them true.
 Simulation shows you where the conditions bite.

 CONTENTS
 --------
 0. Special functions ..... normal CDF/quantile, log-gamma, regularized
                            incomplete beta and gamma, Student t, chi-square,
                            F distributions (CDFs and quantiles)
 1. Samplers .............. inverse-CDF, Box-Muller, Poisson, binomial,
                            geometric, Cauchy, and friends
 2. Probability ........... Bayes and base rates, Monty Hall, birthday problem,
                            coupon collector, gambler's ruin, linearity of
                            expectation, Markov-chain stationary distributions
 3. Limit theorems ........ LLN, CLT, and where the CLT FAILS (Cauchy)
 4. Estimation ............ biased vs unbiased variance, MLE, shrinkage and the
                            bias-variance trade-off, bootstrap standard errors
 5. Confidence intervals .. z vs t, Wald vs Wilson for proportions, bootstrap
                            percentile, coverage checked by simulation
 6. Hypothesis tests ...... one-sample / Welch / paired t, chi-square
                            independence, Mann-Whitney U, permutation test,
                            p-value calibration, power curves
 7. Multiple testing ...... Bonferroni vs Benjamini-Hochberg (FWER vs FDR),
                            the base-rate problem for "significant" results
 8. Regression inference .. OLS standard errors, heteroscedasticity and robust
                            (HC) errors, multicollinearity and VIF, omitted
                            variable bias, regression to the mean
 9. Bayesian inference .... Beta-Binomial updating, credible intervals,
                            Bayesian A/B comparison
 10. Causal inference ..... Simpson's paradox, confounding, stratification,
                            inverse propensity weighting, difference-in-
                            differences
 11. Demos ................ every claim above, checked by simulation
================================================================================
"""

import math
import random

EPS = 1e-12


# =============================================================================
# 0. SPECIAL FUNCTIONS
# =============================================================================
def normal_cdf(x, mu=0.0, sigma=1.0):
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def normal_pdf(x, mu=0.0, sigma=1.0):
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


def _bisect(f, target, lo, hi, iters=200):
    """Invert a monotone increasing function by bisection."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def normal_ppf(p, mu=0.0, sigma=1.0):
    return mu + sigma * _bisect(normal_cdf, p, -40.0, 40.0)


def _betacf(a, b, x, max_iter=500):
    """Continued fraction for the incomplete beta (modified Lentz method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return h


def reg_inc_beta(x, a, b):
    """
    Regularized incomplete beta I_x(a, b) = B(x; a, b) / B(a, b).
    The workhorse behind the Student t, F, and binomial CDFs. Uses the
    continued fraction directly when it converges fast, else the symmetry
    I_x(a,b) = 1 - I_(1-x)(b,a).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
           + a * math.log(x) + b * math.log(1 - x))
    bt = math.exp(lbt)
    if x < (a + 1) / (a + b + 2):
        return bt * _betacf(a, b, x) / a
    return 1 - bt * _betacf(b, a, 1 - x) / b


def reg_inc_gamma_lower(a, x):
    """
    Regularized lower incomplete gamma P(a, x) — the chi-square and Poisson
    CDFs are special cases. Series expansion for x < a + 1, continued
    fraction otherwise (the standard split for fast convergence).
    """
    if x <= 0:
        return 0.0
    if x < a + 1:
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1 / (d if abs(d) > tiny else tiny)
        c = b + an / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return 1 - math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def t_cdf(t, df):
    """Student t CDF via  P(|T| > |t|) = I_{df/(df+t^2)}(df/2, 1/2)."""
    x = df / (df + t * t)
    tail = 0.5 * reg_inc_beta(x, df / 2, 0.5)
    return 1 - tail if t > 0 else tail


def t_ppf(p, df):
    return _bisect(lambda t: t_cdf(t, df), p, -1e3, 1e3)


def chi2_cdf(x, k):
    return reg_inc_gamma_lower(k / 2, x / 2)


def chi2_ppf(p, k):
    return _bisect(lambda x: chi2_cdf(x, k), p, 0.0, 1e4)


def f_cdf(x, d1, d2):
    if x <= 0:
        return 0.0
    return reg_inc_beta(d1 * x / (d1 * x + d2), d1 / 2, d2 / 2)


def binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p), via the incomplete beta identity."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return reg_inc_beta(1 - p, n - k, k + 1)


def binom_pmf(k, n, p):
    if k < 0 or k > n:
        return 0.0
    return math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                    + k * math.log(p) + (n - k) * math.log(1 - p)) if 0 < p < 1 else \
        float((p == 0 and k == 0) or (p == 1 and k == n))


def poisson_pmf(k, lam):
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) if lam > 0 else float(k == 0)


# =============================================================================
# 1. SAMPLERS
# =============================================================================
def sample_exponential(rng, lam=1.0):
    """
    INVERSE-CDF SAMPLING: if U ~ Uniform(0,1) then F^-1(U) has CDF F.
    Exponential: F(x) = 1 - e^(-lam x)  =>  x = -ln(1 - U) / lam.
    Works for any distribution whose CDF you can invert.
    """
    return -math.log(1 - rng.random()) / lam


def sample_normal_box_muller(rng, mu=0.0, sigma=1.0):
    """
    BOX-MULLER: two independent uniforms -> a standard normal.
        R = sqrt(-2 ln U1),  theta = 2 pi U2,  Z = R cos(theta)
    (R sin(theta) is a second, independent normal, discarded here.)
    """
    u1, u2 = 1 - rng.random(), rng.random()
    return mu + sigma * math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def sample_poisson(rng, lam):
    """Knuth's method: count uniforms until their product drops below e^-lam.
    O(lam) per draw — fine for small lam; large lam needs other methods."""
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p < L:
            return k
        k += 1


def sample_binomial(rng, n, p):
    return sum(1 for _ in range(n) if rng.random() < p)


def sample_geometric(rng, p):
    """Number of trials up to and including the first success (support 1, 2, ...)."""
    return int(math.ceil(math.log(1 - rng.random()) / math.log(1 - p))) or 1


def sample_cauchy(rng):
    """Standard Cauchy = tan(pi (U - 1/2)). Heavy-tailed: NO finite mean or
    variance. The canonical counterexample to naive use of the CLT."""
    return math.tan(math.pi * (rng.random() - 0.5))


def mean(xs):
    return sum(xs) / len(xs)


def var(xs, ddof=1):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / (len(xs) - ddof)


def sd(xs, ddof=1):
    return math.sqrt(var(xs, ddof))


def quantile(xs, q):
    s = sorted(xs)
    k = (len(s) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


# =============================================================================
# 2. PROBABILITY — exact answers, checked by simulation
# =============================================================================
def bayes_test(prevalence, sensitivity, specificity):
    """
    P(disease | positive) = P(pos | D) P(D) / P(pos)
        P(pos) = sens * prev + (1 - spec) * (1 - prev)
    THE BASE-RATE FALLACY: with a rare condition, even an accurate test yields
    mostly false positives, because the healthy population is so much larger.
    """
    p_pos = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    return sensitivity * prevalence / p_pos


def simulate_bayes_test(prevalence, sensitivity, specificity, n=200000, seed=0):
    rng = random.Random(seed)
    tp = pos = 0
    for _ in range(n):
        sick = rng.random() < prevalence
        positive = rng.random() < (sensitivity if sick else 1 - specificity)
        if positive:
            pos += 1
            tp += sick
    return tp / pos


def simulate_monty_hall(n=100000, seed=0):
    """
    Three doors, one car. You pick a door; the host, who KNOWS where the car
    is, opens a different door with a goat, then offers a switch.
    Stay wins 1/3, switch wins 2/3. The host's choice is informative because it
    is constrained: he never opens your door or the car's door.
    (If the host opened a door at random and it merely happened to show a goat,
    switching would win only 1/2 — the protocol matters, not just the outcome.)
    """
    rng = random.Random(seed)
    stay = switch = 0
    for _ in range(n):
        car, pick = rng.randrange(3), rng.randrange(3)
        opened = rng.choice([d for d in range(3) if d != pick and d != car])
        other = next(d for d in range(3) if d != pick and d != opened)
        stay += pick == car
        switch += other == car
    return stay / n, switch / n


def birthday_exact(k, days=365):
    """P(at least two of k people share a birthday) = 1 - prod (days-i)/days.
    Assumes birthdays uniform and independent (real ones are neither, which
    slightly RAISES the collision probability)."""
    p_distinct = 1.0
    for i in range(k):
        p_distinct *= (days - i) / days
    return 1 - p_distinct


def simulate_birthday(k, n=40000, seed=0):
    rng = random.Random(seed)
    hits = 0
    for _ in range(n):
        seen = set()
        for _ in range(k):
            d = rng.randrange(365)
            if d in seen:
                hits += 1
                break
            seen.add(d)
    return hits / n


def coupon_collector_exact(n):
    """E[draws to collect all n coupons] = n * H_n = n (1 + 1/2 + ... + 1/n).
    By LINEARITY OF EXPECTATION over the waiting time for each new coupon: when
    i coupons are held, a new one appears w.p. (n-i)/n, so the wait is
    geometric with mean n/(n-i)."""
    return n * sum(1 / i for i in range(1, n + 1))


def simulate_coupon(n, reps=5000, seed=0):
    rng = random.Random(seed)
    total = 0
    for _ in range(reps):
        seen, draws = set(), 0
        while len(seen) < n:
            seen.add(rng.randrange(n))
            draws += 1
        total += draws
    return total / reps


def fixed_points_expectation(n, reps=20000, seed=0):
    """
    Expected number of fixed points of a random permutation of n items is
    EXACTLY 1, for every n. Proof: X = sum of indicators 1[pi(i) = i], each
    with probability 1/n, so E[X] = n * (1/n) = 1 — by linearity, which holds
    even though the indicators are DEPENDENT. The 'hat-check problem'.
    """
    rng = random.Random(seed)
    total = 0
    items = list(range(n))
    for _ in range(reps):
        p = items[:]
        rng.shuffle(p)
        total += sum(1 for i, v in enumerate(p) if i == v)
    return total / reps


def gamblers_ruin_exact(start, target, p):
    """
    P(reach `target` before 0 | start), betting 1 unit with win prob p.
      p = 1/2 : start / target
      else    : (1 - r^start) / (1 - r^target),  r = (1-p)/p
    Even a small house edge makes reaching a distant target very unlikely.
    """
    if abs(p - 0.5) < 1e-12:
        return start / target
    r = (1 - p) / p
    return (1 - r ** start) / (1 - r ** target)


def simulate_gamblers_ruin(start, target, p, reps=20000, seed=0):
    rng = random.Random(seed)
    wins = 0
    for _ in range(reps):
        x = start
        while 0 < x < target:
            x += 1 if rng.random() < p else -1
        wins += x == target
    return wins / reps


def stationary_distribution(P, iters=2000):
    """
    Stationary distribution pi of a Markov chain: pi = pi P, sum(pi) = 1.
    Found here by power iteration (repeatedly multiplying a start vector by P).
    Exists and is unique for an irreducible chain; power iteration CONVERGES to
    it when the chain is also aperiodic. It equals the long-run fraction of
    time spent in each state — the idea behind PageRank and MCMC.
    """
    n = len(P)
    pi = [1.0 / n] * n
    for _ in range(iters):
        pi = [sum(pi[i] * P[i][j] for i in range(n)) for j in range(n)]
    return pi


def simulate_chain(P, steps=200000, seed=0):
    rng = random.Random(seed)
    s, counts = 0, [0] * len(P)
    for _ in range(steps):
        r, acc = rng.random(), 0.0
        for j, pj in enumerate(P[s]):
            acc += pj
            if r <= acc:
                s = j
                break
        counts[s] += 1
    return [c / steps for c in counts]


# =============================================================================
# 3. LIMIT THEOREMS
# =============================================================================
def skewness(xs):
    m, s = mean(xs), sd(xs, 0)
    return sum(((x - m) / s) ** 3 for x in xs) / len(xs) if s > 0 else 0.0


def clt_check(sampler, n, reps=4000, seed=0):
    """
    Simulate `reps` sample means of size n and report:
      * skewness of the sample means — 0 for a normal distribution. For
        Exponential data it should fall like 2/sqrt(n): the CLT is converging,
        but slowly for skewed data.
      * the interquartile range of the sample means — shrinks like 1/sqrt(n)
        when the CLT applies. For Cauchy data it does NOT shrink at all: the
        mean of n Cauchy draws is again standard Cauchy, because the CLT needs a
        FINITE VARIANCE (and the LLN needs a finite mean) that Cauchy lacks.
    (A '|Z| > 1.96' tail check standardized by the sample's own spread is a poor
    diagnostic here: it hovers near 5% even for clearly non-normal data.)
    """
    rng = random.Random(seed)
    means = [mean([sampler(rng) for _ in range(n)]) for _ in range(reps)]
    iqr = quantile(means, 0.75) - quantile(means, 0.25)
    return quantile(means, 0.5), skewness(means), iqr


def running_mean_path(sampler, n, seed=0, checkpoints=(10, 100, 1000, 10000)):
    rng = random.Random(seed)
    total, out = 0.0, {}
    for i in range(1, n + 1):
        total += sampler(rng)
        if i in checkpoints:
            out[i] = total / i
    return out


# =============================================================================
# 4. ESTIMATION
# =============================================================================
def variance_estimator_bias(n=5, sigma=1.0, reps=40000, seed=0):
    """
    E[ (1/n) sum (x - xbar)^2 ]     = (n-1)/n * sigma^2    biased LOW
    E[ (1/(n-1)) sum (x - xbar)^2 ] = sigma^2              unbiased (Bessel)
    The bias comes from measuring spread around xbar, which is itself fitted
    to the data and so sits closer to the points than the true mean does.
    Note: the unbiased VARIANCE estimator does not make s an unbiased
    estimator of sigma (sqrt is concave, so E[s] < sigma by Jensen).
    """
    rng = random.Random(seed)
    mle = unb = sdev = 0.0
    for _ in range(reps):
        xs = [rng.gauss(0, sigma) for _ in range(n)]
        mle += var(xs, 0)
        unb += var(xs, 1)
        sdev += sd(xs, 1)
    return mle / reps, unb / reps, sdev / reps


def shrinkage_mse(true_mu=0.5, sigma=2.0, n=5, reps=20000, seed=0):
    """
    BIAS-VARIANCE for estimators: MSE = bias^2 + variance.
    Compare the sample mean (unbiased) with a shrunk estimate c * xbar. For
    the right c < 1 the shrunk estimator is BIASED yet has LOWER MSE when the
    true mean is small relative to the noise — the same logic as ridge
    regression and the James-Stein phenomenon. Unbiased is not the same as best.
    """
    rng = random.Random(seed)
    out = {}
    for c in [1.0, 0.8, 0.6, 0.4]:
        errs = []
        for _ in range(reps):
            xbar = mean([rng.gauss(true_mu, sigma) for _ in range(n)])
            errs.append(c * xbar - true_mu)
        b = mean(errs)
        out[c] = (b, var(errs, 0), mean([e * e for e in errs]))
    return out


def bootstrap_se(data, stat, B=2000, seed=0):
    """
    BOOTSTRAP standard error: resample the data WITH replacement B times,
    recompute the statistic, take the standard deviation. Works for
    statistics with no closed-form SE (median, correlation, ratios). It
    assumes the sample is representative and i.i.d.; it fails for statistics
    that depend on extremes (e.g. the sample maximum) and for dependent data
    (use block bootstraps for time series).
    """
    rng = random.Random(seed)
    n = len(data)
    reps = [stat([data[rng.randrange(n)] for _ in range(n)]) for _ in range(B)]
    return sd(reps), reps


# =============================================================================
# 5. CONFIDENCE INTERVALS
# =============================================================================
def ci_mean_z(xs, alpha=0.05):
    """Uses the normal quantile with the SAMPLE sd: only valid for large n."""
    z = normal_ppf(1 - alpha / 2)
    m, s = mean(xs), sd(xs)
    h = z * s / math.sqrt(len(xs))
    return m - h, m + h


def ci_mean_t(xs, alpha=0.05):
    """Student t interval: exact for normal data at any n; the t quantile
    accounts for the extra uncertainty from estimating sigma."""
    t = t_ppf(1 - alpha / 2, len(xs) - 1)
    m, s = mean(xs), sd(xs)
    h = t * s / math.sqrt(len(xs))
    return m - h, m + h


def ci_prop_wald(k, n, alpha=0.05):
    """p_hat +/- z sqrt(p_hat(1-p_hat)/n). The textbook interval — and a poor
    one near 0 or 1 or at small n (at k = 0 it has ZERO width)."""
    z = normal_ppf(1 - alpha / 2)
    p = k / n
    h = z * math.sqrt(p * (1 - p) / n)
    return p - h, p + h


def ci_prop_wilson(k, n, alpha=0.05):
    """
    WILSON score interval: invert the score test instead of plugging p_hat into
    the standard error. Much better coverage for small n and extreme p;
    recommended over Wald by Brown, Cai & DasGupta (2001).
    """
    z = normal_ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - h, centre + h


def ci_bootstrap_percentile(xs, stat, B=1000, alpha=0.05, seed=0):
    _, reps = bootstrap_se(xs, stat, B, seed)
    return quantile(reps, alpha / 2), quantile(reps, 1 - alpha / 2)


def coverage(interval_fn, sampler, true_value, n, reps=4000, seed=0):
    """Fraction of intervals that contain the true value — the ONLY thing a
    confidence level promises, and it is a promise about the PROCEDURE over
    repeated samples, not about any one computed interval."""
    rng = random.Random(seed)
    hit = 0
    widths = []
    for _ in range(reps):
        xs = [sampler(rng) for _ in range(n)]
        lo, hi = interval_fn(xs)
        hit += lo <= true_value <= hi
        widths.append(hi - lo)
    return hit / reps, mean(widths)


# =============================================================================
# 6. HYPOTHESIS TESTS
# =============================================================================
def ttest_one_sample(xs, mu0=0.0):
    n = len(xs)
    t = (mean(xs) - mu0) / (sd(xs) / math.sqrt(n))
    return t, 2 * (1 - t_cdf(abs(t), n - 1))


def ttest_welch(a, b):
    """
    WELCH's t-test: does NOT assume equal variances. Degrees of freedom from the
    Welch-Satterthwaite approximation. Use it by default — Student's pooled
    test inflates false positives when variances AND sample sizes both differ.
    """
    va, vb = var(a) / len(a), var(b) / len(b)
    t = (mean(a) - mean(b)) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t, 2 * (1 - t_cdf(abs(t), df)), df


def ttest_student_pooled(a, b):
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * var(a) + (nb - 1) * var(b)) / (na + nb - 2)
    t = (mean(a) - mean(b)) / math.sqrt(sp2 * (1 / na + 1 / nb))
    return t, 2 * (1 - t_cdf(abs(t), na + nb - 2))


def ttest_paired(a, b):
    """Paired t = one-sample t on the differences. Pairing removes the
    between-subject variation, often giving far more power than treating the
    two columns as independent groups."""
    return ttest_one_sample([x - y for x, y in zip(a, b)])


def chi2_independence(table):
    """
    Pearson chi-square test of independence for an r x c table:
        E_ij = row_i * col_j / N ,   X^2 = sum (O - E)^2 / E ,  df = (r-1)(c-1)
    The chi-square approximation needs expected counts that are not too small
    (a common rule of thumb: all E_ij >= 5); otherwise use Fisher's exact test.
    """
    r, c = len(table), len(table[0])
    rows = [sum(row) for row in table]
    cols = [sum(table[i][j] for i in range(r)) for j in range(c)]
    N = sum(rows)
    x2 = 0.0
    min_e = float("inf")
    for i in range(r):
        for j in range(c):
            e = rows[i] * cols[j] / N
            min_e = min(min_e, e)
            x2 += (table[i][j] - e) ** 2 / e
    df = (r - 1) * (c - 1)
    return x2, 1 - chi2_cdf(x2, df), df, min_e


def mann_whitney_u(a, b):
    """
    MANN-WHITNEY U (Wilcoxon rank-sum), normal approximation with tie-averaged
    ranks. It tests whether one group tends to produce larger values —
    P(A > B) != 1/2 — NOT, in general, whether the medians differ (that
    interpretation needs the two distributions to have the same shape).
    Robust to outliers and heavy tails, where the t-test loses power.
    """
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    ra = sum(r for r, (_, g) in zip(ranks, allv) if g == 0)
    na, nb = len(a), len(b)
    U = ra - na * (na + 1) / 2
    mu = na * nb / 2
    sigma = math.sqrt(na * nb * (na + nb + 1) / 12)
    z = (U - mu) / sigma
    return U, z, 2 * (1 - normal_cdf(abs(z)))


def permutation_test(a, b, stat=None, n_perm=2000, seed=0):
    """
    PERMUTATION TEST: under H0 (group labels are exchangeable), shuffling the
    labels should not change the statistic's distribution. p = fraction of
    shuffles at least as extreme as the observed statistic (with +1 in numerator
    and denominator so p is never exactly 0). Almost assumption-free, works for
    any statistic, and exact up to Monte Carlo error.
    """
    stat = stat or (lambda x, y: mean(x) - mean(y))
    rng = random.Random(seed)
    obs = abs(stat(a, b))
    pooled = list(a) + list(b)
    na = len(a)
    extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(stat(pooled[:na], pooled[na:])) >= obs - 1e-12:
            extreme += 1
    return obs, (extreme + 1) / (n_perm + 1)


def false_positive_rate(test, sampler_a, sampler_b, na, nb, reps=3000, alpha=0.05, seed=0):
    """Run a test many times on data where H0 is TRUE; the rejection rate
    should equal alpha. If it doesn't, the test's assumptions are violated."""
    rng = random.Random(seed)
    rej = 0
    for _ in range(reps):
        a = [sampler_a(rng) for _ in range(na)]
        b = [sampler_b(rng) for _ in range(nb)]
        rej += test(a, b) < alpha
    return rej / reps


def power_sim(test, effect, n, sigma=1.0, reps=2000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    rej = 0
    for _ in range(reps):
        a = [rng.gauss(0, sigma) for _ in range(n)]
        b = [rng.gauss(effect, sigma) for _ in range(n)]
        rej += test(a, b) < alpha
    return rej / reps


def power_two_sample_analytic(effect, n, sigma=1.0, alpha=0.05):
    """Normal-approximation power for a two-sided two-sample test with n per
    group:  power ~ Phi( |delta| / (sigma sqrt(2/n)) - z_(1-alpha/2) )."""
    z = normal_ppf(1 - alpha / 2)
    return normal_cdf(abs(effect) / (sigma * math.sqrt(2 / n)) - z)


def n_per_group(effect, sigma=1.0, alpha=0.05, power=0.8):
    """n = 2 (z_(1-a/2) + z_(1-b))^2 sigma^2 / delta^2 per group."""
    za, zb = normal_ppf(1 - alpha / 2), normal_ppf(power)
    return math.ceil(2 * (za + zb) ** 2 * sigma ** 2 / effect ** 2)


# =============================================================================
# 7. MULTIPLE TESTING
# =============================================================================
def bonferroni(pvals, alpha=0.05):
    """Reject p_i <= alpha / m. Controls the FAMILY-WISE ERROR RATE (probability
    of ANY false rejection) under any dependence. Conservative: loses power
    fast as m grows."""
    m = len(pvals)
    return [p <= alpha / m for p in pvals]


def benjamini_hochberg(pvals, q=0.05):
    """
    BENJAMINI-HOCHBERG: sort p-values, find the largest k with
    p_(k) <= (k/m) q, reject the k smallest. Controls the FALSE DISCOVERY RATE
    — the expected FRACTION of rejections that are false — at q, for
    independent (or positively dependent) tests. Far more power than
    Bonferroni when many hypotheses are truly non-null.
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    kmax = 0
    for rank, i in enumerate(order, 1):
        if pvals[i] <= rank / m * q:
            kmax = rank
    rejected = set(order[:kmax])
    return [i in rejected for i in range(m)]


def simulate_multiple_testing(m=200, frac_true=0.1, effect=0.8, n=20, reps=200, seed=0):
    """
    m hypotheses per 'study', a fraction truly non-null. Each is a two-sample
    Welch t-test with n per group. Compare three decision rules on:
      FWER  P(at least one false rejection)
      FDR   E[ false rejections / max(total rejections, 1) ]
      power fraction of true effects detected
    """
    rng = random.Random(seed)
    stats = {k: {"fwer": 0, "fdp": 0.0, "power": 0.0} for k in ("none", "bonf", "bh")}
    for _ in range(reps):
        truth, pvals = [], []
        for i in range(m):
            real = rng.random() < frac_true
            a = [rng.gauss(0, 1) for _ in range(n)]
            b = [rng.gauss(effect if real else 0, 1) for _ in range(n)]
            truth.append(real)
            pvals.append(ttest_welch(a, b)[1])
        rules = {"none": [p < 0.05 for p in pvals], "bonf": bonferroni(pvals),
                 "bh": benjamini_hochberg(pvals)}
        n_true = max(sum(truth), 1)
        for k, rej in rules.items():
            fp = sum(1 for r, t in zip(rej, truth) if r and not t)
            tp = sum(1 for r, t in zip(rej, truth) if r and t)
            stats[k]["fwer"] += fp > 0
            stats[k]["fdp"] += fp / max(fp + tp, 1)
            stats[k]["power"] += tp / n_true
    return {k: {kk: vv / reps for kk, vv in v.items()} for k, v in stats.items()}


def prob_h0_given_significant(prior_true, power, alpha=0.05):
    """
    What fraction of 'significant' results are FALSE positives?
        P(H0 | p < alpha) = alpha (1 - pi) / ( alpha (1 - pi) + power * pi )
    where pi is the fraction of tested hypotheses that are actually true
    effects. A p-value is P(data this extreme | H0), NOT P(H0 | data). With
    low prior plausibility and low power, most 'discoveries' are false even at
    p < 0.05 — the core of the replication crisis argument (Ioannidis 2005).
    """
    fp = alpha * (1 - prior_true)
    tp = power * prior_true
    return fp / (fp + tp)


# =============================================================================
# 8. REGRESSION INFERENCE
# =============================================================================
def _solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def _inv(A):
    n = len(A)
    cols = [_solve(A, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def ols(X, y, robust=False):
    """
    ORDINARY LEAST SQUARES WITH INFERENCE. X gets an intercept column.

        beta = (X'X)^-1 X'y
        classic SE : Var(beta) = s^2 (X'X)^-1,   s^2 = RSS / (n - p)
        HC1 robust : Var(beta) = n/(n-p) (X'X)^-1 [ sum e_i^2 x_i x_i' ] (X'X)^-1

    The classic formula ASSUMES HOMOSCEDASTICITY (constant error variance).
    When the variance depends on x, the coefficient estimates stay unbiased
    but the classic standard errors are wrong — usually too small — so
    confidence intervals under-cover and p-values are too optimistic. The
    'sandwich' (Huber-White / HC) estimator does not need that assumption.

    Returns (beta, standard errors, t statistics, p values, R^2).
    """
    Xc = [[1.0] + list(r) for r in X]
    n, p = len(Xc), len(Xc[0])
    XtX = [[sum(Xc[k][i] * Xc[k][j] for k in range(n)) for j in range(p)] for i in range(p)]
    Xty = [sum(Xc[k][i] * y[k] for k in range(n)) for i in range(p)]
    beta = _solve(XtX, Xty)
    resid = [y[k] - sum(beta[i] * Xc[k][i] for i in range(p)) for k in range(n)]
    rss = sum(e * e for e in resid)
    ybar = mean(y)
    r2 = 1 - rss / sum((v - ybar) ** 2 for v in y)
    XtXi = _inv(XtX)
    if robust:
        meat = [[sum(resid[k] ** 2 * Xc[k][i] * Xc[k][j] for k in range(n))
                 for j in range(p)] for i in range(p)]
        left = [[sum(XtXi[i][a] * meat[a][j] for a in range(p)) for j in range(p)]
                for i in range(p)]
        V = [[sum(left[i][a] * XtXi[a][j] for a in range(p)) * n / (n - p)
              for j in range(p)] for i in range(p)]
    else:
        s2 = rss / (n - p)
        V = [[s2 * XtXi[i][j] for j in range(p)] for i in range(p)]
    se = [math.sqrt(max(V[i][i], 0.0)) for i in range(p)]
    tstat = [b / s if s > 0 else float("nan") for b, s in zip(beta, se)]
    pval = [2 * (1 - t_cdf(abs(t), n - p)) for t in tstat]
    return beta, se, tstat, pval, r2


def slope_ci_coverage(hetero, robust, n=60, reps=1500, true_slope=2.0, seed=0):
    """Does the 95% CI for the slope cover the truth 95% of the time?"""
    rng = random.Random(seed)
    hit = 0
    for _ in range(reps):
        X, y = [], []
        for _ in range(n):
            x = rng.uniform(0, 3)
            noise_sd = (0.2 + 1.5 * x * x) if hetero else 1.0
            X.append([x])
            y.append(1.0 + true_slope * x + rng.gauss(0, noise_sd))
        beta, se, *_ = ols(X, y, robust=robust)
        t = t_ppf(0.975, n - 2)
        hit += beta[1] - t * se[1] <= true_slope <= beta[1] + t * se[1]
    return hit / reps


def vif(X):
    """VARIANCE INFLATION FACTOR for each column: 1 / (1 - R_j^2), where R_j^2 is
    from regressing column j on the others. VIF = 1 means no collinearity; the
    standard error of beta_j is inflated by a factor sqrt(VIF)."""
    out = []
    d = len(X[0])
    for j in range(d):
        yj = [r[j] for r in X]
        others = [[r[k] for k in range(d) if k != j] for r in X]
        r2 = ols(others, yj)[4] if others[0] else 0.0
        out.append(1 / max(1 - r2, 1e-12))
    return out


def omitted_variable_bias(n=2000, seed=0):
    """
    OMITTED VARIABLE BIAS. True model: y = 1 + 2 x + 3 z + e, with x and z
    correlated. Regressing y on x alone gives
        E[b_x] = 2 + 3 * Cov(x, z) / Var(x)
    The omitted variable's effect is absorbed into x's coefficient. No amount
    of data fixes it — the estimate converges to the WRONG number. This is
    confounding, in regression form.
    """
    rng = random.Random(seed)
    X, Z, Y = [], [], []
    for _ in range(n):
        z = rng.gauss(0, 1)
        x = 0.7 * z + rng.gauss(0, 0.7)
        X.append(x)
        Z.append(z)
        Y.append(1 + 2 * x + 3 * z + rng.gauss(0, 1))
    short = ols([[x] for x in X], Y)[0][1]
    full = ols([[x, z] for x, z in zip(X, Z)], Y)[0][1]
    mx, mz = mean(X), mean(Z)
    cov_xz = sum((x - mx) * (z - mz) for x, z in zip(X, Z)) / (n - 1)
    predicted = 2 + 3 * cov_xz / var(X)
    return short, full, predicted


def regression_to_the_mean(n=5000, reliability=0.5, seed=0):
    """
    REGRESSION TO THE MEAN. Test score = true ability + noise. Take the top 10%
    on test 1 and retest them: their average falls, with NO intervention,
    because part of their extreme first score was luck that does not repeat.
    Any 'treat the worst performers, then see them improve' study without a
    control group is contaminated by this.
    """
    rng = random.Random(seed)
    s_true = math.sqrt(reliability)
    s_noise = math.sqrt(1 - reliability)
    people = []
    for _ in range(n):
        a = rng.gauss(0, s_true)
        people.append((a + rng.gauss(0, s_noise), a + rng.gauss(0, s_noise)))
    cut = quantile([p[0] for p in people], 0.9)
    top = [p for p in people if p[0] >= cut]
    bottom_cut = quantile([p[0] for p in people], 0.1)
    bottom = [p for p in people if p[0] <= bottom_cut]
    return (mean([p[0] for p in top]), mean([p[1] for p in top]),
            mean([p[0] for p in bottom]), mean([p[1] for p in bottom]))


# =============================================================================
# 9. BAYESIAN INFERENCE
# =============================================================================
def beta_posterior(successes, trials, a0=1.0, b0=1.0):
    """
    Beta-Binomial CONJUGACY: prior Beta(a, b) + k successes in n trials
    -> posterior Beta(a + k, b + n - k). The prior acts like a + b - 2 'pseudo-
    observations'. With a flat Beta(1,1) prior the posterior mean is
    (k + 1)/(n + 2) — Laplace's rule of succession.
    """
    return a0 + successes, b0 + trials - successes


def beta_ppf(p, a, b):
    return _bisect(lambda x: reg_inc_beta(x, a, b), p, 0.0, 1.0)


def beta_sample(rng, a, b):
    x, y = rng.gammavariate(a, 1), rng.gammavariate(b, 1)
    return x / (x + y)


def bayesian_ab(conv_a, n_a, conv_b, n_b, draws=40000, seed=0):
    """
    P(p_B > p_A | data) under independent Beta(1,1) priors, by Monte Carlo from
    the two posteriors, plus the expected lift. This answers the question
    stakeholders usually THINK a p-value answers ('how likely is B better?') —
    but the number depends on the prior, and repeatedly checking it and
    stopping when it crosses a threshold still inflates error rates.
    """
    rng = random.Random(seed)
    aA, bA = beta_posterior(conv_a, n_a)
    aB, bB = beta_posterior(conv_b, n_b)
    wins = 0
    lifts = []
    for _ in range(draws):
        pa, pb = beta_sample(rng, aA, bA), beta_sample(rng, aB, bB)
        wins += pb > pa
        lifts.append(pb - pa)
    return wins / draws, mean(lifts), quantile(lifts, 0.025), quantile(lifts, 0.975)


# =============================================================================
# 10. CAUSAL INFERENCE
# =============================================================================
KIDNEY_STONES = {
    # Charig et al. (1986), BMJ — the classic Simpson's paradox dataset.
    # (successes, patients) by treatment and stone size.
    ("A", "small"): (81, 87), ("A", "large"): (192, 263),
    ("B", "small"): (234, 270), ("B", "large"): (55, 80),
}


def simpsons_paradox(data=KIDNEY_STONES):
    """
    Treatment A wins WITHIN each stone size, yet B wins in the AGGREGATE.
    Why: stone size is a confounder. Doctors gave A (open surgery) mostly to
    the hard, large-stone cases, which have lower success rates regardless of
    treatment. Aggregating mixes groups of very different difficulty in
    different proportions.

    Which answer is right depends on the CAUSAL structure, not on the numbers:
    if size causes treatment choice (a confounder), condition on it — A is
    better. If a variable is a consequence of treatment (a mediator), you
    generally should NOT condition on it. Data alone cannot tell you which.
    """
    out = {}
    for t in ("A", "B"):
        tot_s = tot_n = 0
        for size in ("small", "large"):
            s, n = data[(t, size)]
            out[(t, size)] = s / n
            tot_s += s
            tot_n += n
        out[(t, "all")] = tot_s / tot_n
    return out


def make_confounded_data(n=4000, true_effect=1.0, seed=0):
    """
    Observational data with a CONFOUNDER: severity z raises the chance of being
    treated AND worsens the outcome directly. The true treatment effect is +1.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        z = rng.gauss(0, 1)
        p_treat = 1 / (1 + math.exp(-(1.5 * z)))
        t = 1 if rng.random() < p_treat else 0
        y = true_effect * t - 2.0 * z + rng.gauss(0, 1)
        rows.append((z, t, y))
    return rows


def _logistic_fit(X, y, epochs=400, lr=0.5):
    d = len(X[0])
    w, b = [0.0] * d, 0.0
    n = len(X)
    for _ in range(epochs):
        gw, gb = [0.0] * d, 0.0
        for x, t in zip(X, y):
            z = sum(a * c for a, c in zip(w, x)) + b
            p = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
            for j in range(d):
                gw[j] += (p - t) * x[j]
            gb += p - t
        w = [wj - lr * g / n for wj, g in zip(w, gw)]
        b -= lr * gb / n
    return w, b


def estimate_effects(rows, n_strata=10):
    """
    Four estimates of the same causal effect (truth = +1):

      naive difference in means    E[y|t=1] - E[y|t=0]      -> biased by z
      regression adjustment        coefficient on t in y ~ t + z
      stratification               difference within quantile bins of z,
                                   averaged by bin size (residual bias from
                                   coarse bins)
      inverse propensity weighting weight each unit by 1/P(t | z) so the
                                   treated and control groups resemble the
                                   whole population

    All three adjustments rely on the SAME untestable assumption: no
    unmeasured confounding (every common cause of t and y is in z). They also
    need OVERLAP — both treated and untreated units at every z. IPW here uses
    self-normalized (Hajek) weights for stability.
    """
    y1 = [y for z, t, y in rows if t == 1]
    y0 = [y for z, t, y in rows if t == 0]
    naive = mean(y1) - mean(y0)
    reg = ols([[t, z] for z, t, y in rows], [y for _, _, y in rows])[0][1]
    zs = sorted(r[0] for r in rows)
    edges = [zs[int(len(zs) * k / n_strata)] for k in range(1, n_strata)]

    def stratum(z):
        k = 0
        while k < len(edges) and z > edges[k]:
            k += 1
        return k
    groups = {}
    for z, t, y in rows:
        groups.setdefault(stratum(z), []).append((t, y))
    strat_num = strat_den = 0.0
    for g in groups.values():
        a = [y for t, y in g if t == 1]
        b = [y for t, y in g if t == 0]
        if a and b:
            strat_num += len(g) * (mean(a) - mean(b))
            strat_den += len(g)
    strat = strat_num / strat_den
    w, b0 = _logistic_fit([[z] for z, _, _ in rows], [t for _, t, _ in rows])
    num1 = den1 = num0 = den0 = 0.0
    for z, t, y in rows:
        e = 1 / (1 + math.exp(-(w[0] * z + b0)))
        e = min(max(e, 0.01), 0.99)
        if t == 1:
            num1 += y / e
            den1 += 1 / e
        else:
            num0 += y / (1 - e)
            den0 += 1 / (1 - e)
    ipw = num1 / den1 - num0 / den0
    return naive, reg, strat, ipw


def difference_in_differences(n_units=400, true_effect=2.0, seed=0):
    """
    DIFFERENCE-IN-DIFFERENCES. Treated units start at a DIFFERENT level (a
    fixed group difference) and everyone shares a common time trend. Then

        DiD = (treated_after - treated_before) - (control_after - control_before)

    removes both the fixed group gap and the shared trend. The key assumption
    is PARALLEL TRENDS: without treatment, both groups would have moved by the
    same amount. A naive before/after comparison on the treated group alone
    mistakes the time trend for a treatment effect.
    """
    rng = random.Random(seed)
    tb, ta, cb, ca = [], [], [], []
    for i in range(n_units):
        treated = i < n_units // 2
        base = (5.0 if treated else 2.0) + rng.gauss(0, 1)
        trend = 1.5
        before = base + rng.gauss(0, 0.5)
        after = base + trend + (true_effect if treated else 0.0) + rng.gauss(0, 0.5)
        (tb if treated else cb).append(before)
        (ta if treated else ca).append(after)
    did = (mean(ta) - mean(tb)) - (mean(ca) - mean(cb))
    before_after = mean(ta) - mean(tb)
    cross_section = mean(ta) - mean(ca)
    return did, before_after, cross_section


# =============================================================================
# 11. DEMOS
# =============================================================================
def _hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def demo_special_functions():
    _hdr("0. SPECIAL FUNCTIONS — validated against standard statistical tables")
    checks = [
        ("t quantile 0.975, df=10", t_ppf(0.975, 10), 2.228139),
        ("t quantile 0.975, df=1", t_ppf(0.975, 1), 12.706205),
        ("t quantile 0.975, df=30", t_ppf(0.975, 30), 2.042272),
        ("chi-square quantile 0.95, df=1", chi2_ppf(0.95, 1), 3.841459),
        ("chi-square quantile 0.95, df=4", chi2_ppf(0.95, 4), 9.487729),
        ("chi-square quantile 0.99, df=10", chi2_ppf(0.99, 10), 23.209251),
        ("normal quantile 0.975", normal_ppf(0.975), 1.959964),
        ("F(3,20) CDF at 3.098391", f_cdf(3.098391, 3, 20), 0.950000),
        ("Binomial(20, 0.3) CDF at 2", binom_cdf(2, 20, 0.3), 0.035483),
    ]
    print(f"  {'quantity':<34}{'computed':>12}{'table':>12}")
    worst = 0.0
    for name, got, ref in checks:
        worst = max(worst, abs(got - ref) / abs(ref))
        print(f"  {name:<34}{got:>12.6f}{ref:>12.6f}")
    print(f"  worst relative error: {worst:.1e}")
    print("  -> every test and interval below is built on these, so they are")
    print("     checked first. The t quantile at df=1 (12.71) versus the normal")
    print("     1.96 shows how much a tiny sample widens an interval.")


def demo_probability():
    _hdr("1. PROBABILITY — exact answers, checked by simulation")
    exact = bayes_test(0.01, 0.99, 0.95)
    sim = simulate_bayes_test(0.01, 0.99, 0.95)
    print("  Base-rate fallacy: disease prevalence 1%, test sensitivity 99%,")
    print("  specificity 95%. P(disease | positive test)?")
    print(f"    exact (Bayes) {exact:.4f} | simulated (200,000 people, seed 0) {sim:.4f}")
    print("    -> only ~1 in 6 positives is actually sick: the 5% false-positive")
    print("       rate applied to 99% healthy people swamps the true positives.\n")

    stay, switch = simulate_monty_hall()
    print(f"  Monty Hall (100,000 games, seed 0): stay wins {stay:.3f}, "
          f"switch wins {switch:.3f}   (exact 1/3 and 2/3)")
    print("    -> holds because the host KNOWS and always reveals a goat.\n")

    print("  Birthday problem (40,000 groups per row, seed 0):")
    for k in [10, 23, 50]:
        print(f"    {k:>3} people: exact {birthday_exact(k):.4f} | "
              f"simulated {simulate_birthday(k):.4f}")
    print("    -> 23 people already pass 50%: pairs grow like k^2/2, so there are")
    print("       253 pairs among 23 people.\n")

    print(f"  Coupon collector, n = 20: exact n*H_n = {coupon_collector_exact(20):.2f}"
          f" | simulated {simulate_coupon(20):.2f} (5,000 runs, seed 0)")
    fp = [fixed_points_expectation(n) for n in [3, 10, 50]]
    print(f"  Expected fixed points of a random permutation, n = 3, 10, 50: "
          f"{', '.join(f'{v:.3f}' for v in fp)}")
    print("    -> exactly 1 for every n, by linearity of expectation — which holds")
    print("       even though the indicator variables are dependent.\n")

    for p in [0.50, 0.47]:
        print(f"  Gambler's ruin, start 10, target 20, win prob {p}: exact "
              f"{gamblers_ruin_exact(10, 20, p):.4f} | simulated "
              f"{simulate_gamblers_ruin(10, 20, p):.4f} (20,000 runs)")
    print("    -> a 3-point edge against you MORE than halves your chance of")
    print("       doubling up (0.50 -> 0.23).\n")

    P = [[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.1, 0.3, 0.6]]
    pi = stationary_distribution(P)
    emp = simulate_chain(P)
    print("  Markov chain stationary distribution (3 states):")
    print(f"    power iteration pi = {[round(v, 4) for v in pi]}")
    print(f"    fraction of 200,000 simulated steps = {[round(v, 4) for v in emp]}")


def demo_limit_theorems():
    _hdr("2. LIMIT THEOREMS — and where they break")
    print("  Sample means of Exponential(1) data (3,000 simulated means each, seed 0):")
    print(f"  {'n':>6}{'skewness':>11}{'theory 2/sqrt(n)':>18}{'IQR of mean':>13}")
    for n in [1, 2, 10, 30, 200]:
        med, sk, iqr = clt_check(lambda r: sample_exponential(r), n, 3000)
        print(f"  {n:>6}{sk:>11.3f}{2 / math.sqrt(n):>18.3f}{iqr:>13.3f}")
    print("  -> skewness falls toward 0 and the spread shrinks like 1/sqrt(n): the")
    print("     CLT at work. (Simulated skewness is itself noisy by ~+/-0.1.)")
    print("     For skewed data 'n >= 30' is not a law: at n = 30 the theoretical")
    print("     skewness of the mean is still 0.37 (0.50 in this simulation).\n")
    print("  The same for Cauchy data, which has NO finite mean or variance:")
    print(f"  {'n':>6}{'IQR of mean':>13}   (standard Cauchy IQR = 2.000)")
    for n in [1, 10, 200]:
        med, sk, iqr = clt_check(sample_cauchy, n, 3000)
        print(f"  {n:>6}{iqr:>13.3f}")
    print("  -> averaging 200 Cauchy draws is no more precise than ONE draw: the")
    print("     mean of n standard Cauchys is again standard Cauchy. The CLT needs")
    print("     finite variance; the LLN needs a finite mean.\n")
    ce = running_mean_path(sample_cauchy, 10000, seed=3)
    ee = running_mean_path(lambda r: sample_exponential(r), 10000, seed=3)
    print("  Running mean after n draws (one sequence each, seed 3):")
    print(f"  {'n':>8}{'Exponential(1), mu=1':>22}{'Cauchy (no mean)':>20}")
    for n in [10, 100, 1000, 10000]:
        print(f"  {n:>8}{ee[n]:>22.4f}{ce[n]:>20.4f}")


def demo_estimation():
    _hdr("3. ESTIMATION — bias, variance, and why unbiased is not always best")
    mle, unb, sdev = variance_estimator_bias(n=5)
    print("  Estimating sigma^2 = 1 from n = 5 normal points (40,000 samples, seed 0):")
    print(f"    divide by n     : mean estimate {mle:.4f}  (theory (n-1)/n = 0.8000)")
    print(f"    divide by n - 1 : mean estimate {unb:.4f}  (theory 1.0000)")
    print(f"    sample sd s     : mean estimate {sdev:.4f}  (sigma = 1: s is still")
    print("                      biased LOW — sqrt is concave, so E[s] < sigma)\n")

    print("  Shrinking the sample mean toward 0: estimate c * xbar")
    print("  (n = 5, sigma = 2, 20,000 samples per row, seed 0)")
    for mu in [0.5, 3.0]:
        print(f"    true mean {mu}:")
        print(f"    {'c':>8}{'bias':>9}{'variance':>11}{'MSE':>9}")
        for c, (b, v, m) in shrinkage_mse(true_mu=mu).items():
            print(f"    {c:>8}{b:>9.3f}{v:>11.3f}{m:>9.3f}")
    print("  -> MSE = bias^2 + variance. When the true mean is small relative to")
    print("     the noise, shrinking trades a little bias for a big variance cut")
    print("     and WINS. When it is large, the same shrinkage badly LOSES. Ridge")
    print("     regression and James-Stein exploit the first case — the benefit")
    print("     is conditional, not free.\n")

    rng = random.Random(4)
    data = [rng.expovariate(1 / 3) for _ in range(40)]
    se_med, _ = bootstrap_se(data, lambda xs: quantile(xs, 0.5), B=2000, seed=0)
    se_mean, _ = bootstrap_se(data, mean, B=2000, seed=0)
    print("  Bootstrap standard errors (one sample of 40 Exponential(mean 3) values,")
    print("  2,000 resamples):")
    print(f"    mean  : bootstrap SE {se_mean:.3f} | formula s/sqrt(n) "
          f"{sd(data) / math.sqrt(40):.3f}")
    print(f"    median: bootstrap SE {se_med:.3f} | (no simple formula — this is")
    print("            where the bootstrap earns its keep)")


def demo_intervals():
    _hdr("4. CONFIDENCE INTERVALS — does '95%' actually mean 95%?")
    norm = lambda r: r.gauss(10, 3)
    print("  Mean of Normal(10, 3^2) data; 3,000 intervals per row (seed 0):")
    print(f"  {'n':>5}{'z coverage':>13}{'t coverage':>13}{'z width':>10}{'t width':>10}")
    for n in [5, 10, 30]:
        cz, wz = coverage(ci_mean_z, norm, 10, n, 3000)
        ct, wt = coverage(ci_mean_t, norm, 10, n, 3000)
        print(f"  {n:>5}{cz:>13.1%}{ct:>13.1%}{wz:>10.2f}{wt:>10.2f}")
    print("  -> plugging the sample sd into a z interval under-covers at small n;")
    print("     the t quantile widens the interval to pay for estimating sigma.\n")

    print("  Proportion intervals, 4,000 simulated samples per row (seed 2):")
    print(f"  {'true p':>8}{'n':>6}{'Wald':>9}{'Wilson':>9}")
    for p, n in [(0.5, 20), (0.05, 20), (0.05, 100), (0.01, 100)]:
        rng = random.Random(2)
        hw = hl = 0
        for _ in range(4000):
            k = sample_binomial(rng, n, p)
            lo, hi = ci_prop_wald(k, n)
            hw += lo <= p <= hi
            lo, hi = ci_prop_wilson(k, n)
            hl += lo <= p <= hi
        print(f"  {p:>8}{n:>6}{hw / 4000:>9.1%}{hl / 4000:>9.1%}")
    print("  -> the textbook Wald interval is fine at p = 0.5 and disastrous for")
    print("     rare events (at k = 0 it has zero width). Use Wilson — this matters")
    print("     for conversion rates, error rates, and any small proportion.\n")

    skew = lambda r: r.expovariate(1.0)
    ct, _ = coverage(ci_mean_t, skew, 1.0, 15, 2000, seed=5)
    cb, _ = coverage(lambda xs: ci_bootstrap_percentile(xs, mean, B=400, seed=1),
                     skew, 1.0, 15, 400, seed=5)
    print("  Skewed data (Exponential, n = 15):")
    print(f"    t interval coverage {ct:.1%} (2,000 samples) | bootstrap percentile "
          f"{cb:.1%} (400 samples)")
    print("  -> neither reaches 95% at n = 15 with skewed data: the t interval")
    print("     assumes normality, and the plain percentile bootstrap is known to")
    print("     under-cover for small samples (BCa or bootstrap-t do better).")
    print("     A nominal level is a claim that depends on assumptions.")


def demo_tests():
    _hdr("5. HYPOTHESIS TESTS — calibration, assumptions, and power")
    norm = lambda r: r.gauss(0, 1)
    print("  False-positive rate when H0 is TRUE (should be 5.0%), 3,000 tests each:")
    fp_w = false_positive_rate(lambda a, b: ttest_welch(a, b)[1], norm, norm, 20, 20)
    print(f"    Welch t, equal variances, n = 20/20          : {fp_w:.1%}")
    wide = lambda r: r.gauss(0, 4)
    fp_s = false_positive_rate(lambda a, b: ttest_student_pooled(a, b)[1],
                               wide, norm, 10, 40)
    fp_w2 = false_positive_rate(lambda a, b: ttest_welch(a, b)[1], wide, norm, 10, 40)
    print(f"    Student pooled t, sd 4 vs 1, n = 10 vs 40    : {fp_s:.1%}")
    print(f"    Welch t, same unequal-variance data          : {fp_w2:.1%}")
    print("  -> the pooled test assumes equal variances; when the SMALLER group has")
    print("     the LARGER variance it fires far too often. Welch holds its level.")
    print("     Default to Welch.\n")

    rng = random.Random(7)
    before = [rng.gauss(50, 10) for _ in range(15)]
    after = [b + 2 + rng.gauss(0, 2) for b in before]
    print("  Paired vs unpaired on the same 15 subjects (true improvement +2, large")
    print("  person-to-person variation, seed 7):")
    print(f"    paired t   p = {ttest_paired(after, before)[1]:.4f}")
    print(f"    Welch t    p = {ttest_welch(after, before)[1]:.4f}")
    print("  -> pairing removes between-subject variation. Ignoring the pairing")
    print("     throws that information away.\n")

    heavy = lambda r: sample_cauchy(r)
    shifted = lambda r: sample_cauchy(r) + 1.0
    rng2 = random.Random(8)
    rej_t = rej_u = 0
    for _ in range(1000):
        a = [heavy(rng2) for _ in range(30)]
        b = [shifted(rng2) for _ in range(30)]
        rej_t += ttest_welch(a, b)[1] < 0.05
        rej_u += mann_whitney_u(a, b)[2] < 0.05
    print("  Power with heavy-tailed data (Cauchy, shift 1, n = 30/30, 1,000 tests):")
    print(f"    Welch t-test   : {rej_t / 1000:.1%}")
    print(f"    Mann-Whitney U : {rej_u / 1000:.1%}")
    print("  -> outliers inflate the t-test's variance estimate and destroy its")
    print("     power; the rank-based test does not care how extreme they are.\n")

    table = [[30, 70], [45, 55]]
    x2, p, df, min_e = chi2_independence(table)
    print(f"  Chi-square independence on [[30, 70], [45, 55]]: X^2 = {x2:.3f}, "
          f"df = {df}, p = {p:.4f}")
    print(f"    (smallest expected count {min_e:.1f} >= 5, so the approximation is OK)")
    rng3 = random.Random(9)
    a = [rng3.gauss(0, 1) for _ in range(25)]
    b = [rng3.gauss(0.6, 1) for _ in range(25)]
    obs, pp = permutation_test(a, b, n_perm=2000, seed=0)
    print(f"  Permutation test vs Welch on one dataset (n = 25/25, seed 9): "
          f"permutation p = {pp:.4f}, Welch p = {ttest_welch(a, b)[1]:.4f}\n")

    print("  Power, two-sample Welch t, effect 0.5 sd (1,000 simulated tests each):")
    print(f"  {'n per group':>12}{'simulated':>11}{'normal approx':>15}")
    for n in [10, 20, 40, 64, 100]:
        ps = power_sim(lambda a, b: ttest_welch(a, b)[1], 0.5, n, reps=1000)
        print(f"  {n:>12}{ps:>11.1%}{power_two_sample_analytic(0.5, n):>15.1%}")
    print(f"  required n per group for 80% power: {n_per_group(0.5)} from the normal")
    print("  approximation; t-based calculations give 64 (the usual textbook figure),")
    print("  because estimating sigma costs a little power at small n.")
    print("  -> a 'non-significant' result from an n = 20 study of a 0.5-sd effect")
    print("     is weak evidence of no effect: that study had ~1-in-3 power.")


def demo_multiple_testing():
    _hdr("6. MULTIPLE TESTING — FWER vs FDR, and what 'significant' implies")
    r = simulate_multiple_testing(m=200, frac_true=0.1, effect=0.8, n=20, reps=1000, seed=1)
    print("  200 hypotheses per study, 10% truly non-null (effect 0.8 sd, n = 20/20),")
    print("  1,000 simulated studies:")
    print(f"  {'rule':<22}{'FWER':>8}{'FDR':>8}{'power':>8}")
    for k, label in [("none", "p < 0.05 each"), ("bonf", "Bonferroni"),
                     ("bh", "Benjamini-Hochberg")]:
        v = r[k]
        print(f"  {label:<22}{v['fwer']:>8.1%}{v['fdp']:>8.1%}"
              f"{v['power']:>8.1%}")
    print("  -> uncorrected: a false positive in EVERY study, and ~39% of all")
    print("     'discoveries' are false. Bonferroni guarantees FWER <= 5% but finds")
    print("     under 9% of real effects. BH holds FDR at its bound (q x fraction")
    print("     null = 4.5%) while more than doubling Bonferroni's power.\n")

    print("  P(the null is TRUE | p < 0.05), by prior plausibility and power:")
    print(f"  {'share of true effects':>22}{'power':>8}{'P(H0 | significant)':>22}")
    for pi, pw in [(0.5, 0.8), (0.1, 0.8), (0.1, 0.3), (0.01, 0.3)]:
        print(f"  {pi:>22.0%}{pw:>8.0%}{prob_h0_given_significant(pi, pw):>22.1%}")
    print("  -> a p-value is P(data | H0), not P(H0 | data). In a field testing")
    print("     long shots with small samples, most significant results are false.")


def demo_regression():
    _hdr("7. REGRESSION INFERENCE — when the standard errors lie")
    print("  95% CI coverage for a slope (n = 60, 1,500 simulated datasets, seed 0):")
    print(f"  {'errors':<20}{'classic SE':>12}{'robust HC1 SE':>15}")
    for het, label in [(False, "homoscedastic"), (True, "heteroscedastic")]:
        c1 = slope_ci_coverage(het, False)
        c2 = slope_ci_coverage(het, True)
        print(f"  {label:<20}{c1:>12.1%}{c2:>15.1%}")
    print("  -> when noise grows with x, the slope stays unbiased but the classic")
    print("     SE is too small and the '95%' interval covers ~87%. Robust SEs fix")
    print("     it without needing to model the variance. They are not free: when")
    print("     errors ARE homoscedastic, HC1 slightly under-covers in small samples")
    print("     (93.3% here); HC3 is the usual small-sample choice.\n")

    rng = random.Random(1)
    Xv = []
    for _ in range(300):
        u = rng.gauss(0, 1)
        Xv.append([u, u + rng.gauss(0, 0.1), rng.gauss(0, 1)])
    yv = [1 + x[0] + x[1] + x[2] + rng.gauss(0, 1) for x in Xv]
    b, se, *_ = ols(Xv, yv)
    print("  Multicollinearity: x2 = x1 + small noise; x3 independent (n = 300):")
    print(f"    VIF   x1 {vif(Xv)[0]:.1f}   x2 {vif(Xv)[1]:.1f}   x3 {vif(Xv)[2]:.1f}")
    print(f"    coef  x1 {b[1]:+.2f} (SE {se[1]:.2f})   x2 {b[2]:+.2f} (SE {se[2]:.2f})"
          f"   x3 {b[3]:+.2f} (SE {se[3]:.2f})   (truth: 1, 1, 1)")
    print("  -> the model can't tell x1 from x2, so their individual coefficients")
    print("     are unstable (SE ~10x larger than x3's), though their SUM is well")
    print("     estimated and predictions are fine.\n")

    short, full, pred = omitted_variable_bias()
    print("  Omitted variable bias (true effect of x = 2; z omitted; n = 2,000):")
    print(f"    y ~ x       : {short:.3f}   (formula predicts {pred:.3f})")
    print(f"    y ~ x + z   : {full:.3f}")
    print("  -> more data does not help: the short regression converges to the")
    print("     wrong number.\n")

    t1, t2, b1, b2 = regression_to_the_mean()
    print("  Regression to the mean (test reliability 0.5, 5,000 people, seed 0):")
    print(f"    top 10% on test 1   : {t1:+.2f} -> retest {t2:+.2f}")
    print(f"    bottom 10% on test 1: {b1:+.2f} -> retest {b2:+.2f}")
    print("  -> both extremes move halfway back with NO intervention. A program")
    print("     that treats the worst scorers and then 'sees improvement' needs a")
    print("     control group to show it did anything.")


def demo_bayesian():
    _hdr("8. BAYESIAN INFERENCE — posteriors and credible intervals")
    a, b = beta_posterior(3, 10)
    print("  3 successes in 10 trials, flat Beta(1,1) prior -> posterior Beta(4, 8)")
    print(f"    posterior mean {a / (a + b):.3f}  (Laplace: (k+1)/(n+2) = 0.333)")
    print(f"    95% credible interval [{beta_ppf(0.025, a, b):.3f}, "
          f"{beta_ppf(0.975, a, b):.3f}]")
    lo, hi = ci_prop_wilson(3, 10)
    print(f"    95% Wilson confidence interval [{lo:.3f}, {hi:.3f}]")
    print("  -> numerically similar here, different in meaning: the credible")
    print("     interval is a probability statement about p GIVEN the prior; the")
    print("     confidence interval is a property of the procedure.\n")

    for k, n in [(3, 10), (30, 100), (300, 1000)]:
        a1, b1 = beta_posterior(k, n, 1, 1)
        a2, b2 = beta_posterior(k, n, 20, 20)
        print(f"    {k}/{n}: flat-prior mean {a1 / (a1 + b1):.3f} | "
              f"Beta(20,20)-prior mean {a2 / (a2 + b2):.3f}")
    print("  -> a strong prior dominates small samples and washes out as data grows.\n")

    pb, lift, lo, hi = bayesian_ab(1000, 10000, 1080, 10000)
    two_sided_p = 2 * (1 - normal_cdf((0.108 - 0.100) / math.sqrt(
        0.104 * 0.896 * 2 / 10000)))
    print("  Bayesian A/B: 1,000/10,000 vs 1,080/10,000, flat priors (40,000 draws):")
    print(f"    P(B > A | data) = {pb:.3f}; expected lift {lift:+.4f}, "
          f"95% interval [{lo:+.4f}, {hi:+.4f}]")
    print(f"    frequentist two-sided p = {two_sided_p:.3f} (one-sided "
          f"{two_sided_p / 2:.3f})")
    print("  -> with a flat prior and large n, P(B > A) ~ 1 - one-sided p. '97%")
    print("     likely better' and 'not significant at 5%' describe the SAME data;")
    print("     they answer different questions.")


def demo_causal():
    _hdr("9. CAUSAL INFERENCE — confounding, Simpson, and adjustment")
    s = simpsons_paradox()
    print("  Kidney stones (Charig et al. 1986) — success rate:")
    print(f"  {'':<12}{'small stones':>14}{'large stones':>14}{'all':>8}")
    for t in ("A", "B"):
        print(f"  {'treatment ' + t:<12}{s[(t, 'small')]:>14.1%}{s[(t, 'large')]:>14.1%}"
              f"{s[(t, 'all')]:>8.1%}")
    print("  -> A wins in EACH group, B wins overall: A was given mostly to the")
    print("     harder, large-stone cases. The data alone cannot say which")
    print("     comparison is right — that needs the causal story.\n")

    rows = make_confounded_data()
    naive, reg, strat, ipw = estimate_effects(rows)
    print("  Observational data, confounder z (severity) drives both treatment and")
    print("  outcome; TRUE effect = +1.00 (n = 4,000, seed 0):")
    print(f"    naive difference in means : {naive:+.2f}")
    print(f"    regression y ~ t + z      : {reg:+.2f}")
    print(f"    stratification (10 bins)  : {strat:+.2f}")
    print(f"    inverse propensity weights: {ipw:+.2f}")
    print("  -> the naive estimate has the WRONG SIGN: sicker patients get treated")
    print("     more and do worse anyway. Adjusting for z recovers ~+1 — but only")
    print("     because z is the only confounder and it was measured. That")
    print("     assumption cannot be checked from the data.\n")

    runs = [difference_in_differences(seed=s) for s in range(40)]
    did = [r[0] for r in runs]
    ba = [r[1] for r in runs]
    cs = [r[2] for r in runs]
    print("  Difference-in-differences, TRUE effect = +2.00 (40 simulations of")
    print("  200 treated + 200 control units; mean and SD across simulations):")
    print(f"    treated before vs after      : {mean(ba):+.2f} (SD {sd(ba):.2f})"
          "   <- absorbs the time trend")
    print(f"    treated vs control, after    : {mean(cs):+.2f} (SD {sd(cs):.2f})"
          "   <- absorbs the group gap")
    print(f"    difference-in-differences    : {mean(did):+.2f} (SD {sd(did):.2f})")
    print("  -> DiD removes both, IF the groups would have followed parallel trends")
    print("     without treatment. Check pre-period trends; it cannot be proven.")


# =============================================================================
# 12. PROBABILITY PUZZLES — each answer checked by simulation
# =============================================================================
def puzzle_two_children(n=200000, seed=0):
    """
    'A family has two children and at least one is a boy. P(both boys)?' -> 1/3.
    'You meet one of the two children and it is a boy. P(both boys)?' -> 1/2.
    Same words, different information-gathering protocol, different answer.
    """
    rng = random.Random(seed)
    at_least = both_given_at_least = met = both_given_met = 0
    for _ in range(n):
        kids = (rng.random() < 0.5, rng.random() < 0.5)       # True = boy
        if kids[0] or kids[1]:
            at_least += 1
            both_given_at_least += kids[0] and kids[1]
        seen = kids[rng.randrange(2)]
        if seen:
            met += 1
            both_given_met += kids[0] and kids[1]
    return both_given_at_least / at_least, both_given_met / met


def puzzle_flips_until(pattern, n=40000, seed=0):
    """Expected coin flips until a pattern first appears: HH -> 6, HT -> 4.
    HT is faster because after a failed attempt (an H followed by H) you are
    still one flip away, whereas HH's failure (H then T) resets you to zero."""
    rng = random.Random(seed)
    total = 0
    for _ in range(n):
        seq = ""
        while not seq.endswith(pattern):
            seq += "H" if rng.random() < 0.5 else "T"
        total += len(seq)
    return total / n


def puzzle_broken_stick(n=200000, seed=0):
    """Break a stick at two uniform random points: P(the three pieces form a
    triangle) = 1/4 (every piece must be shorter than half the stick)."""
    rng = random.Random(seed)
    ok = 0
    for _ in range(n):
        a, b = sorted((rng.random(), rng.random()))
        ok += max(a, b - a, 1 - b) < 0.5
    return ok / n


def puzzle_max_two_dice(n=200000, seed=0):
    """E[max of two fair dice] = sum_k P(max >= k) = 161/36 ~ 4.472."""
    rng = random.Random(seed)
    return sum(max(rng.randint(1, 6), rng.randint(1, 6)) for _ in range(n)) / n


def puzzle_german_tank(true_N=300, k=5, reps=20000, seed=0):
    """
    Serial numbers 1..N; you observe k of them at random (without replacement).
    Sample max m is biased LOW. The minimum-variance unbiased estimator is
        N_hat = m (1 + 1/k) - 1
    (the average gap between observed serials, added back onto the max).
    """
    rng = random.Random(seed)
    naive = mvue = 0.0
    for _ in range(reps):
        m = max(rng.sample(range(1, true_N + 1), k))
        naive += m
        mvue += m * (1 + 1 / k) - 1
    return naive / reps, mvue / reps


def reservoir_sample(stream, k, rng):
    """
    RESERVOIR SAMPLING (Algorithm R): a uniform sample of k items from a stream
    of unknown length in one pass and O(k) memory. Item i (1-indexed) replaces a
    random reservoir slot with probability k/i. By induction every item ends up
    in the reservoir with probability exactly k/n.
    """
    res = []
    for i, x in enumerate(stream, 1):
        if i <= k:
            res.append(x)
        else:
            j = rng.randrange(i)
            if j < k:
                res[j] = x
    return res


def puzzle_reservoir(n=20, k=5, reps=40000, seed=0):
    rng = random.Random(seed)
    counts = [0] * n
    for _ in range(reps):
        for x in reservoir_sample(range(n), k, rng):
            counts[x] += 1
    freqs = [c / reps for c in counts]
    return min(freqs), max(freqs), k / n


def puzzle_secretary(n=100, reps=20000, seed=0):
    """
    SECRETARY PROBLEM: see n candidates in random order, accept or reject each
    immediately. Strategy: reject the first r, then accept the first candidate
    better than all seen so far. With r ~ n/e the chance of picking the single
    best is ~1/e ~ 0.368 — for any large n.
    """
    rng = random.Random(seed)
    r = round(n / math.e)
    wins = 0
    for _ in range(reps):
        ranks = list(range(n))
        rng.shuffle(ranks)                    # higher = better; best is n-1
        bar = max(ranks[:r]) if r else -1
        pick = next((x for x in ranks[r:] if x > bar), ranks[-1])
        wins += pick == n - 1
    return wins / reps


def puzzle_poisson_gap(rate_per_hour=3.0, minutes=20, reps=100000, seed=0):
    """Events arrive as a Poisson process at 3 per hour. P(no event in the next
    20 minutes) = exp(-3 * 20/60) = e^-1 ~ 0.368 — memorylessness means it does
    not matter how long you have already waited."""
    rng = random.Random(seed)
    lam = rate_per_hour / 60
    return sum(1 for _ in range(reps) if rng.expovariate(lam) > minutes) / reps


def demo_puzzles():
    _hdr("10. PROBABILITY PUZZLES — interview classics, answers checked")
    a, b = puzzle_two_children()
    print("  Two children (200,000 families, seed 0):")
    print(f"    told 'at least one is a boy'  -> P(two boys) = {a:.4f}  (exact 1/3)")
    print(f"    you MEET one child, a boy     -> P(two boys) = {b:.4f}  (exact 1/2)")
    print("    -> the same fact ('a boy'), learned by a different protocol, gives a")
    print("       different answer: what matters is HOW the information was obtained.\n")
    print(f"  Expected flips until HH = {puzzle_flips_until('HH'):.3f} (exact 6);"
          f" until HT = {puzzle_flips_until('HT'):.3f} (exact 4)  [40,000 runs]")
    print(f"  Broken stick forms a triangle: {puzzle_broken_stick():.4f} (exact 0.25)")
    print(f"  E[max of two dice]: {puzzle_max_two_dice():.4f} (exact 161/36 = "
          f"{161 / 36:.4f})")
    nv, mv = puzzle_german_tank()
    print(f"  German tanks, N = 300, k = 5 serials seen (20,000 reps): sample max "
          f"averages {nv:.1f}")
    print(f"    (exact E[m] = k(N+1)/(k+1) = {5 * 301 / 6:.1f}: biased low); "
          f"m(1 + 1/k) - 1 averages {mv:.1f} (exact: 300, unbiased)")
    lo, hi, target = puzzle_reservoir()
    print(f"  Reservoir sampling k = 5 from 20 (40,000 runs): inclusion frequency "
          f"per item in [{lo:.4f}, {hi:.4f}], exact {target:.2f}")
    print(f"  Secretary problem, n = 100, skip n/e (20,000 runs): picks the best "
          f"{puzzle_secretary():.3f} of the time (theory ~1/e = {1 / math.e:.3f})")
    print(f"  Poisson, 3/hour: P(no event in 20 min) = {puzzle_poisson_gap():.4f}"
          f" (exact e^-1 = {math.exp(-1):.4f})")


if __name__ == "__main__":
    print("=" * 78)
    print(" PROBABILITY & STATISTICS FROM SCRATCH — every claim checked by simulation")
    print("=" * 78)
    demo_special_functions()
    demo_probability()
    demo_limit_theorems()
    demo_estimation()
    demo_intervals()
    demo_tests()
    demo_multiple_testing()
    demo_regression()
    demo_bayesian()
    demo_causal()
    demo_puzzles()
    print("\n" + "=" * 78)
    print("All results use fixed seeds; rerunning reproduces them exactly.")
    print("=" * 78)
