"""e-means: mini-batch k-means with EMA centroid updates and scaling-law defaults.

A dependency-light, scikit-learn compatible implementation of Algorithm 1 from
Bessa et al., "e-means: outperforming k-means (in one-tenth the epochs)",
arXiv:2611.09501. Constants follow Meta's reference implementation
(faiss/contrib/e_means, MIT license), including its low-dimension rules.

Defaults differ from the paper in two places, both measured: k-means++ seeding
on a subsample instead of random seeding, and no dead-cluster revival. See the
README for the benchmark behind that choice.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data
from threadpoolctl import threadpool_limits

__all__ = ["EMeans", "batch_law", "step_law"]
__version__ = "0.1.0"

_ALPHA_MIN_FACTOR = 0.1  # epsilon in Eq. 5: cosine floor / peak step
_CHECK_UNUSED_EVERY = 5  # revival check period, in updates
_INIT_USAGE = 2.0**-16  # initial EMA mass N_j (paper: 1e-10)
_SPLIT_EPSILON = 1e-4  # relative perturbation when splitting a donor
_UNUSED_THRESHOLD = 0.1  # tau: revive clusters with N_j < tau * mean(N)
_NO_REVIVAL_MAX_D = 2  # faiss: revival hurts on 2-D data
_AUTO_QUALITY_MAX_D = 16  # faiss: use the smaller "quality" batch at d <= 16

# B = c * k**b_k * n**b_n * T**b_T  (Eq. 7), shipped faiss fits.
_BATCH_LAWS = {
    "throughput": (0.035134, 0.134749, 0.876473, 0.813344),
    "quality": (0.00439175, 0.134749, 0.876473, 0.813344),
}
# alpha0 = c * k**a_k * B**a_B * n**a_n * T**a_T  (Eq. 6), shipped faiss fit.
_ALPHA_LAW = (0.8244, -0.4617, 0.9003, -0.4498, -0.1057)


def _round_pow2(value):
    if value <= 1:
        return 1
    low = 2 ** int(math.floor(math.log2(value)))
    return low if value / low < 2 * low / value else 2 * low


def batch_law(n, k, n_epochs, d, law="auto"):
    """Batch size B from the fitted batch law; ``law="auto"`` picks by dimension."""
    if law == "auto":
        law = "quality" if d <= _AUTO_QUALITY_MAX_D else "throughput"
    if law not in _BATCH_LAWS:
        raise ValueError(f"batch_law must be 'auto', 'quality' or 'throughput', got {law!r}")
    c, bk, bn, bt = _BATCH_LAWS[law]
    b = _round_pow2(c * k**bk * n**bn * n_epochs**bt)
    b = min(b, _round_pow2(n / 2)) if law == "throughput" else min(b, n // 2)
    return max(1, min(b, n))


def step_law(n, k, n_epochs, batch_size):
    """Peak EMA step alpha0 from the fitted step law, clamped to [1e-4, 1]."""
    c, ak, ab, an, at = _ALPHA_LAW
    return min(max(c * k**ak * batch_size**ab * n**an * n_epochs**at, 1e-4), 1.0)


def _epoch_rows(n, n_epochs):
    """Rows per epoch: whole passes with any fractional pass before the last one."""
    whole = math.floor(n_epochs)
    rem = int(round((n_epochs - whole) * n))
    if whole == 0:
        return [rem]
    return [n] * (whole - 1) + ([rem] if rem else []) + [n]


_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 1)
    return _POOL


def _assign_chunk(xb, C, half_c_sq):
    p = xb @ C.T  # maximize x.c - |c|^2/2  <=>  minimize |x - c|^2
    p -= half_c_sq
    lab = np.argmax(p, axis=1)
    best = p[np.arange(len(xb)), lab]
    x_sq = np.einsum("ij,ij->i", xb, xb)
    dist = np.maximum(x_sq - 2.0 * best, 0.0)
    return lab, dist


def _assign(X, C, c_sq=None):
    """Nearest centroid labels and squared distances, chunked and multithreaded."""
    if c_sq is None:
        c_sq = np.einsum("ij,ij->i", C, C)
    half_c_sq = (0.5 * c_sq).astype(C.dtype)
    n, k = X.shape[0], C.shape[0]
    mem_rows = max(1, (1 << 22) // max(1, k))  # ~16 MB float32 score block per chunk
    workers = os.cpu_count() or 1
    rows = max(1, min(mem_rows, -(-n // workers)))
    starts = range(0, n, rows)
    if len(starts) == 1 or n * k < (1 << 20):
        parts = [_assign_chunk(X[s : s + rows], C, half_c_sq) for s in starts]
    else:  # numpy releases the GIL in matmul/argmax; keep BLAS single-threaded per worker
        with threadpool_limits(limits=1, user_api="blas"):
            parts = list(_pool().map(lambda s: _assign_chunk(X[s : s + rows], C, half_c_sq), starts))
    labels = np.concatenate([p[0] for p in parts]).astype(np.intp, copy=False)
    dist = np.concatenate([p[1] for p in parts]).astype(np.float64, copy=False)
    return labels, dist


class EMeans(ClusterMixin, TransformerMixin, BaseEstimator):
    """Mini-batch k-means with EMA centroids and a cosine-annealed step (e-means).

    Parameters
    ----------
    n_clusters : int, default=8
    n_epochs : float, default=3
        Passes over the data. Fractional values are allowed (e.g. 1.5).
    batch_size : int or None
        None uses the fitted batch law.
    alpha0 : float or None
        Peak EMA step (1 - decay). None uses the fitted step law for the batch size.
    init : {"k-means++", "random"} or ndarray of shape (n_clusters, n_features), default="k-means++"
        Initial centroids. "k-means++" seeds on a random subsample of
        max(10000, 20 * n_clusters) rows, so seeding stays cheap at any n.
        "random" samples rows (paper default; needs revival to be competitive).
    batch_law : {"auto", "quality", "throughput"}, default="auto"
        "auto" uses "quality" (8x smaller batches) when n_features <= 16.
    revival_threshold : float or None, default=0.0
        Revive clusters whose EMA mass falls below this fraction of the mean
        mass. Off by default: revival discards small clusters that cover
        outliers (2-2.5x worse loss on covtype / california housing) and gains
        <= 1% on clean data. None reproduces the paper / Faiss rule (0.1, or
        off when n_features <= 2).
    random_state : int, RandomState or None
    verbose : bool

    Attributes
    ----------
    cluster_centers_, labels_, inertia_, n_iter_, n_features_in_,
    batch_size_, alpha0_, n_splits_
    """

    def __init__(
        self,
        n_clusters=8,
        *,
        n_epochs=3,
        batch_size=None,
        alpha0=None,
        init="k-means++",
        batch_law="auto",
        revival_threshold=0.0,
        random_state=None,
        verbose=False,
    ):
        self.n_clusters = n_clusters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.init = init
        self.batch_law = batch_law
        self.revival_threshold = revival_threshold
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=[np.float64, np.float32], order="C")
        n, d = X.shape
        k = int(self.n_clusters)
        T = float(self.n_epochs)
        if k < 1:
            raise ValueError(f"n_clusters must be >= 1, got {k}")
        if n < k:
            raise ValueError(f"n_samples={n} should be >= n_clusters={k}.")
        if not T > 0:
            raise ValueError(f"n_epochs must be > 0, got {self.n_epochs}")
        rng = check_random_state(self.random_state)

        if self.batch_size is None:
            B = batch_law(n, k, T, d, self.batch_law)
        else:
            if int(self.batch_size) < 1:
                raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
            B = min(int(self.batch_size), n)
        alpha0 = step_law(n, k, T, B) if self.alpha0 is None else float(self.alpha0)
        if not 0 < alpha0 <= 1:
            raise ValueError(f"alpha0 must be in (0, 1], got {alpha0}")
        tau = self.revival_threshold
        if tau is None:
            tau = 0.0 if d <= _NO_REVIVAL_MAX_D else _UNUSED_THRESHOLD
        tau = float(tau)
        if tau < 0:
            raise ValueError("revival_threshold must be >= 0")

        epochs = _epoch_rows(n, T)
        total_steps = sum(-(-rows // B) for rows in epochs)
        if isinstance(self.init, str) and self.init == "random":
            C = X[rng.choice(n, k, replace=False)].copy()
        elif isinstance(self.init, str) and self.init == "k-means++":
            m = min(n, max(10_000, 20 * k))
            S = X if m == n else X[np.sort(rng.choice(n, m, replace=False))]
            C = kmeans_plusplus(S, k, random_state=rng)[0].astype(X.dtype)
        elif isinstance(self.init, str):
            raise ValueError(f"init must be 'random', 'k-means++' or an array, got {self.init!r}")
        else:
            C = np.array(self.init, dtype=X.dtype)
            if C.shape != (k, d):
                raise ValueError(f"init array has shape {C.shape}, expected {(k, d)}")
        N = np.full(k, _INIT_USAGE)
        step = 0
        n_splits = 0
        for e, rows in enumerate(epochs):
            last = e == len(epochs) - 1
            if last:  # last-epoch-mean finalization accumulators
                pass_counts = np.zeros(k)
                pass_sums = np.zeros((k, d))
            perm = rng.permutation(n)[:rows]
            for s in range(0, rows, B):
                xb = X[perm[s : s + B]]
                lab, _ = _assign(xb, C)
                ind = sparse.csr_matrix(
                    (np.ones(len(lab)), (lab, np.arange(len(lab)))), shape=(k, len(lab))
                )
                counts = np.bincount(lab, minlength=k).astype(np.float64)
                sums = np.asarray(ind @ xb, dtype=np.float64)
                if last:
                    pass_counts += counts
                    pass_sums += sums
                frac = step / max(1, total_steps)
                alpha = alpha0 * (
                    _ALPHA_MIN_FACTOR
                    + (1 - _ALPHA_MIN_FACTOR) * 0.5 * (1 + math.cos(math.pi * frac))
                )
                # EMA on (N_j, S_j) expressed on (N_j, c_j) = (N_j, S_j / N_j).
                N *= 1.0 - alpha
                t = counts > 0
                mass = N[t] + alpha * counts[t]
                w = alpha * counts[t] / mass
                mean = sums[t] / counts[t, None]
                C[t] += (w[:, None] * (mean - C[t])).astype(C.dtype)
                N[t] = mass
                step += 1
                if tau > 0 and step % _CHECK_UNUSED_EVERY == 0:
                    n_splits += self._revive(C, N, tau, rng)
            if self.verbose:
                print(f"epoch {e + 1}/{len(epochs)}: rows={rows} B={B} alpha0={alpha0:.4g}")

        nonempty = pass_counts > 0
        C[nonempty] = (pass_sums[nonempty] / pass_counts[nonempty, None]).astype(C.dtype)

        self.cluster_centers_ = C
        self.labels_, dist = _assign(X, C)
        self.inertia_ = float(dist.sum())
        self.n_iter_ = T
        self.batch_size_ = B
        self.alpha0_ = alpha0
        self.n_splits_ = n_splits
        return self

    @staticmethod
    def _revive(C, N, tau, rng):
        """Split heavy donors (sampled by mass, without replacement) into dead clusters."""
        k = len(N)
        unused = N < tau * N.sum() / k
        donors = ~unused & (N > 0)
        m = min(int(unused.sum()), int(donors.sum()))
        if m == 0:
            return 0
        # Gumbel top-m: sample donors without replacement proportional to N.
        u = np.maximum(rng.random_sample(k), np.finfo(np.float64).tiny)
        keys = np.full(k, -np.inf)
        keys[donors] = np.log(N[donors]) - np.log(-np.log(u[donors]))
        src = np.argsort(keys)[-m:][::-1]
        dst = np.flatnonzero(unused)[:m]
        sign = np.ones(C.shape[1], dtype=C.dtype)
        sign[1::2] = -1
        off = _SPLIT_EPSILON * sign
        base = C[src].copy()
        C[src] = base * (1 - off)
        C[dst] = base * (1 + off)
        same = (C[src] == C[dst]).all(axis=1)  # zero centroids: nudge one coordinate
        if same.any():
            C[dst[same], 0] = np.nextafter(C[dst[same], 0], np.inf)
        N[src] /= 2
        N[dst] = N[src]
        return m

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], order="C", reset=False)
        return _assign(X, self.cluster_centers_.astype(X.dtype, copy=False))[0]

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def transform(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)
        return euclidean_distances(X, self.cluster_centers_)

    def score(self, X, y=None):
        """Negative sum of squared distances to the nearest centroid (as in KMeans)."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], order="C", reset=False)
        return -float(_assign(X, self.cluster_centers_.astype(X.dtype, copy=False))[1].sum())
