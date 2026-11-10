"""Benchmark EMeans defaults and the paper recipe against scikit-learn KMeans / MiniBatchKMeans.

Run from the repo root: uv run --group dev python benchmarks/bench.py

Prints one JSON line per measurement (prefix RESULT) and a summary at the end.
Test MSE is mean squared distance to the nearest centroid on held-out rows.
"""

import json
import os
import time

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import fetch_california_housing, fetch_covtype, make_blobs
from sklearn.preprocessing import StandardScaler

import emeans
from emeans import EMeans, _assign, batch_law, step_law

SEEDS = [0, 1, 2]
DIMS = [8, 32, 128]
BLOBS = dict(n_samples=250_000, n_features=32, centers=1000)
N_TRAIN, N_TEST, K = 200_000, 50_000, 1024
ALPHA_MULTS = [1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8]
PROBE_EPOCHS = 3


def synthetic(d, seed):
    """Gaussian mixture with 256 components of random scale: clustered, non-uniform."""
    rng = np.random.default_rng(1000 + d)  # same distribution across seeds
    centers = rng.normal(size=(256, d))
    scales = rng.uniform(0.05, 0.5, size=256)
    r = np.random.default_rng(seed)
    comp = r.integers(0, 256, size=N_TRAIN + N_TEST)
    X = centers[comp] + scales[comp, None] * r.normal(size=(len(comp), d))
    X = X.astype(np.float32)
    return X[:N_TRAIN], X[N_TRAIN:]


def blobs(seed):
    """Well-separated clusters: where random seeding without revival fails."""
    X = make_blobs(**BLOBS, random_state=1000)[0].astype(np.float32)
    idx = np.random.default_rng(seed).permutation(len(X))
    return X[idx[:N_TRAIN]], X[idx[N_TRAIN:]]


def real(name, seed):
    X = (fetch_covtype() if name == "covtype" else fetch_california_housing()).data
    X = StandardScaler().fit_transform(X).astype(np.float32)
    idx = np.random.default_rng(seed).permutation(len(X))
    cut = int(0.8 * len(X))
    return X[idx[:cut]], X[idx[cut:]]


def mse(X, C):
    return float(_assign(X, C.astype(X.dtype))[1].mean())


def methods(k, seed):
    return {
        "lloyd25_kmeans++": KMeans(k, init="k-means++", n_init=1, max_iter=25, tol=0, random_state=seed),
        "lloyd25_random": KMeans(k, init="random", n_init=1, max_iter=25, tol=0, random_state=seed),
        "sklearn_default": KMeans(k, random_state=seed),  # k-means++, n_init=1 ("auto"), max_iter=300, tol=1e-4
        "minibatch_sklearn": MiniBatchKMeans(k, random_state=seed),
        "emeans_T2": EMeans(k, n_epochs=2, random_state=seed),
        "emeans_T3": EMeans(k, n_epochs=3, random_state=seed),
        "emeans_T3_paper": EMeans(k, n_epochs=3, init="random", revival_threshold=None, random_state=seed),
    }


def emit(**row):
    print("RESULT " + json.dumps(row), flush=True)
    return row


def part_a(datasets):
    rows = []
    for name, loader, k in datasets:
        for seed in SEEDS:
            try:
                Xtr, Xte = loader(seed)
            except Exception as e:  # download failure must not kill the sweep
                print(f"SKIP {name}: {type(e).__name__}: {e}", flush=True)
                break
            for mname, est in methods(k, seed).items():
                t = time.perf_counter()
                est.fit(Xtr)
                sec = time.perf_counter() - t
                rows.append(emit(part="A", dataset=name, d=Xtr.shape[1], n=len(Xtr), k=k, seed=seed,
                                 method=mname, seconds=round(sec, 3),
                                 train_mse=mse(Xtr, est.cluster_centers_),
                                 test_mse=mse(Xte, est.cluster_centers_)))
    return rows


def part_b():
    rows = []
    for d in DIMS:
        for seed in SEEDS:
            Xtr, Xte = synthetic(d, seed)
            B = batch_law(len(Xtr), K, PROBE_EPOCHS, d)
            a_law = step_law(len(Xtr), K, PROBE_EPOCHS, B)
            for m in ALPHA_MULTS:
                a = min(a_law * m, 1.0)
                est = EMeans(K, n_epochs=PROBE_EPOCHS, batch_size=B, alpha0=a, random_state=seed).fit(Xtr)
                rows.append(emit(part="B", d=d, seed=seed, batch=B, alpha_law=a_law, mult=m, alpha0=a,
                                 test_mse=mse(Xte, est.cluster_centers_)))
    return rows


def summarize(a_rows, b_rows):
    print("\n=== Part A: test and train MSE relative to lloyd25_kmeans++ (median over seeds), seconds ===")
    for key in sorted({(r["dataset"], r["d"]) for r in a_rows}, key=lambda x: (x[0] not in ("blobs", "synthetic"), x[0], x[1])):
        sub = [r for r in a_rows if (r["dataset"], r["d"]) == key]
        ref = {r["seed"]: r["test_mse"] for r in sub if r["method"] == "lloyd25_kmeans++"}
        ref_tr = {r["seed"]: r["train_mse"] for r in sub if r["method"] == "lloyd25_kmeans++"}
        parts = []
        print(f"{key[0]:>10} d={key[1]}")
        for mname in dict.fromkeys(r["method"] for r in sub):
            ms = [r for r in sub if r["method"] == mname]
            rel = np.median([r["test_mse"] / ref[r["seed"]] - 1 for r in ms]) * 100
            rel_tr = np.median([r["train_mse"] / ref_tr[r["seed"]] - 1 for r in ms]) * 100
            parts.append(f"{mname}=test{rel:+.2f}%/train{rel_tr:+.2f}%/{np.median([r['seconds'] for r in ms]):.1f}s")
        print("    " + "\n    ".join(parts))

    if not b_rows:
        return
    print("\n=== Part B: best alpha0 multiplier over the step law, per d (median over seeds) ===")
    for d in DIMS:
        sub = [r for r in b_rows if r["d"] == d]
        best_m = []
        for seed in SEEDS:
            s = [r for r in sub if r["seed"] == seed]
            best_m.append(min(s, key=lambda r: r["test_mse"])["mult"])
        at_law = np.median([r["test_mse"] for r in sub if r["mult"] == 1])
        best = min(np.median([r["test_mse"] for r in sub if r["mult"] == m]) for m in ALPHA_MULTS)
        print(f"d={d:<4} B={sub[0]['batch']:<6} alpha_law={sub[0]['alpha_law']:.3g} "
              f"best_mult(per seed)={best_m}  regret_at_law={100 * (at_law / best - 1):.2f}%")


if __name__ == "__main__":
    print(f"emeans {emeans.__version__}  commit={os.environ.get('ORX_COMMIT')}  cpus={os.cpu_count()}")
    print(f"config: n_train={N_TRAIN} n_test={N_TEST} k={K} dims={DIMS} seeds={SEEDS} "
          f"alpha_mults={ALPHA_MULTS} probe_epochs={PROBE_EPOCHS}", flush=True)
    datasets = [("blobs", lambda s: blobs(s), K)]
    datasets += [(f"synthetic", lambda s, d=d: synthetic(d, s), K) for d in DIMS]
    datasets += [("covtype", lambda s: real("covtype", s), K), ("california", lambda s: real("california", s), 64)]
    t0 = time.perf_counter()
    a_rows = part_a(datasets)
    summarize(a_rows, [])
    print(f"\ntotal_seconds={time.perf_counter() - t0:.0f}")
