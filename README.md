# emeans

A drop-in replacement for `sklearn.cluster.KMeans` based on **e-means**
([Bessa et al., 2026](https://arxiv.org/abs/2611.09501)). Pure numpy/scipy, runs on CPU,
no Faiss or GPU needed.

On 200k-row datasets it lands within about 1% of scikit-learn's `KMeans()` loss
and runs 2.4x to 24x faster.

```bash
pip install git+https://github.com/sacredvoid/emeans
```

```python
from emeans import EMeans

km = EMeans(n_clusters=1024).fit(X)      # same API as sklearn KMeans
km.cluster_centers_, km.labels_, km.inertia_
km.predict(X_new)
km.transform(X_new)                       # distances to centroids
```

## What e-means is

Lloyd's k-means makes a full pass over the data for every update. e-means
updates centroids from mini-batches instead, keeping each centroid as an
exponential moving average of the points assigned to it. On top of that, the paper adds:

- a cosine schedule on the step size, so centroids settle down over training
- a last-pass mean at the end, which reliably lowers the training loss
- fitted scaling laws that pick the batch size and step size from `n`, `k` and
  the number of epochs, so there is nothing to tune

The result is k-means quality in 2 to 3 passes over the data instead of 25.
The paper reports 10-45x GPU speedups at billion scale. This package is for the
much more common case of a laptop or a CPU server.

## Benchmarks

`k=1024` clusters (64 for california), 200k training rows (465k for covtype),
3 seeds, 24-core CPU. Loss is held-out MSE relative to scikit-learn Lloyd with
k-means++ seeding run for 25 iterations. Lower is better.

| dataset | `EMeans()` loss | `EMeans()` time | sklearn `KMeans()` time | speedup |
|---|---|---|---|---|
| well-separated blobs, d=32 | -0.26% | 2.5 s | 16.5 s | 6.6x |
| Gaussian mixture, d=8 | -0.09% | 5.4 s | 13.1 s | 2.4x |
| Gaussian mixture, d=32 | +1.13% | 2.6 s | 19.4 s | 7.5x |
| Gaussian mixture, d=128 | +0.96% | 7.5 s | 67.0 s | 8.9x |
| covtype (standardized), d=54 | -0.04% | 3.7 s | 90.8 s | 24.5x |
| california housing, d=8 | -1.79% | 0.2 s | 0.1 s | none (tiny data) |

For comparison, `MiniBatchKMeans` is +101% on the blobs and +237% on covtype.

Reproduce with `uv run --group dev python benchmarks/bench.py`.

**When it will not help:** small data. If Lloyd already finishes in a fraction
of a second, there is nothing to save. The speedup grows with `n`, `k` and `d`.

## Where these defaults differ from the paper

The paper seeds centroids at random and periodically "revives" clusters whose
moving-average point count drops below 10% of the mean, by splitting a busy
cluster in two. Running that recipe (as published, and through Meta's
reference code in Faiss) gave two failure modes:

1. **Data with outliers.** A cluster that covers a handful of far-away points
   looks dead to the revival rule, so it gets moved, and those points end up
   very far from every centroid. On standardized covtype, the worst 1% of points
   carried 54% of the error. Loss was +119% on covtype and +148% on california
   housing, compared with Lloyd.
2. **Turning revival off is not enough.** With random seeding and no revival,
   well-separated blobs come out +253% worse, because revival was quietly
   repairing bad seeds.

Seeding with k-means++ on a subsample of `max(10000, 20*k)` rows, with revival
off, fixes both. It was never more than 2.1% behind Lloyd in our tests. The
one regime where the paper's recipe is slightly better is clean,
high-dimensional embeddings (+0.23% vs +0.96% at d=128), so it is one argument away:

```python
EMeans(k, init="random", revival_threshold=None)   # paper / Faiss recipe
```

## Parameters

| parameter | default | meaning |
|---|---|---|
| `n_clusters` | 8 | number of centroids |
| `n_epochs` | 3 | passes over the data; fractional values work (e.g. `1.5`) |
| `init` | `"k-means++"` | `"k-means++"` (on a subsample), `"random"`, or an array |
| `batch_size`, `alpha0` | `None` | override the scaling laws |
| `batch_law` | `"auto"` | `"quality"` (smaller batches, used when d <= 16) or `"throughput"` |
| `revival_threshold` | `0.0` | `None` = paper rule; a float = revive clusters below that fraction of mean mass |
| `random_state` | `None` | seed |

Fitted attributes: `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`,
`batch_size_`, `alpha0_`, `n_splits_`.

## Use Faiss instead if

You have a GPU, billions of points, or millions of centroids. Meta's reference
implementation lives in
[faiss/contrib/e_means](https://github.com/facebookresearch/faiss/tree/main/contrib/e_means),
with CUDA kernels and distributed training. The scaling-law constants here come
from that code.

## Credit

The algorithm and scaling laws are from:

```bibtex
@article{bessa2026emeans,
  title   = {e-means: outperforming k-means (in one-tenth the epochs)},
  author  = {Bessa, Swann and Mazar{\'e}, Pierre-Emmanuel and Szilvasy, Gergely and
             Touvron, Hugo and Picard, David and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal = {arXiv preprint arXiv:2611.09501},
  year    = {2026}
}
```

This package is an independent implementation and is not affiliated with Meta.

## License

MIT
