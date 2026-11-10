import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from emeans import EMeans, batch_law, step_law


def test_matches_kmeans_quality_on_blobs():
    X, _ = make_blobs(20_000, n_features=32, centers=200, random_state=0)
    em = EMeans(256, n_epochs=3, random_state=0).fit(X)  # library defaults
    km = KMeans(256, init="random", n_init=1, max_iter=25, random_state=0).fit(X)
    assert em.cluster_centers_.shape == (256, 32)
    assert np.array_equal(em.labels_, em.predict(X))
    assert np.isclose(-em.score(X), em.inertia_)
    assert em.inertia_ < 1.05 * km.inertia_  # paper: matches/beats Lloyd in ~2-3 epochs
    assert em.transform(X[:5]).shape == (5, 256)


def test_float32_fractional_epochs_and_tiny_input():
    X = np.random.default_rng(0).normal(size=(500, 2)).astype(np.float32)
    em = EMeans(10, n_epochs=1.5, random_state=0).fit(X)
    assert em.cluster_centers_.dtype == np.float32
    assert len(set(em.labels_)) > 1
    EMeans(3, random_state=0).fit(X[:3])  # n == k must not crash


def test_laws_and_validation():
    # Table 1 setting in the paper: n=1e6, k=16384, T=3, high-d -> throughput law
    B = batch_law(10**6, 16384, 3, d=96)
    assert B == 65536 and 0.3 < step_law(10**6, 16384, 3, B) < 0.4
    assert batch_law(10**6, 16384, 3, d=8) == B // 8
    with pytest.raises(ValueError):
        EMeans(10).fit(np.zeros((5, 2)))
    with pytest.raises(ValueError):
        EMeans(2, alpha0=1.5).fit(np.zeros((5, 2)))


def test_init_options():
    X, _ = make_blobs(2_000, n_features=4, centers=20, random_state=0)
    for init in ["k-means++", X[:16]]:
        assert EMeans(16, init=init, random_state=0).fit(X).cluster_centers_.shape == (16, 4)
    with pytest.raises(ValueError):
        EMeans(16, init="bogus").fit(X)
    with pytest.raises(ValueError):
        EMeans(16, init=X[:3]).fit(X)


def test_default_keeps_outlier_clusters_paper_rule_drops_them():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(20_000, 8)), 100 + rng.normal(size=(5, 8))])  # 5 far outliers
    out = X[-5:]
    init = np.vstack([X[:63], out[:1]])
    paper = EMeans(64, init=init, revival_threshold=0.1, random_state=0).fit(X)
    default = EMeans(64, init=init, random_state=0).fit(X)
    assert np.linalg.norm(default.cluster_centers_ - 100, axis=1).min() < 5
    assert -default.score(out) < -paper.score(out)
