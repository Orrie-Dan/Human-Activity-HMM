"""Tests for src.hmm."""
from itertools import product

import numpy as np
import pytest
from scipy.stats import norm

from src.hmm import GaussianHMM


def small_model():
    """2-state, 1-D model with hand-picked parameters."""
    m = GaussianHMM(2)
    m.startprob = np.array([0.6, 0.4])
    m.transmat = np.array([[0.7, 0.3], [0.2, 0.8]])
    m.means = np.array([[0.0], [3.0]])
    m.variances = np.array([[1.0], [2.0]])
    return m


def brute_force(m, X):
    """Joint probability of every state path (exhaustive enumeration)."""
    paths = {}
    for path in product(range(m.n_states), repeat=len(X)):
        p = m.startprob[path[0]]
        for t, s in enumerate(path):
            if t:
                p *= m.transmat[path[t - 1], s]
            p *= norm.pdf(X[t, 0], m.means[s, 0], np.sqrt(m.variances[s, 0]))
        paths[path] = p
    return paths


X_SMALL = np.array([[0.2], [2.5], [3.1], [-0.4]])


class TestInference:
    def test_log_emission_matches_scipy(self):
        m = small_model()
        expected = norm.logpdf(X_SMALL, m.means[:, 0], np.sqrt(m.variances[:, 0]))
        np.testing.assert_allclose(m.log_emission(X_SMALL), expected)

    def test_forward_likelihood_matches_enumeration(self):
        m = small_model()
        total = sum(brute_force(m, X_SMALL).values())
        assert m.score(X_SMALL) == pytest.approx(np.log(total))

    def test_backward_gives_same_likelihood(self):
        m = small_model()
        log_beta = m.backward(X_SMALL)
        log_b0 = m.log_emission(X_SMALL)[0]
        from_backward = np.logaddexp.reduce(np.log(m.startprob) + log_b0 + log_beta[0])
        assert from_backward == pytest.approx(m.score(X_SMALL))

    def test_posteriors_match_enumeration(self):
        m = small_model()
        paths = brute_force(m, X_SMALL)
        total = sum(paths.values())
        expected = np.zeros((len(X_SMALL), 2))
        for path, p in paths.items():
            expected[np.arange(len(path)), path] += p / total
        np.testing.assert_allclose(m.posteriors(X_SMALL), expected)

    def test_viterbi_matches_enumeration(self):
        m = small_model()
        paths = brute_force(m, X_SMALL)
        best = max(paths, key=paths.get)
        path, logp = m.viterbi(X_SMALL)
        assert tuple(path) == best
        assert logp == pytest.approx(np.log(paths[best]))

    def test_zero_transition_is_respected(self):
        m = small_model()
        m.transmat = np.array([[1.0, 0.0], [0.5, 0.5]])
        m.startprob = np.array([1.0, 0.0])
        # Observations look like state 1, but state 0 can never leave itself.
        assert m.predict(np.array([[3.0], [3.0], [3.0]])).tolist() == [0, 0, 0]

    def test_long_sequence_does_not_underflow(self):
        m = small_model()
        X, _ = m.sample(5000, np.random.default_rng(0))
        assert np.isfinite(m.score(X))


class TestTraining:
    def test_fit_supervised_counts_and_moments(self):
        X = [np.array([[0.0], [2.0], [10.0]]), np.array([[4.0], [12.0]])]
        z = [np.array([0, 0, 1]), np.array([0, 1])]
        m = GaussianHMM(2).fit_supervised(X, z, pseudocount=1.0)
        # Transitions 0->0 once, 0->1 twice; each row gets +1 per cell.
        np.testing.assert_allclose(m.transmat, [[2 / 5, 3 / 5], [1 / 2, 1 / 2]])
        np.testing.assert_allclose(m.startprob, [3 / 4, 1 / 4])
        np.testing.assert_allclose(m.means, [[2.0], [11.0]])
        np.testing.assert_allclose(m.variances, [[8 / 3], [1.0]])

    def test_variance_floor(self):
        X = [np.array([[1.0], [1.0]])]
        m = GaussianHMM(1, variance_floor=0.5).fit_supervised(X, [np.array([0, 0])])
        assert m.variances[0, 0] == 0.5

    def test_baum_welch_requires_initialization(self):
        with pytest.raises(ValueError):
            GaussianHMM(2).fit([X_SMALL])

    def test_baum_welch_likelihood_never_decreases(self):
        rng = np.random.default_rng(1)
        true = small_model()
        seqs = [true.sample(200, rng)[0] for _ in range(3)]
        m = small_model()
        m.means = np.array([[-1.0], [1.0]])
        m.transmat = np.full((2, 2), 0.5)
        history = m.fit(seqs, max_iter=30, tol=0.0)
        assert np.all(np.diff(history) >= -1e-6)

    def test_baum_welch_recovers_parameters(self):
        rng = np.random.default_rng(2)
        true = small_model()
        true.means = np.array([[0.0], [5.0]])
        true.variances = np.array([[1.0], [1.0]])
        seqs = [true.sample(500, rng)[0] for _ in range(4)]
        m = GaussianHMM(2)
        m.startprob = np.array([0.5, 0.5])
        m.transmat = np.full((2, 2), 0.5)
        m.means = np.array([[-1.0], [6.0]])
        m.variances = np.array([[2.0], [2.0]])
        m.fit(seqs, max_iter=100, tol=1e-6)
        np.testing.assert_allclose(m.means, true.means, atol=0.15)
        np.testing.assert_allclose(m.transmat, true.transmat, atol=0.05)
        np.testing.assert_allclose(m.transmat.sum(axis=1), 1.0)
