"""Hidden Markov Model: forward-backward, Viterbi, and Baum-Welch.

A from-scratch HMM with diagonal-covariance Gaussian emissions. All recursions
run in log space so long sequences do not underflow.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import logsumexp

LOG_2PI = np.log(2 * np.pi)


def _safe_log(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(x)


class GaussianHMM:
    """HMM with ``n_states`` hidden states and diagonal Gaussian emissions.

    Parameters (set by ``fit_supervised``/``fit`` or assigned directly):
    ``startprob`` (K,), ``transmat`` (K, K) with rows summing to 1,
    ``means`` (K, D) and ``variances`` (K, D).
    """

    def __init__(self, n_states: int, variance_floor: float = 1e-3):
        self.n_states = n_states
        self.variance_floor = variance_floor
        self.startprob: np.ndarray | None = None
        self.transmat: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.variances: np.ndarray | None = None

    # --- Emissions -----------------------------------------------------------

    def log_emission(self, X: np.ndarray) -> np.ndarray:
        """Log-density of every observation under every state, shape (T, K)."""
        X = np.asarray(X, dtype=float)
        diff = X[:, None, :] - self.means[None, :, :]
        return -0.5 * (
            np.sum(LOG_2PI + np.log(self.variances), axis=1)[None, :]
            + np.sum(diff ** 2 / self.variances[None, :, :], axis=2)
        )

    # --- Inference -----------------------------------------------------------

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Log forward variables ``log_alpha`` (T, K) and the sequence log-likelihood."""
        log_b = self.log_emission(X)
        log_a = _safe_log(self.transmat)
        log_alpha = np.empty_like(log_b)
        log_alpha[0] = _safe_log(self.startprob) + log_b[0]
        for t in range(1, len(log_b)):
            log_alpha[t] = logsumexp(log_alpha[t - 1][:, None] + log_a, axis=0) + log_b[t]
        return log_alpha, float(logsumexp(log_alpha[-1]))

    def backward(self, X: np.ndarray) -> np.ndarray:
        """Log backward variables ``log_beta`` (T, K)."""
        log_b = self.log_emission(X)
        log_a = _safe_log(self.transmat)
        log_beta = np.zeros_like(log_b)
        for t in range(len(log_b) - 2, -1, -1):
            log_beta[t] = logsumexp(log_a + (log_b[t + 1] + log_beta[t + 1])[None, :], axis=1)
        return log_beta

    def score(self, X: np.ndarray) -> float:
        """Log-likelihood of one observation sequence."""
        return self.forward(X)[1]

    def posteriors(self, X: np.ndarray) -> np.ndarray:
        """State posteriors P(state_t | X), shape (T, K)."""
        log_alpha, loglik = self.forward(X)
        return np.exp(log_alpha + self.backward(X) - loglik)

    def viterbi(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Most likely state path and its log-probability."""
        log_b = self.log_emission(X)
        log_a = _safe_log(self.transmat)
        T = len(log_b)
        delta = _safe_log(self.startprob) + log_b[0]
        backptr = np.zeros((T, self.n_states), dtype=int)
        for t in range(1, T):
            scores = delta[:, None] + log_a  # (from, to)
            backptr[t] = np.argmax(scores, axis=0)
            delta = scores[backptr[t], np.arange(self.n_states)] + log_b[t]
        path = np.empty(T, dtype=int)
        path[-1] = int(np.argmax(delta))
        for t in range(T - 1, 0, -1):
            path[t - 1] = backptr[t, path[t]]
        return path, float(delta.max())

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi state path."""
        return self.viterbi(X)[0]

    # --- Training ------------------------------------------------------------

    def fit_supervised(
        self,
        sequences: Sequence[np.ndarray],
        state_sequences: Sequence[np.ndarray],
        pseudocount: float = 1.0,
    ) -> GaussianHMM:
        """Maximum-likelihood parameters from labeled sequences (counts + pseudocounts)."""
        K = self.n_states
        X = np.concatenate(sequences)
        z = np.concatenate(state_sequences).astype(int)
        D = X.shape[1]

        start = np.full(K, pseudocount, dtype=float)
        trans = np.full((K, K), pseudocount, dtype=float)
        for states in state_sequences:
            start[states[0]] += 1
            np.add.at(trans, (states[:-1], states[1:]), 1)
        self.startprob = start / start.sum()
        self.transmat = trans / trans.sum(axis=1, keepdims=True)

        self.means = np.zeros((K, D))
        self.variances = np.ones((K, D))
        for k in range(K):
            Xk = X[z == k]
            if len(Xk):
                self.means[k] = Xk.mean(axis=0)
                self.variances[k] = Xk.var(axis=0)
        self.variances = np.maximum(self.variances, self.variance_floor)
        return self

    def fit(
        self,
        sequences: Sequence[np.ndarray],
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> list[float]:
        """Baum-Welch (EM) from the current parameters; returns total log-likelihood per iteration.

        Stops when the log-likelihood improves by less than ``tol``.
        """
        if self.means is None:
            raise ValueError("initialize parameters (e.g. fit_supervised) before Baum-Welch")
        K = self.n_states
        history: list[float] = []
        for _ in range(max_iter):
            start_acc = np.zeros(K)
            trans_acc = np.zeros((K, K))
            gamma_sum = np.zeros(K)
            x_sum = np.zeros_like(self.means)
            x2_sum = np.zeros_like(self.means)
            total = 0.0

            log_a = _safe_log(self.transmat)
            for X in sequences:
                X = np.asarray(X, dtype=float)
                log_b = self.log_emission(X)
                log_alpha, loglik = self.forward(X)
                log_beta = self.backward(X)
                total += loglik

                gamma = np.exp(log_alpha + log_beta - loglik)
                start_acc += gamma[0]
                if len(X) > 1:
                    log_xi = (
                        log_alpha[:-1, :, None]
                        + log_a[None, :, :]
                        + (log_b[1:] + log_beta[1:])[:, None, :]
                        - loglik
                    )
                    trans_acc += np.exp(log_xi).sum(axis=0)
                gamma_sum += gamma.sum(axis=0)
                x_sum += gamma.T @ X
                x2_sum += gamma.T @ X ** 2

            history.append(total)
            if len(history) > 1 and history[-1] - history[-2] < tol:
                break

            self.startprob = start_acc / start_acc.sum()
            row = trans_acc.sum(axis=1, keepdims=True)
            self.transmat = np.where(row > 0, trans_acc / np.where(row > 0, row, 1), self.transmat)
            used = gamma_sum > 0
            means = x_sum[used] / gamma_sum[used, None]
            self.variances[used] = np.maximum(
                x2_sum[used] / gamma_sum[used, None] - means ** 2, self.variance_floor
            )
            self.means[used] = means
        return history

    # --- Sampling (for tests and demonstrations) -----------------------------

    def sample(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw ``n`` observations and their hidden states."""
        states = np.empty(n, dtype=int)
        states[0] = rng.choice(self.n_states, p=self.startprob)
        for t in range(1, n):
            states[t] = rng.choice(self.n_states, p=self.transmat[states[t - 1]])
        X = rng.normal(self.means[states], np.sqrt(self.variances[states]))
        return X, states
