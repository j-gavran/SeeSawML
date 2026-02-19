from __future__ import annotations

import logging
from typing import Any

import hist
import numpy as np


class ToyDistributions:
    def __init__(self, dist_type: str) -> None:
        """Different pt-like shape distributions.

        Parameters
        ---------
        dist_type : str
            Type of distribution: ('signal', 'background', 'resonance').

        Other Parameters
        ----------------
        dist_x : np.ndarray or None
            x values of the distribution.
        dist_y : np.ndarray or None
            y values of the distribution.
        n_samples : int or None
            Number of samples to generate.
        dist_kwargs : Any
            Additional parameters for the distribution.

        Returns:
        --------
        np.ndarray
            Values of the distribution for the given x.

        """
        self.dist_type = dist_type

        self.n_samples: int | None = None
        self._sampled_dist: np.ndarray | None = None

    @property
    def sampled_dist(self) -> np.ndarray:
        """Sampled distribution."""
        if self._sampled_dist is None:
            raise ValueError("Error. No samples generated yet.")
        return self._sampled_dist

    def signal(self, x: np.ndarray, a: float = 2.0, b: float = 10.0, c: float = 3.0, d: float = 0.01) -> np.ndarray:
        """Signal-like shape distribution, drops off at high pt."""
        return a / ((x + b) ** c) * np.exp(-d * x)

    def background(self, x: np.ndarray, alpha: float = 1.2, beta: float = 0.2, gamma: float = 0.005) -> np.ndarray:
        """Background: typical shapes for QCD background process."""
        return alpha * (x ** (-beta)) * np.exp(-gamma * x)

    def resonance(self, x: np.ndarray, mu: float = 500.0, sigma: float = 30.0, norm: float = 1.0) -> np.ndarray:
        """Gaussian shape for resonance (peaks in particulal kinematic region)."""
        return norm * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def qcd(self, x, s: float = 13000.0, A: float = 1.0, b: float = 5.0, c: float = 6.0, d: float = 0.0) -> np.ndarray:
        """QCD-like 4-parameter background: f(m) = A * (1 - x)^b / x^{c + d*ln(x)}, with x = m/sqrt(s)."""
        x_hat = x / s
        x_hat = np.clip(x_hat, 1e-12, 1.0 - 1e-12)
        exponent = c + d * np.log(x_hat)
        return A * (1.0 - x_hat) ** b / (x_hat**exponent)

    def _evaluate(
        self, x: np.ndarray, tail_boost: str | None = None, tail_boost_factor: float = 2.0, **kwargs: Any
    ) -> np.ndarray:
        """Evaluate the distribution function in the given range."""

        if self.dist_type == "signal":
            y = self.signal(x, **kwargs)
        elif self.dist_type == "background":
            y = self.background(x, **kwargs)
        elif self.dist_type == "resonance":
            y = self.resonance(x, **kwargs)
        elif self.dist_type == "qcd":
            y = self.qcd(x, **kwargs)
        else:
            raise ValueError(f"Error. Not valid distribution type: {self.dist_type}")

        x_min, x_max = np.min(x), np.max(x)

        if tail_boost is None:
            tail_mask = None
        elif tail_boost == "both":
            lower_mask = x < (x_min + (x_max - x_min) * 0.1)
            upper_mask = x > (x_min + (x_max - x_min) * 0.9)
            tail_mask = lower_mask | upper_mask
        elif tail_boost == "lower":
            tail_mask = x < (x_min + (x_max - x_min) * 0.1)
        elif tail_boost == "upper":
            tail_mask = x > (x_min + (x_max - x_min) * 0.9)
        else:
            raise ValueError("Error. Not valid tail_boost value.")

        if tail_mask is not None:
            y[tail_mask] *= tail_boost_factor  # increase probabilities in tails

        return y

    def calculate_samples(
        self, x_min: float, x_max: float, n_samples: int, x_points: int = 1000, **kwargs: Any
    ) -> ToyDistributions:
        logging.info(f"Sampling {self.dist_type} distribution with parameters: {kwargs}")

        self.n_samples = n_samples

        x = np.linspace(x_min, x_max, x_points)
        y = self._evaluate(x, **kwargs)

        y_max = np.max(y)
        samples: list[np.ndarray] = []

        while len(samples) < n_samples:
            x_rand = np.random.uniform(x_min, x_max, n_samples - len(samples))
            y_rand = np.random.uniform(0, y_max, n_samples - len(samples))
            samples.extend(x_rand[y_rand < self._evaluate(x_rand, **kwargs)])

        self._sampled_dist = np.array(samples[:n_samples], dtype=np.float32)

        return self

    def fill_hist(self, name: str, bins: list[float]) -> hist.Hist:
        """Fill histogram with the sampled distribution."""
        h = hist.Hist(hist.axis.Variable(bins, name=name))
        h.fill(self.sampled_dist)
        return h


class WeightedToyDistributions(ToyDistributions):
    def __init__(self, dist_type: str, weight_type: str) -> None:
        super().__init__(dist_type)
        """Generate weights for the toy distributions.

        Parameters
        ----------
        weight_type : str
            Type of weight distribution: ('lognormal', 'normal', 'uniform', 'cauchy', 'unity').

        Other Parameters
        ----------------
        weights : np.ndarray or None
            Weights for the samples.

        """
        self.weight_type = weight_type

        self._weights: np.ndarray | None = None

    @property
    def weights(self) -> np.ndarray:
        """Weights for the samples."""
        if self._weights is None:
            raise ValueError("Error. No weights generated yet.")
        return self._weights

    def lognormal(self, mean: float = 3.5, sigma: float = 0.5) -> np.ndarray:
        """Generate log-normal distributed weights."""
        return np.random.lognormal(mean=mean, sigma=sigma, size=self.n_samples).astype(np.float32)

    def normal(self, mean: float = 1.0, sigma: float = 0.1) -> np.ndarray:
        """Generate normal distributed weights."""
        return np.random.normal(loc=mean, scale=sigma, size=self.n_samples).astype(np.float32)

    def cauchy(self, x_0: float = 0.0, gamma: float = 1.0, clip_tails: float | None = None) -> np.ndarray:
        """Generate Cauchy distributed weights, https://en.wikipedia.org/wiki/Cauchy_distribution."""
        y: np.ndarray = x_0 + gamma * np.random.standard_cauchy(size=self.n_samples)  # type: ignore[assignment]

        if clip_tails is not None:
            y[np.abs(y) > clip_tails] = x_0

        return y.astype(np.float32)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> np.ndarray:
        """Generate uniform distributed weights."""
        return np.random.uniform(low=low, high=high, size=self.n_samples).astype(np.float32)

    def calculate_weights(self, **kwargs: Any) -> WeightedToyDistributions:
        if self.n_samples is None:
            raise ValueError("Error. No samples generated yet.")

        logging.info(f"Generating {self.weight_type} weights with parameters: {kwargs}")

        if self.weight_type == "lognormal":
            self._weights = self.lognormal(**kwargs)
        elif self.weight_type == "normal":
            self._weights = self.normal(**kwargs)
        elif self.weight_type == "cauchy":
            self._weights = self.cauchy(**kwargs)
        elif self.weight_type == "uniform":
            self._weights = self.uniform(**kwargs)
        elif self.weight_type == "unity":
            self._weights = np.ones(self.n_samples, dtype=np.float32)
        else:
            raise ValueError(f"Error. Not valid weight type: {self.weight_type}")

        return self

    def fill_hist(self, name: str, bins: list[float]) -> hist.Hist:
        """Fill histogram with the sampled distribution and weights."""
        h = hist.Hist(hist.axis.Variable(bins, name=name), storage=hist.storage.Weight())
        h.fill(self.sampled_dist, weight=self.weights)
        return h
