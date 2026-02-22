"""Significance metrics for signal/background classification.

This module provides functions for computing Asimov significance in
signal/background classification problems, including threshold scans
and binned significance calculations.

References
----------
.. [1] Cowan, G., Cranmer, K., Gross, E., & Vitells, O. (2011).
       Asymptotic formulae for likelihood-based tests of new physics.
       Eur. Phys. J. C, 71, 1554. arXiv:1007.1727
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

# Numeric constants
_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class SignificanceResult:
    """Result of a significance threshold scan.

    Attributes
    ----------
    max_significance : float
        Maximum Asimov significance found across all thresholds.
    optimal_threshold : float
        Threshold value at which maximum significance occurs.
    thresholds : np.ndarray
        Array of threshold values scanned, shape ``(n_points,)``.
    significances : np.ndarray
        Asimov significance at each threshold, shape ``(n_points,)``.
    """

    max_significance: float
    optimal_threshold: float
    thresholds: np.ndarray
    significances: np.ndarray


@dataclass(frozen=True, slots=True)
class BinnedSignificanceResult:
    """Result of binned significance calculation.

    Attributes
    ----------
    integrated_significance : float
        Quadrature sum of per-bin significances: ``sqrt(sum(Z_i^2))``.
    bin_significances : np.ndarray
        Asimov significance in each bin, shape ``(n_bins,)``.
    bin_edges : np.ndarray
        Bin edge values, shape ``(n_bins + 1,)``.
    bin_centers : np.ndarray
        Bin center values, shape ``(n_bins,)``.
    signal_yields : np.ndarray
        Weighted signal yield in each bin, shape ``(n_bins,)``.
    background_yields : np.ndarray
        Weighted background yield in each bin, shape ``(n_bins,)``.
    """

    integrated_significance: float
    bin_significances: np.ndarray
    bin_edges: np.ndarray
    bin_centers: np.ndarray
    signal_yields: np.ndarray
    background_yields: np.ndarray


def significance(
    s: float | np.ndarray,
    b: float | np.ndarray,
) -> float | np.ndarray:
    """Compute Asimov significance for given signal and background yields.

    Uses the Asimov approximation for discovery significance from a
    profile likelihood ratio test [1]_.

    Parameters
    ----------
    s : float or np.ndarray
        Weighted sum of signal events.
    b : float or np.ndarray
        Weighted sum of background events.

    Returns
    -------
    float or np.ndarray
        Asimov significance value (in units of standard deviations).
        Returns float if inputs are scalar, array otherwise.

    Notes
    -----
    The Asimov significance formula is:

    .. math::

        Z = \\sqrt{2 \\left[ (s + b) \\ln\\left(1 + \\frac{s}{b}\\right) - s \\right]}

    Edge cases:

    - If ``b <= 0``: returns 0.0 (no background means no discovery sensitivity)
    - If ``s < 0``: clips to 0.0 (can occur with negative MC weights)

    References
    ----------
    .. [1] Cowan et al. (2011), Eur. Phys. J. C 71, 1554.
    """
    s = np.asarray(s)
    b = np.asarray(b)
    scalar_input = s.ndim == 0 and b.ndim == 0

    s = np.atleast_1d(s).astype(np.float64)
    b = np.atleast_1d(b).astype(np.float64)

    result = np.zeros_like(s)

    valid = b > _EPSILON
    s_valid = np.maximum(s[valid], 0.0)
    b_valid = b[valid]

    ratio = s_valid / b_valid
    inner = 2.0 * ((s_valid + b_valid) * np.log1p(ratio) - s_valid)
    result[valid] = np.sqrt(np.maximum(inner, 0.0))

    if scalar_input:
        return float(result[0])
    return result


def scan_significance(
    scores: np.ndarray,
    weights: np.ndarray,
    is_signal: np.ndarray,
    n_points: int = 200,
) -> SignificanceResult:
    """Scan significance vs threshold to find optimal working point.

    For each threshold, computes the Asimov significance for events
    with score >= threshold.

    Parameters
    ----------
    scores : np.ndarray
        Classifier scores, shape ``(n_events,)``.
    weights : np.ndarray
        MC weights per event, shape ``(n_events,)``.
    is_signal : np.ndarray
        Boolean mask indicating signal events, shape ``(n_events,)``.
    n_points : int, optional
        Number of threshold points to scan. Default is 200.

    Returns
    -------
    SignificanceResult
        Dataclass containing max significance, optimal threshold,
        and arrays of all thresholds and significances.

    Notes
    -----
    The scan uses cumulative sums with binary search for O(n log n)
    complexity. Events are sorted by score in descending order, then
    for each threshold the cumulative weighted yield is computed via
    ``searchsorted``.

    See Also
    --------
    significance : The Asimov significance formula used at each threshold.
    binned_significance : Alternative using binned yields.
    """
    sig_scores, sig_weights = scores[is_signal], weights[is_signal]
    bkg_scores, bkg_weights = scores[~is_signal], weights[~is_signal]

    # Sort descending by score, compute cumulative sums
    sig_order = np.argsort(sig_scores)[::-1]
    bkg_order = np.argsort(bkg_scores)[::-1]

    sig_scores_sorted = sig_scores[sig_order]
    bkg_scores_sorted = bkg_scores[bkg_order]
    sig_cumsum = np.cumsum(sig_weights[sig_order])
    bkg_cumsum = np.cumsum(bkg_weights[bkg_order])

    thresholds = np.linspace(scores.min(), scores.max(), n_points)

    # For each threshold, find number of events passing (score >= threshold)
    # searchsorted on reversed array gives count of events with score < threshold
    sig_n_passing = len(sig_scores_sorted) - np.searchsorted(sig_scores_sorted[::-1], thresholds, side="left")
    bkg_n_passing = len(bkg_scores_sorted) - np.searchsorted(bkg_scores_sorted[::-1], thresholds, side="left")

    # Vectorized lookup of cumulative yields
    # Prepend 0 to handle the case where n_passing == 0
    sig_cumsum_padded = np.concatenate([[0.0], sig_cumsum])
    bkg_cumsum_padded = np.concatenate([[0.0], bkg_cumsum])

    s_yields = sig_cumsum_padded[sig_n_passing]
    b_yields = bkg_cumsum_padded[bkg_n_passing]

    # Vectorized significance computation
    significances = cast(np.ndarray, significance(s_yields, b_yields))

    best_idx = int(np.argmax(significances))

    return SignificanceResult(
        max_significance=float(significances[best_idx]),
        optimal_threshold=float(thresholds[best_idx]),
        thresholds=thresholds,
        significances=significances,
    )


def binned_significance(
    scores: np.ndarray,
    weights: np.ndarray,
    is_signal: np.ndarray,
    n_bins: int = 10,
    score_range: tuple[float, float] | None = None,
) -> BinnedSignificanceResult:
    """Compute binned and integrated significance.

    Bins the classifier score and computes Asimov significance per bin.
    The integrated significance combines bins in quadrature.

    Parameters
    ----------
    scores : np.ndarray
        Classifier scores, shape ``(n_events,)``.
    weights : np.ndarray
        MC weights per event, shape ``(n_events,)``.
    is_signal : np.ndarray
        Boolean mask indicating signal events, shape ``(n_events,)``.
    n_bins : int, optional
        Number of score bins. Default is 10.
    score_range : tuple[float, float], optional
        Range ``(min, max)`` for binning. If None, uses data range.

    Returns
    -------
    BinnedSignificanceResult
        Dataclass containing integrated significance, per-bin values,
        bin edges/centers, and signal/background yields per bin.

    Notes
    -----
    The integrated significance is computed as:

    .. math::

        Z_{\\text{int}} = \\sqrt{\\sum_{i=1}^{N} Z_i^2}

    This represents the total discriminating power when using the full
    shape of the classifier distribution, as in a binned likelihood fit.

    See Also
    --------
    scan_significance : Alternative using threshold cuts.
    """
    if score_range is None:
        score_range = (float(scores.min()), float(scores.max()))

    bin_edges = np.linspace(score_range[0], score_range[1], n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    sig_scores, sig_weights = scores[is_signal], weights[is_signal]
    bkg_scores, bkg_weights = scores[~is_signal], weights[~is_signal]

    signal_yields, _ = np.histogram(sig_scores, bins=bin_edges, weights=sig_weights)
    background_yields, _ = np.histogram(bkg_scores, bins=bin_edges, weights=bkg_weights)

    # Vectorized per-bin significance
    bin_significances = cast(np.ndarray, significance(signal_yields, background_yields))

    integrated_significance = float(np.sqrt(np.sum(bin_significances**2)))

    return BinnedSignificanceResult(
        integrated_significance=integrated_significance,
        bin_significances=bin_significances,
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        signal_yields=signal_yields,
        background_yields=background_yields,
    )


def compute_signal_score(
    predictions: np.ndarray,
    signal_indices: list[int],
    background_indices: list[int] | None = None,
    use_discriminant: bool = False,
) -> np.ndarray:
    """Compute signal score from multiclass predictions.

    Parameters
    ----------
    predictions : np.ndarray
        Softmax predictions, shape ``(n_events, n_classes)``.
    signal_indices : list[int]
        Class indices considered as signal.
    background_indices : list[int], optional
        Class indices considered as background. Required only if
        ``use_discriminant=True``.
    use_discriminant : bool, optional
        If True, return ``log(sum_sig / sum_bkg)``.
        Otherwise return sum of signal probabilities. Default is False.

    Returns
    -------
    np.ndarray
        Score per event, shape ``(n_events,)``.

    Raises
    ------
    ValueError
        If ``use_discriminant=True`` but ``background_indices`` is None.

    Notes
    -----
    The default score (sum of signal probabilities) ranges from 0 to 1
    for well-calibrated softmax outputs. The discriminant form provides
    better separation but is unbounded.
    """
    sum_sig = predictions[:, signal_indices].sum(axis=1)

    if not use_discriminant:
        return sum_sig

    if background_indices is None:
        raise ValueError("background_indices required when use_discriminant=True")

    sum_bkg = predictions[:, background_indices].sum(axis=1)
    return np.log(sum_sig / (sum_bkg + _EPSILON))
