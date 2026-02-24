# Significance Calculation

## Asimov Significance Formula

The framework uses the Asimov significance formula for a counting experiment:

$$
Z = \sqrt{2 \left[ (s + b) \ln\left(1 + \frac{s}{b}\right) - s \right]}
$$

where:
- $s$ = weighted sum of signal events
- $b$ = weighted sum of background events

This formula approximates the median discovery significance assuming the background-only hypothesis. It is derived from the profile likelihood ratio and is valid in the asymptotic limit (large statistics). The formula comes from the standard approach in high-energy physics for computing expected discovery significance without running full toy Monte Carlo experiments.

### Edge Cases

The formula requires careful handling at the boundaries:

- **If $b \leq 0$:** Return $Z = 0$. This occurs when all background is removed by a cut, which would naively give infinite significance but is unphysical.
- **If $s < 0$:** Clip to $s = 0$. Negative signal yields can occur due to negative MC weights (common in NLO generators like MC@NLO). The significance cannot be negative, so we clip.

### Physical Interpretation

The Asimov significance $Z$ can be interpreted as the number of standard deviations by which the observed data would deviate from the background-only hypothesis if the signal is present at the expected rate. A significance of $Z = 5$ corresponds to a p-value of approximately $3 \times 10^{-7}$, the conventional threshold for discovery in particle physics.

---

## Signal Score Definition

For multiclass classification, the signal score is computed from the softmax predictions of the neural network:

$$
\text{score} = \sum_{i \in \text{signal classes}} P_i
$$

where $P_i$ is the softmax probability for class $i$.

The signal and background class groupings are defined in the configuration. For example, in a ttH analysis:

```yaml
ttH_vs_ttbar:
  signal:
    - ttH_bb
    - ttH_cc
  background:
    - ttbar_enhanced_HF1b1B
    - ttbar_enhanced_HF1c1C
    - ttbar_enhanced_HFlight
```

This configuration defines which physics processes are considered signal and which are background for the purpose of significance calculation. The classifier may have many output classes, but for significance we group them into two categories: signal-like and background-like.

When computing the signal score, we sum the softmax probabilities of all signal classes. This gives a single number in $[0, 1]$ representing how signal-like an event is according to the classifier.

---

## Maximum Significance (Threshold Scan)

The maximum significance metric finds the optimal working point by scanning over threshold cuts on the signal score.

### Procedure

1. **Define thresholds:** Generate a set of threshold values $t \in [0, 1]$. By default, 200 evenly spaced points are used.

2. **For each threshold $t$:**
   - Select events with signal score $\geq t$
   - Compute signal yield: $s = \sum w_i$ for signal events passing cut
   - Compute background yield: $b = \sum w_i$ for background events passing cut
   - Calculate $Z(t)$ using the Asimov formula

3. **Find the optimum:**
   - $Z_{\max} = \max_t Z(t)$
   - $t_{\text{opt}}$ = threshold value that achieves $Z_{\max}$

### Interpretation

The maximum significance represents the **best achievable significance with a single cut** on the classifier output. This is directly relevant for cut-based analyses where a working point needs to be chosen.

The optimal threshold $t_{\text{opt}}$ is the classifier score cut that maximizes discovery potential. Events with scores above this threshold would be selected in a cut-based analysis.

---

## Integrated Significance (Binned)

The integrated significance metric uses the full shape of the classifier output distribution by binning the signal score.

### Procedure

1. **Divide the score range:** Split $[0, 1]$ into $N$ bins (default: 10 bins with equal width)

2. **For each bin $i$:**
   - $s_i$ = weighted signal yield in bin
   - $b_i$ = weighted background yield in bin
   - $Z_i$ = Asimov significance computed for that bin

3. **Combine in quadrature:**

$$
Z_{\text{int}} = \sqrt{\sum_{i=1}^{N} Z_i^2}
$$

---

## MC Weight Handling

All significance calculations use MC event weights $w_i$. This is essential because:

- Different physics processes have different cross-sections
- Events are normalized to the expected luminosity
- NLO generators can produce negative weights

### Negative Weights

Negative weights (common in NLO MC generators like Powheg and MC@NLO) require special handling:

- In threshold scans: negative weights can cause the yield to decrease as more events are added
- In binned significance: bins can have negative yields if negative-weight events dominate

When negative yields occur, they are clipped to zero with a warning. This is a conservative approach that avoids unphysical (imaginary) significance values.

### Weight Sum Interpretation

The weighted sum $s = \sum w_i$ represents the expected number of events in the analysis region, normalized to the target luminosity. For example, if training on Run 2 data scaled to 140 fb$^{-1}$, the significance is computed for that luminosity.

---

## Training Integration

During model training, significance is automatically tracked and logged. This allows monitoring whether the classifier is improving in terms of physics performance, not just classification metrics like loss or accuracy.

### Configuration

Significance tracking is configured in the training configuration file:

```yaml
significance_config:
  ttH_vs_ttbar:
    signal:
      - ttH_bb
      - ttH_cc
    background:
      - ttbar
```

Multiple significance configurations can be defined to track different signal hypotheses simultaneously.

### Logged Metrics

The following metrics are logged to MLflow at each evaluation:

- `significance/max_significance` - The $Z_{\max}$ value
- `significance/optimal_threshold` - The threshold achieving $Z_{\max}$
- `significance/integrated_significance` - The $Z_{\text{int}}$ value

### Generated Plots

The tracker automatically generates diagnostic plots:

1. **Significance vs Threshold Curve:** Shows $Z(t)$ for all thresholds scanned. The peak indicates the optimal working point. A broad peak suggests robustness to threshold choice.

2. **Per-Bin Significance Distribution:** Shows $Z_i$ for each score bin. Ideally, high-score bins should dominate the significance. If low-score bins contribute significantly, it may indicate classification issues.

---

## References

- Cowan, G., Cranmer, K., Gross, E., & Vitells, O. (2011). Asymptotic formulae for likelihood-based tests of new physics. *Eur. Phys. J. C*, 71, 1554. [arXiv:1007.1727](https://arxiv.org/abs/1007.1727)

- The original Asimov paper derives the asymptotic formulae and defines the "Asimov dataset" as a representative dataset where all observations equal their expected values.
