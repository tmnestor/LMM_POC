# Appendix C: Cohen's d Effect Size Methodology

## Overview

Cohen's d is a standardized measure of effect size that quantifies the magnitude of difference between two groups in terms of standard deviation units. While p-values indicate whether a difference is statistically significant (unlikely due to chance), Cohen's d indicates whether the difference is practically meaningful.

## Formula: Paired Samples

For comparing models across the same schema fields, we use the **paired samples** formula:

$$d = \frac{\bar{D}}{s_D}$$

Where:
- $\bar{D}$ = Mean of the difference scores (Model A F1 - Model B F1 per field)
- $s_D$ = Standard deviation of the difference scores

This paired approach is appropriate because each schema field serves as its own control, and the same 17 fields are evaluated across all models.

## Interpretation Thresholds

Cohen (1988) proposed conventional thresholds for interpreting effect sizes:

| Absolute Value of d | Interpretation | Practical Meaning |
|:-------------------:|:--------------:|:------------------|
| < 0.2 | Negligible | Difference too small to be meaningful |
| 0.2 – 0.5 | Small | Real but subtle difference |
| 0.5 – 0.8 | Medium | Difference visible in practice |
| > 0.8 | Large | Substantial, obvious difference |

## Application in This Study

Effect sizes were computed for all pairwise model comparisons:

| Comparison | Cohen's d | Interpretation |
|:-----------|:---------:|:--------------:|
| Llama-11B vs IVL3.5-8B | 0.72 | Medium |
| Llama-11B vs IVL3-2B | 0.76 | Medium |
| IVL3.5-8B vs IVL3-2B | 0.68 | Medium |

All comparisons show **medium effect sizes**, indicating that the performance differences between models are not only statistically significant but also practically meaningful for field-level extraction tasks.

## Why Effect Size Matters

Statistical significance (p-value) and practical significance (effect size) address different questions:

- **p-value**: "Is this difference real or due to chance?"
- **Cohen's d**: "Is this difference large enough to matter?"

With small sample sizes (n=17 schema fields), relying solely on p-values can be misleading. A study might show:
- Significant p-value with negligible effect size: statistically real but trivial
- Non-significant p-value with large effect size: meaningful difference but underpowered

Reporting both metrics provides a complete picture of model performance differences.

## References

Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

Lakens, D. (2013). Calculating and reporting effect sizes to facilitate cumulative science: A practical primer for t-tests and ANOVAs. *Frontiers in Psychology*, 4, 863. https://doi.org/10.3389/fpsyg.2013.00863
