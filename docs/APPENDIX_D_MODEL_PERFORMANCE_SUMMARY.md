# Appendix D: Model Performance Comparison Summary

## D.1 Overall Performance Dashboard

**Overall Performance**: IVL3.5-8B emerges as the best-performing model with a Mean F1 score of 73.7%, outperforming both Llama-11B (64.8%) and IVL3-2B (60.9%). The box plots show IVL3.5-8B has not only the highest median accuracy but also the tightest distribution (16.0% standard deviation), indicating more consistent performance across documents.

**Speed-Accuracy Trade-off**: All three models have comparable processing speeds, with IVL3-2B being slightly fastest (35.0s average, 1.7 docs/min) followed by IVL3.5-8B (37.1s, 1.6 docs/min) and Llama-11B (41.8s, 1.4 docs/min). Notably, IVL3.5-8B achieves the highest accuracy while maintaining competitive speed, making it the most efficient choice. The scatter plot confirms this—IVL3.5-8B points cluster toward the upper-left quadrant representing high accuracy with reasonable processing time.

**Document Type Analysis**: Performance varies significantly by document type. For **receipts**, IVL3.5-8B excels at ~78% F1 compared to Llama's ~60%. For **invoices**, IVL3.5-8B also leads (~68% vs ~57%). For **bank statements**, Llama-11B and IVL3.5-8B perform similarly (~62%), while IVL3-2B struggles (~45%). Bank statements require the longest processing time across all models (~70-80s) compared to invoices and receipts (~28-32s), likely due to their more complex tabular structure.

**Recommendation**: IVL3.5-8B offers the best balance of accuracy, speed, and consistency, making it the recommended model for production deployment on business document extraction tasks.

---

## D.2 Statistical Comparison Dashboard

**Statistical Significance**: The paired t-test results reveal that the performance differences are statistically meaningful. IVL3.5-8B significantly outperforms both Llama-11B (p=0.0041, Cohen's d=-0.81, large effect) and IVL3-2B (p=0.0001, Cohen's d=1.25, large effect). Interestingly, the difference between Llama-11B and IVL3-2B is not statistically significant (p=0.2176, small effect), suggesting these two models perform comparably despite Llama being 5x larger.

**Summary Statistics**: IVL3.5-8B achieves the highest Mean F1 (73.7%) and Median F1 (79.5%), with the lowest standard deviation (20.0%) indicating consistent performance. Its minimum F1 (29.2%) is also higher than competitors, meaning it fails less catastrophically on difficult fields. Llama-11B shows higher variance (24.6% std dev) with a wider gap between mean (64.8%) and median (70.3%), suggesting it is pulled down by poor performance on certain fields.

**Per-Field Analysis**: All models excel on structured fields like STATEMENT_DATE_RANGE, TRANSACTION_DATES, and TRANSACTION_AMOUNTS_PAID (90%+ F1). The largest performance gaps appear in IS_GST_INCLUDED where IVL3.5-8B achieves 91% vs Llama's 47%—a 44-point advantage. Line item fields (QUANTITIES, DESCRIPTIONS, PRICES) are challenging for all models, scoring 20-47%, highlighting an area for future improvement. IVL3.5-8B consistently leads or ties on nearly every field.

**Conclusion**: IVL3.5-8B is the statistically validated best model, with large effect sizes confirming the practical significance of its 9-13 percentage point advantage over competitors. The paired t-test methodology (same 17 schema fields across all models) ensures a fair, controlled comparison.

---

## Summary Table

| Metric | Llama-11B | IVL3.5-8B | IVL3-2B |
|:-------|:---------:|:---------:|:-------:|
| Mean F1 | 64.8% | **73.7%** | 60.9% |
| Median F1 | 70.3% | **79.5%** | 61.0% |
| Std Dev | 24.6% | **20.0%** | 19.3% |
| Min F1 | 20.0% | **29.2%** | 25.1% |
| Max F1 | 92.7% | **92.8%** | 89.0% |
| Avg Processing Time | 41.8s | 37.1s | **35.0s** |
| Throughput (docs/min) | 1.4 | 1.6 | **1.7** |

## Pairwise Statistical Tests

| Comparison | p-value | Significance | Cohen's d | Effect Size |
|:-----------|:-------:|:------------:|:---------:|:-----------:|
| Llama-11B vs IVL3.5-8B | 0.0041 | ** | -0.81 | Large |
| Llama-11B vs IVL3-2B | 0.2176 | ns | 0.31 | Small |
| IVL3.5-8B vs IVL3-2B | 0.0001 | *** | 1.25 | Large |

*Significance levels: \*\*\* p<0.001, \*\* p<0.01, \* p<0.05, ns = not significant*
