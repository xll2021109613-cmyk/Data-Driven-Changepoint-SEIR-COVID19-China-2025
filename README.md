# Data-Driven-Changepoint-SEIR-COVID19-China-2025
Data-driven changepoint SEIR model fitted to China's 2025 COVID-19 wave using official China CDC data — with sensitivity analysis, residual diagnostics, and full reproducibility.

## Key Results

| Metric | Constant-beta SEIR | Free-changepoint SEIR |
|--------|-------------------|----------------------|
| R² | 0.669 | **0.869** |
| RMSE | 2,955.4 | **1,860.3** |
| AIC | 2,451.4 | **2,313.7** |

- **Estimated changepoint:** May 24, 2025 (AIC-selected; precedes observed peak of May 26 by 2 days)
- **Reproduction numbers:** R_before = 1.40 (> 1, growth phase); R_after = 0.88 (< 1, decline phase)
- **Model-predicted peak:** May 23, 2025 (20,384 cases/day) vs. observed peak May 26 (22,684 cases/day)
- **Robustness:** conclusions hold across all 19 sensitivity scenarios (5 changepoints × 5 observation-scale factors × 9 biological parameter combinations)
- **Residual diagnostics:** significant weekly reporting cycle (weekend vs. weekday, Welch p < 0.001; lag-7 autocorrelation 0.70)

## Repository Structure

