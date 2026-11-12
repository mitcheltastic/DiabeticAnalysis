# Retraining Round v3: Protocol Amendment 1

- **Document**: `v3/PROTOCOL_AMENDMENT_1.md`
- **Parent Protocol**: `v3/PROTOCOL.md` (Commit: `1db8cb346d0a7dbf2be37a9fc77103fa723aa9e3`)
- **Date**: November 2026
- **Status**: Formally Adopted & Locked

---

## 1. Amendment Provisions

### A. Metric Consistency
1. **Experiments E3 (Honest Imputation Benchmark) and E4 (Honest Tuning)**:
   - Primary hypothesis testing (Nadeau-Bengio corrected resampled $t$-test and TOST equivalence test) is conducted on **fold-level ROC-AUC** across the 50 outer folds.
   - The TOST equivalence margin for ROC-AUC is set to $\Delta = \pm 0.02$.
   - Fold-level Accuracy tests (with equivalence margin $\Delta = \pm 0.02$) are computed and reported as secondary evaluations.
2. **Experiment E2 (Performance Inflation Ladder)**:
   - Report test ROC-AUC alongside Accuracy across all 100 seeds for every rung (R0 through R5, and R4').

### B. Probability Calibration Estimation
1. **Model Specification**:
   - Calibration slope and intercept are estimated via unpenalized logistic regression using `statsmodels.api.Logit`.
   - Predicted probabilities $p$ are clipped to $[10^{-6}, 1 - 10^{-6}]$ prior to logit transformation: $\text{logit}(p) = \ln(p / (1 - p))$.
   - **Calibration Slope**: Fitted via unpenalized regression:
     $$\text{logit}(P(Y=1)) = \alpha + \beta \cdot \text{logit}(p)$$
     where $\beta$ is the calibration slope.
   - **Calibration Intercept (Calibration-in-the-Large)**: Estimated with the slope fixed to $1.0$ (acting as an offset):
     $$\text{logit}(P(Y=1)) = \alpha_0 + 1.0 \cdot \text{logit}(p)$$
     where $\alpha_0$ represents calibration-in-the-large.
2. **Unit Test Standard**:
   - Evaluated on $N=5,000$ simulated instances with $p \sim \text{Uniform}(0.1, 0.9)$ and $y \sim \text{Bernoulli}(p)$.
   - Acceptance tolerance: Slope $\beta \in [0.90, 1.10]$ and Intercept $\alpha_0 \in [-0.10, +0.10]$.

### C. TOST Variance and Subsampling Ratios
1. **Realistic Cross-Validation Benchmarks**:
   - TOST test suite in `test_stats_utils.py` must include a synthetic verification with realistic fold variability ($s_d \approx 0.04$, $k=50$, $n_{\text{test}}/n_{\text{train}} = 1/9$).
   - The corrected standard error $\text{SE}_{\text{corr}} = \sqrt{\left(\frac{1}{k} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) s_d^2}$ and resulting $p_{\text{TOST}}$ are explicitly verified.
2. **Ladder Subsampling Ratio**:
   - For the 100-seed 80:20 train/test evaluation (Experiment E2), the ratio $n_{\text{test}} / n_{\text{train}}$ is explicitly fixed at $154 / 614 \approx 0.250814$.

### D. Honest RLTR Imputation Specification
1. **Similarity Function**:
   - For an instance $i$ missing target attribute $T$, similarity with donor candidate $j$ is evaluated on available attributes $A \in \{\text{Glucose}, \text{BMI}\} \setminus \{T\}$.
   - A donor candidate $j$ qualifies if and only if for all available attributes $a \in A$:
     $$\frac{|x_{i, a} - x_{j, a}|}{\max(|x_{i, a}|, |x_{j, a}|)} \le \epsilon$$
   - Both $x_{i, a}$ and $x_{j, a}$ must be observed and non-zero.
2. **Donor Pool Eligibility**:
   - Donor candidates are restricted strictly to the current training fold indices where target attribute $T$ is observed.
   - Eligibility never conditions on the class target label `Outcome`.
3. **Deterministic Fallback**:
   - If no candidate satisfies the $\epsilon$-similarity criterion (or if no similarity attribute is observed), the imputed value defaults to the training-fold median of $T$.
   - All fallback occurrences are logged per attribute per fold to `v3/results/rltr_fallback_log.csv`. If the fallback rate exceeds 50% for any attribute, it is flagged in the report.
4. **Tolerance Settings**:
   - Evaluated at $\epsilon = 0.15$ (published upperclassmen parameter) and $\epsilon = 0.25$.

### E. Sanity Anchors (Mandatory Hard Stops)
1. **Anchor 1 (RLTR Target Leakage Audit)**:
   - In Experiment E1, analysis of `RLTR_Imputed.csv` on missing insulin rows must reproduce the established Phase 2 estimates ($\beta_{\text{Outcome}} \approx +28.68\text{ mg/dL}$, partial correlation $r(\text{Insulin}, \text{Outcome} \mid \text{Glucose}, \text{BMI}) \approx +0.82$) before executing audits on the remaining four files.
2. **Anchor 2 (Paper Baseline Reproduction)**:
   - In Experiment E2, Rung R3 evaluated on holdout `seed=42` with default XGBoost must reproduce the exact baseline accuracy of $0.8506$ ($90\text{ TN}, 10\text{ FP}, 13\text{ FN}, 41\text{ TP}$). Any discrepancy requires immediate execution halt and reporting prior to proceeding with further rungs.

### F. A-Priori Methodological Decisions
1. **Pipeline Imputer**:
   - Simple in-fold median imputation is designated *a priori* as the standard imputer for Experiments E4 (Tuning), E5 (Clinical Calibration), and E7 (SHAP Stability) based on structural parsimony, computational stability, and non-inferiority, completely independent of E3 benchmark outcomes.
2. **Hypothesis Formulation**:
   - The class-conditional RLTR reconstruction evaluated in Experiment E1 is an independent mathematical hypothesis derived strictly from textual descriptions of rough-set decision tables, executed without access to or reconstruction of proprietary cohort scripts.
