# Retraining Round v3: Comprehensive Experimental Audit and Results Report

- **Document**: `v3/RESULTS_v3.md`
- **Scope**: Complete experimental findings covering Experiments E1 through E7, Replication Study E2b, and Headline Subgroup Benchmarks
- **Dataset**: PIMA Indian Diabetes Cohort ($N = 768$, $d = 8$, binary classification)
- **Principal Investigator / Supervisory Team**: Dr. R. Yunendah Nur Fu'adah, S.T., M.T., Ph.D.; Student: Mitchel Mohamad Affandi
- **Date**: 2026-11-12
- **Standard Protocol**: Pre-registered protocol (`v3/PROTOCOL.md`), Protocol Amendment 1 (`v3/PROTOCOL_AMENDMENT_1.md`), Protocol Amendment 2 (`v3/PROTOCOL_AMENDMENT_2.md`)

---

## 1. Methods Summary

### 1.1 Dataset and Missing Data Topology
The PIMA Indian Diabetes cohort contains 768 female patients of Pima Indian heritage aged $\ge 21$ years, evaluated for type 2 diabetes mellitus (Outcome: 500 negative [$65.10\%$], 268 positive [$34.90\%$]). Biologically impossible zero values represent missing physiological measurements:
- `Glucose`: 5 missing ($0.65\%$)
- `BloodPressure`: 35 missing ($4.56\%$)
- `SkinThickness`: 227 missing ($29.56\%$)
- `Insulin`: 374 missing ($48.70\%$)
- `BMI`: 11 missing ($1.43\%$)
- `Pregnancies`, `DiabetesPedigreeFunction`, `Age`: 0 missing ($0.00\%$)

A total of 376 patients ($48.96\%$) contain at least one missing value. Complete biological records (listwise complete) exist for 392 patients ($51.04\%$), while 394 patients have observed `Insulin` values.

### 1.2 Evaluation Protocols and Partitioning
To resolve data leakage and evaluation optimism present in earlier literature, all experiments adhere to strict data hygiene:
1. **100-Seed Stratified Holdout Benchmark**: 100 independent random splits ($80\%$ train [$n=614$], $20\%$ test [$n=154$], stratified on Outcome, seeds $0$ to $99$). Preprocessing (imputation, scaling) is fit exclusively on training rows and applied to test rows. Enforced at runtime via `assert_disjoint(train_idx, test_idx)`.
2. **Nested 50-Fold Cross-Validation**: $10 \times 5$-fold Repeated Stratified Cross-Validation (`RepeatedStratifiedKFold`, random state 42). Hyperparameter optimization and operating threshold selection execute inside inner 5-fold cross-validation loops, evaluating exclusively on unseen outer test folds.

### 1.3 Statistical Testing Framework
All paired model comparisons across resampled folds or split seeds employ the **Nadeau-Bengio corrected resampled $t$-test** to account for non-independence of overlapping training partitions:
$$\text{Var}_{\text{NB}}(\bar{d}) = \left(\frac{1}{k} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) S^2$$
where $k$ is the number of resamples (100 for split seeds, 50 for cross-validation), $n_{\text{test}} = 154$, and $n_{\text{train}} = 614$. Equivalence testing employs Two-One-Sided Tests (TOST) against pre-registered margins ($\pm 0.015$ accuracy, $\pm 0.020$ ROC-AUC). Multi-model comparisons control familywise error rates using Holm-Bonferroni step-down adjustments.

---

## 2. Experiment E1: Data Leakage Audit of Imputed Artifacts

- **Data Sources**: `v3/results/E1_leakage_all_files.csv`, `v3/results/E1_rltr_row_agreement.csv`, `v3/results/E1_spec_reconciliation.md`

### 2.1 Three-Tier Evidence Structure
Experiment E1 audited the five rough-set imputation files utilized in published literature: Tolerance Relation (`TR`), Limited Tolerance Relation (`LTR`), Relative Limited Tolerance Relation (`RLTR`), Non-Symmetric Similarity Relation (`NSSR`), and Similarity Relation (`SIM`).

| Evidence Tier | Benchmark Description | Observed Data Baseline ($N=394$) | RLTR Imputed File ($N=768$) | Statistical Comparison | Finding |
|---|---|:---:|:---:|:---:|:---:|
| **Tier 1: Bivariate Target Association** | Point-biserial correlation $r(\text{Insulin}, \text{Outcome})$ | $r = +0.2163$<br>($95\%\text{ CI: } [0.1200, 0.3082]$) | $r = +0.6729$<br>($95\%\text{ CI: } [0.6322, 0.7099]$) | Fisher $z$-test: $z = 10.05$<br>$p = 4.41 \times 10^{-24}$ | Imputed insulin correlation is three-fold higher than observed biology. |
| **Tier 2: Conditional Independence** | Multiple OLS regression coefficient of Outcome on Insulin controlling for all 7 covariates | $\beta = -6.16\text{ mg/dL}$<br>($t = -0.484, p = 0.6285$) | $\beta = +28.68\text{ mg/dL}$<br>($t = 26.77, p = 3.97 \times 10^{-88}$) | Contrast: $\Delta\beta = +34.84\text{ mg/dL}$<br>$95\%\text{ CI: } [26.58, 30.79]$ | Conditioning on biology eliminates target correlation in raw data, but artificial dependency remains in RLTR. |
| **Tier 3: Specification Reconciliation** | Algorithmic lineage and formula reconstruction | Unimputed raw clinical measurements | Non-parametric rough-set relative tolerance relation | Relational approximation matrix incorporates Outcome class | The imputation method incorporates class labels into the neighborhood relation. |

### 2.2 Complete 5-File Conditional Dependency Audit
Multiple OLS regression coefficients for `Outcome` predicting `Insulin` while adjusting for all seven remaining clinical predictors (`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `BMI`, `DiabetesPedigreeFunction`, `Age`):

| File Key | Imputation Relation | Sample Size ($N$) | Outcome Coefficient ($\beta$) | Standard Error | $t$-statistic | $p$-value | $95\%$ Confidence Interval | Target Dependency Classification |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raw Observed** | Complete cases (unimputed) | 392 | $-6.16\text{ mg/dL}$ | $12.74$ | $-0.484$ | $0.6285$ | $[-31.18, +18.86]$ | No target dependency ($p > 0.60$) |
| **TR** | Tolerance Relation | 768 | $+0.81\text{ mg/dL}$ | $1.10$ | $+0.738$ | $0.4608$ | $[-1.34, +2.97]$ | Compatible with weak dependency ($p = 0.46$) |
| **LTR** | Limited Tolerance Relation | 768 | $+17.58\text{ mg/dL}$ | $1.44$ | $+12.18$ | $1.01 \times 10^{-30}$ | $[+14.75, +20.41]$ | Statistically detectable target dependency |
| **RLTR** | Relative Limited Tolerance Relation | 768 | $+28.68\text{ mg/dL}$ | $1.07$ | $+26.77$ | $3.97 \times 10^{-88}$ | $[+26.58, +30.79]$ | Severe artificial target dependency |
| **NSSR** | Non-Symmetric Similarity Relation | 768 | $-2.62\text{ mg/dL}$ | $2.29$ | $-1.146$ | $0.2520$ | $[-7.11, +1.87]$ | Compatible with weak dependency ($p = 0.25$) |
| **SIM** | Similarity Relation | 768 | $+0.48\text{ mg/dL}$ | $1.07$ | $+0.447$ | $0.6552$ | $[-1.63, +2.58]$ | Compatible with weak dependency ($p = 0.65$) |

*Source: `v3/results/E1_leakage_all_files.csv`*.

---

## 3. Experiment E2: The Inflation Ladder Benchmark (100 Seeds)

- **Data Sources**: `v3/results/E2_inflation_ladder.csv`, `v3/results/E2_seed_level_results.csv`, `v3/results/E2_sensitivity_true_defaults.csv`, `v3/results/E2_sensitivity_inflation_gaps.csv`

### 3.1 Inflation Ladder Across 100 Seeds
The inflation ladder decomposes performance inflation from leak-free modeling up to direct test-set optimization. Each step evaluates 100 stratified $80/20$ splits:

| Ladder Rung | Methodological Condition | 100-Seed Accuracy (Mean ± SD) | 95% Percentile Range | Paired $\Delta\text{Acc}$ vs. R0 ($p_{\text{NB}}$) | 100-Seed ROC-AUC (Mean ± SD) | Seed 42 Accuracy | Seed 42 Confusion Matrix |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **R0** | Leak-Free Baseline (In-fold median + scale, XGB reference) | $75.25\% \pm 3.06\%$ | $[69.14\%, 80.86\%]$ | Baseline | $0.8163 \pm 0.0296$ | $75.32\%$ | $(84, 16, 22, 32)$ |
| **R1** | Global Scaling Leakage (Pre-split scaling + in-fold median) | $75.23\% \pm 3.09\%$ | $[68.83\%, 80.86\%]$ | $-0.01\text{ pp}$ ($p = 0.9414$) | $0.8166 \pm 0.0297$ | $75.32\%$ | $(84, 16, 22, 32)$ |
| **R2** | Global MICE Leakage (Pre-split MICE + scaling) | $75.05\% \pm 3.20\%$ | $[68.83\%, 81.17\%]$ | $-0.20\text{ pp}$ ($p = 0.6548$) | $0.8166 \pm 0.0298$ | $74.68\%$ | $(84, 16, 23, 31)$ |
| **R3a_TR** | TR Imputed As-Is (Pre-split MinMax) | $75.45\% \pm 3.20\%$ | $[68.83\%, 81.82\%]$ | $+0.20\text{ pp}$ ($p = 0.8554$) | $0.8270 \pm 0.0315$ | $75.32\%$ | $(84, 16, 22, 32)$ |
| **R3a_LTR** | LTR Imputed As-Is (Pre-split MinMax) | $74.37\% \pm 2.99\%$ | $[68.83\%, 80.52\%]$ | $-0.88\text{ pp}$ ($p = 0.4845$) | $0.8147 \pm 0.0293$ | $74.68\%$ | $(83, 17, 22, 32)$ |
| **R3a_NSSR** | NSSR Imputed As-Is (Pre-split MinMax) | $74.86\% \pm 3.14\%$ | $[68.18\%, 80.52\%]$ | $-0.39\text{ pp}$ ($p = 0.7406$) | $0.8127 \pm 0.0312$ | $73.38\%$ | $(82, 18, 23, 31)$ |
| **R3a_SIM** | SIM Imputed As-Is (Pre-split MinMax) | $75.75\% \pm 3.12\%$ | $[69.48\%, 81.82\%]$ | $+0.50\text{ pp}$ ($p = 0.6870$) | $0.8358 \pm 0.0308$ | $74.03\%$ | $(84, 16, 24, 30)$ |
| **R3a_RLTR** | RLTR Imputed As-Is (Published Paper Baseline) | **$83.69\% \pm 2.65\%$** | $[78.57\%, 88.96\%]$ | **$+8.45\text{ pp}$ ($p = 3.90 \times 10^{-7}$)** | **$0.8998 \pm 0.0217$** | **$85.06\%$** | **$(90, 10, 13, 41)$** |
| **R3b_RLTR** | RLTR with In-Fold Glucose Imputation Fix | $83.69\% \pm 2.67\%$ | $[77.92\%, 88.96\%]$ | $+8.45\text{ pp}$ ($p = 3.91 \times 10^{-7}$) | $0.8996 \pm 0.0217$ | $85.06\%$ | $(90, 10, 13, 41)$ |
| **R4'** | Honest Train-Set CV Tuning (Optuna 50 trials) | $83.21\% \pm 2.62\%$ | $[77.92\%, 88.31\%]$ | $+7.96\text{ pp}$ ($p = 1.09 \times 10^{-6}$) | $0.9009 \pm 0.0215$ | $84.42\%$ | $(89, 11, 13, 41)$ |
| **R4-50** | Test-Set Objective Tuning (Optuna 50 trials) | $85.86\% \pm 2.19\%$ | $[81.82\%, 89.61\%]$ | $+10.61\text{ pp}$ ($p = 8.16 \times 10^{-10}$) | $0.8930 \pm 0.0223$ | $87.01\%$ | $(92, 8, 12, 42)$ |
| **R4-100** | Test-Set Objective Tuning (Optuna 100 trials) | $86.45\% \pm 2.06\%$ | $[82.47\%, 90.26\%]$ | $+11.20\text{ pp}$ ($p = 1.54 \times 10^{-10}$) | $0.8911 \pm 0.0222$ | $87.01\%$ | $(92, 8, 12, 42)$ |
| **R5** | Maximum Seed Selection Effect | $89.61\%$ | Single split maximum | — | $0.9231$ | — | $(94, 6, 10, 44)$ |

*Note*: The difference between honest inner tuning (R4') and direct test-set objective tuning (R4-50) is $\Delta\text{Accuracy} = +2.65\text{ pp}$ ($95\%\text{ CI: } [2.01, 3.29]\text{ pp}, p_{\text{NB}} = 1.15 \times 10^{-11}$).

### 3.2 Sensitivity Check: True Library Defaults vs. Reference Configuration
Sensitivity testing across 100 seeds compares the XGBoost reference configuration (`learning_rate=0.1, max_depth=6, n_estimators=100`) against true `xgboost 3.4` library defaults (`learning_rate=0.3`):
- **Leak-Free R0 Baseline**: True defaults yield $74.33\% \pm 2.84\%$ accuracy and $0.7990 \pm 0.0306$ ROC-AUC.
- **RLTR R3a**: True defaults yield $83.54\% \pm 2.64\%$ accuracy and $0.8971 \pm 0.0226$ ROC-AUC.
- **Inflation Gap ($\text{RLTR} - \text{R0}$)**: Under true defaults, the gap is $\Delta\text{Accuracy} = +9.21\text{ pp}$ ($95\%\text{ CI: } [6.95, 11.47]\text{ pp}, p_{\text{NB}} = 3.34 \times 10^{-9}$) and $\Delta\text{AUC} = +0.0981$ ($p_{\text{NB}} = 6.88 \times 10^{-12}$).
- **Invariance**: Target leakage inflation remains statistically significant and of comparable magnitude regardless of whether learning rate is $0.1$ or $0.3$.

*Source: `v3/results/E2_sensitivity_true_defaults.csv`, `v3/results/E2_sensitivity_inflation_gaps.csv`*.

---

## 4. Experiment E2b: Paper Replication and Implied Matrix Derivations

- **Data Sources**: `v3/results/E2b_paper1_table2.csv`, `v3/results/E2b_paper2_replication.csv`

### 4.1 Implied Confusion Matrix Derivation
On $n_{\text{test}} = 154$ with known marginal totals (negative $N_0 = 100$, positive $N_1 = 54$):
- Specificity: $\text{Spec} = TN / 100 \implies TN = \text{round}(100 \times \text{Spec})$
- Sensitivity / Recall: $\text{Sens} = TP / 54 \implies TP = \text{round}(54 \times \text{Sens})$
- False Positives: $FP = 100 - TN$; False Negatives: $FN = 54 - TP$
- Accuracy: $\text{Acc} = (TN + TP) / 154$

### 4.2 Replication Evidence Tiers
1. **Tier 1 (Exact Deterministic Reproduction)**:
   - **Paper 1 Table II XGBoost Row**: Reported Acc $85.06\%$, Precision $0.80$, Recall $0.76$, Specificity $0.90$, $F_1 = 0.78$.
   - **Our R3a RLTR Reference Model**: Acc $85.0649\%$, Precision $0.8039$, Recall $0.7593$, Specificity $0.9000$, $F_1 = 0.7810$, Matrix: **$(90, 10, 13, 41)$**. Replicated exactly on Seed 42.
2. **Tier 2 (Consistent Optimization Matches)**:
   - **Paper 2 RLTR XGB+Optuna Row**: Reported Acc $87.01\%$, Weighted Precision $86.90\%$, Recall $87.01\%$, $F_1 = 86.89\%$.
   - **Our Direct Test-Tuned R4-50 Model**: Acc $87.0130\%$, Weighted Precision $86.90\%$, Recall $87.01\%$, $F_1 = 86.89\%$, Matrix: **$(92, 8, 12, 42)$**. Implied matrix matches uniquely.
   - **Paper 2 LTR XGB+Optuna Row**: Reported Acc $77.27\%$, Weighted Precision $76.81\%$, Recall $77.27\%$, $F_1 = 76.87\%$.
   - **Our Honest Inner-CV Tuned LTR Model**: Acc $77.2727\%$, Weighted Precision $76.81\%$, Recall $77.27\%$, $F_1 = 76.87\%$, Matrix: **$(86, 14, 21, 33)$**.
3. **Tier 3 (Coincidental Matches & Matrix Divergence)**:
   - **Paper 1 KNN**: Reported $74.00\%$. Sklearn default $k=5$ yields $77.27\%$ ($85, 15, 20, 34$). Setting $k=4$ yields $74.03\%$ with matrix $(81, 19, 21, 33)$, which diverges from the implied matrix of Paper 1. Labeled coincidental.

*Source: `v3/results/E2b_paper1_table2.csv`, `v3/results/E2b_paper2_replication.csv`*.

---

## 5. Experiment E3: Honest Imputation Benchmark Across 50 Folds

- **Data Sources**: `v3/results/E3_honest_imputation_benchmark.csv`, `v3/results/E3_fold_level_results.csv`

Experiment E3 evaluated six leak-free imputation strategies inside a 50-fold cross-validation scheme ($10 \times 5$-fold Repeated Stratified K-Fold) across three model families:

| Model Family | Imputation Strategy | 50-Fold Accuracy (Mean ± SD) | Paired $\Delta\text{Acc}$ vs. Median ($p_{\text{NB}}$) | TOST Equivalence ($\pm 0.015, p_{\text{TOST}}$) | 50-Fold ROC-AUC (Mean ± SD) |
|---|---|:---:|:---:|:---:|:---:|
| **Logistic Regression** | Median | $0.7719 \pm 0.0357$ | Baseline | — | $0.8354 \pm 0.0374$ |
| | Mean | $0.7717 \pm 0.0360$ | $-0.03\text{ pp}$ ($p = 0.902$) | $p < 0.0001$ | $0.8351 \pm 0.0375$ |
| | MICE (BayesianRidge) | $0.7725 \pm 0.0360$ | $+0.05\text{ pp}$ ($p = 0.812$) | $p < 0.0001$ | $0.8357 \pm 0.0373$ |
| | MissForest | $0.7722 \pm 0.0358$ | $+0.03\text{ pp}$ ($p = 0.902$) | $p < 0.0001$ | $0.8364 \pm 0.0375$ |
| | Honest RLTR ($\epsilon=0.15$) | $0.7727 \pm 0.0355$ | $+0.08\text{ pp}$ ($p = 0.707$) | $p < 0.0001$ | $0.8359 \pm 0.0374$ |
| | Honest RLTR ($\epsilon=0.25$) | $0.7714 \pm 0.0361$ | $-0.05\text{ pp}$ ($p = 0.818$) | $p < 0.0001$ | $0.8354 \pm 0.0375$ |
| **XGBoost Reference** | Median | $0.7758 \pm 0.0396$ | Baseline | — | $0.8306 \pm 0.0392$ |
| | Mean | $0.7745 \pm 0.0401$ | $-0.13\text{ pp}$ ($p = 0.697$) | $p < 0.0001$ | $0.8305 \pm 0.0392$ |
| | MICE (BayesianRidge) | $0.7745 \pm 0.0400$ | $-0.13\text{ pp}$ ($p = 0.669$) | $p < 0.0001$ | $0.8300 \pm 0.0389$ |
| | MissForest | $0.7758 \pm 0.0405$ | $+0.00\text{ pp}$ ($p = 1.000$) | $p < 0.0001$ | $0.8307 \pm 0.0392$ |
| | Honest RLTR ($\epsilon=0.15$) | $0.7748 \pm 0.0393$ | $-0.10\text{ pp}$ ($p = 0.760$) | $p < 0.0001$ | $0.8296 \pm 0.0396$ |
| | Honest RLTR ($\epsilon=0.25$) | $0.7742 \pm 0.0396$ | $-0.16\text{ pp}$ ($p = 0.627$) | $p < 0.0001$ | $0.8301 \pm 0.0394$ |
| **CatBoost** | Median | $0.7797 \pm 0.0366$ | Baseline | — | $0.8406 \pm 0.0362$ |
| | Mean | $0.7807 \pm 0.0369$ | $+0.10\text{ pp}$ ($p = 0.742$) | $p < 0.0001$ | $0.8415 \pm 0.0362$ |
| | MICE (BayesianRidge) | $0.7797 \pm 0.0371$ | $+0.00\text{ pp}$ ($p = 1.000$) | $p < 0.0001$ | $0.8411 \pm 0.0362$ |
| | MissForest | $0.7802 \pm 0.0373$ | $+0.05\text{ pp}$ ($p = 0.865$) | $p < 0.0001$ | $0.8410 \pm 0.0361$ |
| | Honest RLTR ($\epsilon=0.15$) | $0.7812 \pm 0.0367$ | $+0.15\text{ pp}$ ($p = 0.627$) | $p < 0.0001$ | $0.8407 \pm 0.0362$ |
| | Honest RLTR ($\epsilon=0.25$) | $0.7815 \pm 0.0368$ | $+0.18\text{ pp}$ ($p = 0.569$) | $p < 0.0001$ | $0.8414 \pm 0.0362$ |

**Finding**: No imputation method produces a statistically detectable improvement over simple in-fold median imputation across any model family ($p \ge 0.442$). All 15 comparisons satisfy TOST statistical equivalence within the $\pm 0.015$ accuracy margin ($p_{\text{TOST}} < 0.0001$).

---

## 6. Experiment E4: Honest Nested CV Tuning and Final Model Selection

- **Data Sources**: `v3/results/E4_honest_tuning_nested_cv.csv`, `v3/results/E4_hyperparameter_stability.csv`, `v3/results/E4_fold_level_results.csv`

### 6.1 Outer-Fold Performance and Protocol Amendment 2 Decision Rule
In accordance with pre-registered Protocol Amendment 2:
- If no tuned model beats LogReg-L2 default on outer-fold ROC-AUC with a Holm-adjusted Nadeau-Bengio $p < 0.05$, the final model is LogReg-L2 default (parsimony).
- Candidate evaluation over 50 outer folds:

| Model | Hyperparameter Status | Outer-Fold ROC-AUC (Mean ± SD) | Outer-Fold Accuracy (Mean ± SD) | Brier Score | Calibration Slope | CITL | Paired $\Delta\text{AUC}$ vs. Baseline | Unadjusted $p_{\text{NB}}$ | Holm-Adjusted $p_{\text{Holm}}$ | TOST Equivalence ($\pm 0.02\text{ AUC}$) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **LogReg-L2** | Default Baseline ($C=1.0$) | **$0.8357 \pm 0.0381$** | **$0.7745 \pm 0.0366$** | $0.1601 \pm 0.0169$ | $0.9859 \pm 0.2312$ | $-0.0211 \pm 0.2185$ | Baseline | — | — | Baseline |
| **LogReg-EN** | Tuned (50 trials) | $0.8369 \pm 0.0383$ | $0.7745 \pm 0.0361$ | $0.1598 \pm 0.0170$ | $0.9822 \pm 0.2321$ | $-0.0215 \pm 0.2188$ | $+0.0012$ | $0.5847$ | $1.0000$ | $p < 0.0001$ |
| **XGBoost** | Tuned (50 trials) | $0.8379 \pm 0.0396$ | $0.7600 \pm 0.0401$ | $0.1610 \pm 0.0188$ | $0.9634 \pm 0.2451$ | $-0.0189 \pm 0.2215$ | $+0.0022$ | $0.6121$ | $1.0000$ | $p < 0.0001$ |
| **CatBoost** | Tuned (50 trials) | $0.8417 \pm 0.0363$ | $0.7748 \pm 0.0371$ | $0.1582 \pm 0.0165$ | $0.9712 \pm 0.2398$ | $-0.0201 \pm 0.2201$ | $+0.0060$ | $0.4492$ | $1.0000$ | $p = 0.0014$ |

**Decision**: No tuned model achieves statistically significant superiority over LogReg-L2 default ($p_{\text{Holm}} = 1.0000$). All models are statistically equivalent within the $\pm 0.020$ AUC margin. **LogReg-L2 default with in-fold median imputation and StandardScaler is selected as the final model.**

### 6.2 Tuning Optimism and Search Space Saturation
- **Tuning Optimism**: Inner CV validation AUC exceeded outer-fold test AUC across all models: XGBoost $+0.0147$ ($0.8526$ inner vs. $0.8379$ outer); CatBoost $+0.0103$ ($0.8520$ inner vs. $0.8417$ outer); LogReg-EN $+0.0038$ ($0.8407$ inner vs. $0.8369$ outer).
- **Lower Search-Space Boundary Saturation**:
  - XGBoost `max_depth` (range $[2, 8]$) selected the lower bound ($=2$) in **$78.0\%$ ($39/50$)** of outer folds.
  - CatBoost `depth` (range $[3, 8]$) selected the lower bound ($=3$) in **$68.0\%$ ($34/50$)** of outer folds.
  - Consistent with heavy regularization being necessary on small tabular data ($N=768$).

---

## 7. Experiment E5: Operating Decision Thresholds, Calibration, and DCA

- **Data Sources**: `v3/results/E5_threshold_calibration_summary.csv`, `v3/results/E5_seed42_holdout_results.csv`, `v3/results/E5_decision_curve_analysis.csv`

### 7.1 Nested 50-Fold Cross-Validation Operating Policies
Evaluated strictly through out-of-fold outer predictions with inner-fold threshold optimization:

| Clinical Operating Policy | Inner-Loop Selection Rule | Mean Threshold ($\tau$) | Sensitivity (Mean ± SD) | Specificity (Mean ± SD) | $F_1$ Score (Mean ± SD) | Accuracy (Mean ± SD) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Default** | Standard uncalibrated probability cut | $0.500$ | $0.5580 \pm 0.0827$ | $0.8788 \pm 0.0416$ | $0.6242 \pm 0.0638$ | $0.7669 \pm 0.0371$ |
| **Policy A** | Maximize $F_1$ score | $0.297 \pm 0.041$ | $0.7747 \pm 0.0934$ | $0.7020 \pm 0.0664$ | $0.6646 \pm 0.0526$ | $0.7275 \pm 0.0461$ |
| **Policy B** | Sensitivity $\ge 0.80$ with maximal specificity | $0.280 \pm 0.038$ | $0.8043 \pm 0.0888$ | $0.6848 \pm 0.0716$ | $0.6725 \pm 0.0531$ | $0.7265 \pm 0.0467$ |

### 7.2 Seed 42 Holdout Evaluation (`train_test_split`, $n=154$)
Thresholds selected on 614 training rows and evaluated on the unseen test partition:

| Operating Policy | Applied Threshold ($\tau$) | Accuracy | ROC-AUC | Brier Score | Calibration Slope | CITL | Sensitivity | Specificity | $F_1$ Score | Confusion Matrix |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Default** | $0.50$ | $70.78\%$ | $0.8130$ | $0.1754$ | $0.7608$ | $-0.0062$ | $0.5000$ | $0.8200$ | $0.5455$ | $(82, 18, 27, 27)$ |
| **Policy A** | $0.41$ | $71.43\%$ | $0.8130$ | $0.1754$ | $0.7608$ | $-0.0062$ | $0.5926$ | $0.7800$ | $0.5926$ | $(78, 22, 22, 32)$ |
| **Policy B** | $0.28$ | $72.73\%$ | $0.8130$ | $0.1754$ | $0.7608$ | $-0.0062$ | $0.8333$ | $0.6700$ | $0.6818$ | $(67, 33, 9, 45)$ |

### 7.3 Decision Curve Analysis (DCA)
Decision Curve Analysis confirms the final model provides positive clinical net benefit over "treat all" across threshold probabilities $p_t \in [0.10, 0.55]$. At $p_t = 0.20$, clinical net benefit is $0.2458$ (equivalent to identifying 25 true positive cases per 100 patients without false positive intervention).

---

## 8. Experiment E6: Subgroup Audit and Feature Ablation (100 Seeds)

- **Data Sources**: `v3/results/E6_multiseed_subgroup_summary.csv`, `v3/results/E6_multiseed_ablation_summary.csv`, `v3/results/E6_headline_subgroup_tests.csv`, `v3/results/E6_seed42_exact_subgroups.csv`, `v3/results/E6_seed42_exact_ablation.csv`

### 8.1 Subgroup Performance by Insulin Missingness (100 Seeds)
Evaluated across 100 independent stratified splits:

| Model Architecture | Test Partition Subgroup | Accuracy (Mean ± SD) | 95% Percentile Interval | ROC-AUC (Mean ± SD) | Paired $\Delta$ (Observed − Missing) | Nadeau-Bengio 95% CI | Corrected $p$-value |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **RLTR R3a Reference XGBoost** | Missing Insulin ($n\approx 84$) | $92.00\% \pm 3.31\%$ | $[85.37\%, 97.50\%]$ | $0.9700 \pm 0.0174$ | — | — | — |
| | Observed Insulin ($n\approx 70$) | $75.75\% \pm 3.87\%$ | $[67.81\%, 82.50\%]$ | $0.8060 \pm 0.0449$ | $\Delta\text{Acc} = -16.25\text{ pp}$<br>$\Delta\text{AUC} = -0.1640$ | $[-21.15, -11.35]\text{ pp}$<br>$[-0.2133, -0.1147]$ | $p = 2.25 \times 10^{-9}$<br>$p = 2.10 \times 10^{-9}$ |
| **Leak-Free LogReg-L2 (Final Model)** | Missing Insulin ($n\approx 84$) | $75.43\% \pm 3.64\%$ | $[67.78\%, 81.23\%]$ | $0.8316 \pm 0.0364$ | — | — | — |
| | Observed Insulin ($n\approx 70$) | $77.68\% \pm 3.63\%$ | $[71.60\%, 85.37\%]$ | $0.8446 \pm 0.0362$ | $\Delta\text{Acc} = +2.26\text{ pp}$<br>$\Delta\text{AUC} = +0.0130$ | $[-2.90, +7.41]\text{ pp}$<br>$[-0.0372, +0.0631]$ | $p = 0.3871$<br>$p = 0.6085$ |
| **Leak-Free Tuned XGBoost** | Missing Insulin ($n\approx 84$) | $73.59\% \pm 4.40\%$ | $[63.26\%, 81.48\%]$ | $0.8285 \pm 0.0426$ | — | — | — |
| | Observed Insulin ($n\approx 70$) | $78.10\% \pm 3.93\%$ | $[71.43\%, 86.17\%]$ | $0.8501 \pm 0.0375$ | $\Delta\text{Acc} = +4.51\text{ pp}$<br>$\Delta\text{AUC} = +0.0216$ | $[-1.28, +10.30]\text{ pp}$<br>$[-0.0335, +0.0767]$ | $p = 0.1251$<br>$p = 0.4387$ |

### 8.2 Headline Subgroup Tests
1. **Model Comparison on Observed-Insulin Patients**:
   - On the genuine biological subgroup ($n\approx 70$), Leak-Free LogReg-L2 achieves higher discrimination than the RLTR-trained model:
     $$\Delta\text{ROC-AUC} (\text{LogReg} - \text{RLTR}) = +0.0386 \quad (95\%\text{ CI: } [+0.0039, +0.0732], p_{\text{NB}} = 0.0295)$$
   - Accuracy difference is not statistically significant: $\Delta\text{Accuracy} = +1.93\text{ pp}$ ($p_{\text{NB}} = 0.3435$).
2. **Difference-in-Differences Interaction**:
   - Contrast: $[(\text{RLTR Observed} - \text{RLTR Missing}) - (\text{LogReg Observed} - \text{LogReg Missing})]$:
     $$\Delta\Delta\text{Accuracy} = -18.51\text{ pp} \quad (95\%\text{ CI: } [-23.56, -13.45]\text{ pp}, p_{\text{NB}} = 8.98 \times 10^{-11})$$
     $$\Delta\Delta\text{ROC-AUC} = -0.1770 \quad (95\%\text{ CI: } [-0.2266, -0.1274], p_{\text{NB}} = 2.15 \times 10^{-10})$$
   - Consistent with artificial performance inflation in RLTR being isolated to patients whose insulin was synthetically imputed.

### 8.3 Retrained Feature Ablation (100 Seeds)
Models retrained from scratch on reduced feature sets:
- **Leak-Free LogReg-L2**:
  - Full Model (8 features): Acc $76.60\% \pm 2.62\%$, AUC $0.8369 \pm 0.0267$
  - No Insulin (7 features): Acc $76.36\% \pm 2.66\%$, AUC $0.8384 \pm 0.0262$ ($\Delta\text{Acc} = -0.24\text{ pp}, p = 0.4960$; $\Delta\text{AUC} = +0.0015, p = 0.3781$)
  - No Insulin + SkinThickness (6 features): Acc $76.49\% \pm 2.61\%$, AUC $0.8395 \pm 0.0264$ ($\Delta\text{Acc} = -0.10\text{ pp}, p = 0.8006$; $\Delta\text{AUC} = +0.0025, p = 0.2223$)
- **Leak-Free Tuned XGBoost**:
  - Full Model (8 features): Acc $75.90\% \pm 3.07\%$, AUC $0.8400 \pm 0.0290$
  - No Insulin (7 features): Acc $76.03\% \pm 3.07\%$, AUC $0.8390 \pm 0.0292$ ($\Delta\text{Acc} = +0.12\text{ pp}, p = 0.8456$; $\Delta\text{AUC} = -0.0010, p = 0.7100$)
  - No Insulin + SkinThickness (6 features): Acc $76.12\% \pm 3.13\%$, AUC $0.8393 \pm 0.0296$ ($\Delta\text{Acc} = +0.21\text{ pp}, p = 0.7674$; $\Delta\text{AUC} = -0.0006, p = 0.8287$)

---

## 9. Experiment E7: Bootstrap SHAP Feature Importance Stability

- **Data Source**: `v3/results/E7_bootstrap_feature_importance.csv`

Evaluated across 200 bootstrap refits on the 614 training rows and explained on the holdout partition ($n=154$):

| Feature | LogReg-L2 Mean \|SHAP\| | LogReg-L2 Median Rank [IQR] | LogReg-L2 Top-3 Freq | Tuned XGBoost Mean \|SHAP\| | Tuned XGBoost Median Rank [IQR] | Tuned XGBoost Top-3 Freq |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Glucose** | $1.0535 \pm 0.1088$ | **$1.0$ [$0.0$]** | $100.0\%$ | $0.8407 \pm 0.0926$ | **$1.0$ [$0.0$]** | $100.0\%$ |
| **BMI** | $0.5809 \pm 0.1066$ | **$2.0$ [$0.0$]** | $100.0\%$ | $0.4835 \pm 0.0845$ | **$2.0$ [$0.0$]** | $100.0\%$ |
| **Pregnancies** | $0.3298 \pm 0.1198$ | $3.0$ [$0.0$] | $77.0\%$ | $0.1178 \pm 0.0587$ | $6.0$ [$2.0$] | $3.5\%$ |
| **DiabetesPedigreeFunction** | $0.1800 \pm 0.0894$ | $4.0$ [$1.25$] | $7.5\%$ | $0.1984 \pm 0.0718$ | $4.0$ [$1.0$] | $18.0\%$ |
| **Age** | $0.1417 \pm 0.0933$ | $5.0$ [$2.0$] | $14.5\%$ | $0.2919 \pm 0.0887$ | $3.0$ [$1.0$] | $74.5\%$ |
| **Insulin** | $0.0902 \pm 0.0612$ | **$6.0$ [$2.0$]** | $1.0\%$ | $0.1421 \pm 0.0641$ | **$5.0$ [$2.0$]** | $3.5\%$ |
| **SkinThickness** | $0.0615 \pm 0.0455$ | **$7.0$ [$2.0$]** | $0.0\%$ | $0.0656 \pm 0.0463$ | **$7.0$ [$2.0$]** | $0.5\%$ |
| **BloodPressure** | $0.0727 \pm 0.0551$ | **$7.0$ [$2.0$]** | $0.0\%$ | $0.0414 \pm 0.0294$ | **$8.0$ [$1.0$]** | $0.0\%$ |

**Finding**: Glucose and BMI consistently occupy the top two ranks in $100.0\%$ of bootstrap refits across both linear and non-linear models. Insulin and SkinThickness rank in the bottom tier (median ranks 5 to 7), confirming that once Glucose and BMI are known, missing-indicator features offer negligible marginal discriminatory contribution.

---

## 10. Study Limitations

1. **Tree Depth Search Space Saturation**: In Experiment E4, hyperparameter optimization hit the lower boundary in $78\%$ of XGBoost folds (`max_depth=2`) and $68\%$ of CatBoost folds (`depth=3`), consistent with small-sample tabular constraints where shallow trees regularize variance.
2. **Holdout Sample Variability (Seed 42 vs. 100-Seed Mean)**: On Seed 42, the final model achieved $70.78\%$ accuracy, whereas across 100 random seeds it averaged $76.60\% \pm 2.62\%$ ($95\%$ percentile range: $70.78\%\text{–}81.51\%$). Single test splits on $n=154$ are subject to sampling variance.
3. **Listwise Complete Sample Size**: Unimputed baseline regression is restricted to complete records ($N=392$), reducing statistical power relative to the full cohort ($N=768$).
4. **Single-Cohort Geographic Constraint**: Data originates exclusively from female patients of Pima Indian heritage collected between 1965 and 1990; findings require validation on contemporary multi-ethnic cohorts.
5. **Accuracy Discreteness on Test Set ($n=154$)**: A single classification outcome alters accuracy by $1/154 \approx 0.649\text{ percentage points}$. Coincidences in accuracy across models or seeds can arise by chance.
6. **Upperclassmen Search Space Underspecification**: Original published manuscripts do not fully specify random seed schedules, Optuna parameter search intervals, or complete dependency manifests.

---

## 11. Cross-Reference to Documentation Corrections

All historical discrepancies, parameter clarifications, and narrative updates from Phase 2/Phase 3 reports are documented in:
- [CORRECTIONS_TO_V2.md](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/5.1%20EXTRA%20BANTU%20BU%20YUNEN/DiabeticAnalysis/v3/results/CORRECTIONS_TO_V2.md) (Items 1 through 12).
