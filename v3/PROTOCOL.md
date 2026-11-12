# Retraining Round v3: Experimental Protocol & Specifications

- **Investigator / Student**: Mitchel Mohamad Affandi
- **Principal Investigator / Supervisor**: Dr. R. Yunendah Nur Fu'adah, S.T., M.T., Ph.D.
- **Protocol Version**: 3.0 (Pre-registered)
- **Git Commit Hash**: `ed854740205b9c816df909c699635d03ba7f3b99`
- **Date**: November 2026
- **Status**: Registered & Locked Prior to Execution

---

## 1. Primary Objectives & Scientific Mandate

The objective of Round v3 is to execute a rigorous, leak-free re-evaluation of machine learning pipelines for diabetes diagnosis on the PIMA Indian dataset ($N=768$), comparing directly against findings and baselines reported in prior literature from upperclassmen while maintaining methodological defensibility.

### 1.1 Independence Policy
All experiments use only:
1. The published raw dataset (`Dataset Diabetes.csv`).
2. The five published imputation files (`LTR_Imputed.csv`, `NSSR_Imputed.csv`, `RLTR_Imputed.csv`, `SIM_Imputed.csv`, `TR_Imputed.csv`).
3. Explicit textual descriptions in upperclassmen papers (e.g., RLTR on Glucose, BMI, and Insulin with $\epsilon = 0.15$; 80:20 stratified split; random_state=42; default tree architectures; Optuna TPE tuning).

No proprietary or unreleased code from previous cohorts is used, requested, or reconstructed. Where previous literature is silent on implementation details, assumptions are made explicit in this protocol.

### 1.2 Scientific Tone & Neutrality
All analyses and report deliverables adhere to formal, neutral scientific terminology:
- Observed phenomena are described as *"statistically consistent with"* specific mechanisms (e.g., class-conditional donor selection), avoiding subjective or speculative verbs.
- No promotional phrasing or decorative iconography is used in technical reports.

---

## 2. Evaluation Metrics Hierarchy

### 2.1 Primary Metric
- **ROC-AUC** computed under nested repeated cross-validation.
- **Co-Primary**: Probability Calibration, quantified by:
  - Calibration Slope (target: 1.0)
  - Calibration Intercept (target: 0.0)
  - Brier Score Loss

### 2.2 Secondary Metrics
- Accuracy
- Sensitivity (Recall of diabetic class, $Y=1$)
- Specificity (Recall of healthy class, $Y=0$)
- $F_1$-score (Positive/diabetic class)
- Precision-Recall AUC (PR-AUC / Average Precision)
- Weighted Precision, Weighted Recall, Weighted $F_1$-score (to ensure direct comparability with upperclassmen Paper 2)

---

## 3. Methodological Ground Rules

1. **Strict Temporal and Fold Isolation**:
   - Every stateful transformation—including imputation, feature scaling, hyperparameter tuning, feature selection, and decision threshold calibration—must be fitted strictly on the training indices of the active loop.
   - Test indices are evaluated strictly out-of-sample.
   - Enforced programmatically via an assertion check: `assert_disjoint(train_idx, test_idx)` called in every cross-validation and bootstrap loop.

2. **Validation Frameworks**:
   - **Primary Model Estimate**: Repeated Nested Cross-Validation on all 768 rows:
     - Outer loop: `RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)` ($k=50$ outer test folds).
     - Inner loop: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` ($5$ inner folds for model selection/tuning).
   - **Secondary Benchmark (Paper-Comparable)**: Single holdout 80:20 stratified split (`test_size=0.20, random_state=42`, $N_{\text{train}}=614, N_{\text{test}}=154$). Any model selection or tuning uses only the 614 training instances. The test split is evaluated exactly once per pipeline.

3. **Physiological Data Hygiene**:
   - Across raw data and all five published files, non-physiological zeros in `Glucose` ($N=5$) are treated as missing data (`np.nan`) and imputed strictly inside training folds. Global imputation across the full dataset is prohibited in leak-free tracks.

4. **Statistical Inference Tools (`v3/scripts/stats_utils.py`)**:
   - **Nadeau-Bengio Corrected Resampled t-Test**: Used for all paired cross-validation score comparisons ($k=50$, $n_{\text{test}}/n_{\text{train}} = 1/9$).
   - **Two One-Sided Tests (TOST) Equivalence**: Evaluated using Nadeau-Bengio corrected variance under an equivalence margin of $\Delta = \pm 0.02$ accuracy / AUC.
   - **Holm-Bonferroni Procedure**: Step-down family-wise error rate control for all multi-hypothesis testing suites.
   - **McNemar Exact Test**: Two-tailed exact binomial test on discordant test set predictions ($b$ vs $c$) for paired holdout comparisons.

---

## 4. Experiment Specifications (E1 through E7)

### Experiment E1: Comparative Leakage Audit of All Five Imputation Files
- **Data Examined**: `LTR_Imputed.csv`, `NSSR_Imputed.csv`, `RLTR_Imputed.csv`, `SIM_Imputed.csv`, `TR_Imputed.csv`, compared against raw observed cases in `Dataset Diabetes.csv`.
- **Target Subsets**:
  1. Patients with missing Insulin in raw data (`Insulin == 0`, $N=374$).
  2. Patients with missing SkinThickness in raw data (`SkinThickness == 0`, $N=227$).
- **Statistical Tests**:
  - Multiple OLS regression: $\text{Insulin} = \beta_0 + \beta_1 \text{Glucose} + \beta_2 \text{BMI} + \beta_3 \text{Outcome}$. Report $\beta_3$, $t$-statistic, and $p$-value.
  - Partial correlation: $r(\text{Insulin}, \text{Outcome} \mid \text{Glucose}, \text{BMI})$.
  - Bivariate Pearson correlation: $r(\text{Insulin}, \text{Outcome})$.
- **Row-Level Reconstruction Verification**:
  - Re-implement RLTR with class-conditional donor restriction (`Outcome == target`).
  - Measure agreement with provided `RLTR_Imputed.csv` on the 374 missing insulin entries: Mean Absolute Difference (MAD), Spearman rank correlation ($\rho$), and proportion of predictions within $\pm 5\text{ mg/dL}$.
- **Output**: `v3/results/E1_leakage_all_files.csv` and `v3/results/E1_rltr_row_agreement.csv`.

### Experiment E2: The Performance Inflation Ladder (Centerpiece)
- **Model Architecture**: Fixed to XGBoost (`XGBClassifier`, baseline hyperparameters: `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `eval_metric='logloss'`).
- **Evaluation**: 100 stratified random splits (80:20, seeds $0 \le s \le 99$).
- **Rungs of the Inflation Ladder**:
  - **R0 (Leak-Free Baseline)**: In-fold Median imputation + in-fold scaling + default XGBoost.
  - **R1 (Global Scaling Leakage)**: Feature scaling fitted on all 768 rows prior to split + in-fold median imputation + default XGBoost.
  - **R2 (Global Unsupervised Imputation Leakage)**: MICE (`IterativeImputer` with BayesianRidge, label-free) fitted on all 768 rows prior to split + global scaling + default XGBoost.
  - **R3 (Class-Conditional Feature File)**: Evaluation on `RLTR_Imputed.csv` with 5 glucose zeros imputed in-fold + global MinMaxScaler (matching upperclassmen setup).
  - **R4 (Direct Test Set Optimization)**: R3 setup + Optuna hyperparameter optimization (100 trials, TPE sampler) directly maximizing test set accuracy.
  - **R5 (Selection Bias via Multi-Seed Search)**: Maximum test accuracy observed across seeds $0 \le s \le 29$ under R4.
  - **R4' (Honest Hyperparameter Tuning on RLTR)**: R3 setup + Optuna hyperparameter optimization (100 trials, TPE sampler) maximizing 5-fold CV accuracy on training data only.
- **Specific Checkpoints**:
  - R3 performance on `seed=42` holdout (target check: exact reproduction of $0.8506$ / confusion matrix $90/10/13/41$).
  - R4 / R4' performance on `seed=42` holdout (target check against Paper 2's reported $0.8701$).
- **Outputs**: `v3/results/E2_inflation_ladder.csv` and waterfall diagram `v3/figures/E2_inflation_ladder_waterfall.png`.

### Experiment E3: Honest Benchmark of Imputation Strategies
- **Classifiers Tested**:
  1. Logistic Regression with $L_2$ penalty (StandardScaler).
  2. XGBoost (`XGBClassifier`, default settings).
- **Imputation Strategies Evaluated**:
  1. Median Imputation (`SimpleImputer(strategy='median')`).
  2. Mean Imputation (`SimpleImputer(strategy='mean')`).
  3. $k$-Nearest Neighbors (`KNNImputer(n_neighbors=5)`).
  4. MICE (`IterativeImputer(estimator=BayesianRidge(), max_iter=10)`).
  5. MissForest (`IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=50), max_iter=10)`).
  6. Honest RLTR ($\epsilon=0.15$, distance computed strictly on standardized Glucose and BMI; donor pool restricted to training fold; no Outcome used).
  7. Honest RLTR ($\epsilon=0.25$, same setup).
  8. Native missing value handling + missing indicators (evaluated for XGBoost and HistGradientBoosting).
- **Cross-Validation**: 50 outer folds (10-fold CV repeated 5 times).
- **Statistical Testing**:
  - Nadeau-Bengio corrected resampled difference vs. Median Imputer.
  - Holm-Bonferroni corrected $p$-values.
  - TOST equivalence test ($\Delta = \pm 0.02$).
- **Output**: `v3/results/E3_honest_imputation_benchmark.csv`.

### Experiment E4: Honest Hyperparameter Optimization (Nested CV)
- **Validation**: 50 outer folds (10x5 RepeatedStratifiedKFold), each with an inner 5-fold StratifiedKFold.
- **A Priori Selected Imputer**: Simple Median Imputation (selected based on parsimony and non-inferiority demonstrated in E3).
- **Models Tuned**:
  1. XGBoost (`max_depth`: 2–8, `n_estimators`: 50–300, `learning_rate`: 0.01–0.3, `subsample`: 0.6–1.0, `colsample_bytree`: 0.6–1.0).
  2. CatBoost (`depth`: 3–8, `iterations`: 100–500, `learning_rate`: 0.01–0.2, `l2_leaf_reg`: 1–10).
  3. Logistic Regression with ElasticNet penalty (`C`: $10^{-3}$ to $10^2$, `l1_ratio`: 0.0 to 1.0, solver `saga`).
- **Inner Optimization**: Optuna with TPE sampler (50 trials per inner loop) maximizing ROC-AUC.
- **Evaluation**: Compare tuned models against their default counterparts using the Nadeau-Bengio corrected test. Log chosen hyperparameters across all 50 folds.
- **Output**: `v3/results/E4_honest_tuning_nested_cv.csv` and `v3/results/E4_hyperparameter_stability.csv`.

### Experiment E5: Clinical Decision Threshold Calibration & Decision Curve Analysis
- **Threshold Calibration**: Determined strictly inside the inner cross-validation loop:
  - Policy A: Threshold maximizing $F_1$-score on out-of-fold predictions.
  - Policy B: Threshold achieving Sensitivity $\ge 0.80$ while maximizing Specificity.
- **Out-of-Sample Evaluation**: Apply inner-calibrated thresholds to outer test folds; report test sensitivity, specificity, and $F_1$.
- **Calibration Analysis**:
  - Calibration slope and intercept via logistic calibration.
  - Reliability diagrams (10 bins).
  - Decision Curve Analysis (DCA): Net Benefit across clinical threshold probabilities $p_t \in [0.05, 0.60]$.
- **Holdout Validation**: Evaluate the same calibration rules on the seed-42 holdout set.
- **Outputs**: `v3/results/E5_threshold_calibration.csv`, `v3/figures/E5_reliability_diagram.png`, `v3/figures/E5_decision_curve_analysis.png`.

### Experiment E6: Subgroup Reliability & Deployment Realism
- **Cohort Stratification (Seed-42 Holdout, $N=154$)**:
  - Subgroup A: Patients with observed raw insulin ($N_{\text{obs}} \approx 79$).
  - Subgroup B: Patients with missing raw insulin ($N_{\text{mis}} \approx 75$).
- **Comparative Models**:
  - Model 1: Upperclassmen baseline XGBoost on `RLTR_Imputed.csv`.
  - Model 2: Leak-free tuned pipeline model.
- **Deployment Realism Simulation**:
  - Simulate low-resource primary care clinics where laboratory insulin and skinfold caliper measurements are unavailable at point of care.
  - At inference time, omit Insulin and SkinThickness from test instances (re-imputing or evaluating without them in the leak-free pipeline).
  - Quantify performance degradation.
- **Output**: `v3/results/E6_subgroup_and_deployment_realism.csv`.

### Experiment E7: Explainability Stability under Bootstrap Resampling
- **Methodology**: 200 non-parametric bootstrap resamples ($B=200$) of the training cohort ($N=614$).
- **Refitting**: Refit the final leak-free model on each bootstrap sample.
- **SHAP Computation**: Calculate TreeSHAP attribution values on the holdout test set for each model.
- **Metric**: Rank stability distribution: percentage of iterations each feature occupies Rank 1, 2, 3, etc. Compare against the rank stability of the RLTR model.
- **Outputs**: `v3/results/E7_shap_rank_stability.csv` and `v3/figures/E7_shap_rank_distribution.png`.

---

## 5. Explicit Assumptions Where Prior Papers Are Silent

1. **Pre-split Scaling**: Upperclassmen Paper 1 describes applying MinMaxScaler to features. We assume this was applied globally prior to splitting in their initial exploratory scripts, consistent with standard novice pipeline construction.
2. **RLTR Donor Restriction**: The textual descriptions mention Robust Linear Trend Regression with $\epsilon=0.15$. The text does not explicitly document conditioning on `Outcome`; our experiment assesses whether donor restriction by class reproduces their exact values.
3. **Hyperparameter Tuning Strategy**: Upperclassmen Paper 2 mentions Optuna TPE tuning. Where exact search boundaries were omitted in the text, standard broad bounds are adopted as defined in Section 4.

---

## 6. Estimated Compute Runtimes

| Experiment | Configuration & Workload | Estimated Runtime | Status / Action Rule |
|---|---|:---:|---|
| **E1** | OLS & Correlation audits across 5 CSVs (768 rows) | ~15 seconds | Run full |
| **E2** | 100 seeds × R0–R3 (fast default XGB fits: ~20s)<br>R4 (Optuna 100 trials on test set × 100 seeds: ~8 min)<br>R4' (Optuna 100 trials on 5-fold CV × 100 seeds: ~35 min) | ~45–55 minutes | Well under 3h threshold; run full 100 seeds |
| **E3** | 50 outer folds × 8 imputers × 2 models = 800 fits | ~4–6 minutes | Run full |
| **E4** | 50 outer folds × 3 models × 50 inner trials × 5 folds<br>(XGBoost: ~6 min; LogReg: ~1 min; CatBoost: ~15 min) | ~25–35 minutes | Run full |
| **E5** | 50 outer folds inner threshold calibration + DCA | ~2 minutes | Run full |
| **E6** | Seed-42 holdout subgroup partition & feature drop | ~10 seconds | Run full |
| **E7** | 200 bootstrap refits + TreeSHAP calculations | ~1–2 minutes | Run full |
| **Total Suite** | All 7 experiments sequentially | **~1.5 hours** | Proceed sequentially with per-experiment reports |
