# Defensible Machine Learning for Early Diabetes Diagnosis: Leak-Free PIMA Benchmark & Target Leakage Audit

**Project**: Academic Research Paper Preparation with Dr. R. Yunendah Nur Fu'adah, S.T., M.T., Ph.D.  
**Student / Investigator**: Mitchel Mohamad Affandi  
**Date**: October 2026 (Patch v2 Applied)  
**Execution Environment**: Python 3.12 (scikit-learn, XGBoost, CatBoost, LightGBM, Statsmodels, SciPy, SHAP)  
**Mandate**: Maximum Defensibility — every number must withstand rigorous scrutiny without data leakage.

---

## Executive Summary: The Defensibility Mandate

In medical machine learning, **a lower honest number beats an inflated flawed number every single time**. A metric that collapses under examination by a peer reviewer or thesis examiner damages academic credibility, whereas a rigorous, fully audited result that accounts for every percentage point represents genuine, publishable scientific progress.

This benchmark resolves the fundamental anomalies identified in the preliminary study:
1. **The Upperclassmen Baseline (0.8506)**: Successfully replicated to the exact single patient ($\text{TN}=90, \text{FP}=10, \text{FN}=13, \text{TP}=41$, Accuracy $0.8506$, Precision $0.8039$, Recall $0.7593$, Specificity $0.9000$, $F_1 = 0.7810$). *Note: No peer-reviewed evidence supports colloquial claims of "0.88".*
2. **Empirical Proof of Target Leakage via Class-Conditional Mechanism**:
   - In multiple OLS regression controlling for Glucose and BMI, `Outcome` has an artificial coefficient of $+28.68\text{ mg/dL}$ ($t = 26.77, p = 4.24 \times 10^{-77}$) in `RLTR_Imputed.csv`, whereas in real observed data, `Outcome` has no statistically significant predictive effect on insulin ($t = -0.45, p = 0.653$).
   - Partial correlation $r(\text{Insulin}, \text{Outcome} \mid \text{Glucose}, \text{BMI})$ is $+0.8208$ ($p < 10^{-90}$) in `RLTR_Imputed.csv` versus $-0.0168$ ($p = 0.740$) in real data.
   - When we explicitly tested the class-conditional hypothesis by re-implementing RLTR and restricting donor pools by target class, the resulting imputed insulin correlation with Outcome reached **$r = 0.7908$**, matching the provided file's **$r = 0.7895$** to within $0.0013$.
3. **Restoring Biological Reality via Leak-Free Explainability (Rank-Only SHAP)**:
   - In `RLTR_Imputed.csv`, SHAP ranked Insulin #1 and Glucose #2 (or #4 with feature engineering).
   - Under our leak-free protocol, **Glucose returns to Rank #1** and BMI rises to Rank #3, fully restoring endocrinological consistency.
4. **Honest Leak-Free Benchmark & Nadeau-Bengio Statistical Testing**:
   - On 50 repeated stratified folds, leak-free models achieve **$76.5\% - 77.5\%$ cross-validated accuracy** ($0.843$ ROC-AUC).
   - Under the Nadeau-Bengio corrected resampled t-test (accounting for fold overlap variance), no complex imputer statistically significantly outperforms simple median imputation ($p > 0.05$ after Holm correction).
   - On the untouched 154-patient holdout test set (`random_state=42`), the frozen leak-free ensemble scores **$74.68\%$ accuracy** ($95\%\text{ CI } [67.5\% - 81.8\%]$, ROC-AUC $0.8135$).
5. **Contextual Disclosure of Holdout Seed 12**:
   - The 89.61% accuracy figure was identified during an exploratory multi-seed scan ($0 \le \text{seed} \le 29$). Reporting the maximum across splits introduces selection bias; across all 30 splits, the mean was $\approx 83.5\%$, and on the standard seed 42 split, it was $83.77\%$. Combined with target leakage in `RLTR_Imputed.csv`, the 89.61% figure was doubly confounded.

---

## Phase 1: Baseline Reproduction & Statistical Tests

We replicated the exact methodology of the upperclassmen paper: default XGBoost on `RLTR_Imputed.csv`, 80:20 train/test split with `random_state=42`, stratified by target ($N_{\text{test}} = 154$: 100 healthy, 54 diabetic).

### Table 1: Upperclassmen Baseline Replication & Model Comparison

| Evaluation Split | Model Architecture | TN | FP | FN | TP | Accuracy [95% CI] | Sensitivity [95% CI] | Specificity [95% CI] | F1-Score [95% CI] | McNemar $p$-value vs Paper |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Seed 42 (Paper Split)** | **Upperclassmen XGBoost** | **90** | **10** | **13** | **41** | **0.8506** [0.792 - 0.903] | **0.7593** [0.638 - 0.869] | **0.9000** [0.840 - 0.952] | **0.7810** [0.684 - 0.863] | Baseline ($b=0, c=0$) |
| Seed 42 (Paper Split) | Consensus Ensemble (v1) | 88 | 12 | 13 | 41 | 0.8377 [0.779 - 0.896] | 0.7593 [0.633 - 0.870] | 0.8800 [0.813 - 0.939] | 0.7664 [0.667 - 0.851] | $p = 0.7539$ ($b=6, c=4$) |
| **Seed 12 (v1 Demo Split)** | Upperclassmen XGBoost | 92 | 8 | 10 | 44 | 0.8831 [0.831 - 0.929] | 0.8148 [0.708 - 0.913] | 0.9200 [0.863 - 0.968] | 0.8302 [0.748 - 0.898] | Baseline ($b=0, c=0$) |
| Seed 12 (v1 Demo Split) | Consensus Ensemble (v1) | 93 | 7 | 9 | 45 | 0.8961 [0.844 - 0.942] | 0.8333 [0.729 - 0.927] | 0.9300 [0.876 - 0.979] | 0.8491 [0.773 - 0.911] | $p = 0.7266$ ($b=3, c=5$) |

> **Key Finding 1**: The upperclassmen paper result of `0.8506` is replicated **to the exact single patient**.  
> **Key Finding 2**: The McNemar exact test shows **no statistically significant difference** ($p = 0.7539$ on Seed 42; $p = 0.7266$ on Seed 12) between the Paper XGBoost and the Consensus Ensemble.  
> **Key Finding 3**: The shift from `0.8506` to `0.8831` (and `0.8961`) was driven purely by changing the holdout seed from 42 to 12. Single-split holdout metrics have a 95% margin of error of $\pm 5.5\%$, proving why cross-validation must be the primary metric.

---

## Phase 2: RLTR Target Leakage Audit

We tested whether the imputed values in `RLTR_Imputed.csv` (374 patients with missing insulin) were mathematically derived using the target label (`Outcome`).

### Table 2: Mathematical Proof of Target Leakage in RLTR

| Mathematical Test | Real Observed Data ($N=394$) | Honest RLTR Re-implementation | Class-Conditional RLTR Experiment | Provided `RLTR_Imputed.csv` ($N=374$) | Theoretical / Clinical Interpretation |
|---|:---:|:---:|:---:|:---:|---|
| **Partial Correlation $r(\text{Ins}, Y \mid \text{Gluc}, \text{BMI})$** | **$-0.0168$** ($p = 0.740$) | **$+0.1240$** | **$+0.8262$** ($p < 10^{-90}$) | **$+0.8208$** ($p = 3.65 \times 10^{-93}$) | Massive artificial diagnostic signal preserved after removing all glucose and adiposity variance. |
| **Outcome Regression Coef ($\beta_{\text{Outcome}}$)** | $-5.72\text{ mg/dL}$ | $+7.74\text{ mg/dL}$ | $+29.08\text{ mg/dL}$ | **$+28.68\text{ mg/dL}$** | An artificial $+28.68\text{ mg/dL}$ was directly injected into diabetic patients' missing insulin. |
| **Outcome $t$-Statistic** | $t = -0.449$ ($p = 0.653$) | $t = 1.056$ ($p = 0.292$) | $t = 27.60$ ($p < 10^{-75}$) | **$t = 26.772$** ($p = 4.24 \times 10^{-77}$) | In real clinical biology, Outcome is NOT significant once Glucose and BMI are known. |
| **Correlation with Target $r(\text{Insulin}, \text{Outcome})$** | $r = 0.3014$ | $r = 0.5129$ | **$r = 0.7908$** | **$r = 0.7895$** | Class-conditional RLTR matches provided file to within $0.0013$. |
| **Mean Absolute Difference vs Honest RLTR** | N/A | Baseline ($0.0\text{ mg/dL}$) | $32.48\text{ mg/dL}$ | **$33.66\text{ mg/dL}$** | Deviates substantially from genuine epsilon-similarity imputation. |

$^\dagger$*Footnote on Linear Upper Bound ($R_{\max}$)*: The theoretical maximum linear multiple correlation possible from the other 7 clinical features on the missing rows is $R_{\max} = \sqrt{\mathbf{r}_{YX}^T \mathbf{R}_{XX}^{-1} \mathbf{r}_{YX}} = 0.5299$. The correlation observed in `RLTR_Imputed.csv` ($r = 0.7895$) exceeds this linear upper bound by $+49\%$. While non-linear feature interactions can theoretically exceed linear bounds, the partial correlation and class-conditional re-imputation experiments confirm that this elevation was directly caused by conditioning donor pools on `Outcome`.

> **Official Audit Verdict**: **LABEL LEAKAGE CONFIRMED**. Restricting donor pools to matching target classes (`Outcome == target`) directly reproduces $r = 0.7908 \approx 0.7895$ and $\beta = 29.08 \approx 28.68$. Consequently, `RLTR_Imputed.csv` cannot be used to claim clinical predictive accuracy. The paper's scientific contribution is the discovery, documentation, and mathematical proof of this leakage.

---

## Phase 3: Leak-Free Imputation Benchmark (50 Folds)

Evaluated under `RepeatedStratifiedKFold(n_splits=10, n_repeats=5)` (50 paired folds). All imputers and scalers were fitted strictly inside training folds. Zero test data was touched.

*Methodological Note on Honest RLTR*: In preliminary script iterations, integer column indexing (`c != 4`) against pandas DataFrames with string column names caused `HonestRLTRImputer` to fall back to median values. With integer-safe array indexing restored, Honest RLTR computes genuine donor-pool averages strictly on training data. On Logistic Regression, Honest RLTR achieves $0.7666 \pm 0.0430$ (vs Median $0.7676 \pm 0.0425$, Mean Diff $-0.0010$, 95% CI $[-0.0067, +0.0046]$, $t = -0.367, p = 0.7150$), verifying completely independent runs.

### Table 3: Imputer Benchmark & Nadeau-Bengio Corrected Resampled t-Tests vs Median Baseline

Statistical significance was evaluated using the **Nadeau-Bengio (2003) corrected resampled t-test**, which corrects for the variance underestimation inherent in repeated cross-validation where training folds overlap ($n_2 / n_1 = 1/9$). All $p$-values are family-wise adjusted using the **Holm-Bonferroni step-down procedure** ($\alpha = 0.05$).

| Imputation Strategy | Top Classifier Architecture | 50-Fold CV Accuracy (Mean $\pm$ Std) | ROC-AUC (Mean) | F1-Score (Mean) | Mean Diff vs Median (CatBoost) | Nadeau-Bengio 95% CI | Nadeau-Bengio $t$-stat | $p$-value (Uncorrected) | Holm-Adjusted $p$ | Significant After Holm ($\alpha=0.05$)? |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **MissForest (ExtraTrees)** | XGBoost | 0.7705 $\pm$ 0.0486 | **0.8431** | 0.6474 | +0.0112 | [-0.0051, +0.0275] | $t = 1.376$ | $p = 0.1750$ | $p = 0.8751$ | No (CI includes 0) |
| **MICE (BayesianRidge)** | XGBoost | **0.7726** $\pm$ 0.0549 | 0.8415 | **0.6560** | +0.0073 | [-0.0077, +0.0222] | $t = 0.978$ | $p = 0.3329$ | $p = 1.0000$ | No (CI includes 0) |
| **Honest RLTR ($\epsilon=0.25$)** | XGBoost | 0.7671 $\pm$ 0.0504 | 0.8407 | 0.6446 | +0.0039 | [-0.0116, +0.0193] | $t = 0.503$ | $p = 0.6172$ | $p = 1.0000$ | No (CI includes 0) |
| **KNN ($k=5$)** | Logistic Regression | 0.7692 $\pm$ 0.0454 | 0.8372 | 0.6258 | +0.0029 | [-0.0151, +0.0208] | $t = 0.320$ | $p = 0.7504$ | $p = 1.0000$ | No (CI includes 0) |
| **Mean (SimpleImputer)** | Logistic Regression | 0.7679 $\pm$ 0.0441 | 0.8363 | 0.6218 | +0.0015 | [-0.0094, +0.0125] | $t = 0.284$ | $p = 0.7775$ | $p = 1.0000$ | No (CI includes 0) |
| **Median (SimpleImputer)** | Logistic Regression | 0.7676 $\pm$ 0.0425 | 0.8361 | 0.6208 | Baseline (0.0) | Reference | Reference | Reference | Reference | Reference |

*Visual Artifact: See [`v2/figures_v2/A1_imputer_x_model.png`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/5.1%20EXTRA%20BANTU%20BU%20YUNEN/DiabeticAnalysis/v2/figures_v2/A1_imputer_x_model.png).*

> **Key Finding**: While MissForest (+1.12 pp) and MICE (+0.73 pp) achieve higher point estimates, the Nadeau-Bengio corrected resampled t-test shows that **no complex imputation method statistically significantly outperforms median imputation** ($p > 0.05$ for all comparisons; all 95% CIs encompass zero). This demonstrates that simple median imputation is an exceptionally robust, defensible baseline for PIMA.

---

## Phase 4: Squeeze the Modeling (Nested CV Protocol)

### Table 4A: Feature Ablation Study (Paired Deltas with 95% CI vs Base 8 Features)

Evaluated under 10-Fold Stratified CV using CatBoost on leak-free data. Fold-by-fold paired differences ($d_i = \text{score}_{\text{cand}, i} - \text{score}_{\text{base}, i}$) were calculated. Features are classified as helpful or harmful **only if the 95% confidence interval strictly excludes 0**; otherwise, they are designated as neutral/inconclusive.

| Feature Added to Base 8 Predictors | 10-Fold CV Accuracy (Mean $\pm$ Std) | Delta vs Base 8 Features | Paired SE | 95% Confidence Interval | $t$-statistic | $p$-value | Classification (Excludes 0?) | Retained in Final Pipeline? |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **None (Base 8 Clinical Features)** | **0.7734 $\pm$ 0.0427** | **Baseline (0.0000)** | — | — | — | — | **Reference** | **YES (Champion)** |
| + Missing Indicator Flags | 0.7708 $\pm$ 0.0327 | $-0.0026$ | 0.0054 | `[-0.0148, +0.0097]` | $t = -0.477$ | $p = 0.6449$ | Neutral (CI spans 0) | NO |
| + Glucose $\times$ BMI | 0.7707 $\pm$ 0.0406 | $-0.0026$ | 0.0064 | `[-0.0171, +0.0118]` | $t = -0.411$ | $p = 0.6905$ | Neutral (CI spans 0) | NO |
| + Glucose $\times$ Age | 0.7707 $\pm$ 0.0434 | $-0.0026$ | 0.0075 | `[-0.0195, +0.0142]` | $t = -0.355$ | $p = 0.7308$ | Neutral (CI spans 0) | NO |
| + Insulin $\times$ BMI | 0.7721 $\pm$ 0.0393 | $-0.0013$ | 0.0041 | `[-0.0105, +0.0080]` | $t = -0.313$ | $p = 0.7612$ | Neutral (CI spans 0) | NO |
| + $\log(\text{Insulin})$ | 0.7721 $\pm$ 0.0407 | $-0.0013$ | 0.0023 | `[-0.0066, +0.0040]` | $t = -0.557$ | $p = 0.5911$ | Neutral (CI spans 0) | NO |
| + BMI Bins (Categorical) | 0.7694 $\pm$ 0.0470 | $-0.0039$ | 0.0067 | `[-0.0192, +0.0113]` | $t = -0.584$ | $p = 0.5737$ | Neutral (CI spans 0) | NO |
| + Age Bins (Categorical) | 0.7721 $\pm$ 0.0386 | $-0.0013$ | 0.0060 | `[-0.0148, +0.0122]` | $t = -0.218$ | $p = 0.8324$ | Neutral (CI spans 0) | NO |
| + RiskScore (ADA Criteria) | 0.7695 $\pm$ 0.0367 | $-0.0039$ | 0.0073 | `[-0.0204, +0.0125]` | $t = -0.538$ | $p = 0.6036$ | Neutral (CI spans 0) | NO |
| + HOMA-IR | 0.7668 $\pm$ 0.0422 | $-0.0065$ | 0.0045 | `[-0.0166, +0.0035]` | $t = -1.469$ | $p = 0.1759$ | Neutral (CI spans 0) | NO |

> **Key Finding**: In genuine leak-free machine learning, **all ablation 95% confidence intervals encompass zero**. None of the interaction terms provide statistically distinguishable improvements. The principle of parsimony strictly dictates retaining only the original 8 clinical features.

### Table 4B: Model Architecture Diversity Benchmark

| Model Architecture | 10-Fold CV Accuracy | ROC-AUC | F1-Score | Primary Hyperparameters |
|---|:---:|:---:|:---:|---|
| **CatBoost** | **0.7734 $\pm$ 0.0427** | **0.8448** | 0.6461 | depth=4, lr=0.03, iterations=150 |
| **XGBoost** | **0.7732 $\pm$ 0.0679** | **0.8438** | **0.6642** | max_depth=3, lr=0.03, n_est=150 |
| **Logistic Regression (ElasticNet)** | 0.7721 $\pm$ 0.0204 | 0.8372 | 0.6317 | L1_ratio=0.5, solver=saga |
| **Logistic Regression (L2)** | 0.7695 $\pm$ 0.0210 | 0.8374 | 0.6278 | C=1.0, max_iter=1000 |
| **HistGradientBoosting** | 0.7681 $\pm$ 0.0591 | 0.8399 | 0.6496 | max_iter=150, max_depth=3 |
| **Random Forest** | 0.7668 $\pm$ 0.0478 | 0.8422 | 0.6363 | n_est=200, max_depth=6 |
| **LightGBM** | 0.7668 $\pm$ 0.0603 | 0.8406 | 0.6454 | max_depth=3, lr=0.03, n_est=150 |
| **Gradient Boosting** | 0.7655 $\pm$ 0.0543 | 0.8409 | 0.6402 | max_depth=3, lr=0.03, n_est=150 |
| **Extra Trees** | 0.7591 $\pm$ 0.0302 | 0.8464 | 0.5710 | n_est=200, max_depth=6 |
| **SVM (RBF Kernel)** | 0.7565 $\pm$ 0.0410 | 0.8297 | 0.6075 | $C=1.0, \gamma=\text{scale}$ |
| **MLP Neural Network** | 0.7383 $\pm$ 0.0426 | 0.8052 | 0.5154 | hidden_layers=(64, 32), early_stopping |

### Table 4C: Imbalance Handling & Operating Threshold Optimization

| Optimization Layer | Configuration / Threshold | Accuracy | Sensitivity (Recall) | Specificity | F1-Score | Clinical Role |
|---|---|:---:|:---:|:---:|:---:|---|
| **Default Threshold** | Threshold = 0.50 | 0.7669 | 0.5897 | **0.8620** | 0.6339 | Minimizes false alarms |
| **Class Weighting** | `class_weight='balanced'` | 0.7642 | **0.7910** | 0.7490 | 0.6993 | Major recall boost (+20.1 pp) |
| **Calibrated Optimal F1** | **Threshold = 0.37** | **0.7682** | **0.7985** | **0.7520** | **0.7063** | **Optimal clinical screening point** |

*Visual Artifact: See [`v2/figures_v2/A2_threshold_tradeoff_curve.png`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/5.1%20EXTRA%20BANTU%20BU%20YUNEN/DiabeticAnalysis/v2/figures_v2/A2_threshold_tradeoff_curve.png).*

---

## Phase 5: Leak-Free Explainability (Rank-Only SHAP Analysis)

*Methodological Note on Comparing SHAP Values*: Comparing raw numerical SHAP magnitudes across disparate models and preprocessing pipelines is methodologically unsound because SHAP values are expressed in model log-odds relative to each model's specific base value $E[f(X)]$. Instead, **ordinal feature rankings** provide an invariant, rigorous comparison of clinical importance.

### Table 5: Feature Importance Comparison (Ordinal SHAP Rankings Only)

| Feature | Leak-Free Model Rank (TreeExplainer) | Leaked RLTR Model Rank (Base Features) | Leaked RLTR Model Rank (With FE) | Directional Rank Shift | Clinical Endocrinological Assessment |
|---|:---:|:---:|:---:|:---:|---|
| **Glucose** | **#1** | #2 | #4 | **Rose to #1** | **Restores physiological truth: Blood glucose is the definitive diagnostic biomarker of diabetes mellitus.** |
| **Insulin** | **#2** | **#1** | **#1** | **Dropped from #1** | Artificially inflated in RLTR due to target-conditional donor matching; drops once leakage is removed. |
| **BMI** | **#3** | #5 | #8 | **Rose from #5 to #3** | Body mass index / adiposity is a primary physiological risk factor in type 2 diabetes. |
| **Age** | **#4** | #3 | #6 | Rank #4 | Reflects progressive $\beta$-cell exhaustion and age-related insulin resistance. |
| **DiabetesPedigreeFunction** | **#5** | #6 | #7 | Rank #5 | Familial genetic predisposition. |
| **SkinThickness** | **#6** | #4 | (unranked) | **Dropped from #4 to #6** | Peripheral skinfold thickness is a secondary adiposity marker, not a primary driver. |
| **Pregnancies** | **#7** | #7 | (unranked) | Rank #7 | Gestational metabolic stress. |
| **BloodPressure** | **#8** | #8 | (unranked) | Rank #8 | Weakest univariate discriminator in the PIMA cohort. |

*Visual Artifacts: See [`v2/figures_v2/A3_leak_free_shap_ranking.png`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/5.1%20EXTRA%20BANTU%20BU%20YUNEN/DiabeticAnalysis/v2/figures_v2/A3_leak_free_shap_ranking.png) and [`v2/figures_v2/A4_partial_dependence_profiles.png`](file:///d:/1.%20COLLEGE/1.%20HERE%20WE%20GO/8.%20TUGAS%20AKHIR/5.1%20EXTRA%20BANTU%20BU%20YUNEN/DiabeticAnalysis/v2/figures_v2/A4_partial_dependence_profiles.png).*

---

## Phase 6: Final Evaluation on Untouched Holdout Test Set

The entire pipeline was **frozen** and evaluated on the untouched 154-patient holdout test set (`random_state=42`, stratified: 100 healthy, 54 diabetic) **exactly once**.

### Table 6: Final Benchmark Confrontation on Untouched Holdout

| Pipeline | Imputation & Preprocessing Protocol | Holdout Accuracy [95% CI] | Sensitivity [95% CI] | Specificity [95% CI] | Precision [95% CI] | F1-Score [95% CI] | ROC-AUC [95% CI] | PR-AUC [95% CI] | Brier Score [95% CI] | McNemar $p$-value vs Baseline |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Upperclassmen Baseline** | `RLTR_Imputed.csv` (Pre-split norm, Target Leakage) | 0.8506 [0.792 - 0.903] | 0.7593 [0.638 - 0.869] | 0.9000 [0.840 - 0.952] | 0.8039 [0.690 - 0.909] | 0.7810 [0.684 - 0.863] | 0.8931 [0.841 - 0.941] | 0.8124 [0.718 - 0.899] | 0.1182 [0.084 - 0.158] | Baseline |
| **Our Frozen Pipeline** | **Fold-Internal MICE + RobustScaler + 5-Model Soft Voting** | **0.7468** [0.675 - 0.818] | **0.5741** [0.436 - 0.717] | **0.8400** [0.763 - 0.909] | **0.6596** [0.514 - 0.792] | **0.6139** [0.494 - 0.718] | **0.8135** [0.736 - 0.878] | **0.6528** [0.525 - 0.797] | **0.1690** [0.133 - 0.208] | **$p = 0.0004$** ($b=18, c=2$) |

> **Academic Insight**: The McNemar exact test yields $p = 0.0004$, demonstrating that the performance disparity between the upperclassmen baseline on `RLTR_Imputed.csv` and our leak-free model is statistically significant and **consistent with the presence of target leakage and pre-split contamination**, rather than asserting that the difference is exclusively caused by any single factor.

---

## Methodological Clarification: Origin & Role of Holdout `random_state=12` (89.61%)

In preliminary presentations and exploratory scripts, an accuracy of **`89.61%` (138 / 154 correct)** was reported on `RLTR_Imputed.csv`. For academic transparency, its origin must be documented:

1. **Exploratory Seed Sweeps**: During preliminary model tuning, an exploratory scan across random split seeds ($0 \le \text{seed} \le 29$) was conducted to evaluate test variance. Split `random_state=12` produced an empirical maximum of 89.61% for the tuned soft-voting ensemble.
2. **Selection Bias (Optimism Bias)**: Reporting the maximum performance across multiple test splits introduces optimization-on-the-test-set. Across all 30 splits, the mean test accuracy was $\approx 83.5\%$, and on the standard seed 42 split, it was $83.77\%$.
3. **Compound Confounding**: Because `RLTR_Imputed.csv` contained class-conditional target leakage ($r = 0.7895$), the 89.61% figure was doubly confounded: split selection bias compounded by target leakage.
4. **Defensible Reporting Standard**: In peer-reviewed manuscripts, single-split peak metrics must not be presented as baseline benchmarks. The true, defensible performance of leak-free models on PIMA is **$76.5\% - 77.5\%$ cross-validated accuracy** ($0.843$ ROC-AUC).

---

## Claims We CAN Defend in Front of Examiners

1. **Replication of Baseline**: We replicated the upperclassmen paper baseline to the exact single patient ($\text{Acc} = 0.8506$, 90/10/13/41 matrix).
2. **Identification & Proof of Target Leakage**: We provided mathematical and empirical proof (partial correlation $r = 0.8208, p < 10^{-90}$; regression $t = 26.77, p < 10^{-75}$; class-conditional re-imputation reproducing $r = 0.7908$) that `RLTR_Imputed.csv` contains target leakage.
3. **Rectification of Un-Imputed Glucose Zeros**: We discovered and documented that all five imputed files left 5 patients with blood glucose equal to zero, distorting decision trees.
4. **Rigorous Imputation Benchmarking**: In a 50-fold leak-free repeated CV benchmark under Nadeau-Bengio corrected resampled t-tests, simple median imputation proves to be an exceptionally solid baseline with no complex imputer achieving a statistically significant advantage after Holm-Bonferroni correction.
5. **Endocrinological Restoration via SHAP**: In our leak-free pipeline, Glucose is the undisputed #1 predictive feature, and BMI is #3, correcting the physiological distortion where Insulin was artificially dominant.
6. **Defensible Benchmark Standard**: On real, uncorrupted PIMA data evaluated leak-free, the true performance frontier is **$76.5\% - 77.5\%$ cross-validated accuracy** ($0.843$ ROC-AUC).

---

## Claims We Must NOT Make (To Avoid Failure in Peer Review)

1. **DO NOT claim we "cleared the 0.88 benchmark" or "hit 0.8961" as a general finding**: 89.61% was obtained on `RLTR_Imputed.csv` (which contains target leakage) on a single favorable split (`random_state=12`). On the paper's split (`random_state=42`), the same model gets 83.77%.
2. **DO NOT claim that the upperclassmen paper achieved 0.88**: The upperclassmen paper achieved 0.8506. "0.88" does not appear anywhere in their published manuscript.
3. **DO NOT claim feature engineering (+FE) boosted accuracy to 0.90**: Rigorous 10-fold nested ablation shows that none of the engineered interaction features beat the 8 raw features under a leak-free protocol (all 95% CIs encompass 0).
4. **DO NOT claim RLTR is a "proven superior" imputation algorithm**: RLTR's apparent superiority was driven by target correlation. An honest re-implementation of RLTR achieves parity with median imputation (0.7671 vs 0.7676).

---

## Slide Deck Numeric Diff List (For Synchronizing Presentations)

| Slide # | Graphic File | Old Claim / Value in Preliminary Draft | Corrected Defensible Value | Ground-Truth Source File |
|:---:|---|---|---|---|
| **Slide 06** | `figures/01_accuracy_heatmap_all_models_datasets.png` | CatBoost 84.89%, LightGBM 84.50% | **CatBoost 85.02%, LightGBM 83.46%** | `figures/01_accuracy_heatmap_all_models_datasets.png` |
| **Slide 07** | `figures/02_roc_auc_heatmap_all_models_datasets.png` | "+0.07 to +0.10 ROC-AUC jump" | **"+0.05 to +0.07 ROC-AUC jump"** | `figures/02_roc_auc_heatmap_all_models_datasets.png` |
| **Slide 09** | `figures/10_shap_feature_importance_ranking.png` | "Insulin dominates because of metabolic resistance" | **"Insulin dominance (Rank #1) is an artifact of RLTR leakage; leak-free SHAP restores Glucose to Rank #1 and BMI to Rank #3"** | `v2/results/phase5_shap_importance.csv` |
| **Slide 10** | `figures/05_cv_fold_stability_top_models.png` | "Ensemble fold center is above 85%" | **"Boxplot shows 4 individual models (medians 0.83 to 0.85), peak fold 94.81% is from single boosted trees"** | `figures/05_cv_fold_stability_top_models.png` |
| **Slide 11** | `figures/12_consensus_model_confusion_matrix.png` | Consensus F1 = 0.8519; "0.88 benchmark cleared" | **Consensus F1 = 0.8491 (TN 93, FP 7, FN 9, TP 45); 10-fold CV ensemble is 0.8528; true leak-free holdout is 0.7468** | `v2/results/phase1_baseline_reproduction.csv` & `v2/results/phase6_final_evaluation.csv` |
| **Slide 12** | Roadmap / Pillars | "RLTR proven superior; FE pushed accuracy past 0.90" | **"PIMA leak-free ceiling is ~77% (0.844 AUC); paper's core contribution is uncovering RLTR leakage (t = 26.77) and establishing leak-free MissForest protocol"** | `v2/results/phase2_leakage_verdict.md` & `v2/results/imputer_benchmark.csv` |

---

## Exact Commands to Reproduce (Deterministic Seeds)

All scripts run from the repository root using the project virtual environment:

```powershell
# Phase 1: Reproduce paper baseline (0.8506), McNemar tests, and 2000x bootstrap CIs
.\venv\Scripts\python.exe v2/scripts/run_phase1.py

# Phase 2: Run mathematical leakage audit, partial correlation, and class-conditional test
.\venv\Scripts\python.exe v2/scripts/run_phase2.py

# Phase 3: Run 50-fold leak-free imputer benchmark & Nadeau-Bengio corrected resampled t-tests
.\venv\Scripts\python.exe v2/scripts/run_phase3.py

# Phase 4: Run nested feature ablation with 95% CIs, model diversity, and threshold scan
.\venv\Scripts\python.exe v2/scripts/run_phase4.py

# Phase 5: Run leak-free SHAP explainability (ordinal ranking) and PDP analysis
.\venv\Scripts\python.exe v2/scripts/run_phase5.py

# Phase 6: Run final frozen pipeline evaluation on untouched holdout once
.\venv\Scripts\python.exe v2/scripts/run_phase6.py
```
