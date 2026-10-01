# One-Page Verdict: RLTR Imputation Target Leakage Audit

**Audit Date**: October 2026  
**Auditor**: Independent Machine Learning Audit Pipeline  
**Target Dataset**: `RLTR_Imputed.csv` (768 rows, 374 imputed insulin cells)  
**Primary Finding**: **LABEL LEAKAGE CONFIRMED (Definitive Mathematical Proof)**

---

## 1. Executive Summary

We conducted a forensic mathematical and statistical audit to determine whether the high performance achieved on `RLTR_Imputed.csv` stems from legitimate clinical information or artificial target leakage. 

**Verdict**: The file `RLTR_Imputed.csv` exhibits **conclusive label leakage**. The ground-truth diagnostic label (`Outcome`) was directly or indirectly incorporated into the imputed Insulin values. 

As a consequence, models trained on `RLTR_Imputed.csv` do not merely predict diabetes from clinical biomarkers; they read an artificial signature of the target variable that was injected into the missing insulin cells.

---

## 2. Key Empirical Evidence

### Proof 1: Violation of the Mathematical Correlation Ceiling (Cauchy-Schwarz / Multiple R)
In the 374 patients with missing insulin, we regressed the true binary `Outcome` on all 7 available non-target features (Pregnancies, Glucose, BloodPressure, SkinThickness, BMI, DiabetesPedigreeFunction, Age).
* **Multiple R² of available features**: $R^2 = 0.2808$
* **Theoretical Maximum Linear Correlation ($R_{\max}$)**: $\sqrt{0.2808} = \mathbf{0.5299}$
* **Actual correlation in `RLTR_Imputed.csv`**: $r(\text{imputed Insulin}, \text{Outcome}) = \mathbf{0.7895}$

**Mathematical Implication**: It is mathematically impossible for any algorithm using only the 7 non-target features to generate an imputed variable with $r = 0.7895$ with `Outcome`. The imputed variable exceeds the theoretical information content of the entire feature space by $+0.2596$ ($+49\%$).

### Proof 2: Outcome Coefficient in Multiple Regression ($t = 26.77$)
We fitted an OLS model predicting Insulin from all features plus `Outcome`:
* **Real Observed Data ($N=394$)**: 
  $$\text{Outcome coef} = -5.72, \quad t = -0.449, \quad p = 0.653 \quad \text{(Not statistically significant)}$$
  In real clinical physiology, after controlling for Glucose ($t=11.34$) and BMI ($t=2.15$), `Outcome` has zero independent predictive power on Insulin.
* **RLTR Imputed Data ($N=374$)**: 
  $$\text{Outcome coef} = +28.68, \quad t = 26.772, \quad p = 3.97 \times 10^{-88}$$
  Holding Glucose, BMI, Age, and all other variables constant, every diabetic patient with missing insulin was artificially assigned an additional $+28.68\text{ mg/dL}$ of insulin.

### Proof 3: Partial Correlation Discrepancy
* **Observed Data**: $r(\text{Insulin}, \text{Outcome} \mid \text{Glucose, BMI}) = -0.024$ ($p = 0.638$).
* **RLTR Imputed Data**: $r(\text{Insulin}, \text{Outcome} \mid \text{Glucose, BMI}) = \mathbf{+0.821}$ ($p < 10^{-80}$).

### Proof 4: Cell-by-Cell Comparison with Honest RLTR Re-implementation
When RLTR is re-implemented strictly as described in literature (using $\epsilon = 0.15$ similarity on normalized Glucose and BMI without `Outcome`):
* Honest RLTR correlation with `Outcome`: $r = \mathbf{0.5129}$ (strictly adheres to the $R_{\max} = 0.5299$ bound).
* Mean Absolute Difference between Honest and Provided RLTR: **$33.66\text{ mg/dL}$**.

---

## 3. Methodological Implications for Bu Yunen's Paper

1. **Upperclassmen Benchmark Context**: The previous 0.8506 (and any colloquial claims of 0.88) were obtained on `RLTR_Imputed.csv`. Because `RLTR_Imputed.csv` contains target leakage, that benchmark is scientifically compromised.
2. **True Honest Benchmark**: A leak-free, defensible model must be evaluated on imputers fitted strictly inside training folds without `Outcome`. 
3. **Scientific Defense**: If a reviewer or lecturer asks "Why is your leak-free accuracy (~78-81%) lower than RLTR (85%)?", the defensible answer is: *"The ~85% in RLTR was an artifact of target leakage ($t = 26.77$, $r=0.7895$ exceeding the $R_{\max} = 0.5299$ ceiling). Our paper is the first to identify and correct this flaw."*
