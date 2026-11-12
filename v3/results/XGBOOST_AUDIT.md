# Audit of XGBoost Hyperparameter Configurations ("Default" vs. "Ladder Configuration")

- **Document**: `v3/results/XGBOOST_AUDIT.md`
- **Scope**: Audit of all XGBoost model instantiations across Phase 1, Phase 2, and Retraining Round v3 experiments
- **Date**: 2026-11-12
- **XGBoost Package Version**: `3.4.1`

---

## 1. Inventory of XGBoost Parameters in Experiments Labeled "Default"

Across every experiment previously referring to "default XGBoost", the model was explicitly instantiated with a fixed hyperparameter tuple:

| Experiment / File | Code Location | Exact Parameters Instantiated | Deviation from True Library Defaults |
|---|---|---|---|
| **Phase 1** | `v2/scripts/run_phase1.py` Line 42 | `XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss")` | `learning_rate=0.1` (True library default is `0.3`) |
| **E2 (Rungs R0–R3)** | `v3/scripts/run_e2_inflation_ladder.py` Lines 78, 114, 149, 189 | `XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=4)` | `learning_rate=0.1` (True library default is `0.3`) |
| **E2b (Paper Replication)** | `v3/scripts/run_e2b_paper_replication.py` Line 55 | `XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=4)` | `learning_rate=0.1` (True library default is `0.3`) |
| **E3 (Honest Imputation)** | `v3/scripts/run_e3_honest_imputation_benchmark.py` Line 97 | `XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=fold_seed, eval_metric="logloss", n_jobs=2)` | `learning_rate=0.1` (True library default is `0.3`) |
| **E4 (Default Baseline Rows)** | `v3/scripts/run_e4_honest_tuning_nested_cv.py` Line 96 | `XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=outer_fold_seed, eval_metric="logloss", n_jobs=2)` | `learning_rate=0.1` (True library default is `0.3`) |
| **E6 (R3a RLTR Rows)** | `v3/scripts/run_e6_multiseed_subgroups_ablation.py` Line 87 | `XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=4)` | `learning_rate=0.1` (True library default is `0.3`) |

---

## 2. Benchmark on Seed 42 RLTR: True Library Defaults vs. Ladder Configuration

Evaluated on `RLTR_Imputed.csv` (as published, pre-split MinMaxScaler) with `train_test_split(..., test_size=0.20, random_state=42, stratify=y)`:

| Configuration | Parameters | Accuracy | ROC-AUC | Confusion Matrix (TN, FP, FN, TP) | Reproduces 0.8506 (90,10,13,41)? |
|---|---|:---:|:---:|:---:|:---:|
| **True Library Defaults** (`xgboost 3.4.1`) | `learning_rate=None` ($\to 0.3$), `max_depth=None` ($\to 6$), `n_estimators=None` ($\to 100$) | **$83.7662\%$** | **$0.9081$** | **$(89, 11, 14, 40)$** | **NO** (Differs by 2 classification decisions) |
| **Ladder Configuration** | `learning_rate=0.1`, `max_depth=6`, `n_estimators=100` | **$85.0649\%$** | **$0.9057$** | **$(90, 10, 13, 41)$** | **YES (Exact Match)** |

### Finding
The published Paper 1 / Paper 2 baseline of **$85.0649\%$** ($90\text{ TN}, 10\text{ FP}, 13\text{ FN}, 41\text{ TP}$) **does NOT reproduce with true library defaults**. It reproduces **ONLY when `learning_rate=0.1` is explicitly specified**. 

---

## 3. Formal Relabeling Specification

To maintain complete scientific rigor and avoid misleading readers:
- Any configuration using `max_depth=6, n_estimators=100, learning_rate=0.1` is formally designated as the **"ladder configuration (lr=0.1, depth=6, n_est=100)"**.
- The term "default XGBoost" is restricted exclusively to unparameterized calls matching the native library specification (`learning_rate=0.3`).
