import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import optuna

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Import stats helpers
sys.path.append(os.path.abspath("v3/scripts"))
from stats_utils import assert_disjoint, nadeau_bengio_ttest, tost_equivalence

print("="*75)
print("EXPERIMENT E4: HONEST NESTED CV HYPERPARAMETER OPTIMIZATION (50 OUTER FOLDS)")
print("="*75)

os.makedirs("v3/results", exist_ok=True)

# ------------------------------------------------------------------------------
# 1. LOAD DATA & IN-FOLD MEDIAN SETUP
# ------------------------------------------------------------------------------
raw = pd.read_csv("Dataset Diabetes.csv", sep=";")
y = raw["Outcome"].values
n_samples = len(raw)

# Replace non-physiological zeros with NaN
raw_X = raw.drop(columns=["Outcome"]).copy()
physiological_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in physiological_cols:
    raw_X.loc[raw_X[col] == 0, col] = np.nan

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
n_folds = 50

fold_results_file = "v3/results/E4_fold_level_results.csv"
stability_file = "v3/results/E4_hyperparameter_stability.csv"
summary_file = "v3/results/E4_honest_tuning_nested_cv.csv"

# Remove existing files to ensure clean append-only generation
if os.path.exists(fold_results_file):
    os.remove(fold_results_file)
if os.path.exists(stability_file):
    os.remove(stability_file)

print("Setup: 50 outer folds, inner 5-fold CV, 50 Optuna trials per model, in-fold median imputation.")
start_time = time.time()

fold_records = []
hyperparam_records = []

for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(raw_X, y)):
    t_fold_start = time.time()
    outer_fold_seed = 42 + fold_idx
    
    # Assert disjoint indices
    assert_disjoint(train_idx, test_idx)
    
    y_tr, y_te = y[train_idx], y[test_idx]
    
    X_tr_raw = raw_X.iloc[train_idx].copy().reset_index(drop=True)
    X_te_raw = raw_X.iloc[test_idx].copy().reset_index(drop=True)
    
    # In-fold Median Imputation
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = pd.DataFrame(imputer.fit_transform(X_tr_raw), columns=raw_X.columns)
    X_te_imp = pd.DataFrame(imputer.transform(X_te_raw), columns=raw_X.columns)
    
    # In-fold Scaling for Logistic Regression
    scaler = StandardScaler()
    X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr_imp), columns=raw_X.columns)
    X_te_scaled = pd.DataFrame(scaler.transform(X_te_imp), columns=raw_X.columns)
    
    # Inner CV splitter
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=outer_fold_seed)
    
    # ==========================================================================
    # MODEL 1: XGBOOST
    # ==========================================================================
    # Default model
    xgb_def = XGBClassifier(
        max_depth=6,
        n_estimators=100,
        learning_rate=0.1,
        random_state=outer_fold_seed,
        eval_metric="logloss",
        n_jobs=2
    )
    xgb_def.fit(X_tr_imp.values, y_tr)
    pred_def_x = xgb_def.predict(X_te_imp.values)
    prob_def_x = xgb_def.predict_proba(X_te_imp.values)[:, 1]
    
    # Inner Optuna tuning (50 trials)
    def obj_xgb(trial):
        clf = XGBClassifier(
            max_depth=trial.suggest_int("max_depth", 2, 8),
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=outer_fold_seed,
            eval_metric="logloss",
            n_jobs=2
        )
        scores = cross_val_score(clf, X_tr_imp.values, y_tr, cv=cv_inner, scoring="roc_auc", n_jobs=1)
        return float(np.mean(scores))

    study_xgb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=outer_fold_seed))
    study_xgb.optimize(obj_xgb, n_trials=50)
    best_xgb_params = study_xgb.best_params
    
    xgb_tuned = XGBClassifier(
        **best_xgb_params,
        random_state=outer_fold_seed,
        eval_metric="logloss",
        n_jobs=2
    )
    xgb_tuned.fit(X_tr_imp.values, y_tr)
    pred_tun_x = xgb_tuned.predict(X_te_imp.values)
    prob_tun_x = xgb_tuned.predict_proba(X_te_imp.values)[:, 1]
    
    hyperparam_records.append({
        "fold": fold_idx,
        "model": "XGBoost",
        "inner_cv_best_auc": study_xgb.best_value,
        **best_xgb_params
    })
    
    # ==========================================================================
    # MODEL 2: CATBOOST
    # ==========================================================================
    # Default model
    cb_def = CatBoostClassifier(
        depth=6,
        iterations=300,
        learning_rate=0.05,
        random_seed=outer_fold_seed,
        verbose=False,
        thread_count=2
    )
    cb_def.fit(X_tr_imp.values, y_tr)
    pred_def_c = cb_def.predict(X_te_imp.values)
    prob_def_c = cb_def.predict_proba(X_te_imp.values)[:, 1]
    
    # Inner Optuna tuning (50 trials)
    def obj_cb(trial):
        clf = CatBoostClassifier(
            depth=trial.suggest_int("depth", 3, 8),
            iterations=trial.suggest_int("iterations", 100, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            random_seed=outer_fold_seed,
            verbose=False,
            thread_count=2
        )
        scores = cross_val_score(clf, X_tr_imp.values, y_tr, cv=cv_inner, scoring="roc_auc", n_jobs=1)
        return float(np.mean(scores))

    study_cb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=outer_fold_seed))
    study_cb.optimize(obj_cb, n_trials=50)
    best_cb_params = study_cb.best_params
    
    cb_tuned = CatBoostClassifier(
        **best_cb_params,
        random_seed=outer_fold_seed,
        verbose=False,
        thread_count=2
    )
    cb_tuned.fit(X_tr_imp.values, y_tr)
    pred_tun_c = cb_tuned.predict(X_te_imp.values)
    prob_tun_c = cb_tuned.predict_proba(X_te_imp.values)[:, 1]
    
    hyperparam_records.append({
        "fold": fold_idx,
        "model": "CatBoost",
        "inner_cv_best_auc": study_cb.best_value,
        **best_cb_params
    })
    
    # ==========================================================================
    # MODEL 3: LOGISTIC REGRESSION (ELASTICNET)
    # ==========================================================================
    # Default model (L2, C=1.0)
    lr_def = LogisticRegression(
        penalty="l2",
        C=1.0,
        max_iter=1000,
        random_state=outer_fold_seed
    )
    lr_def.fit(X_tr_scaled.values, y_tr)
    pred_def_l = lr_def.predict(X_te_scaled.values)
    prob_def_l = lr_def.predict_proba(X_te_scaled.values)[:, 1]
    
    # Inner Optuna tuning (50 trials)
    def obj_lr(trial):
        clf = LogisticRegression(
            penalty="elasticnet",
            C=trial.suggest_float("C", 1e-3, 1e2, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
            solver="saga",
            max_iter=1000,
            random_state=outer_fold_seed
        )
        scores = cross_val_score(clf, X_tr_scaled.values, y_tr, cv=cv_inner, scoring="roc_auc", n_jobs=1)
        return float(np.mean(scores))

    study_lr = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=outer_fold_seed))
    study_lr.optimize(obj_lr, n_trials=50)
    best_lr_params = study_lr.best_params
    
    lr_tuned = LogisticRegression(
        penalty="elasticnet",
        **best_lr_params,
        solver="saga",
        max_iter=1000,
        random_state=outer_fold_seed
    )
    lr_tuned.fit(X_tr_scaled.values, y_tr)
    pred_tun_l = lr_tuned.predict(X_te_scaled.values)
    prob_tun_l = lr_tuned.predict_proba(X_te_scaled.values)[:, 1]
    
    hyperparam_records.append({
        "fold": fold_idx,
        "model": "LogReg-EN",
        "inner_cv_best_auc": study_lr.best_value,
        **best_lr_params
    })
    
    # Evaluate and record all metrics for fold
    eval_list = [
        ("XGBoost", "Default", pred_def_x, prob_def_x),
        ("XGBoost", "Tuned", pred_tun_x, prob_tun_x),
        ("CatBoost", "Default", pred_def_c, prob_def_c),
        ("CatBoost", "Tuned", pred_tun_c, prob_tun_c),
        ("LogReg-EN", "Default", pred_def_l, prob_def_l),
        ("LogReg-EN", "Tuned", pred_tun_l, prob_tun_l),
    ]
    
    curr_fold_rows = []
    for m_name, m_type, y_pred, y_prob in eval_list:
        acc = accuracy_score(y_te, y_pred)
        auc = roc_auc_score(y_te, y_prob)
        prauc = average_precision_score(y_te, y_prob)
        brier = brier_score_loss(y_te, y_prob)
        cm = confusion_matrix(y_te, y_pred).ravel()
        tn, fp, fn, tp = cm
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_te, y_pred, zero_division=0)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_te, y_pred, average="weighted", zero_division=0)
        
        row_dict = {
            "fold": fold_idx,
            "model": m_name,
            "type": m_type,
            "accuracy": float(acc),
            "roc_auc": float(auc),
            "pr_auc": float(prauc),
            "brier": float(brier),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "f1": float(f1),
            "weighted_precision": float(p_w),
            "weighted_recall": float(r_w),
            "weighted_f1": float(f1_w),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }
        curr_fold_rows.append(row_dict)
        fold_records.append(row_dict)
        
    # INCREMENTAL APPEND: Write fold results to disk
    df_curr_f = pd.DataFrame(curr_fold_rows)
    df_curr_f.to_csv(fold_results_file, mode="a", header=not os.path.exists(fold_results_file), index=False)
    
    # INCREMENTAL APPEND: Write hyperparameter results to disk
    df_curr_hp = pd.DataFrame(hyperparam_records[-3:])
    df_curr_hp.to_csv(stability_file, mode="a", header=not os.path.exists(stability_file), index=False)
    
    if (fold_idx + 1) % 5 == 0 or fold_idx == 0:
        elapsed = time.time() - start_time
        print(f"  Completed {fold_idx + 1}/50 folds ({elapsed:.1f}s elapsed, last fold: {time.time() - t_fold_start:.2f}s)", flush=True)

print(f"\nAll 50 folds completed successfully in {time.time() - start_time:.1f}s.", flush=True)

# ------------------------------------------------------------------------------
# 2. STATISTICAL COMPARISON: TUNED VS DEFAULT PER MODEL
# ------------------------------------------------------------------------------
df_folds = pd.DataFrame(fold_records)
k = 50
n_test = 154
n_train = 614

summary_records = []
models = ["XGBoost", "CatBoost", "LogReg-EN"]

for m in models:
    df_m = df_folds[df_folds["model"] == m]
    
    def_sub = df_m[df_m["type"] == "Default"].sort_values("fold")
    tun_sub = df_m[df_m["type"] == "Tuned"].sort_values("fold")
    
    # Fold values
    auc_def = def_sub["roc_auc"].values
    auc_tun = tun_sub["roc_auc"].values
    acc_def = def_sub["accuracy"].values
    acc_tun = tun_sub["accuracy"].values
    
    # Paired differences
    d_auc = auc_tun - auc_def
    d_acc = acc_tun - acc_def
    
    tt_auc = nadeau_bengio_ttest(d_auc, k, n_test, n_train)
    tost_auc = tost_equivalence(d_auc, k, n_test, n_train, margin=0.02)
    
    tt_acc = nadeau_bengio_ttest(d_acc, k, n_test, n_train)
    tost_acc = tost_equivalence(d_acc, k, n_test, n_train, margin=0.02)
    
    summary_records.append({
        "model": m,
        "auc_default_mean": float(np.mean(auc_def)),
        "auc_default_std": float(np.std(auc_def, ddof=1)),
        "auc_tuned_mean": float(np.mean(auc_tun)),
        "auc_tuned_std": float(np.std(auc_tun, ddof=1)),
        "delta_auc_tuned_vs_default": float(tt_auc["mean_diff"]),
        "auc_nb_se_corr": float(tt_auc["se_corr"]),
        "auc_nb_ci_lower": float(tt_auc["ci_lower"]),
        "auc_nb_ci_upper": float(tt_auc["ci_upper"]),
        "auc_nb_t_stat": float(tt_auc["t_stat"]),
        "auc_nb_p_value": float(tt_auc["p_value"]),
        "auc_tost_p": float(tost_auc["p_tost"]),
        "auc_equivalent": bool(tost_auc["equivalent"]),
        "acc_default_mean": float(np.mean(acc_def)),
        "acc_default_std": float(np.std(acc_def, ddof=1)),
        "acc_tuned_mean": float(np.mean(acc_tun)),
        "acc_tuned_std": float(np.std(acc_tun, ddof=1)),
        "delta_acc_tuned_vs_default": float(tt_acc["mean_diff"]),
        "acc_nb_se_corr": float(tt_acc["se_corr"]),
        "acc_nb_ci_lower": float(tt_acc["ci_lower"]),
        "acc_nb_ci_upper": float(tt_acc["ci_upper"]),
        "acc_nb_t_stat": float(tt_acc["t_stat"]),
        "acc_nb_p_value": float(tt_acc["p_value"]),
        "acc_tost_p": float(tost_acc["p_tost"]),
        "acc_equivalent": bool(tost_acc["equivalent"]),
        "prauc_tuned_mean": float(tun_sub["pr_auc"].mean()),
        "brier_tuned_mean": float(tun_sub["brier"].mean()),
        "sensitivity_tuned_mean": float(tun_sub["sensitivity"].mean()),
        "specificity_tuned_mean": float(tun_sub["specificity"].mean()),
        "f1_tuned_mean": float(tun_sub["f1"].mean()),
        "weighted_f1_tuned_mean": float(tun_sub["weighted_f1"].mean())
    })

df_summary = pd.DataFrame(summary_records)
df_summary.to_csv(summary_file, index=False)
print(f"Saved E4 tuning summary to: {summary_file}")

print("\n" + "="*75)
print("EXPERIMENT E4 COMPLETED SUCCESSFULLY")
print("="*75)

