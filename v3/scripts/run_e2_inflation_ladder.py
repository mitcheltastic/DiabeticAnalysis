import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import optuna

# Import project stats utils
sys.path.append(os.path.abspath("v3/scripts"))
from stats_utils import assert_disjoint

# Suppress Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("="*75, flush=True)
print("EXPERIMENT E2: THE PERFORMANCE INFLATION LADDER (100 SEEDS with R3a/R3b SPLIT)", flush=True)
print("="*75, flush=True)

os.makedirs("v3/results", exist_ok=True)
os.makedirs("v3/figures", exist_ok=True)

# ------------------------------------------------------------------------------
# 1. LOAD DATASETS & PRE-COMPUTATIONS
# ------------------------------------------------------------------------------
raw = pd.read_csv("Dataset Diabetes.csv", sep=";")
y = raw["Outcome"].values
n_samples = len(raw)

imputed_files = {
    "LTR": "LTR_Imputed.csv",
    "NSSR": "NSSR_Imputed.csv",
    "RLTR": "RLTR_Imputed.csv",
    "SIM": "SIM_Imputed.csv",
    "TR": "TR_Imputed.csv"
}

dfs_imputed = {k: pd.read_csv(v) for k, v in imputed_files.items()}

# Raw data with zeros replaced by NaN for physiological features
missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
raw_nan = raw.copy()
for col in missing_cols:
    raw_nan[col] = raw_nan[col].replace(0, np.nan)
X_raw = raw_nan.drop(columns=["Outcome"]).copy()

# Precompute global transformations for leakage rungs R1 and R2
scaler_global_std = StandardScaler()
X_global_scaled = pd.DataFrame(
    scaler_global_std.fit_transform(X_raw),
    columns=X_raw.columns
)

mice_global = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
X_raw_mice = pd.DataFrame(
    mice_global.fit_transform(X_raw),
    columns=X_raw.columns
)
scaler_global_mice = StandardScaler()
X_global_mice_scaled = pd.DataFrame(
    scaler_global_mice.fit_transform(X_raw_mice),
    columns=X_raw.columns
)

# Pre-split MinMaxScaler for R3a variants (published files as-is, glucose zeros intact)
dfs_r3a_scaled = {}
for k, df in dfs_imputed.items():
    X_file = df.drop(columns=["Outcome"]).copy()
    mms = MinMaxScaler()
    dfs_r3a_scaled[k] = pd.DataFrame(mms.fit_transform(X_file), columns=X_file.columns)

# ------------------------------------------------------------------------------
# 2. RUN 100 SEEDS FOR LADDER RUNGS
# ------------------------------------------------------------------------------
n_seeds = 100
seeds = list(range(n_seeds))

results_by_seed = []
seed42_details = {}

print(f"\nExecuting 100-seed evaluation across ladder rungs (Seeds 0 to {n_seeds - 1})...", flush=True)
start_time = time.time()

for s_idx, s in enumerate(seeds):
    t_seed_start = time.time()
    
    # 80:20 stratified split
    train_idx, test_idx = train_test_split(
        np.arange(n_samples),
        test_size=0.20,
        random_state=s,
        stratify=y
    )
    assert_disjoint(train_idx, test_idx)
    y_tr, y_te = y[train_idx], y[test_idx]
    
    seed_row = {"seed": s}
    
    # --------------------------------------------------------------------------
    # Rung R0: Leak-Free Baseline (In-fold Median + In-fold StandardScaler)
    # --------------------------------------------------------------------------
    X_tr_r0 = X_raw.iloc[train_idx].copy()
    X_te_r0 = X_raw.iloc[test_idx].copy()
    
    imp_r0 = SimpleImputer(strategy="median")
    X_tr_r0_imp = imp_r0.fit_transform(X_tr_r0)
    X_te_r0_imp = imp_r0.transform(X_te_r0)
    
    scaler_r0 = StandardScaler()
    X_tr_r0_s = scaler_r0.fit_transform(X_tr_r0_imp)
    X_te_r0_s = scaler_r0.transform(X_te_r0_imp)
    
    xgb_r0 = XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=4)
    xgb_r0.fit(X_tr_r0_s, y_tr)
    pred_r0 = xgb_r0.predict(X_te_r0_s)
    prob_r0 = xgb_r0.predict_proba(X_te_r0_s)[:, 1]
    
    seed_row["R0_acc"] = float(accuracy_score(y_te, pred_r0))
    seed_row["R0_auc"] = float(roc_auc_score(y_te, prob_r0))
    
    # --------------------------------------------------------------------------
    # Rung R1: Global Scaling Leakage (Pre-split Scaling + In-fold Median)
    # --------------------------------------------------------------------------
    X_tr_r1 = X_global_scaled.iloc[train_idx].copy()
    X_te_r1 = X_global_scaled.iloc[test_idx].copy()
    
    imp_r1 = SimpleImputer(strategy="median")
    X_tr_r1_imp = imp_r1.fit_transform(X_tr_r1)
    X_te_r1_imp = imp_r1.transform(X_te_r1)
    
    xgb_r1 = XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=4)
    xgb_r1.fit(X_tr_r1_imp, y_tr)
    pred_r1 = xgb_r1.predict(X_te_r1_imp)
    prob_r1 = xgb_r1.predict_proba(X_te_r1_imp)[:, 1]
    
    seed_row["R1_acc"] = float(accuracy_score(y_te, pred_r1))
    seed_row["R1_auc"] = float(roc_auc_score(y_te, prob_r1))
    
    # --------------------------------------------------------------------------
    # Rung R2: Global Imputation Leakage (Pre-split MICE + Pre-split Scaling)
    # --------------------------------------------------------------------------
    X_tr_r2 = X_global_mice_scaled.iloc[train_idx].values
    X_te_r2 = X_global_mice_scaled.iloc[test_idx].values
    
    xgb_r2 = XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=4)
    xgb_r2.fit(X_tr_r2, y_tr)
    pred_r2 = xgb_r2.predict(X_te_r2)
    prob_r2 = xgb_r2.predict_proba(X_te_r2)[:, 1]
    
    seed_row["R2_acc"] = float(accuracy_score(y_te, pred_r2))
    seed_row["R2_auc"] = float(roc_auc_score(y_te, prob_r2))
    
    # --------------------------------------------------------------------------
    # Rung R3a: Published Imputed Files AS-IS (Glucose Zeros Intact) + Pre-split MinMax
    # --------------------------------------------------------------------------
    for file_key, df_scaled in dfs_r3a_scaled.items():
        X_tr_3a = df_scaled.iloc[train_idx].values
        X_te_3a = df_scaled.iloc[test_idx].values
        
        xgb_3a = XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=4)
        xgb_3a.fit(X_tr_3a, y_tr)
        pred_3a = xgb_3a.predict(X_te_3a)
        prob_3a = xgb_3a.predict_proba(X_te_3a)[:, 1]
        
        acc_3a = float(accuracy_score(y_te, pred_3a))
        auc_3a = float(roc_auc_score(y_te, prob_3a))
        
        seed_row[f"R3a_{file_key}_acc"] = acc_3a
        seed_row[f"R3a_{file_key}_auc"] = auc_3a
        
        if file_key == "RLTR":
            seed_row["R3a_acc"] = acc_3a
            seed_row["R3a_auc"] = auc_3a
            X_tr_rltr_3a = X_tr_3a.copy()
            X_te_rltr_3a = X_te_3a.copy()
            
            # --- MANDATORY SANITY ANCHOR CHECK ON SEED 42 FOR R3a ---
            if s == 42:
                cm_42_3a = confusion_matrix(y_te, pred_3a)
                tn, fp, fn, tp = cm_42_3a.ravel()
                print(f"\n>>> Checking Mandatory Sanity Anchor E2 on Seed 42 for RLTR (R3a - Published Zeros Intact):", flush=True)
                print(f"    Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}", flush=True)
                print(f"    Accuracy: {acc_3a:.6f} (Expected: 0.850649)", flush=True)
                
                # Hard stop assertions
                assert abs(acc_3a - 0.850649) < 1e-4, f"HARD STOP: Seed 42 R3a accuracy ({acc_3a}) does not match 0.850649!"
                assert (tn, fp, fn, tp) == (90, 10, 13, 41), f"HARD STOP: Seed 42 R3a confusion matrix ({tn},{fp},{fn},{tp}) does not match (90, 10, 13, 41)!"
                print("    [SANITY ANCHOR CONFIRMED]: R3a reproduced exact upperclassmen baseline (0.8506, TN 90, FP 10, FN 13, TP 41).\n", flush=True)
                
        if s == 42:
            seed42_details[f"{file_key}_R3a"] = {
                "file": file_key,
                "variant": "R3a (Published Zeros Intact)",
                "accuracy": acc_3a,
                "roc_auc": auc_3a,
                "cm": confusion_matrix(y_te, pred_3a).tolist()
            }
            
    # --------------------------------------------------------------------------
    # Rung R3b: Imputed Files with In-fold Glucose Fix (Zeros set to NaN & Imputed in-fold)
    # --------------------------------------------------------------------------
    for file_key, df_f in dfs_imputed.items():
        X_f = df_f.drop(columns=["Outcome"]).copy()
        X_tr_3b = X_f.iloc[train_idx].copy()
        X_te_3b = X_f.iloc[test_idx].copy()
        
        # In-fold glucose fix
        X_tr_3b.loc[X_tr_3b["Glucose"] == 0, "Glucose"] = np.nan
        X_te_3b.loc[X_te_3b["Glucose"] == 0, "Glucose"] = np.nan
        med_g_3b = X_tr_3b["Glucose"].median()
        X_tr_3b["Glucose"] = X_tr_3b["Glucose"].fillna(med_g_3b)
        X_te_3b["Glucose"] = X_te_3b["Glucose"].fillna(med_g_3b)
        
        mms_3b = MinMaxScaler()
        X_tr_3b_s = mms_3b.fit_transform(X_tr_3b)
        X_te_3b_s = mms_3b.transform(X_te_3b)
        
        xgb_3b = XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=4)
        xgb_3b.fit(X_tr_3b_s, y_tr)
        pred_3b = xgb_3b.predict(X_te_3b_s)
        prob_3b = xgb_3b.predict_proba(X_te_3b_s)[:, 1]
        
        acc_3b = float(accuracy_score(y_te, pred_3b))
        auc_3b = float(roc_auc_score(y_te, prob_3b))
        
        seed_row[f"R3b_{file_key}_acc"] = acc_3b
        seed_row[f"R3b_{file_key}_auc"] = auc_3b
        
        if file_key == "RLTR":
            seed_row["R3b_acc"] = acc_3b
            seed_row["R3b_auc"] = auc_3b
            
        if s == 42:
            seed42_details[f"{file_key}_R3b"] = {
                "file": file_key,
                "variant": "R3b (In-fold Glucose Fix)",
                "accuracy": acc_3b,
                "roc_auc": auc_3b,
                "cm": confusion_matrix(y_te, pred_3b).tolist()
            }
            
    # --------------------------------------------------------------------------
    # Rung R4: Direct Test Set Optimization on RLTR (100 Optuna Trials)
    # Evaluated on RLTR R3a (published features) to match Paper 2 optimization
    # --------------------------------------------------------------------------
    def obj_test_r4(trial):
        clf = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=42,
            eval_metric="logloss",
            n_jobs=4
        )
        clf.fit(X_tr_rltr_3a, y_tr)
        preds = clf.predict(X_te_rltr_3a)
        return float(accuracy_score(y_te, preds))

    study_r4 = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_r4.optimize(obj_test_r4, n_trials=100)
    
    best_acc_r4 = float(study_r4.best_value)
    best_params_r4 = study_r4.best_params
    best_clf_r4 = XGBClassifier(**best_params_r4, random_state=42, eval_metric="logloss", n_jobs=4)
    best_clf_r4.fit(X_tr_rltr_3a, y_tr)
    prob_r4 = best_clf_r4.predict_proba(X_te_rltr_3a)[:, 1]
    best_auc_r4 = float(roc_auc_score(y_te, prob_r4))
    
    seed_row["R4_acc"] = best_acc_r4
    seed_row["R4_auc"] = best_auc_r4
    
    if s == 42:
        seed42_details["R4_Tuned_Test"] = {
            "file": "RLTR",
            "variant": "R4 (Direct Test Set Tuned, 100 trials)",
            "accuracy": best_acc_r4,
            "roc_auc": best_auc_r4,
            "cm": confusion_matrix(y_te, best_clf_r4.predict(X_te_rltr_3a)).tolist()
        }

    # --------------------------------------------------------------------------
    # Rung R4': Honest CV Tuning on Training Data Only (50 Trials, 5-Fold CV)
    # --------------------------------------------------------------------------
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    def obj_cv_r4_prime(trial):
        clf = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=42,
            eval_metric="logloss",
            n_jobs=4
        )
        scores = cross_val_score(clf, X_tr_rltr_3a, y_tr, cv=cv_inner, scoring="accuracy", n_jobs=1)
        return float(np.mean(scores))

    study_r4_prime = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_r4_prime.optimize(obj_cv_r4_prime, n_trials=50)
    
    best_params_r4_prime = study_r4_prime.best_params
    best_clf_r4_prime = XGBClassifier(**best_params_r4_prime, random_state=42, eval_metric="logloss", n_jobs=4)
    best_clf_r4_prime.fit(X_tr_rltr_3a, y_tr)
    pred_r4_prime = best_clf_r4_prime.predict(X_te_rltr_3a)
    prob_r4_prime = best_clf_r4_prime.predict_proba(X_te_rltr_3a)[:, 1]
    
    seed_row["R4_prime_acc"] = float(accuracy_score(y_te, pred_r4_prime))
    seed_row["R4_prime_auc"] = float(roc_auc_score(y_te, prob_r4_prime))
    
    if s == 42:
        seed42_details["R4_Prime_Honest_Tuned"] = {
            "file": "RLTR",
            "variant": "R4' (Honest Train CV Tuned, 50 trials)",
            "accuracy": seed_row["R4_prime_acc"],
            "roc_auc": seed_row["R4_prime_auc"],
            "cm": confusion_matrix(y_te, pred_r4_prime).tolist()
        }

    results_by_seed.append(seed_row)
    
    if (s + 1) % 10 == 0 or s == 42 or s == 0:
        elapsed = time.time() - start_time
        print(f"  Processed {s + 1}/{n_seeds} seeds ({elapsed:.1f}s elapsed, last seed: {time.time() - t_seed_start:.1f}s)", flush=True)

# ------------------------------------------------------------------------------
# 3. COMPILE AGGREGATE SUMMARY TABLE & WATERFALL
# ------------------------------------------------------------------------------
df_seeds = pd.DataFrame(results_by_seed)
df_seeds.to_csv("v3/results/E2_seed_level_results.csv", index=False)
print("\nSaved seed-level results to: v3/results/E2_seed_level_results.csv", flush=True)

# Compute R5: Selection effect across seeds 0..29
r5_max_acc_r4 = float(df_seeds.loc[df_seeds["seed"] < 30, "R4_acc"].max())
r5_max_auc_r4 = float(df_seeds.loc[df_seeds["seed"] < 30, "R4_auc"].max())

# Ladder summary table
ladder_rungs = [
    ("R0_Leak_Free", "R0_acc", "R0_auc", "In-fold median + in-fold scaling, default XGB"),
    ("R1_Global_Scaling", "R1_acc", "R1_auc", "R0 + StandardScaler fit on all 768 rows pre-split"),
    ("R2_Global_MICE", "R2_acc", "R2_auc", "R1 + MICE fit on all 768 rows pre-split (label-free)"),
    ("R3a_RLTR_Published", "R3a_acc", "R3a_auc", "RLTR file AS PUBLISHED (zeros intact) + pre-split MinMax"),
    ("R3b_RLTR_GlucoseFix", "R3b_acc", "R3b_auc", "RLTR file with in-fold glucose fix + MinMax"),
    ("R3a_LTR", "R3a_LTR_acc", "R3a_LTR_auc", "LTR file as published (zeros intact) + pre-split MinMax"),
    ("R3b_LTR", "R3b_LTR_acc", "R3b_LTR_auc", "LTR file with in-fold glucose fix + MinMax"),
    ("R3a_NSSR", "R3a_NSSR_acc", "R3a_NSSR_auc", "NSSR file as published (zeros intact) + pre-split MinMax"),
    ("R3b_NSSR", "R3b_NSSR_acc", "R3b_NSSR_auc", "NSSR file with in-fold glucose fix + MinMax"),
    ("R3a_SIM", "R3a_SIM_acc", "R3a_SIM_auc", "SIM file as published (zeros intact) + pre-split MinMax"),
    ("R3b_SIM", "R3b_SIM_acc", "R3b_SIM_auc", "SIM file with in-fold glucose fix + MinMax"),
    ("R3a_TR", "R3a_TR_acc", "R3a_TR_auc", "TR file as published (zeros intact) + pre-split MinMax"),
    ("R3b_TR", "R3b_TR_acc", "R3b_TR_auc", "TR file with in-fold glucose fix + MinMax"),
    ("R4_Prime_Honest_Tuned", "R4_prime_acc", "R4_prime_auc", "R3a + Optuna 50 trials maximizing 5-fold CV on train only"),
    ("R4_Direct_Test_Tuned", "R4_acc", "R4_auc", "R3a + Optuna 100 trials directly maximizing test accuracy"),
]

summary_records = []
for rung_name, acc_col, auc_col, desc in ladder_rungs:
    acc_s = df_seeds[acc_col]
    auc_s = df_seeds[auc_col]
    
    val_42_acc = float(df_seeds.loc[df_seeds["seed"] == 42, acc_col].iloc[0])
    val_42_auc = float(df_seeds.loc[df_seeds["seed"] == 42, auc_col].iloc[0])
    
    summary_records.append({
        "rung": rung_name,
        "description": desc,
        "accuracy_mean": float(acc_s.mean()),
        "accuracy_std": float(acc_s.std(ddof=1)),
        "accuracy_p025": float(np.percentile(acc_s, 2.5)),
        "accuracy_median": float(acc_s.median()),
        "accuracy_p975": float(np.percentile(acc_s, 97.5)),
        "accuracy_seed42": val_42_acc,
        "auc_mean": float(auc_s.mean()),
        "auc_std": float(auc_s.std(ddof=1)),
        "auc_p025": float(np.percentile(auc_s, 2.5)),
        "auc_median": float(auc_s.median()),
        "auc_p975": float(np.percentile(auc_s, 97.5)),
        "auc_seed42": val_42_auc
    })

# Add R5 explicitly
summary_records.append({
    "rung": "R5_Selection_Effect",
    "description": "Maximum test performance over seeds 0..29 under R4",
    "accuracy_mean": r5_max_acc_r4,
    "accuracy_std": 0.0,
    "accuracy_p025": r5_max_acc_r4,
    "accuracy_median": r5_max_acc_r4,
    "accuracy_p975": r5_max_acc_r4,
    "accuracy_seed42": float(df_seeds.loc[df_seeds["seed"] == 42, "R4_acc"].iloc[0]),
    "auc_mean": r5_max_auc_r4,
    "auc_std": 0.0,
    "auc_p025": r5_max_auc_r4,
    "auc_median": r5_max_auc_r4,
    "auc_p975": r5_max_auc_r4,
    "auc_seed42": float(df_seeds.loc[df_seeds["seed"] == 42, "R4_auc"].iloc[0])
})

df_summary = pd.DataFrame(summary_records)
df_summary.to_csv("v3/results/E2_inflation_ladder.csv", index=False)
print("Saved inflation ladder summary to: v3/results/E2_inflation_ladder.csv", flush=True)

# Save Seed 42 Comparison Across All Files and Variants
records_42 = []
for key, d in seed42_details.items():
    cm = d["cm"]
    tn, fp = cm[0]
    fn, tp = cm[1]
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
    records_42.append({
        "key": key,
        "dataset_file": d["file"],
        "variant": d["variant"],
        "accuracy": d["accuracy"],
        "roc_auc": d["roc_auc"],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1
    })
df_42 = pd.DataFrame(records_42)
df_42.to_csv("v3/results/E2_seed42_file_comparison.csv", index=False)
print("Saved Seed 42 comparison across all 5 files to: v3/results/E2_seed42_file_comparison.csv", flush=True)

# ------------------------------------------------------------------------------
# 4. PLOT WATERFALL FIGURE (300 DPI)
# ------------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

plot_rungs = [
    ("R0 (Leak-Free)", "R0_Leak_Free"),
    ("R1 (+Global Scale)", "R1_Global_Scaling"),
    ("R2 (+Global MICE)", "R2_Global_MICE"),
    ("R3a (RLTR As-Is)", "R3a_RLTR_Published"),
    ("R3b (RLTR GlucFix)", "R3b_RLTR_GlucoseFix"),
    ("R4' (Honest Tuned)", "R4_Prime_Honest_Tuned"),
    ("R4 (Test Tuned)", "R4_Direct_Test_Tuned"),
    ("R5 (Seed Selection)", "R5_Selection_Effect")
]

x_labels = [p[0] for p in plot_rungs]
rung_keys = [p[1] for p in plot_rungs]

sub_summary = df_summary.set_index("rung").loc[rung_keys]

acc_means = sub_summary["accuracy_mean"].values
acc_sds = sub_summary["accuracy_std"].values
acc_seed42 = sub_summary["accuracy_seed42"].values

auc_means = sub_summary["auc_mean"].values
auc_sds = sub_summary["auc_std"].values
auc_seed42 = sub_summary["auc_seed42"].values

colors = ["#2b5c8f", "#41729f", "#588bae", "#d95f02", "#f1a340", "#998ec3", "#e7298a", "#7570b3"]

# Panel 1: Accuracy
ax1.bar(range(len(x_labels)), acc_means, yerr=acc_sds, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=1)
ax1.scatter(range(len(x_labels)), acc_seed42, color="red", s=65, zorder=5, label="Seed 42 Holdout")
ax1.set_title("A. Test Accuracy Inflation Ladder Across 100 Seeds", fontsize=12, fontweight="bold", pad=10)
ax1.set_ylabel("Accuracy", fontsize=11)
ax1.set_ylim(0.65, 0.95)
ax1.set_xticks(range(len(x_labels)))
ax1.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=10)
ax1.axhline(0.8506, color="black", linestyle="--", alpha=0.6, label="Paper 1 Baseline (0.8506)")
ax1.axhline(0.8701, color="purple", linestyle=":", alpha=0.6, label="Paper 2 Target (0.8701)")
ax1.legend(loc="upper left", frameon=True, fontsize=9)

# Panel 2: ROC-AUC
ax2.bar(range(len(x_labels)), auc_means, yerr=auc_sds, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=1)
ax2.scatter(range(len(x_labels)), auc_seed42, color="red", s=65, zorder=5, label="Seed 42 Holdout")
ax2.set_title("B. Test ROC-AUC Progression Across 100 Seeds", fontsize=12, fontweight="bold", pad=10)
ax2.set_ylabel("ROC-AUC", fontsize=11)
ax2.set_ylim(0.70, 0.98)
ax2.set_xticks(range(len(x_labels)))
ax2.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=10)
ax2.legend(loc="upper left", frameon=True, fontsize=9)

plt.tight_layout()
fig_path = "v3/figures/E2_inflation_ladder_waterfall.png"
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"Saved 300 DPI waterfall plot to: {fig_path}", flush=True)

print("\n" + "="*75, flush=True)
print("EXPERIMENT E2 COMPLETED SUCCESSFULLY", flush=True)
print("="*75, flush=True)
