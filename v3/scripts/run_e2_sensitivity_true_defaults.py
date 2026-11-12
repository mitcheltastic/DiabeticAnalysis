import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath("v3/scripts"))
from stats_utils import assert_disjoint, nadeau_bengio_ttest

def main():
    print("="*75)
    print("EXPERIMENT E2 SENSITIVITY CHECK: TRUE LIBRARY DEFAULTS VS REFERENCE CONFIG (100 SEEDS)")
    print("="*75)
    
    raw = pd.read_csv("Dataset Diabetes.csv", sep=";")
    y = raw["Outcome"].values
    n_samples = len(raw)
    
    # Missing columns for R0 leak-free
    missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    raw_nan = raw.copy()
    for col in missing_cols:
        raw_nan[col] = raw_nan[col].replace(0, np.nan)
    X_raw = raw_nan.drop(columns=["Outcome"]).copy()
    
    # Load published imputed files for R3a (pre-split MinMaxScaler)
    files = {
        "TR": "TR_Imputed.csv",
        "LTR": "LTR_Imputed.csv",
        "RLTR": "RLTR_Imputed.csv",
        "NSSR": "NSSR_Imputed.csv",
        "SIM": "SIM_Imputed.csv"
    }
    dfs_r3a_scaled = {}
    for k, fname in files.items():
        df_k = pd.read_csv(fname)
        X_k = df_k.drop(columns=["Outcome"]).copy()
        mms = MinMaxScaler()
        dfs_r3a_scaled[k] = pd.DataFrame(mms.fit_transform(X_k), columns=X_k.columns)
        
    n_seeds = 100
    
    records = []
    
    print(f"Running 100 seeds across R0 and R3a (5 files) for both configurations...")
    for s in range(n_seeds):
        train_idx, test_idx = train_test_split(
            np.arange(n_samples),
            test_size=0.20,
            random_state=s,
            stratify=y
        )
        assert_disjoint(train_idx, test_idx)
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # ----------------------------------------------------------------------
        # 1. Rung R0: Leak-Free Baseline
        # ----------------------------------------------------------------------
        X_tr_r0 = X_raw.iloc[train_idx].copy()
        X_te_r0 = X_raw.iloc[test_idx].copy()
        
        imp_r0 = SimpleImputer(strategy="median")
        X_tr_r0_imp = imp_r0.fit_transform(X_tr_r0)
        X_te_r0_imp = imp_r0.transform(X_te_r0)
        
        scaler_r0 = StandardScaler()
        X_tr_r0_s = scaler_r0.fit_transform(X_tr_r0_imp)
        X_te_r0_s = scaler_r0.transform(X_te_r0_imp)
        
        # Reference configuration (lr=0.1, depth=6, n_est=100)
        xgb_r0_ref = XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=4)
        xgb_r0_ref.fit(X_tr_r0_s, y_tr)
        p_r0_ref = xgb_r0_ref.predict_proba(X_te_r0_s)[:, 1]
        y_r0_ref = xgb_r0_ref.predict(X_te_r0_s)
        
        # True library defaults (learning_rate=0.3, max_depth=6, n_estimators=100)
        xgb_r0_true = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=4)
        xgb_r0_true.fit(X_tr_r0_s, y_tr)
        p_r0_true = xgb_r0_true.predict_proba(X_te_r0_s)[:, 1]
        y_r0_true = xgb_r0_true.predict(X_te_r0_s)
        
        row = {
            "seed": s,
            "R0_ref_acc": accuracy_score(y_te, y_r0_ref),
            "R0_ref_auc": roc_auc_score(y_te, p_r0_ref),
            "R0_true_acc": accuracy_score(y_te, y_r0_true),
            "R0_true_auc": roc_auc_score(y_te, p_r0_true),
        }
        
        # ----------------------------------------------------------------------
        # 2. Rung R3a: Published Files as-is (5 files)
        # ----------------------------------------------------------------------
        for f_name, df_scaled in dfs_r3a_scaled.items():
            X_tr_3a = df_scaled.iloc[train_idx].values
            X_te_3a = df_scaled.iloc[test_idx].values
            
            # Reference configuration
            xgb_3a_ref = XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss", n_jobs=4)
            xgb_3a_ref.fit(X_tr_3a, y_tr)
            p_3a_ref = xgb_3a_ref.predict_proba(X_te_3a)[:, 1]
            y_3a_ref = xgb_3a_ref.predict(X_te_3a)
            
            # True library defaults
            xgb_3a_true = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=4)
            xgb_3a_true.fit(X_tr_3a, y_tr)
            p_3a_true = xgb_3a_true.predict_proba(X_te_3a)[:, 1]
            y_3a_true = xgb_3a_true.predict(X_te_3a)
            
            row[f"R3a_{f_name}_ref_acc"] = accuracy_score(y_te, y_3a_ref)
            row[f"R3a_{f_name}_ref_auc"] = roc_auc_score(y_te, p_3a_ref)
            row[f"R3a_{f_name}_true_acc"] = accuracy_score(y_te, y_3a_true)
            row[f"R3a_{f_name}_true_auc"] = roc_auc_score(y_te, p_3a_true)
            
        records.append(row)
        if (s + 1) % 10 == 0:
            print(f"  Completed {s + 1}/{n_seeds} seeds...", flush=True)
        
    df_seeds = pd.DataFrame(records)
    df_seeds.to_csv("v3/results/E2_sensitivity_true_defaults_seed_level.csv", index=False)
    
    # Compile comparison table
    k = 100
    n_test = 154
    n_train = 614
    
    targets = [
        ("R0 (Leak-Free Baseline)", "R0_ref_acc", "R0_ref_auc", "R0_true_acc", "R0_true_auc"),
        ("R3a_TR", "R3a_TR_ref_acc", "R3a_TR_ref_auc", "R3a_TR_true_acc", "R3a_TR_true_auc"),
        ("R3a_LTR", "R3a_LTR_ref_acc", "R3a_LTR_ref_auc", "R3a_LTR_true_acc", "R3a_LTR_true_auc"),
        ("R3a_RLTR", "R3a_RLTR_ref_acc", "R3a_RLTR_ref_auc", "R3a_RLTR_true_acc", "R3a_RLTR_true_auc"),
        ("R3a_NSSR", "R3a_NSSR_ref_acc", "R3a_NSSR_ref_auc", "R3a_NSSR_true_acc", "R3a_NSSR_true_auc"),
        ("R3a_SIM", "R3a_SIM_ref_acc", "R3a_SIM_ref_auc", "R3a_SIM_true_acc", "R3a_SIM_true_auc"),
    ]
    
    summary = []
    for label, ref_acc_col, ref_auc_col, true_acc_col, true_auc_col in targets:
        ref_acc = df_seeds[ref_acc_col].values
        ref_auc = df_seeds[ref_auc_col].values
        true_acc = df_seeds[true_acc_col].values
        true_auc = df_seeds[true_auc_col].values
        
        # Seed 42 values
        s42 = df_seeds[df_seeds["seed"] == 42].iloc[0]
        
        # Paired test: true defaults vs reference configuration
        diff_acc = true_acc - ref_acc
        diff_auc = true_auc - ref_auc
        tt_acc = nadeau_bengio_ttest(diff_acc, k, n_test, n_train)
        tt_auc = nadeau_bengio_ttest(diff_auc, k, n_test, n_train)
        
        summary.append({
            "target": label,
            "ref_acc_mean": np.mean(ref_acc),
            "ref_acc_std": np.std(ref_acc, ddof=1),
            "ref_acc_seed42": s42[ref_acc_col],
            "ref_auc_mean": np.mean(ref_auc),
            "ref_auc_std": np.std(ref_auc, ddof=1),
            "ref_auc_seed42": s42[ref_auc_col],
            "true_acc_mean": np.mean(true_acc),
            "true_acc_std": np.std(true_acc, ddof=1),
            "true_acc_seed42": s42[true_acc_col],
            "true_auc_mean": np.mean(true_auc),
            "true_auc_std": np.std(true_auc, ddof=1),
            "true_auc_seed42": s42[true_auc_col],
            "diff_acc_true_minus_ref": tt_acc["mean_diff"],
            "nb_p_acc": tt_acc["p_value"],
            "diff_auc_true_minus_ref": tt_auc["mean_diff"],
            "nb_p_auc": tt_auc["p_value"]
        })
        
    df_summary = pd.DataFrame(summary)
    summary_path = "v3/results/E2_sensitivity_true_defaults.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved sensitivity summary to: {summary_path}")
    
    # Compute R3a - R0 inflation gaps under both configurations
    print("\n" + "="*75)
    print("INFLATION GAPS (R3a File - R0 Baseline) UNDER BOTH CONFIGURATIONS")
    print("="*75)
    gaps = []
    for f_name in ["TR", "LTR", "RLTR", "NSSR", "SIM"]:
        # Reference configuration gap:
        gap_acc_ref = df_seeds[f"R3a_{f_name}_ref_acc"].values - df_seeds["R0_ref_acc"].values
        gap_auc_ref = df_seeds[f"R3a_{f_name}_ref_auc"].values - df_seeds["R0_ref_auc"].values
        tt_gap_acc_ref = nadeau_bengio_ttest(gap_acc_ref, k, n_test, n_train)
        tt_gap_auc_ref = nadeau_bengio_ttest(gap_auc_ref, k, n_test, n_train)
        
        # True defaults gap:
        gap_acc_true = df_seeds[f"R3a_{f_name}_true_acc"].values - df_seeds["R0_true_acc"].values
        gap_auc_true = df_seeds[f"R3a_{f_name}_true_auc"].values - df_seeds["R0_true_auc"].values
        tt_gap_acc_true = nadeau_bengio_ttest(gap_acc_true, k, n_test, n_train)
        tt_gap_auc_true = nadeau_bengio_ttest(gap_auc_true, k, n_test, n_train)
        
        gaps.append({
            "file": f_name,
            # Reference configuration
            "gap_acc_ref_mean": tt_gap_acc_ref["mean_diff"],
            "gap_acc_ref_ci_low": tt_gap_acc_ref["ci_lower"],
            "gap_acc_ref_ci_high": tt_gap_acc_ref["ci_upper"],
            "gap_acc_ref_p": tt_gap_acc_ref["p_value"],
            "gap_auc_ref_mean": tt_gap_auc_ref["mean_diff"],
            "gap_auc_ref_ci_low": tt_gap_auc_ref["ci_lower"],
            "gap_auc_ref_ci_high": tt_gap_auc_ref["ci_upper"],
            "gap_auc_ref_p": tt_gap_auc_ref["p_value"],
            # True library defaults
            "gap_acc_true_mean": tt_gap_acc_true["mean_diff"],
            "gap_acc_true_ci_low": tt_gap_acc_true["ci_lower"],
            "gap_acc_true_ci_high": tt_gap_acc_true["ci_upper"],
            "gap_acc_true_p": tt_gap_acc_true["p_value"],
            "gap_auc_true_mean": tt_gap_auc_true["mean_diff"],
            "gap_auc_true_ci_low": tt_gap_auc_true["ci_lower"],
            "gap_auc_true_ci_high": tt_gap_auc_true["ci_upper"],
            "gap_auc_true_p": tt_gap_auc_true["p_value"],
        })
        
    df_gaps = pd.DataFrame(gaps)
    gap_path = "v3/results/E2_sensitivity_inflation_gaps.csv"
    df_gaps.to_csv(gap_path, index=False)
    print(f"Saved inflation gaps summary to: {gap_path}")
    print(df_gaps)

if __name__ == "__main__":
    main()

