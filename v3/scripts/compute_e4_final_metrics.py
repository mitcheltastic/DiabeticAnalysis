import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.abspath("v3/scripts"))
from stats_utils import (
    compute_calibration_metrics,
    roc_auc_score,
    accuracy_score,
    brier_score_loss,
    nadeau_bengio_ttest,
    tost_equivalence,
    holm_bonferroni
)

def run_analysis():
    fold_csv = "v3/results/E4_fold_level_results.csv"
    stab_csv = "v3/results/E4_hyperparameter_stability.csv"
    
    if not os.path.exists(fold_csv) or not os.path.exists(stab_csv):
        print("Required CSV files not found.")
        return
        
    df_f = pd.read_csv(fold_csv)
    df_hp = pd.read_csv(stab_csv)
    
    folds = sorted(df_f["fold"].unique())
    n_folds = len(folds)
    print(f"Total outer folds in CSV: {n_folds}")
    
    # 1. Compute calibration metrics for each fold and model
    df_raw = pd.read_csv("Dataset Diabetes.csv", sep=";")
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df_raw[c] = df_raw[c].replace(0, np.nan)
        
    X = df_raw.drop(columns=["Outcome"]).copy()
    y = df_raw["Outcome"].values
    
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    splits = list(rskf.split(X, y))
    
    calib_records = []
    
    print(f"Extracting predicted probabilities and calibration metrics for {n_folds} folds...")
    for f_idx in folds:
        outer_fold_seed = 42 + f_idx
        train_idx, test_idx = splits[f_idx]
        X_tr_raw = X.iloc[train_idx].copy().reset_index(drop=True)
        X_te_raw = X.iloc[test_idx].copy().reset_index(drop=True)
        y_tr, y_te = y[train_idx], y[test_idx]
        
        imp = SimpleImputer(strategy="median")
        X_tr_imp = pd.DataFrame(imp.fit_transform(X_tr_raw), columns=X.columns)
        X_te_imp = pd.DataFrame(imp.transform(X_te_raw), columns=X.columns)
        
        scaler = StandardScaler()
        X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr_imp), columns=X.columns)
        X_te_scaled = pd.DataFrame(scaler.transform(X_te_imp), columns=X.columns)
        
        # XGB Default
        xgb_def = XGBClassifier(
            max_depth=6, n_estimators=100, learning_rate=0.1,
            random_state=outer_fold_seed, eval_metric="logloss", n_jobs=2
        )
        xgb_def.fit(X_tr_imp.values, y_tr)
        p_xgb_def = xgb_def.predict_proba(X_te_imp.values)[:, 1]
        cal_xgb_def = compute_calibration_metrics(y_te, p_xgb_def)
        calib_records.append({
            "fold": f_idx, "model": "XGBoost", "type": "Default",
            "calibration_slope": cal_xgb_def["calibration_slope"],
            "calibration_intercept": cal_xgb_def["calibration_intercept"]
        })
        
        # XGB Tuned
        sub_xgb = df_hp[(df_hp["fold"] == f_idx) & (df_hp["model"] == "XGBoost")].iloc[0]
        xgb_tun = XGBClassifier(
            max_depth=int(sub_xgb["max_depth"]),
            n_estimators=int(sub_xgb["n_estimators"]),
            learning_rate=float(sub_xgb["learning_rate"]),
            subsample=float(sub_xgb["subsample"]),
            colsample_bytree=float(sub_xgb["colsample_bytree"]),
            random_state=outer_fold_seed, eval_metric="logloss", n_jobs=2
        )
        xgb_tun.fit(X_tr_imp.values, y_tr)
        p_xgb_tun = xgb_tun.predict_proba(X_te_imp.values)[:, 1]
        cal_xgb_tun = compute_calibration_metrics(y_te, p_xgb_tun)
        calib_records.append({
            "fold": f_idx, "model": "XGBoost", "type": "Tuned",
            "calibration_slope": cal_xgb_tun["calibration_slope"],
            "calibration_intercept": cal_xgb_tun["calibration_intercept"]
        })
        
        # CatBoost Default
        cb_def = CatBoostClassifier(
            depth=6, iterations=300, learning_rate=0.05,
            random_seed=outer_fold_seed, verbose=False, thread_count=2
        )
        cb_def.fit(X_tr_imp.values, y_tr)
        p_cb_def = cb_def.predict_proba(X_te_imp.values)[:, 1]
        cal_cb_def = compute_calibration_metrics(y_te, p_cb_def)
        calib_records.append({
            "fold": f_idx, "model": "CatBoost", "type": "Default",
            "calibration_slope": cal_cb_def["calibration_slope"],
            "calibration_intercept": cal_cb_def["calibration_intercept"]
        })
        
        # CatBoost Tuned
        sub_cb = df_hp[(df_hp["fold"] == f_idx) & (df_hp["model"] == "CatBoost")].iloc[0]
        cb_tun = CatBoostClassifier(
            depth=int(sub_cb["depth"]),
            iterations=int(sub_cb["iterations"]),
            learning_rate=float(sub_cb["learning_rate"]),
            l2_leaf_reg=float(sub_cb["l2_leaf_reg"]),
            random_seed=outer_fold_seed, verbose=False, thread_count=2
        )
        cb_tun.fit(X_tr_imp.values, y_tr)
        p_cb_tun = cb_tun.predict_proba(X_te_imp.values)[:, 1]
        cal_cb_tun = compute_calibration_metrics(y_te, p_cb_tun)
        calib_records.append({
            "fold": f_idx, "model": "CatBoost", "type": "Tuned",
            "calibration_slope": cal_cb_tun["calibration_slope"],
            "calibration_intercept": cal_cb_tun["calibration_intercept"]
        })
        
        # LogReg-EN Default
        lr_def = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=outer_fold_seed)
        lr_def.fit(X_tr_scaled.values, y_tr)
        p_lr_def = lr_def.predict_proba(X_te_scaled.values)[:, 1]
        cal_lr_def = compute_calibration_metrics(y_te, p_lr_def)
        calib_records.append({
            "fold": f_idx, "model": "LogReg-EN", "type": "Default",
            "calibration_slope": cal_lr_def["calibration_slope"],
            "calibration_intercept": cal_lr_def["calibration_intercept"]
        })
        
        # LogReg-EN Tuned
        sub_lr = df_hp[(df_hp["fold"] == f_idx) & (df_hp["model"] == "LogReg-EN")].iloc[0]
        lr_tun = LogisticRegression(
            penalty="elasticnet",
            C=float(sub_lr["C"]),
            l1_ratio=float(sub_lr["l1_ratio"]),
            solver="saga", max_iter=1000, random_state=outer_fold_seed
        )
        lr_tun.fit(X_tr_scaled.values, y_tr)
        p_lr_tun = lr_tun.predict_proba(X_te_scaled.values)[:, 1]
        cal_lr_tun = compute_calibration_metrics(y_te, p_lr_tun)
        calib_records.append({
            "fold": f_idx, "model": "LogReg-EN", "type": "Tuned",
            "calibration_slope": cal_lr_tun["calibration_slope"],
            "calibration_intercept": cal_lr_tun["calibration_intercept"]
        })

    df_cal = pd.DataFrame(calib_records)
    df_merged = pd.merge(df_f, df_cal, on=["fold", "model", "type"])
    
    # Save enriched fold-level results
    enriched_file = "v3/results/E4_fold_level_results_enriched.csv"
    df_merged.to_csv(enriched_file, index=False)
    print(f"Saved enriched fold results to {enriched_file}")
    
    # Statistical Comparisons (Default vs Tuned per model)
    models = ["XGBoost", "CatBoost", "LogReg-EN"]
    metrics_list = ["roc_auc", "calibration_slope", "calibration_intercept", "brier", "accuracy"]
    k = n_folds
    n_test = 154
    n_train = 614
    
    print("\n" + "="*85)
    print("PART 1: DEFAULT VS TUNED PER MODEL (50-FOLD NESTED CV)")
    print("="*85)
    
    auc_p_values = []
    comparison_results = {}
    
    for m in models:
        df_m = df_merged[df_merged["model"] == m]
        def_df = df_m[df_m["type"] == "Default"].sort_values("fold")
        tun_df = df_m[df_m["type"] == "Tuned"].sort_values("fold")
        
        print(f"\nModel: {m}")
        print(f"{'-'*75}")
        
        model_stats = {}
        for met in metrics_list:
            v_def = def_df[met].values
            v_tun = tun_df[met].values
            d = v_tun - v_def
            
            tt = nadeau_bengio_ttest(d, k, n_test, n_train)
            tost = tost_equivalence(d, k, n_test, n_train, margin=0.02) if met == "roc_auc" else None
            
            mean_def, std_def = np.mean(v_def), np.std(v_def, ddof=1)
            mean_tun, std_tun = np.mean(v_tun), np.std(v_tun, ddof=1)
            
            model_stats[met] = {
                "def_mean": mean_def, "def_std": std_def,
                "tun_mean": mean_tun, "tun_std": std_tun,
                "diff": tt["mean_diff"], "ci_lower": tt["ci_lower"], "ci_upper": tt["ci_upper"],
                "p_value": tt["p_value"], "t_stat": tt["t_stat"],
                "tost": tost
            }
            
            if met == "roc_auc":
                auc_p_values.append((m, tt["p_value"]))
                print(f"  {met.upper()} (PRIMARY): Default = {mean_def:.4f} +- {std_def:.4f} | Tuned = {mean_tun:.4f} +- {std_tun:.4f}")
                print(f"    Diff = {tt['mean_diff']:+.4f}, NB 95% CI [{tt['ci_lower']:+.4f}, {tt['ci_upper']:+.4f}], t = {tt['t_stat']:+.3f}, p = {tt['p_value']:.4f}")
                print(f"    TOST (+-0.02 AUC): p_TOST = {tost['p_tost']:.4e}, Equivalent = {tost['equivalent']}")
            else:
                print(f"  {met}: Default = {mean_def:.4f} +- {std_def:.4f} | Tuned = {mean_tun:.4f} +- {std_tun:.4f} | Diff = {tt['mean_diff']:+.4f} [{tt['ci_lower']:+.4f}, {tt['ci_upper']:+.4f}], p = {tt['p_value']:.4f}")
                
        comparison_results[m] = model_stats
        
    # Apply Holm correction across the 3 models for ROC-AUC
    raw_ps = [p for _, p in auc_p_values]
    adj_ps = holm_bonferroni(raw_ps)
    print("\nHolm-Bonferroni Adjusted p-values for ROC-AUC Tuned vs Default:")
    for (m, raw_p), adj_p in zip(auc_p_values, adj_ps):
        print(f"  {m}: raw p = {raw_p:.4f}, Holm adjusted p = {adj_p:.4f}")
        
    # PART 2: Best Tuned Model vs LogReg-L2 Default (Simplest Baseline)
    print("\n" + "="*85)
    print("PART 2: BEST TUNED MODEL VS LOGREG-L2 DEFAULT (SIMPLEST BASELINE)")
    print("="*85)
    
    mean_tuned_aucs = {m: comparison_results[m]["roc_auc"]["tun_mean"] for m in models}
    best_m = max(mean_tuned_aucs, key=mean_tuned_aucs.get)
    print(f"Best Tuned Model by 50-Fold ROC-AUC: {best_m} (Mean AUC = {mean_tuned_aucs[best_m]:.4f})")
    
    # LogReg-L2 Default
    lr_def_df = df_merged[(df_merged["model"] == "LogReg-EN") & (df_merged["type"] == "Default")].sort_values("fold")
    best_tun_df = df_merged[(df_merged["model"] == best_m) & (df_merged["type"] == "Tuned")].sort_values("fold")
    
    for met in metrics_list:
        v_lr = lr_def_df[met].values
        v_best = best_tun_df[met].values
        d = v_best - v_lr
        
        tt = nadeau_bengio_ttest(d, k, n_test, n_train)
        tost = tost_equivalence(d, k, n_test, n_train, margin=0.02) if met == "roc_auc" else None
        
        m_lr, s_lr = np.mean(v_lr), np.std(v_lr, ddof=1)
        m_best, s_best = np.mean(v_best), np.std(v_best, ddof=1)
        
        print(f"\n{met.upper()} Comparison: {best_m} Tuned vs LogReg-L2 Default")
        print(f"  LogReg-L2 Default: {m_lr:.4f} +- {s_lr:.4f}")
        print(f"  {best_m} Tuned:      {m_best:.4f} +- {s_best:.4f}")
        print(f"  Diff = {tt['mean_diff']:+.4f}, NB 95% CI [{tt['ci_lower']:+.4f}, {tt['ci_upper']:+.4f}], t = {tt['t_stat']:+.3f}, p = {tt['p_value']:.4f}")
        if tost:
            print(f"  TOST (+-0.02 AUC): p_TOST = {tost['p_tost']:.4e}, Equivalent = {tost['equivalent']}")
            
    # PART 3: Hyperparameter Stability
    print("\n" + "="*85)
    print("PART 3: HYPERPARAMETER STABILITY ACROSS 50 FOLDS")
    print("="*85)
    
    bounds = {
        "XGBoost": {
            "max_depth": (2, 8, "int"),
            "n_estimators": (50, 300, "int"),
            "learning_rate": (0.01, 0.3, "float_log"),
            "subsample": (0.6, 1.0, "float"),
            "colsample_bytree": (0.6, 1.0, "float")
        },
        "CatBoost": {
            "depth": (3, 8, "int"),
            "iterations": (100, 500, "int"),
            "learning_rate": (0.01, 0.2, "float"),
            "l2_leaf_reg": (1.0, 10.0, "float")
        },
        "LogReg-EN": {
            "C": (1e-3, 1e2, "float_log"),
            "l1_ratio": (0.0, 1.0, "float")
        }
    }
    
    for m, p_dict in bounds.items():
        print(f"\nModel: {m}")
        sub = df_hp[df_hp["model"] == m]
        for p, (low, high, ptype) in p_dict.items():
            vals = sub[p].dropna().values
            if len(vals) == 0:
                continue
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            iqr = q75 - q25
            if ptype == "int":
                at_min = np.sum(vals == low)
                at_max = np.sum(vals == high)
            elif ptype == "float_log":
                at_min = np.sum(vals <= low * 1.05)
                at_max = np.sum(vals >= high * 0.95)
            else:
                at_min = np.sum(vals <= low + (high - low) * 0.01)
                at_max = np.sum(vals >= high - (high - low) * 0.01)
            pct_boundary = (at_min + at_max) / len(vals) * 100
            print(f"  {p:18s}: Median = {q50:9.4f}, IQR = {iqr:8.4f} [Q25={q25:8.4f}, Q75={q75:8.4f}] | Boundary Hits: {at_min + at_max:2d}/{len(vals)} ({pct_boundary:5.1f}%) [min={at_min:2d}, max={at_max:2d}]")

    # PART 4: Inner-CV AUC vs Outer-Fold AUC (Tuning Optimism)
    print("\n" + "="*85)
    print("PART 4: TUNING OPTIMISM (INNER-CV BEST AUC VS OUTER-FOLD AUC)")
    print("="*85)
    for m in models:
        sub_hp = df_hp[df_hp["model"] == m].sort_values("fold")
        sub_f = df_merged[(df_merged["model"] == m) & (df_merged["type"] == "Tuned")].sort_values("fold")
        
        inner_auc = sub_hp["inner_cv_best_auc"].values
        outer_auc = sub_f["roc_auc"].values
        optimism = inner_auc - outer_auc
        
        tt_opt = nadeau_bengio_ttest(optimism, k, n_test, n_train)
        print(f"\nModel: {m}")
        print(f"  Inner-CV AUC: {np.mean(inner_auc):.4f} +- {np.std(inner_auc, ddof=1):.4f}")
        print(f"  Outer-Fold AUC: {np.mean(outer_auc):.4f} +- {np.std(outer_auc, ddof=1):.4f}")
        print(f"  Optimism (Inner - Outer): {tt_opt['mean_diff']:+.4f} +- {np.std(optimism, ddof=1):.4f}")
        print(f"  NB 95% CI: [{tt_opt['ci_lower']:+.4f}, {tt_opt['ci_upper']:+.4f}], t = {tt_opt['t_stat']:+.3f}, p = {tt_opt['p_value']:.4f}")

if __name__ == "__main__":
    run_analysis()
