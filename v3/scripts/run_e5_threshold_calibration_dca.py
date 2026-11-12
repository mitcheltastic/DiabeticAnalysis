import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_recall_fscore_support,
    brier_score_loss,
    confusion_matrix,
    f1_score
)

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath("v3/scripts"))
from stats_utils import assert_disjoint, compute_calibration_metrics

def find_threshold_max_f1(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_th = 0.50
    best_f1 = -1.0
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return best_th, best_f1

def find_threshold_high_sensitivity(y_true, y_prob, target_sens=0.80):
    thresholds = np.linspace(0.05, 0.95, 91)
    valid_candidates = []
    for th in thresholds:
        pred = (y_prob >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        if sens >= target_sens:
            valid_candidates.append((th, sens, spec))
    if not valid_candidates:
        return 0.50, 0.0, 0.0
    # Maximize specificity among candidates with sens >= target_sens
    valid_candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)
    best_th, best_sens, best_spec = valid_candidates[0]
    return best_th, best_sens, best_spec

def compute_dca(y_true, y_prob, pt_range=np.linspace(0.05, 0.60, 56)):
    n = len(y_true)
    prev = np.mean(y_true)
    dca_records = []
    for pt in pt_range:
        pred = (y_prob >= pt).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        
        weight = pt / (1.0 - pt)
        nb_model = (tp / n) - (fp / n) * weight
        nb_all = prev - (1.0 - prev) * weight
        nb_none = 0.0
        
        dca_records.append({
            "threshold_probability": pt,
            "net_benefit_model": nb_model,
            "net_benefit_all": nb_all,
            "net_benefit_none": nb_none
        })
    return pd.DataFrame(dca_records)

def main():
    print("="*75)
    print("EXPERIMENT E5: THRESHOLD CALIBRATION & DECISION CURVE ANALYSIS")
    print("="*75)
    os.makedirs("v3/results", exist_ok=True)
    os.makedirs("v3/figures", exist_ok=True)
    
    # Load dataset
    raw = pd.read_csv("Dataset Diabetes.csv", sep=";")
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        raw[c] = raw[c].replace(0, np.nan)
        
    X = raw.drop(columns=["Outcome"]).copy()
    y = raw["Outcome"].values
    
    # --------------------------------------------------------------------------
    # PART 1: 50 OUTER FOLDS NESTED THRESHOLD SELECTION
    # --------------------------------------------------------------------------
    print("Running 50 outer folds nested threshold selection...")
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    
    fold_records = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        assert_disjoint(train_idx, test_idx)
        outer_seed = 42 + fold_idx
        
        X_tr = X.iloc[train_idx].copy().reset_index(drop=True)
        X_te = X.iloc[test_idx].copy().reset_index(drop=True)
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # In-fold median imputation
        imp = SimpleImputer(strategy="median")
        X_tr_imp = pd.DataFrame(imp.fit_transform(X_tr), columns=X.columns)
        X_te_imp = pd.DataFrame(imp.transform(X_te), columns=X.columns)
        
        # In-fold StandardScaler
        scaler = StandardScaler()
        X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr_imp), columns=X.columns)
        X_te_sc = pd.DataFrame(scaler.transform(X_te_imp), columns=X.columns)
        
        # Inner CV on X_tr for threshold selection (leak-free)
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=outer_seed)
        clf_inner = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=outer_seed)
        
        # Out-of-fold predictions on training set
        y_tr_prob = cross_val_predict(clf_inner, X_tr_sc.values, y_tr, cv=inner_cv, method="predict_proba")[:, 1]
        
        # Select thresholds on inner OOF predictions
        th_max_f1, inner_f1 = find_threshold_max_f1(y_tr, y_tr_prob)
        th_high_sens, inner_sens, inner_spec = find_threshold_high_sensitivity(y_tr, y_tr_prob, target_sens=0.80)
        
        # Fit final model on full training fold
        clf_outer = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=outer_seed)
        clf_outer.fit(X_tr_sc.values, y_tr)
        y_te_prob = clf_outer.predict_proba(X_te_sc.values)[:, 1]
        
        # Calibration metrics on outer fold
        cal_met = compute_calibration_metrics(y_te, y_te_prob)
        auc_val = roc_auc_score(y_te, y_te_prob)
        
        # Policies evaluation on unseen test fold
        policies = [
            ("Default (0.50)", 0.50),
            ("Policy A (Max F1)", th_max_f1),
            ("Policy B (Sensitivity >= 0.80)", th_high_sens)
        ]
        
        for p_name, th in policies:
            y_pred = (y_te_prob >= th).astype(int)
            acc = accuracy_score(y_te, y_pred)
            cm = confusion_matrix(y_te, y_pred).ravel()
            tn, fp, fn, tp = cm
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1 = f1_score(y_te, y_pred, zero_division=0)
            p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_te, y_pred, average="weighted", zero_division=0)
            
            fold_records.append({
                "fold": fold_idx,
                "policy": p_name,
                "threshold": th,
                "roc_auc": auc_val,
                "calibration_slope": cal_met["calibration_slope"],
                "calibration_intercept": cal_met["calibration_intercept"],
                "brier": cal_met["brier_score"],
                "accuracy": acc,
                "sensitivity": sens,
                "specificity": spec,
                "f1": f1,
                "weighted_f1": f1_w,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp
            })
            
    df_folds = pd.DataFrame(fold_records)
    nested_res_path = "v3/results/E5_nested_thresholds_results.csv"
    df_folds.to_csv(nested_res_path, index=False)
    print(f"Saved nested CV threshold results to: {nested_res_path}")
    
    # Compute summary across 50 folds
    summary_records = []
    for p_name in ["Default (0.50)", "Policy A (Max F1)", "Policy B (Sensitivity >= 0.80)"]:
        sub = df_folds[df_folds["policy"] == p_name]
        summary_records.append({
            "policy": p_name,
            "threshold_mean": sub["threshold"].mean(),
            "threshold_std": sub["threshold"].std(),
            "accuracy_mean": sub["accuracy"].mean(),
            "accuracy_std": sub["accuracy"].std(),
            "sensitivity_mean": sub["sensitivity"].mean(),
            "sensitivity_std": sub["sensitivity"].std(),
            "specificity_mean": sub["specificity"].mean(),
            "specificity_std": sub["specificity"].std(),
            "f1_mean": sub["f1"].mean(),
            "f1_std": sub["f1"].std(),
            "weighted_f1_mean": sub["weighted_f1"].mean(),
            "weighted_f1_std": sub["weighted_f1"].std(),
            "roc_auc_mean": sub["roc_auc"].mean(),
            "brier_mean": sub["brier"].mean(),
            "cal_slope_mean": sub["calibration_slope"].mean(),
            "cal_intercept_mean": sub["calibration_intercept"].mean()
        })
    df_summary = pd.DataFrame(summary_records)
    summary_path = "v3/results/E5_threshold_calibration_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved threshold summary to: {summary_path}")
    
    # --------------------------------------------------------------------------
    # PART 2: SEED 42 HOLDOUT EVALUATION (CLINICAL TEST BENCHMARK)
    # --------------------------------------------------------------------------
    print("\nRunning Seed 42 holdout evaluation (614 train / 154 test)...")
    skf_seed42 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(skf_seed42.split(X, y))
    assert_disjoint(train_idx, test_idx)
    
    X_tr42 = X.iloc[train_idx].copy().reset_index(drop=True)
    X_te42 = X.iloc[test_idx].copy().reset_index(drop=True)
    y_tr42, y_te42 = y[train_idx], y[test_idx]
    
    imp42 = SimpleImputer(strategy="median")
    X_tr42_imp = pd.DataFrame(imp42.fit_transform(X_tr42), columns=X.columns)
    X_te42_imp = pd.DataFrame(imp42.transform(X_te42), columns=X.columns)
    
    scaler42 = StandardScaler()
    X_tr42_sc = pd.DataFrame(scaler42.fit_transform(X_tr42_imp), columns=X.columns)
    X_te42_sc = pd.DataFrame(scaler42.transform(X_te42_imp), columns=X.columns)
    
    # Inner CV on 614 training rows ONLY
    inner_cv42 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_inner42 = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
    y_tr42_prob = cross_val_predict(clf_inner42, X_tr42_sc.values, y_tr42, cv=inner_cv42, method="predict_proba")[:, 1]
    
    th_max_f1_42, tr_f1_42 = find_threshold_max_f1(y_tr42, y_tr42_prob)
    th_high_sens_42, tr_sens_42, tr_spec_42 = find_threshold_high_sensitivity(y_tr42, y_tr42_prob, target_sens=0.80)
    
    # Fit final model on 614 training rows
    clf_final = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
    clf_final.fit(X_tr42_sc.values, y_tr42)
    y_te42_prob = clf_final.predict_proba(X_te42_sc.values)[:, 1]
    
    cal_met42 = compute_calibration_metrics(y_te42, y_te42_prob)
    auc42 = roc_auc_score(y_te42, y_te42_prob)
    
    holdout_records = []
    policies_42 = [
        ("Default (0.50)", 0.50),
        ("Policy A (Max F1)", th_max_f1_42),
        ("Policy B (Sensitivity >= 0.80)", th_high_sens_42)
    ]
    for p_name, th in policies_42:
        y_pred = (y_te42_prob >= th).astype(int)
        acc = accuracy_score(y_te42, y_pred)
        cm = confusion_matrix(y_te42, y_pred).ravel()
        tn, fp, fn, tp = cm
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_te42, y_pred, zero_division=0)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_te42, y_pred, average="weighted", zero_division=0)
        
        holdout_records.append({
            "policy": p_name,
            "threshold": th,
            "roc_auc": auc42,
            "calibration_slope": cal_met42["calibration_slope"],
            "calibration_intercept": cal_met42["calibration_intercept"],
            "brier": cal_met42["brier_score"],
            "accuracy": acc,
            "sensitivity": sens,
            "specificity": spec,
            "f1": f1,
            "weighted_f1": f1_w,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp
        })
    df_holdout = pd.DataFrame(holdout_records)
    holdout_path = "v3/results/E5_seed42_holdout_results.csv"
    df_holdout.to_csv(holdout_path, index=False)
    print(f"Saved Seed 42 holdout results to: {holdout_path}")
    
    # --------------------------------------------------------------------------
    # PART 3: RELIABILITY DIAGRAM (CALIBRATION CURVE)
    # --------------------------------------------------------------------------
    prob_true, prob_pred = calibration_curve(y_te42, y_te42_prob, n_bins=10, strategy="uniform")
    
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Slope = 1.0, CITL = 0.0)")
    plt.plot(prob_pred, prob_true, "s-", color="#1f77b4", label=f"Final Model (LogReg-L2)\nSlope = {cal_met42['calibration_slope']:.3f}, CITL = {cal_met42['calibration_intercept']:.3f}")
    plt.title("Reliability Diagram: Final Model Calibration (Seed 42 Holdout, n=154)", fontsize=12, pad=12)
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Observed Fraction of Positives", fontsize=11)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    cal_fig_path = "v3/figures/E5_reliability_diagram_seed42.png"
    plt.savefig(cal_fig_path, dpi=300)
    plt.close()
    print(f"Saved Reliability Diagram to: {cal_fig_path}")
    
    # --------------------------------------------------------------------------
    # PART 4: DECISION CURVE ANALYSIS (DCA)
    # --------------------------------------------------------------------------
    df_dca = compute_dca(y_te42, y_te42_prob, pt_range=np.linspace(0.05, 0.60, 56))
    dca_csv_path = "v3/results/E5_decision_curve_analysis.csv"
    df_dca.to_csv(dca_csv_path, index=False)
    print(f"Saved DCA table to: {dca_csv_path}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(df_dca["threshold_probability"], df_dca["net_benefit_model"], "-", color="#1f77b4", linewidth=2.2, label="Final Model (LogReg-L2)")
    plt.plot(df_dca["threshold_probability"], df_dca["net_benefit_all"], "--", color="#d62728", linewidth=1.8, label="Treat All")
    plt.plot(df_dca["threshold_probability"], df_dca["net_benefit_none"], ":", color="gray", linewidth=1.5, label="Treat None")
    plt.title("Decision Curve Analysis: Clinical Net Benefit (Seed 42 Holdout)", fontsize=12, pad=12)
    plt.xlabel("Threshold Probability (p_t)", fontsize=11)
    plt.ylabel("Net Benefit", fontsize=11)
    plt.xlim([0.05, 0.60])
    plt.ylim([-0.05, max(df_dca["net_benefit_model"].max() * 1.15, 0.35)])
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    dca_fig_path = "v3/figures/E5_decision_curve_analysis.png"
    plt.savefig(dca_fig_path, dpi=300)
    plt.close()
    print(f"Saved DCA figure to: {dca_fig_path}")
    
    print("\n" + "="*75)
    print("EXPERIMENT E5 COMPLETED SUCCESSFULLY")
    print("="*75)

if __name__ == "__main__":
    main()
