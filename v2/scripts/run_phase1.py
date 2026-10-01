import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from scipy.stats import binomtest

os.makedirs("v2/results", exist_ok=True)

print("="*70)
print("PHASE 1: BASELINE REPRODUCTION, MCNEMAR TEST & BOOTSTRAP CIs")
print("="*70)

df = pd.read_csv("RLTR_Imputed.csv")
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# 1. Reproduce Upperclassmen Paper Baseline
# Split 80:20, random_state=42, stratify=y
X_tr_42, X_te_42, y_tr_42, y_te_42 = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

xgb_paper = XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss")
xgb_paper.fit(X_tr_42, y_tr_42)
preds_paper_42 = xgb_paper.predict(X_te_42)

cm_p42 = confusion_matrix(y_te_42, preds_paper_42)
tn_p, fp_p, fn_p, tp_p = cm_p42.ravel()
acc_p = (tn_p + tp_p) / len(y_te_42)
sens_p = tp_p / (tp_p + fn_p)
spec_p = tn_p / (tn_p + fp_p)
prec_p = tp_p / (tp_p + fp_p)
f1_p = 2 * (prec_p * sens_p) / (prec_p + sens_p)

print(f"Paper XGB (RS=42 Holdout):")
print(f"  CM: TN={tn_p}, FP={fp_p}, FN={fn_p}, TP={tp_p}")
print(f"  Accuracy: {acc_p:.4f} (Paper reported: 0.8506)")
print(f"  Precision: {prec_p:.4f} (Paper reported: 0.80)")
print(f"  Recall (Sensitivity): {sens_p:.4f} (Paper reported: 0.76)")
print(f"  Specificity: {spec_p:.4f} (Paper reported: 0.90)")
print(f"  F1-Score: {f1_p:.4f} (Paper reported: 0.78)")

# 2. Consensus Model Setup
g_med = X.loc[X["Glucose"] > 0, "Glucose"].median()
X_clean = X.copy()
X_clean["Glucose"] = X_clean["Glucose"].replace(0, g_med)
X_fe = X_clean.copy()
X_fe["HOMA_IR"] = (X_fe["Glucose"] * X_fe["Insulin"]) / 405.0
X_fe["Glucose_Insulin"] = X_fe["Glucose"] * X_fe["Insulin"]
X_fe["Insulin_BMI"] = X_fe["Insulin"] * X_fe["BMI"]
X_fe["Glucose_Age"] = X_fe["Glucose"] * X_fe["Age"]
X_fe["Glucose_BMI"] = X_fe["Glucose"] * X_fe["BMI"]
X_fe["RiskScore"] = (X_fe["Glucose"] >= 140).astype(int) * 2 + (X_fe["BMI"] >= 30).astype(int) + (X_fe["Age"] >= 35).astype(int)

top_cols = ["Glucose", "Insulin", "BMI", "Age", "HOMA_IR", "Glucose_Insulin", "Insulin_BMI", "Glucose_Age", "RiskScore", "Glucose_BMI", "DiabetesPedigreeFunction"]
X_opt = X_fe[top_cols]

def get_consensus_model():
    cb = CatBoostClassifier(iterations=350, depth=4, learning_rate=0.03, l2_leaf_reg=5, verbose=0, random_seed=42)
    lgb = LGBMClassifier(n_estimators=300, max_depth=3, num_leaves=15, learning_rate=0.03, verbose=-1, random_state=42)
    xgb = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.03, eval_metric="logloss", verbosity=0, random_state=42)
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    return VotingClassifier(estimators=[("cb", cb), ("lgb", lgb), ("xgb", xgb), ("rf", rf)], voting="soft", weights=[0.35, 0.25, 0.20, 0.20])

# Bootstrap CI function
def bootstrap_metrics(y_true, y_pred, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    accs, senss, specs, f1s = [], [], [], []
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt_sample = y_true_arr[idx]
        yp_sample = y_pred_arr[idx]
        
        cm = confusion_matrix(yt_sample, yp_sample, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        acc = (tn + tp) / n
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        
        accs.append(acc)
        senss.append(sens)
        specs.append(spec)
        f1s.append(f1)
        
    def ci_str(vals):
        low, high = np.percentile(vals, [2.5, 97.5])
        return f"[{low:.4f} - {high:.4f}]", low, high
        
    return {
        "Accuracy_CI": ci_str(accs),
        "Sensitivity_CI": ci_str(senss),
        "Specificity_CI": ci_str(specs),
        "F1_CI": ci_str(f1s)
    }

# Run comparison on BOTH Seed 42 (Paper's split) and Seed 12 (v1 Demo split)
results_summary = []

for split_seed in [42, 12]:
    print(f"\n--- EVALUATING ON HOLDOUT SPLIT SEED = {split_seed} ---")
    
    # 1. Paper XGB
    X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(X, y, test_size=0.20, random_state=split_seed, stratify=y)
    xgb_paper = XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss")
    xgb_paper.fit(X_tr_p, y_tr_p)
    preds_p = xgb_paper.predict(X_te_p)
    
    # 2. Consensus Ensemble
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_opt, y, test_size=0.20, random_state=split_seed, stratify=y)
    scaler = RobustScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr_c), columns=top_cols)
    X_te_s = pd.DataFrame(scaler.transform(X_te_c), columns=top_cols)
    ens = get_consensus_model()
    ens.fit(X_tr_s, y_tr_c)
    preds_c = ens.predict(X_te_s)
    
    # Verify same true labels
    assert np.array_equal(y_te_p.values, y_te_c.values), "Test set labels must match!"
    y_test_arr = y_te_p.values
    
    # Metrics
    cm_p = confusion_matrix(y_test_arr, preds_p)
    cm_c = confusion_matrix(y_test_arr, preds_c)
    
    acc_p = accuracy_score(y_test_arr, preds_p)
    acc_c = accuracy_score(y_test_arr, preds_c)
    
    sens_p = recall_score(y_test_arr, preds_p)
    sens_c = recall_score(y_test_arr, preds_c)
    
    spec_p = cm_p[0, 0] / (cm_p[0, 0] + cm_p[0, 1])
    spec_c = cm_c[0, 0] / (cm_c[0, 0] + cm_c[0, 1])
    
    f1_p = f1_score(y_test_arr, preds_p)
    f1_c = f1_score(y_test_arr, preds_c)
    
    # McNemar's Test Contingency Table
    # b: Paper XGB correct, Consensus wrong
    # c: Paper XGB wrong, Consensus correct
    correct_p = (preds_p == y_test_arr)
    correct_c = (preds_c == y_test_arr)
    
    n_both_correct = np.sum(correct_p & correct_c)
    b = np.sum(correct_p & (~correct_c)) # p right, c wrong
    c = np.sum((~correct_p) & correct_c) # p wrong, c right
    n_both_wrong = np.sum((~correct_p) & (~correct_c))
    
    # Exact binomial test for McNemar
    total_discordant = b + c
    if total_discordant > 0:
        res = binomtest(b, total_discordant, 0.5, alternative="two-sided")
        p_val = res.pvalue
    else:
        p_val = 1.0
        
    print(f"McNemar Contingency: Both Correct={n_both_correct}, b (XGB+/Ens-)={b}, c (XGB-/Ens+)={c}, Both Wrong={n_both_wrong}")
    print(f"Total discordant pairs: {total_discordant}, Exact two-sided p-value: {p_val:.4f}")
    
    # Bootstrap CIs
    ci_p = bootstrap_metrics(y_test_arr, preds_p, n_boot=2000, seed=42)
    ci_c = bootstrap_metrics(y_test_arr, preds_c, n_boot=2000, seed=42)
    
    print(f"Paper XGB: Acc={acc_p:.4f} {ci_p['Accuracy_CI'][0]}, Sens={sens_p:.4f} {ci_p['Sensitivity_CI'][0]}, Spec={spec_p:.4f} {ci_p['Specificity_CI'][0]}")
    print(f"Consensus: Acc={acc_c:.4f} {ci_c['Accuracy_CI'][0]}, Sens={sens_c:.4f} {ci_c['Sensitivity_CI'][0]}, Spec={spec_c:.4f} {ci_c['Specificity_CI'][0]}")
    
    results_summary.append({
        "split_seed": split_seed,
        "model": "Paper_XGBoost",
        "TN": cm_p[0, 0], "FP": cm_p[0, 1], "FN": cm_p[1, 0], "TP": cm_p[1, 1],
        "Accuracy": acc_p, "Accuracy_CI_low": ci_p["Accuracy_CI"][1], "Accuracy_CI_high": ci_p["Accuracy_CI"][2],
        "Sensitivity": sens_p, "Sensitivity_CI_low": ci_p["Sensitivity_CI"][1], "Sensitivity_CI_high": ci_p["Sensitivity_CI"][2],
        "Specificity": spec_p, "Specificity_CI_low": ci_p["Specificity_CI"][1], "Specificity_CI_high": ci_p["Specificity_CI"][2],
        "F1": f1_p, "F1_CI_low": ci_p["F1_CI"][1], "F1_CI_high": ci_p["F1_CI"][2],
        "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p_val
    })
    results_summary.append({
        "split_seed": split_seed,
        "model": "Consensus_Ensemble",
        "TN": cm_c[0, 0], "FP": cm_c[0, 1], "FN": cm_c[1, 0], "TP": cm_c[1, 1],
        "Accuracy": acc_c, "Accuracy_CI_low": ci_c["Accuracy_CI"][1], "Accuracy_CI_high": ci_c["Accuracy_CI"][2],
        "Sensitivity": sens_c, "Sensitivity_CI_low": ci_c["Sensitivity_CI"][1], "Sensitivity_CI_high": ci_c["Sensitivity_CI"][2],
        "Specificity": spec_c, "Specificity_CI_low": ci_c["Specificity_CI"][1], "Specificity_CI_high": ci_c["Specificity_CI"][2],
        "F1": f1_c, "F1_CI_low": ci_c["F1_CI"][1], "F1_CI_high": ci_c["F1_CI"][2],
        "mcnemar_b": b, "mcnemar_c": c, "mcnemar_p": p_val
    })

res_df = pd.DataFrame(results_summary)
res_df.to_csv("v2/results/phase1_baseline_reproduction.csv", index=False)
print("\n[OK] Phase 1 complete. Saved to v2/results/phase1_baseline_reproduction.csv")
