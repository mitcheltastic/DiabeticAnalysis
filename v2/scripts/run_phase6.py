import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score, brier_score_loss
)
from scipy.stats import binomtest

os.makedirs("v2/results", exist_ok=True)

print("="*70)
print("PHASE 6: FROZEN PIPELINE FINAL EVALUATION ON UNTOUCHED HOLDOUT (EVALUATED ONCE)")
print("="*70)

# Load Raw Data
raw = pd.read_csv("Dataset Diabetes.csv", sep=";")
X_raw = raw.drop(columns=["Outcome"]).copy()
y = raw["Outcome"].copy()

missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X_nan = X_raw.copy()
for col in missing_cols:
    X_nan[col] = X_nan[col].replace(0, np.nan)

feature_names = X_raw.columns.tolist()

# 1. Untouched Holdout Split: 80:20, random_state=42, stratify=y (Identical 154-patient test set)
X_tr, X_te, y_tr, y_te = train_test_split(X_nan, y, test_size=0.20, random_state=42, stratify=y)
print(f"Train set: {len(X_tr)} (Healthy: {(y_tr==0).sum()}, Diabetic: {(y_tr==1).sum()})")
print(f"Test set:  {len(X_te)} (Healthy: {(y_te==0).sum()}, Diabetic: {(y_te==1).sum()})")

# 2. Fit Imputer Strictly on X_tr
imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
X_tr_imp = imp.fit_transform(X_tr)
X_te_imp = imp.transform(X_te)

# 3. Fit Scaler Strictly on X_tr_imp
scaler = RobustScaler()
X_tr_s = scaler.fit_transform(X_tr_imp)
X_te_s = scaler.transform(X_te_imp)

# 4. Fit Frozen Ensemble on Training Set Only
cb = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.03, verbose=0, random_seed=42)
lgb = LGBMClassifier(n_estimators=150, max_depth=3, learning_rate=0.03, verbose=-1, random_state=42)
gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.03, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

ensemble = VotingClassifier(
    estimators=[("cb", cb), ("lgb", lgb), ("gb", gb), ("rf", rf), ("lr", lr)],
    voting="soft",
    weights=[0.30, 0.25, 0.20, 0.15, 0.10]
)

print("Fitting frozen leak-free ensemble on training data...")
ensemble.fit(X_tr_s, y_tr)

# 5. Evaluate ONCE on Untouched Holdout Test Set
test_preds = ensemble.predict(X_te_s)
test_probs = ensemble.predict_proba(X_te_s)[:, 1]

cm = confusion_matrix(y_te, test_preds)
tn, fp, fn, tp = cm.ravel()

acc = accuracy_score(y_te, test_preds)
sens = recall_score(y_te, test_preds)
spec = tn / (tn + fp)
prec = precision_score(y_te, test_preds)
f1 = f1_score(y_te, test_preds)
roc_auc = roc_auc_score(y_te, test_probs)
pr_auc = average_precision_score(y_te, test_probs)
brier = brier_score_loss(y_te, test_probs)

print("\n--- FINAL TEST SET PERFORMANCE (STRICTLY LEAK-FREE) ---")
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Accuracy:    {acc:.4f} ({tn+tp}/{len(y_te)} correct)")
print(f"Sensitivity: {sens:.4f} ({tp}/{tp+fn})")
print(f"Specificity: {spec:.4f} ({tn}/{tn+fp})")
print(f"Precision:   {prec:.4f}")
print(f"F1-Score:    {f1:.4f}")
print(f"ROC-AUC:     {roc_auc:.4f}")
print(f"PR-AUC:      {pr_auc:.4f}")
print(f"Brier Score: {brier:.4f}")

# 6. Bootstrap (2000x) 95% Confidence Intervals
rng = np.random.RandomState(42)
n_boot = 2000
boot_acc, boot_sens, boot_spec, boot_prec, boot_f1, boot_auc, boot_pr, boot_brier = [], [], [], [], [], [], [], []

y_te_arr = y_te.values

for _ in range(n_boot):
    idx = rng.randint(0, len(y_te), size=len(y_te))
    y_s = y_te_arr[idx]
    pred_s = test_preds[idx]
    prob_s = test_probs[idx]
    
    cm_s = confusion_matrix(y_s, pred_s, labels=[0, 1])
    tns, fps, fns, tps = cm_s.ravel()
    
    boot_acc.append((tns + tps) / len(y_s))
    boot_sens.append(tps / (tps + fns) if (tps + fns) > 0 else 0)
    boot_spec.append(tns / (tns + fps) if (tns + fps) > 0 else 0)
    boot_prec.append(tps / (tps + fps) if (tps + fps) > 0 else 0)
    boot_f1.append(2*tps / (2*tps + fps + fns) if (2*tps + fps + fns) > 0 else 0)
    if len(np.unique(y_s)) > 1:
        boot_auc.append(roc_auc_score(y_s, prob_s))
        boot_pr.append(average_precision_score(y_s, prob_s))
    boot_brier.append(brier_score_loss(y_s, prob_s))

def get_ci(arr):
    low, high = np.percentile(arr, [2.5, 97.5])
    return f"[{low:.4f} - {high:.4f}]", low, high

print("\n--- 2000x BOOTSTRAP 95% CONFIDENCE INTERVALS ---")
print(f"Accuracy:    {acc:.4f} {get_ci(boot_acc)[0]}")
print(f"Sensitivity: {sens:.4f} {get_ci(boot_sens)[0]}")
print(f"Specificity: {spec:.4f} {get_ci(boot_spec)[0]}")
print(f"Precision:   {prec:.4f} {get_ci(boot_prec)[0]}")
print(f"F1-Score:    {f1:.4f} {get_ci(boot_f1)[0]}")
print(f"ROC-AUC:     {roc_auc:.4f} {get_ci(boot_auc)[0]}")
print(f"PR-AUC:      {pr_auc:.4f} {get_ci(boot_pr)[0]}")
print(f"Brier:       {brier:.4f} {get_ci(boot_brier)[0]}")

# 7. McNemar Test vs Upperclassmen Paper Baseline
# Upperclassmen predictions on Seed 42:
rltr_df = pd.read_csv("RLTR_Imputed.csv")
X_rltr = rltr_df.drop(columns=["Outcome"])
y_rltr = rltr_df["Outcome"]
X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(X_rltr, y_rltr, test_size=0.20, random_state=42, stratify=y_rltr)
xgb_paper = XGBClassifier(max_depth=6, n_estimators=100, learning_rate=0.1, random_state=42, eval_metric="logloss")
xgb_paper.fit(X_tr_p, y_tr_p)
paper_preds = xgb_paper.predict(X_te_p)

# McNemar contingency table
correct_paper = (paper_preds == y_te_arr)
correct_leakfree = (test_preds == y_te_arr)

b = np.sum(correct_paper & (~correct_leakfree))
c = np.sum((~correct_paper) & correct_leakfree)
total_disc = b + c
p_mcnemar = binomtest(b, total_disc, 0.5, alternative="two-sided").pvalue if total_disc > 0 else 1.0

print("\n--- McNEMAR PAIRED TEST: LEAK-FREE ENSEMBLE vs PAPER BASELINE ---")
print(f"Both Correct: {np.sum(correct_paper & correct_leakfree)}, Paper+/Free-: {b}, Paper-/Free+: {c}, Both Wrong: {np.sum((~correct_paper) & (~correct_leakfree))}")
print(f"Discordant Pairs: {total_disc}, Exact McNemar p-value: {p_mcnemar:.4f}")

# Save final benchmark results
final_results = pd.DataFrame([{
    "Pipeline": "LeakFree_MICE_VotingEnsemble",
    "Holdout_Seed": 42,
    "N_Test": len(y_te),
    "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    "Accuracy": acc, "Accuracy_CI": get_ci(boot_acc)[0],
    "Sensitivity": sens, "Sensitivity_CI": get_ci(boot_sens)[0],
    "Specificity": spec, "Specificity_CI": get_ci(boot_spec)[0],
    "Precision": prec, "Precision_CI": get_ci(boot_prec)[0],
    "F1_Score": f1, "F1_CI": get_ci(boot_f1)[0],
    "ROC_AUC": roc_auc, "ROC_AUC_CI": get_ci(boot_auc)[0],
    "PR_AUC": pr_auc, "PR_AUC_CI": get_ci(boot_pr)[0],
    "Brier_Score": brier, "Brier_CI": get_ci(boot_brier)[0],
    "McNemar_vs_Paper_b": b,
    "McNemar_vs_Paper_c": c,
    "McNemar_p_value": p_mcnemar
}])
final_results.to_csv("v2/results/phase6_final_evaluation.csv", index=False)
print("\n[OK] Phase 6 successfully completed and saved to v2/results/phase6_final_evaluation.csv!")
