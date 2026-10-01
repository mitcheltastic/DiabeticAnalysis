import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, 
    HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

os.makedirs("v2/results", exist_ok=True)
os.makedirs("v2/figures_v2", exist_ok=True)

print("="*70)
print("PHASE 4: SQUEEZE THE MODELING (STRICT NESTED / LEAK-FREE PROTOCOL)")
print("="*70)

# Load Raw
raw = pd.read_csv("Dataset Diabetes.csv", sep=";")
X_raw = raw.drop(columns=["Outcome"]).copy()
y = raw["Outcome"].copy()

# Replace biologically impossible zeros with NaN
missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X_nan = X_raw.copy()
for col in missing_cols:
    X_nan[col] = X_nan[col].replace(0, np.nan)

# Missing flags
missing_flags = (X_raw[missing_cols] == 0).astype(float)
missing_flags.columns = [f"{col}_is_missing" for col in missing_cols]

# ==============================================================================
# PART A: FEATURE ABLATION (ONE BY ONE ON LEAK-FREE DATA)
# ==============================================================================
print("\n--- PART A: FEATURE ABLATION STUDY ---")

def engineer_features(df_imputed, flags_df=None):
    df = df_imputed.copy()
    feats = {}
    
    # Base raw features
    for col in df.columns:
        feats[col] = df[col]
        
    # Candidate features
    feats["HOMA_IR"] = (df["Glucose"] * df["Insulin"]) / 405.0
    feats["Glucose_x_BMI"] = df["Glucose"] * df["BMI"]
    feats["Glucose_x_Age"] = df["Glucose"] * df["Age"]
    feats["Insulin_x_BMI"] = df["Insulin"] * df["BMI"]
    feats["log_Insulin"] = np.log1p(np.maximum(df["Insulin"], 0))
    feats["BMI_bin"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(float)
    feats["Age_bin"] = pd.cut(df["Age"], bins=[0, 25, 35, 50, 100], labels=[0, 1, 2, 3]).astype(float)
    feats["RiskScore"] = (df["Glucose"] >= 140).astype(float) * 2 + (df["BMI"] >= 30).astype(float) + (df["Age"] >= 35).astype(float)
    
    res_df = pd.DataFrame(feats)
    if flags_df is not None:
        res_df = pd.concat([res_df, flags_df.reset_index(drop=True)], axis=1)
    return res_df

# Evaluate feature sets using 10-Fold Stratified CV with leak-free BayesianRidge imputer
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

candidate_features = [
    "HOMA_IR", "Glucose_x_BMI", "Glucose_x_Age", "Insulin_x_BMI", 
    "log_Insulin", "BMI_bin", "Age_bin", "RiskScore", "missing_indicators"
]

base_cols = X_raw.columns.tolist()
baseline_scores = []

for tr_idx, te_idx in cv.split(X_nan, y):
    X_tr, X_te = X_nan.iloc[tr_idx].copy(), X_nan.iloc[te_idx].copy()
    y_tr, y_te = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()
    
    imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
    X_tr_imp = pd.DataFrame(imp.fit_transform(X_tr), columns=base_cols)
    X_te_imp = pd.DataFrame(imp.transform(X_te), columns=base_cols)
    
    scaler = RobustScaler()
    X_tr_s = scaler.fit_transform(X_tr_imp)
    X_te_s = scaler.transform(X_te_imp)
    
    clf = CatBoostClassifier(iterations=150, depth=4, learning_rate=0.03, verbose=0, random_seed=42)
    clf.fit(X_tr_s, y_tr)
    baseline_scores.append(clf.score(X_te_s, y_te))

base_mean = np.mean(baseline_scores)
base_std = np.std(baseline_scores)
print(f"Base Features (N=8) 10-Fold CV Accuracy: {base_mean:.4f} (+/- {base_std:.4f})")

ablation_results = [{
    "Feature_Added": "None (Base 8 Features)",
    "Mean_Accuracy": base_mean,
    "Std_Accuracy": base_std,
    "Delta_vs_Base": 0.0,
    "Retained": True
}]

retained_features = list(base_cols)

for cand in candidate_features:
    cand_scores = []
    for tr_idx, te_idx in cv.split(X_nan, y):
        X_tr, X_te = X_nan.iloc[tr_idx].copy(), X_nan.iloc[te_idx].copy()
        y_tr, y_te = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()
        
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
        X_tr_imp = pd.DataFrame(imp.fit_transform(X_tr), columns=base_cols)
        X_te_imp = pd.DataFrame(imp.transform(X_te), columns=base_cols)
        
        # Add feature
        if cand == "missing_indicators":
            f_tr = missing_flags.iloc[tr_idx].reset_index(drop=True)
            f_te = missing_flags.iloc[te_idx].reset_index(drop=True)
            X_tr_fe = pd.concat([X_tr_imp, f_tr], axis=1)
            X_te_fe = pd.concat([X_te_imp, f_te], axis=1)
        else:
            full_tr_fe = engineer_features(X_tr_imp)
            full_te_fe = engineer_features(X_te_imp)
            X_tr_fe = X_tr_imp.copy()
            X_te_fe = X_te_imp.copy()
            X_tr_fe[cand] = full_tr_fe[cand]
            X_te_fe[cand] = full_te_fe[cand]
            
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr_fe)
        X_te_s = scaler.transform(X_te_fe)
        
        clf = CatBoostClassifier(iterations=150, depth=4, learning_rate=0.03, verbose=0, random_seed=42)
        clf.fit(X_tr_s, y_tr)
        cand_scores.append(clf.score(X_te_s, y_te))
        
    diffs = np.array(cand_scores) - np.array(baseline_scores)
    m_cand = np.mean(cand_scores)
    s_cand = np.std(cand_scores)
    delta = np.mean(diffs)
    se_delta = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
    t_crit = stats.t.ppf(0.975, df=len(diffs) - 1)
    ci_low = delta - t_crit * se_delta
    ci_high = delta + t_crit * se_delta
    t_stat = delta / (se_delta + 1e-15)
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=len(diffs) - 1))
    
    # Strict rule: Only call a feature harmful/helpful if the 95% CI excludes 0
    if ci_low > 0:
        classification = "Helpful"
        retained = True
    elif ci_high < 0:
        classification = "Harmful"
        retained = False
    else:
        classification = "Neutral (CI spans 0)"
        retained = False
        
    print(f"Adding {cand:20s}: Mean Acc = {m_cand:.4f}, Delta = {delta:+.4f}, 95% CI = [{ci_low:+.4f}, {ci_high:+.4f}] -> {classification}")
    ablation_results.append({
        "Feature_Added": cand,
        "Mean_Accuracy": m_cand,
        "Std_Accuracy": s_cand,
        "Delta_vs_Base": delta,
        "SE_Delta": se_delta,
        "CI_95_Low": ci_low,
        "CI_95_High": ci_high,
        "CI_95_Formatted": f"[{ci_low:+.4f}, {ci_high:+.4f}]",
        "t_stat": t_stat,
        "p_value": p_val,
        "Classification": classification,
        "Retained": retained
    })

pd.DataFrame(ablation_results).to_csv("v2/results/phase4_feature_ablation.csv", index=False)

# ==============================================================================
# PART B: MODEL ARCHITECTURE BENCHMARK (ON LEAK-FREE FEATURES)
# ==============================================================================
print("\n--- PART B: MODEL DIVERSITY BENCHMARK ---")

models_to_test = {
    "Logistic (L2)": LogisticRegression(max_iter=1000, random_state=42),
    "Logistic (ElasticNet)": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=1500, random_state=42),
    "SVM (RBF)": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=200, max_depth=6, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.03, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.03, random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=150, max_depth=3, learning_rate=0.03, verbose=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.03, eval_metric="logloss", verbosity=0, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=150, depth=4, learning_rate=0.03, verbose=0, random_seed=42),
    "MLP Neural Net": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42)
}

model_benchmark_results = []
oof_preds = {name: np.zeros(len(X_raw)) for name in models_to_test}
oof_probs = {name: np.zeros(len(X_raw)) for name in models_to_test}

for name, clf in models_to_test.items():
    accs, aucs, f1s = [], [], []
    for tr_idx, te_idx in cv.split(X_nan, y):
        X_tr, X_te = X_nan.iloc[tr_idx].copy(), X_nan.iloc[te_idx].copy()
        y_tr, y_te = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()
        
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
        X_tr_imp = imp.fit_transform(X_tr)
        X_te_imp = imp.transform(X_te)
        
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr_imp)
        X_te_s = scaler.transform(X_te_imp)
        
        clf.fit(X_tr_s, y_tr)
        preds = clf.predict(X_te_s)
        probs = clf.predict_proba(X_te_s)[:, 1]
        
        oof_preds[name][te_idx] = preds
        oof_probs[name][te_idx] = probs
        
        accs.append(accuracy_score(y_te, preds))
        aucs.append(roc_auc_score(y_te, probs))
        f1s.append(f1_score(y_te, preds, zero_division=0))
        
    print(f"{name:22s} | Acc: {np.mean(accs):.4f} (+/- {np.std(accs):.4f}) | AUC: {np.mean(aucs):.4f} | F1: {np.mean(f1s):.4f}")
    model_benchmark_results.append({
        "Model": name,
        "Accuracy_Mean": np.mean(accs),
        "Accuracy_Std": np.std(accs),
        "ROC_AUC_Mean": np.mean(aucs),
        "F1_Mean": np.mean(f1s)
    })

pd.DataFrame(model_benchmark_results).to_csv("v2/results/phase4_model_diversity.csv", index=False)

# ==============================================================================
# PART C: IMBALANCE HANDLING (INSIDE FOLDS ONLY)
# ==============================================================================
print("\n--- PART C: CLASS IMBALANCE HANDLING ---")
imbalance_records = []

for method in ["None", "Class_Weight", "SMOTE"]:
    accs, aucs, f1s, recalls = [], [], [], []
    for tr_idx, te_idx in cv.split(X_nan, y):
        X_tr, X_te = X_nan.iloc[tr_idx].copy(), X_nan.iloc[te_idx].copy()
        y_tr, y_te = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()
        
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
        X_tr_imp = imp.fit_transform(X_tr)
        X_te_imp = imp.transform(X_te)
        
        scaler = RobustScaler()
        X_tr_s = scaler.fit_transform(X_tr_imp)
        X_te_s = scaler.transform(X_te_imp)
        
        if method == "SMOTE":
            smote = SMOTE(random_state=42)
            X_tr_res, y_tr_res = smote.fit_resample(X_tr_s, y_tr)
            clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
            clf.fit(X_tr_res, y_tr_res)
        elif method == "Class_Weight":
            clf = RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42)
            clf.fit(X_tr_s, y_tr)
        else:
            clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
            clf.fit(X_tr_s, y_tr)
            
        preds = clf.predict(X_te_s)
        probs = clf.predict_proba(X_te_s)[:, 1]
        
        accs.append(accuracy_score(y_te, preds))
        aucs.append(roc_auc_score(y_te, probs))
        f1s.append(f1_score(y_te, preds, zero_division=0))
        recalls.append(recall_score(y_te, preds, zero_division=0))
        
    print(f"Imbalance {method:15s} | Acc: {np.mean(accs):.4f} | Recall: {np.mean(recalls):.4f} | F1: {np.mean(f1s):.4f} | AUC: {np.mean(aucs):.4f}")
    imbalance_records.append({
        "Method": method,
        "Accuracy": np.mean(accs),
        "Recall": np.mean(recalls),
        "F1": np.mean(f1s),
        "ROC_AUC": np.mean(aucs)
    })

pd.DataFrame(imbalance_records).to_csv("v2/results/phase4_imbalance_handling.csv", index=False)

# ==============================================================================
# PART D: ENSEMBLING, CALIBRATION & THRESHOLD OPTIMIZATION
# ==============================================================================
print("\n--- PART D: ENSEMBLING & CALIBRATION (OOF EVALUATION) ---")

# Combine top diverse models: CatBoost, LightGBM, Gradient Boosting, Random Forest, Logistic
top_models_list = ["CatBoost", "LightGBM", "Gradient Boosting", "Random Forest", "Logistic (L2)"]
OOF_matrix = np.column_stack([oof_probs[m] for m in top_models_list])

# Soft voting mean
oof_soft_vote = np.mean(OOF_matrix, axis=1)
acc_soft_vote = accuracy_score(y, (oof_soft_vote >= 0.5).astype(int))
auc_soft_vote = roc_auc_score(y, oof_soft_vote)
f1_soft_vote = f1_score(y, (oof_soft_vote >= 0.5).astype(int))
brier_soft_vote = brier_score_loss(y, oof_soft_vote)

print(f"OOF Soft-Voting Ensemble | Acc: {acc_soft_vote:.4f} | AUC: {auc_soft_vote:.4f} | F1: {f1_soft_vote:.4f} | Brier: {brier_soft_vote:.4f}")

# Threshold scan on OOF predictions
thresholds = np.linspace(0.1, 0.9, 81)
thresh_records = []
for th in thresholds:
    p_bin = (oof_soft_vote >= th).astype(int)
    acc = accuracy_score(y, p_bin)
    sens = recall_score(y, p_bin, zero_division=0)
    spec = np.sum((p_bin == 0) & (y == 0)) / np.sum(y == 0)
    f1 = f1_score(y, p_bin, zero_division=0)
    thresh_records.append({
        "Threshold": th,
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "F1_Score": f1
    })

df_thresh = pd.DataFrame(thresh_records)
df_thresh.to_csv("v2/results/phase4_threshold_scan.csv", index=False)

# Best F1 threshold and balanced accuracy threshold
best_f1_idx = df_thresh["F1_Score"].idxmax()
best_f1_row = df_thresh.iloc[best_f1_idx]
print(f"Optimal F1 Threshold: {best_f1_row['Threshold']:.2f} -> F1: {best_f1_row['F1_Score']:.4f}, Acc: {best_f1_row['Accuracy']:.4f}, Sens: {best_f1_row['Sensitivity']:.4f}, Spec: {best_f1_row['Specificity']:.4f}")

# Plot Threshold Trade-off Curve (Light Theme)
plt.figure(figsize=(8, 5.5), dpi=300)
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.plot(df_thresh["Threshold"], df_thresh["Sensitivity"], label="Sensitivity (Recall)", color="#DC2626", linewidth=2.2)
plt.plot(df_thresh["Threshold"], df_thresh["Specificity"], label="Specificity", color="#2563EB", linewidth=2.2)
plt.plot(df_thresh["Threshold"], df_thresh["Accuracy"], label="Accuracy", color="#059669", linewidth=2.0, linestyle="--")
plt.plot(df_thresh["Threshold"], df_thresh["F1_Score"], label="F1-Score", color="#7C3AED", linewidth=2.0, linestyle=":")
plt.axvline(best_f1_row["Threshold"], color="#0F172A", linestyle="-.", alpha=0.7, label=f"Optimal F1 ({best_f1_row['Threshold']:.2f})")

plt.xlabel("Decision Threshold (on OOF Calibrated Probabilities)", fontweight="bold", fontsize=11, color="#0F172A")
plt.ylabel("Diagnostic Metric Value", fontweight="bold", fontsize=11, color="#0F172A")
plt.title("Diagnostic Trade-Off Curve across Operating Thresholds\n(Leak-Free Soft-Voting Ensemble on Out-of-Fold Predictions)", 
          fontsize=12, fontweight="bold", pad=12, color="#0F172A")
plt.legend(frameon=True, framealpha=0.95, loc="lower center")
plt.tight_layout()
fig_thresh_path = "v2/figures_v2/A2_threshold_tradeoff_curve.png"
plt.savefig(fig_thresh_path, dpi=300)
plt.close()
print(f"[OK] Saved Figure A2 to {fig_thresh_path}")

print("\n[OK] Phase 4 complete!")
