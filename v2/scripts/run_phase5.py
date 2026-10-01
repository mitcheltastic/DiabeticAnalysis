import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.inspection import PartialDependenceDisplay
from catboost import CatBoostClassifier

os.makedirs("v2/results", exist_ok=True)
os.makedirs("v2/figures_v2", exist_ok=True)

print("="*70)
print("PHASE 5: LEAK-FREE EXPLAINABILITY (SHAP, PDP, ERROR ANALYSIS)")
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

# 1. Fit Leak-Free Model on Stratified K-Fold to gather OOF SHAP values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_shap_values = np.zeros_like(X_raw.values, dtype=float)
all_oof_preds = np.zeros(len(X_raw), dtype=int)
all_oof_probs = np.zeros(len(X_raw), dtype=float)
imputed_X_full = np.zeros_like(X_raw.values, dtype=float)

cb_model = None

for tr_idx, te_idx in cv.split(X_nan, y):
    X_tr, X_te = X_nan.iloc[tr_idx].copy(), X_nan.iloc[te_idx].copy()
    y_tr, y_te = y.iloc[tr_idx].copy(), y.iloc[te_idx].copy()
    
    imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
    X_tr_imp = imp.fit_transform(X_tr)
    X_te_imp = imp.transform(X_te)
    
    scaler = RobustScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr_imp), columns=feature_names)
    X_te_s = pd.DataFrame(scaler.transform(X_te_imp), columns=feature_names)
    
    cb = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.03, verbose=0, random_seed=42)
    cb.fit(X_tr_s, y_tr)
    cb_model = cb # save for PDP
    
    explainer = shap.TreeExplainer(cb)
    shap_vals_fold = explainer.shap_values(X_te_s)
    
    all_shap_values[te_idx] = shap_vals_fold
    all_oof_preds[te_idx] = cb.predict(X_te_s)
    all_oof_probs[te_idx] = cb.predict_proba(X_te_s)[:, 1]
    imputed_X_full[te_idx] = X_te_imp

# 2. SHAP Feature Importance Ranking (Leak-Free)
mean_abs_shap = np.abs(all_shap_values).mean(axis=0)
shap_summary_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Abs_SHAP": mean_abs_shap
}).sort_values("Mean_Abs_SHAP", ascending=False)

print("\n--- LEAK-FREE SHAP FEATURE IMPORTANCE ---")
print(shap_summary_df)
shap_summary_df.to_csv("v2/results/phase5_shap_importance.csv", index=False)

# Plot Figure A3: Leak-Free SHAP Ranking vs Leaky RLTR SHAP
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
sns.set_theme(style="whitegrid", font_scale=1.0)
palette = sns.color_palette("mako", n_colors=len(shap_summary_df))[::-1]
bars = ax.barh(shap_summary_df["Feature"][::-1], shap_summary_df["Mean_Abs_SHAP"][::-1], 
              color=palette, edgecolor="#0F172A", linewidth=0.6)

for bar, val in zip(bars, shap_summary_df["Mean_Abs_SHAP"][::-1]):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
            f"{val:.4f}", va="center", fontweight="bold", fontsize=10, color="#0F172A")

ax.set_xlabel("Mean |SHAP Value| (Impact on Model Log-Odds)", fontweight="bold", fontsize=11, color="#0F172A")
ax.set_title("Leak-Free SHAP Global Feature Importance (TreeExplainer)\n(Notice: Glucose Dominates as Expected in Real Clinical Endocrinology)", 
             fontsize=11, fontweight="bold", pad=12, color="#0F172A")
ax.set_xlim(0, max(shap_summary_df["Mean_Abs_SHAP"]) * 1.2)
plt.tight_layout()
fig_shap_path = "v2/figures_v2/A3_leak_free_shap_ranking.png"
plt.savefig(fig_shap_path, dpi=300)
plt.close()
print(f"[OK] Saved Figure A3 to {fig_shap_path}")

# 3. Figure A4: Partial Dependence Plots for Core Biomarkers
fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)
sns.set_theme(style="whitegrid", font_scale=1.0)

# Full leak-free preprocessed dataset for PDP display
imp_full = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
X_full_imp = pd.DataFrame(imp_full.fit_transform(X_nan), columns=feature_names)
scaler_full = RobustScaler()
X_full_s = pd.DataFrame(scaler_full.fit_transform(X_full_imp), columns=feature_names)
cb_full = CatBoostClassifier(iterations=200, depth=4, learning_rate=0.03, verbose=0, random_seed=42)
cb_full.fit(X_full_s, y)

pdp_features = [("Glucose", 1), ("BMI", 5), ("Age", 7), ("Insulin", 4)]
for ax, (feat_name, col_idx) in zip(axes.ravel(), pdp_features):
    PartialDependenceDisplay.from_estimator(
        cb_full, X_full_s, [feat_name], ax=ax, 
        line_kw={"color": "#2563EB", "linewidth": 2.2}
    )
    ax.set_title(f"Partial Dependence: {feat_name}", fontweight="bold", fontsize=11, color="#0F172A")
    ax.set_xlabel(f"{feat_name} (Standardized)", fontweight="bold", color="#0F172A")
    ax.set_ylabel("Partial Dependence", fontweight="bold", color="#0F172A")

plt.suptitle("Partial Dependence Profiles for Core Metabolic Biomarkers (Leak-Free Pipeline)", 
             fontsize=13, fontweight="bold", color="#0F172A", y=0.99)
plt.tight_layout()
fig_pdp_path = "v2/figures_v2/A4_partial_dependence_profiles.png"
plt.savefig(fig_pdp_path, dpi=300)
plt.close()
print(f"[OK] Saved Figure A4 to {fig_pdp_path}")

# 4. Error Analysis on Out-of-Fold Predictions
df_analysis = pd.DataFrame(imputed_X_full, columns=feature_names)
df_analysis["True_Outcome"] = y.values
df_analysis["OOF_Pred"] = all_oof_preds
df_analysis["OOF_Prob"] = all_oof_probs
df_analysis["Insulin_Was_Missing"] = (raw["Insulin"] == 0).values

df_analysis["Error_Type"] = "Correct"
df_analysis.loc[(df_analysis["True_Outcome"] == 1) & (df_analysis["OOF_Pred"] == 0), "Error_Type"] = "False_Negative"
df_analysis.loc[(df_analysis["True_Outcome"] == 0) & (df_analysis["OOF_Pred"] == 1), "Error_Type"] = "False_Positive"

err_summary = df_analysis.groupby("Error_Type").agg({
    "Glucose": ["count", "mean", "std"],
    "BMI": ["mean", "std"],
    "Age": ["mean", "std"],
    "Insulin": ["mean", "std"],
    "Insulin_Was_Missing": "mean"
})
print("\n--- ERROR ANALYSIS SUMMARY ---")
print(err_summary)
err_summary.to_csv("v2/results/phase5_error_analysis.csv")

print("\n[OK] Phase 5 complete!")
