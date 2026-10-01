import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, ExtraTreesRegressor
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

os.makedirs("v2/results", exist_ok=True)
os.makedirs("v2/figures_v2", exist_ok=True)

print("="*70, flush=True)
print("PHASE 3: LEAK-FREE IMPUTATION BENCHMARK (OPTIMIZED 50-FOLD REPEATED CV)", flush=True)
print("="*70, flush=True)

raw = pd.read_csv("Dataset Diabetes.csv", sep=";")
X_raw = raw.drop(columns=["Outcome"]).copy()
y = raw["Outcome"].copy()

feature_names = X_raw.columns.tolist()
missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

X_nan = X_raw.copy()
for col in missing_cols:
    X_nan[col] = X_nan[col].replace(0, np.nan)

missing_indicators = (X_raw[missing_cols] == 0).astype(float)
missing_indicators.columns = [f"{col}_missing" for col in missing_cols]

class HonestRLTRImputer(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=0.15):
        self.epsilon = epsilon
        
    def fit(self, X, y=None):
        X_df = pd.DataFrame(X).copy()
        self.medians_ = X_df.median()
        self.mins_ = X_df.min()
        self.maxs_ = X_df.max()
        self.ranges_ = (self.maxs_ - self.mins_).replace(0, 1.0)
        self.donors_ = X_df.dropna().copy()
        return self
        
    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for c in X_df.columns:
            if c != 4:
                X_df[c] = X_df[c].fillna(self.medians_[c])
                
        if 4 in X_df.columns and len(self.donors_) > 0:
            ins_missing = X_df[4].isna()
            if ins_missing.any():
                donor_norm = (self.donors_[[1, 5]] - self.mins_[[1, 5]]) / self.ranges_[[1, 5]]
                target_norm = (X_df.loc[ins_missing, [1, 5]] - self.mins_[[1, 5]]) / self.ranges_[[1, 5]]
                
                for idx, t_row in target_norm.iterrows():
                    dists = np.sqrt(((donor_norm - t_row)**2).sum(axis=1))
                    matches = self.donors_.loc[dists <= self.epsilon, 4]
                    if len(matches) > 0:
                        X_df.loc[idx, 4] = matches.mean()
                    else:
                        X_df.loc[idx, 4] = self.medians_[4]
                        
        return X_df.fillna(self.medians_).values

imputers = {
    "Median": SimpleImputer(strategy="median"),
    "Mean": SimpleImputer(strategy="mean"),
    "KNN (k=5)": KNNImputer(n_neighbors=5),
    "MICE (BayesianRidge)": IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42),
    "MissForest (ExtraTrees)": IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=15, random_state=42, n_jobs=-1), max_iter=3, random_state=42),
    "Honest RLTR (eps=0.15)": HonestRLTRImputer(epsilon=0.15)
}

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42),
    "LightGBM": LGBMClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, verbose=-1, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric="logloss", verbosity=0, random_state=42),
    "CatBoost": CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05, verbose=0, random_seed=42)
}

rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
fold_indices = list(rskf.split(X_nan, y))
print(f"Total Folds: {len(fold_indices)} (10 splits x 5 repeats)", flush=True)

records = []
fold_scores_tracker = {}

t_start = time.time()

for imp_name, imputer in imputers.items():
    t_imp = time.time()
    print(f"\nProcessing Imputer: {imp_name}...", flush=True)
    
    # Precompute imputed folds for both With_Indicator=False and True
    imputed_folds = []
    for tr_idx, te_idx in fold_indices:
        X_tr = X_nan.iloc[tr_idx].copy()
        X_te = X_nan.iloc[te_idx].copy()
        y_tr = y.iloc[tr_idx].copy()
        y_te = y.iloc[te_idx].copy()
        
        imputer.fit(X_tr)
        X_tr_imp = imputer.transform(X_tr)
        X_te_imp = imputer.transform(X_te)
        
        ind_tr = missing_indicators.iloc[tr_idx].values
        ind_te = missing_indicators.iloc[te_idx].values
        
        imputed_folds.append((X_tr_imp, X_te_imp, ind_tr, ind_te, y_tr, y_te))
        
    print(f"  Folds imputed in {time.time() - t_imp:.1f}s. Evaluating classifiers...", flush=True)
    
    for with_indicator in [False, True]:
        variant_label = f"{imp_name} + Flag" if with_indicator else imp_name
        
        for clf_name, clf in classifiers.items():
            acc_list = []
            auc_list = []
            f1_list = []
            
            for X_tr_imp, X_te_imp, ind_tr, ind_te, y_tr, y_te in imputed_folds:
                if with_indicator:
                    X_tr_cur = np.hstack([X_tr_imp, ind_tr])
                    X_te_cur = np.hstack([X_te_imp, ind_te])
                else:
                    X_tr_cur = X_tr_imp
                    X_te_cur = X_te_imp
                    
                scaler = RobustScaler()
                X_tr_s = scaler.fit_transform(X_tr_cur)
                X_te_s = scaler.transform(X_te_cur)
                
                clf.fit(X_tr_s, y_tr)
                preds = clf.predict(X_te_s)
                
                acc = accuracy_score(y_te, preds)
                f1 = f1_score(y_te, preds, zero_division=0)
                
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X_te_s)[:, 1]
                    auc = roc_auc_score(y_te, probs)
                else:
                    auc = acc
                    
                acc_list.append(acc)
                auc_list.append(auc)
                f1_list.append(f1)
                
            mean_acc = np.mean(acc_list)
            std_acc = np.std(acc_list)
            mean_auc = np.mean(auc_list)
            mean_f1 = np.mean(f1_list)
            
            key = (imp_name, clf_name, with_indicator)
            fold_scores_tracker[key] = acc_list
            
            records.append({
                "Imputer": imp_name,
                "With_Missing_Indicator": with_indicator,
                "Imputer_Variant": variant_label,
                "Classifier": clf_name,
                "Accuracy_Mean": mean_acc,
                "Accuracy_Std": std_acc,
                "ROC_AUC_Mean": mean_auc,
                "F1_Mean": mean_f1
            })

df_res = pd.DataFrame(records)
df_res.to_csv("v2/results/imputer_benchmark.csv", index=False)
print(f"\n[OK] Imputer benchmark completed in {time.time() - t_start:.1f}s!", flush=True)

# Paired Wilcoxon tests vs Median Baseline on CatBoost
print("\n--- PAIRED STATISTICAL TESTS vs MEDIAN IMPUTER ---", flush=True)
stat_tests = []
median_scores = fold_scores_tracker[("Median", "CatBoost", False)]

for imp_name in imputers:
    if imp_name == "Median":
        continue
    comp_scores = fold_scores_tracker[(imp_name, "CatBoost", False)]
    stat, pval = stats.wilcoxon(comp_scores, median_scores)
    mean_diff = np.mean(comp_scores) - np.mean(median_scores)
    stat_tests.append({
        "Comparison": f"{imp_name} vs Median (CatBoost)",
        "Mean_Diff_Acc": mean_diff,
        "Wilcoxon_W": stat,
        "p_value": pval
    })

stat_df = pd.DataFrame(stat_tests)
stat_df = stat_df.sort_values("p_value").reset_index(drop=True)
stat_df["Holm_Threshold"] = 0.05 / (len(stat_df) - stat_df.index)
stat_df["Significant_After_Holm"] = stat_df["p_value"] < stat_df["Holm_Threshold"]
stat_df.to_csv("v2/results/phase3_imputer_statistical_tests.csv", index=False)
print(stat_df.to_string(), flush=True)

# Heatmap Figure A1 (300 DPI, Light Publication Theme)
pivot_acc = df_res[df_res["With_Missing_Indicator"] == False].pivot(
    index="Classifier", columns="Imputer", values="Accuracy_Mean"
)

plt.figure(figsize=(10, 6.5), dpi=300)
sns.set_theme(style="white", font_scale=1.0)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"

ax = sns.heatmap(
    pivot_acc, annot=True, fmt=".4f", cmap="Blues",
    cbar_kws={'label': 'Mean 50-Fold Repeated CV Accuracy'},
    linewidths=0.8, linecolor="#E2E8F0",
    annot_kws={"size": 11, "fontweight": "bold", "color": "#0F172A"}
)

plt.title("Leak-Free Imputation Benchmark: Classifier Accuracy across Imputers\n(50 Folds: Repeated Stratified 10-Fold CV, Fitted Inside Folds Only)", 
          fontsize=12, fontweight="bold", pad=14, color="#0F172A")
plt.xlabel("Imputation Strategy (Evaluated Without Data Leakage)", fontweight="bold", fontsize=11, color="#0F172A")
plt.ylabel("Classifier Architecture", fontweight="bold", fontsize=11, color="#0F172A")
plt.tight_layout()

fig_path = "v2/figures_v2/A1_imputer_x_model.png"
plt.savefig(fig_path, dpi=300)
plt.close()
print(f"[OK] Saved Figure A1 to {fig_path}", flush=True)
