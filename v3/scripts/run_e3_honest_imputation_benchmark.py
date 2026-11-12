import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)
from xgboost import XGBClassifier

# Import stats helpers
sys.path.append(os.path.abspath("v3/scripts"))
from stats_utils import assert_disjoint, nadeau_bengio_ttest, tost_equivalence, holm_bonferroni

print("="*75)
print("EXPERIMENT E3: HONEST BENCHMARK OF IMPUTATION STRATEGIES (50 OUTER FOLDS)")
print("="*75)

os.makedirs("v3/results", exist_ok=True)

# ------------------------------------------------------------------------------
# 1. LOAD DATA & DEFINE HONEST RLTR
# ------------------------------------------------------------------------------
raw = pd.read_csv("Dataset Diabetes.csv", sep=";")
y = raw["Outcome"].values
n_samples = len(raw)

# Replace non-physiological zeros with NaN
raw_X = raw.drop(columns=["Outcome"]).copy()
physiological_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in physiological_cols:
    raw_X.loc[raw_X[col] == 0, col] = np.nan

class HonestRLTRImputer:
    """
    Honest RLTR imputer complying strictly with Protocol Amendment 1 Section D:
    - Relative difference on available Glucose and BMI
    - Donors restricted strictly to training fold where target attribute is observed
    - No Outcome label conditioning
    - Fallback to training-fold median
    """
    def __init__(self, eps=0.15):
        self.eps = eps
        self.train_df = None
        self.medians = {}
        self.fallback_counts = {}
        self.missing_counts = {}
        
    def fit(self, X_train):
        self.train_df = X_train.copy()
        for col in self.train_df.columns:
            self.medians[col] = float(self.train_df[col].median())
        return self
        
    def transform(self, X_eval):
        X_out = X_eval.copy()
        sim_attrs_all = ["Glucose", "BMI"]
        target_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        
        fallback_counts = {c: 0 for c in target_cols}
        missing_counts = {c: 0 for c in target_cols}
        
        for col in target_cols:
            sim_attrs = [a for a in sim_attrs_all if a != col]
            missing_mask = X_out[col].isna()
            missing_idx = X_out[missing_mask].index
            missing_counts[col] = len(missing_idx)
            
            if len(missing_idx) == 0:
                continue
                
            donor_candidates = self.train_df[~self.train_df[col].isna()]
            if len(donor_candidates) == 0:
                X_out.loc[missing_idx, col] = self.medians[col]
                fallback_counts[col] += len(missing_idx)
                continue
                
            for idx in missing_idx:
                row = X_out.loc[idx]
                valid_sim = [a for a in sim_attrs if not pd.isna(row[a])]
                
                if len(valid_sim) == 0:
                    X_out.loc[idx, col] = self.medians[col]
                    fallback_counts[col] += 1
                    continue
                    
                mask = pd.Series(True, index=donor_candidates.index)
                for a in valid_sim:
                    val_i = row[a]
                    vals_j = donor_candidates[a]
                    a_valid = ~vals_j.isna()
                    diff = np.abs(val_i - vals_j) / np.maximum(np.abs(val_i), np.abs(vals_j))
                    mask = mask & a_valid & (diff <= self.eps)
                    
                donors = donor_candidates[mask]
                if len(donors) > 0:
                    X_out.loc[idx, col] = donors[col].mean()
                else:
                    X_out.loc[idx, col] = self.medians[col]
                    fallback_counts[col] += 1
                    
        self.fallback_counts = fallback_counts
        self.missing_counts = missing_counts
        return X_out

# ------------------------------------------------------------------------------
# 2. SETUP BENCHMARK CONFIGURATIONS
# ------------------------------------------------------------------------------
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
n_folds = 50

imputer_names = [
    "Median",
    "Mean",
    "KNN-5",
    "MICE-Linear",
    "MissForest",
    "Honest-RLTR-0.15",
    "Honest-RLTR-0.25",
    "Native-Missing"
]

classifier_names = ["LogReg-L2", "XGBoost"]

fold_results_file = "v3/results/E3_fold_level_results.csv"
fallback_log_file = "v3/results/rltr_fallback_log.csv"

# Remove existing files to ensure clean append-only generation
if os.path.exists(fold_results_file):
    os.remove(fold_results_file)
if os.path.exists(fallback_log_file):
    os.remove(fallback_log_file)

print(f"Beginning 50-fold cross-validation across {len(imputer_names)} imputers and {len(classifier_names)} classifiers...")
start_time = time.time()

fold_rows = []
fallback_rows = []

for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(raw_X, y)):
    t_fold_start = time.time()
    
    # Assert disjoint indices
    assert_disjoint(train_idx, test_idx)
    
    fold_seed = 42 + fold_idx
    y_tr, y_te = y[train_idx], y[test_idx]
    
    X_tr_raw = raw_X.iloc[train_idx].copy().reset_index(drop=True)
    X_te_raw = raw_X.iloc[test_idx].copy().reset_index(drop=True)
    
    # Pre-impute matrices for each strategy
    imputed_matrices = {}
    
    # 1. Median
    imp_med = SimpleImputer(strategy="median")
    imputed_matrices["Median"] = (
        pd.DataFrame(imp_med.fit_transform(X_tr_raw), columns=raw_X.columns),
        pd.DataFrame(imp_med.transform(X_te_raw), columns=raw_X.columns)
    )
    
    # 2. Mean
    imp_mean = SimpleImputer(strategy="mean")
    imputed_matrices["Mean"] = (
        pd.DataFrame(imp_mean.fit_transform(X_tr_raw), columns=raw_X.columns),
        pd.DataFrame(imp_mean.transform(X_te_raw), columns=raw_X.columns)
    )
    
    # 3. KNN-5
    imp_knn = KNNImputer(n_neighbors=5)
    imputed_matrices["KNN-5"] = (
        pd.DataFrame(imp_knn.fit_transform(X_tr_raw), columns=raw_X.columns),
        pd.DataFrame(imp_knn.transform(X_te_raw), columns=raw_X.columns)
    )
    
    # 4. MICE-Linear
    imp_mice = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=fold_seed)
    imputed_matrices["MICE-Linear"] = (
        pd.DataFrame(imp_mice.fit_transform(X_tr_raw), columns=raw_X.columns),
        pd.DataFrame(imp_mice.transform(X_te_raw), columns=raw_X.columns)
    )
    
    # 5. MissForest
    imp_mf = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=30, random_state=fold_seed), max_iter=5, random_state=fold_seed)
    imputed_matrices["MissForest"] = (
        pd.DataFrame(imp_mf.fit_transform(X_tr_raw), columns=raw_X.columns),
        pd.DataFrame(imp_mf.transform(X_te_raw), columns=raw_X.columns)
    )
    
    # 6. Honest-RLTR-0.15
    rltr_15 = HonestRLTRImputer(eps=0.15).fit(X_tr_raw)
    X_tr_r15 = rltr_15.transform(X_tr_raw)
    X_te_r15 = rltr_15.transform(X_te_raw)
    imputed_matrices["Honest-RLTR-0.15"] = (X_tr_r15, X_te_r15)
    
    for c in physiological_cols:
        mc = rltr_15.missing_counts[c]
        fc = rltr_15.fallback_counts[c]
        pct = (fc / mc * 100) if mc > 0 else 0.0
        fallback_rows.append({
            "fold": fold_idx,
            "epsilon": 0.15,
            "attribute": c,
            "missing_count": mc,
            "fallback_count": fc,
            "fallback_pct": pct
        })
        
    # 7. Honest-RLTR-0.25
    rltr_25 = HonestRLTRImputer(eps=0.25).fit(X_tr_raw)
    X_tr_r25 = rltr_25.transform(X_tr_raw)
    X_te_r25 = rltr_25.transform(X_te_raw)
    imputed_matrices["Honest-RLTR-0.25"] = (X_tr_r25, X_te_r25)
    
    for c in physiological_cols:
        mc = rltr_25.missing_counts[c]
        fc = rltr_25.fallback_counts[c]
        pct = (fc / mc * 100) if mc > 0 else 0.0
        fallback_rows.append({
            "fold": fold_idx,
            "epsilon": 0.25,
            "attribute": c,
            "missing_count": mc,
            "fallback_count": fc,
            "fallback_pct": pct
        })
        
    # 8. Native-Missing (XGBoost native, LogReg median + indicator)
    imp_nat = SimpleImputer(strategy="median", add_indicator=True)
    imputed_matrices["Native-Missing"] = (
        pd.DataFrame(imp_nat.fit_transform(X_tr_raw)),
        pd.DataFrame(imp_nat.transform(X_te_raw))
    )
    
    # Fit and evaluate models
    current_fold_records = []
    
    for imp_name in imputer_names:
        X_tr_imp, X_te_imp = imputed_matrices[imp_name]
        
        # Scaling for LogReg
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_imp)
        X_te_scaled = scaler.transform(X_te_imp)
        
        # A. Logistic Regression
        lr = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=fold_seed)
        lr.fit(X_tr_scaled, y_tr)
        pred_lr = lr.predict(X_te_scaled)
        prob_lr = lr.predict_proba(X_te_scaled)[:, 1]
        
        acc_lr = accuracy_score(y_te, pred_lr)
        auc_lr = roc_auc_score(y_te, prob_lr)
        prauc_lr = average_precision_score(y_te, prob_lr)
        brier_lr = brier_score_loss(y_te, prob_lr)
        cm_lr = confusion_matrix(y_te, pred_lr).ravel()
        tn_l, fp_l, fn_l, tp_l = cm_lr
        sens_lr = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0.0
        spec_lr = tn_l / (tn_l + fp_l) if (tn_l + fp_l) > 0 else 0.0
        f1_lr = f1_score(y_te, pred_lr, zero_division=0)
        p_w_lr, r_w_lr, f1_w_lr, _ = precision_recall_fscore_support(y_te, pred_lr, average="weighted", zero_division=0)
        
        rec_lr = {
            "fold": fold_idx,
            "imputer": imp_name,
            "classifier": "LogReg-L2",
            "accuracy": float(acc_lr),
            "roc_auc": float(auc_lr),
            "pr_auc": float(prauc_lr),
            "brier": float(brier_lr),
            "sensitivity": float(sens_lr),
            "specificity": float(spec_lr),
            "f1": float(f1_lr),
            "weighted_precision": float(p_w_lr),
            "weighted_recall": float(r_w_lr),
            "weighted_f1": float(f1_w_lr),
            "tn": int(tn_l), "fp": int(fp_l), "fn": int(fn_l), "tp": int(tp_l)
        }
        current_fold_records.append(rec_lr)
        fold_rows.append(rec_lr)
        
        # B. XGBoost
        # For Native-Missing, feed raw unscaled with NaNs directly
        if imp_name == "Native-Missing":
            X_tr_xgb = X_tr_raw.values
            X_te_xgb = X_te_raw.values
        else:
            X_tr_xgb = X_tr_imp.values
            X_te_xgb = X_te_imp.values
            
        xgb = XGBClassifier(
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1,
            random_state=fold_seed,
            eval_metric="logloss",
            n_jobs=2
        )
        xgb.fit(X_tr_xgb, y_tr)
        pred_xgb = xgb.predict(X_te_xgb)
        prob_xgb = xgb.predict_proba(X_te_xgb)[:, 1]
        
        acc_xgb = accuracy_score(y_te, pred_xgb)
        auc_xgb = roc_auc_score(y_te, prob_xgb)
        prauc_xgb = average_precision_score(y_te, prob_xgb)
        brier_xgb = brier_score_loss(y_te, prob_xgb)
        cm_xgb = confusion_matrix(y_te, pred_xgb).ravel()
        tn_x, fp_x, fn_x, tp_x = cm_xgb
        sens_xgb = tp_x / (tp_x + fn_x) if (tp_x + fn_x) > 0 else 0.0
        spec_xgb = tn_x / (tn_x + fp_x) if (tn_x + fp_x) > 0 else 0.0
        f1_xgb = f1_score(y_te, pred_xgb, zero_division=0)
        p_w_xgb, r_w_xgb, f1_w_xgb, _ = precision_recall_fscore_support(y_te, pred_xgb, average="weighted", zero_division=0)
        
        rec_xgb = {
            "fold": fold_idx,
            "imputer": imp_name,
            "classifier": "XGBoost",
            "accuracy": float(acc_xgb),
            "roc_auc": float(auc_xgb),
            "pr_auc": float(prauc_xgb),
            "brier": float(brier_xgb),
            "sensitivity": float(sens_xgb),
            "specificity": float(spec_xgb),
            "f1": float(f1_xgb),
            "weighted_precision": float(p_w_xgb),
            "weighted_recall": float(r_w_xgb),
            "weighted_f1": float(f1_w_xgb),
            "tn": int(tn_x), "fp": int(fp_x), "fn": int(fn_x), "tp": int(tp_x)
        }
        current_fold_records.append(rec_xgb)
        fold_rows.append(rec_xgb)

    # INCREMENTAL APPEND: Write fold results to disk
    df_curr_fold = pd.DataFrame(current_fold_records)
    df_curr_fold.to_csv(fold_results_file, mode="a", header=not os.path.exists(fold_results_file), index=False)
    
    # INCREMENTAL APPEND: Write fallback results to disk
    df_curr_fb = pd.DataFrame(fallback_rows[-10:])
    df_curr_fb.to_csv(fallback_log_file, mode="a", header=not os.path.exists(fallback_log_file), index=False)
    
    if (fold_idx + 1) % 10 == 0 or fold_idx == 0:
        elapsed = time.time() - start_time
        print(f"  Completed {fold_idx + 1}/50 folds ({elapsed:.1f}s elapsed, last fold: {time.time() - t_fold_start:.2f}s)", flush=True)

print(f"\nAll 50 folds completed successfully in {time.time() - start_time:.1f}s.", flush=True)

# ------------------------------------------------------------------------------
# 3. STATISTICAL ANALYSIS & SUMMARY TABLE
# ------------------------------------------------------------------------------
df_all = pd.DataFrame(fold_rows)

k = 50
n_test = 154
n_train = 614

summary_records = []

for clf_name in classifier_names:
    df_clf = df_all[df_all["classifier"] == clf_name]
    
    # Reference: Median
    med_df = df_clf[df_clf["imputer"] == "Median"].sort_values("fold")
    med_auc_vals = med_df["roc_auc"].values
    med_acc_vals = med_df["accuracy"].values
    
    # Raw p-values for Holm adjustment across the 7 non-reference imputers
    p_auc_list = []
    p_acc_list = []
    temp_rows = []
    
    for imp_name in imputer_names:
        sub_df = df_clf[df_clf["imputer"] == imp_name].sort_values("fold")
        auc_vals = sub_df["roc_auc"].values
        acc_vals = sub_df["accuracy"].values
        
        auc_mean = float(np.mean(auc_vals))
        auc_std = float(np.std(auc_vals, ddof=1))
        acc_mean = float(np.mean(acc_vals))
        acc_std = float(np.std(acc_vals, ddof=1))
        
        prauc_mean = float(sub_df["pr_auc"].mean())
        brier_mean = float(sub_df["brier"].mean())
        sens_mean = float(sub_df["sensitivity"].mean())
        spec_mean = float(sub_df["specificity"].mean())
        f1_mean = float(sub_df["f1"].mean())
        w_f1_mean = float(sub_df["weighted_f1"].mean())
        
        if imp_name == "Median":
            temp_rows.append({
                "classifier": clf_name,
                "imputer": imp_name,
                "roc_auc_mean": auc_mean, "roc_auc_std": auc_std,
                "delta_auc_vs_median": 0.0,
                "nb_p_auc": 1.0, "tost_p_auc": 0.0, "equivalent_auc": True,
                "accuracy_mean": acc_mean, "accuracy_std": acc_std,
                "delta_acc_vs_median": 0.0,
                "nb_p_acc": 1.0, "tost_p_acc": 0.0, "equivalent_acc": True,
                "pr_auc_mean": prauc_mean, "brier_mean": brier_mean,
                "sensitivity_mean": sens_mean, "specificity_mean": spec_mean,
                "f1_mean": f1_mean, "weighted_f1_mean": w_f1_mean
            })
        else:
            d_auc = auc_vals - med_auc_vals
            d_acc = acc_vals - med_acc_vals
            
            tt_auc = nadeau_bengio_ttest(d_auc, k, n_test, n_train)
            tost_auc = tost_equivalence(d_auc, k, n_test, n_train, margin=0.02)
            
            tt_acc = nadeau_bengio_ttest(d_acc, k, n_test, n_train)
            tost_acc = tost_equivalence(d_acc, k, n_test, n_train, margin=0.02)
            
            p_auc_list.append(tt_auc["p_value"])
            p_acc_list.append(tt_acc["p_value"])
            
            temp_rows.append({
                "classifier": clf_name,
                "imputer": imp_name,
                "roc_auc_mean": auc_mean, "roc_auc_std": auc_std,
                "delta_auc_vs_median": float(tt_auc["mean_diff"]),
                "nb_p_auc": float(tt_auc["p_value"]),
                "tost_p_auc": float(tost_auc["p_tost"]),
                "equivalent_auc": bool(tost_auc["equivalent"]),
                "accuracy_mean": acc_mean, "accuracy_std": acc_std,
                "delta_acc_vs_median": float(tt_acc["mean_diff"]),
                "nb_p_acc": float(tt_acc["p_value"]),
                "tost_p_acc": float(tost_acc["p_tost"]),
                "equivalent_acc": bool(tost_acc["equivalent"]),
                "pr_auc_mean": prauc_mean, "brier_mean": brier_mean,
                "sensitivity_mean": sens_mean, "specificity_mean": spec_mean,
                "f1_mean": f1_mean, "weighted_f1_mean": w_f1_mean
            })
            
    # Apply Holm correction across the 7 non-reference tests
    adj_p_auc = holm_bonferroni(p_auc_list)
    adj_p_acc = holm_bonferroni(p_acc_list)
    
    # Insert adjusted p-values
    idx_non_ref = 0
    for r in temp_rows:
        if r["imputer"] == "Median":
            r["holm_p_auc"] = 1.0
            r["holm_p_acc"] = 1.0
        else:
            r["holm_p_auc"] = float(adj_p_auc[idx_non_ref])
            r["holm_p_acc"] = float(adj_p_acc[idx_non_ref])
            idx_non_ref += 1
        summary_records.append(r)

df_summary = pd.DataFrame(summary_records)
summary_file = "v3/results/E3_honest_imputation_benchmark.csv"
df_summary.to_csv(summary_file, index=False)
print(f"Saved complete benchmark summary to: {summary_file}")

# ------------------------------------------------------------------------------
# 4. AUDIT HONEST RLTR FALLBACK RATES
# ------------------------------------------------------------------------------
df_fb = pd.DataFrame(fallback_rows)
fb_summary = df_fb.groupby(["epsilon", "attribute"])[["missing_count", "fallback_count"]].sum().reset_index()
fb_summary["fallback_pct"] = (fb_summary["fallback_count"] / fb_summary["missing_count"]) * 100
print("\n=== HONEST RLTR FALLBACK AUDIT ACROSS ALL 50 FOLDS ===")
print(fb_summary.to_string(index=False))

# Check for > 50% fallback rate flag
flagged = fb_summary[fb_summary["fallback_pct"] > 50.0]
if len(flagged) > 0:
    print("\n[WARNING]: The following attributes exceeded 50% fallback rate in Honest RLTR:")
    print(flagged.to_string(index=False))
else:
    print("\n[CONFIRMED]: Zero attributes exceeded the 50% fallback threshold in Honest RLTR.")

print("\n" + "="*75)
print("EXPERIMENT E3 COMPLETED SUCCESSFULLY")
print("="*75)

