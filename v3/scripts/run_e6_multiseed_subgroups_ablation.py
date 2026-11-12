import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
    precision_recall_fscore_support,
    f1_score
)

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath("v3/scripts"))
from stats_utils import assert_disjoint, nadeau_bengio_ttest

def eval_detailed(y_true, y_prob, y_pred):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = cm
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return {
        "n_samples": len(y_true),
        "accuracy": acc,
        "roc_auc": auc,
        "brier": brier,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "weighted_f1": f1_w,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }

def main():
    print("="*75)
    print("EXPERIMENT E6 (REVISED): MULTI-SEED SUBGROUP & ABLATION AUDIT (100 SEEDS)")
    print("="*75)
    
    raw = pd.read_csv("Dataset Diabetes.csv", sep=";")
    y = raw["Outcome"].values
    n_samples = len(raw)
    raw_insulin = raw["Insulin"].values
    
    # 1. RLTR pre-split scaled data (R3a specification)
    df_rltr = pd.read_csv("RLTR_Imputed.csv")
    X_rltr = df_rltr.drop(columns=["Outcome"]).copy()
    mms = MinMaxScaler()
    X_rltr_scaled = pd.DataFrame(mms.fit_transform(X_rltr), columns=X_rltr.columns)
    
    # 2. Leak-free raw data with impossible zeros as NaN
    missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    raw_nan = raw.copy()
    for col in missing_cols:
        raw_nan[col] = raw_nan[col].replace(0, np.nan)
    X_nan = raw_nan.drop(columns=["Outcome"]).copy()
    
    # E4 median hyperparameters for XGBoost
    df_hp = pd.read_csv("v3/results/E4_hyperparameter_stability.csv")
    sub_xgb = df_hp[df_hp["model"] == "XGBoost"]
    xgb_honest_params = {
        "max_depth": int(sub_xgb["max_depth"].median()),
        "n_estimators": int(sub_xgb["n_estimators"].median()),
        "learning_rate": float(sub_xgb["learning_rate"].median()),
        "subsample": float(sub_xgb["subsample"].median()),
        "colsample_bytree": float(sub_xgb["colsample_bytree"].median()),
        "eval_metric": "logloss",
        "n_jobs": 4
    }
    
    # R4-50 exact test-tuned parameters for Seed 42 from E2
    r4_50_params = {
        "n_estimators": 136,
        "max_depth": 8,
        "learning_rate": 0.18454894574190198,
        "subsample": 0.6936895426281456,
        "colsample_bytree": 0.6315948762251937,
        "eval_metric": "logloss",
        "n_jobs": 4
    }
    
    n_seeds = 100
    
    subgroup_seed_records = []
    ablation_seed_records = []
    seed42_subgroup_records = []
    seed42_ablation_records = []
    
    print(f"Executing 100 seeds for Subgroup and Ablation benchmarks...")
    for s in range(n_seeds):
        train_idx, test_idx = train_test_split(
            np.arange(n_samples),
            test_size=0.20,
            random_state=s,
            stratify=y
        )
        assert_disjoint(train_idx, test_idx)
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # Subgroup masks on test set:
        ins_te = raw_insulin[test_idx]
        mask_missing = (ins_te == 0) # missing / imputed insulin (n=84 in seed 42)
        mask_obs = (ins_te > 0)      # observed insulin (n=70 in seed 42)
        
        # Model 1: RLTR R3a Default XGBoost (as in E2 ladder)
        X_tr_rltr = X_rltr_scaled.iloc[train_idx].values
        X_te_rltr = X_rltr_scaled.iloc[test_idx].values
        xgb_rltr_def = XGBClassifier(
            max_depth=6, n_estimators=100, learning_rate=0.1,
            random_state=42, eval_metric="logloss", n_jobs=4
        )
        xgb_rltr_def.fit(X_tr_rltr, y_tr)
        p_rltr_def = xgb_rltr_def.predict_proba(X_te_rltr)[:, 1]
        y_rltr_def = xgb_rltr_def.predict(X_te_rltr)
        
        # Model 2: Leak-Free Final Model (LogReg-L2 Default, in-fold median + StandardScaler)
        X_tr_nan = X_nan.iloc[train_idx].copy()
        X_te_nan = X_nan.iloc[test_idx].copy()
        
        imp_full = SimpleImputer(strategy="median")
        X_tr_imp = imp_full.fit_transform(X_tr_nan)
        X_te_imp = imp_full.transform(X_te_nan)
        
        sc_full = StandardScaler()
        X_tr_sc = sc_full.fit_transform(X_tr_imp)
        X_te_sc = sc_full.transform(X_te_imp)
        
        lr_final = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
        lr_final.fit(X_tr_sc, y_tr)
        p_lr_final = lr_final.predict_proba(X_te_sc)[:, 1]
        y_lr_final = lr_final.predict(X_te_sc)
        
        # Model 3: Leak-Free Tuned XGBoost (E4 median params, in-fold median)
        xgb_honest = XGBClassifier(**xgb_honest_params, random_state=42)
        xgb_honest.fit(X_tr_imp, y_tr)
        p_xgb_honest = xgb_honest.predict_proba(X_te_imp)[:, 1]
        y_xgb_honest = xgb_honest.predict(X_te_imp)
        
        # Sanity check on seed 42 for R3a and test-tuned R4-50
        if s == 42:
            acc_check_r3a = accuracy_score(y_te, y_rltr_def)
            cm_check_r3a = confusion_matrix(y_te, y_rltr_def).ravel()
            assert abs(acc_check_r3a - 0.850649) < 1e-4, f"Mismatch on seed 42 RLTR default! Acc={acc_check_r3a}"
            assert tuple(cm_check_r3a) == (90, 10, 13, 41), f"Mismatch on seed 42 RLTR default CM! {cm_check_r3a}"
            
            # Model 4 (Seed 42 specific): RLTR R4-50 Test-Tuned XGBoost
            xgb_rltr_r4_50 = XGBClassifier(**r4_50_params, random_state=42)
            xgb_rltr_r4_50.fit(X_tr_rltr, y_tr)
            p_rltr_r4_50 = xgb_rltr_r4_50.predict_proba(X_te_rltr)[:, 1]
            y_rltr_r4_50 = xgb_rltr_r4_50.predict(X_te_rltr)
            
            acc_check_r4_50 = accuracy_score(y_te, y_rltr_r4_50)
            cm_check_r4_50 = confusion_matrix(y_te, y_rltr_r4_50).ravel()
            assert abs(acc_check_r4_50 - 0.870130) < 1e-4, f"Mismatch on seed 42 RLTR R4-50! Acc={acc_check_r4_50}"
            assert tuple(cm_check_r4_50) == (92, 8, 12, 42), f"Mismatch on seed 42 RLTR R4-50 CM! {cm_check_r4_50}"
            
            # Record detailed seed 42 subgroup results for all 4 models
            seed42_models = [
                ("RLTR R3a Default XGBoost", p_rltr_def, y_rltr_def),
                ("RLTR R4-50 Test-Tuned XGBoost", p_rltr_r4_50, y_rltr_r4_50),
                ("Leak-Free LogReg-L2 (Final Model)", p_lr_final, y_lr_final),
                ("Leak-Free Tuned XGBoost", p_xgb_honest, y_xgb_honest)
            ]
            for m_name, probs, preds in seed42_models:
                res_full = eval_detailed(y_te, probs, preds)
                res_mis = eval_detailed(y_te[mask_missing], probs[mask_missing], preds[mask_missing])
                res_obs = eval_detailed(y_te[mask_obs], probs[mask_obs], preds[mask_obs])
                seed42_subgroup_records.append({
                    "model": m_name, "subgroup": "Full Cohort (n=154)", **res_full
                })
                seed42_subgroup_records.append({
                    "model": m_name, "subgroup": "Missing Insulin (n=84)", **res_mis
                })
                seed42_subgroup_records.append({
                    "model": m_name, "subgroup": "Observed Insulin (n=70)", **res_obs
                })
        
        # 100-seed subgroup records
        models_subgroup = [
            ("RLTR R3a Default XGBoost", p_rltr_def, y_rltr_def),
            ("Leak-Free LogReg-L2", p_lr_final, y_lr_final),
            ("Leak-Free Tuned XGBoost", p_xgb_honest, y_xgb_honest)
        ]
        
        for m_name, probs, preds in models_subgroup:
            acc_full = accuracy_score(y_te, preds)
            auc_full = roc_auc_score(y_te, probs)
            acc_mis = accuracy_score(y_te[mask_missing], preds[mask_missing])
            auc_mis = roc_auc_score(y_te[mask_missing], probs[mask_missing])
            acc_obs = accuracy_score(y_te[mask_obs], preds[mask_obs])
            auc_obs = roc_auc_score(y_te[mask_obs], probs[mask_obs])
            
            subgroup_seed_records.append({
                "seed": s, "model": m_name,
                "acc_full": acc_full, "auc_full": auc_full,
                "acc_missing": acc_mis, "auc_missing": auc_mis,
                "acc_observed": acc_obs, "auc_observed": auc_obs,
                "diff_acc_obs_minus_mis": acc_obs - acc_mis,
                "diff_auc_obs_minus_mis": auc_obs - auc_mis
            })
            
        # ----------------------------------------------------------------------
        # ABLATION BENCHMARKS OVER 100 SEEDS (Full / No Insulin / No Insulin+Skin)
        # ----------------------------------------------------------------------
        ablation_configs = [
            ("Full (8 Features)", list(X_nan.columns)),
            ("No Insulin (7 Features)", [c for c in X_nan.columns if c != "Insulin"]),
            ("No Insulin+Skin (6 Features)", [c for c in X_nan.columns if c not in ["Insulin", "SkinThickness"]])
        ]
        
        for abl_name, cols in ablation_configs:
            X_tr_c = X_tr_nan[cols].copy()
            X_te_c = X_te_nan[cols].copy()
            
            imp_c = SimpleImputer(strategy="median")
            X_tr_c_imp = imp_c.fit_transform(X_tr_c)
            X_te_c_imp = imp_c.transform(X_te_c)
            
            sc_c = StandardScaler()
            X_tr_c_sc = sc_c.fit_transform(X_tr_c_imp)
            X_te_c_sc = sc_c.transform(X_te_c_imp)
            
            # LogReg retrained
            lr_c = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
            lr_c.fit(X_tr_c_sc, y_tr)
            p_lr_c = lr_c.predict_proba(X_te_c_sc)[:, 1]
            y_lr_c = lr_c.predict(X_te_c_sc)
            
            # XGBoost retrained
            xgb_c = XGBClassifier(**xgb_honest_params, random_state=42)
            xgb_c.fit(X_tr_c_imp, y_tr)
            p_xgb_c = xgb_c.predict_proba(X_te_c_imp)[:, 1]
            y_xgb_c = xgb_c.predict(X_te_c_imp)
            
            ablation_seed_records.append({
                "seed": s, "model": "Leak-Free LogReg-L2", "condition": abl_name,
                "n_features": len(cols),
                "accuracy": accuracy_score(y_te, y_lr_c),
                "roc_auc": roc_auc_score(y_te, p_lr_c)
            })
            ablation_seed_records.append({
                "seed": s, "model": "Leak-Free Tuned XGBoost", "condition": abl_name,
                "n_features": len(cols),
                "accuracy": accuracy_score(y_te, y_xgb_c),
                "roc_auc": roc_auc_score(y_te, p_xgb_c)
            })
            
            if s == 42:
                seed42_ablation_records.append({
                    "model": "Leak-Free LogReg-L2", "condition": abl_name, "n_features": len(cols),
                    "features_passed": str(cols),
                    "coefficients": str(dict(zip(cols, np.round(lr_c.coef_[0], 4)))),
                    "intercept": float(np.round(lr_c.intercept_[0], 4)),
                    "accuracy": accuracy_score(y_te, y_lr_c),
                    "roc_auc": roc_auc_score(y_te, p_lr_c),
                    "brier": brier_score_loss(y_te, p_lr_c),
                    "cm": str(confusion_matrix(y_te, y_lr_c).ravel().tolist())
                })
                seed42_ablation_records.append({
                    "model": "Leak-Free Tuned XGBoost", "condition": abl_name, "n_features": len(cols),
                    "features_passed": str(cols),
                    "accuracy": accuracy_score(y_te, y_xgb_c),
                    "roc_auc": roc_auc_score(y_te, p_xgb_c),
                    "brier": brier_score_loss(y_te, p_xgb_c),
                    "cm": str(confusion_matrix(y_te, y_xgb_c).ravel().tolist())
                })

    # Save per-seed records
    df_sg_seeds = pd.DataFrame(subgroup_seed_records)
    df_sg_seeds.to_csv("v3/results/E6_subgroup_seed_level.csv", index=False)
    
    df_abl_seeds = pd.DataFrame(ablation_seed_records)
    df_abl_seeds.to_csv("v3/results/E6_ablation_seed_level.csv", index=False)
    
    # --------------------------------------------------------------------------
    # COMPILE 100-SEED SUBGROUP SUMMARY TABLE
    # --------------------------------------------------------------------------
    k = 100
    n_test = 154
    n_train = 614
    
    sg_summary = []
    for m_name in ["RLTR R3a Default XGBoost", "Leak-Free LogReg-L2", "Leak-Free Tuned XGBoost"]:
        sub = df_sg_seeds[df_sg_seeds["model"] == m_name]
        
        diff_acc = sub["diff_acc_obs_minus_mis"].values
        diff_auc = sub["diff_auc_obs_minus_mis"].values
        
        tt_acc = nadeau_bengio_ttest(diff_acc, k, n_test, n_train)
        tt_auc = nadeau_bengio_ttest(diff_auc, k, n_test, n_train)
        
        sg_summary.append({
            "model": m_name,
            # Full holdout
            "acc_full_mean": sub["acc_full"].mean(),
            "acc_full_std": sub["acc_full"].std(),
            "auc_full_mean": sub["auc_full"].mean(),
            "auc_full_std": sub["auc_full"].std(),
            # Missing subgroup
            "acc_mis_mean": sub["acc_missing"].mean(),
            "acc_mis_std": sub["acc_missing"].std(),
            "acc_mis_p2_5": np.percentile(sub["acc_missing"], 2.5),
            "acc_mis_p97_5": np.percentile(sub["acc_missing"], 97.5),
            "auc_mis_mean": sub["auc_missing"].mean(),
            "auc_mis_std": sub["auc_missing"].std(),
            "auc_mis_p2_5": np.percentile(sub["auc_missing"], 2.5),
            "auc_mis_p97_5": np.percentile(sub["auc_missing"], 97.5),
            # Observed subgroup
            "acc_obs_mean": sub["acc_observed"].mean(),
            "acc_obs_std": sub["acc_observed"].std(),
            "acc_obs_p2_5": np.percentile(sub["acc_observed"], 2.5),
            "acc_obs_p97_5": np.percentile(sub["acc_observed"], 97.5),
            "auc_obs_mean": sub["auc_observed"].mean(),
            "auc_obs_std": sub["auc_observed"].std(),
            "auc_obs_p2_5": np.percentile(sub["auc_observed"], 2.5),
            "auc_obs_p97_5": np.percentile(sub["auc_observed"], 97.5),
            # Paired Difference (Observed - Missing)
            "nb_diff_acc": tt_acc["mean_diff"],
            "nb_ci_acc_low": tt_acc["ci_lower"],
            "nb_ci_acc_high": tt_acc["ci_upper"],
            "nb_p_acc": tt_acc["p_value"],
            "nb_diff_auc": tt_auc["mean_diff"],
            "nb_ci_auc_low": tt_auc["ci_lower"],
            "nb_ci_auc_high": tt_auc["ci_upper"],
            "nb_p_auc": tt_auc["p_value"]
        })
    df_sg_summary = pd.DataFrame(sg_summary)
    df_sg_summary.to_csv("v3/results/E6_multiseed_subgroup_summary.csv", index=False)
    print("Saved 100-seed subgroup summary to: v3/results/E6_multiseed_subgroup_summary.csv")
    
    # --------------------------------------------------------------------------
    # COMPILE 100-SEED ABLATION SUMMARY TABLE
    # --------------------------------------------------------------------------
    abl_summary = []
    for m_name in ["Leak-Free LogReg-L2", "Leak-Free Tuned XGBoost"]:
        sub_full = df_abl_seeds[(df_abl_seeds["model"] == m_name) & (df_abl_seeds["condition"] == "Full (8 Features)")].sort_values("seed")
        acc_full = sub_full["accuracy"].values
        auc_full = sub_full["roc_auc"].values
        
        for cond in ["Full (8 Features)", "No Insulin (7 Features)", "No Insulin+Skin (6 Features)"]:
            sub = df_abl_seeds[(df_abl_seeds["model"] == m_name) & (df_abl_seeds["condition"] == cond)].sort_values("seed")
            acc_v = sub["accuracy"].values
            auc_v = sub["roc_auc"].values
            
            # NB paired difference vs Full
            diff_acc = acc_v - acc_full
            diff_auc = auc_v - auc_full
            tt_acc = nadeau_bengio_ttest(diff_acc, k, n_test, n_train)
            tt_auc = nadeau_bengio_ttest(diff_auc, k, n_test, n_train)
            
            abl_summary.append({
                "model": m_name,
                "condition": cond,
                "n_features": sub["n_features"].iloc[0],
                "accuracy_mean": np.mean(acc_v),
                "accuracy_std": np.std(acc_v, ddof=1),
                "acc_p2_5": np.percentile(acc_v, 2.5),
                "acc_p97_5": np.percentile(acc_v, 97.5),
                "auc_mean": np.mean(auc_v),
                "auc_std": np.std(auc_v, ddof=1),
                "auc_p2_5": np.percentile(auc_v, 2.5),
                "auc_p97_5": np.percentile(auc_v, 97.5),
                "nb_diff_acc_vs_full": tt_acc["mean_diff"],
                "nb_p_acc_vs_full": tt_acc["p_value"],
                "nb_diff_auc_vs_full": tt_auc["mean_diff"],
                "nb_p_auc_vs_full": tt_auc["p_value"]
            })
    df_abl_summary = pd.DataFrame(abl_summary)
    df_abl_summary.to_csv("v3/results/E6_multiseed_ablation_summary.csv", index=False)
    print("Saved 100-seed ablation summary to: v3/results/E6_multiseed_ablation_summary.csv")
    
    # Save seed 42 detailed validation tables
    df_s42_sg = pd.DataFrame(seed42_subgroup_records)
    df_s42_sg.to_csv("v3/results/E6_seed42_exact_subgroups.csv", index=False)
    
    df_s42_abl = pd.DataFrame(seed42_ablation_records)
    df_s42_abl.to_csv("v3/results/E6_seed42_exact_ablation.csv", index=False)
    print("Saved Seed 42 exact subgroup and ablation validation tables.")
    
    print("\n" + "="*75)
    print("EXPERIMENT E6 REVISED RUN COMPLETED SUCCESSFULLY")
    print("="*75)

if __name__ == "__main__":
    main()
