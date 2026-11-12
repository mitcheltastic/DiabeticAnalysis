import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
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
from stats_utils import assert_disjoint

def eval_subgroup(y_true, y_prob, y_pred):
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
    print("EXPERIMENT E6: SUBGROUP AUDIT & FEATURE ABLATION (SEED 42)")
    print("="*75)
    os.makedirs("v3/results", exist_ok=True)
    
    # 1. LOAD RAW DATASET & SEED 42 SPLIT
    df_raw = pd.read_csv("Dataset Diabetes.csv", sep=";")
    y = df_raw["Outcome"].values
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(skf.split(df_raw, y))
    assert_disjoint(train_idx, test_idx)
    
    # Identify Insulin missingness on test set
    raw_insulin_test = df_raw.iloc[test_idx]["Insulin"].values
    test_missing_ins = (raw_insulin_test == 0)
    test_obs_ins = (raw_insulin_test > 0)
    
    print(f"Test Set (n=154): Missing Insulin n={np.sum(test_missing_ins)}, Observed Insulin n={np.sum(test_obs_ins)}")
    
    # 2. FIT LEAK-FREE FINAL MODEL (LogReg-L2 Default, in-fold median + StandardScaler)
    raw_nan = df_raw.copy()
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        raw_nan[c] = raw_nan[c].replace(0, np.nan)
    X_nan = raw_nan.drop(columns=["Outcome"]).copy()
    
    X_tr_nan, X_te_nan = X_nan.iloc[train_idx].copy().reset_index(drop=True), X_nan.iloc[test_idx].copy().reset_index(drop=True)
    y_tr, y_te = y[train_idx], y[test_idx]
    
    imp = SimpleImputer(strategy="median")
    X_tr_imp = pd.DataFrame(imp.fit_transform(X_tr_nan), columns=X_nan.columns)
    X_te_imp = pd.DataFrame(imp.transform(X_te_nan), columns=X_nan.columns)
    
    scaler = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr_imp), columns=X_nan.columns)
    X_te_sc = pd.DataFrame(scaler.transform(X_te_imp), columns=X_nan.columns)
    
    clf_final = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
    clf_final.fit(X_tr_sc.values, y_tr)
    prob_final = clf_final.predict_proba(X_te_sc.values)[:, 1]
    pred_final = clf_final.predict(X_te_sc.values)
    
    # 3. FIT LEAK-FREE TUNED XGBOOST (median best params from E4)
    df_hp = pd.read_csv("v3/results/E4_hyperparameter_stability.csv")
    sub_xgb = df_hp[df_hp["model"] == "XGBoost"]
    xgb_honest = XGBClassifier(
        max_depth=int(sub_xgb["max_depth"].median()),
        n_estimators=int(sub_xgb["n_estimators"].median()),
        learning_rate=float(sub_xgb["learning_rate"].median()),
        subsample=float(sub_xgb["subsample"].median()),
        colsample_bytree=float(sub_xgb["colsample_bytree"].median()),
        random_state=42, eval_metric="logloss", n_jobs=2
    )
    xgb_honest.fit(X_tr_imp.values, y_tr)
    prob_xgb_honest = xgb_honest.predict_proba(X_te_imp.values)[:, 1]
    pred_xgb_honest = xgb_honest.predict(X_te_imp.values)
    
    # 4. FIT UPPERCLASSMEN RLTR XGBOOST (Default and Test-Tuned)
    df_rltr = pd.read_csv("RLTR_Imputed.csv")
    X_rltr = df_rltr.drop(columns=["Outcome"]).copy()
    mms = MinMaxScaler()
    X_rltr_scaled = mms.fit_transform(X_rltr)
    X_tr_rltr, X_te_rltr = X_rltr_scaled[train_idx], X_rltr_scaled[test_idx]
    
    xgb_rltr_def = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=2)
    xgb_rltr_def.fit(X_tr_rltr, y_tr)
    prob_rltr_def = xgb_rltr_def.predict_proba(X_te_rltr)[:, 1]
    pred_rltr_def = xgb_rltr_def.predict(X_te_rltr)
    
    xgb_rltr_tuned = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", n_jobs=2
    )
    xgb_rltr_tuned.fit(X_tr_rltr, y_tr)
    prob_rltr_tun = xgb_rltr_tuned.predict_proba(X_te_rltr)[:, 1]
    pred_rltr_tun = xgb_rltr_tuned.predict(X_te_rltr)
    
    # Evaluate subgroups
    models_to_eval = [
        ("Leak-Free Final Model (LogReg-L2)", prob_final, pred_final),
        ("Leak-Free Tuned XGBoost", prob_xgb_honest, pred_xgb_honest),
        ("RLTR-file XGBoost (Default)", prob_rltr_def, pred_rltr_def),
        ("RLTR-file XGBoost (Test-Tuned)", prob_rltr_tun, pred_rltr_tun)
    ]
    
    subgroups = [
        ("Full Cohort", np.ones(len(y_te), dtype=bool)),
        ("Missing Insulin (Raw=0)", test_missing_ins),
        ("Observed Insulin (Raw>0)", test_obs_ins)
    ]
    
    subgroup_records = []
    for m_name, probs, preds in models_to_eval:
        for sg_name, mask in subgroups:
            res = eval_subgroup(y_te[mask], probs[mask], preds[mask])
            subgroup_records.append({
                "model": m_name,
                "subgroup": sg_name,
                **res
            })
            
    df_sg = pd.DataFrame(subgroup_records)
    sg_path = "v3/results/E6_subgroup_analysis.csv"
    df_sg.to_csv(sg_path, index=False)
    print(f"Saved Subgroup Analysis to: {sg_path}")
    
    # --------------------------------------------------------------------------
    # PART 2: FEATURE ABLATION / OMISSION DURING INFERENCE
    # --------------------------------------------------------------------------
    print("\nRunning Feature Ablation (Insulin and SkinThickness removal)...")
    ablation_sets = [
        ("Full 8 Features", list(X_nan.columns)),
        ("Insulin Removed (7 Features)", [c for c in X_nan.columns if c != "Insulin"]),
        ("Insulin & SkinThickness Removed (6 Features)", [c for c in X_nan.columns if c not in ["Insulin", "SkinThickness"]])
    ]
    
    ablation_records = []
    
    for abl_name, feat_cols in ablation_sets:
        # A. Honest retraining on reduced features
        X_tr_sub = X_tr_nan[feat_cols].copy()
        X_te_sub = X_te_nan[feat_cols].copy()
        
        imp_sub = SimpleImputer(strategy="median")
        X_tr_sub_imp = pd.DataFrame(imp_sub.fit_transform(X_tr_sub), columns=feat_cols)
        X_te_sub_imp = pd.DataFrame(imp_sub.transform(X_te_sub), columns=feat_cols)
        
        sc_sub = StandardScaler()
        X_tr_sub_sc = pd.DataFrame(sc_sub.fit_transform(X_tr_sub_imp), columns=feat_cols)
        X_te_sub_sc = pd.DataFrame(sc_sub.transform(X_te_sub_imp), columns=feat_cols)
        
        # 1. LogReg Retrained
        lr_abl = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
        lr_abl.fit(X_tr_sub_sc.values, y_tr)
        p_lr = lr_abl.predict_proba(X_te_sub_sc.values)[:, 1]
        y_lr = lr_abl.predict(X_te_sub_sc.values)
        res_lr = eval_subgroup(y_te, p_lr, y_lr)
        ablation_records.append({
            "model": "LogReg-L2",
            "ablation_type": "Retrained without feature(s)",
            "feature_set": abl_name,
            "n_features": len(feat_cols),
            **res_lr
        })
        
        # 2. XGBoost Retrained
        xgb_abl = XGBClassifier(
            max_depth=int(sub_xgb["max_depth"].median()),
            n_estimators=int(sub_xgb["n_estimators"].median()),
            learning_rate=float(sub_xgb["learning_rate"].median()),
            subsample=float(sub_xgb["subsample"].median()),
            colsample_bytree=float(sub_xgb["colsample_bytree"].median()),
            random_state=42, eval_metric="logloss", n_jobs=2
        )
        xgb_abl.fit(X_tr_sub_imp.values, y_tr)
        p_xgb = xgb_abl.predict_proba(X_te_sub_imp.values)[:, 1]
        y_xgb = xgb_abl.predict(X_te_sub_imp.values)
        res_xgb = eval_subgroup(y_te, p_xgb, y_xgb)
        ablation_records.append({
            "model": "Tuned XGBoost",
            "ablation_type": "Retrained without feature(s)",
            "feature_set": abl_name,
            "n_features": len(feat_cols),
            **res_xgb
        })
        
        # B. Inference-time zeroing / masking (model trained on full 8, but missing features set to training median at inference)
        if abl_name != "Full 8 Features":
            X_te_masked_imp = X_te_imp.copy()
            for col_to_mask in [c for c in X_nan.columns if c not in feat_cols]:
                X_te_masked_imp[col_to_mask] = X_tr_imp[col_to_mask].median()
                
            X_te_masked_sc = pd.DataFrame(scaler.transform(X_te_masked_imp), columns=X_nan.columns)
            
            p_lr_mask = clf_final.predict_proba(X_te_masked_sc.values)[:, 1]
            y_lr_mask = clf_final.predict(X_te_masked_sc.values)
            res_lr_mask = eval_subgroup(y_te, p_lr_mask, y_lr_mask)
            ablation_records.append({
                "model": "LogReg-L2",
                "ablation_type": "Inference-time Masked (Fixed to Median)",
                "feature_set": abl_name,
                "n_features": len(feat_cols),
                **res_lr_mask
            })
            
            p_xgb_mask = xgb_honest.predict_proba(X_te_masked_imp.values)[:, 1]
            y_xgb_mask = xgb_honest.predict(X_te_masked_imp.values)
            res_xgb_mask = eval_subgroup(y_te, p_xgb_mask, y_xgb_mask)
            ablation_records.append({
                "model": "Tuned XGBoost",
                "ablation_type": "Inference-time Masked (Fixed to Median)",
                "feature_set": abl_name,
                "n_features": len(feat_cols),
                **res_xgb_mask
            })

    df_abl = pd.DataFrame(ablation_records)
    abl_path = "v3/results/E6_feature_ablation.csv"
    df_abl.to_csv(abl_path, index=False)
    print(f"Saved Feature Ablation results to: {abl_path}")
    
    print("\n" + "="*75)
    print("EXPERIMENT E6 COMPLETED SUCCESSFULLY")
    print("="*75)

if __name__ == "__main__":
    main()
