import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
    f1_score,
    recall_score
)
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath("v3/scripts"))
from stats_utils import assert_disjoint

def main():
    print("="*75)
    print("EXPERIMENT E8: SHAP BEESWARM & TOP-K FEATURE ABLATION AUDIT")
    print("="*75)
    
    os.makedirs("v3/results", exist_ok=True)
    os.makedirs("v3/figures", exist_ok=True)
    
    # 1. Load Data & Replace Impossible Biological Zeros
    df_raw = pd.read_csv("Dataset Diabetes.csv", sep=";")
    missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for c in missing_cols:
        df_raw[c] = df_raw[c].replace(0, np.nan)
        
    X_raw = df_raw.drop(columns=["Outcome"]).copy()
    y = df_raw["Outcome"].values
    feature_names = list(X_raw.columns)
    
    # 2. Fit Imputer & Scaler on Seed 42 Train Set for Canonical SHAP Plots
    train_idx, test_idx = train_test_split(
        np.arange(len(df_raw)),
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    assert_disjoint(train_idx, test_idx)
    
    X_tr_raw = X_raw.iloc[train_idx].copy().reset_index(drop=True)
    X_te_raw = X_raw.iloc[test_idx].copy().reset_index(drop=True)
    y_tr, y_te = y[train_idx], y[test_idx]
    
    imputer_canonical = SimpleImputer(strategy="median")
    X_tr_imp = pd.DataFrame(imputer_canonical.fit_transform(X_tr_raw), columns=feature_names)
    X_te_imp = pd.DataFrame(imputer_canonical.transform(X_te_raw), columns=feature_names)
    
    scaler_canonical = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler_canonical.fit_transform(X_tr_imp), columns=feature_names)
    X_te_sc = pd.DataFrame(scaler_canonical.transform(X_te_imp), columns=feature_names)
    
    # Tuned XGBoost parameters from E4
    df_hp = pd.read_csv("v3/results/E4_hyperparameter_stability.csv")
    sub_xgb = df_hp[df_hp["model"] == "XGBoost"]
    xgb_params = {
        "max_depth": int(sub_xgb["max_depth"].median()),
        "n_estimators": int(sub_xgb["n_estimators"].median()),
        "learning_rate": float(sub_xgb["learning_rate"].median()),
        "subsample": float(sub_xgb["subsample"].median()),
        "colsample_bytree": float(sub_xgb["colsample_bytree"].median()),
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 4
    }
    
    # --------------------------------------------------------------------------
    # PART 1: COMPUTE SHAP VALUES & BEESWARM PLOTS
    # --------------------------------------------------------------------------
    print("\n--- Part 1: Computing SHAP Explanations ---")
    # A. Tuned XGBoost
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr_imp, y_tr)
    explainer_xgb = shap.TreeExplainer(xgb_model)
    shap_values_xgb = explainer_xgb(X_te_imp)
    
    # B. Logistic Regression (Final Model)
    lr_model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
    lr_model.fit(X_tr_sc, y_tr)
    explainer_lr = shap.LinearExplainer(lr_model, X_tr_sc)
    shap_values_lr = explainer_lr(X_te_sc)
    
    # Plot Beeswarm 1: Tuned XGBoost
    plt.figure(figsize=(9, 5))
    shap.plots.beeswarm(shap_values_xgb, max_display=8, show=False)
    plt.title("SHAP Beeswarm Plot — Tuned XGBoost (Leak-Free Holdout Set)", fontsize=12, pad=12)
    plt.tight_layout()
    fig_xgb_path = "v3/figures/E8_shap_beeswarm_xgb.png"
    plt.savefig(fig_xgb_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_xgb_path}")
    
    # Plot Beeswarm 2: Logistic Regression
    plt.figure(figsize=(9, 5))
    shap.plots.beeswarm(shap_values_lr, max_display=8, show=False)
    plt.title("SHAP Beeswarm Plot — Logistic Regression (Leak-Free Holdout Set)", fontsize=12, pad=12)
    plt.tight_layout()
    fig_lr_path = "v3/figures/E8_shap_beeswarm_lr.png"
    plt.savefig(fig_lr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_lr_path}")
    
    # --------------------------------------------------------------------------
    # PART 2: TOP-K CUMULATIVE FEATURE ABLATION (100 SEEDS)
    # --------------------------------------------------------------------------
    print("\n--- Part 2: Running 100-Seed Top-K Cumulative Feature Ablation ---")
    # Ranked features strictly aligned with Official Final Model (LogReg-L2) Bootstrap Median Rank from E7:
    # 1. Glucose, 2. BMI, 3. Pregnancies, 4. DiabetesPedigreeFunction, 5. Age, 6. Insulin, 7. BloodPressure, 8. SkinThickness
    ranked_features = [
        "Glucose",
        "BMI",
        "Pregnancies",
        "DiabetesPedigreeFunction",
        "Age",
        "Insulin",
        "BloodPressure",
        "SkinThickness"
    ]
    
    n_seeds = 100
    seed_records = []
    
    for s in range(n_seeds):
        tr_idx, te_idx = train_test_split(
            np.arange(len(df_raw)),
            test_size=0.20,
            random_state=s,
            stratify=y
        )
        assert_disjoint(tr_idx, te_idx)
        
        y_train_s, y_test_s = y[tr_idx], y[te_idx]
        X_train_raw_s = X_raw.iloc[tr_idx].copy().reset_index(drop=True)
        X_test_raw_s = X_raw.iloc[te_idx].copy().reset_index(drop=True)
        
        for k in range(1, 9):
            sub_feats = ranked_features[:k]
            
            X_tr_k = X_train_raw_s[sub_feats].copy()
            X_te_k = X_test_raw_s[sub_feats].copy()
            
            # In-fold median imputation
            imp_k = SimpleImputer(strategy="median")
            X_tr_k_imp = imp_k.fit_transform(X_tr_k)
            X_te_k_imp = imp_k.transform(X_te_k)
            
            # In-fold standard scaling (for LogReg)
            sc_k = StandardScaler()
            X_tr_k_sc = sc_k.fit_transform(X_tr_k_imp)
            X_te_k_sc = sc_k.transform(X_te_k_imp)
            
            # A. Logistic Regression
            lr_k = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
            lr_k.fit(X_tr_k_sc, y_train_s)
            p_lr = lr_k.predict_proba(X_te_k_sc)[:, 1]
            pred_lr = lr_k.predict(X_te_k_sc)
            
            seed_records.append({
                "seed": s,
                "model": "Logistic Regression (L2)",
                "k": k,
                "feature_added": ranked_features[k-1],
                "features_used": "+".join(sub_feats),
                "accuracy": accuracy_score(y_test_s, pred_lr),
                "roc_auc": roc_auc_score(y_test_s, p_lr),
                "sensitivity": recall_score(y_test_s, pred_lr, zero_division=0),
                "f1": f1_score(y_test_s, pred_lr, zero_division=0)
            })
            
            # B. Tuned XGBoost
            xgb_k = XGBClassifier(**xgb_params)
            xgb_k.fit(X_tr_k_imp, y_train_s)
            p_xgb = xgb_k.predict_proba(X_te_k_imp)[:, 1]
            pred_xgb = xgb_k.predict(X_te_k_imp)
            
            seed_records.append({
                "seed": s,
                "model": "Tuned XGBoost",
                "k": k,
                "feature_added": ranked_features[k-1],
                "features_used": "+".join(sub_feats),
                "accuracy": accuracy_score(y_test_s, pred_xgb),
                "roc_auc": roc_auc_score(y_test_s, p_xgb),
                "sensitivity": recall_score(y_test_s, pred_xgb, zero_division=0),
                "f1": f1_score(y_test_s, pred_xgb, zero_division=0)
            })
            
    df_seeds = pd.DataFrame(seed_records)
    csv_seeds_path = "v3/results/E8_topk_ablation_seeds.csv"
    df_seeds.to_csv(csv_seeds_path, index=False)
    print(f"Saved: {csv_seeds_path}")
    
    # --------------------------------------------------------------------------
    # PART 3: AGGREGATE SUMMARY TABLE (MEAN, STD, 95% CI)
    # --------------------------------------------------------------------------
    summary_records = []
    for (model, k), group in df_seeds.groupby(["model", "k"]):
        summary_records.append({
            "model": model,
            "k": k,
            "feature_added": group["feature_added"].iloc[0],
            "features_used": group["features_used"].iloc[0],
            "acc_mean": group["accuracy"].mean(),
            "acc_std": group["accuracy"].std(),
            "acc_ci_low": np.percentile(group["accuracy"], 2.5),
            "acc_ci_high": np.percentile(group["accuracy"], 97.5),
            "auc_mean": group["roc_auc"].mean(),
            "auc_std": group["roc_auc"].std(),
            "auc_ci_low": np.percentile(group["roc_auc"], 2.5),
            "auc_ci_high": np.percentile(group["roc_auc"], 97.5),
            "sens_mean": group["sensitivity"].mean(),
            "f1_mean": group["f1"].mean()
        })
        
    df_summary = pd.DataFrame(summary_records).sort_values(["model", "k"])
    csv_summary_path = "v3/results/E8_topk_ablation_summary.csv"
    df_summary.to_csv(csv_summary_path, index=False)
    print(f"Saved: {csv_summary_path}")
    
    # Print neat console view
    print("\n--- Summary of Top-K Ablation across 100 Seeds ---")
    view_cols = ["model", "k", "feature_added", "acc_mean", "acc_ci_low", "acc_ci_high", "auc_mean", "auc_ci_low", "auc_ci_high"]
    df_view = df_summary[view_cols].copy()
    df_view["acc_mean"] = (df_view["acc_mean"] * 100).round(2).astype(str) + "%"
    df_view["auc_mean"] = df_view["auc_mean"].round(4)
    print(df_view.to_string(index=False))
    
    # --------------------------------------------------------------------------
    # PART 4: VISUALIZATION OF TOP-K ABLATION CURVES
    # --------------------------------------------------------------------------
    print("\n--- Part 4: Generating High-Resolution Publication Figure ---")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharex=True)
    
    palette = {"Logistic Regression (L2)": "#1f77b4", "Tuned XGBoost": "#ff7f0e"}
    
    k_labels = [f"k={k}\n+{feat}" for k, feat in enumerate(ranked_features, 1)]
    k_ticks = np.arange(1, 9)
    
    for model_name, col in palette.items():
        sub = df_summary[df_summary["model"] == model_name]
        
        # Left Panel: Accuracy
        axes[0].plot(sub["k"], sub["acc_mean"] * 100, marker="o", linewidth=2.5, label=model_name, color=col)
        axes[0].fill_between(sub["k"], sub["acc_ci_low"] * 100, sub["acc_ci_high"] * 100, color=col, alpha=0.15)
        
        # Right Panel: ROC-AUC
        axes[1].plot(sub["k"], sub["auc_mean"], marker="s", linewidth=2.5, label=model_name, color=col)
        axes[1].fill_between(sub["k"], sub["auc_ci_low"], sub["auc_ci_high"], color=col, alpha=0.15)
        
    # Styling Left Panel (Accuracy)
    axes[0].set_title("Test Accuracy vs. Top-k Features (100 Seeds, 95% CI Band)", fontsize=12, pad=12)
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].set_xlabel("Number of Features Included (LogReg-L2 SHAP Order)", fontsize=11)
    axes[0].set_xticks(k_ticks)
    axes[0].set_xticklabels(k_labels, fontsize=9, rotation=20, ha='right')
    axes[0].set_ylim(68, 83)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right", fontsize=10)
    
    # Annotate Top-4 plateau
    axes[0].axvline(x=4, color="#2ca02c", linestyle="--", alpha=0.7)
    axes[0].annotate(
        "Top-4 Parsimonious Plateau\n(Acc ~76.6% - 76.8%)",
        xy=(4, 76.6), xytext=(4.2, 70.5),
        arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9, color="#1e7123", fontweight="bold"
    )
    
    # Styling Right Panel (ROC-AUC)
    axes[1].set_title("Test ROC-AUC vs. Top-k Features (100 Seeds, 95% CI Band)", fontsize=12, pad=12)
    axes[1].set_ylabel("ROC-AUC", fontsize=11)
    axes[1].set_xlabel("Number of Features Included (LogReg-L2 SHAP Order)", fontsize=11)
    axes[1].set_xticks(k_ticks)
    axes[1].set_xticklabels(k_labels, fontsize=9, rotation=20, ha='right')
    axes[1].set_ylim(0.70, 0.92)  # Expanded to prevent 95% CI upper band clipping
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower right", fontsize=10)
    
    axes[1].axvline(x=4, color="#2ca02c", linestyle="--", alpha=0.7)
    axes[1].annotate(
        "AUC Plateau (~0.838 - 0.840)\nRemaining 4 features add negligible gain",
        xy=(4, 0.839), xytext=(4.2, 0.730),
        arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
        fontsize=9, color="#1e7123", fontweight="bold"
    )
    
    plt.suptitle("Clinical Parsimony: Impact of Incremental Feature Addition on Leak-Free Performance", fontsize=13, y=0.98)
    plt.tight_layout()
    fig_ablation_path = "v3/figures/E8_topk_feature_ablation_curve.png"
    plt.savefig(fig_ablation_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_ablation_path}")
    
    print("\n" + "="*75)
    print("EXPERIMENT E8 COMPLETED SUCCESSFULLY")
    print("="*75)

if __name__ == "__main__":
    main()
