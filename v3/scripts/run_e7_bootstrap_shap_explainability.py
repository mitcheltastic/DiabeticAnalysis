import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import shap

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath("v3/scripts"))
from stats_utils import assert_disjoint

def main():
    print("="*75)
    print("EXPERIMENT E7: BOOTSTRAP SHAP EXPLAINABILITY (200 REFITS, SEED 42)")
    print("="*75)
    os.makedirs("v3/results", exist_ok=True)
    os.makedirs("v3/figures", exist_ok=True)
    
    # 1. Load data & seed 42 split
    df_raw = pd.read_csv("Dataset Diabetes.csv", sep=";")
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        df_raw[c] = df_raw[c].replace(0, np.nan)
        
    X = df_raw.drop(columns=["Outcome"]).copy()
    y = df_raw["Outcome"].values
    feature_names = list(X.columns)
    n_features = len(feature_names)
    
    train_idx, test_idx = train_test_split(
        np.arange(len(df_raw)),
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    assert_disjoint(train_idx, test_idx)
    
    X_tr_raw = X.iloc[train_idx].copy().reset_index(drop=True)
    X_te_raw = X.iloc[test_idx].copy().reset_index(drop=True)
    y_tr, y_te = y[train_idx], y[test_idx]
    
    n_train = len(y_tr)
    print(f"Training on {n_train} samples, explaining holdout of {len(y_te)} samples across 200 bootstrap refits...")
    
    # Tuned XGBoost parameters from E4
    df_hp = pd.read_csv("v3/results/E4_hyperparameter_stability.csv")
    sub_xgb = df_hp[df_hp["model"] == "XGBoost"]
    base_xgb_params = {
        "max_depth": int(sub_xgb["max_depth"].median()),
        "n_estimators": int(sub_xgb["n_estimators"].median()),
        "learning_rate": float(sub_xgb["learning_rate"].median()),
        "subsample": float(sub_xgb["subsample"].median()),
        "colsample_bytree": float(sub_xgb["colsample_bytree"].median()),
        "eval_metric": "logloss",
        "n_jobs": 2
    }
    
    n_boot = 200
    np.random.seed(42)
    
    # Matrices to store importance: shape (n_boot, n_features)
    imp_lr_boot = np.zeros((n_boot, n_features))
    imp_xgb_boot = np.zeros((n_boot, n_features))
    
    # Matrices to store ranks: shape (n_boot, n_features)
    rank_lr_boot = np.zeros((n_boot, n_features))
    rank_xgb_boot = np.zeros((n_boot, n_features))
    
    coef_lr_boot = np.zeros((n_boot, n_features))
    
    print("Executing 200 bootstrap refits...")
    for b in range(n_boot):
        # Bootstrap resample with replacement from training rows
        boot_idx = np.random.choice(n_train, size=n_train, replace=True)
        X_b_raw = X_tr_raw.iloc[boot_idx].copy().reset_index(drop=True)
        y_b = y_tr[boot_idx]
        
        # In-sample imputer + scaler
        imp_b = SimpleImputer(strategy="median")
        X_b_imp = pd.DataFrame(imp_b.fit_transform(X_b_raw), columns=feature_names)
        X_te_b_imp = pd.DataFrame(imp_b.transform(X_te_raw), columns=feature_names)
        
        sc_b = StandardScaler()
        X_b_sc = pd.DataFrame(sc_b.fit_transform(X_b_imp), columns=feature_names)
        X_te_b_sc = pd.DataFrame(sc_b.transform(X_te_b_imp), columns=feature_names)
        
        # 1. Final Model (LogReg-L2 default)
        lr = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42 + b)
        lr.fit(X_b_sc.values, y_b)
        coef_lr_boot[b, :] = lr.coef_[0]
        
        # Exact analytical SHAP for linear model on test set:
        # phi_j(x) = beta_j * (x_j - E[x_j]), where E[x_j] on training data is 0 after StandardScaler
        # Mean absolute SHAP = mean_i |beta_j * x_ij| = |beta_j| * mean_i |x_ij|
        shap_lr = lr.coef_[0] * (X_te_b_sc.values - np.mean(X_b_sc.values, axis=0))
        mean_abs_lr = np.mean(np.abs(shap_lr), axis=0)
        imp_lr_boot[b, :] = mean_abs_lr
        
        # Ranks for LogReg (rank 1 = largest importance)
        order_lr = np.argsort(-mean_abs_lr)
        ranks_lr = np.empty_like(order_lr)
        ranks_lr[order_lr] = np.arange(1, n_features + 1)
        rank_lr_boot[b, :] = ranks_lr
        
        # 2. Tuned XGBoost
        p_xgb = base_xgb_params.copy()
        p_xgb["random_state"] = 42 + b
        xgb = XGBClassifier(**p_xgb)
        xgb.fit(X_b_imp.values, y_b)
        
        explainer_xgb = shap.TreeExplainer(xgb)
        shap_xgb = explainer_xgb.shap_values(X_te_b_imp.values)
        mean_abs_xgb = np.mean(np.abs(shap_xgb), axis=0)
        imp_xgb_boot[b, :] = mean_abs_xgb
        
        order_xgb = np.argsort(-mean_abs_xgb)
        ranks_xgb = np.empty_like(order_xgb)
        ranks_xgb[order_xgb] = np.arange(1, n_features + 1)
        rank_xgb_boot[b, :] = ranks_xgb
        
        if (b + 1) % 50 == 0:
            print(f"  Completed {b + 1}/{n_boot} bootstrap refits...")
            
    # Compile summary statistics per feature
    records = []
    for j, feat in enumerate(feature_names):
        # LogReg-L2 stats
        mean_shap_lr = np.mean(imp_lr_boot[:, j])
        std_shap_lr = np.std(imp_lr_boot[:, j], ddof=1)
        ci_shap_lr_low = np.percentile(imp_lr_boot[:, j], 2.5)
        ci_shap_lr_high = np.percentile(imp_lr_boot[:, j], 97.5)
        med_rank_lr = np.median(rank_lr_boot[:, j])
        iqr_rank_lr = np.percentile(rank_lr_boot[:, j], 75) - np.percentile(rank_lr_boot[:, j], 25)
        top1_lr = np.mean(rank_lr_boot[:, j] == 1) * 100
        top3_lr = np.mean(rank_lr_boot[:, j] <= 3) * 100
        mean_coef_lr = np.mean(coef_lr_boot[:, j])
        ci_coef_lr_low = np.percentile(coef_lr_boot[:, j], 2.5)
        ci_coef_lr_high = np.percentile(coef_lr_boot[:, j], 97.5)
        
        # XGBoost stats
        mean_shap_xgb = np.mean(imp_xgb_boot[:, j])
        std_shap_xgb = np.std(imp_xgb_boot[:, j], ddof=1)
        ci_shap_xgb_low = np.percentile(imp_xgb_boot[:, j], 2.5)
        ci_shap_xgb_high = np.percentile(imp_xgb_boot[:, j], 97.5)
        med_rank_xgb = np.median(rank_xgb_boot[:, j])
        iqr_rank_xgb = np.percentile(rank_xgb_boot[:, j], 75) - np.percentile(rank_xgb_boot[:, j], 25)
        top1_xgb = np.mean(rank_xgb_boot[:, j] == 1) * 100
        top3_xgb = np.mean(rank_xgb_boot[:, j] <= 3) * 100
        
        records.append({
            "feature": feat,
            # Final Model (LogReg-L2)
            "lr_mean_abs_shap": mean_shap_lr,
            "lr_std_abs_shap": std_shap_lr,
            "lr_shap_ci_95_low": ci_shap_lr_low,
            "lr_shap_ci_95_high": ci_shap_lr_high,
            "lr_median_rank": med_rank_lr,
            "lr_iqr_rank": iqr_rank_lr,
            "lr_pct_top1": top1_lr,
            "lr_pct_top3": top3_lr,
            "lr_standardized_coef_mean": mean_coef_lr,
            "lr_coef_ci_95_low": ci_coef_lr_low,
            "lr_coef_ci_95_high": ci_coef_lr_high,
            # Tuned XGBoost
            "xgb_mean_abs_shap": mean_shap_xgb,
            "xgb_std_abs_shap": std_shap_xgb,
            "xgb_shap_ci_95_low": ci_shap_xgb_low,
            "xgb_shap_ci_95_high": ci_shap_xgb_high,
            "xgb_median_rank": med_rank_xgb,
            "xgb_iqr_rank": iqr_rank_xgb,
            "xgb_pct_top1": top1_xgb,
            "xgb_pct_top3": top3_xgb
        })
        
    df_res = pd.DataFrame(records).sort_values("lr_median_rank")
    csv_path = "v3/results/E7_bootstrap_feature_importance.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"Saved E7 Bootstrap Feature Importance to: {csv_path}")
    
    # --------------------------------------------------------------------------
    # VISUALIZATION 1: RANK DISTRIBUTION BOXPLOT (LOGREG-L2 VS XGBOOST)
    # --------------------------------------------------------------------------
    rank_df_list = []
    for b in range(n_boot):
        for j, feat in enumerate(feature_names):
            rank_df_list.append({"Feature": feat, "Rank": rank_lr_boot[b, j], "Model": "Final Model (LogReg-L2)"})
            rank_df_list.append({"Feature": feat, "Rank": rank_xgb_boot[b, j], "Model": "Tuned XGBoost"})
    df_plot_ranks = pd.DataFrame(rank_df_list)
    
    order_feats = df_res["feature"].tolist()
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_plot_ranks,
        x="Feature",
        y="Rank",
        hue="Model",
        order=order_feats,
        palette=["#1f77b4", "#ff7f0e"]
    )
    plt.gca().invert_yaxis()
    plt.xticks(rotation=25, ha='right', fontsize=10)
    plt.title("Feature Importance Rank Stability across 200 Bootstrap Refits (Rank 1 = Most Important)", fontsize=12, pad=12)
    plt.xlabel("Feature (Ordered by Final Model Median Rank)", fontsize=11)
    plt.ylabel("Assigned Rank (1 to 8)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6, axis="y")
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    fig1_path = "v3/figures/E7_feature_importance_ranks.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved Rank Boxplot to: {fig1_path}")
    
    # --------------------------------------------------------------------------
    # VISUALIZATION 2: MEAN ABSOLUTE SHAP BAR COMPARISON WITH 95% CI
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    
    # Unified ordering: order by LR mean absolute SHAP ascending (so largest is at the top)
    df_res_sorted = df_res.sort_values("lr_mean_abs_shap", ascending=True)
    y_pos = np.arange(n_features)
    
    # Panel 1: Logistic Regression
    axes[0].barh(
        y_pos,
        df_res_sorted["lr_mean_abs_shap"],
        xerr=[
            df_res_sorted["lr_mean_abs_shap"] - df_res_sorted["lr_shap_ci_95_low"],
            df_res_sorted["lr_shap_ci_95_high"] - df_res_sorted["lr_mean_abs_shap"]
        ],
        color="#1f77b4",
        alpha=0.85,
        capsize=4
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(df_res_sorted["feature"], fontsize=10)
    axes[0].set_xlabel("Mean |SHAP Value| (Impact on Log-Odds)", fontsize=11)
    axes[0].set_title("Final Model (LogReg-L2 Default)", fontsize=12)
    axes[0].grid(True, linestyle=":", alpha=0.6, axis="x")
    
    # Panel 2: Tuned XGBoost (using its own XGBoost values, explicitly aligned with df_res_sorted['feature'])
    axes[1].barh(
        y_pos,
        df_res_sorted["xgb_mean_abs_shap"],
        xerr=[
            df_res_sorted["xgb_mean_abs_shap"] - df_res_sorted["xgb_shap_ci_95_low"],
            df_res_sorted["xgb_shap_ci_95_high"] - df_res_sorted["xgb_mean_abs_shap"]
        ],
        color="#ff7f0e",
        alpha=0.85,
        capsize=4
    )
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(df_res_sorted["feature"], fontsize=10)
    axes[1].set_xlabel("Mean |SHAP Value| (Impact on Margin)", fontsize=11)
    axes[1].set_title("Tuned XGBoost (Aligned Feature Order)", fontsize=12)
    axes[1].grid(True, linestyle=":", alpha=0.6, axis="x")
    
    plt.suptitle("Feature Importance Comparison: 200 Bootstrap Refits on Seed 42 (95% CI Error Bars)", fontsize=13, y=0.98)
    plt.tight_layout()
    fig2_path = "v3/figures/E7_shap_importance_comparison.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved SHAP Bar Chart to: {fig2_path}")
    
    print("\n" + "="*75)
    print("EXPERIMENT E7 COMPLETED SUCCESSFULLY")
    print("="*75)

if __name__ == "__main__":
    main()
