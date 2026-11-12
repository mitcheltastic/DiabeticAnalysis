import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath("v3/scripts"))
from stats_utils import nadeau_bengio_ttest

def main():
    df = pd.read_csv("v3/results/E6_subgroup_seed_level.csv")
    rltr = df[df["model"] == "RLTR R3a Default XGBoost"].sort_values("seed").reset_index(drop=True)
    lr = df[df["model"] == "Leak-Free LogReg-L2"].sort_values("seed").reset_index(drop=True)

    k = 100
    n_test = 154
    n_train = 614

    # 1. RLTR R3a XGBoost vs leak-free LogReg-L2 on OBSERVED-insulin subgroup
    diff_acc_obs = rltr["acc_observed"].values - lr["acc_observed"].values
    diff_auc_obs = rltr["auc_observed"].values - lr["auc_observed"].values

    tt_acc_obs = nadeau_bengio_ttest(diff_acc_obs, k, n_test, n_train)
    tt_auc_obs = nadeau_bengio_ttest(diff_auc_obs, k, n_test, n_train)

    # 2. Difference-in-differences:
    # (RLTR observed - RLTR missing) - (LogReg observed - LogReg missing)
    did_acc = rltr["diff_acc_obs_minus_mis"].values - lr["diff_acc_obs_minus_mis"].values
    did_auc = rltr["diff_auc_obs_minus_mis"].values - lr["diff_auc_obs_minus_mis"].values

    tt_did_acc = nadeau_bengio_ttest(did_acc, k, n_test, n_train)
    tt_did_auc = nadeau_bengio_ttest(did_auc, k, n_test, n_train)

    results = [
        {
            "comparison": "RLTR vs LogReg-L2 on Observed Subgroup",
            "metric": "Accuracy",
            "model_1_mean": rltr["acc_observed"].mean(),
            "model_1_std": rltr["acc_observed"].std(),
            "model_2_mean": lr["acc_observed"].mean(),
            "model_2_std": lr["acc_observed"].std(),
            "mean_diff": tt_acc_obs["mean_diff"],
            "nb_ci_lower": tt_acc_obs["ci_lower"],
            "nb_ci_upper": tt_acc_obs["ci_upper"],
            "nb_p_value": tt_acc_obs["p_value"]
        },
        {
            "comparison": "RLTR vs LogReg-L2 on Observed Subgroup",
            "metric": "ROC-AUC",
            "model_1_mean": rltr["auc_observed"].mean(),
            "model_1_std": rltr["auc_observed"].std(),
            "model_2_mean": lr["auc_observed"].mean(),
            "model_2_std": lr["auc_observed"].std(),
            "mean_diff": tt_auc_obs["mean_diff"],
            "nb_ci_lower": tt_auc_obs["ci_lower"],
            "nb_ci_upper": tt_auc_obs["ci_upper"],
            "nb_p_value": tt_auc_obs["p_value"]
        },
        {
            "comparison": "Difference-in-Differences [(RLTR Obs-Mis) - (LogReg Obs-Mis)]",
            "metric": "Accuracy",
            "model_1_mean": rltr["diff_acc_obs_minus_mis"].mean(),
            "model_1_std": rltr["diff_acc_obs_minus_mis"].std(),
            "model_2_mean": lr["diff_acc_obs_minus_mis"].mean(),
            "model_2_std": lr["diff_acc_obs_minus_mis"].std(),
            "mean_diff": tt_did_acc["mean_diff"],
            "nb_ci_lower": tt_did_acc["ci_lower"],
            "nb_ci_upper": tt_did_acc["ci_upper"],
            "nb_p_value": tt_did_acc["p_value"]
        },
        {
            "comparison": "Difference-in-Differences [(RLTR Obs-Mis) - (LogReg Obs-Mis)]",
            "metric": "ROC-AUC",
            "model_1_mean": rltr["diff_auc_obs_minus_mis"].mean(),
            "model_1_std": rltr["diff_auc_obs_minus_mis"].std(),
            "model_2_mean": lr["diff_auc_obs_minus_mis"].mean(),
            "model_2_std": lr["diff_auc_obs_minus_mis"].std(),
            "mean_diff": tt_did_auc["mean_diff"],
            "nb_ci_lower": tt_did_auc["ci_lower"],
            "nb_ci_upper": tt_did_auc["ci_upper"],
            "nb_p_value": tt_did_auc["p_value"]
        }
    ]

    df_out = pd.DataFrame(results)
    out_path = "v3/results/E6_headline_subgroup_tests.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Saved headline subgroup tests to: {out_path}")
    print(df_out)

if __name__ == "__main__":
    main()
