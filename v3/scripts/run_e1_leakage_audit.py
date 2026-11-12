import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Ensure reproducibility
np.random.seed(42)

print("="*75)
print("EXPERIMENT E1: COMPARATIVE LEAKAGE AUDIT OF ALL FIVE IMPUTATION FILES")
print("="*75)

os.makedirs("v3/results", exist_ok=True)

# ------------------------------------------------------------------------------
# 1. LOAD DATASETS
# ------------------------------------------------------------------------------
raw = pd.read_csv("Dataset Diabetes.csv", sep=";")
imputed_files = {
    "LTR": "LTR_Imputed.csv",
    "NSSR": "NSSR_Imputed.csv",
    "RLTR": "RLTR_Imputed.csv",
    "SIM": "SIM_Imputed.csv",
    "TR": "TR_Imputed.csv"
}

m_ins = (raw["Insulin"] == 0)
m_skin = (raw["SkinThickness"] == 0)

n_obs_ins = (~m_ins).sum()
n_mis_ins = m_ins.sum()
n_obs_skin = (~m_skin).sum()
n_mis_skin = m_skin.sum()

print(f"Cohort Summary: Total N = {len(raw)}")
print(f"Insulin: Observed = {n_obs_ins}, Missing (Zeros) = {n_mis_ins}")
print(f"SkinThickness: Observed = {n_obs_skin}, Missing (Zeros) = {n_mis_skin}")

all_covariates_ins = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "BMI", "DiabetesPedigreeFunction", "Age"]
all_covariates_skin = ["Pregnancies", "Glucose", "BloodPressure", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# ------------------------------------------------------------------------------
# 2. SANITY ANCHOR CHECK (HARD STOP REQUIREMENT)
# ------------------------------------------------------------------------------
print("\n>>> Checking Sanity Anchor E1: Reproducing RLTR Phase 2 Values...")
df_rltr = pd.read_csv(imputed_files["RLTR"])
sub_rltr_ins = df_rltr[m_ins].copy()

# OLS with all predictors + Outcome
ols_rltr_all = sm.OLS(sub_rltr_ins["Insulin"], sm.add_constant(sub_rltr_ins[all_covariates_ins + ["Outcome"]])).fit()
beta_anchor = ols_rltr_all.params["Outcome"]
t_anchor = ols_rltr_all.tvalues["Outcome"]

# Partial correlation r(Insulin, Outcome | Glucose, BMI)
rx_anchor = sm.OLS(sub_rltr_ins["Insulin"], sm.add_constant(sub_rltr_ins[["Glucose", "BMI"]])).fit().resid
ry_anchor = sm.OLS(sub_rltr_ins["Outcome"], sm.add_constant(sub_rltr_ins[["Glucose", "BMI"]])).fit().resid
r_part_anchor, p_part_anchor = stats.pearsonr(rx_anchor, ry_anchor)

print(f"  RLTR Missing Insulin: Outcome beta = {beta_anchor:.4f} (Phase 2 target: ~+28.68)")
print(f"  RLTR Missing Insulin: Outcome t-stat = {t_anchor:.3f} (Phase 2 target: ~26.77)")
print(f"  RLTR Missing Insulin: Partial r = {r_part_anchor:.4f} (Phase 2 target: ~+0.82)")

# Verify anchor tolerance
assert abs(beta_anchor - 28.68) < 0.20, f"Anchor Failure: beta {beta_anchor:.4f} diverges from expected ~28.68!"
assert abs(r_part_anchor - 0.8208) < 0.01, f"Anchor Failure: partial r {r_part_anchor:.4f} diverges from expected ~0.82!"
print("  [SANITY ANCHOR CONFIRMED]: RLTR target leakage estimates reproduced to exact precision.\n")

# ------------------------------------------------------------------------------
# 3. AUDIT ALL 5 FILES FOR INSULIN & SKINTHICKNESS
# ------------------------------------------------------------------------------
audit_records = []

def run_leakage_diagnostics(df_subset, target_attr, covars_base, all_covars, dataset_name, subset_label):
    y_target = df_subset[target_attr]
    outcome = df_subset["Outcome"]
    
    # 1. Base OLS: Target ~ Glucose + BMI + Outcome
    X_base = sm.add_constant(df_subset[covars_base + ["Outcome"]])
    ols_base = sm.OLS(y_target, X_base).fit()
    beta_base = float(ols_base.params["Outcome"])
    t_base = float(ols_base.tvalues["Outcome"])
    p_base = float(ols_base.pvalues["Outcome"])
    
    # 2. Partial correlation: r(Target, Outcome | Glucose, BMI)
    X_cov_only = sm.add_constant(df_subset[covars_base])
    rx = sm.OLS(y_target, X_cov_only).fit().resid
    ry = sm.OLS(outcome, X_cov_only).fit().resid
    r_part, p_part = stats.pearsonr(rx, ry)
    
    # 3. Bivariate correlation: r(Target, Outcome)
    r_biv, p_biv = stats.pearsonr(y_target, outcome)
    
    # 4. Full OLS: Target ~ All Covariates + Outcome
    X_all = sm.add_constant(df_subset[all_covars + ["Outcome"]])
    ols_all = sm.OLS(y_target, X_all).fit()
    beta_all = float(ols_all.params["Outcome"])
    t_all = float(ols_all.tvalues["Outcome"])
    p_all = float(ols_all.pvalues["Outcome"])
    
    return {
        "dataset": dataset_name,
        "attribute": target_attr,
        "subset": subset_label,
        "n_samples": len(df_subset),
        "ols_beta_base": beta_base,
        "ols_t_base": t_base,
        "ols_p_base": p_base,
        "partial_r": float(r_part),
        "partial_p": float(p_part),
        "bivariate_r": float(r_biv),
        "bivariate_p": float(p_biv),
        "ols_beta_all": beta_all,
        "ols_t_all": t_all,
        "ols_p_all": p_all
    }

# A. Real Observed Data Baselines
df_obs_ins = raw[~m_ins].copy()
audit_records.append(run_leakage_diagnostics(
    df_obs_ins, "Insulin", ["Glucose", "BMI"], all_covariates_ins, "Raw_Observed", f"Observed_Insulin (N={n_obs_ins})"
))

df_obs_skin = raw[~m_skin].copy()
audit_records.append(run_leakage_diagnostics(
    df_obs_skin, "SkinThickness", ["Glucose", "BMI"], all_covariates_skin, "Raw_Observed", f"Observed_SkinThickness (N={n_obs_skin})"
))

# B. Imputed Files Audits
for name, filename in imputed_files.items():
    df_imp = pd.read_csv(filename)
    
    # 1. Missing Insulin rows
    sub_ins = df_imp[m_ins].copy()
    audit_records.append(run_leakage_diagnostics(
        sub_ins, "Insulin", ["Glucose", "BMI"], all_covariates_ins, name, f"Missing_Insulin (N={n_mis_ins})"
    ))
    
    # 2. Missing SkinThickness rows
    sub_skin = df_imp[m_skin].copy()
    audit_records.append(run_leakage_diagnostics(
        sub_skin, "SkinThickness", ["Glucose", "BMI"], all_covariates_skin, name, f"Missing_SkinThickness (N={n_mis_skin})"
    ))

df_audit = pd.DataFrame(audit_records)
df_audit.to_csv("v3/results/E1_leakage_all_files.csv", index=False)
print("Saved complete audit table to: v3/results/E1_leakage_all_files.csv")

# ------------------------------------------------------------------------------
# 4. ROW-LEVEL AGREEMENT OF CLASS-CONDITIONAL RLTR RECONSTRUCTION vs RLTR_IMPUTED
# ------------------------------------------------------------------------------
print("\n>>> Computing Row-Level Agreement for Class-Conditional RLTR Hypothesis Test...")

y_missing_ins = raw.loc[m_ins, "Outcome"].values
provided_rltr_ins = df_rltr.loc[m_ins, "Insulin"].values

# Donors pool from raw observed data
donors_raw = raw[~m_ins].copy()

# Setup MinMax normalized features for Euclidean distance matching
norm_df = raw.copy()
for f in ["Glucose", "BMI"]:
    norm_df[f] = (raw[f] - raw[f].min()) / (raw[f].max() - raw[f].min())
donors_norm = norm_df[~m_ins].copy()

eps_grid = [0.10, 0.15, 0.20, 0.25]
agreement_records = []

# Spec 1: Euclidean Distance on MinMax-scaled Glucose and BMI (Phase 2 formulation)
for eps in eps_grid:
    reconstructed = []
    fallback_count = 0
    for idx, row in norm_df.iterrows():
        if row["Insulin"] > 0:
            continue
        c_donors = donors_norm[donors_norm["Outcome"] == row["Outcome"]]
        diffs = c_donors[["Glucose", "BMI"]] - row[["Glucose", "BMI"]]
        dist = np.sqrt((diffs**2).sum(axis=1))
        matches = c_donors.loc[dist <= eps, "Insulin"]
        if len(matches) > 0:
            reconstructed.append(matches.mean())
        else:
            fallback_count += 1
            reconstructed.append(c_donors["Insulin"].median())
            
    rec_arr = np.array(reconstructed)
    mad = float(np.mean(np.abs(rec_arr - provided_rltr_ins)))
    rho, rho_p = stats.spearmanr(rec_arr, provided_rltr_ins)
    r_target, _ = stats.pearsonr(rec_arr, y_missing_ins)
    pct_5 = float(np.mean(np.abs(rec_arr - provided_rltr_ins) <= 5.0) * 100.0)
    pct_10 = float(np.mean(np.abs(rec_arr - provided_rltr_ins) <= 10.0) * 100.0)
    fallback_rate = float(fallback_count / len(rec_arr) * 100.0)
    
    agreement_records.append({
        "specification": "Euclidean_Normalized",
        "epsilon": eps,
        "MAD_mg_dL": mad,
        "spearman_rho": float(rho),
        "target_correlation_r": float(r_target),
        "pct_within_5mg_dL": pct_5,
        "pct_within_10mg_dL": pct_10,
        "fallback_count": fallback_count,
        "fallback_rate_pct": fallback_rate
    })

# Spec 2: Relative Difference |a - b| / max(|a|, |b|) <= eps (Protocol Amendment 1 Item D)
for eps in eps_grid:
    reconstructed = []
    fallback_count = 0
    for idx, row in raw.iterrows():
        if row["Insulin"] > 0:
            continue
        c_donors = donors_raw[donors_raw["Outcome"] == row["Outcome"]]
        g_row, b_row = row["Glucose"], row["BMI"]
        
        # Available similarity attributes
        cond_g = np.abs(c_donors["Glucose"] - g_row) / np.maximum(c_donors["Glucose"], g_row) <= eps
        cond_b = np.abs(c_donors["BMI"] - b_row) / np.maximum(c_donors["BMI"], b_row) <= eps
        matches = c_donors.loc[cond_g & cond_b, "Insulin"]
        
        if len(matches) > 0:
            reconstructed.append(matches.mean())
        else:
            fallback_count += 1
            reconstructed.append(c_donors["Insulin"].median())
            
    rec_arr = np.array(reconstructed)
    mad = float(np.mean(np.abs(rec_arr - provided_rltr_ins)))
    rho, rho_p = stats.spearmanr(rec_arr, provided_rltr_ins)
    r_target, _ = stats.pearsonr(rec_arr, y_missing_ins)
    pct_5 = float(np.mean(np.abs(rec_arr - provided_rltr_ins) <= 5.0) * 100.0)
    pct_10 = float(np.mean(np.abs(rec_arr - provided_rltr_ins) <= 10.0) * 100.0)
    fallback_rate = float(fallback_count / len(rec_arr) * 100.0)
    
    agreement_records.append({
        "specification": "Relative_Difference",
        "epsilon": eps,
        "MAD_mg_dL": mad,
        "spearman_rho": float(rho),
        "target_correlation_r": float(r_target),
        "pct_within_5mg_dL": pct_5,
        "pct_within_10mg_dL": pct_10,
        "fallback_count": fallback_count,
        "fallback_rate_pct": fallback_rate
    })

df_agreement = pd.DataFrame(agreement_records)
df_agreement.to_csv("v3/results/E1_rltr_row_agreement.csv", index=False)
print("Saved row-level agreement table to: v3/results/E1_rltr_row_agreement.csv")

print("\n" + "="*75)
print("EXPERIMENT E1 COMPLETED SUCCESSFULLY")
print("="*75)
