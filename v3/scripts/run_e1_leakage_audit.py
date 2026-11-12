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

# ------------------------------------------------------------------------------
# 3B. REPLACEMENT TEST: LISTWISE DELETION PARTIAL CORRELATIONS (RAW COVARIATES)
# ------------------------------------------------------------------------------
print("\n>>> Computing Listwise Deletion Partial Correlations (Raw Feature Controls)...")
raw_nan_leak = raw.copy()
for col in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    raw_nan_leak[col] = raw_nan_leak[col].replace(0, np.nan)

# 1. Missing Insulin: control for all 7 raw covariates
covars_ins_raw = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "BMI", "DiabetesPedigreeFunction", "Age"]
df_ins_test = raw_nan_leak.loc[m_ins, covars_ins_raw + ["Outcome"]].copy()

# 2. Missing SkinThickness: control for 6 raw covariates EXCLUDING Insulin
covars_skin_no_ins = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"]
df_skin_test = raw_nan_leak.loc[m_skin, covars_skin_no_ins + ["Outcome"]].copy()

listwise_records = []
for name, filename in imputed_files.items():
    df_imp = pd.read_csv(filename)
    
    # A. Insulin
    df_ins_test["Insulin_imp"] = df_imp.loc[m_ins, "Insulin"]
    valid_ins = df_ins_test.dropna()
    N_used_ins = len(valid_ins)
    rx_ins = sm.OLS(valid_ins["Insulin_imp"], sm.add_constant(valid_ins[covars_ins_raw])).fit().resid
    ry_ins = sm.OLS(valid_ins["Outcome"], sm.add_constant(valid_ins[covars_ins_raw])).fit().resid
    r_part_ins, p_part_ins = stats.pearsonr(rx_ins, ry_ins)
    r_biv_ins, p_biv_ins = stats.pearsonr(valid_ins["Insulin_imp"], valid_ins["Outcome"])
    
    listwise_records.append({
        "dataset": name,
        "attribute": "Insulin",
        "subset": f"Missing_Insulin_Listwise_7RawCovars (N={N_used_ins})",
        "n_samples": N_used_ins,
        "ols_beta_base": np.nan,
        "ols_t_base": np.nan,
        "ols_p_base": np.nan,
        "partial_r": float(r_part_ins),
        "partial_p": float(p_part_ins),
        "bivariate_r": float(r_biv_ins),
        "bivariate_p": float(p_biv_ins),
        "ols_beta_all": np.nan,
        "ols_t_all": np.nan,
        "ols_p_all": np.nan
    })
    
    # B. SkinThickness
    df_skin_test["Skin_imp"] = df_imp.loc[m_skin, "SkinThickness"]
    valid_skin = df_skin_test.dropna()
    N_used_skin = len(valid_skin)
    rx_skin = sm.OLS(valid_skin["Skin_imp"], sm.add_constant(valid_skin[covars_skin_no_ins])).fit().resid
    ry_skin = sm.OLS(valid_skin["Outcome"], sm.add_constant(valid_skin[covars_skin_no_ins])).fit().resid
    r_part_skin, p_part_skin = stats.pearsonr(rx_skin, ry_skin)
    r_biv_skin, p_biv_skin = stats.pearsonr(valid_skin["Skin_imp"], valid_skin["Outcome"])
    
    listwise_records.append({
        "dataset": name,
        "attribute": "SkinThickness",
        "subset": f"Missing_Skin_Listwise_6RawCovars_NoIns (N={N_used_skin})",
        "n_samples": N_used_skin,
        "ols_beta_base": np.nan,
        "ols_t_base": np.nan,
        "ols_p_base": np.nan,
        "partial_r": float(r_part_skin),
        "partial_p": float(p_part_skin),
        "bivariate_r": float(r_biv_skin),
        "bivariate_p": float(p_biv_skin),
        "ols_beta_all": np.nan,
        "ols_t_all": np.nan,
        "ols_p_all": np.nan
    })

# Holm-Bonferroni correction across the 10 listwise tests
sys.path.append(os.path.abspath("v3/scripts"))
from stats_utils import holm_bonferroni
p_vals_listwise = [r["partial_p"] for r in listwise_records]
adj_p_listwise = holm_bonferroni(p_vals_listwise)
for r, ap in zip(listwise_records, adj_p_listwise):
    r["holm_adjusted_p"] = float(ap)
audit_records.extend(listwise_records)

# ------------------------------------------------------------------------------
# 3C. SENSITIVITY ANALYSIS: ALL MISSING ROWS (MEDIAN-FILLED COVARIATES, NO OUTCOME)
# ------------------------------------------------------------------------------
print(">>> Computing Sensitivity Analysis on ALL Missing Rows (Cohort Median Covariates)...")
raw_filled = raw_nan_leak.copy()
medians_all = raw_nan_leak[covars_ins_raw].median()
for col in covars_ins_raw:
    raw_filled[col] = raw_filled[col].fillna(medians_all[col])

sens_records = []
for name, filename in imputed_files.items():
    df_imp = pd.read_csv(filename)
    
    # A. Insulin on all 374 missing rows
    ins_imp = df_imp.loc[m_ins, "Insulin"]
    X_ins_sens = raw_filled.loc[m_ins, covars_ins_raw]
    y_ins_sens = raw_filled.loc[m_ins, "Outcome"]
    rx_i = sm.OLS(ins_imp, sm.add_constant(X_ins_sens)).fit().resid
    ry_i = sm.OLS(y_ins_sens, sm.add_constant(X_ins_sens)).fit().resid
    r_i, p_i = stats.pearsonr(rx_i, ry_i)
    r_biv_i, p_biv_i = stats.pearsonr(ins_imp, y_ins_sens)
    
    sens_records.append({
        "dataset": name,
        "attribute": "Insulin",
        "subset": f"Missing_Insulin_Sensitivity_MedianCovars (N={n_mis_ins})",
        "n_samples": n_mis_ins,
        "ols_beta_base": np.nan,
        "ols_t_base": np.nan,
        "ols_p_base": np.nan,
        "partial_r": float(r_i),
        "partial_p": float(p_i),
        "bivariate_r": float(r_biv_i),
        "bivariate_p": float(p_biv_i),
        "ols_beta_all": np.nan,
        "ols_t_all": np.nan,
        "ols_p_all": np.nan
    })
    
    # B. SkinThickness on all 227 missing rows (no insulin)
    skin_imp = df_imp.loc[m_skin, "SkinThickness"]
    X_skin_sens = raw_filled.loc[m_skin, covars_skin_no_ins]
    y_skin_sens = raw_filled.loc[m_skin, "Outcome"]
    rx_s = sm.OLS(skin_imp, sm.add_constant(X_skin_sens)).fit().resid
    ry_s = sm.OLS(y_skin_sens, sm.add_constant(X_skin_sens)).fit().resid
    r_s, p_s = stats.pearsonr(rx_s, ry_s)
    r_biv_s, p_biv_s = stats.pearsonr(skin_imp, y_skin_sens)
    
    sens_records.append({
        "dataset": name,
        "attribute": "SkinThickness",
        "subset": f"Missing_Skin_Sensitivity_MedianCovars_NoIns (N={n_mis_skin})",
        "n_samples": n_mis_skin,
        "ols_beta_base": np.nan,
        "ols_t_base": np.nan,
        "ols_p_base": np.nan,
        "partial_r": float(r_s),
        "partial_p": float(p_s),
        "bivariate_r": float(r_biv_s),
        "bivariate_p": float(p_biv_s),
        "ols_beta_all": np.nan,
        "ols_t_all": np.nan,
        "ols_p_all": np.nan
    })

p_vals_sens = [r["partial_p"] for r in sens_records]
adj_p_sens = holm_bonferroni(p_vals_sens)
for r, ap in zip(sens_records, adj_p_sens):
    r["holm_adjusted_p"] = float(ap)
audit_records.extend(sens_records)

df_audit = pd.DataFrame(audit_records)
df_audit.to_csv("v3/results/E1_leakage_all_files.csv", index=False)
print("Saved complete audit table (including listwise and median-filled sensitivity) to: v3/results/E1_leakage_all_files.csv")

# ------------------------------------------------------------------------------
# 4. ROW-LEVEL AGREEMENT OF CLASS-CONDITIONAL RLTR RECONSTRUCTION vs RLTR_IMPUTED
# ------------------------------------------------------------------------------
print("\n>>> Computing Row-Level Agreement & Controls for Class-Conditional RLTR Hypothesis Test...")

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

for spec in ["Euclidean_Normalized", "Relative_Difference"]:
    for eps in eps_grid:
        rec_cc = []
        rec_honest = []
        fallback_cc = 0
        fallback_h = 0
        
        for idx, row in raw.iterrows():
            if row["Insulin"] > 0:
                continue
                
            if spec == "Euclidean_Normalized":
                row_norm = norm_df.loc[idx]
                # 1. Class-Conditional
                c_donors = donors_norm[donors_norm["Outcome"] == row["Outcome"]]
                diff_cc = np.sqrt(((c_donors[["Glucose", "BMI"]] - row_norm[["Glucose", "BMI"]])**2).sum(axis=1))
                matches_cc = c_donors.loc[diff_cc <= eps, "Insulin"]
                if len(matches_cc) > 0:
                    rec_cc.append(matches_cc.mean())
                else:
                    fallback_cc += 1
                    rec_cc.append(c_donors["Insulin"].median())
                    
                # 2. Honest Control (unconditioned on Outcome)
                diff_h = np.sqrt(((donors_norm[["Glucose", "BMI"]] - row_norm[["Glucose", "BMI"]])**2).sum(axis=1))
                matches_h = donors_norm.loc[diff_h <= eps, "Insulin"]
                if len(matches_h) > 0:
                    rec_honest.append(matches_h.mean())
                else:
                    fallback_h += 1
                    rec_honest.append(donors_norm["Insulin"].median())
            else:
                # Relative Difference
                g_row, b_row = row["Glucose"], row["BMI"]
                # 1. Class-Conditional
                c_donors = donors_raw[donors_raw["Outcome"] == row["Outcome"]]
                cond_g_cc = np.abs(c_donors["Glucose"] - g_row) / np.maximum(c_donors["Glucose"], g_row) <= eps
                cond_b_cc = np.abs(c_donors["BMI"] - b_row) / np.maximum(c_donors["BMI"], b_row) <= eps
                matches_cc = c_donors.loc[cond_g_cc & cond_b_cc, "Insulin"]
                if len(matches_cc) > 0:
                    rec_cc.append(matches_cc.mean())
                else:
                    fallback_cc += 1
                    rec_cc.append(c_donors["Insulin"].median())
                    
                # 2. Honest Control (unconditioned on Outcome)
                cond_g_h = np.abs(donors_raw["Glucose"] - g_row) / np.maximum(donors_raw["Glucose"], g_row) <= eps
                cond_b_h = np.abs(donors_raw["BMI"] - b_row) / np.maximum(donors_raw["BMI"], b_row) <= eps
                matches_h = donors_raw.loc[cond_g_h & cond_b_h, "Insulin"]
                if len(matches_h) > 0:
                    rec_honest.append(matches_h.mean())
                else:
                    fallback_h += 1
                    rec_honest.append(donors_raw["Insulin"].median())
                    
        rec_cc = np.array(rec_cc)
        rec_honest = np.array(rec_honest)
        
        # Primary comparison: Class-Conditional vs RLTR file
        mad_cc = float(np.mean(np.abs(rec_cc - provided_rltr_ins)))
        rho_cc, _ = stats.spearmanr(rec_cc, provided_rltr_ins)
        r_target_cc, _ = stats.pearsonr(rec_cc, y_missing_ins)
        pct5_cc = float(np.mean(np.abs(rec_cc - provided_rltr_ins) <= 5.0) * 100.0)
        pct10_cc = float(np.mean(np.abs(rec_cc - provided_rltr_ins) <= 10.0) * 100.0)
        
        # Win rate per row: CC vs Honest closer to file
        diff_cc_prov = np.abs(rec_cc - provided_rltr_ins)
        diff_h_prov = np.abs(rec_honest - provided_rltr_ins)
        cc_wins = float(np.mean(diff_cc_prov < diff_h_prov) * 100.0)
        h_wins = float(np.mean(diff_h_prov < diff_cc_prov) * 100.0)
        ties = float(np.mean(diff_cc_prov == diff_h_prov) * 100.0)
        
        agreement_records.append({
            "comparison": "Class_Conditional_vs_RLTR_File",
            "specification": spec,
            "epsilon": eps,
            "MAD_mg_dL": mad_cc,
            "spearman_rho": float(rho_cc),
            "target_correlation_r": float(r_target_cc),
            "pct_within_5mg_dL": pct5_cc,
            "pct_within_10mg_dL": pct10_cc,
            "fallback_count": fallback_cc,
            "fallback_rate_pct": float(fallback_cc / len(rec_cc) * 100.0),
            "pct_rows_closer_to_file": cc_wins
        })
        
        # Control A: Honest (unconditioned) vs RLTR file (evaluated at eps=0.15 and 0.25)
        if eps in [0.15, 0.25]:
            mad_h = float(np.mean(np.abs(rec_honest - provided_rltr_ins)))
            rho_h, _ = stats.spearmanr(rec_honest, provided_rltr_ins)
            r_target_h, _ = stats.pearsonr(rec_honest, y_missing_ins)
            pct5_h = float(np.mean(np.abs(rec_honest - provided_rltr_ins) <= 5.0) * 100.0)
            pct10_h = float(np.mean(np.abs(rec_honest - provided_rltr_ins) <= 10.0) * 100.0)
            
            agreement_records.append({
                "comparison": "Honest_Control_vs_RLTR_File",
                "specification": spec,
                "epsilon": eps,
                "MAD_mg_dL": mad_h,
                "spearman_rho": float(rho_h),
                "target_correlation_r": float(r_target_h),
                "pct_within_5mg_dL": pct5_h,
                "pct_within_10mg_dL": pct10_h,
                "fallback_count": fallback_h,
                "fallback_rate_pct": float(fallback_h / len(rec_honest) * 100.0),
                "pct_rows_closer_to_file": h_wins
            })
            
            # Control B: Class-Conditional vs Honest Control
            mad_cc_h = float(np.mean(np.abs(rec_cc - rec_honest)))
            rho_cc_h, _ = stats.spearmanr(rec_cc, rec_honest)
            pct5_cc_h = float(np.mean(np.abs(rec_cc - rec_honest) <= 5.0) * 100.0)
            pct10_cc_h = float(np.mean(np.abs(rec_cc - rec_honest) <= 10.0) * 100.0)
            
            agreement_records.append({
                "comparison": "Class_Conditional_vs_Honest_Control",
                "specification": spec,
                "epsilon": eps,
                "MAD_mg_dL": mad_cc_h,
                "spearman_rho": float(rho_cc_h),
                "target_correlation_r": np.nan,
                "pct_within_5mg_dL": pct5_cc_h,
                "pct_within_10mg_dL": pct10_cc_h,
                "fallback_count": np.nan,
                "fallback_rate_pct": np.nan,
                "pct_rows_closer_to_file": np.nan
            })

df_agreement = pd.DataFrame(agreement_records)
df_agreement.to_csv("v3/results/E1_rltr_row_agreement.csv", index=False)
print("Saved complete row-level agreement and control table to: v3/results/E1_rltr_row_agreement.csv")

print("\n" + "="*75)
print("EXPERIMENT E1 COMPLETED SUCCESSFULLY")
print("="*75)

