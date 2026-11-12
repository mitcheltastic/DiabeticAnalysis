import os
import hashlib
import numpy as np
import pandas as pd

raw_path = "Dataset Diabetes.csv"
imputed_files = {
    "RLTR": "RLTR_Imputed.csv",
    "LTR": "LTR_Imputed.csv",
    "NSSR": "NSSR_Imputed.csv",
    "SIM": "SIM_Imputed.csv",
    "TR": "TR_Imputed.csv"
}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

raw_df = pd.read_csv(raw_path, sep=";")
print(f"Raw Dataset Diabetes.csv: SHA256 = {sha256_file(raw_path)}, Rows = {len(raw_df)}")

# Check duplicate rows in raw
print(f"Raw duplicates: {raw_df.duplicated().sum()}")

invariant_cols = ["Outcome", "Pregnancies", "Age", "DiabetesPedigreeFunction"]
physiological_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

results = []

for name, path in imputed_files.items():
    f_sha = sha256_file(path)
    df = pd.read_csv(path)
    n_rows = len(df)
    n_dupes = df.duplicated().sum()
    
    # 1. Row count match
    row_count_match = (n_rows == len(raw_df))
    
    # 2. Invariant columns mismatch
    mismatch_counts = {}
    for col in invariant_cols:
        # Check if values match row by row
        if col in df.columns:
            # Float comparison with tolerance for DPF
            diff = np.abs(df[col].values - raw_df[col].values)
            mismatch_counts[col] = int(np.sum(diff > 1e-4))
        else:
            mismatch_counts[col] = "Column Missing"
            
    # 3. Non-imputed cells altered (raw cells > 0)
    altered_cells = {}
    for col in physiological_cols:
        raw_vals = raw_df[col].values
        imp_vals = df[col].values
        # non-zero mask in raw
        nonzero_mask = (raw_vals > 0)
        # Check how many non-zero raw cells changed in imputed file
        diff = np.abs(imp_vals[nonzero_mask] - raw_vals[nonzero_mask])
        altered_cells[col] = int(np.sum(diff > 1e-4))
        
    results.append({
        "dataset": name,
        "file": path,
        "sha256": f_sha,
        "rows": n_rows,
        "duplicates": n_dupes,
        "mismatches_Outcome": mismatch_counts["Outcome"],
        "mismatches_Pregnancies": mismatch_counts["Pregnancies"],
        "mismatches_Age": mismatch_counts["Age"],
        "mismatches_DPF": mismatch_counts["DiabetesPedigreeFunction"],
        "altered_Glucose": altered_cells["Glucose"],
        "altered_BloodPressure": altered_cells["BloodPressure"],
        "altered_SkinThickness": altered_cells["SkinThickness"],
        "altered_Insulin": altered_cells["Insulin"],
        "altered_BMI": altered_cells["BMI"]
    })

res_df = pd.DataFrame(results)
print("\n" + res_df.to_string(index=False))

res_df.to_csv("v3/results/data_provenance_audit.csv", index=False)
print("\nSaved provenance report to v3/results/data_provenance_audit.csv")

