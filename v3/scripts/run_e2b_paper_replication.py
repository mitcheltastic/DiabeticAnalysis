import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import optuna

# Suppress Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("="*75)
print("EXPERIMENT E2b: PAPER 1 & PAPER 2 TABLE REPLICATION (SEED 42 ONLY)")
print("="*75)

os.makedirs("v3/results", exist_ok=True)

raw = pd.read_csv("Dataset Diabetes.csv", sep=";")
y = raw["Outcome"].values
n_samples = len(raw)

imputed_files = {
    "TR": "TR_Imputed.csv",
    "LTR": "LTR_Imputed.csv",
    "NSSR": "NSSR_Imputed.csv",
    "SIM": "SIM_Imputed.csv",
    "RLTR": "RLTR_Imputed.csv"
}

# Split indices (seed 42, 80/20 stratified)
train_idx, test_idx = train_test_split(
    np.arange(n_samples),
    test_size=0.20,
    random_state=42,
    stratify=y
)
y_tr, y_te = y[train_idx], y[test_idx]

# Paper 2 Reported Values (Table 3: Default, Table 4: Optuna)
paper2_reported = {
    "TR":   {"default_acc": 74.68, "optuna_acc": 79.87},
    "LTR":  {"default_acc": 72.73, "optuna_acc": 77.27},
    "NSSR": {"default_acc": 74.68, "optuna_acc": 79.22},
    "SIM":  {"default_acc": 73.38, "optuna_acc": 77.92},
    "RLTR": {"default_acc": 85.06, "optuna_acc": 87.01}
}

p2_records = []

for file_key, csv_path in imputed_files.items():
    print(f"\nProcessing {file_key} ({csv_path})...")
    df_raw = pd.read_csv(csv_path)
    X = df_raw.drop(columns=["Outcome"]).copy()
    
    # As published (R3a): published features + pre-split MinMaxScaler
    mms = MinMaxScaler()
    X_scaled = mms.fit_transform(X)
    X_tr, X_te = X_scaled[train_idx], X_scaled[test_idx]
    
    # 1. XGB Default
    xgb_def = XGBClassifier(
        max_depth=6,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=4
    )
    xgb_def.fit(X_tr, y_tr)
    pred_def = xgb_def.predict(X_te)
    acc_def = accuracy_score(y_te, pred_def) * 100
    p_def, r_def, f1_def, _ = precision_recall_fscore_support(y_te, pred_def, average="weighted")
    cm_def = confusion_matrix(y_te, pred_def).ravel()
    
    p2_records.append({
        "dataset_file": file_key,
        "variant": "XGB Default",
        "eval_objective": "None (Default Parameters)",
        "trials": 0,
        "accuracy": acc_def,
        "weighted_precision": p_def * 100,
        "weighted_recall": r_def * 100,
        "weighted_f1": f1_def * 100,
        "tn": cm_def[0], "fp": cm_def[1], "fn": cm_def[2], "tp": cm_def[3],
        "paper2_reported_acc": paper2_reported[file_key]["default_acc"],
        "acc_delta_vs_reported": acc_def - paper2_reported[file_key]["default_acc"]
    })
    
    # 2. Optuna 50 trials (Direct Test-Set Objective)
    def obj_test_50(trial):
        clf = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=42,
            eval_metric="logloss",
            n_jobs=4
        )
        clf.fit(X_tr, y_tr)
        return float(accuracy_score(y_te, clf.predict(X_te)))

    study_test = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_test.optimize(obj_test_50, n_trials=100)
    
    # Slice 50 trials
    trials_50 = [t for t in study_test.trials if t.number < 50 and t.value is not None]
    best_t_50 = max(trials_50, key=lambda t: t.value)
    clf_50 = XGBClassifier(**best_t_50.params, random_state=42, eval_metric="logloss", n_jobs=4)
    clf_50.fit(X_tr, y_tr)
    pred_50 = clf_50.predict(X_te)
    acc_50 = accuracy_score(y_te, pred_50) * 100
    p_50, r_50, f1_50, _ = precision_recall_fscore_support(y_te, pred_50, average="weighted")
    cm_50 = confusion_matrix(y_te, pred_50).ravel()
    
    p2_records.append({
        "dataset_file": file_key,
        "variant": "XGB Optuna (50 trials)",
        "eval_objective": "Direct Test-Set Accuracy",
        "trials": 50,
        "accuracy": acc_50,
        "weighted_precision": p_50 * 100,
        "weighted_recall": r_50 * 100,
        "weighted_f1": f1_50 * 100,
        "tn": cm_50[0], "fp": cm_50[1], "fn": cm_50[2], "tp": cm_50[3],
        "paper2_reported_acc": paper2_reported[file_key]["optuna_acc"],
        "acc_delta_vs_reported": acc_50 - paper2_reported[file_key]["optuna_acc"]
    })
    
    # 3. Optuna 100 trials (Direct Test-Set Objective)
    best_t_100 = study_test.best_trial
    clf_100 = XGBClassifier(**best_t_100.params, random_state=42, eval_metric="logloss", n_jobs=4)
    clf_100.fit(X_tr, y_tr)
    pred_100 = clf_100.predict(X_te)
    acc_100 = accuracy_score(y_te, pred_100) * 100
    p_100, r_100, f1_100, _ = precision_recall_fscore_support(y_te, pred_100, average="weighted")
    cm_100 = confusion_matrix(y_te, pred_100).ravel()
    
    p2_records.append({
        "dataset_file": file_key,
        "variant": "XGB Optuna (100 trials)",
        "eval_objective": "Direct Test-Set Accuracy",
        "trials": 100,
        "accuracy": acc_100,
        "weighted_precision": p_100 * 100,
        "weighted_recall": r_100 * 100,
        "weighted_f1": f1_100 * 100,
        "tn": cm_100[0], "fp": cm_100[1], "fn": cm_100[2], "tp": cm_100[3],
        "paper2_reported_acc": paper2_reported[file_key]["optuna_acc"],
        "acc_delta_vs_reported": acc_100 - paper2_reported[file_key]["optuna_acc"]
    })
    
    # 4. Optuna 50 trials (Honest 5-Fold CV Objective on Training Data)
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    def obj_cv_50(trial):
        clf = XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            random_state=42,
            eval_metric="logloss",
            n_jobs=4
        )
        scores = cross_val_score(clf, X_tr, y_tr, cv=cv_inner, scoring="accuracy", n_jobs=1)
        return float(np.mean(scores))

    study_cv = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study_cv.optimize(obj_cv_50, n_trials=50)
    
    clf_cv = XGBClassifier(**study_cv.best_params, random_state=42, eval_metric="logloss", n_jobs=4)
    clf_cv.fit(X_tr, y_tr)
    pred_cv = clf_cv.predict(X_te)
    acc_cv = accuracy_score(y_te, pred_cv) * 100
    p_cv, r_cv, f1_cv, _ = precision_recall_fscore_support(y_te, pred_cv, average="weighted")
    cm_cv = confusion_matrix(y_te, pred_cv).ravel()
    
    p2_records.append({
        "dataset_file": file_key,
        "variant": "XGB Optuna Honest CV (50 trials)",
        "eval_objective": "Honest 5-Fold CV on Training Data",
        "trials": 50,
        "accuracy": acc_cv,
        "weighted_precision": p_cv * 100,
        "weighted_recall": r_cv * 100,
        "weighted_f1": f1_cv * 100,
        "tn": cm_cv[0], "fp": cm_cv[1], "fn": cm_cv[2], "tp": cm_cv[3],
        "paper2_reported_acc": paper2_reported[file_key]["optuna_acc"],
        "acc_delta_vs_reported": acc_cv - paper2_reported[file_key]["optuna_acc"]
    })

df_p2 = pd.DataFrame(p2_records)
df_p2.to_csv("v3/results/E2b_paper2_replication.csv", index=False)
print("\nSaved Paper 2 replication table to: v3/results/E2b_paper2_replication.csv")

# ------------------------------------------------------------------------------
# Paper 1 Table II Replication (RLTR Normalized, Seed 42)
# ------------------------------------------------------------------------------
print("\nReplicating Paper 1 Table II (RLTR Normalized, Seed 42)...")
df_rltr = pd.read_csv("RLTR_Imputed.csv")
X_rltr = df_rltr.drop(columns=["Outcome"]).copy()
X_rltr_scaled = MinMaxScaler().fit_transform(X_rltr)
X_tr_rltr, X_te_rltr = X_rltr_scaled[train_idx], X_rltr_scaled[test_idx]

p1_models = {
    "Random Forest (RF)": (RandomForestClassifier(random_state=42), 0.83),
    "Support Vector Machine (SVM)": (SVC(random_state=42), 0.75),
    "K-Nearest Neighbors (KNN - default k=5)": (KNeighborsClassifier(n_neighbors=5), 0.74),
    "K-Nearest Neighbors (KNN - k=4)": (KNeighborsClassifier(n_neighbors=4), 0.74),
    "Logistic Regression (LR)": (LogisticRegression(random_state=42, max_iter=1000), 0.70)
}

p1_records = []
for name, (clf, rep_val) in p1_models.items():
    clf.fit(X_tr_rltr, y_tr)
    preds = clf.predict(X_te_rltr)
    acc = accuracy_score(y_te, preds)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_te, preds, average="weighted")
    cm = confusion_matrix(y_te, preds).ravel()
    
    p1_records.append({
        "model": name,
        "accuracy_reproduced": acc,
        "accuracy_reported_paper1": rep_val,
        "delta": acc - rep_val,
        "weighted_precision": p_w,
        "weighted_recall": r_w,
        "weighted_f1": f1_w,
        "tn": cm[0], "fp": cm[1], "fn": cm[2], "tp": cm[3]
    })

df_p1 = pd.DataFrame(p1_records)
df_p1.to_csv("v3/results/E2b_paper1_table2.csv", index=False)
print("Saved Paper 1 replication table to: v3/results/E2b_paper1_table2.csv")

print("\n" + "="*75)
print("EXPERIMENT E2b COMPLETED SUCCESSFULLY")
print("="*75)

