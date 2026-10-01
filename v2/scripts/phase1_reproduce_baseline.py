import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from scipy.stats import binomtest
import json

print("--- PHASE 1: REPRODUCE PAPER BASELINE ---")

df = pd.read_csv("RLTR_Imputed.csv")
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Test different scalers and splitting options to match exact 90/10/13/41 matrix
scalers = {
    "None": None,
    "MinMax": MinMaxScaler(),
    "Standard": StandardScaler(),
    "Robust": RobustScaler()
}

print(f"Dataset shape: {df.shape}, Outcomes: {y.value_counts().to_dict()}")

found_match = False
best_match_info = None

for strat in [True, False]:
    for rs in [42, 12, 0, 1]:
        for scaler_name, scaler in scalers.items():
            if scaler is not None:
                # Scaled before split as described in paper
                X_s = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            else:
                X_s = X.copy()
            
            strat_param = y if strat else None
            X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.20, random_state=rs, stratify=strat_param)
            
            # XGB defaults
            xgb = XGBClassifier(random_state=rs, eval_metric="logloss")
            xgb.fit(X_tr, y_tr)
            preds = xgb.predict(X_te)
            
            cm = confusion_matrix(y_te, preds)
            acc = accuracy_score(y_te, preds)
            
            # Check for exact 90, 10, 13, 41
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                if tn == 90 and fp == 10 and fn == 13 and tp == 41:
                    print(f"EXACT MATCH FOUND! Strat={strat}, RS={rs}, Scaler={scaler_name}")
                    print(f"CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}, Acc={acc:.4f}")
                    found_match = True
                    best_match_info = {
                        "stratify": strat,
                        "random_state": rs,
                        "scaler": scaler_name,
                        "cm": [tn, fp, fn, tp],
                        "acc": acc
                    }
                    break
                elif abs(acc - 0.8506) < 0.005:
                    print(f"Close match: Strat={strat}, RS={rs}, Scaler={scaler_name} -> Acc={acc:.4f}, CM: {cm.ravel()}")
        if found_match:
            break
    if found_match:
        break

if not found_match:
    print("Exact 90/10/13/41 not hit by raw loop, testing specific variants (e.g. Glucose 0 fix, default XGB seed, max_depth)...")
