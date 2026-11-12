# Protocol Amendment 2: Final-Model Selection Rule

Final-model selection rule (fixed before E4 summary statistics were computed or inspected):
- Candidate set: LogReg-L2 default, and each E4 tuned model (XGBoost, CatBoost, LogReg-EN).
- If no tuned model beats LogReg-L2 default on outer-fold ROC-AUC with a Holm-adjusted Nadeau-Bengio p < 0.05, the final model is LogReg-L2 default (parsimony).
- Otherwise, the final model is the tuned model with the highest mean outer-fold ROC-AUC.
- Imputer: in-fold median (a priori, Amendment 1 F).
