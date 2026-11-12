import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)


def assert_disjoint(train_idx, test_idx):
    """
    Assert that training and testing index sets are strictly disjoint.
    Raises AssertionError if any overlap exists.
    """
    train_set = set(train_idx)
    test_set = set(test_idx)
    intersection = train_set & test_set
    if len(intersection) > 0:
        raise AssertionError(
            f"Data leakage detected: {len(intersection)} indices overlap between train and test sets! "
            f"Overlap sample: {list(intersection)[:5]}"
        )


def nadeau_bengio_variance(d, k, n_test, n_train):
    """
    Compute Nadeau-Bengio (2003) corrected variance for resampled / repeated cross-validation:
        Var_corr = (1/k + n_test / n_train) * Var(d, ddof=1)
    where:
        d: array-like of paired differences of length k
        k: total number of evaluations (e.g., 50 for 10x5 RepeatedStratifiedKFold)
        n_test: number of samples in test fold
        n_train: number of samples in train fold
    """
    d = np.asarray(d, dtype=float)
    if len(d) != k:
        raise ValueError(f"Length of differences ({len(d)}) does not match k ({k})")
    
    sample_var = np.var(d, ddof=1)
    correction_factor = (1.0 / k) + (float(n_test) / float(n_train))
    corr_var = correction_factor * sample_var
    return corr_var


def nadeau_bengio_ttest(d, k, n_test, n_train):
    """
    Nadeau-Bengio corrected resampled t-test for paired fold scores.
    Returns:
        dict with keys: 'mean_diff', 'std_diff', 'se_corr', 't_stat', 'df', 'p_value', 'ci_lower', 'ci_upper'
    """
    d = np.asarray(d, dtype=float)
    k_len = len(d)
    if k_len != k:
        raise ValueError(f"Length of differences ({k_len}) does not match k ({k})")
    
    mean_diff = float(np.mean(d))
    std_diff = float(np.std(d, ddof=1))
    
    corr_var = nadeau_bengio_variance(d, k, n_test, n_train)
    se_corr = float(np.sqrt(corr_var))
    
    df = k - 1
    if se_corr == 0.0:
        t_stat = 0.0
        p_val = 1.0
        ci_lower = mean_diff
        ci_upper = mean_diff
    else:
        t_stat = float(mean_diff / se_corr)
        p_val = float(2.0 * stats.t.sf(np.abs(t_stat), df=df))
        t_crit = float(stats.t.ppf(0.975, df=df))
        ci_lower = float(mean_diff - t_crit * se_corr)
        ci_upper = float(mean_diff + t_crit * se_corr)
    
    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se_corr": se_corr,
        "t_stat": t_stat,
        "df": df,
        "p_value": p_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }


def tost_equivalence(d, k, n_test, n_train, margin=0.02, alpha=0.05):
    """
    Two One-Sided Tests (TOST) for statistical equivalence using Nadeau-Bengio corrected variance.
    Tests H0: |mean_diff| >= margin versus H1: -margin < mean_diff < margin.
    
    Returns:
        dict with keys: 'mean_diff', 'margin', 'se_corr', 't1', 'p1', 't2', 'p2', 'p_tost', 'equivalent'
    """
    d = np.asarray(d, dtype=float)
    mean_diff = float(np.mean(d))
    corr_var = nadeau_bengio_variance(d, k, n_test, n_train)
    se_corr = float(np.sqrt(corr_var))
    df = k - 1
    
    if se_corr == 0.0:
        if abs(mean_diff) < margin:
            return {
                "mean_diff": mean_diff,
                "margin": margin,
                "se_corr": 0.0,
                "t1": np.inf,
                "p1": 0.0,
                "t2": -np.inf,
                "p2": 0.0,
                "p_tost": 0.0,
                "equivalent": True
            }
        else:
            return {
                "mean_diff": mean_diff,
                "margin": margin,
                "se_corr": 0.0,
                "t1": -np.inf,
                "p1": 1.0,
                "t2": np.inf,
                "p2": 1.0,
                "p_tost": 1.0,
                "equivalent": False
            }
            
    # H01: mean_diff <= -margin vs H11: mean_diff > -margin
    t1 = float((mean_diff - (-margin)) / se_corr)
    p1 = float(stats.t.sf(t1, df=df))
    
    # H02: mean_diff >= +margin vs H12: mean_diff < +margin
    t2 = float((mean_diff - margin) / se_corr)
    p2 = float(stats.t.cdf(t2, df=df))
    
    p_tost = float(max(p1, p2))
    equivalent = bool(p_tost < alpha)
    
    return {
        "mean_diff": mean_diff,
        "margin": margin,
        "se_corr": se_corr,
        "t1": t1,
        "p1": p1,
        "t2": t2,
        "p2": p2,
        "p_tost": p_tost,
        "equivalent": equivalent
    }


def holm_bonferroni(p_values):
    """
    Holm-Bonferroni step-down family-wise error rate adjustment.
    Input:
        p_values: list or array of unadjusted p-values
    Returns:
        np.ndarray of adjusted p-values in original order
    """
    p_arr = np.asarray(p_values, dtype=float)
    m = len(p_arr)
    if m == 0:
        return np.array([])
    
    sort_idx = np.argsort(p_arr)
    sorted_p = p_arr[sort_idx]
    
    adjusted_sorted = np.zeros(m, dtype=float)
    for i in range(m):
        rank = i + 1
        adj_p = (m - rank + 1) * sorted_p[i]
        adjusted_sorted[i] = adj_p
        
    # Enforce monotonicity: adjusted p cannot decrease
    for i in range(1, m):
        if adjusted_sorted[i] < adjusted_sorted[i - 1]:
            adjusted_sorted[i] = adjusted_sorted[i - 1]
            
    # Cap at 1.0
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    
    # Restore original order
    adjusted = np.zeros(m, dtype=float)
    adjusted[sort_idx] = adjusted_sorted
    return adjusted


def mcnemar_exact(y_true, y_pred1, y_pred2):
    """
    Exact McNemar test using binomial distribution for paired holdout predictions.
    b: model 1 correct, model 2 incorrect
    c: model 1 incorrect, model 2 correct
    """
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)
    
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)
    
    b = int(np.sum(correct1 & ~correct2))
    c = int(np.sum(~correct1 & correct2))
    n_discordant = b + c
    
    if n_discordant == 0:
        p_value = 1.0
    else:
        test_res = stats.binomtest(b, n=n_discordant, p=0.5, alternative='two-sided')
        p_value = float(test_res.pvalue)
        
    return {
        "b": b,
        "c": c,
        "n_discordant": n_discordant,
        "p_value": p_value
    }


def compute_calibration_metrics(y_true, y_prob):
    """
    Compute calibration slope and intercept via logistic regression:
        logit(y) = intercept + slope * logit(p)
    Clipped to avoid inf in logit.
    Also returns Brier score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    eps = 1e-6
    p_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logit_p = np.log(p_clipped / (1.0 - p_clipped))
    
    # Logistic regression of y on logit(p) unpenalized
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e9, solver='lbfgs', max_iter=200)
        lr.fit(logit_p.reshape(-1, 1), y_true)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
    except Exception:
        slope = np.nan
        intercept = np.nan
        
    brier = float(brier_score_loss(y_true, y_prob))
    return {
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "brier_score": brier
    }


def compute_all_metrics(y_true, y_pred, y_prob=None):
    """
    Compute comprehensive metrics:
      - accuracy
      - roc_auc (if y_prob provided)
      - pr_auc (average precision score, if y_prob provided)
      - brier_score (if y_prob provided)
      - sensitivity (recall of positive/diabetic class)
      - specificity (recall of negative/healthy class)
      - f1 (positive/diabetic class)
      - weighted_precision
      - weighted_recall
      - weighted_f1
      - calibration_slope & intercept (if y_prob provided)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f1_diabetic = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    
    w_prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    w_rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    w_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    metrics = {
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1_diabetic,
        "weighted_precision": w_prec,
        "weighted_recall": w_rec,
        "weighted_f1": w_f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }
    
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = np.nan
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["pr_auc"] = np.nan
            
        calib = compute_calibration_metrics(y_true, y_prob)
        metrics["brier"] = calib["brier_score"]
        metrics["calibration_slope"] = calib["calibration_slope"]
        metrics["calibration_intercept"] = calib["calibration_intercept"]
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
        metrics["brier"] = np.nan
        metrics["calibration_slope"] = np.nan
        metrics["calibration_intercept"] = np.nan
        
    return metrics
