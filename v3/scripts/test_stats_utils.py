import sys
import numpy as np
from scipy import stats
from stats_utils import (
    assert_disjoint,
    nadeau_bengio_variance,
    nadeau_bengio_ttest,
    tost_equivalence,
    holm_bonferroni,
    mcnemar_exact,
    compute_all_metrics
)


def test_assert_disjoint():
    print("Testing assert_disjoint...")
    train_idx = np.array([0, 1, 2, 3, 4])
    test_idx = np.array([5, 6, 7])
    # Should pass without error
    assert_disjoint(train_idx, test_idx)
    
    # Overlapping indices should raise AssertionError
    overlapping_test = np.array([4, 5, 6])
    try:
        assert_disjoint(train_idx, overlapping_test)
        raise Exception("Failed: assert_disjoint did not raise AssertionError on overlapping indices!")
    except AssertionError as e:
        print(f"  [PASS] Correctly caught overlap: {e}")


def test_nadeau_bengio():
    print("\nTesting nadeau_bengio_ttest...")
    k = 50
    n_test = 77
    n_train = 691
    np.random.seed(42)
    # Synthetic paired differences: mean ~ 0.01, std ~ 0.02
    d = np.random.normal(loc=0.01, scale=0.02, size=k)
    
    res = nadeau_bengio_ttest(d, k, n_test, n_train)
    
    # Manual verification
    sample_var = np.var(d, ddof=1)
    expected_factor = (1.0 / k) + (n_test / n_train)
    expected_var = expected_factor * sample_var
    expected_se = np.sqrt(expected_var)
    expected_t = np.mean(d) / expected_se
    expected_p = 2.0 * stats.t.sf(np.abs(expected_t), df=k - 1)
    
    assert np.isclose(res["se_corr"], expected_se), "SE mismatch!"
    assert np.isclose(res["t_stat"], expected_t), "t-stat mismatch!"
    assert np.isclose(res["p_value"], expected_p), "p-value mismatch!"
    assert res["ci_lower"] < res["mean_diff"] < res["ci_upper"], "CI invalid!"
    print(f"  [PASS] Nadeau-Bengio: mean_diff={res['mean_diff']:.4f}, se_corr={res['se_corr']:.4f}, t={res['t_stat']:.3f}, p={res['p_value']:.4f}")


def test_tost():
    print("\nTesting tost_equivalence...")
    k = 50
    n_test = 77
    n_train = 691
    margin = 0.02
    
    # Case 1: differences very tightly clustered near 0 (mean = 0.001, std = 0.005)
    np.random.seed(123)
    d_equiv = np.random.normal(loc=0.001, scale=0.005, size=k)
    res_equiv = tost_equivalence(d_equiv, k, n_test, n_train, margin=margin, alpha=0.05)
    assert res_equiv["equivalent"] is True, "Expected equivalence to be True!"
    print(f"  [PASS] Equivalence True case: mean_diff={res_equiv['mean_diff']:.4f}, p_tost={res_equiv['p_tost']:.4e}, equivalent={res_equiv['equivalent']}")
    
    # Case 2: differences centered far from 0 (mean = 0.05, std = 0.01)
    d_nonequiv = np.random.normal(loc=0.05, scale=0.01, size=k)
    res_nonequiv = tost_equivalence(d_nonequiv, k, n_test, n_train, margin=margin, alpha=0.05)
    assert res_nonequiv["equivalent"] is False, "Expected equivalence to be False!"
    print(f"  [PASS] Equivalence False case: mean_diff={res_nonequiv['mean_diff']:.4f}, p_tost={res_nonequiv['p_tost']:.4f}, equivalent={res_nonequiv['equivalent']}")


def test_holm_bonferroni():
    print("\nTesting holm_bonferroni...")
    p_raw = [0.001, 0.04, 0.02, 0.10]
    p_adj = holm_bonferroni(p_raw)
    
    # Sorted order: 0.001 (m=4 -> *4=0.004), 0.02 (m=3 -> *3=0.06), 0.04 (m=2 -> *2=0.08), 0.10 (m=1 -> *1=0.10)
    expected_sorted = [0.004, 0.06, 0.08, 0.10]
    # Check corresponding positions:
    # idx 0: 0.001 -> 0.004
    # idx 1: 0.04 -> 0.08
    # idx 2: 0.02 -> 0.06
    # idx 3: 0.10 -> 0.10
    expected = [0.004, 0.08, 0.06, 0.10]
    assert np.allclose(p_adj, expected), f"Holm-Bonferroni mismatch! Got {p_adj}, expected {expected}"
    print(f"  [PASS] Holm-Bonferroni: raw {p_raw} -> adjusted {p_adj.tolist()}")


def test_mcnemar():
    print("\nTesting mcnemar_exact...")
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # Model 1 correct on all
    y_p1   = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    # Model 2 wrong on 4 positives (b=4, c=0)
    y_p2   = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    res = mcnemar_exact(y_true, y_p1, y_p2)
    assert res["b"] == 4 and res["c"] == 0, "Discordant counts mismatch!"
    # binomtest(4, 4, 0.5) = (0.5)^4 * 2 = 0.125
    assert np.isclose(res["p_value"], 0.125), f"McNemar p-value mismatch: {res['p_value']}"
    print(f"  [PASS] McNemar: b={res['b']}, c={res['c']}, p={res['p_value']:.4f}")


def test_metrics():
    print("\nTesting compute_all_metrics...")
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.8, 0.85, 0.9])
    
    m = compute_all_metrics(y_true, y_pred, y_prob)
    assert "accuracy" in m and "roc_auc" in m and "pr_auc" in m and "brier" in m
    assert "weighted_f1" in m and "calibration_slope" in m
    print(f"  [PASS] Metrics computed: Acc={m['accuracy']:.2f}, Sens={m['sensitivity']:.2f}, Spec={m['specificity']:.2f}, AUC={m['roc_auc']:.2f}, Brier={m['brier']:.3f}, CalSlope={m['calibration_slope']:.3f}")


if __name__ == "__main__":
    test_assert_disjoint()
    test_nadeau_bengio()
    test_tost()
    test_holm_bonferroni()
    test_mcnemar()
    test_metrics()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
