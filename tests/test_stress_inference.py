"""tests for src/stress.py and src/inference.py -- run: pytest test_stress_inference.py -q"""
import numpy as np
import pandas as pd
import pytest

import src stress
import src inference


def _series(vals, start="2000-01-31"):
    idx = pd.date_range(start, periods=len(vals), freq="ME")
    return pd.Series(vals, index=idx)


# ---------------------------------------------------------------- stress: L1
def test_max_drawdown_known_path():
    # +10%, -50%, +10%: trough at 0.55/1.10 - 1 = -50%
    r = _series([0.10, -0.50, 0.10])
    assert stress.max_drawdown(r) == pytest.approx(-0.50, abs=1e-12)


def test_recovery_months_counts_to_new_peak():
    # peak, -20%, then +30% recovers past peak in 1 month after trough
    r = _series([0.05, -0.20, 0.30])
    assert stress.recovery_months(r) == 1.0


def test_recovery_inf_when_never_recovers():
    r = _series([0.05, -0.30, 0.01])
    assert stress.recovery_months(r) == float("inf")


def test_regime_table_slices_correct_window():
    idx = pd.date_range("1999-01-31", "2003-12-31", freq="ME")
    df = pd.DataFrame({"a": 0.01}, index=idx)
    # inject a crash inside the dot-com window only
    df.loc["2001-09-30", "a"] = -0.30
    tab = stress.regime_table(df, {"dot_com_unwind": ("2000-03-01", "2002-10-31")})
    row = tab.iloc[0]
    assert row["months"] == 32
    assert row["max_drawdown"] < -0.25


# ---------------------------------------------------------------- stress: L2
def test_bootstrap_paths_shape_and_values_from_source():
    r = _series(np.linspace(-0.02, 0.02, 60))
    paths = stress.block_bootstrap_paths(r, n_paths=50, block_len=6, horizon=60, seed=1)
    assert paths.shape == (50, 60)
    assert set(np.round(paths.ravel(), 10)).issubset(set(np.round(r.to_numpy(), 10)))


def test_bootstrap_risk_profile_orders_quantiles():
    rng = np.random.default_rng(0)
    r = _series(rng.normal(0.007, 0.045, 300))
    prof = stress.bootstrap_risk_profile(r, n_paths=300, seed=2)
    assert prof["maxdd_p05"] <= prof["maxdd_median"] <= prof["maxdd_p95"] <= 0
    assert prof["monthly_cvar95"] <= prof["monthly_var95"] < 0


# ---------------------------------------------------------------- stress: L3
def test_unwind_scenario_hand_computed():
    w = pd.Series({1: 0.30, 2: 0.10, 3: 0.60})
    res = stress.unwind_scenario(w, mega_ids=[1, 2], shock_mega=-0.40, shock_rest=-0.10)
    # 40% mega at -40%, 60% rest at -10% => -0.16 - 0.06 = -0.22
    assert res["mega_weight"] == pytest.approx(0.40)
    assert res["portfolio_return"] == pytest.approx(-0.22)


def test_unwind_table_common_megacap_set_and_protection_sign():
    cap = pd.Series({i: (10 - i) for i in range(1, 21)}, dtype=float)  # concentrated
    ew = pd.Series(1.0, index=cap.index)                                # equal weight
    tab = stress.unwind_table({"cap_weight": cap / cap.sum(), "equal": ew / ew.sum()},
                              benchmark_scheme="cap_weight", top_k=5)
    # equal weight holds less of the benchmark's top-5, so it must lose less
    assert tab.loc["equal", "portfolio_return"] > tab.loc["cap_weight", "portfolio_return"]
    assert tab.loc["equal", "protection_vs_benchmark_bps"] > 0


# ---------------------------------------------------------------- inference
def test_sharpe_diff_hac_exact_null_via_scale_invariance():
    # Sharpe is scale-invariant: 2*r has IDENTICAL Sharpe to r, so the null
    # is exactly true with a nondegenerate variance estimate.
    rng = np.random.default_rng(3)
    r = _series(rng.normal(0.006, 0.04, 300))
    res = inference.sharpe_diff_hac(r, 2.0 * r)
    assert abs(res["diff_ann"]) < 1e-8
    assert res["p_value"] > 0.95


def test_sharpe_diff_hac_detects_large_true_difference():
    rng = np.random.default_rng(4)
    base = rng.normal(0.0, 0.04, 600)
    a = _series(base + 0.010)        # Sharpe ~ 0.87 ann
    b = _series(base + 0.001)        # Sharpe ~ 0.09 ann; highly correlated pair
    res = inference.sharpe_diff_hac(a, b)
    assert res["diff_ann"] > 0.5
    assert res["p_value"] < 0.01


def test_sharpe_diff_bootstrap_agrees_directionally():
    rng = np.random.default_rng(5)
    base = rng.normal(0.0, 0.04, 360)
    a = _series(base + 0.008)
    b = _series(base + 0.001)
    res = inference.sharpe_diff_bootstrap(a, b, n_boot=800, seed=6)
    assert res["diff_ann"] > 0
    assert res["p_value"] < 0.05
    lo, hi = res["ci95_ann"]
    assert lo < res["diff_ann"] < hi


def test_ir_test_zero_alpha_high_pvalue():
    rng = np.random.default_rng(7)
    bench = _series(rng.normal(0.006, 0.04, 300))
    noise = rng.normal(0.0, 0.005, 300)
    scheme = bench + (noise - noise.mean())        # exactly zero mean active return
    res = inference.ir_test(scheme, bench)
    assert res["p_value"] > 0.05


def test_inference_table_shapes_and_columns():
    rng = np.random.default_rng(8)
    base = rng.normal(0.005, 0.04, 240)
    df = pd.DataFrame({
        "cap_weight": base,
        "equal": base + rng.normal(0, 0.004, 240),
        "minvar": base * 0.7,
    }, index=pd.date_range("2000-01-31", periods=240, freq="ME"))
    tab = inference.inference_table(df, benchmark="cap_weight")
    assert list(tab.index) == ["equal", "minvar"]
    for col in ["sharpe_diff_ann", "p_hac", "p_bootstrap", "ir_ann", "p_ir"]:
        assert col in tab.columns
