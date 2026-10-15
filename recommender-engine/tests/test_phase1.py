"""Intuition: five fast tests for the phase-1 pieces that fail silently if wrong: NDCG math,
search-row mask semantics, train-only scaling, positive weights, and user eligibility."""
import numpy as np
import pandas as pd

from feed.data import DENSE, fit_scaler, make_targets, standardize
from feed.eval import bootstrap_ci, eligible_user_mask, paired_bootstrap, per_user_ndcg
from feed.train import compute_pos_weights

GAINS = [0, 1, 2, 3, 4]


def test_ndcg_toy_example():
    users = np.array([7, 7, 7])
    tiers = np.array([0, 1, 4])
    perfect_users, perfect = per_user_ndcg(np.array([1.0, 2.0, 3.0]), tiers, users, GAINS, k=3)
    _, worst = per_user_ndcg(np.array([3.0, 2.0, 1.0]), tiers, users, GAINS, k=3)
    assert perfect_users.tolist() == [7]
    assert np.isclose(perfect[0], 1.0)
    ideal = 4.0 + 1.0 / np.log2(3.0)
    assert np.isclose(worst[0], (1.0 / np.log2(3.0) + 2.0) / ideal)


def test_search_rows_mask_non_click_heads():
    df = pd.DataFrame({
        "click": [1, 1], "like": [0, 0], "follow": [0, 0], "forward": [0, 0],
        "watch": [0.5, np.nan],
        "obs_like": [1, 0], "obs_follow": [1, 0], "obs_forward": [1, 0], "obs_watch": [1, 0],
    })
    y, m = make_targets(df)
    assert m[0].all() and np.isnan(y).sum() == 0
    assert m[1, 0] == 1 and (m[1, 1:] == 0).all()
    assert y[1, 4] == 0.0


def test_scaler_fits_on_train_only():
    train = pd.DataFrame({f: np.arange(4, dtype=np.float64) for f in DENSE})
    train["i_n"] = [0.0, 1.0, 3.0, 7.0]
    shifted = train + 1000.0
    scaler = fit_scaler(train)
    j = DENSE.index("u_pos_rate")
    assert np.isclose(scaler["mean"][j], 1.5)
    z_train = standardize(train, scaler)
    z_shifted = standardize(shifted, scaler)
    assert np.isfinite(z_train).all() and np.isfinite(z_shifted).all()
    assert np.isclose(z_train[:, j].mean(), 0.0, atol=1e-6)
    assert not np.isclose(z_shifted[:, j].mean(), 0.0)


def test_pos_weight_formula_and_cap():
    y = np.array([[1.0], [0.0], [1.0], [0.0]])
    m = np.ones_like(y)
    assert np.isclose(compute_pos_weights(y, m, cap=10.0)[0], 1.0)
    y_rare = np.zeros((101, 1))
    y_rare[0, 0] = 1.0
    pw = compute_pos_weights(y_rare, np.ones_like(y_rare), cap=10.0)[0]
    assert pw == 10.0


def test_eligibility_and_bootstrap():
    users = np.array([1, 1, 2, 2, 2, 3])
    tiers = np.array([1, 2, 1, 1, 2, 1])
    rows_ok, uniq, keep = eligible_user_mask(users, tiers)
    assert uniq.tolist() == [1, 2, 3]
    assert keep.tolist() == [True, True, False]
    assert rows_ok.tolist() == [True, True, True, True, True, False]
    ci = bootstrap_ci(np.array([0.4, 0.6, 0.5]), n_boot=200, seed=42)
    assert ci["lo"] < 0.5 < ci["hi"] and ci["n_users"] == 3
    paired = paired_bootstrap(np.array([0.6, 0.7]), np.array([0.5, 0.5]), n_boot=200, seed=42)
    assert paired["mean_diff"] > 0 and paired["lo"] >= 0
