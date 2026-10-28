"""Exakte Einheiten: E[min], Pfadaufzählung des Logit, Kette der besten Antwort, Treffzeiten, Formel der ersten Abweichung."""

import numpy as np
import pytest

import swg_constants as C
import swg_exact as X
import swg_scenario as S


def test_uniform_expected_min_by_hand(uniform2):
    assert X.uniform_expected_min(uniform2["n"]) == pytest.approx(uniform2["value"])
    assert X.uniform_expected_min(1) == pytest.approx(4.0)           # ein Spieler: Mittel 1..7
    assert [round(X.uniform_expected_min(n), 3) for n in (2, 4, 8, 16)] == [2.857, 1.948, 1.372, 1.090]


def test_expected_min_independent_for_different_players():
    # zwei Spieler: einer sicher auf 3, einer gleichverteilt auf 1..4: E[min] = (1 + 2 + 3 + 3)/4 = 2,25
    p = np.array([[0, 0, 1.0, 0], [0.25, 0.25, 0.25, 0.25]])
    assert X.expected_min_independent(p) == pytest.approx(2.25)
    assert X.expected_min_iid(np.array([0.25] * 4), 2) == pytest.approx((16 + 9 + 4 + 1) / 16)


def test_logit_exact_path_one_step_by_hand(logit3):
    s = S.Settings(n=2, levels=3, start=C.START_TOP, lam=5.0, phi=0.9, rounds=2, normalize=False)
    out = X.logit_exact_path(s, 2)
    assert out[0] == pytest.approx(3.0) and out[1] == pytest.approx(logit3["emin2"], rel=1e-5)


def test_logit_exact_path_in_the_mean_version_by_hand(logit3):
    s = S.Settings(n=2, levels=3, start=C.START_TOP, rule=C.RULE_MEAN, lam=50.0, phi=0.9, rounds=2, normalize=False)
    out = X.logit_exact_path(s, 2)
    p3 = logit3["p3_mean"]
    p1 = np.exp(-10) / (1 + np.exp(-5) + np.exp(-10))
    p2 = np.exp(-5) / (1 + np.exp(-5) + np.exp(-10))
    assert p3 == pytest.approx(1 / (1 + np.exp(-5) + np.exp(-10)), abs=1e-6) and (1 - p1) == pytest.approx(p2 + p3)
    assert out[1] == pytest.approx(1.0 + (1 - p1) ** 2 + p3 ** 2, abs=1e-5)


def test_logit_exact_path_rejects_too_many_paths_and_best_reply():
    with pytest.raises(ValueError):
        X.logit_exact_path(S.Settings(n=4, rounds=4), 4)
    with pytest.raises(ValueError):
        X.logit_exact_path(S.Settings(n=2, rule=C.RULE_BEST), 2)


def test_first_round_of_the_exact_path_is_the_random_minimum():
    out = X.logit_exact_path(S.Settings(n=3, levels=4, rounds=1), 1)
    assert out[0] == pytest.approx(X.uniform_expected_min(3, 4))


def test_chain_has_the_right_shape_and_rows_sum_to_one():
    mat, mins = X.br_chain(2, 0.2, levels=2)
    assert mat.shape == (4, 4) and np.allclose(mat.sum(axis=1), 1.0) and mins.tolist() == [1, 1, 1, 2]
    mat3, _ = X.br_chain(3, 0.05)
    assert mat3.shape == (343, 343) and np.allclose(mat3.sum(axis=1), 1.0)


def test_stationary_expected_min_by_hand(chain2):
    mat, mins = X.br_chain(2, 0.2, levels=2)
    assert X.stationary(mat) @ mins == pytest.approx(chain2["stationary"])
    mat7, mins7 = X.br_chain(2, 0.01)
    assert X.stationary(mat7) @ mins7 == pytest.approx(chain2["seven"], rel=1e-9)


def test_chain_expected_min_starts_at_the_random_minimum():
    mat, mins = X.br_chain(3, 0.05)
    path = X.chain_expected_min(mat, mins, 5)
    assert path[0] == pytest.approx(X.uniform_expected_min(3))


def test_hitting_time_by_hand_in_a_two_state_chain():
    # Zustand 0 geht mit 0,25 in den Zielzustand 1: erwartete Zeit 4 (geometrisch)
    mat = np.array([[0.75, 0.25], [0.0, 1.0]])
    assert X.hitting_time(mat, 0, np.array([False, True])) == pytest.approx(4.0)
    assert X.hitting_time(mat, 1, np.array([False, True])) == 0.0


def test_first_drop_formula_by_hand():
    # n = 1, ε = 0,07: q = 1 − 0,07 + 0,01 = 0,94, 1 + 1/(1 − 0,94) = 17,667
    assert X.first_drop_rounds(1, 0.07) == pytest.approx(1 + 1 / 0.06)
    assert X.first_drop_rounds(2, 0.0) == float("inf")
