"""Experimente: Zellen, Seeds, Reihen (nicht konstant), exakte Bezugskurven der Hauptansicht, Formate der JSON-Wandlung."""

import json

import numpy as np
import pytest

import swg_constants as C
import swg_experiments as XP
import swg_exact as X
import swg_scenario as S


def test_seed_of_is_a_fixed_function_of_the_label():
    """FNV-1a ist reine Ganzzahlarithmetik: dieselbe Bezeichnung gibt immer denselben Seed, andere Bezeichnungen andere Seeds (nicht Pythons gesalzenes hash)."""
    assert XP.seed_of("e1|a|b|2") == XP.seed_of("e1|a|b|2")
    labels = [f"cell|{i}" for i in range(200)]
    assert len({XP.seed_of(x) for x in labels}) == 200 and all(0 <= XP.seed_of(x) <= C.SEED_MAX for x in labels)
    assert XP.seed_of("") == 0xCBF29CE484222325 % (C.SEED_MAX + 1)


def test_cell_reports_mean_se_and_top_share_in_the_chosen_rounds():
    c = XP.cell(S.Settings(n=3, runs=100, rounds=5, start=C.START_TOP, rule=C.RULE_FICT), (1, 3, 5))
    assert c["rounds"] == [1, 3, 5] and c["runs"] == 100 and c["mean"] == [7.0, 7.0, 7.0] and c["se"] == [0.0, 0.0, 0.0] and c["top"] == [1.0, 1.0, 1.0]


def test_cell_runs_to_the_largest_requested_round_only():
    c = XP.cell(S.Settings(n=4, runs=50), (1, 10))
    assert len(c["mean"]) == 2 and c["mean"][0] > c["mean"][1] > 0


def test_e1_cells_cover_every_rule_and_group_size():
    cells = XP.e1_cells(C.START_UNIFORM, 40, rounds=(1, 10), ns=(2, 4))
    assert set(cells) == set(XP.RULE_CONFIGS) and all(set(v) == {2, 4} for v in cells.values())


def test_e1_round_one_is_the_exact_random_minimum_for_every_rule_within_four_standard_errors():
    cells = XP.e1_cells(C.START_UNIFORM, 600, rounds=(1, 5), ns=(3, 8))
    for name, per_n in cells.items():
        for n, c in per_n.items():
            assert abs(c["mean"][0] - X.uniform_expected_min(n)) < 4 * max(c["se"][0], 1e-9), (name, n)


def test_e1_live_uses_the_given_settings_and_group_sizes():
    live = XP.e1_live(S.Settings(rule=C.RULE_MEAN, lam=50.0, start=C.START_TOP), 40, rounds=(1, 10), ns=(2, 4))
    assert set(live) == {2, 4} and live[2]["mean"][0] == 7.0


def test_sharpness_series_is_not_a_constant_column_in_any_convention():
    """Nullspalten-Regel: die Schärfe-Reihe darf nicht konstant sein; das Einfrieren bei großer Schärfe ist ein Befund, die Reihe als Ganzes bewegt sich."""
    for conv in ("sum", "mean"):
        cells = XP.e4_sharpness_cells(C.START_UNIFORM, 4, 200, conv, rounds=(1, 60))
        last = [c["mean"][-1] for c in cells.values()]
        assert max(last) - min(last) > 0.3, conv
    top = XP.e4_sharpness_cells(C.START_TOP, 4, 200, "sum", rounds=(1, 60))
    assert max(c["mean"][-1] for c in top.values()) - min(c["mean"][-1] for c in top.values()) > 3.0


def test_forgetting_series_is_not_constant():
    cells = XP.e4_forgetting_cells(C.START_TOP, 8, 200, rounds=(1, 60))
    last = [c["mean"][-1] for c in cells.values()]
    assert max(last) - min(last) > 1.0


def test_price_and_premium_series_move_the_minimum():
    price = XP.e5_price_cells(4, 200, rounds=(1, 60))
    assert price[0.01]["mean"][-1] > price[0.19]["mean"][-1] + 3.0
    norm = XP.e5_premium_cells(4, 200, True, rounds=(1, 60))
    raw = XP.e5_premium_cells(4, 200, False, rounds=(1, 60))
    assert norm[2.0]["mean"][-1] > norm[0.2]["mean"][-1] + 3.0 and norm[2.0]["mean"][-1] > raw[2.0]["mean"][-1] + 1.0
    gap = abs(norm[0.2]["mean"][-1] - raw[0.2]["mean"][-1])         # bei der Bezugsprämie a = 0,2 ist die Normierung ohne Wirkung (hier andere Zellen-Seeds: nur der Zufall trennt sie)
    assert gap < 4 * (norm[0.2]["se"][-1] ** 2 + raw[0.2]["se"][-1] ** 2) ** 0.5


def test_teams_prior_combo_and_grow_cells_have_the_documented_keys():
    teams = XP.e6_team_cells(30, rounds=(1, 5))
    assert set(teams) == {f"{k}|{f}" for k in C.POOL_TEAMS for f in (0, 1)}
    prior = XP.e6_prior_cells(4, 30, rounds=(1, 5))
    assert list(prior) == list(C.E6_PRIORS)
    combo = XP.e6_combo_cells(8, 30, rounds=(1, 5))
    assert list(combo) == ["0.2|0", "0.2|5", "1.0|0", "1.0|5", "1.0|30"]
    grow = XP.e6_grow_cells(8, 30, "sum", rounds=(1, 20))
    assert list(grow) == ["fest", "alle 10", "alle 25"]


def test_prior_series_is_monotone_in_the_early_rounds_and_moves_the_minimum():
    prior = XP.e6_prior_cells(4, 300, rounds=(1, 10))
    tenth = [c["mean"][1] for c in prior.values()]
    assert tenth[-1] > tenth[0] + 4.0 and all(y >= x - 0.35 for x, y in zip(tenth, tenth[1:]))


def test_growing_group_delays_the_collapse_in_this_model():
    grow = XP.e6_grow_cells(8, 200, "sum", rounds=(60,))
    assert grow["alle 25"]["mean"][0] > grow["fest"]["mean"][0] + 3.0


def test_chain_rows_give_the_exact_hand_checked_numbers_as_bands():
    rows = XP.e2_chain_rows(0.01, ns=(2, 3))
    assert [r["n"] for r in rows] == [2, 3] and rows[0]["states"] == 49
    assert rows[0]["up"] == pytest.approx(328.0, rel=1e-3) and rows[0]["down"] == pytest.approx(116.9, rel=1e-3) and rows[0]["stationary"] == pytest.approx(140 / 49, rel=1e-6)
    assert 1e6 < rows[1]["up"] < 1e7 and rows[1]["down"] == pytest.approx(78.1, rel=1e-2)


def test_first_drop_rows_compare_formula_and_simulation():
    rows = XP.e2_first_drop_rows(0.1, 400, ns=(2, 8), rounds=60)
    assert [r["n"] for r in rows] == [2, 8]
    for r in rows:
        assert abs(r["sim"] - r["formula"]) < 4 * r["se"] and r["censored"] == 0.0


def test_exact_overlay_for_the_chain_and_for_two_logit_players():
    best = XP.exact_overlay(S.Settings(rule=C.RULE_BEST, eps=0.05, n=3, rounds=20))
    assert best["x"] == list(range(1, 21)) and best["y"][0] == pytest.approx(X.uniform_expected_min(3))
    top = XP.exact_overlay(S.Settings(rule=C.RULE_BEST, eps=0.05, n=2, rounds=5, start=C.START_TOP))
    assert top["y"][0] == pytest.approx(7.0)
    logit = XP.exact_overlay(S.Settings(n=2, rounds=10))
    assert logit["x"] == [1, 2, 3] and logit["y"][0] == pytest.approx(X.uniform_expected_min(2))
    assert XP.exact_overlay(S.Settings(n=8)) is None and XP.exact_overlay(S.Settings(rule=C.RULE_FICT, n=2)) is None and XP.exact_overlay(S.Settings(n=2, rounds=2)) is None


def test_unchanged_share_is_one_for_the_noise_free_best_reply():
    out = XP.unchanged_share(60, ns=(2, 3), rounds=30)
    assert out["Beste Antwort (ε = 0)"][2]["unchanged"] == 1.0 and out["Beste Antwort (ε = 0)"][3]["unchanged"] == 1.0
    assert out["Fiktives Spiel"][2]["unchanged"] < 1.0                 # n = 2: Gleichstände der Bewertung wirken als Zufall


def test_to_jsonable_turns_keys_and_numpy_numbers_into_text_and_python_numbers():
    obj = {2: {0.5: np.float64(1.5)}, "k": [np.int64(3), (1, 2)]}
    out = XP.to_jsonable(obj)
    assert out == {"2": {"0.5": 1.5}, "k": [3, [1, 2]]} and json.dumps(out)


def test_start_share_boundaries():
    assert XP.start_share(7.0, 1.4, 1.0) == pytest.approx(5.6 / 6.0)
    assert XP.start_share(7.0, 7.0, 1.0) == pytest.approx(0.0)
    assert XP.start_share(7.0, 7.0, 7.0) is None and XP.start_share(7.0, 3.0, 7.5) is None
