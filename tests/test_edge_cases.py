"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste Gruppe, inklusive Grenzen, Reglerränder, Entartungen, Überläufe."""

import math

import numpy as np
import pytest

import swg_constants as C
import swg_evaluation as V
import swg_exact as X
import swg_experiments as XP
import swg_game as G
import swg_learning as L
import swg_replicator as RP
import swg_scenario as S


def test_exactly_two_partners_is_the_smallest_chain():
    """n = 2: das Minimum der anderen ist der Einsatz des Mitspielers; Gleichgewichte, Treffzeit und Toleranz greifen nicht auf einen Zweig `n < 2` zurück."""
    assert G.excl_min(np.array([2, 6])).tolist() == [6, 2]
    assert len(G.equilibria(2, 0.2, 0.1, 0.6)) == 7 and G.tolerance(2, 0.1) == pytest.approx(0.9)
    mat, mins = X.br_chain(2, 0.05)
    assert mat.shape == (49, 49) and X.hitting_time(mat, 0, mins >= 4) > 0
    assert np.isfinite(L.simulate(S.Settings(n=2, runs=5, rounds=5)).mins).all()


def test_one_run_and_one_round_do_not_divide_by_zero():
    v = V.analyse(S.Settings(n=3, runs=1, rounds=1))
    assert v.diff_se == 0.0 and np.isfinite(v.mean_min).all() and v.se_min.tolist() == [0.0]
    kind, text = V.verdict_start_or_learning(v)
    assert kind == "info" and "Nur Runde 1" in text


def test_a_run_with_one_run_but_many_rounds_has_finite_numbers():
    v = V.analyse(S.Settings(n=3, runs=1, rounds=20))
    assert np.isfinite(v.mean_min).all() and np.isfinite(v.pay_end) and v.unchanged in (0.0, 1.0)


def test_price_equal_to_prize_is_a_tie_and_still_runs():
    """b = a (kein Zugewinn aus höherer Stufe): die beste Antwort nimmt bei Gleichstand die höhere Stufe; das Lernen rechnet."""
    r = L.simulate(S.Settings(n=3, a=0.2, b=0.2, rule=C.RULE_BEST, eps=0.0, runs=10, rounds=10))
    assert (r.mins == r.mins[:, :1]).all()
    assert np.isfinite(L.simulate(S.Settings(n=3, a=0.2, b=0.2, runs=10, rounds=10)).mins).all()


def test_tolerance_at_the_boundaries():
    assert G.tolerance(5, 1.0) == pytest.approx(0.0)              # b = a: keine Toleranz
    assert G.tolerance(5, 0.0) == pytest.approx(1.0)              # b = 0: jede Verfehlung ist tragbar
    assert 0.0 < G.tolerance(32, 0.95) < 0.01


def test_zero_sharpness_chooses_uniformly_and_zero_trembling_chooses_exactly():
    r = L.simulate(S.Settings(n=4, lam=0.0, runs=2000, rounds=3, seed=4))
    uniform = X.uniform_expected_min(4)
    se = r.mins[:, 2].std(ddof=1) / math.sqrt(2000)
    assert abs(r.mins[:, 2].mean() - uniform) < 4 * se                        # λ = 0: jede Stufe gleich wahrscheinlich, also der Zufall von Runde 1
    full = L.simulate(S.Settings(n=4, rule=C.RULE_BEST, eps=1.0, runs=2000, rounds=3, seed=4))
    se2 = full.mins[:, 2].std(ddof=1) / math.sqrt(2000)
    assert abs(full.mins[:, 2].mean() - uniform) < 4 * se2                    # ε = 1: immer zufällig


def test_forgetting_zero_and_one_are_both_valid():
    for phi in (0.0, 1.0):
        for rule in (C.RULE_SUM, C.RULE_MEAN):
            r = L.simulate(S.Settings(n=4, phi=phi, rule=rule, lam=20.0, runs=20, rounds=20))
            assert np.isfinite(r.mins).all()


def test_huge_sharpness_and_huge_prior_do_not_overflow_the_exponential():
    """Logit-Argument bis 200 · 2 / 0,2 · Summe über 400 Runden: exp(z − max) bleibt endlich, die Wahl ist nie NaN."""
    r = L.simulate(S.Settings(n=8, lam=200.0, phi=1.0, a=2.0, normalize=False, prior=60.0, runs=20, rounds=400))
    assert np.isfinite(r.mins).all() and r.final.min() >= 1 and r.final.max() <= 7
    big = L.simulate(S.Settings(n=8, rule=C.RULE_FICT, prior=60.0, runs=20, rounds=400))
    assert np.isfinite(big.mins).all()


def test_slider_maximum_runs_within_the_budget():
    """n = 32, T = 400: die App kürzt die Läufe auf RUN_BUDGET / (n · T); der Lauf rechnet und liefert endliche Zahlen (kein Zeitvergleich, nur Lauffähigkeit)."""
    s = S.Settings(n=C.N_MAX, rounds=C.T_MAX, runs=C.RUNS_MAX, rule=C.RULE_MEAN, lam=50.0)
    assert S.budget_runs(s) == C.RUN_BUDGET // (C.N_MAX * C.T_MAX) == 781
    v = V.analyse(S.with_(s, runs=S.budget_runs(s) // 8))
    assert np.isfinite(v.mean_min).all() and v.mean_min.shape == (C.T_MAX,)


def test_budget_never_drops_below_one_run_and_never_exceeds_the_request():
    assert S.budget_runs(S.Settings(n=32, rounds=400, runs=50)) == 50
    assert S.budget_runs(S.Settings(n=2, rounds=10, runs=1000)) == 1000
    assert S.budget_runs(S.Settings(n=32, rounds=400, runs=1)) == 1


def test_classify_boundary_is_inclusive():
    """Genau vier Standardfehler Abstand zählt als „bleibt“ (≤), knapp darüber nicht."""
    assert V.classify(1.0, 0.1, 1.4, 0.0) == 0 and V.classify(1.0, 0.1, 1.4001, 0.0) == 1 and V.classify(1.0, 0.1, 0.5999, 0.0) == -1


def test_explicit_zero_standard_error_distinguishes_any_difference():
    assert V.classify(3.0, 0.0, 3.0, 0.0) == 0 and V.classify(3.0, 0.0, 2.9, 0.0) == -1


def test_all_top_start_with_one_stage_game_is_degenerate_but_fine():
    """Eine einzige Stufe: nichts zu wählen, das Minimum ist 1 und bleibt es."""
    r = L.simulate(S.Settings(n=3, levels=1, runs=5, rounds=4))
    assert (r.mins == 1).all()
    assert X.uniform_expected_min(5, 1) == pytest.approx(1.0)


def test_replicator_with_two_players_and_extreme_ratio_stays_a_distribution():
    p0 = RP.random_starts(50, 2)
    p, _ = RP.run(p0, 2, 0.2, 0.19, 0.6, steps=3000)
    assert np.allclose(p.sum(axis=1), 1.0) and (p >= 0).all() and np.isfinite(p).all()
    p32, _ = RP.run(p0, 32, 0.2, 0.01, 0.6, steps=3000)
    assert np.allclose(p32.sum(axis=1), 1.0) and np.isfinite(p32).all()


def test_replicator_without_any_step_returns_the_start():
    p0 = RP.random_starts(4, 3)
    p, used = RP.run(p0, 4, 0.2, 0.1, 0.6, steps=0)
    assert np.array_equal(p, p0) and used.tolist() == [0, 0, 0, 0]


def test_first_drop_row_handles_n_equal_two_and_the_largest_group():
    rows = XP.e2_first_drop_rows(0.1, 50, ns=(2, 32), rounds=60)
    assert [r["n"] for r in rows] == [2, 32] and all(np.isfinite(r["sim"]) for r in rows)


def test_big_number_format_switches_to_powers_of_ten_at_ten_thousand():
    assert V.big(328.0) == "328" and V.big(9999.0) == "9.999" and V.big(9.4e8) == "9,4·10^8" and V.big(58.7) == "58,7"
