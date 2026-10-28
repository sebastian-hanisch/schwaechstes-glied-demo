"""Lernregeln: jede Einheit an einer Handrechnung, dann Eigenschaften des Laufs (Seed, Form, Rauschfreiheit, Zweige jeder Regel)."""

import numpy as np
import pytest

import swg_constants as C
import swg_learning as L
import swg_scenario as S
from swg_rng import uniform_block


# --- Strom -------------------------------------------------------------------------------------------------------------------------------------
def test_stream_blocks_continue_the_splitmix_stream():
    st = L.Stream(123)
    first, second = st.take(10), st.take(5)
    assert np.array_equal(np.concatenate([first, second]), uniform_block(123, 15))
    assert st.pos == 15


# --- Einheiten ---------------------------------------------------------------------------------------------------------------------------------
def test_logit_probs_by_hand(logit3):
    p = L.logit_probs(5.0 * np.array(logit3["pay"]))
    assert p == pytest.approx(logit3["p_sum"], abs=1e-6)
    p_mean = L.logit_probs(50.0 * np.array(logit3["pay"]))
    assert p_mean[2] == pytest.approx(logit3["p3_mean"], abs=1e-6)


def test_logit_probs_survive_huge_arguments():
    p = L.logit_probs(np.array([-1e5, 0.0, 1e5]))
    assert p.tolist() == [0.0, 0.0, 1.0] and np.isfinite(p).all()


def test_update_experience_by_hand():
    num, den = L.update_experience(np.array([[1.0, 2.0]]), np.array([[3.0]]), np.array([[0.5, 0.25]]), 0.9)
    assert num.tolist() == [[pytest.approx(1.4), pytest.approx(2.05)]] and den.tolist() == [[pytest.approx(3.7)]]


def test_logit_argument_conventions():
    num, den = np.array([[2.0, 4.0]]), np.array([[4.0]])
    assert L.logit_argument(num, den, 5.0, "sum").tolist() == [[10.0, 20.0]]
    assert L.logit_argument(num, den, 5.0, "mean").tolist() == [[2.5, 5.0]]


def test_draw_logit_inverts_the_cumulative_distribution(logit3):
    z = 5.0 * np.array(logit3["pay"])
    # kumulierte Verteilung: 0,186329 / 0,493525 / 1
    u = np.array([0.0, 0.18, 0.19, 0.49, 0.50, 0.999999])
    assert L.draw_logit(np.tile(z, (6, 1)), u).tolist() == [1, 1, 2, 2, 3, 3]


def test_draw_logit_never_exceeds_the_top_level_even_for_u_just_below_one():
    assert L.draw_logit(np.zeros((1, 7)), np.array([np.nextafter(1.0, 0.0)])).tolist() == [7]


def test_initial_profiles():
    u = np.array([[0.0, 0.14, 0.15, 0.999999]])
    assert L.initial_profile(C.START_UNIFORM, 7, u).tolist() == [[1, 1, 2, 7]]
    assert L.initial_profile(C.START_TOP, 7, u).tolist() == [[7, 7, 7, 7]]
    assert L.initial_profile(C.START_MID, 7, u).tolist() == [[4, 4, 4, 4]]
    assert L.initial_profile(C.START_MID, 3, u).tolist() == [[3, 3, 3, 3]]
    with pytest.raises(ValueError):
        L.initial_profile("Quatsch", 7, u)


def test_best_reply_choice_with_and_without_tremble():
    br = np.arange(1, 8)
    m = np.array([[5, 2, 2]])
    calm = L.best_reply_choice(m, br, 0.0, np.array([[0.0, 0.0, 0.0]]), np.array([[0.9, 0.9, 0.9]]), 7)
    assert calm.tolist() == [[5, 2, 2]]                                    # ε = 0: nie Zittern (u < 0 ist nie wahr)
    shaky = L.best_reply_choice(m, br, 0.1, np.array([[0.05, 0.2, 0.0999]]), np.array([[0.0, 0.9, 0.5]]), 7)
    assert shaky.tolist() == [[1, 2, 4]]                                   # erster und dritter zittern: Stufen 1 und 4


def test_team_excl_min_in_fixed_and_shuffled_teams():
    e = np.array([[2, 5, 6, 3]])
    m, gmin = L.team_excl_min(e, 2, np.array([[0, 1, 2, 3]]))              # Teams (2, 5) und (6, 3)
    assert m.tolist() == [[5, 2, 3, 6]] and gmin.tolist() == [[2, 3]]
    m2, gmin2 = L.team_excl_min(e, 2, np.array([[0, 2, 1, 3]]))             # Teams (2, 6) und (5, 3)
    assert m2.tolist() == [[6, 3, 2, 5]] and gmin2.tolist() == [[2, 3]]
    m3, gmin3 = L.team_excl_min(e, 4, np.array([[0, 1, 2, 3]]))             # ein Team: Minimum der anderen wie G.excl_min
    assert m3.tolist() == [[3, 2, 2, 2]] and gmin3.tolist() == [[2]]


# --- Läufe --------------------------------------------------------------------------------------------------------------------------------------
def test_same_seed_same_run_and_other_seed_other_run():
    a = L.simulate(S.Settings(n=4, runs=20, rounds=20, seed=5))
    b = L.simulate(S.Settings(n=4, runs=20, rounds=20, seed=5))
    c = L.simulate(S.Settings(n=4, runs=20, rounds=20, seed=6))
    assert np.array_equal(a.mins, b.mins) and np.array_equal(a.trace, b.trace) and not np.array_equal(a.mins, c.mins)


def test_run_shapes_and_value_ranges():
    r = L.simulate(S.Settings(n=5, runs=30, rounds=12))
    assert r.mins.shape == (30, 12) and r.top.shape == (30, 12) and r.trace.shape == (12, 5) and r.final.shape == (30, 5)
    assert r.trace.min() >= 1 and r.trace.max() <= 7 and r.final.min() >= 1 and r.final.max() <= 7 and r.runs == 30
    assert np.array_equal(r.mins[0], r.trace.min(axis=1))                  # Kettenminimum des ersten Laufs aus der Spur
    assert np.array_equal(r.mins[:, -1], r.final.min(axis=1))


def test_start_all_top_begins_at_the_top_and_all_mid_at_four():
    assert (L.simulate(S.Settings(n=3, runs=10, rounds=3, start=C.START_TOP)).mins[:, 0] == 7).all()
    assert (L.simulate(S.Settings(n=3, runs=10, rounds=3, start=C.START_MID)).mins[:, 0] == 4).all()


def test_round_one_is_pure_chance_for_every_rule():
    """Runde 1 hängt von der Lernregel nicht ab: bei gleichverteiltem Start ist das Kettenminimum der Zufall (der Strom ist derselbe, die Regel kommt erst danach)."""
    first = {rule: L.simulate(S.Settings(n=6, runs=50, rounds=2, rule=rule)).mins[:, 0] for rule in C.RULES}
    ref = first[C.RULE_SUM]
    assert all(np.array_equal(ref, v) for v in first.values())


def test_best_reply_without_trembling_never_moves_the_minimum_in_any_run():
    """Die beste Antwort ohne Zittern erhält das Minimum der Vorrunde (jeder spielt die Stufe des Minimums der anderen): Zahl für Zahl konstant, für jedes n."""
    for n in (2, 3, 5, 12):
        r = L.simulate(S.Settings(n=n, rule=C.RULE_BEST, eps=0.0, runs=100, rounds=40))
        assert (r.mins == r.mins[:, :1]).all(), n


def test_two_partners_without_trembling_swap_their_choices():
    r = L.simulate(S.Settings(n=2, rule=C.RULE_BEST, eps=0.0, runs=5, rounds=6))
    assert (r.trace[:-1, ::-1] == r.trace[1:]).all()


def test_the_rules_differ_on_at_least_one_cell():
    """Zweig-Test: keine Regel ist stillschweigend eine andere (Nullspalten-Regel)."""
    base = dict(n=4, runs=200, rounds=30, start=C.START_TOP)
    means = {name: L.simulate(S.Settings(**base, **fields)).mins[:, -1].mean() for name, fields in
             {"sum": dict(rule=C.RULE_SUM), "mean": dict(rule=C.RULE_MEAN, lam=50.0), "best": dict(rule=C.RULE_BEST, eps=0.01),
              "best0": dict(rule=C.RULE_BEST, eps=0.0), "fict": dict(rule=C.RULE_FICT)}.items()}
    assert means["best0"] == 7.0 and means["fict"] == 7.0                   # rauschfrei: bleibt oben
    assert means["sum"] < 4.0 and means["mean"] > 6.0 and 2.0 < means["best"] < 6.5   # drei verschiedene Arten, Rauschen zu haben
    assert len({round(means[k], 1) for k in ("sum", "mean", "best")}) == 3
    # fiktives Spiel und beste Antwort ohne Zittern erhalten beide das Minimum, spielen aber nicht dieselben Stufen (Gedächtnis gegen Vorrunde, bei Gleichstand der Bewertung Zufall)
    traces = [(L.simulate(S.Settings(rule=C.RULE_BEST, eps=0.0, n=5, runs=1, rounds=30, seed=k)).trace, L.simulate(S.Settings(rule=C.RULE_FICT, n=5, runs=1, rounds=30, seed=k)).trace) for k in range(1, 9)]
    assert sum(not np.array_equal(a, b) for a, b in traces) >= 1


def test_prize_and_price_and_prior_change_the_minimum():
    base = dict(n=4, runs=300, rounds=40)
    ref = L.simulate(S.Settings(**base)).mins[:, -1].mean()
    cheap = L.simulate(S.Settings(b=0.02, **base)).mins[:, -1].mean()
    prize = L.simulate(S.Settings(a=2.0, **base)).mins[:, -1].mean()
    prior = L.simulate(S.Settings(prior=10.0, **base)).mins[:, -1].mean()
    assert cheap > ref + 1.0 and prize > ref + 1.0 and prior > ref + 1.0


def test_normalisation_changes_the_run_only_when_the_prize_differs_from_the_reference():
    base = dict(n=4, runs=100, rounds=20)
    same = [L.simulate(S.Settings(a=0.2, normalize=flag, **base)).mins for flag in (True, False)]
    other = [L.simulate(S.Settings(a=1.0, normalize=flag, **base)).mins for flag in (True, False)]
    assert np.array_equal(same[0], same[1]) and not np.array_equal(other[0], other[1])


def test_prior_weight_enters_the_first_choice_after_round_one():
    r = L.simulate(S.Settings(n=3, runs=300, rounds=2, prior=60.0))
    assert r.mins[:, 0].mean() < 3.0 and r.mins[:, 1].mean() > 6.5          # Runde 1 Zufall, danach fast sicher oben


def test_fictitious_play_ignores_lambda_and_phi():
    a = L.simulate(S.Settings(n=4, rule=C.RULE_FICT, lam=1.0, phi=0.0, runs=50, rounds=20, seed=3))
    b = L.simulate(S.Settings(n=4, rule=C.RULE_FICT, lam=200.0, phi=1.0, runs=50, rounds=20, seed=3))
    assert np.array_equal(a.mins, b.mins)


def test_pool_with_team_equal_to_the_pool_is_the_single_chain():
    s = S.Settings(n=4, runs=200, rounds=15, seed=11)
    pool = L.simulate(s, team=4)
    single = L.simulate(s)
    assert np.array_equal(pool.mins, single.mins)


def test_teams_see_only_their_own_minimum():
    r = L.simulate(S.Settings(n=8, runs=100, rounds=3, start=C.START_TOP, rule=C.RULE_FICT), team=2)
    assert (r.mins == 7.0).all() and (r.top == 1.0).all()
    with pytest.raises(ValueError):
        L.simulate(S.Settings(n=8, runs=2, rounds=2), team=3)


def test_growing_group_starts_small_and_gets_to_full_size():
    s = S.Settings(n=6, runs=20, rounds=40, start=C.START_TOP, rule=C.RULE_MEAN, lam=50.0, seed=2)
    r = L.simulate(s, grow=(2, 10))
    assert (r.mins[:, 0] == 7).all()
    assert r.trace[0, 2:].tolist() == [7, 7, 7, 7]                         # noch nicht dabei: oberste Stufe
    with pytest.raises(ValueError):
        L.simulate(S.Settings(n=6, runs=2, rounds=2, rule=C.RULE_BEST), grow=(2, 5))


def test_growing_group_with_start_size_n_equals_the_fixed_group():
    s = S.Settings(n=4, runs=50, rounds=20, start=C.START_TOP, seed=8)
    assert np.array_equal(L.simulate(s).mins, L.simulate(s, grow=(4, 5)).mins)


def test_one_round_runs_and_reports_the_start():
    r = L.simulate(S.Settings(n=3, runs=7, rounds=1))
    assert r.mins.shape == (7, 1) and r.final.shape == (7, 3)
