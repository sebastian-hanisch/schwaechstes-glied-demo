"""Jede Zahl aus der README und den Hilfetexten, nachgerechnet (DEMO-PLAYBOOK §4).

Zwei Quellen, beide plattformsicher:
* die gespeicherten Messreihen (`data/measurements.json`, 1 000 Läufe je Zelle, Erzeuger-Hash geprüft in `test_measurements.py`): die README-Zahl ist der gespeicherte Wert, gerundet;
  eine Stichprobe der Zellen wird in `test_measurements.py` mit anderem Seed innerhalb von vier Standardfehlern reproduziert;
* exakte Rechnung (Aufzählung, Formeln, lineare Systeme): Gleitkommagleichheit nur mit `approx`; Treffzeiten der schlecht konditionierten Kette (n = 3, 4) nur als Band.
Keine Zeit-Assertion, keine exakten Zufallswerte, Argmax nur dort, wo der Abstand der besten Zellen weit über dem Standardfehler liegt."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import swg_constants as C
import swg_evaluation as V
import swg_exact as X
import swg_experiments as XP
import swg_game as G
import swg_learning as L
import swg_scenario as S

ROOT = Path(__file__).resolve().parent.parent
SUM, MEAN, BR, BR0, FP = "Logit, Summe (λ = 5)", "Logit, Mittel (Λ = 50)", "Beste Antwort (ε = 0,01)", "Beste Antwort (ε = 0)", "Fiktives Spiel"
NS = [str(n) for n in C.E1_NS]


@pytest.fixture(scope="module")
def meas():
    return json.loads(XP.MEAS_FILE.read_text(encoding="utf-8"))


def r2(x):
    return round(x, 2)


def rounds(cell, *idx):
    """Mittel in den gespeicherten Runden (Index in (1, 10, 60, 150, 400))."""
    return [r2(cell["mean"][i]) for i in idx]


# --- Spiel ------------------------------------------------------------------------------------------------------------------------------------
def test_payoffs_of_the_extreme_profiles_and_the_hand_instance():
    assert round(G.payoff(7, 7, 0.2, 0.1, 0.6), 2) == 1.30 and round(G.payoff(1, 1, 0.2, 0.1, 0.6), 2) == 0.70
    assert [round(float(G.payoff(x, m, 0.2, 0.1, 0.6)), 2) for x, m in ((3, 5), (5, 3))] == [0.9, 0.7]


def test_seven_strict_symmetric_equilibria_for_two_to_four_partners():
    for n in (2, 3, 4):
        assert G.equilibria(n, 0.2, 0.1, 0.6) == [(k,) * n for k in range(1, 8)]


def test_tolerance_table_in_the_readme():
    ns = (2, 3, 4, 6, 8, 16, 32)
    assert [round(G.tolerance(n, 0.5), 3) for n in ns] == [0.500, 0.293, 0.206, 0.129, 0.094, 0.045, 0.022]
    assert [round(G.tolerance(n, 0.1), 3) for n in ns] == [0.900, 0.684, 0.536, 0.369, 0.280, 0.142, 0.072]
    assert round(G.tolerance(8, 0.05), 3) == 0.348                      # b/a = 0,05, n = 8: die Schwelle ist weit, das Lernen trotzdem langsam


def test_uniform_random_minimum_table():
    assert [round(X.uniform_expected_min(n), 3) for n in (2, 3, 4, 6, 8, 16, 32)] == [2.857, 2.286, 1.948, 1.571, 1.372, 1.090, 1.007]


def test_first_observation_makes_the_sum_version_a_tenth_as_sharp_as_in_the_end():
    """λ = 5, φ = 0,9: nach der ersten Beobachtung gilt die Schärfe λ, am Ende λ/(1−φ) = 50: ein Zehntel. Von Hand: p(7) = 1/Σ_j e^(−0,5 j) (j = 0..6) = 0,41."""
    s = S.Settings()
    assert s.lam / (s.lam / (1 - s.phi)) == pytest.approx(0.1)
    p7 = 1.0 / sum(math.exp(-0.5 * j) for j in range(7))
    assert round(p7, 2) == 0.41
    z = 5.0 * np.array([G.payoff(x, 7, 0.2, 0.1, 0.6) for x in range(1, 8)])
    assert L.logit_probs(z)[-1] == pytest.approx(p7, rel=1e-12)


# --- H1 V4-Hook ---------------------------------------------------------------------------------------------------------------------------------
def test_h1_sum_version_uniform_start_round_ten_and_sixty(meas):
    cells = meas["e1_uniform"][SUM]
    assert [r2(cells[n]["mean"][1]) for n in NS] == [3.59, 1.64, 1.18, 1.02, 1.00, 1.00, 1.00]
    assert [r2(cells[n]["mean"][2]) for n in NS] == [3.91, 1.61, 1.13, 1.00, 1.00, 1.00, 1.00]
    assert [r2(cells[n]["mean"][0]) for n in NS][:3] == [2.79, 2.26, 1.98]
    assert all(1.0 <= cells[n]["mean"][1] <= 1.2 for n in NS[2:])                       # n ≥ 4: binnen 10 Runden zwischen 1,0 und 1,2
    assert cells["3"]["mean"][1] > 1.2                                                  # Ausnahme: n = 3 liegt bei 1,64


# --- H2 Start gegen Lernen --------------------------------------------------------------------------------------------------------------------
def test_h2_round_one_equals_the_exact_random_minimum_in_all_seven_group_sizes(meas):
    cells = meas["e1_uniform"][SUM]
    for n in NS:
        assert abs(cells[n]["mean"][0] - X.uniform_expected_min(int(n))) < 4 * cells[n]["se"][0], n


def test_h2_share_of_the_fall_that_happens_at_the_start(meas):
    cells = meas["e1_uniform"][SUM]
    shares = {n: XP.start_share(7.0, cells[n]["mean"][0], cells[n]["mean"][2]) for n in NS}
    assert shares["2"] > 1.0                                                              # n = 2: das Minimum fällt nicht, es steigt (kein Anteil am Abfall)
    assert [round(100 * shares[n]) for n in NS[1:]] == [88, 86, 90, 94, 98, 100]
    assert [round(cells[n]["mean"][2], 2) for n in NS[1:]][:3] == [1.61, 1.13, 1.00]


# --- H3 Rauschen ------------------------------------------------------------------------------------------------------------------------------
def test_h3_cooperative_start_round_four_hundred_for_the_three_noisy_rules(meas):
    top = meas["e1_top"]
    assert [r2(top[SUM][n]["mean"][4]) for n in NS] == [6.17, 4.06, 2.72, 1.16, 1.00, 1.00, 1.00]
    assert [r2(top[MEAN][n]["mean"][4]) for n in NS] == [6.99, 6.95, 6.71, 4.85, 2.16, 1.00, 1.00]
    assert [r2(top[BR][n]["mean"][4]) for n in NS] == [2.93, 1.21, 1.09, 1.03, 1.01, 1.00, 1.00]


def test_h3_fourteen_of_eighteen_cells_end_below_four_and_the_four_exceptions(meas):
    top = meas["e1_top"]
    below, above = [], []
    for rule in (SUM, MEAN, BR):
        for n in NS[1:]:                                                                 # n = 3 … 32
            (below if top[rule][n]["mean"][4] < 4.0 else above).append((rule, n))
    assert len(below) == 14 and len(below) + len(above) == 18
    assert above == [(SUM, "3"), (MEAN, "3"), (MEAN, "4"), (MEAN, "6")]


def test_h3_speed_depends_on_the_version_first_measured_round_below_four_for_eight_partners(meas):
    top = meas["e1_top"]
    first = {rule: V.first_round_below(top[rule]["8"], 4.0) for rule in (SUM, MEAN, BR)}
    assert first == {SUM: 10, MEAN: 400, BR: 60}
    assert [r2(top[MEAN]["8"]["mean"][i]) for i in (1, 2, 3, 4)] == [6.70, 6.00, 4.90, 2.16]      # 60 Runden genügen nicht
    assert [r2(top[SUM]["8"]["mean"][i]) for i in (1, 2)] == [1.44, 1.15]


def test_h3_mean_version_with_uniform_start_barely_moves_for_three_and_four_partners(meas):
    cells = meas["e1_uniform"][MEAN]
    assert [r2(cells["3"]["mean"][i]) for i in (0, 4)] == [2.23, 2.19] and [r2(cells["4"]["mean"][i]) for i in (0, 4)] == [1.92, 1.80]


# --- H4 rauschfreie Regeln ---------------------------------------------------------------------------------------------------------------------
def test_h4_best_reply_without_trembling_never_changes_the_minimum(meas):
    row = meas["unchanged"][BR0]
    assert all(row[n]["unchanged"] == 1.0 and row[n]["runs"] == 1000 for n in NS)
    assert [r2(row[n]["mean_first"]) for n in NS] == [2.81, 2.38, 1.91, 1.61, 1.38, 1.10, 1.00]


def test_h4_fictitious_play_keeps_the_minimum_from_three_partners_on_but_not_for_two(meas):
    row = meas["unchanged"][FP]
    assert all(row[n]["unchanged"] == 1.0 for n in NS[1:])
    assert round(100 * row["2"]["unchanged"], 1) == 39.1 and [r2(row["2"]["mean_first"]), r2(row["2"]["mean_last"])] == [2.84, 3.92]


def test_h4_with_two_partners_best_reply_swaps_and_the_stationary_value_is_exact_two_point_eight_five_seven():
    mat, mins = X.br_chain(2, 0.01)
    assert X.stationary(mat) @ mins == pytest.approx(140 / 49, rel=1e-9) and round(140 / 49, 3) == 2.857


# --- H5 Ratsche -------------------------------------------------------------------------------------------------------------------------------
def test_h5_exact_hitting_times_of_the_chain_as_bands():
    rows = XP.e2_chain_rows(0.01)
    assert [r["n"] for r in rows] == [2, 3, 4] and [r["states"] for r in rows] == [49, 343, 2401]
    assert rows[0]["up"] == pytest.approx(328.0, rel=1e-3) and rows[1]["up"] == pytest.approx(5.27e6, rel=1e-2) and rows[2]["up"] == pytest.approx(9.38e8, rel=1e-2)
    assert [round(r["down"], 1) for r in rows] == [116.9, 78.1, 58.7]
    assert [round(r["stationary"], 4) for r in rows][:2] == [2.8571, 1.0002] and rows[2]["stationary"] < 1.001
    assert all(r["stationary"] < 1.03 for r in rows[1:])


def test_h5_going_up_takes_longer_than_going_down_by_an_order_of_magnitude_growing_with_n():
    rows = XP.e2_chain_rows(0.01)
    ratios = [r["up"] / r["down"] for r in rows]
    assert 2.5 < ratios[0] < 3.1 and ratios[1] > 1e4 and ratios[2] > 1e6 and ratios[0] < ratios[1] < ratios[2]


def test_h5_first_drop_formula_values_and_simulation_agreement(meas):
    f = meas["first_drop"]
    assert [round(r["formula"], 1) for r in f["0.01"]] == [59.6, 30.5, 16.0, 8.8, 5.2] and [round(r["formula"], 1) for r in f["0.05"]] == [12.9, 7.2, 4.4, 3.0, 2.3]
    inside = sum(1 for eps in f for r in f[eps] if abs(r["sim"] - r["formula"]) <= 4 * r["se"])
    assert inside == 10 and all(r["censored"] <= 0.001 for eps in f for r in f[eps])
    assert [round(r["sim"], 1) for r in f["0.01"]] == [58.9, 31.6, 16.3, 8.9, 5.1]


# --- H6 Einzugsbereich -----------------------------------------------------------------------------------------------------------------------
def test_h6_basin_of_the_top_level_shrinks_with_the_group_size(meas):
    top01 = [meas["basin"][f"{n}|0.1"]["share"][-1] for n in C.BASIN_NS]
    top05 = [meas["basin"][f"{n}|0.5"]["share"][-1] for n in C.BASIN_NS]
    assert [round(x, 3) for x in top01] == [0.858, 0.398, 0.148, 0.023, 0.005, 0.000, 0.000]
    assert [round(x, 3) for x in top05] == [0.075, 0.003, 0.000, 0.000, 0.000, 0.000, 0.000]
    assert all(a > b for a, b in zip(top01[:5], top01[1:6]))                              # streng fallend bis n = 8
    assert all(a >= b for a, b in zip(top01, top01[1:])) and all(a >= b for a, b in zip(top05, top05[1:]))


def test_h6_uniform_start_of_the_replicator_ends_at_four_for_two_and_at_one_from_three_partners():
    from swg_replicator import end_level, run
    uni = np.full((1, 7), 1 / 7)
    assert [int(end_level(run(uni, n, 0.2, 0.1, 0.6)[0])[0]) for n in (2, 3, 4, 8)] == [4, 1, 1, 1]
    assert [int(end_level(run(uni, n, 0.2, 0.02, 0.6)[0])[0]) for n in (2, 3)] == [7, 7]


# --- H7 Einsatzpreis -------------------------------------------------------------------------------------------------------------------------
def test_h7_price_helps_small_groups_fully_and_large_groups_only_partly(meas):
    price = meas["price"]
    assert [r2(price["2"]["0.05"]["mean"][2]), r2(price["2"]["0.02"]["mean"][2])] == [6.98, 7.00] and [round(100 * price["2"]["0.05"]["top"][i], 1) for i in (2, 3)] == [97.8, 99.8]
    assert [r2(price["4"]["0.01"]["mean"][2]), r2(price["4"]["0.02"]["mean"][2]), r2(price["4"]["0.02"]["mean"][3]), r2(price["4"]["0.05"]["mean"][2])] == [6.88, 5.57, 6.89, 2.28]
    assert [r2(price["8"]["0.01"]["mean"][2]), r2(price["8"]["0.01"]["mean"][3]), r2(price["8"]["0.02"]["mean"][2])] == [3.16, 4.93, 1.71]
    assert [r2(price["16"]["0.01"]["mean"][2]), r2(price["16"]["0.01"]["mean"][3])] == [1.12, 1.13]
    assert max(price["8"][b]["top"][3] for b in price["8"]) == 0.0 and max(price["16"][b]["top"][3] for b in price["16"]) == 0.0      # n ≥ 8: nie die oberste Stufe


# --- H8 Prämie ---------------------------------------------------------------------------------------------------------------------------------
def test_h8_premium_normalised_helps_small_groups_and_large_ones_only_partly(meas):
    norm = meas["premium_norm"]
    assert [r2(norm["2"]["2.0"]["mean"][3]), r2(norm["4"]["2.0"]["mean"][2]), r2(norm["4"]["2.0"]["mean"][3])] == [7.00, 6.88, 7.00]
    assert [r2(norm["8"]["2.0"]["mean"][2]), r2(norm["8"]["2.0"]["mean"][3]), norm["8"]["2.0"]["top"][3]] == [3.13, 4.96, 0.0]
    assert [r2(norm["16"]["2.0"]["mean"][2]), r2(norm["16"]["2.0"]["mean"][3])] == [1.11, 1.12]


def test_h8_without_normalising_the_premium_does_much_less(meas):
    raw, norm = meas["premium_raw"], meas["premium_norm"]
    assert [r2(raw["4"]["2.0"]["mean"][3]), r2(norm["4"]["2.0"]["mean"][3])] == [2.70, 7.00]
    assert [r2(raw["8"]["2.0"]["mean"][3]), r2(raw["2"]["0.3"]["mean"][3]), r2(norm["2"]["0.3"]["mean"][3])] == [1.37, 5.86, 6.75]


def test_h8_only_the_ratio_b_over_a_counts_when_the_sharpness_is_normalised():
    """a = 1, b = 0,1 (normiert) und a = 0,2, b = 0,02 sind dasselbe Spiel in anderen Einheiten: gleicher Zufall, gleiche Läufe (bis auf Gleitkommarauschen)."""
    a = L.simulate(S.Settings(n=4, a=1.0, b=0.1, runs=300, rounds=60, seed=17)).mins
    b = L.simulate(S.Settings(n=4, a=0.2, b=0.02, runs=300, rounds=60, seed=17)).mins
    assert (a == b).mean() > 0.999


def test_h8_sharpness_series_summed_version_four_partners_is_not_monotone_and_freezes(meas):
    sharp = meas["sharp_sum"]["4"]
    lams = list(sharp)
    last = [r2(sharp[k]["mean"][2]) for k in lams]
    assert lams == ["0.2", "0.5", "1.0", "2.0", "5.0", "10.0", "20.0", "50.0", "200.0"] and last == [1.64, 1.23, 1.02, 1.00, 1.13, 1.49, 1.78, 1.98, 1.92]
    best = lams[int(np.argmin(last))]
    assert best == "2.0"                                                                   # Minimum der Reihe bei λ = 2, danach steigt sie wieder (Einfrieren am Startwert)
    for k in ("50.0", "200.0"):
        assert abs(sharp[k]["mean"][2] - sharp[k]["mean"][0]) < 4 * math.hypot(sharp[k]["se"][0], sharp[k]["se"][2])
    two = meas["sharp_sum"]["2"]
    assert [r2(two[k]["mean"][2]) for k in ("10.0", "50.0", "200.0")] == [4.04, 4.04, 2.90] and max(c["mean"][2] for c in two.values()) < 4.05


def test_h8_cooperative_start_keeps_the_top_with_sharp_learners(meas):
    sharp = meas["sharp_sum_top"]["4"]
    assert [r2(sharp[k]["mean"][2]) for k in ("1.0", "5.0", "10.0", "20.0", "50.0", "200.0")] == [1.02, 2.80, 5.21, 6.48, 6.97, 7.00]
    mean_top = meas["sharp_mean_top"]["8"]
    assert [r2(mean_top[k]["mean"][3]) for k in ("2.0", "5.0", "10.0")] == [1.00, 2.10, 7.00]


def test_h8_forgetting_delays_the_fall_in_the_mean_version(meas):
    phi = meas["phi_mean_top"]["8"]
    assert [r2(phi[k]["mean"][3]) for k in ("0.0", "0.5", "0.8", "0.9", "0.95", "1.0")] == [1.00, 1.00, 1.06, 2.20, 4.34, 6.50]
    assert all(phi[a]["mean"][3] <= phi[b]["mean"][3] + 1e-9 for a, b in zip(list(phi), list(phi)[1:]))      # monoton in φ
    four = meas["phi_mean_top"]["4"]
    assert r2(four["1.0"]["mean"][3]) == 6.92 and r2(four["0.0"]["mean"][3]) == 1.08
    uni = meas["phi_mean_uniform"]["4"]
    assert [r2(uni["0.0"]["mean"][2]), r2(uni["1.0"]["mean"][2])] == [1.39, 1.99]


# --- H9 Vorabsprache -------------------------------------------------------------------------------------------------------------------------
def test_h9_prior_fades_for_larger_groups(meas):
    prior = meas["prior"]
    assert [r2(prior["8"]["30"]["mean"][i]) for i in (1, 2, 3)] == [7.00, 6.70, 2.62]
    assert [r2(prior["16"]["30"]["mean"][i]) for i in (1, 2, 3)] == [7.00, 5.79, 1.00]
    assert [r2(prior["4"]["30"]["mean"][i]) for i in (1, 3)] == [7.00, 6.79] and round(100 * prior["4"]["30"]["top"][3]) == 81
    assert [r2(prior["2"]["5"]["mean"][2]), round(100 * prior["2"]["5"]["top"][2], 1)] == [6.99, 98.5]
    assert [r2(prior["8"]["0"]["mean"][3]), r2(prior["16"]["0"]["mean"][3])] == [1.00, 1.00]


# --- H10 Prämie und Vorabsprache -------------------------------------------------------------------------------------------------------------
def test_h10_premium_and_prior_together_hold_sixteen_partners_to_round_four_hundred(meas):
    combo = meas["combo"]
    assert [r2(combo["8"]["1.0|5"]["mean"][i]) for i in (1, 2, 3)] == [6.99, 7.00, 7.00] and round(100 * combo["8"]["1.0|5"]["top"][3], 1) == 99.9
    assert [r2(combo["16"]["1.0|5"]["mean"][i]) for i in (1, 2, 3)] == [6.91, 6.95, 6.95] and round(100 * combo["16"]["1.0|5"]["top"][3], 1) == 94.6
    assert [r2(combo["16"]["1.0|0"]["mean"][3]), r2(combo["16"]["0.2|5"]["mean"][3]), r2(combo["16"]["0.2|0"]["mean"][3])] == [1.03, 1.00, 1.00]
    assert [r2(combo["8"]["1.0|0"]["mean"][3]), r2(combo["8"]["0.2|5"]["mean"][3])] == [1.66, 1.26]
    assert round(100 * combo["16"]["1.0|30"]["top"][3]) == 100


# --- H11 wachsende Gruppen --------------------------------------------------------------------------------------------------------------------
def test_h11_growth_delays_the_fall_but_does_not_hold_the_group(meas):
    g = meas["grow"]
    assert [r2(g["8|sum"]["fest"]["mean"][i]) for i in range(5)] == [7.00, 1.42, 1.14, 1.04, 1.00]
    assert [r2(g["8|sum"]["alle 10"]["mean"][i]) for i in range(5)] == [7.00, 5.85, 5.44, 4.36, 1.87]
    assert [r2(g["8|sum"]["alle 25"]["mean"][i]) for i in range(5)] == [7.00, 5.80, 6.04, 5.65, 2.67]
    assert [r2(g["8|mean"]["fest"]["mean"][i]) for i in range(5)] == [7.00, 6.72, 6.11, 4.96, 2.14]
    assert [r2(g["8|mean"]["alle 25"]["mean"][i]) for i in range(5)] == [7.00, 6.98, 6.97, 6.56, 3.54]
    assert [r2(g["16|sum"]["alle 25"]["mean"][i]) for i in (3, 4)] == [5.75, 1.10] and [r2(g["16|mean"]["alle 25"]["mean"][i]) for i in (3, 4)] == [6.55, 1.19]
    assert [r2(g[f"16|{c}"][k]["mean"][4]) for c in ("sum", "mean") for k in ("fest", "alle 10")] == [1.00, 1.00, 1.00, 1.00]
    later, better, cells_total = [], 0, 0
    for key, cells in g.items():
        ref = V.first_round_below(cells["fest"], 4.0) or 10 ** 9
        for k in ("alle 10", "alle 25"):
            cells_total += 1
            if (V.first_round_below(cells[k], 4.0) or 10 ** 9) > ref:
                later.append((key, k))
            better += V.classify(cells["fest"]["mean"][4], cells["fest"]["se"][4], cells[k]["mean"][4], cells[k]["se"][4]) > 0
    assert cells_total == 8 and len(later) == 6 and better == 6                          # gemessene Runden 1, 10, 60, 150, 400
    assert {key for key in g} - {key for key, _ in later} == {"8|mean"}                  # n = 8, Mittel: alle drei Gruppen fallen erst nach Runde 150 unter 4


# --- H12 Teams --------------------------------------------------------------------------------------------------------------------------------
def test_h12_re_drawn_teams_are_not_better_than_fixed_teams(meas):
    teams = meas["teams"]
    fixed = [r2(teams[f"{k}|1"]["mean"][2]) for k in C.POOL_TEAMS]
    drawn = [r2(teams[f"{k}|0"]["mean"][2]) for k in C.POOL_TEAMS]
    assert fixed == [3.99, 1.14, 1.00, 1.00] and drawn == [4.00, 1.00, 1.00, 1.00]
    assert round(100 * teams["2|1"]["top"][2], 1) == 2.8 and teams["2|0"]["top"][2] == 0.0
    for k in C.POOL_TEAMS:
        f, d = teams[f"{k}|1"], teams[f"{k}|0"]
        assert V.classify(f["mean"][2], f["se"][2], d["mean"][2], d["se"][2]) <= 0                    # gelost nie besser (bei 4 Teamgrößen)


# --- Plan-Zahlen, die sich nicht reproduzieren oder anders ausfallen ---------------------------------------------------------------------------
def test_corrections_against_the_plan(meas):
    uni = meas["e1_uniform"][SUM]
    assert [r2(uni["2"]["mean"][1]), r2(uni["2"]["mean"][2])] == [3.59, 3.91]           # der Plan: 3,75 / 4,04 (4 Seeds × 250 Läufe)
    assert [round(meas["basin"][f"{n}|0.1"]["share"][-1], 3) for n in (2, 3, 4, 6, 8)] == [0.858, 0.398, 0.148, 0.023, 0.005]         # der Plan: 0,863 / 0,406 / 0,158 / 0,028 / 0,007


# --- Hilfetexte der Presets -----------------------------------------------------------------------------------------------------------------------
def test_preset_help_numbers_are_the_stored_measurements(meas):
    help_ = C.PRESET_HELP
    e1u, e1t = meas["e1_uniform"], meas["e1_top"]
    de = V.de
    need = {
        "Standard (V4-Hook, n = 8)": [de(e1u[SUM]["8"]["mean"][0]), de(e1u[SUM]["8"]["mean"][2])],
        "Kooperativer Start, Summe": [de(e1t[SUM]["8"]["mean"][1]), de(e1t[SUM]["8"]["mean"][2])],
        "Kooperativer Start, Mittel": [de(e1t[MEAN]["8"]["mean"][i]) for i in (2, 3, 4)],
        "Nur zwei": [de(e1u[SUM]["2"]["mean"][0]), de(X.uniform_expected_min(2)), de(e1u[SUM]["2"]["mean"][2])],
        "Ohne Rauschen": [de(meas["unchanged"][BR0]["8"]["mean_first"])],
        "Billiger Einsatz": [de(meas["price"]["4"]["0.02"]["mean"][2]), de(meas["price"]["4"]["0.02"]["mean"][3]), de(meas["price"]["8"]["0.02"]["mean"][2])],
        "Prämie": [de(meas["premium_norm"]["8"]["2.0"]["mean"][2]), de(meas["premium_norm"]["8"]["2.0"]["mean"][3]), de(meas["premium_raw"]["8"]["2.0"]["mean"][3])],
        "Vorabsprache verblasst": [de(meas["prior"]["8"]["30"]["mean"][i]) for i in (1, 2, 3)],
        "Prämie plus Vorabsprache": [de(meas["combo"]["16"]["1.0|5"]["mean"][3]), V.pct(meas["combo"]["16"]["1.0|5"]["top"][3]), de(meas["combo"]["16"]["1.0|0"]["mean"][3]), de(meas["combo"]["16"]["0.2|5"]["mean"][3])],
    }
    assert set(need) == set(help_)
    for name, numbers in need.items():
        for number in numbers:
            assert number in help_[name], (name, number, help_[name])


def test_the_parameters_named_in_the_preset_help_texts_are_the_preset_values():
    """Die Parameter-Zahlen der Hilfetexte (b, a, s, n) stehen in den Presets; die Messwerte prüft der Test oben."""
    assert "b = 0,02" in C.PRESET_HELP["Billiger Einsatz"] and C.PRESETS["Billiger Einsatz"]["b"] == 0.02 and C.PRESETS["Billiger Einsatz"]["n"] == 4
    assert "a = 2,0" in C.PRESET_HELP["Prämie"] and C.PRESETS["Prämie"]["a"] == 2.0
    assert "s = 30" in C.PRESET_HELP["Vorabsprache verblasst"] and C.PRESETS["Vorabsprache verblasst"]["prior"] == 30
    assert "a = 1,0 und Vorabsprache s = 5" in C.PRESET_HELP["Prämie plus Vorabsprache"] and C.PRESETS["Prämie plus Vorabsprache"]["a"] == 1.0 and C.PRESETS["Prämie plus Vorabsprache"]["prior"] == 5


def test_h8_h11_small_additions_named_in_the_readme(meas):
    sharp = meas["sharp_sum"]["4"]
    assert min(c["mean"][0] for c in sharp.values()) > 1.91 and max(c["mean"][0] for c in sharp.values()) < 2.02          # Start 1,92 bis 2,01
    g = meas["grow"]
    assert max(g[f"16|{c}"][k]["mean"][4] for c in ("sum", "mean") for k in ("fest", "alle 10", "alle 25")) < 1.195


def test_the_readme_contains_the_measured_numbers_it_cites(meas):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    de = V.de
    u, t = meas["e1_uniform"][SUM], meas["e1_top"]
    wanted = [de(u[n]["mean"][1]) for n in NS] + [de(u["2"]["mean"][0]), de(u["2"]["mean"][2])]
    wanted += [de(X.uniform_expected_min(int(n)), 3) for n in NS]
    wanted += [de(t[SUM]["3"]["mean"][4]), de(t[MEAN]["3"]["mean"][4]), de(t[MEAN]["4"]["mean"][4]), de(t[MEAN]["6"]["mean"][4]), de(t[MEAN]["8"]["mean"][4])]
    wanted += [de(v) for v in (meas["premium_norm"]["8"]["2.0"]["mean"][2], meas["premium_norm"]["8"]["2.0"]["mean"][3], meas["premium_raw"]["4"]["2.0"]["mean"][3], meas["prior"]["8"]["30"]["mean"][3])]
    wanted += [V.pct(meas["combo"]["16"]["1.0|5"]["top"][3]), V.pct(meas["unchanged"][FP]["2"]["unchanged"]), de(meas["unchanged"][FP]["2"]["mean_last"])]
    wanted += [V.pct(meas["basin"][f"{n}|0.1"]["share"][-1]).replace(" %", "") for n in C.BASIN_NS[:5]]
    missing = [w for w in wanted if w not in readme]
    assert not missing, missing
    assert "Fünftes Stück der Reihe" in readme
