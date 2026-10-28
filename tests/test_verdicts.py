"""Verdikte als reine Funktionen (DEMO-PLAYBOOK §12, Punkt 2): Art, jede Zahl im Text und jeder Zweig an handgesetzten Werten; Formate."""

import numpy as np
import pytest

import swg_constants as C
import swg_evaluation as V
import swg_scenario as S


def mv(settings=None, *, mean=(1.4, 1.0), unchanged=0.0, diff=-0.4, diff_se=0.01, runs=200, random_min=1.37, hist=None, pay_end=0.7, tol=0.094):
    s = settings or S.Settings(rounds=len(mean), runs=runs)
    mean = np.array(mean, dtype=float)
    return V.MainView(settings=s, runs_used=runs, mean_min=mean, se_min=np.full(len(mean), 0.01), q25=mean, q75=mean, unchanged=unchanged, diff=diff, diff_se=diff_se, random_min=random_min,
                      hist=np.array(hist if hist is not None else [1.0, 0, 0, 0, 0, 0, 0]), pay_end=pay_end, pay_top=1.3, pay_bottom=0.7, trace=np.zeros((len(mean), s.n), dtype=int), tolerance=tol)


def cell(mean, se, top=None, rounds=(1, 10, 60)):
    return {"rounds": list(rounds), "runs": 1000, "mean": list(mean), "se": list(se), "top": list(top if top is not None else [0.0] * len(mean))}


# --- Formate ---------------------------------------------------------------------------------------------------------------------------------
def test_de_thousands_pct_big():
    assert V.de(1.5, 1) == "1,5" and V.de(3.14159, 3) == "3,142" and V.de(7, 0) == "7"
    assert V.de(-0.91, 3) == "−0,910" and V.de(-0.0001, 3) == "0,000"
    assert V.thousands(1234567) == "1.234.567" and V.thousands(999.6) == "1.000" and V.thousands(0) == "0"
    assert V.pct(0.0943) == "9,4 %" and V.pct(0.0, 0) == "0 %" and V.pct(1.0) == "100,0 %"
    assert V.big(5.269e6) == "5,3·10^6" and V.big(116.9) == "117" and V.big(78.11) == "78,1"


def test_ns_text_and_rule_label():
    assert V.ns_text([3, 4]) == "n = 3, 4" and V.ns_text([]) == "keine"
    assert V.rule_label(S.Settings(rule=C.RULE_BEST, eps=0.01)) == "Beste Antwort mit Zittern, ε = 0,010"
    assert V.rule_label(S.Settings(rule=C.RULE_FICT)) == "Fiktives Spiel"
    assert V.rule_label(S.Settings(lam=5.0, phi=0.9)) == "Logit, Summe (V4-Fassung), Schärfe 5,0, Vergessen φ = 0,90"


def test_payoff_level_is_a_minus_b_times_level_plus_c():
    assert V.payoff_level(7, 0.2, 0.1, 0.6) == pytest.approx(1.3) and V.payoff_level(1, 0.2, 0.1, 0.6) == pytest.approx(0.7)


# --- Hauptansicht ----------------------------------------------------------------------------------------------------------------------------
def test_start_or_learning_one_round():
    kind, text = V.verdict_start_or_learning(mv(mean=(2.9,), settings=S.Settings(rounds=1)))
    assert kind == "info" and "Nur Runde 1" in text and "2,90" in text and "1,37" in text


def test_start_or_learning_nothing_moves():
    kind, text = V.verdict_start_or_learning(mv(mean=(1.33, 1.33), unchanged=1.0, diff=0.0, diff_se=0.0, settings=S.Settings(rule=C.RULE_BEST, eps=0.0, rounds=2, runs=200)))
    assert kind == "info" and text.startswith("**Das Minimum bewegt sich nicht.**") and "In allen 200 Läufen" in text and "Mittel 1,33" in text and "kein Rauschen" in text


def test_start_or_learning_nothing_moves_with_a_noisy_rule_does_not_claim_the_rule_has_no_noise():
    text = V.verdict_start_or_learning(mv(mean=(7.0, 7.0), unchanged=1.0, diff=0.0, diff_se=0.0, settings=S.Settings(rule=C.RULE_MEAN, lam=200.0, start=C.START_TOP, rounds=2, runs=200)))[1]
    assert text.startswith("**Das Minimum bewegt sich nicht.**") and "so scharf" in text and "kein Rauschen" not in text


def test_start_or_learning_no_visible_difference():
    kind, text = V.verdict_start_or_learning(mv(mean=(2.0, 2.01), unchanged=0.4, diff=0.01, diff_se=0.05))
    assert kind == "info" and "Kein erkennbarer Unterschied" in text and "60 % der Läufe unterscheiden sich" in text


def test_start_or_learning_rise():
    kind, text = V.verdict_start_or_learning(mv(mean=(2.86, 4.0), diff=1.14, diff_se=0.05, settings=S.Settings(n=2, rounds=2, runs=200)))
    assert kind == "success" and text.startswith("**Lernen hebt das Minimum:**") and "von 2,86 in Runde 1 auf 4,00 in Runde 2" in text


def test_start_or_learning_fall_from_a_cooperative_start_is_all_learning():
    kind, text = V.verdict_start_or_learning(mv(mean=(7.0, 1.1), diff=-5.9, settings=S.Settings(start=C.START_TOP, rounds=2, runs=200)))
    assert kind == "warning" and "vom Start 7 (Runde 1) auf 1,10" in text and "Der ganze Abfall ist Lernen" in text


def test_start_or_learning_fall_from_a_random_start_splits_the_distance():
    kind, text = V.verdict_start_or_learning(mv(mean=(1.4, 1.0), diff=-0.4))
    # Strecke 7 → 1,0 = 6,0; davon in Runde 1 schon 5,6 (93 %)
    assert kind == "warning" and "überwiegend der Start" in text and "93 % der Strecke sind Zufall, 7 % kommen durch das Lernen" in text and "exakt 1,37" in text


def test_start_or_learning_fall_with_learning_as_main_part():
    kind, text = V.verdict_start_or_learning(mv(mean=(6.0, 1.0), diff=-5.0, random_min=6.0, settings=S.Settings(n=2, rounds=2, runs=200)))
    # Strecke 7 → 1,0 = 6,0; davon in Runde 1 nur 1,0 (17 %)
    assert kind == "warning" and "überwiegend das Lernen" in text and "17 % der Strecke sind Zufall, 83 % kommen durch das Lernen" in text


def test_start_or_learning_half_and_half_counts_as_start():
    text = V.verdict_start_or_learning(mv(mean=(4.0, 1.0), diff=-3.0, random_min=4.0, settings=S.Settings(n=2, rounds=2, runs=200)))[1]
    assert "überwiegend der Start" in text and "50 % der Strecke sind Zufall, 50 % kommen durch das Lernen" in text


def test_a_peak_in_the_middle_of_the_run_is_named_and_a_monotone_run_has_none():
    peak = V.verdict_start_or_learning(mv(mean=(1.4, 7.0, 6.0, 2.6), diff=1.2, diff_se=0.05, settings=S.Settings(n=8, rounds=4, runs=200)))[1]
    assert "Zwischendurch erreicht das Mittel 7,00 (Runde 2) und fällt danach wieder auf 2,60." in peak
    flat = V.verdict_start_or_learning(mv(mean=(1.4, 2.0, 2.6), diff=1.2, diff_se=0.05, settings=S.Settings(n=8, rounds=3, runs=200)))[1]
    assert "Zwischendurch" not in flat
    assert V.peak_note(mv(mean=(1.0, 2.0))) == "" and V.peak_note(mv(mean=(3.0, 3.0, 3.0))) == ""
    assert V.peak_note(mv(mean=(5.0, 5.01, 1.0))) == ""                     # Gipfel nicht über 4 Standardfehler (0,01) hinaus: 0,01 < 4 · 0,014


def test_payoff_verdict_numbers():
    kind, text = V.verdict_payoff(mv(pay_end=1.0))
    assert kind == "info" and "im Mittel 1,000 je Partner" in text and "Stufe 1 bringen 0,70" in text and "Stufe 7 1,30" in text and "50 % dieses Abstands" in text


def test_tolerance_verdict_names_n_minus_one_others():
    kind, text = V.verdict_tolerance(mv(tol=0.0943))
    assert kind == "info" and "7 anderen" in text and "δ* = 9,4 %" in text and "b/a = 0,50" in text


# --- E1 --------------------------------------------------------------------------------------------------------------------------------------
def test_e1_counts_falls_stays_and_rises_and_names_the_exceptions():
    cells = {2: cell((2.86, 3.7, 4.0), (0.05, 0.05, 0.05)), 4: cell((1.95, 1.2, 1.1), (0.05, 0.02, 0.02)), 8: cell((1.35, 1.0, 1.0), (0.04, 0.0, 0.0)), 16: cell((1.09, 1.09, 1.09), (0.03, 0.03, 0.03))}
    kind, text = V.verdict_e1(cells, "Logit, Summe", C.START_UNIFORM)
    assert kind == "warning"
    assert "fällt das Minimum bei 2 (n = 4, 8), bleibt bei 1 (n = 16) und steigt bei 1 (n = 2)" in text
    # Anteil des Starts: n = 4: (7 − 1,95)/(7 − 1,1) = 85,6 %, n = 8: (7 − 1,35)/6 = 94,2 %
    assert "zwischen 86 % (n = 4) und 94 % (n = 8)" in text


def test_e1_without_a_fall_is_info():
    cells = {3: cell((7, 7, 7), (0, 0, 0)), 8: cell((7, 7, 7), (0, 0, 0))}
    kind, text = V.verdict_e1(cells, "Fiktives Spiel", C.START_TOP)
    assert kind == "info" and "fällt das Minimum bei 0 (keine), bleibt bei 2 (n = 3, 8)" in text


# --- E2 --------------------------------------------------------------------------------------------------------------------------------------
def test_e2_counts_how_many_group_sizes_match_the_formula():
    chain = [{"n": 2, "up": 328.0, "down": 117.0}, {"n": 3, "up": 5.27e6, "down": 78.1}]
    drops = [{"n": 2, "formula": 59.6, "sim": 58.4, "se": 1.9, "censored": 0.0}, {"n": 8, "formula": 16.0, "sim": 18.1, "se": 0.3, "censored": 0.0}, {"n": 4, "formula": 30.5, "sim": 31.0, "se": 1.0, "censored": 0.5}]
    kind, text = V.verdict_e2(chain, drops, 0.01)
    assert kind == "info" and "n = 2: 328 Runden hinauf, 117 hinab" in text and "n = 3: 5,3·10^6 Runden hinauf, 78,1 hinab" in text
    assert "innerhalb von 4 Standardfehlern der Formel" in text and "in 1 von 3 Gruppengrößen" in text and "(nicht bei n = 8, 4)" in text and "2,8-mal (n = 2)" in text


# --- E3 --------------------------------------------------------------------------------------------------------------------------------------
def test_e3_says_whether_the_share_shrinks_in_every_step():
    def basin(top):
        return {"starts": 5000, "share": [0, 0, 0, 0, 0, 1 - top, top]}
    kind, text = V.verdict_e3({2: basin(0.86), 4: basin(0.15), 8: basin(0.01)}, 0.1)
    assert kind == "info" and "n = 2: 86,0 %, n = 4: 15,0 %, n = 8: 1,0 %" in text and "sinkt mit der Gruppengröße in jedem Schritt" in text and "5.000 Zufallsstarts" in text
    assert "nicht in jedem Schritt" in V.verdict_e3({2: basin(0.2), 4: basin(0.3)}, 0.1)[1]


# --- Reihen ----------------------------------------------------------------------------------------------------------------------------------
def test_is_flat_flags_constant_columns():
    assert V.is_flat([1.0, 1.0, 1.0]) and not V.is_flat([1.0, 1.0, 1.1]) and not V.is_flat([])


def test_e4_sharpness_reports_peak_monotonicity_and_freezing():
    cells = {1.0: cell((1.9, 1.3, 1.0), (0.05, 0.03, 0.0)), 5.0: cell((1.9, 1.2, 1.1), (0.05, 0.03, 0.02)), 50.0: cell((2.0, 1.98, 1.98), (0.05, 0.05, 0.05))}
    kind, text = V.verdict_e4_sharpness(cells, 4, C.START_UNIFORM, "sum")
    assert kind == "info" and "zwischen 1,00 und 1,98, am höchsten bei Schärfe 50,0" in text and "die Reihe ist monoton in der Schärfe" in text
    assert "Bei Schärfe 50,0 ist das Minimum am Ende nicht von dem in Runde 1 zu unterscheiden" in text
    top = {5.0: cell((7, 6.0, 5.9), (0, 0.05, 0.05)), 50.0: cell((7, 7, 7), (0, 0, 0))}
    text2 = V.verdict_e4_sharpness(top, 4, C.START_TOP, "mean")[1]
    assert "Bei Schärfe 50,0 bleibt das Minimum in der letzten Runde über 6,9" in text2 and "Einfrieren" not in text2


def test_e4_forgetting_names_best_and_worst():
    cells = {0.0: cell((7, 3, 1.0), (0, 0.1, 0.0)), 1.0: cell((7, 6.5, 6.0), (0, 0.1, 0.1))}
    text = V.verdict_e4_forgetting(cells, 8, C.START_TOP)[1]
    assert "φ = 0,00: 1,00, φ = 1,00: 6,00" in text and "höchste Minimum hat φ = 1,00, das niedrigste φ = 0,00" in text


def test_e5_price_branches():
    cells = {0.01: cell((1.3, 2.0, 5.0), (0.01, 0.05, 0.1), top=[0, 0, 0.25]), 0.1: cell((1.3, 1.0, 1.0), (0.01, 0, 0))}
    text = V.verdict_e5_price(cells, 8)[1]
    assert "b = 0,01: 5,00, b = 0,10: 1,00" in text and "bis 25,0 % (b = 0,01)" in text
    flat = V.verdict_e5_price({0.01: cell((1.3, 1.0, 1.0), (0, 0, 0))}, 8)[1]
    assert "In keinem Preis erreicht ein Lauf" in flat and "bis" not in flat


def test_e5_premium_counts_premiums_where_normalising_is_better():
    norm = {0.2: cell((1, 1, 1.0), (0, 0, 0)), 2.0: cell((1, 5, 7.0), (0, 0.1, 0.0))}
    raw = {0.2: cell((1, 1, 1.0), (0, 0, 0)), 2.0: cell((1, 2, 2.0), (0, 0.1, 0.1))}
    text = V.verdict_e5_premium(norm, raw, 4)[1]
    assert "a = 0,2: 1,00, a = 2,0: 7,00" in text and "bei 1 von 2 Prämien über der nicht normierten (a = 2,0)" in text


def test_e6_teams_compares_fixed_and_drawn_teams():
    cells = {"2|1": cell((2.9, 3.6, 4.0), (0.05, 0.05, 0.05)), "2|0": cell((2.9, 3.5, 4.0), (0.05, 0.05, 0.05)), "4|1": cell((2, 1.2, 1.14), (0.05, 0.02, 0.02)), "4|0": cell((2, 1.0, 1.0), (0.05, 0, 0))}
    text = V.verdict_e6_teams(cells)[1]
    assert "k = 2: fest 4,00, gelost 4,00; k = 4: fest 1,14, gelost 1,00" in text and "Jede Runde neu gelöste Teams sind bei 0 von 2 Teamgrößen besser als feste (keine)." in text
    cells["8|1"], cells["8|0"] = cell((1.4, 1.5, 2.0), (0.05, 0.05, 0.05)), cell((1.4, 2.5, 3.0), (0.05, 0.05, 0.05))
    assert "bei 1 von 3 Teamgrößen besser als feste (k = 8)." in V.verdict_e6_teams(cells)[1]


def test_e6_prior_and_combo_texts():
    prior = {0: cell((2, 1.2, 1.0, 1.0), (0, 0, 0, 0), rounds=(1, 10, 60, 400)), 30: cell((2, 7.0, 6.7, 2.6), (0, 0, 0.1, 0.1), rounds=(1, 10, 60, 400))}
    text = V.verdict_e6_prior(prior, 8)[1]
    assert "s = 30: Minimum in Runde 10: 7,00, Runde 60: 6,70, Runde 400: 2,60" in text and "(s = 0) in Runde 400: 1,00" in text
    combo = {"1.0|5": cell((1, 7, 6.9, 6.9), (0, 0, 0, 0), rounds=(1, 10, 60, 400))}
    assert "a = 1,0, s = 5: 6,90" in V.verdict_e6_combo(combo, 16)[1]


def test_e6_grow_reports_delay_and_final_comparison():
    rounds = (1, 10, 60, 150, 400)
    cells = {"fest": cell((7, 1.4, 1.1, 1.0, 1.0), (0, 0.05, 0.02, 0, 0), rounds=rounds), "alle 10": cell((7, 5.9, 5.4, 4.3, 1.9), (0, 0.05, 0.05, 0.05, 0.05), rounds=rounds),
             "alle 25": cell((7, 5.9, 6.1, 5.7, 1.0), (0, 0.05, 0.05, 0.05, 0.0), rounds=rounds)}
    text = V.verdict_e6_grow(cells, 8, "sum")[1]
    assert "fest: 7,00 / 1,40 / 1,10 / 1,00 / 1,00 (unter 4 ab Runde 10)" in text and "alle 10: 7,00 / 5,90 / 5,40 / 4,30 / 1,90 (unter 4 ab Runde 400)" in text
    assert "kommt bei 2 von 2 Taktungen später" in text and "in der letzten Runde ist das Wachsen bei 1 von 2 Taktungen besser" in text
    assert V.first_round_below(cells["fest"], 4.0) == 10 and V.first_round_below(cell((7, 7), (0, 0), rounds=(1, 10)), 4.0) is None
