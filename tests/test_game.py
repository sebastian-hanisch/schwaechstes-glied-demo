"""Das Spiel: Auszahlung, Minimum der anderen, Gleichgewichte, beste Antwort und Fehlertoleranz an den Mini-Instanzen von Hand."""

import numpy as np
import pytest

import swg_game as G


def test_payoff_of_two_partners_by_hand(pay2):
    excl = G.excl_min(np.array(pay2["profile"]))
    assert tuple(excl) == pay2["excl"] and min(pay2["profile"]) == pay2["min"]
    got = [G.payoff(x, m, 0.2, 0.1, 0.6) for x, m in zip(pay2["profile"], excl)]
    assert got == pytest.approx(pay2["pay"])


def test_payoff_of_four_partners_at_the_top_and_the_bottom(pay4):
    assert G.payoff(7, 7, 0.2, 0.1, 0.6) == pytest.approx(pay4["all7"])
    assert G.payoff(1, 1, 0.2, 0.1, 0.6) == pytest.approx(pay4["all1"])


def test_payoff_table_has_the_payoff_in_every_cell():
    t = G.payoff_table(0.2, 0.1, 0.6)
    assert t.shape == (7, 7)
    for x in range(1, 8):
        for m in range(1, 8):
            assert t[x - 1, m - 1] == pytest.approx(0.2 * min(x, m) - 0.1 * x + 0.6)


def test_excl_min_gives_the_second_smallest_only_to_a_sole_minimum_holder():
    assert tuple(G.excl_min(np.array([2, 5, 5]))) == (5, 2, 2)
    assert tuple(G.excl_min(np.array([2, 2, 5]))) == (2, 2, 2)          # Gleichstand im Minimum: jeder sieht 2
    assert tuple(G.excl_min(np.array([4, 4]))) == (4, 4)
    batch = G.excl_min(np.array([[3, 5], [1, 1]]))
    assert batch.tolist() == [[5, 3], [1, 1]]


def test_the_best_reply_to_m_is_m_when_the_prize_exceeds_the_price():
    assert G.best_reply_table(0.2, 0.1, 0.6).tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert G.best_reply_table(0.1, 0.2, 0.6).tolist() == [1] * 7            # Einsatz teurer als die Prämie: so tief wie möglich
    assert G.best_reply_table(0.2, 0.2, 0.6).tolist() == [1, 2, 3, 4, 5, 6, 7]   # Gleichstand: die höhere Stufe


def test_there_are_exactly_seven_symmetric_strict_equilibria_for_two_to_four_partners():
    for n in (2, 3, 4):
        found = G.equilibria(n, 0.2, 0.1, 0.6)
        assert found == [(k,) * n for k in range(1, 8)]
        assert all(G.is_strict(n, prof, 0.2, 0.1, 0.6) for prof in found)


def test_when_the_price_exceeds_the_prize_only_the_lowest_level_is_an_equilibrium():
    assert G.equilibria(3, 0.1, 0.2, 0.6) == [(1, 1, 1)]


def test_equilibria_of_a_small_game_include_no_mixed_levels():
    assert G.equilibria(2, 0.2, 0.1, 0.6, levels=3) == [(1, 1), (2, 2), (3, 3)]


def test_tolerance_by_hand(tol3):
    assert G.tolerance(tol3["n"], tol3["ratio"]) == pytest.approx(tol3["delta"])
    assert G.tolerance(2, 0.5) == pytest.approx(0.5)
    assert G.tolerance(2, 0.1) == pytest.approx(0.9)


def test_tolerance_shrinks_with_the_group_size():
    values = [G.tolerance(n, 0.5) for n in (2, 3, 4, 6, 8, 16, 32)]
    assert all(x > y for x, y in zip(values, values[1:]))
    assert [round(v, 3) for v in values] == [0.5, 0.293, 0.206, 0.129, 0.094, 0.045, 0.022]


def test_marginal_gain_is_zero_exactly_at_the_tolerance():
    for n in (2, 3, 8):
        delta = G.tolerance(n, 0.25)
        assert G.marginal_gain(1 - delta, n, 0.2, 0.05) == pytest.approx(0.0, abs=1e-12)
        assert G.marginal_gain(1 - delta + 0.01, n, 0.2, 0.05) > 0 > G.marginal_gain(1 - delta - 0.01, n, 0.2, 0.05)


def test_margin_of_the_two_partner_example(margin2):
    t = G.payoff_table(0.2, 0.1, 0.6)
    assert t[margin2["to"] - 1, margin2["m"] - 1] - t[margin2["from"] - 1, margin2["m"] - 1] == pytest.approx(margin2["gain"])
