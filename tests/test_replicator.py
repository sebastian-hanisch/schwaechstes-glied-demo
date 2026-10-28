"""Replikator: Schritt von Hand, Fixpunkte, Konvergenz, Zufallsstarts, Einzugsbereich."""

import numpy as np
import pytest

import swg_replicator as RP


def test_one_step_by_hand(repl2):
    p = np.array([repl2["p"]])
    u = RP.utilities(p, 2, 0.2, 0.1, 0.6)
    assert u[0].tolist() == pytest.approx(repl2["u"])
    q = RP.step(p, 2, 0.2, 0.1, 0.6)
    assert q[0].tolist() == pytest.approx(repl2["next"])


def test_the_interior_point_is_a_fixed_point(repl2):
    p = np.array([repl2["fixed"]])
    assert RP.step(p, 2, 0.2, 0.1, 0.6)[0].tolist() == pytest.approx(list(repl2["fixed"]))


def test_survival_function():
    p = np.array([[0.1, 0.2, 0.3, 0.4]])
    assert RP.survival(p)[0].tolist() == pytest.approx([1.0, 0.9, 0.7, 0.4])


def test_time_scale_follows_the_prize_but_the_path_depends_only_on_the_ratio():
    p = RP.random_starts(5, 3)
    a = RP.run(p, 4, 0.2, 0.05, 0.6, steps=300)[0]
    b = RP.run(p, 4, 1.0, 0.25, 0.6, steps=300)[0]                          # gleiches Verhältnis b/a = 0,25, fünffache Prämie
    assert np.allclose(a, b, atol=1e-9)


def test_every_pure_level_is_a_fixed_point_and_stable_to_small_disturbances():
    for lev in range(1, 8):
        p = np.zeros((1, 7))
        p[0, lev - 1] = 1.0
        assert RP.step(p, 6, 0.2, 0.1, 0.6)[0].tolist() == pytest.approx(p[0].tolist())
        d = np.full((1, 7), 1e-3 / 6)
        d[0, lev - 1] = 1 - 1e-3
        end = RP.run(d, 6, 0.2, 0.1, 0.6, steps=6000)[0]
        assert end[0, lev - 1] > 0.999999


def test_random_starts_are_distributions_and_reproducible():
    p = RP.random_starts(100, 7)
    assert p.shape == (100, 7) and np.allclose(p.sum(axis=1), 1.0) and (p > 0).all()
    assert np.array_equal(p, RP.random_starts(100, 7)) and not np.array_equal(p, RP.random_starts(100, 8))


def test_uniform_start_two_partners_ends_at_level_four_and_larger_groups_at_one():
    uni = np.full((1, 7), 1 / 7)
    assert RP.end_level(RP.run(uni, 2, 0.2, 0.1, 0.6)[0]).tolist() == [4]
    assert RP.end_level(RP.run(uni, 3, 0.2, 0.1, 0.6)[0]).tolist() == [1]
    assert RP.end_level(RP.run(uni, 2, 0.2, 0.02, 0.6)[0]).tolist() == [7]       # b/a = 0,1: Stufe 7


def test_end_level_marks_unconverged_rows_with_zero():
    p = np.array([[0.5, 0.5, 0, 0, 0, 0, 0], [0.995, 0.005, 0, 0, 0, 0, 0]])
    assert RP.end_level(p).tolist() == [0, 1]


def test_basin_shares_sum_to_one_and_shrink_with_the_group_size():
    cells = [RP.basin(n, 0.1, 600, 5) for n in (2, 4, 8)]
    for cell in cells:
        assert cell["none"] + sum(cell["share"]) == pytest.approx(1.0) and cell["none"] == 0.0
    top = [cell["share"][-1] for cell in cells]
    assert top[0] > top[1] > top[2] and top[0] > 0.8


def test_mutation_keeps_a_floor_under_every_level():
    uni = np.full((1, 7), 1 / 7)
    p, used = RP.run(uni, 4, 0.2, 0.1, 0.6, steps=500, mu=0.05)
    assert (p > 0.05 / 7 * 0.9).all() and used[0] == 500


def test_mean_minimum_of_the_end_state_equals_the_level_for_a_pure_end():
    cell = RP.basin(2, 0.9, 100, 1)
    assert 1.0 <= cell["emin"] <= 7.0 and cell["emin"] < 1.5          # b/a = 0,9: fast alle enden bei Stufe 1
