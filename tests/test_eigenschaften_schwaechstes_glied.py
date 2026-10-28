"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2, metamorph): Aussagen über das Verfahren, die ohne Kenntnis des richtigen Ergebnisses gelten müssen.

* Das Grundniveau c verschiebt alle Stufen gleich: weder die Wahl der Lernregeln noch das Ergebnis ändern sich (bis auf Gleitkommarauschen in seltenen Grenzfällen der Ziehung).
* Prämie und Einsatzpreis zusammen mit k > 0 multipliziert (Schärfe normiert) ändern nichts: nur b/a zählt.
* Umbenennen der Partner permutiert das Minimum der anderen mit; die Zahl der Gleichgewichte hängt nicht an der Reihenfolge.
* Verschieben von Wahrscheinlichkeit nach oben hebt E[min] (stochastische Dominanz).
* Gleicher Zufalls-Seed gibt in einem zweiten Prozess (andere Hash-Randomisierung) dieselbe Ausgabe."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import swg_constants as C
import swg_exact as X
import swg_game as G
import swg_learning as L
import swg_replicator as RP
import swg_scenario as S
from swg_rng import uniform_block

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("rule,extra", [(C.RULE_SUM, {}), (C.RULE_MEAN, {"lam": 50.0}), (C.RULE_BEST, {"eps": 0.05}), (C.RULE_FICT, {})])
def test_a_constant_shift_of_all_payoffs_changes_nothing(rule, extra):
    a = L.simulate(S.Settings(rule=rule, c=0.6, n=5, runs=200, rounds=40, seed=5, **extra)).mins
    b = L.simulate(S.Settings(rule=rule, c=3.0, n=5, runs=200, rounds=40, seed=5, **extra)).mins
    assert (a == b).mean() > 0.999


def test_scaling_prize_and_price_together_changes_nothing_when_the_sharpness_is_normalised():
    base = L.simulate(S.Settings(n=6, a=0.2, b=0.05, runs=300, rounds=40, seed=9)).mins
    for k in (2.0, 5.0, 10.0):
        scaled = L.simulate(S.Settings(n=6, a=0.2 * k, b=0.05 * k, runs=300, rounds=40, seed=9)).mins
        assert (scaled == base).mean() > 0.999, k


def test_without_normalising_the_scale_does_matter():
    base = L.simulate(S.Settings(n=6, a=0.2, b=0.02, runs=300, rounds=40, seed=9, normalize=False)).mins
    scaled = L.simulate(S.Settings(n=6, a=1.0, b=0.1, runs=300, rounds=40, seed=9, normalize=False)).mins
    assert (scaled == base).mean() < 0.9


def test_renaming_the_partners_permutes_the_minimum_of_the_others():
    u = uniform_block(3, 40 * 6).reshape(40, 6)
    profiles = 1 + (u * 7).astype(int)
    order = np.array([3, 0, 5, 1, 4, 2])
    assert np.array_equal(G.excl_min(profiles)[:, order], G.excl_min(profiles[:, order]))
    assert np.array_equal(profiles.min(axis=1), profiles[:, order].min(axis=1))


def test_payoff_is_nondecreasing_in_the_minimum_of_the_others_and_decreasing_in_the_own_level_above_it():
    t = G.payoff_table(0.2, 0.1, 0.6)                                       # t[x−1, m−1]
    assert (np.diff(t, axis=1) >= -1e-15).all()                              # mehr Minimum der anderen: nie weniger
    for m in range(1, 8):
        above = t[m - 1:, m - 1]                                             # eigene Stufe ab dem Minimum der anderen
        assert (np.diff(above) < 0).all()                                    # darüber wird jede Stufe teurer, ohne zu nützen


def test_moving_probability_mass_upwards_raises_the_expected_minimum():
    rng_u = uniform_block(11, 20 * 7).reshape(20, 7)
    for row in rng_u:
        p = row / row.sum()
        for n in (2, 5):
            base = X.expected_min_iid(p, n)
            q = p.copy()
            j = int(np.argmax(q[:-1] > 0))                                    # erste Stufe mit Masse (nicht die oberste)
            moved = q[j] / 2
            q[j] -= moved
            q[j + 1] += moved
            assert X.expected_min_iid(q, n) > base - 1e-15


def test_expected_minimum_falls_with_the_group_size_for_any_distribution():
    p = np.array([0.05, 0.1, 0.15, 0.2, 0.2, 0.2, 0.1])
    values = [X.expected_min_iid(p, n) for n in range(1, 12)]
    assert all(x > y for x, y in zip(values, values[1:]))


def test_the_replicator_keeps_a_distribution_in_every_step():
    p = RP.random_starts(200, 4)
    for n in (2, 7, 32):
        q = p.copy()
        for _ in range(50):
            q = RP.step(q, n, 0.2, 0.1, 0.6)
            assert np.allclose(q.sum(axis=1), 1.0) and (q >= 0).all()


def test_the_number_of_equilibria_does_not_depend_on_the_price_inside_the_open_interval():
    for b in (0.01, 0.05, 0.1, 0.15, 0.19):
        assert len(G.equilibria(3, 0.2, b, 0.6)) == 7


def test_best_reply_does_not_depend_on_lambda_or_phi():
    a = L.simulate(S.Settings(rule=C.RULE_BEST, eps=0.05, lam=1.0, phi=0.0, runs=100, rounds=20, seed=2)).mins
    b = L.simulate(S.Settings(rule=C.RULE_BEST, eps=0.05, lam=200.0, phi=1.0, runs=100, rounds=20, seed=2)).mins
    assert np.array_equal(a, b)


def test_a_second_process_with_other_hash_randomisation_computes_the_same_run():
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); import hashlib, swg_scenario as S, swg_learning as L; "
            "r = L.simulate(S.Settings(n=6, runs=50, rounds=30, seed=77)); print(hashlib.sha256(r.mins.tobytes() + r.final.tobytes()).hexdigest())")
    outs = []
    for seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        res = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
        assert res.returncode == 0, res.stderr
        outs.append(res.stdout.strip())
    assert outs[0] == outs[1] and len(outs[0]) == 64
    r = L.simulate(S.Settings(n=6, runs=50, rounds=30, seed=77))
    import hashlib
    assert hashlib.sha256(r.mins.tobytes() + r.final.tobytes()).hexdigest() == outs[0]
