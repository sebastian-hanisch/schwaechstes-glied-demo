"""Orakel (ORACLE-PLAYBOOK §1): jede Kernroutine gegen einen ANDEREN Rechenweg, in reinem Python (math, itertools) oder durch Simulation mit Marge von vier Standardfehlern.

* Gleichgewichte: Aufzählung aller Profile mit Python-Zahlen gegen `swg_game.equilibria` (numpy) und `is_strict`.
* E[min]: Aufzählung aller Profile gegen die Formel Σ_k Π_i P(x_i ≥ k).
* Logit-Lernen: Pfadaufzählung in reinem Python (Rekursion, math.exp) gegen die vektorisierte Pfadaufzählung `logit_exact_path`; beide gegen `simulate`.
* Beste Antwort mit Zittern: eine in Python aufgebaute Markov-Kette gegen `br_chain`; stationäre Verteilung per Potenzieren; Simulation gegen die exakte Kette.
* Replikator: Schleife in Python gegen die vektorisierte Fassung; u_x gegen die Aufzählung der Mitspieler.
* Fehlertoleranz: Grenznutzen durch Aufzählung der Mitspieler-Ergebnisse.
Alles unter 10 Sekunden. Keine Zeit-Assertion, keine exakten Zufallswerte."""

import itertools
import math

import numpy as np
import pytest

import swg_constants as C
import swg_exact as X
import swg_game as G
import swg_learning as L
import swg_replicator as RP
import swg_scenario as S
from swg_rng import uniform_block


def pay(x, m, a, b, c):
    return a * min(x, m) - b * x + c


# --- O1 Gleichgewichte -----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("b", [0.02, 0.1, 0.19])
def test_equilibria_by_enumeration_of_all_profiles(n, b):
    found = []
    for prof in itertools.product(range(1, 8), repeat=n):
        ok = True
        for i in range(n):
            m = min(prof[:i] + prof[i + 1:])
            cur = pay(prof[i], m, 0.2, b, 0.6)
            if any(pay(y, m, 0.2, b, 0.6) > cur + 1e-12 for y in range(1, 8)):
                ok = False
                break
        if ok:
            found.append(prof)
    assert found == [(k,) * n for k in range(1, 8)]                           # genau sieben, alle symmetrisch
    assert G.equilibria(n, 0.2, b, 0.6) == found
    for prof in found:                                                       # alle strikt: jede Abweichung verschlechtert
        for i in range(n):
            m = min(prof[:i] + prof[i + 1:])
            assert all(pay(y, m, 0.2, b, 0.6) < pay(prof[i], m, 0.2, b, 0.6) - 1e-12 for y in range(1, 8) if y != prof[i])
        assert G.is_strict(n, prof, 0.2, b, 0.6)


# --- O2 E[min] -------------------------------------------------------------------------------------------------------------------------------
def test_expected_min_by_enumeration_matches_the_formula_for_random_distributions():
    u = uniform_block(5, 3 * 4 * 3)
    for trial in range(3):
        q = u[trial * 12:(trial + 1) * 12].reshape(3, 4)
        q = q / q.sum(axis=1, keepdims=True)
        exact = sum(np.prod([q[i][prof[i] - 1] for i in range(3)]) * min(prof) for prof in itertools.product(range(1, 5), repeat=3))
        assert X.expected_min_independent(q) == pytest.approx(exact, rel=1e-12)


def test_uniform_expected_min_by_enumeration():
    for n in (2, 3, 4):
        total = sum(min(prof) for prof in itertools.product(range(1, 8), repeat=n)) / 7 ** n
        assert X.uniform_expected_min(n) == pytest.approx(total, rel=1e-12)


# --- O3 Logit: Pfadaufzählung ------------------------------------------------------------------------------------------------------------------
def python_logit_path(n, levels, a, b, c, lam, phi, conv, rounds, prior=0.0, start="uniform"):
    """E[min] in den Runden 1..rounds durch Rekursion über alle Pfade, in reinem Python."""
    res = [0.0] * rounds

    def softmax(zs):
        mx = max(zs)
        w = [math.exp(z - mx) for z in zs]
        s = sum(w)
        return [x / s for x in w]

    def rec(prof, num, den, t, prob):
        res[t] += prob * min(prof)
        if t == rounds - 1:
            return
        den2 = phi * den + 1.0
        num2 = []
        for i in range(n):
            m = min(prof[:i] + prof[i + 1:])
            num2.append([phi * num[i][x - 1] + pay(x, m, a, b, c) for x in range(1, levels + 1)])
        ps = [softmax([lam * v if conv == "sum" else lam * v / den2 for v in num2[i]]) for i in range(n)]
        for nxt in itertools.product(range(levels), repeat=n):
            pr = 1.0
            for i in range(n):
                pr *= ps[i][nxt[i]]
            if pr > 0.0:
                rec(tuple(x + 1 for x in nxt), num2, den2, t + 1, prob * pr)

    num0 = [[prior * pay(x, levels, a, b, c) for x in range(1, levels + 1)] for _ in range(n)]
    starts = list(itertools.product(range(1, levels + 1), repeat=n)) if start == "uniform" else [(levels,) * n]
    for prof in starts:
        rec(prof, num0, prior, 0, 1.0 / len(starts))
    return res


CASES = [
    dict(n=2, levels=4, lam=5.0, phi=0.9, rule=C.RULE_SUM),
    dict(n=2, levels=7, lam=5.0, phi=0.9, rule=C.RULE_SUM),
    dict(n=3, levels=3, lam=50.0, phi=0.9, rule=C.RULE_MEAN),
    dict(n=2, levels=5, lam=20.0, phi=0.5, rule=C.RULE_MEAN),
    dict(n=2, levels=4, lam=5.0, phi=0.9, rule=C.RULE_SUM, prior=3.0),
    dict(n=3, levels=3, lam=5.0, phi=0.9, rule=C.RULE_SUM, start=C.START_TOP),
]


@pytest.mark.parametrize("case", CASES)
def test_vectorised_path_enumeration_equals_the_python_recursion(case):
    s = S.Settings(normalize=False, rounds=3, **case)
    ref = python_logit_path(case["n"], case["levels"], 0.2, 0.1, 0.6, case["lam"], case["phi"], s.convention, 3, case.get("prior", 0.0), "uniform" if case.get("start") is None else "top")
    assert X.logit_exact_path(s, 3) == pytest.approx(ref, rel=1e-9)


@pytest.mark.parametrize("case", CASES)
def test_simulation_lies_within_four_standard_errors_of_the_exact_path(case):
    s = S.Settings(normalize=False, rounds=3, runs=300_000, seed=101, **case)
    exact = X.logit_exact_path(s, 3)
    m = L.simulate(s).mins
    se = m.std(axis=0, ddof=1) / math.sqrt(m.shape[0])
    z = np.abs(m.mean(axis=0) - exact) / np.maximum(se, 1e-12)
    assert (z < 4.0).all(), z


# --- O4 beste Antwort: Kette -----------------------------------------------------------------------------------------------------------------
def python_br_chain(n, levels, eps):
    profs = list(itertools.product(range(1, levels + 1), repeat=n))
    index = {q: i for i, q in enumerate(profs)}
    mat = [[0.0] * len(profs) for _ in profs]
    for q in profs:
        dist = []
        for i in range(n):
            m = min(q[:i] + q[i + 1:])
            d = [eps / levels] * levels
            d[m - 1] += 1 - eps
            dist.append(d)
        for nxt in itertools.product(range(levels), repeat=n):
            pr = 1.0
            for i in range(n):
                pr *= dist[i][nxt[i]]
            mat[index[q]][index[tuple(x + 1 for x in nxt)]] += pr
    return profs, mat


@pytest.mark.parametrize("n,levels,eps", [(2, 4, 0.1), (3, 3, 0.05), (3, 4, 0.2)])
def test_best_reply_chain_equals_a_chain_built_in_python(n, levels, eps):
    profs, mat = python_br_chain(n, levels, eps)
    got, mins = X.br_chain(n, eps, levels=levels)
    assert np.allclose(got, np.array(mat), atol=1e-14, rtol=0) and mins.tolist() == [min(q) for q in profs]


def test_stationary_distribution_equals_repeated_multiplication():
    profs, mat = python_br_chain(3, 4, 0.2)
    got, mins = X.br_chain(3, 0.2, levels=4)
    v = np.full(len(profs), 1.0 / len(profs))
    for _ in range(400):
        v = v @ got
    assert X.stationary(got) == pytest.approx(v, abs=1e-9) and X.stationary(got) @ mins == pytest.approx(v @ mins, abs=1e-9)


def test_simulated_best_reply_stays_within_four_standard_errors_of_the_chain():
    for n, eps in ((2, 0.05), (3, 0.05), (3, 0.01)):
        mat, mins = X.br_chain(n, eps)
        exact = X.chain_expected_min(mat, mins, 60)
        m = L.simulate(S.Settings(rule=C.RULE_BEST, eps=eps, n=n, runs=100_000, rounds=60, seed=7)).mins
        se = m.std(axis=0, ddof=1) / math.sqrt(m.shape[0])
        z = np.abs(m.mean(axis=0) - exact) / np.maximum(se, 1e-12)
        assert (z < 4.0).all(), (n, eps, z.max())


def test_first_drop_formula_equals_the_hitting_time_of_the_chain_plus_one_round():
    for n in (2, 3):
        mat, mins = X.br_chain(n, 0.05)
        hit = X.hitting_time(mat, len(mat) - 1, mins <= 6)
        assert X.first_drop_rounds(n, 0.05) == pytest.approx(1.0 + hit, rel=1e-9)


# --- O5 Replikator -----------------------------------------------------------------------------------------------------------------------------
def test_replicator_equals_a_plain_python_loop_for_300_steps():
    p0 = [0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2]
    vec = RP.run(np.array([p0]), 4, 0.2, 0.1, 0.6, steps=300, tol=0.0)[0][0]
    q = list(p0)
    for _ in range(300):
        s = [sum(q[k:]) for k in range(7)]
        u = [0.2 * sum(s[k - 1] ** 3 for k in range(1, x + 1)) - 0.1 * x + 0.6 for x in range(1, 8)]
        ub = sum(qi * ui for qi, ui in zip(q, u))
        q = [qi * (1 + 0.5 * (ui - ub)) for qi, ui in zip(q, u)]
        tot = sum(q)
        q = [qi / tot for qi in q]
    assert vec == pytest.approx(q, abs=1e-12)


def test_replicator_utilities_equal_the_enumeration_over_the_other_players():
    p = np.array([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2])
    for n in (2, 3, 4):
        u = RP.utilities(p[None, :], n, 0.2, 0.1, 0.6)[0]
        for x in range(1, 8):
            enum = sum(np.prod([p[y - 1] for y in prof]) * pay(x, min(prof), 0.2, 0.1, 0.6) for prof in itertools.product(range(1, 8), repeat=n - 1))
            assert u[x - 1] == pytest.approx(enum, rel=1e-12), (n, x)


def test_replicator_expected_minimum_equals_the_enumeration():
    p = np.array([0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2])
    for n in (2, 3, 4):
        enum = sum(np.prod([p[y - 1] for y in prof]) * min(prof) for prof in itertools.product(range(1, 8), repeat=n))
        assert X.expected_min_iid(p, n) == pytest.approx(enum, rel=1e-12)


# --- O6 Pool -----------------------------------------------------------------------------------------------------------------------------------
def test_pool_round_one_is_the_exact_random_minimum_of_a_team():
    for k in (2, 4):
        for fixed in (True, False):
            r = L.simulate(S.Settings(n=16, runs=4000, rounds=2, seed=3), team=k, fixed=fixed)
            exact = X.uniform_expected_min(k)
            se = r.mins[:, 0].std(ddof=1) / math.sqrt(4000)
            assert abs(r.mins[:, 0].mean() - exact) < 4 * se, (k, fixed)


# --- O7 Fehlertoleranz --------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [2, 3, 4])
def test_tolerance_equals_the_zero_of_the_enumerated_marginal_gain(n):
    a, b, level = 0.2, 0.05, 4
    ratio = b / a
    delta = G.tolerance(n, ratio)

    def gain(q):
        total = 0.0
        for outcome in itertools.product((True, False), repeat=n - 1):          # True: dieser Mitspieler erreicht mindestens die Stufe
            pr = np.prod([q if o else 1 - q for o in outcome])
            m = level if all(outcome) else level - 1                             # das Minimum der anderen ist ≥ level genau, wenn alle es erreichen
            total += pr * (pay(level, m, a, b, 0.6) - pay(level - 1, m, a, b, 0.6))
        return total

    assert gain(1 - delta) == pytest.approx(0.0, abs=1e-12)
    assert gain(1 - delta + 0.02) > 0 > gain(1 - delta - 0.02)
    assert gain(0.9) == pytest.approx(G.marginal_gain(0.9, n, a, b), abs=1e-12)


# --- O8 zweiter Rechenweg mit scipy (nur Test, nie zur Laufzeit) ------------------------------------------------------------------------------------
def test_softmax_equals_scipy_softmax():
    special = pytest.importorskip("scipy.special")
    z = np.array([[3.5, 4.0, 4.5], [-1e5, 0.0, 1e5], [0.0, 0.0, 0.0]])
    assert L.logit_probs(z) == pytest.approx(special.softmax(z, axis=-1), abs=1e-15)


def test_stationary_distribution_equals_the_null_space_of_the_generator():
    linalg = pytest.importorskip("scipy.linalg")
    mat, mins = X.br_chain(3, 0.2, levels=4)
    ns = linalg.null_space(mat.T - np.eye(len(mat)))
    assert ns.shape[1] == 1
    pi = ns[:, 0] / ns[:, 0].sum()
    assert X.stationary(mat) == pytest.approx(pi, abs=1e-9)


def test_hitting_time_equals_the_sparse_solver():
    sparse = pytest.importorskip("scipy.sparse")
    splinalg = pytest.importorskip("scipy.sparse.linalg")
    mat, mins = X.br_chain(3, 0.05)
    target = mins <= 6
    rest = np.where(~target)[0]
    a = sparse.csc_matrix(np.eye(len(rest)) - mat[np.ix_(rest, rest)])
    t = splinalg.spsolve(a, np.ones(len(rest)))
    start = len(mat) - 1
    assert X.hitting_time(mat, start, target) == pytest.approx(float(t[np.searchsorted(rest, start)]), rel=1e-9)
