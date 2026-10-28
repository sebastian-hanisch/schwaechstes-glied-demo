"""Exakte Rechnungen ohne Simulation: Erwartungswert des Minimums unabhängiger Einsätze, Pfadaufzählung des Logit-Lernens in den ersten Runden und die Markov-Kette
der besten Antwort mit Zittern (Ratsche).

Die Kette hat levels^n Zustände (alle Profile); n ≤ 4 mit 7 Stufen sind 2 401 Zustände. Erwartete Treffzeiten kommen aus einem linearen System
(I − Q) t = 1; sie sind bei kleinem ε sehr groß (Größenordnung ε^−n), das System ist dann schlecht konditioniert und die Stellen hängen vom Löser ab:
Tests und README nennen solche Zeiten nur als Größenordnung bzw. Band."""

from __future__ import annotations

import itertools
import math

import numpy as np

import swg_constants as C
import swg_game as G
import swg_learning as L
import swg_scenario as S

MAX_PATHS = 3_000_000


def expected_min_independent(p: np.ndarray) -> float:
    """E[min] unabhängiger Einsätze: Σ_k Π_i P(x_i ≥ k). `p` hat die Form (n, E) (Verteilung je Spieler) oder (E,) (für alle gleich; dann ist n = `n` nötig → expected_min_iid)."""
    surv = np.cumsum(p[:, ::-1], axis=1)[:, ::-1]        # surv[i, k−1] = P(x_i ≥ k)
    return float(np.prod(surv, axis=0).sum())


def expected_min_iid(p: np.ndarray, n: int) -> float:
    """E[min von n unabhängigen Ziehungen aus p] = Σ_k S(k)^n mit S(k) = P(x ≥ k)."""
    surv = np.cumsum(p[::-1])[::-1]
    return float((surv ** n).sum())


def uniform_expected_min(n: int, levels: int = C.E_LEVELS) -> float:
    """E[min] bei gleichverteilten Stufen: Σ_k ((E−k+1)/E)^n (Runde 1 aus gleichverteiltem Start ist reiner Zufall)."""
    k = np.arange(1, levels + 1)
    return float((((levels - k + 1) / levels) ** n).sum())


# --------------------------------------------------------------------------------- Logit: Pfadaufzählung
def logit_exact_path(s: S.Settings, rounds: int) -> np.ndarray:
    """E[Minimum] in den Runden 1..rounds exakt für die Logit-Regeln, durch Aufzählung aller Pfade der Profile (Start gleichverteilt oder fest, Vorabsprache und Vergessen wie in
    `swg_learning.simulate`). Gedacht für wenige Spieler und Runden: die Zahl der Pfade wächst wie levels^(n·(rounds−1)) und ist durch MAX_PATHS begrenzt (sonst ValueError)."""
    if s.rule not in (C.RULE_SUM, C.RULE_MEAN):
        raise ValueError("Pfadaufzählung gibt es nur für die Logit-Regeln")
    n, levels = s.n, s.levels
    if levels ** (n * (rounds - 1)) * (1 if s.start != C.START_UNIFORM else levels ** n) > MAX_PATHS:
        raise ValueError("zu viele Pfade für die Aufzählung")
    table = G.payoff_table(s.a, s.b, s.c, levels)
    table_t = np.ascontiguousarray(table.T)
    lam, phi, conv = S.sharpness(s), S.forgetting(s), s.convention
    every = np.array(list(itertools.product(range(1, levels + 1), repeat=n)), dtype=np.int64)        # (levels^n, n), erster Spieler am langsamsten
    if s.start == C.START_UNIFORM:
        prof, prob = every, np.full(len(every), 1.0 / len(every))
    else:
        level = levels if s.start == C.START_TOP else min(C.MID_LEVEL, levels)
        prof, prob = np.full((1, n), level, dtype=np.int64), np.ones(1)
    num = np.zeros((len(prof), n, levels))
    den = 0.0
    if s.prior > 0:
        num += s.prior * table[:, levels - 1][None, None, :]
        den += s.prior
    out = np.empty(rounds)
    out[0] = float(prob @ prof.min(axis=1))
    for t in range(1, rounds):
        m = G.excl_min(prof)
        num = phi * num + table_t[m - 1]
        den = phi * den + 1.0
        p = L.logit_probs(lam * num if conv == "sum" else lam * num / den)          # (Pfade, n, levels)
        child = p[:, 0, :]
        for i in range(1, n):
            child = (child[:, :, None] * p[:, i, None, :]).reshape(len(prof), -1)       # (Pfade, levels^(i+1)), Reihenfolge wie `every`
        prob = (prob[:, None] * child).reshape(-1)
        prof = np.tile(every, (len(num), 1))
        num = np.repeat(num, len(every), axis=0)
        out[t] = float(prob @ prof.min(axis=1))
    return out


# --------------------------------------------------------------------------------- beste Antwort mit Zittern: Kette
def br_chain(n: int, eps: float, a: float = C.DEFAULT_A, b: float = C.DEFAULT_B, c: float = C.DEFAULT_C, levels: int = C.E_LEVELS) -> tuple[np.ndarray, np.ndarray]:
    """Übergangsmatrix der besten Antwort mit Zittern über alle levels^n Profile und das Minimum je Profil (Zustand i ↔ Profil `i` in der Reihenfolge von itertools.product).

    Je Spieler: Verteilung ε/E gleichmäßig plus (1 − ε) auf der besten Antwort auf das Minimum der anderen; die Spieler wählen unabhängig."""
    profs = np.array(list(itertools.product(range(1, levels + 1), repeat=n)), dtype=np.int64)
    br = G.best_reply_table(a, b, c, levels)
    m = G.excl_min(profs)                                      # (Zustände, n)
    d = np.full((len(profs), n, levels), eps / levels)
    d[np.arange(len(profs))[:, None], np.arange(n)[None, :], br[m - 1] - 1] += 1.0 - eps
    mat = d[:, 0, :]
    for i in range(1, n):
        mat = (mat[:, :, None] * d[:, i, None, :]).reshape(len(profs), -1)
    return mat, profs.min(axis=1).astype(float)


def hitting_time(mat: np.ndarray, start: int, target: np.ndarray) -> float:
    """Erwartete Zahl der Schritte vom Zustand `start` bis erstmals ein Zustand der Menge `target` (Maske) erreicht ist; löst (I − Q) t = 1 auf den Zuständen außerhalb."""
    if target[start]:
        return 0.0
    rest = np.where(~target)[0]
    q = mat[np.ix_(rest, rest)]
    t = np.linalg.solve(np.eye(len(rest)) - q, np.ones(len(rest)))
    return float(t[np.searchsorted(rest, start)])


def stationary(mat: np.ndarray) -> np.ndarray:
    """Stationäre Verteilung der Kette (π P = π, Σπ = 1) aus dem linearen System."""
    k = len(mat)
    a = mat.T - np.eye(k)
    a[-1, :] = 1.0
    rhs = np.zeros(k)
    rhs[-1] = 1.0
    return np.linalg.solve(a, rhs)


def chain_expected_min(mat: np.ndarray, mins: np.ndarray, rounds: int, v0: np.ndarray | None = None) -> np.ndarray:
    """E[Minimum] in den Runden 1..rounds; Start `v0` (Verteilung über die Zustände), ohne Angabe gleichverteilt (Runde 1 = Zufall)."""
    v = np.full(len(mat), 1.0 / len(mat)) if v0 is None else np.asarray(v0, dtype=float)
    out = np.empty(rounds)
    for t in range(rounds):
        out[t] = v @ mins
        v = v @ mat
    return out


def point_start(n: int, level: int, levels: int = C.E_LEVELS) -> np.ndarray:
    """Startverteilung der Kette: alle n Partner auf `level` (Zustandsnummer wie in itertools.product, erster Partner am langsamsten)."""
    v = np.zeros(levels ** n)
    index = 0
    for _ in range(n):
        index = index * levels + (level - 1)
    v[index] = 1.0
    return v


def first_drop_rounds(n: int, eps: float, levels: int = C.E_LEVELS) -> float:
    """Erwartete Runde (1 = der Start), in der aus „alle auf der obersten Stufe“ erstmals ein Spieler darunter liegt: 1 + 1/(1 − q^n) mit q = 1 − ε + ε/E
    (q = Wahrscheinlichkeit, dass ein Spieler wieder die oberste Stufe wählt; nur das Profil „alle oben“ hat das oberste Minimum)."""
    if eps <= 0:
        return math.inf                                  # ohne Zittern verlässt die Kette „alle oben“ nie
    q = 1.0 - eps + eps / levels
    return 1.0 + 1.0 / (1.0 - q ** n)
