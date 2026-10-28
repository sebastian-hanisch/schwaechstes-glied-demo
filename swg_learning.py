"""Lernregeln auf dem Spiel vom schwächsten Glied: Logit-Erfahrungsbewertung (Summe und Mittel), beste Antwort mit Zittern, fiktives Spiel.

Jede Regel sieht nach jeder Runde nur das Minimum der ANDEREN und bewertet daraufhin ALLE Stufen gegenfaktisch (was hätte Stufe x gebracht?).

**Logit-Erfahrungsbewertung** mit Vergessen φ: Zähler N_x ← φ·N_x + π(x, m), Nenner W ← φ·W + 1 (je Spieler). Gewählt wird Stufe x mit Wahrscheinlichkeit ∝ exp(z_x):
* Fassung „Summe“ (die des früheren Skripts V4): z = λ·N. Die Schärfe wächst in den ersten Runden mit der Summe der Gewichte bis λ/(1−φ).
* Fassung „Mittel“: z = Λ·N/W, das gewichtete Mittel der Auszahlungen, Schärfe von Anfang an Λ.
**Beste Antwort mit Zittern**: x = m (die beste Antwort auf das Minimum der Vorrunde, solange a > b), mit Wahrscheinlichkeit ε stattdessen eine gleichverteilte Stufe.
**Fiktives Spiel**: Mittel über alle Runden (φ = 1) mit Schärfe C.FICT_LAM, praktisch die beste Antwort auf die gesamte Erfahrung.

Alle Läufe werden gleichzeitig gerechnet (numpy über Läufe, Spieler, Stufen); die Zufallszahlen kommen blockweise aus SplitMix64 (swg_rng) und sind damit auf jeder
Plattform dieselben Gleichverteilten. Der Block einer Runde ist `Stream.take(Läufe · Spieler)`; die Reihenfolge der Ziehungen steht an jeder Stelle im Code."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import swg_constants as C
import swg_game as G
import swg_scenario as S
from swg_rng import _GAMMA, _MASK, uniform_block


# --------------------------------------------------------------------------------- Zufall
def block_seed(seed: int, offset: int) -> int:
    """Seed des SplitMix64-Stroms ab Position `offset` (der Generator ist ein Zähler: Seed + offset·γ)."""
    return (seed + offset * _GAMMA) & _MASK


class Stream:
    """Fortlaufender SplitMix64-Strom in Blöcken: `take(count)` liefert die nächsten `count` Gleichverteilten, `Stream(seed).take(a)` und danach `take(b)` zusammen
    genau `uniform_block(seed, a + b)`."""

    def __init__(self, seed: int):
        self.seed = seed & _MASK
        self.pos = 0

    def take(self, count: int) -> np.ndarray:
        u = uniform_block(block_seed(self.seed, self.pos), count)
        self.pos += count
        return u


# --------------------------------------------------------------------------------- Einheiten
def initial_profile(start: str, levels: int, u: np.ndarray) -> np.ndarray:
    """Startstufen aller Spieler: gleichverteilt auf 1..levels (aus den Gleichverteilten `u`), „alle 7“ (oberste Stufe) oder „alle 4“ (Mitte, höchstens die oberste Stufe)."""
    if start == C.START_UNIFORM:
        return 1 + np.minimum((u * levels).astype(np.int64), levels - 1)
    if start == C.START_TOP:
        return np.full(u.shape, levels, dtype=np.int64)
    if start == C.START_MID:
        return np.full(u.shape, min(C.MID_LEVEL, levels), dtype=np.int64)
    raise ValueError(f"unbekannter Start: {start}")


def logit_probs(z: np.ndarray) -> np.ndarray:
    """Softmax über die letzte Achse (numerisch stabil: das Maximum wird abgezogen)."""
    w = np.exp(z - z.max(axis=-1, keepdims=True))
    return w / w.sum(axis=-1, keepdims=True)


def draw_logit(z: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Zieht je Zeile eine Stufe 1..E mit Wahrscheinlichkeit ∝ exp(z) aus der Gleichverteilten `u` (Inversion der kumulierten Verteilung)."""
    w = np.exp(z - z.max(axis=-1, keepdims=True))
    cum = np.cumsum(w, axis=-1)
    cum /= cum[..., -1:]
    return np.minimum(1 + (u[..., None] > cum).sum(axis=-1), z.shape[-1])


def update_experience(num: np.ndarray, den: np.ndarray, pay: np.ndarray, phi: float) -> tuple[np.ndarray, np.ndarray]:
    """Zähler und Nenner der Erfahrungsbewertung: N ← φ·N + π, W ← φ·W + 1."""
    return phi * num + pay, phi * den + 1.0


def logit_argument(num: np.ndarray, den: np.ndarray, lam: float, convention: str) -> np.ndarray:
    """Logit-Argument je Stufe: „sum“ → λ·N, „mean“ → λ·N/W."""
    return lam * num if convention == "sum" else lam * num / den


def best_reply_choice(m: np.ndarray, br: np.ndarray, eps: float, u_tremble: np.ndarray, u_level: np.ndarray, levels: int) -> np.ndarray:
    """Beste Antwort auf m; mit Wahrscheinlichkeit ε (u_tremble < ε) stattdessen die gleichverteilte Stufe aus `u_level`."""
    rnd = 1 + np.minimum((u_level * levels).astype(np.int64), levels - 1)
    return np.where(u_tremble < eps, rnd, br[m - 1])


def team_excl_min(e: np.ndarray, k: int, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum der anderen im Team und Team-Minimum, wenn der Pool (Form (Läufe, N)) in der Reihenfolge `order` in Teams der Größe k zerlegt wird.

    Rückgabe: m (Form (Läufe, N), in der Reihenfolge der Spieler) und gmin (Form (Läufe, N/k))."""
    runs, pool = e.shape
    if k == pool:                                       # eine Kette: kein Umsortieren nötig
        return G.excl_min(e), e.min(axis=1, keepdims=True)
    eg = np.take_along_axis(e, order, axis=1).reshape(runs, pool // k, k)
    mg = G.excl_min(eg).reshape(runs, pool)
    m = np.empty_like(mg)
    np.put_along_axis(m, order, mg, axis=1)
    return m, eg.min(axis=2)


# --------------------------------------------------------------------------------- Lauf
@dataclass(frozen=True)
class Run:
    mins: np.ndarray       # (Läufe, Runden): Kettenminimum; im Pool das Mittel der Team-Minima
    top: np.ndarray        # (Läufe, Runden): Anteil der Ketten (Teams) mit dem Minimum der obersten Stufe
    trace: np.ndarray      # (Runden, n): Stufen aller Spieler im ersten Lauf
    final: np.ndarray      # (Läufe, n): Stufen der letzten Runde

    @property
    def runs(self) -> int:
        return self.mins.shape[0]


def simulate(s: S.Settings, *, team: int | None = None, fixed: bool = True, grow: tuple[int, int] | None = None) -> Run:
    """`s.runs` unabhängige Läufe über `s.rounds` Runden. `team`: Pool von `s.n` Spielern in Teams dieser Größe (fest oder jede Runde neu gelost), sonst eine Kette aus allen
    n Spielern. `grow=(Start, Takt)`: nur eine Kette und nur Logit; die Gruppe startet mit `Start` Spielern und wächst alle `Takt` Runden um einen, wer dazukommt übernimmt die
    Erfahrung eines zufälligen Mitglieds. Reihenfolge der Ziehungen je Runde: Neuzugänge (Quelle), Wahl (Logit: ein Block; beste Antwort: Zittern, dann Stufe),
    Teamlosung (nur gelost); vorweg der Start."""
    runs, n, rounds, levels = s.runs, s.n, s.rounds, s.levels
    k = n if team is None else team
    if n % k:
        raise ValueError("die Teamgröße muss den Pool teilen")
    best = s.rule == C.RULE_BEST
    if grow is not None and (best or team is not None):
        raise ValueError("wachsende Gruppen gibt es nur für Logit-Regeln in einer Kette")
    table = G.payoff_table(s.a, s.b, s.c, levels)
    table_t = np.ascontiguousarray(table.T)            # table_t[m−1, x−1]
    br = G.best_reply_table(s.a, s.b, s.c, levels)
    lam, phi, conv = S.sharpness(s), S.forgetting(s), s.convention
    stream = Stream(s.seed)
    num = np.zeros((runs, n, levels))
    den = np.zeros((runs, n, 1))
    if s.prior > 0:
        num += s.prior * table[:, levels - 1][None, None, :]
        den += s.prior
    e = initial_profile(s.start, levels, stream.take(runs * n).reshape(runs, n))
    active = n if grow is None else min(n, grow[0])
    e[:, active:] = levels                              # noch nicht dabei: oberste Stufe, wirkt auf kein Minimum
    mins = np.empty((runs, rounds))
    top = np.empty((runs, rounds))
    trace = np.empty((rounds, n), dtype=np.int64)
    ar = np.arange(runs)
    order = np.broadcast_to(np.arange(n), (runs, n))
    for t in range(rounds):
        m, gmin = team_excl_min(e, k, order)
        mins[:, t] = gmin.mean(axis=1)
        top[:, t] = (gmin == levels).mean(axis=1)
        trace[t] = e[0]
        if t == rounds - 1:
            break
        if not best:
            num, den = update_experience(num, den, table_t[m - 1], phi)
        if grow is not None:
            nxt = min(n, grow[0] + (t + 1) // grow[1])
            if nxt > active:
                src = np.minimum((stream.take(runs) * active).astype(np.int64), active - 1)
                for j in range(active, nxt):
                    num[:, j] = num[ar, src]
                    den[:, j] = den[ar, src]
                active = nxt
        if best:
            u = stream.take(2 * runs * n).reshape(2, runs, n)
            e = best_reply_choice(m, br, s.eps, u[0], u[1], levels)
        else:
            e = draw_logit(logit_argument(num, den, lam, conv), stream.take(runs * n).reshape(runs, n))
        e[:, active:] = levels
        if team is not None and not fixed and k < n:
            order = np.argsort(stream.take(runs * n).reshape(runs, n), axis=1, kind="stable")
    return Run(mins, top, trace, e)
