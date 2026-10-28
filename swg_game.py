"""Das Spiel vom schwächsten Glied: Auszahlung, Minimum der anderen, Gleichgewichte und Fehlertoleranz (alle Einheiten reine Funktionen).

n Partner wählen je eine Leistungsstufe x in 1..E; die Kette läuft mit der langsamsten Stufe. Auszahlung eines Partners (Zeitersparnis je Sendung):
π(x, m) = a · min(x, m) − b · x + c, wobei m das Minimum der Stufen der ANDEREN ist. Die Zahlen a = 0,2, b = 0,1, c = 0,6 sind die des klassischen
Laborspiels, hier nur Modellparameter."""

from __future__ import annotations

import itertools

import numpy as np

import swg_constants as C


def payoff(x, m, a: float, b: float, c: float):
    """Auszahlung bei eigener Stufe x und Minimum m der anderen (Zahl oder Array)."""
    return a * np.minimum(x, m) - b * x + c


def payoff_table(a: float, b: float, c: float, levels: int = C.E_LEVELS) -> np.ndarray:
    """pi[x−1, m−1] für x, m in 1..levels."""
    x = np.arange(1, levels + 1)[:, None]
    m = np.arange(1, levels + 1)[None, :]
    return payoff(x, m, a, b, c)


def excl_min(e: np.ndarray) -> np.ndarray:
    """Minimum der ANDEREN je Spieler; `e` hat die Form (..., n) mit n ≥ 2.

    Der Spieler, der allein das Minimum hält, sieht das zweitkleinste; alle anderen (und bei Gleichstand auch der Minimumhalter) sehen das Minimum selbst."""
    s = np.sort(e, axis=-1)
    m1 = s[..., 0:1]
    m2 = s[..., 1:2]
    return np.where((e == m1) & (m2 > m1), m2, m1)


def best_reply_table(a: float, b: float, c: float, levels: int = C.E_LEVELS) -> np.ndarray:
    """Beste Antwort br[m−1] (Stufe 1..levels) auf das Minimum m der anderen; bei Gleichstand die höhere Stufe. Für a > b ist es x = m."""
    table = payoff_table(a, b, c, levels)
    return levels - np.argmax(table[::-1, :], axis=0)


def equilibria(n: int, a: float, b: float, c: float, levels: int = C.E_LEVELS, tol: float = 1e-12) -> list[tuple[int, ...]]:
    """ALLE Gleichgewichte in reinen Strategien durch Aufzählung aller levels^n Profile (nur für kleine n gedacht).

    Ein Profil ist Gleichgewicht, wenn kein Spieler durch Wechsel seiner Stufe mehr als `tol` gewinnt."""
    table = payoff_table(a, b, c, levels)
    found = []
    for prof in itertools.product(range(1, levels + 1), repeat=n):
        arr = np.array(prof)
        m = excl_min(arr)
        ok = all(table[arr[i] - 1, m[i] - 1] >= table[:, m[i] - 1].max() - tol for i in range(n))
        if ok:
            found.append(prof)
    return found


def is_strict(n: int, prof: tuple[int, ...], a: float, b: float, c: float, levels: int = C.E_LEVELS, tol: float = 1e-12) -> bool:
    """Strikt: jede einzelne Abweichung eines Spielers verschlechtert ihn um mehr als `tol`."""
    table = payoff_table(a, b, c, levels)
    arr = np.array(prof)
    m = excl_min(arr)
    for i in range(n):
        cur = table[arr[i] - 1, m[i] - 1]
        for y in range(1, levels + 1):
            if y != arr[i] and table[y - 1, m[i] - 1] >= cur - tol:
                return False
    return True


def tolerance(n: int, ratio: float) -> float:
    """Fehlertoleranz je Partner δ* = 1 − (b/a)^(1/(n−1)): „Stufe L halten“ lohnt gegenüber „eine Stufe tiefer“ genau dann, wenn jeder der n−1 anderen
    das Niveau mit höchstens dieser Wahrscheinlichkeit verfehlt, denn der Zugewinn ist a · P(Minimum der anderen ≥ L) − b = a · (1−δ)^(n−1) − b."""
    return 1.0 - ratio ** (1.0 / (n - 1))


def marginal_gain(q: float, n: int, a: float, b: float) -> float:
    """Zugewinn von „Stufe L halten“ gegenüber „eine Stufe tiefer“, wenn jeder der n−1 anderen das Niveau L mit Wahrscheinlichkeit q erreicht:
    a · q^(n−1) − b. Er hängt nicht an L selbst, denn π(L, m) − π(L−1, m) = a · [m ≥ L] − b."""
    return a * q ** (n - 1) - b
