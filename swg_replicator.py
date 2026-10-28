"""Replikator im Mittelwertfeld: eine große Gruppe, aus der n−1 unabhängige Mitspieler gezogen werden; die Verteilung p der Stufen entwickelt sich deterministisch.

Auszahlung der Stufe x gegen n−1 Mitspieler aus p: u_x = a · Σ_{k≤x} S(k)^(n−1) − b · x + c mit S(k) = P(Stufe ≥ k) (das Minimum der anderen ist ≥ k mit Wahrscheinlichkeit S(k)^(n−1)).
Schritt (Euler): p_x ← p_x · (1 + η · κ · (u_x − ū)); κ = A_REF / a stellt die Zeit auf Einheiten der Prämie um (der Pfad der Verteilungen hängt nur von b/a ab).
Mutation μ mischt je Schritt den Anteil μ gleichverteilt zu."""

from __future__ import annotations

import numpy as np

import swg_constants as C
import swg_rng as RNG


def survival(p: np.ndarray) -> np.ndarray:
    """S(k) = P(Stufe ≥ k) für k = 1..E, je Zeile (Form (..., E))."""
    return np.cumsum(p[..., ::-1], axis=-1)[..., ::-1]


def utilities(p: np.ndarray, n: int, a: float, b: float, c: float) -> np.ndarray:
    """u_x für x = 1..E gegen n−1 unabhängige Mitspieler aus p."""
    levels = p.shape[-1]
    x = np.arange(1, levels + 1)
    return a * np.cumsum(survival(p) ** (n - 1), axis=-1) - b * x + c


def step(p: np.ndarray, n: int, a: float, b: float, c: float, eta: float = C.E3_ETA, mu: float = 0.0) -> np.ndarray:
    """Ein Euler-Schritt des Replikators (Form (K, E)); negative Anteile werden auf 0 gekappt, dann normiert; danach die Mutation μ."""
    u = utilities(p, n, a, b, c)
    ubar = (p * u).sum(axis=-1, keepdims=True)
    q = np.clip(p * (1.0 + eta * (C.A_REF / a) * (u - ubar)), 0.0, None)
    q /= q.sum(axis=-1, keepdims=True)
    if mu > 0:
        q = (1.0 - mu) * q + mu / p.shape[-1]
    return q


def run(p0: np.ndarray, n: int, a: float, b: float, c: float, steps: int = C.E3_STEPS, eta: float = C.E3_ETA, mu: float = 0.0, tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """`steps` Schritte für alle Startverteilungen `p0` (Form (K, E)). Starts, deren größter Anteil über 1 − tol liegt, werden eingefroren (ohne Mutation); das ändert die
    Einordnung „endet bei Stufe k“ nicht. Rückgabe: Endverteilungen und die Zahl der Schritte je Start."""
    p = p0.copy()
    used = np.zeros(len(p), dtype=np.int64)
    alive = np.arange(len(p))
    for t in range(1, steps + 1):
        p[alive] = step(p[alive], n, a, b, c, eta, mu)
        used[alive] = t
        if mu == 0.0 and t % 25 == 0:
            alive = alive[p[alive].max(axis=-1) < 1.0 - tol]
            if len(alive) == 0:
                break
    return p, used


def random_starts(count: int, seed: int, levels: int = C.E_LEVELS) -> np.ndarray:
    """Dirichlet(1, …, 1): normierte Exponentialverteilte −ln(1 − u) aus dem SplitMix64-Block des Seeds (Form (count, E))."""
    g = -np.log(1.0 - RNG.uniform_block(seed, count * levels)).reshape(count, levels)
    return g / g.sum(axis=1, keepdims=True)


def end_level(p: np.ndarray, threshold: float = 0.99) -> np.ndarray:
    """Endstufe 1..E, wo der größte Anteil über `threshold` liegt, sonst 0 (nicht konvergiert). Bei Gleichstand der Anteile gäbe es keinen Anteil über 0,99."""
    return np.where(p.max(axis=-1) > threshold, p.argmax(axis=-1) + 1, 0)


def basin(n: int, ratio: float, starts: int, seed: int, steps: int = C.E3_STEPS) -> dict:
    """Einzugsbereiche für b/a = `ratio` (a = A_REF): Anteil der Zufallsstarts, die bei Stufe 1..E enden, und der nicht konvergierten (Stufe 0)."""
    p0 = random_starts(starts, seed)
    p, used = run(p0, n, C.A_REF, C.A_REF * ratio, C.DEFAULT_C, steps)
    lev = end_level(p)
    shares = [float((lev == k).mean()) for k in range(0, C.E_LEVELS + 1)]
    emin = float((survival(p) ** n).sum(axis=-1).mean())
    return {"n": n, "ratio": ratio, "starts": starts, "none": shares[0], "share": shares[1:], "emin": emin, "max_steps": int(used.max())}
