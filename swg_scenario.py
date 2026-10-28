"""Szenario: die Einstellungen der App (ein Lauf der Lernregeln) und ihre abgeleiteten Größen. Der Zufall kommt aus SplitMix64 (swg_rng)."""

from __future__ import annotations

from dataclasses import dataclass, replace

import swg_constants as C


@dataclass(frozen=True, kw_only=True)
class Settings:
    """Alle Regler der Seitenleiste. Nur mit Schlüsselwörtern anlegen: ein eingefügtes Feld darf nie still Werte verschieben."""
    rule: str = C.RULE_SUM
    start: str = C.START_UNIFORM
    n: int = C.DEFAULT_N
    a: float = C.DEFAULT_A            # Prämie: Zeitgewinn je Kettenstufe
    b: float = C.DEFAULT_B            # Einsatzpreis: Eigenaufwand je Stufe
    c: float = C.DEFAULT_C            # Grundniveau (kürzt sich im Logit heraus, verschiebt nur die Höhe der Auszahlung)
    lam: float = C.DEFAULT_LAM        # Schärfe: Summe λ, Mittel Λ
    phi: float = C.DEFAULT_PHI        # Vergessen: Gewicht der Vorrunden
    eps: float = C.DEFAULT_EPS        # Zittern der besten Antwort
    prior: float = 0.0                # Vorabsprache s: fiktive Runden „alle spielen die oberste Stufe“
    normalize: bool = True            # Schärfe in Einheiten von a / A_REF (nur das Verhältnis b/a zählt)
    levels: int = C.E_LEVELS
    rounds: int = C.DEFAULT_T
    runs: int = C.DEFAULT_RUNS
    seed: int = C.DEFAULT_SEED

    @property
    def convention(self) -> str:
        """„sum“ (Logit-Argument λ·N) oder „mean“ (λ·N/W), siehe swg_learning."""
        return "sum" if self.rule == C.RULE_SUM else "mean"

    @property
    def noisy(self) -> bool:
        """Enthält die Regel Zufall in der Wahl (Logit immer, beste Antwort nur mit ε > 0)? Das fiktive Spiel mit Schärfe 1 000 zählt als rauschfrei."""
        return self.rule in (C.RULE_SUM, C.RULE_MEAN) or (self.rule == C.RULE_BEST and self.eps > 0)


def settings_of_preset(name: str) -> Settings:
    return Settings(**C.PRESETS[name])


def with_(s: Settings, **changes) -> Settings:
    """Kopie mit geänderten Feldern (nur Schlüsselwörter)."""
    return replace(s, **changes)


def sharpness(s: Settings) -> float:
    """Wirksame Schärfe des Logit: normiert auf die Prämie, beim fiktiven Spiel fest C.FICT_LAM."""
    if s.rule == C.RULE_FICT:
        return C.FICT_LAM
    return s.lam * (C.A_REF / s.a if s.normalize else 1.0)


def forgetting(s: Settings) -> float:
    """Wirksames Vergessen: das fiktive Spiel mittelt über alle Runden (φ = 1)."""
    return 1.0 if s.rule == C.RULE_FICT else s.phi


def budget_runs(s: Settings) -> int:
    """Läufe, mit denen die App tatsächlich rechnet: höchstens RUN_BUDGET / (n · Runden), mindestens 1."""
    return max(1, min(s.runs, C.RUN_BUDGET // (s.n * s.rounds)))


def estimated_seconds(s: Settings) -> float:
    """Grobe Zeitschätzung eines Laufs für die Anzeige am Spinner: Spielerrunden (Läufe · n · Runden) durch C.PLAYER_ROUNDS_PER_SECOND."""
    return budget_runs(s) * s.n * s.rounds / C.PLAYER_ROUNDS_PER_SECOND
