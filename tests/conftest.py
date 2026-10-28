"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Alle Erwartungswerte stehen als Rechnung dabei."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def pay2():
    """Zwei Partner, Einsätze (3, 5), a = 0,2, b = 0,1, c = 0,6. Minimum der Kette 3.
    Spieler 1 (x = 3) sieht das Minimum der anderen m = 5: π = 0,2·min(3, 5) − 0,1·3 + 0,6 = 0,6 − 0,3 + 0,6 = 0,9.
    Spieler 2 (x = 5) sieht m = 3: π = 0,2·3 − 0,1·5 + 0,6 = 0,6 − 0,5 + 0,6 = 0,7."""
    return {"profile": (3, 5), "min": 3, "excl": (5, 3), "pay": (0.9, 0.7)}


@pytest.fixture
def pay4():
    """Vier Partner, alle auf 7: π = 0,2·7 − 0,1·7 + 0,6 = 1,4 − 0,7 + 0,6 = 1,3. Alle auf 1: 0,2 − 0,1 + 0,6 = 0,7. Der Unterschied 0,6 = (a − b)·(7 − 1)."""
    return {"all7": 1.3, "all1": 0.7}


@pytest.fixture
def uniform2():
    """Gleichverteilung auf 1..7, zwei Spieler: E[min] = Σ_k P(beide ≥ k) = Σ_k ((8 − k)/7)² = (49 + 36 + 25 + 16 + 9 + 4 + 1)/49 = 140/49 = 2,857."""
    return {"n": 2, "value": 140 / 49}


@pytest.fixture
def tol3():
    """δ* für n = 3, b/a = 0,25: 1 − (0,25)^(1/2) = 1 − 0,5 = 0,5. Für n = 2 ist δ* = 1 − b/a (b/a = 0,5: 0,5)."""
    return {"n": 3, "ratio": 0.25, "delta": 0.5}


@pytest.fixture
def margin2():
    """Zweierspiel mit Einsätzen (4, 7), b/a = 0,5: Grenznutzen von Stufe 5 gegen Stufe 4 für Spieler 1 (das Minimum der anderen ist 7): 0,2·(min(5, 7) − min(4, 7)) − 0,1 = 0,2 − 0,1 = +0,1."""
    return {"m": 7, "from": 4, "to": 5, "gain": 0.1}


@pytest.fixture
def logit3():
    """Logit-Schritt von Hand: drei Stufen, zwei Spieler, beide auf Stufe 3 (m = 3), a = 0,2, b = 0,1, c = 0,6, φ = 0,9, λ = 5, Start ohne Vorerfahrung.
    Auszahlungen der Stufen 1..3 gegen m = 3: 0,7 / 0,8 / 0,9. Summe: z = 5·(0,7, 0,8, 0,9) = (3,5; 4,0; 4,5); Gewichte e^−1, e^−0,5, 1 = 0,367879 / 0,606531 / 1; Summe 1,974410;
    p = (0,186324; 0,307196; 0,506480). E[min] der nächsten Runde (zwei unabhängige Spieler) = Σ_k S(k)² mit S(1) = 1, S(2) = 0,813676, S(3) = 0,506480: 1 + 0,662068 + 0,256523 = 1,918591.
    Mittel mit Λ = 50 und W = 1: z = (35, 40, 45), p₃ = 1/(1 + e^−5 + e^−10) = 0,993262."""
    return {"pay": (0.7, 0.8, 0.9), "p_sum": (0.186324, 0.307196, 0.506480), "emin2": 1.918591, "p3_mean": 0.993262}


@pytest.fixture
def chain2():
    """Beste Antwort mit Zittern, zwei Spieler, zwei Stufen, ε = 0,2: jeder wählt die Stufe des anderen mit 0,9 und die andere mit 0,1 (ε/E = 0,1 zufällig plus 0,8 beste Antwort).
    Aus einem gleichen Paar bleibt das Paar gleich mit 0,82 (beide oder keiner wechseln: 0,81 + 0,01), aus einem ungleichen wird es gleich mit 0,18; stationär also P(gleich) = 0,18/(0,18 + 0,18) = ½
    und P(beide oben) = ¼: E[min] = 1·¾ + 2·¼ = 1,25. Bei sieben Stufen und zwei Spielern ist die stationäre Verteilung gleichverteilt (die Übergangsmatrix ist doppelt stochastisch): 140/49."""
    return {"stationary": 1.25, "seven": 140 / 49}


@pytest.fixture
def repl2():
    """Replikator, zwei Stufen, n = 2, a = 0,2, b = 0,1, c = 0,6, η = 0,5: u₁ = a − b + c = 0,7, u₂ = a(1 + p₂) − 2b + c = 0,7 + a·p₂ − b. Bei p = (0,4; 0,6) ist u₂ = 0,72,
    ū = 0,4·0,7 + 0,6·0,72 = 0,712, p₁' = 0,4·(1 + 0,5·(0,7 − 0,712)) = 0,3976, p₂' = 0,6·(1 + 0,5·0,008) = 0,6024. Der innere Punkt p₂ = b/a = ½ ist ein Fixpunkt."""
    return {"p": (0.4, 0.6), "u": (0.7, 0.72), "next": (0.3976, 0.6024), "fixed": (0.5, 0.5)}
