"""Experimente E1 bis E6: Zellen und Tabellen aus dem Kern (live, mit wenigen Läufen) und die vorgerechneten Messreihen (data/measurements.json, mit vielen Läufen).

Eine Zelle ist ein Lauf `swg_learning.simulate` mit festem Zufalls-Seed; ausgewiesen werden Mittel und Standardfehler des Kettenminimums in ausgewählten Runden.
Die vorgerechneten Reihen (tools/make_measurements.py) benutzen genau diese Funktionen mit C.MEAS_RUNS Läufen und tragen den Hash der Dateien, aus deren Code sie stammen."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

import swg_constants as C
import swg_exact as X
import swg_learning as L
import swg_replicator as RP
import swg_scenario as S

ROOT = Path(__file__).resolve().parent
MEAS_FILE = ROOT / "data" / "measurements.json"
HASHED_FILES = ("swg_game.py", "swg_learning.py", "swg_scenario.py", "swg_replicator.py", "swg_exact.py", "swg_experiments.py", "swg_rng.py")

# Regeln der Vergleichstabellen: (Name, Settings-Felder). Schärfe und Vergessen wie im früheren Skript V4 (Summe λ = 5, φ = 0,9) bzw. Mittel mit Λ = 50.
RULE_CONFIGS = {
    "Logit, Summe (λ = 5)": dict(rule=C.RULE_SUM, lam=5.0, phi=0.9),
    "Logit, Mittel (Λ = 50)": dict(rule=C.RULE_MEAN, lam=50.0, phi=0.9),
    "Beste Antwort (ε = 0,01)": dict(rule=C.RULE_BEST, eps=0.01),
    "Beste Antwort (ε = 0)": dict(rule=C.RULE_BEST, eps=0.0),
    "Fiktives Spiel": dict(rule=C.RULE_FICT),
}


# --------------------------------------------------------------------------------- Hilfen
def seed_of(label: str) -> int:
    """Zufalls-Seed aus einer Zellenbezeichnung (FNV-1a über die UTF-8-Bytes; Python-`hash` ist je Prozess gesalzen und taugt nicht)."""
    h = 0xCBF29CE484222325
    for byte in label.encode("utf-8"):
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h % (C.SEED_MAX + 1)


def cell(s: S.Settings, rounds: tuple[int, ...], *, team: int | None = None, fixed: bool = True, grow: tuple[int, int] | None = None) -> dict:
    """Mittel, Standardfehler und Anteil „oberste Stufe“ des Kettenminimums in den Runden `rounds` (1-gezählt); gerechnet wird bis zur größten davon."""
    run = L.simulate(S.with_(s, rounds=max(rounds)), team=team, fixed=fixed, grow=grow)
    m = run.mins
    idx = [r - 1 for r in rounds]
    r = run.runs
    return {"rounds": list(rounds), "runs": r, "mean": [float(m[:, i].mean()) for i in idx], "se": [float(m[:, i].std(ddof=1) / math.sqrt(r)) if r > 1 else 0.0 for i in idx],
            "top": [float(run.top[:, i].mean()) for i in idx]}


def base_settings(runs: int, seed_label: str, **fields) -> S.Settings:
    return S.Settings(runs=runs, seed=seed_of(seed_label), **fields)


# --------------------------------------------------------------------------------- E1 Regel × n × Start
def e1_cells(start: str, runs: int, rounds=C.MEAS_ROUNDS, ns=C.E1_NS, configs=RULE_CONFIGS) -> dict:
    """Regel × Gruppengröße bei gegebenem Start: {Regelname: {n: Zelle}}. Gleiche Seed-Bezeichnung für alle n einer Regel ist kein Problem: die Läufe sind je n verschieden lang."""
    out = {}
    for name, fields in configs.items():
        out[name] = {n: cell(base_settings(runs, f"e1|{name}|{start}|{n}", n=n, start=start, **fields), rounds) for n in ns}
    return out


def e1_live(s: S.Settings, runs: int, rounds=(1, 10, 60), ns=C.E1_NS) -> dict:
    """Dieselbe Reihe wie E1, aber mit den Einstellungen der Seitenleiste (Regel, Schärfe, Start, Preise): {n: Zelle}."""
    return {n: cell(S.with_(s, n=n, runs=runs, seed=seed_of(f"e1live|{s.seed}|{n}")), rounds) for n in ns}


def start_share(start_value: float, m_first: float, m_end: float) -> float | None:
    """Anteil des Abfalls von `start_value` (kooperativ: oberste Stufe) auf `m_end`, der schon in Runde 1 geschehen ist: (start_value − m1) / (start_value − m_end).
    None, wenn das Minimum nicht unter den Startwert gefallen ist."""
    total = start_value - m_end
    if total <= 1e-9:
        return None
    return (start_value - m_first) / total


# --------------------------------------------------------------------------------- E2 Ratsche
def e2_chain_rows(eps: float, ns=(2, 3, 4)) -> list[dict]:
    """Exakte Zeiten der Kette der besten Antwort mit Zittern (alle levels^n Profile): von „alle 1“ bis Minimum ≥ 4, von „alle 7“ bis Minimum ≤ 3, stationäres E[min]."""
    rows = []
    for n in ns:
        mat, mins = X.br_chain(n, eps)
        rows.append({"n": n, "up": X.hitting_time(mat, 0, mins >= C.MID_LEVEL), "down": X.hitting_time(mat, len(mat) - 1, mins <= C.MID_LEVEL - 1),
                     "stationary": float(X.stationary(mat) @ mins), "states": len(mat)})
    return rows


def e2_first_drop_rows(eps: float, runs: int, ns=(2, 4, 8, 16, 32), rounds: int = 400, seed_label: str = "e2") -> list[dict]:
    """Erste Runde, in der aus „alle 7“ das Minimum unter 7 fällt: Formel 1 + 1/(1 − q^n) gegen den Mittelwert der Simulation (Standardfehler der Läufe)."""
    rows = []
    for n in ns:
        s = base_settings(runs, f"{seed_label}|{n}|{eps}", n=n, rule=C.RULE_BEST, eps=eps, start=C.START_TOP, rounds=rounds)
        m = L.simulate(s).mins
        below = m < C.E_LEVELS
        first = np.where(below.any(axis=1), below.argmax(axis=1) + 1, rounds + 1)
        censored = float((~below.any(axis=1)).mean())
        rows.append({"n": n, "formula": X.first_drop_rounds(n, eps), "sim": float(first.mean()), "se": float(first.std(ddof=1) / math.sqrt(len(first))), "censored": censored})
    return rows


def unchanged_share(runs: int, ns=C.E1_NS, rounds: int = 400) -> dict:
    """Rauschfreie Regeln (beste Antwort ohne Zittern, fiktives Spiel), gleichverteilter Start: Anteil der Läufe, deren Minimum in der letzten Runde dem von Runde 1 gleicht, je n."""
    out = {}
    for name, fields in (("Beste Antwort (ε = 0)", RULE_CONFIGS["Beste Antwort (ε = 0)"]), ("Fiktives Spiel", RULE_CONFIGS["Fiktives Spiel"])):
        out[name] = {}
        for n in ns:
            m = L.simulate(base_settings(runs, f"h4|{name}|{n}", n=n, rounds=rounds, **fields)).mins
            out[name][n] = {"runs": runs, "unchanged": float((m[:, -1] == m[:, 0]).mean()), "mean_first": float(m[:, 0].mean()), "mean_last": float(m[:, -1].mean())}
    return out


# --------------------------------------------------------------------------------- E3 Einzugsbereich
def e3_basin_grid(starts: int, ns=C.BASIN_NS, ratios=C.BASIN_RATIOS) -> dict:
    """Einzugsbereiche des Replikators: {(n, b/a): basin-Ergebnis}."""
    return {f"{n}|{r}": RP.basin(n, r, starts, seed_of(f"basin|{n}|{r}")) for n in ns for r in ratios}


# --------------------------------------------------------------------------------- E4 Schärfe und Gedächtnis
def e4_sharpness_cells(start: str, n: int, runs: int, convention: str, rounds=(1, 10, 60), lams=C.E4_LAMS_SUM) -> dict:
    """Schärfe-Reihe: {λ: Zelle}. Summe: λ aus E4_LAMS_SUM, Mittel: Λ = 10·λ (gleiche Endschärfe λ/(1−φ) bei φ = 0,9)."""
    out = {}
    for lam in lams:
        fields = dict(rule=C.RULE_SUM, lam=lam) if convention == "sum" else dict(rule=C.RULE_MEAN, lam=lam * 10.0)
        out[lam] = cell(base_settings(runs, f"e4|{convention}|{start}|{n}|{lam}", n=n, start=start, **fields), rounds)
    return out


def e4_forgetting_cells(start: str, n: int, runs: int, rounds=(1, 10, 60), phis=C.E4_PHIS) -> dict:
    """Vergessen-Reihe in der Mittelfassung (Schärfe 50): {φ: Zelle}."""
    return {phi: cell(base_settings(runs, f"e4phi|{start}|{n}|{phi}", n=n, start=start, rule=C.RULE_MEAN, lam=50.0, phi=phi), rounds) for phi in phis}


# --------------------------------------------------------------------------------- E5 Einsatzpreis und Prämie
def e5_price_cells(n: int, runs: int, rounds=(1, 10, 60), prices=C.E5_BS) -> dict:
    """Einsatzpreis-Reihe (a = 0,2, Summe, λ = 5, gleichverteilter Start): {b: Zelle}."""
    return {b: cell(base_settings(runs, f"e5b|{n}|{b}", n=n, b=b), rounds) for b in prices}


def e5_premium_cells(n: int, runs: int, normalize: bool, rounds=(1, 10, 60), premiums=C.E5_AS) -> dict:
    """Prämien-Reihe (b = 0,1, Summe, λ = 5, gleichverteilter Start): {a: Zelle}. `normalize`: Schärfe in Einheiten von a (reines Verhältnis b/a) oder in Auszahlungseinheiten."""
    return {a: cell(base_settings(runs, f"e5a|{n}|{a}|{normalize}", n=n, a=a, normalize=normalize), rounds) for a in premiums}


# --------------------------------------------------------------------------------- E6 Teams, Vorabsprache, Kombination, wachsende Gruppen
def e6_team_cells(runs: int, rounds=(1, 10, 60), pool: int = C.POOL_SIZE, teams=C.POOL_TEAMS) -> dict:
    """Pool von `pool` Spielern in Teams der Größe k: {(k, fest?): Zelle}, Kennzahl ist das Mittel der Team-Minima (top = Anteil Teams mit oberstem Minimum)."""
    out = {}
    for k in teams:
        for fixed in (True, False):
            out[f"{k}|{int(fixed)}"] = cell(base_settings(runs, f"e6t|{k}|{fixed}", n=pool), rounds, team=k, fixed=fixed)
    return out


def e6_prior_cells(n: int, runs: int, rounds=(1, 10, 60, 400), priors=C.E6_PRIORS) -> dict:
    """Vorabsprache s (fiktive Runden „alle oben“) bei gleichverteilter erster Wahl, Summe: {s: Zelle}."""
    return {s_: cell(base_settings(runs, f"e6p|{n}|{s_}", n=n, prior=float(s_)), rounds) for s_ in priors}


def e6_combo_cells(n: int, runs: int, rounds=(1, 10, 60, 400)) -> dict:
    """Prämie und Vorabsprache zusammen (b = 0,1): {(a, s): Zelle}, Schärfe normiert."""
    combos = ((0.2, 0), (0.2, 5), (1.0, 0), (1.0, 5), (1.0, 30))
    return {f"{a}|{s_}": cell(base_settings(runs, f"e6c|{n}|{a}|{s_}", n=n, a=a, prior=float(s_)), rounds) for a, s_ in combos}


def e6_grow_cells(n: int, runs: int, convention: str, rounds=(1, 10, 60, 150, 400)) -> dict:
    """Wachsende Gruppen (H11): Start mit 2 Spielern, alle `every` Runden einer mehr bis n; gegen die von Anfang an volle Gruppe. Kooperativer Start (alle oben)."""
    fields = dict(rule=C.RULE_SUM, lam=5.0) if convention == "sum" else dict(rule=C.RULE_MEAN, lam=50.0)
    out = {"fest": cell(base_settings(runs, f"e6g|{n}|{convention}|fest", n=n, start=C.START_TOP, **fields), rounds)}
    for every in (10, 25):
        out[f"alle {every}"] = cell(base_settings(runs, f"e6g|{n}|{convention}|{every}", n=n, start=C.START_TOP, **fields), rounds, grow=(C.GROW_START, every))
    return out


# --------------------------------------------------------------------------------- vorgerechnete Reihen
def generator_hash() -> str:
    """Hash der Dateien, aus deren Code die Messreihen stammen (Zeilenenden normalisiert, damit CRLF/LF denselben Hash geben)."""
    h = hashlib.sha256()
    for name in HASHED_FILES:
        text = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        h.update(name.encode("utf-8") + b"\0" + text + b"\0")
    return h.hexdigest()[:16]


def measure_all(runs: int = C.MEAS_RUNS, starts: int = C.BASIN_STARTS, progress=None) -> dict:
    """Alle vorgerechneten Reihen (ein Wörterbuch, JSON-fähig; Schlüssel der Wörterbücher als Text)."""
    def note(msg):
        if progress:
            progress(msg)

    out: dict = {"runs": runs, "basin_starts": starts, "rounds": list(C.MEAS_ROUNDS)}
    note("E1 Regel × n (gleichverteilter Start)")
    out["e1_uniform"] = e1_cells(C.START_UNIFORM, runs)
    note("E1 Regel × n (alle 7)")
    out["e1_top"] = e1_cells(C.START_TOP, runs)
    note("H4 unverändertes Minimum bei rauschfreien Regeln")
    out["unchanged"] = unchanged_share(runs)
    note("E3 Einzugsbereiche")
    out["basin"] = e3_basin_grid(starts)
    note("E4 Schärfe und Vergessen")
    out["sharp_sum"] = {str(n): e4_sharpness_cells(C.START_UNIFORM, n, runs, "sum") for n in (2, 4, 8)}
    out["sharp_sum_top"] = {str(n): e4_sharpness_cells(C.START_TOP, n, runs, "sum") for n in (2, 4, 8)}
    out["sharp_mean_top"] = {str(n): e4_sharpness_cells(C.START_TOP, n, runs, "mean", rounds=(1, 10, 60, 400)) for n in (4, 8)}
    out["phi_mean_top"] = {str(n): e4_forgetting_cells(C.START_TOP, n, runs, rounds=(1, 10, 60, 400)) for n in (4, 8)}
    out["phi_mean_uniform"] = {str(n): e4_forgetting_cells(C.START_UNIFORM, n, runs) for n in (4,)}
    note("E5 Einsatzpreis und Prämie")
    out["price"] = {str(n): e5_price_cells(n, runs, rounds=(1, 10, 60, 400)) for n in (2, 4, 8, 16)}
    out["premium_norm"] = {str(n): e5_premium_cells(n, runs, True, rounds=(1, 10, 60, 400)) for n in (2, 4, 8, 16)}
    out["premium_raw"] = {str(n): e5_premium_cells(n, runs, False, rounds=(1, 10, 60, 400)) for n in (2, 4, 8, 16)}
    note("E6 Teams, Vorabsprache, Kombination, wachsende Gruppen")
    out["teams"] = e6_team_cells(runs)
    out["prior"] = {str(n): e6_prior_cells(n, runs) for n in (2, 4, 8, 16)}
    out["combo"] = {str(n): e6_combo_cells(n, runs) for n in (8, 16)}
    out["grow"] = {f"{n}|{conv}": e6_grow_cells(n, runs, conv) for n in (8, 16) for conv in ("sum", "mean")}
    note("E2 Ratsche: erste Abweichung")
    out["first_drop"] = {str(eps): e2_first_drop_rows(eps, 5 * runs) for eps in (0.01, 0.05)}          # billig (beste Antwort), mehr Läufe für die enge Prüfung gegen die Formel
    return out


def to_jsonable(obj):
    """Wörterbücher mit Zahlen- oder Tupelschlüsseln in JSON-fähige (Text-)Schlüssel umwandeln."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def write_measurements(progress=None) -> dict:
    payload = {"generator_hash": generator_hash(), **to_jsonable(measure_all(progress=progress))}
    MEAS_FILE.parent.mkdir(exist_ok=True)
    MEAS_FILE.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    return payload


def load_measurements() -> dict | None:
    """Die vorgerechneten Reihen, oder None, wenn die Datei fehlt oder zu einem anderen Rechencode gehört (dann zeigt die App sie nicht)."""
    if not MEAS_FILE.exists():
        return None
    data = json.loads(MEAS_FILE.read_text(encoding="utf-8"))
    return data if data.get("generator_hash") == generator_hash() else None


# --------------------------------------------------------------------------------- exakte Bezugskurve der Hauptansicht
def exact_overlay(s: S.Settings) -> dict | None:
    """Exakte Bezugskurve zu den Einstellungen, wenn es sie gibt: Markov-Kette der besten Antwort mit Zittern (n ≤ 3, alle Runden) oder Pfadaufzählung des Logit
    (n = 2, die ersten drei Runden). Sonst None."""
    if s.rule == C.RULE_BEST and s.n <= 3:
        mat, mins = X.br_chain(s.n, s.eps, s.a, s.b, s.c, s.levels)
        v0 = None
        if s.start == C.START_TOP:
            v0 = X.point_start(s.n, s.levels, s.levels)
        elif s.start == C.START_MID:
            v0 = X.point_start(s.n, min(C.MID_LEVEL, s.levels), s.levels)
        y = X.chain_expected_min(mat, mins, s.rounds, v0)
        return {"label": "exakt (Markov-Kette über alle Profile)", "x": list(range(1, s.rounds + 1)), "y": [float(v) for v in y]}
    if s.rule in (C.RULE_SUM, C.RULE_MEAN) and s.n == 2 and s.rounds >= 3:
        y = X.logit_exact_path(s, 3)
        return {"label": "exakt (Pfadaufzählung, Runde 1 bis 3)", "x": [1, 2, 3], "y": [float(v) for v in y]}
    return None
