"""Kennzahlen der Hauptansicht und Verdikte (reine Funktionen: Art und Text, die App ruft `getattr(st, Art)(Text)`; DEMO-PLAYBOOK §12).

Jedes Verdikt zählt über die Zellen, die es beschreibt (k von m), und nennt die Ausnahme im selben Satz (DEMO-PLAYBOOK §11a). „Fällt“, „bleibt“ und „steigt“ heißen: der Unterschied zweier
Mittel liegt unter −4, innerhalb von ±4 oder über +4 Standardfehlern (Fehler der beiden Mittel quadratisch addiert, also eher streng)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import swg_constants as C
import swg_exact as X
import swg_experiments as XP
import swg_game as G
import swg_learning as L
import swg_scenario as S

Z = 4.0                       # Marge in Standardfehlern


# --- Formate ---------------------------------------------------------------------------------------------------------------------------------
def de(x: float, nd: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und echtem Minuszeichen (−)."""
    text = f"{x:.{nd}f}"
    if float(text) == 0.0:
        text = text.lstrip("-")                      # „−0,000“ gibt es nicht
    return text.replace(".", ",").replace("-", "−")


def thousands(n: int | float) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner."""
    return f"{int(round(n)):,}".replace(",", ".")


def pct(x: float, nd: int = 1) -> str:
    """Anteil als Prozentzahl mit Komma und Leerzeichen vor dem Prozentzeichen."""
    return f"{de(100.0 * x, nd)} %"


def big(x: float) -> str:
    """Eine Zeit in Runden: bis 10 000 ganz mit Tausenderpunkt, darüber als Zehnerpotenz (z. B. 9,4·10^8)."""
    if x < 10_000:
        return thousands(x) if x >= 100 else de(x, 1)
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant.replace('.', ',')}·10^{int(exp)}"


def ns_text(ns) -> str:
    """„n = 3, 4, 6“ bzw. „keine“."""
    ns = list(ns)
    return "n = " + ", ".join(str(n) for n in ns) if ns else "keine"


def rule_label(s: S.Settings) -> str:
    if s.rule == C.RULE_BEST:
        return f"{s.rule}, ε = {de(s.eps, 3)}"
    if s.rule == C.RULE_FICT:
        return s.rule
    return f"{s.rule}, Schärfe {de(s.lam, 1)}, Vergessen φ = {de(s.phi, 2)}"


def payoff_level(level: int, a: float, b: float, c: float) -> float:
    """Auszahlung eines Partners, wenn alle auf `level` sind: (a − b)·level + c."""
    return float(G.payoff(level, level, a, b, c))


# --- Hauptansicht -------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MainView:
    settings: S.Settings
    runs_used: int                  # Läufe, mit denen tatsächlich gerechnet wurde (Budget)
    mean_min: np.ndarray            # (Runden,) Mittel des Kettenminimums über die Läufe
    se_min: np.ndarray              # (Runden,) Standardfehler
    q25: np.ndarray
    q75: np.ndarray
    unchanged: float                # Anteil der Läufe, deren Minimum in der letzten Runde dem der ersten gleicht
    diff: float                     # Mittel von (Minimum letzte Runde − Minimum Runde 1)
    diff_se: float                  # Standardfehler dieses Mittels (Differenz je Lauf)
    random_min: float               # exakter Erwartungswert des Minimums bei gleichverteilter Wahl
    hist: np.ndarray                # (E,) Anteil der Läufe je Kettenminimum in der letzten Runde
    pay_end: float                  # mittlere Auszahlung je Partner in der letzten Runde
    pay_top: float
    pay_bottom: float
    trace: np.ndarray               # (Runden, n) Stufen aller Partner im ersten Lauf
    tolerance: float                # δ* für n und b/a

    @property
    def share_top(self) -> float:
        return float(self.hist[-1])


def analyse(s: S.Settings) -> MainView:
    """Ein Lauf der gewählten Lernregel mit höchstens S.budget_runs(s) Läufen und die Kennzahlen daraus."""
    used = S.budget_runs(s)
    run = L.simulate(S.with_(s, runs=used))
    m = run.mins
    r = run.runs
    d = m[:, -1] - m[:, 0]
    excl = G.excl_min(run.final)
    pay_end = float(G.payoff(run.final, excl, s.a, s.b, s.c).mean())
    hist = np.array([(m[:, -1] == lev).mean() for lev in range(1, s.levels + 1)])
    return MainView(
        settings=s, runs_used=r, mean_min=m.mean(axis=0), se_min=m.std(axis=0, ddof=1) / math.sqrt(r) if r > 1 else np.zeros(m.shape[1]),
        q25=np.percentile(m, 25, axis=0), q75=np.percentile(m, 75, axis=0), unchanged=float((d == 0).mean()), diff=float(d.mean()),
        diff_se=float(d.std(ddof=1) / math.sqrt(r)) if r > 1 else 0.0, random_min=X.uniform_expected_min(s.n, s.levels), hist=hist, pay_end=pay_end,
        pay_top=payoff_level(s.levels, s.a, s.b, s.c), pay_bottom=payoff_level(1, s.a, s.b, s.c), trace=run.trace, tolerance=G.tolerance(s.n, s.b / s.a))


def start_value(s: S.Settings) -> float | None:
    """Das Minimum zu Beginn, wenn es feststeht (alle 7: 7, alle 4: 4), sonst None (gleichverteilt: Zufall)."""
    return {C.START_TOP: float(s.levels), C.START_MID: float(min(C.MID_LEVEL, s.levels))}.get(s.start)


def peak_note(v: MainView) -> str:
    """Satz über einen Gipfel im Verlauf: liegt das mittlere Minimum in einer Zwischenrunde um mehr als 4 Standardfehler über Anfang und Ende, fällt es danach wieder (Vorabsprache)."""
    m = v.mean_min
    if len(m) < 3:
        return ""
    i = int(np.argmax(m))
    if i == 0 or i == len(m) - 1:
        return ""
    ends = max(m[0], m[-1])
    j = 0 if m[0] >= m[-1] else len(m) - 1
    if m[i] - ends <= Z * math.hypot(float(v.se_min[i]), float(v.se_min[j])):
        return ""
    return f" Zwischendurch erreicht das Mittel {de(float(m[i]), 2)} (Runde {i + 1}) und fällt danach wieder auf {de(float(m[-1]), 2)}."


def verdict_start_or_learning(v: MainView) -> tuple[str, str]:
    """Kernfrage der Hauptansicht: Wie viel des Minimums ist Start, wie viel Lernen? Zählt nur den Unterschied von Runde 1 zur letzten Runde."""
    s, t = v.settings, v.settings.rounds
    m1, mt = float(v.mean_min[0]), float(v.mean_min[-1])
    runs = v.runs_used
    if t == 1:
        return "info", f"Nur Runde 1: das Kettenminimum ist im Mittel {de(m1, 2)} (bei gleichverteilter Wahl exakt {de(v.random_min, 2)}). Ab Runde 2 wirkt die Lernregel."
    if v.unchanged == 1.0:
        return "info", (f"**Das Minimum bewegt sich nicht.** In allen {thousands(runs)} Läufen steht es in Runde {t} auf demselben Wert wie in Runde 1 (Mittel {de(m1, 2)}). "
                        f"{rule_label(s)}: " + ("Die Wahl ist so scharf, dass in diesen Läufen nie ein Partner von der Stufe des Minimums abwich; mit Rauschen in der Wahl wäre das Minimum nicht garantiert." if s.noisy else
                                "Die Regel enthält kein Rauschen, ein einzelner Partner kann das Minimum daher weder senken noch heben; was in Runde 1 ausgelost wurde (oder der Start vorgibt), bleibt stehen."))
    if abs(v.diff) <= Z * v.diff_se:
        return "info", (f"**Kein erkennbarer Unterschied** zwischen Runde 1 ({de(m1, 2)}) und Runde {t} ({de(mt, 2)}): die Differenz je Lauf liegt innerhalb von {Z:.0f} Standardfehlern "
                        f"({thousands(runs)} Läufe). Das Minimum ändert sich in einzelnen Läufen ({pct(1 - v.unchanged, 0)} der Läufe unterscheiden sich), im Mittel nicht." + peak_note(v))
    if v.diff > 0:
        return "success", (f"**Lernen hebt das Minimum:** von {de(m1, 2)} in Runde 1 auf {de(mt, 2)} in Runde {t} (Mittel über {thousands(runs)} Läufe). "
                           f"Bei gleichverteilter Wahl wäre es {de(v.random_min, 2)}; hier gilt {rule_label(s)} mit n = {s.n}." + peak_note(v))
    sv = start_value(s)
    if sv is not None:
        return "warning", (f"**Das Minimum fällt durch Lernen:** vom Start {de(sv, 0)} (Runde 1) auf {de(mt, 2)} in Runde {t} (Mittel über {thousands(runs)} Läufe). "
                           f"Der ganze Abfall ist Lernen, denn der Start ist kooperativ; {rule_label(s)} mit n = {s.n}." + peak_note(v))
    top = float(s.levels)
    share = XP.start_share(top, m1, mt)
    if share is None:
        return "info", f"Das Minimum liegt in Runde {t} bei {de(mt, 2)}."
    if share >= 0.5:
        who = "überwiegend der Start (Zufall in Runde 1)"
    else:
        who = "überwiegend das Lernen"
    return "warning", (f"**Von {de(top, 0)} auf {de(mt, 2)}: {who}.** Schon in Runde 1 liegt das Minimum bei {de(m1, 2)} (bei gleichverteilter Wahl exakt {de(v.random_min, 2)}): "
                       f"{pct(share, 0)} der Strecke sind Zufall, {pct(1 - share, 0)} kommen durch das Lernen (Mittel über {thousands(runs)} Läufe, {rule_label(s)}, n = {s.n}).")


def verdict_payoff(v: MainView) -> tuple[str, str]:
    """Wie viel der möglichen Verbesserung (alle auf 1 → alle auf 7) erreicht die Kette in der letzten Runde?"""
    s = v.settings
    span = v.pay_top - v.pay_bottom
    share = (v.pay_end - v.pay_bottom) / span if span > 0 else float("nan")
    return "info", (f"**Auszahlung in Runde {s.rounds}:** im Mittel {de(v.pay_end, 3)} je Partner und Sendung. Alle auf Stufe 1 bringen {de(v.pay_bottom, 2)}, alle auf Stufe {s.levels} "
                    f"{de(v.pay_top, 2)}; die Kette erreicht {pct(share, 0)} dieses Abstands. Jedes dieser sieben Profile ist ein Gleichgewicht: keiner verbessert sich allein.")


def verdict_tolerance(v: MainView) -> tuple[str, str]:
    """Fehlertoleranz je Partner δ* = 1 − (b/a)^(1/(n−1)): bis zu welcher Verfehlungswahrscheinlichkeit der anderen lohnt es sich, eine Stufe zu halten?"""
    s = v.settings
    return "info", (f"**Fehlertoleranz je Partner:** Eine Stufe zu halten lohnt sich gegenüber einer Stufe tiefer nur, solange jeder der {s.n - 1} anderen sie mit höchstens "
                    f"δ* = {pct(v.tolerance, 1)} verfehlt (b/a = {de(s.b / s.a, 2)}). Die Schwelle ist eine Rechnung aus dem Spiel und sagt nichts über die Lerngeschwindigkeit.")


# --- E1 Start oder Lernen --------------------------------------------------------------------------------------------------------------------
def classify(first: float, se_first: float, last: float, se_last: float) -> int:
    """−1 (fällt), 0 (bleibt) oder +1 (steigt): Unterschied von `last` und `first` gegen die quadratisch addierten Standardfehler."""
    se = math.hypot(se_first, se_last)
    d = last - first
    if abs(d) <= Z * se:
        return 0
    return 1 if d > 0 else -1


def verdict_e1(cells: dict, rule: str, start: str, top: float = float(C.E_LEVELS)) -> tuple[str, str]:
    """Zellen {n: Zelle} einer Regel und eines Starts, zwei oder mehr Runden: Wie viele Gruppengrößen fallen, bleiben, steigen, und wie groß ist der Anteil des Starts am Abfall?"""
    ns = sorted(cells, key=int)
    kinds = {n: classify(cells[n]["mean"][0], cells[n]["se"][0], cells[n]["mean"][-1], cells[n]["se"][-1]) for n in ns}
    falls = [n for n in ns if kinds[n] < 0]
    stays = [n for n in ns if kinds[n] == 0]
    rises = [n for n in ns if kinds[n] > 0]
    last_round = cells[ns[0]]["rounds"][-1]
    text = (f"**{rule}, Start {start}, Runde 1 gegen Runde {last_round}:** von {len(ns)} Gruppengrößen fällt das Minimum bei {len(falls)} ({ns_text(falls)}), "
            f"bleibt bei {len(stays)} ({ns_text(stays)}) und steigt bei {len(rises)} ({ns_text(rises)}).")
    if start == C.START_UNIFORM and falls:
        shares = {n: XP.start_share(top, cells[n]["mean"][0], cells[n]["mean"][-1]) for n in falls}
        shares = {n: v for n, v in shares.items() if v is not None}
        if shares:
            lo_n, hi_n = min(shares, key=shares.get), max(shares, key=shares.get)
            text += (f" Der Anteil des Starts (Runde 1 = Zufall) an der Strecke von {de(top, 0)} bis zum Wert der letzten Runde liegt bei den fallenden Gruppen zwischen "
                     f"{pct(shares[lo_n], 0)} (n = {lo_n}) und {pct(shares[hi_n], 0)} (n = {hi_n}); der Rest ist Lernen.")
    return ("warning" if falls else "info"), text


# --- E2 Ratsche ------------------------------------------------------------------------------------------------------------------------------
def verdict_e2(chain_rows: list[dict], drop_rows: list[dict], eps: float) -> tuple[str, str]:
    """Exakte Treffzeiten der Kette (Zeilen mit n, up, down) und die Formel der ersten Abweichung gegen die Simulation (Zeilen mit n, formula, sim, se, censored)."""
    first = "; ".join(f"n = {r['n']}: {big(r['up'])} Runden hinauf, {big(r['down'])} hinab" for r in chain_rows)
    ratios = [r["up"] / r["down"] for r in chain_rows]
    inside = [r for r in drop_rows if r["censored"] == 0 and abs(r["sim"] - r["formula"]) <= Z * max(r["se"], 1e-12)]
    out = [r["n"] for r in drop_rows if r not in inside]
    text = (f"**Ratsche bei Zittern ε = {de(eps, 3)}:** Erwartete Runden von „alle auf 1“ bis zum Minimum ≥ {C.MID_LEVEL} (hinauf) und von „alle auf {C.E_LEVELS}“ bis zum Minimum "
            f"≤ {C.MID_LEVEL - 1} (hinab): {first}. Hinauf dauert {de(ratios[0], 1)}-mal (n = {chain_rows[0]['n']}) bis {big(ratios[-1])}-mal (n = {chain_rows[-1]['n']}) so lange wie hinab. "
            f"Die erste Abweichung aus „alle {C.E_LEVELS}“ kommt nach 1 + 1/(1 − q^n) Runden: die Simulation liegt in {len(inside)} von {len(drop_rows)} Gruppengrößen innerhalb von "
            f"{Z:.0f} Standardfehlern der Formel" + (f" (nicht bei {ns_text(out)})." if out else "."))
    return "info", text


# --- E3 Einzugsbereich -----------------------------------------------------------------------------------------------------------------------
def verdict_e3(cells: dict, ratio: float) -> tuple[str, str]:
    """`cells`: {n: basin-Ergebnis} bei einem festen b/a: Anteil der Zufallsstarts, die bei der obersten Stufe enden, und bei Stufe 1."""
    ns = sorted(cells, key=int)
    tops = [cells[n]["share"][-1] for n in ns]
    monotone = all(x >= y - 1e-12 for x, y in zip(tops, tops[1:]))
    starts = cells[ns[0]]["starts"]
    parts = ", ".join(f"n = {n}: {pct(t, 1)}" for n, t in zip(ns, tops))
    return "info", (f"**Einzugsbereich der obersten Stufe bei b/a = {de(ratio, 2)}** (Anteil von {thousands(starts)} Zufallsstarts, die der Replikator bei Stufe {C.E_LEVELS} enden lässt): {parts}. "
                    f"{'Der Anteil sinkt mit der Gruppengröße in jedem Schritt.' if monotone else 'Der Anteil sinkt nicht in jedem Schritt mit der Gruppengröße.'}")


# --- Reihen: Einsatzpreis, Prämie, Schärfe, Vergessen, Vorabsprache (eine Spalte pro Zelle) ---------------------------------------------------------------------
def _fmt_series(cells: dict, key_fmt, idx: int) -> str:
    return ", ".join(f"{key_fmt(k)}: {de(cells[k]['mean'][idx], 2)}" for k in cells)


def is_flat(values, tol: float = 1e-9) -> bool:
    """Alle Werte gleich (bis `tol`)? Eine solche Spalte wäre ein Bug-Signal (Nullspalten-Regel), außer sie ist als Befund erklärt."""
    v = list(values)
    return bool(v) and max(v) - min(v) <= tol


def verdict_e4_sharpness(cells: dict, n: int, start: str, convention: str) -> tuple[str, str]:
    """Schärfe-Reihe: wo liegt das Minimum der letzten Runde am höchsten, ist die Reihe monoton, wo friert es am Startwert ein, wo bleibt es oben?"""
    ks = list(cells)
    last = [cells[k]["mean"][-1] for k in ks]
    best = ks[int(np.argmax(last))]
    lo, hi = min(last), max(last)
    mono = all(x >= y for x, y in zip(last, last[1:])) or all(x <= y for x, y in zip(last, last[1:]))
    top_level = C.E_LEVELS - 0.1
    stays_top = [k for k in ks if cells[k]["mean"][-1] > top_level]
    frozen = [k for k in ks if k not in stays_top and abs(cells[k]["mean"][-1] - cells[k]["mean"][0]) <= Z * math.hypot(cells[k]["se"][0], cells[k]["se"][-1])]
    text = (f"**Schärfe-Reihe ({'Summe' if convention == 'sum' else 'Mittel'}), n = {n}, Start {start}:** das Minimum der letzten Runde liegt zwischen {de(lo, 2)} und {de(hi, 2)}, am höchsten bei Schärfe "
            f"{de(best, 1)}; die Reihe ist {'monoton' if mono else 'nicht monoton'} in der Schärfe. ")
    if stays_top:
        text += f"Bei Schärfe {', '.join(de(k, 1) for k in stays_top)} bleibt das Minimum in der letzten Runde über {de(top_level, 1)}. "
    if frozen:
        text += f"Bei Schärfe {', '.join(de(k, 1) for k in frozen)} ist das Minimum am Ende nicht von dem in Runde 1 zu unterscheiden (Einfrieren am Startwert, innerhalb von {Z:.0f} Standardfehlern). "
    return "info", text.rstrip()


def verdict_e4_forgetting(cells: dict, n: int, start: str) -> tuple[str, str]:
    ks = list(cells)
    last = [cells[k]["mean"][-1] for k in ks]
    return "info", (f"**Vergessen (Mittelfassung), n = {n}, Start {start}, Runde {cells[ks[0]]['rounds'][-1]}:** " + ", ".join(f"φ = {de(k, 2)}: {de(v, 2)}" for k, v in zip(ks, last)) + ". "
                    f"Das höchste Minimum hat φ = {de(ks[int(np.argmax(last))], 2)}, das niedrigste φ = {de(ks[int(np.argmin(last))], 2)}.")


def verdict_e5_price(cells: dict, n: int) -> tuple[str, str]:
    ks = list(cells)
    last = [cells[k]["mean"][-1] for k in ks]
    top = [cells[k]["top"][-1] for k in ks]
    rounds = cells[ks[0]]["rounds"][-1]
    text = f"**Einsatzpreis, n = {n}, Runde {rounds} (a = 0,2):** " + ", ".join(f"b = {de(k, 2)}: {de(v, 2)}" for k, v in zip(ks, last)) + ". "
    if max(top) > 0:
        text += f"Läufe mit Minimum {C.E_LEVELS} in Runde {rounds}: bis {pct(max(top), 1)} (b = {de(ks[int(np.argmax(top))], 2)})."
    else:
        text += f"In keinem Preis erreicht ein Lauf in Runde {rounds} das Minimum {C.E_LEVELS}."
    return "info", text


def verdict_e5_premium(norm: dict, raw: dict, n: int) -> tuple[str, str]:
    ks = list(norm)
    rounds = norm[ks[0]]["rounds"][-1]
    worse = [k for k in ks if classify(raw[k]["mean"][-1], raw[k]["se"][-1], norm[k]["mean"][-1], norm[k]["se"][-1]) > 0]
    return "info", (f"**Prämie, n = {n}, Runde {rounds} (b = 0,1):** mit auf die Prämie normierter Schärfe " + ", ".join(f"a = {de(k, 1)}: {de(norm[k]['mean'][-1], 2)}" for k in ks) +
                    "; ohne Normierung (die Prämie macht die Wahl zugleich schärfer) " + ", ".join(f"{de(raw[k]['mean'][-1], 2)}" for k in ks) +
                    f". Die normierte Reihe liegt bei {len(worse)} von {len(ks)} Prämien über der nicht normierten ({', '.join('a = ' + de(k, 1) for k in worse) if worse else 'keine'}).")


def verdict_e6_teams(cells: dict) -> tuple[str, str]:
    """`cells`: {"k|fest": Zelle}; Vergleich fest gegen jede Runde neu gelost je Teamgröße."""
    ks = sorted({int(k.split("|")[0]) for k in cells})
    parts = []
    better = []
    for k in ks:
        f, r = cells[f"{k}|1"], cells[f"{k}|0"]
        parts.append(f"k = {k}: fest {de(f['mean'][-1], 2)}, gelost {de(r['mean'][-1], 2)}")
        if classify(f["mean"][-1], f["se"][-1], r["mean"][-1], r["se"][-1]) > 0:
            better.append(k)
    return "info", (f"**Teams im Pool von {C.POOL_SIZE}, Team-Minimum in Runde {cells[f'{ks[0]}|1']['rounds'][-1]}:** " + "; ".join(parts) + ". "
                    f"Jede Runde neu gelöste Teams sind bei {len(better)} von {len(ks)} Teamgrößen besser als feste ({ns_text(better).replace('n = ', 'k = ')}).")


def verdict_e6_prior(cells: dict, n: int) -> tuple[str, str]:
    ks = list(cells)
    rounds = cells[ks[0]]["rounds"]
    big_s = ks[-1]
    return "info", (f"**Vorabsprache, n = {n} (fiktive Runden „alle oben“, erste Wahl gleichverteilt):** s = {big_s}: Minimum in Runde {rounds[1]}: {de(cells[big_s]['mean'][1], 2)}, "
                    f"Runde {rounds[2]}: {de(cells[big_s]['mean'][2], 2)}, Runde {rounds[-1]}: {de(cells[big_s]['mean'][-1], 2)}. Ohne Absprache (s = 0) in Runde {rounds[-1]}: {de(cells[ks[0]]['mean'][-1], 2)}.")


def verdict_e6_combo(cells: dict, n: int) -> tuple[str, str]:
    keys = list(cells)
    r = cells[keys[0]]["rounds"][-1]
    return "info", (f"**Prämie und Vorabsprache, n = {n}, Runde {r}:** " + "; ".join(f"a = {de(float(k.split('|')[0]), 1)}, s = {k.split('|')[1]}: {de(cells[k]['mean'][-1], 2)}" for k in keys) + ".")


def first_round_below(cell: dict, level: float) -> int | None:
    """Erste der gemessenen Runden, in der das Mittel unter `level` liegt, sonst None."""
    for r, m in zip(cell["rounds"], cell["mean"]):
        if m < level:
            return r
    return None


def verdict_e6_grow(cells: dict, n: int, convention: str) -> tuple[str, str]:
    """Wachsende Gruppen gegen die von Anfang an volle Gruppe: Minimum in den gemessenen Runden, erste gemessene Runde mit Mittel unter 4 und Vergleich in der letzten Runde."""
    keys = list(cells)
    rounds = cells[keys[0]]["rounds"]
    ref = cells["fest"]
    parts = []
    for k in keys:
        below = first_round_below(cells[k], float(C.MID_LEVEL))
        parts.append(f"{k}: " + " / ".join(de(cells[k]["mean"][i], 2) for i in range(len(rounds))) + f" (unter {C.MID_LEVEL} ab Runde {below if below is not None else 'nie'})")
    kinds = [classify(ref["mean"][-1], ref["se"][-1], cells[k]["mean"][-1], cells[k]["se"][-1]) for k in keys if k != "fest"]
    better = sum(1 for x in kinds if x > 0)
    never = 10 ** 9
    ref_below = first_round_below(ref, float(C.MID_LEVEL)) or never
    later = sum(1 for k in keys if k != "fest" and (first_round_below(cells[k], float(C.MID_LEVEL)) or never) > ref_below)
    return "info", (f"**Wachsende Gruppen ({'Summe' if convention == 'sum' else 'Mittel'}), n = {n}, kooperativer Start, Minimum in den Runden {', '.join(str(r) for r in rounds)}:** " + "; ".join(parts) +
                    f". Der Einbruch unter {C.MID_LEVEL} kommt bei {later} von {len(kinds)} Taktungen später als bei der von Anfang an vollen Gruppe; in der letzten Runde ist das Wachsen bei {better} von {len(kinds)} Taktungen besser.")
