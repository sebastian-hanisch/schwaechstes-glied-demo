"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Rechenaufwand am Slider-Maximum (DEMO-PLAYBOOK §1c): ein Lauf der Lernregeln kostet O(Läufe · n · Stufen · Runden) (Logit: Auszahlungsvektor, Softmax, Ziehung je Spieler und Runde),
am Maximum (n = 32, 400 Runden, 1 000 Läufe) 12,8 Mio. Spielerrunden mit je 7 Stufen; die App begrenzt Läufe · n · Runden auf RUN_BUDGET und meldet die Kürzung.
Exakte Kette der besten Antwort: E^n Zustände (n = 3: 343, n = 4: 2 401 mit einem dichten linearen System von rund 6 Mio. Einträgen, ein bis zwei Sekunden).
Pfadaufzählung des Logit-Lernens: E^(n·Runden) Pfade, nur n = 2 mit höchstens 3 Runden live. Replikator: O(Schritte · Starts · Stufen) mit Abbruch je Start bei Konvergenz."""

# --- Spiel -----------------------------------------------------------------------------------------------------------------------------------
E_LEVELS = 7                     # Leistungsstufen 1 bis 7
MID_LEVEL = 4                    # Start „alle 4“
A_REF = 0.2                      # Bezugsprämie: die Schärfe gilt in Einheiten von a / A_REF, wenn „normieren“ an ist
DEFAULT_A, DEFAULT_B, DEFAULT_C = 0.2, 0.1, 0.6    # a: Zeitgewinn je Kettenstufe, b: Eigenaufwand je Stufe, c: Grundniveau (Modellparameter des Lehrbuchspiels)

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 2, 32, 8
A_MIN, A_MAX = 0.2, 2.0          # a ≥ 0,2 > b: alle sieben Stufen bleiben strikte Gleichgewichte
B_MIN, B_MAX = 0.01, 0.19
LAM_MIN, LAM_MAX, DEFAULT_LAM = 0.2, 200.0, 5.0     # Schärfe: Summe λ (Endschärfe λ/(1−φ)), Mittel Λ
PHI_MIN, PHI_MAX, DEFAULT_PHI = 0.0, 1.0, 0.9
EPS_MIN, EPS_MAX, DEFAULT_EPS = 0.0, 0.2, 0.01
PRIOR_MIN, PRIOR_MAX = 0, 60     # Vorabsprache: fiktive Runden „alle spielen 7“
T_MIN, T_MAX, DEFAULT_T = 10, 400, 60
RUNS_MIN, RUNS_MAX, DEFAULT_RUNS = 50, 1000, 200
SEED_MAX, DEFAULT_SEED = 999999, 35
PLAYER_ROUNDS_PER_SECOND = 4_000_000   # grobe Schätzung für die Zeitangabe (gemessen: 12,8 Mio. Spielerrunden in rund 2,5 s auf dem Entwicklungsrechner)
RUN_BUDGET = 10_000_000          # Läufe · n · Runden höchstens; darüber rechnet die App mit weniger Läufen und sagt es
FICT_LAM = 1000.0                # fiktives Spiel: Mittel über alle Runden (φ = 1) mit dieser Schärfe, praktisch die beste Antwort auf die gesamte Erfahrung

# --- Experimente ----------------------------------------------------------------------------------------------------------------------------
EXP_RUNS = 300                   # Läufe je Zelle der Live-Experimente
EXP_SEED = 20261028
POOL_SIZE = 16                   # E6: Pool der Teams
POOL_TEAMS = (2, 4, 8, 16)
E1_NS = (2, 3, 4, 6, 8, 16, 32)
E3_LIVE_MAX_STARTS = 500
E3_STEPS = 20_000
E3_ETA = 0.5
E4_LAMS_SUM = (0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 200.0)
E4_PHIS = (0.0, 0.5, 0.8, 0.9, 0.95, 1.0)
E5_BS = (0.01, 0.02, 0.05, 0.1, 0.15, 0.19)
E5_AS = (0.2, 0.3, 0.5, 1.0, 2.0)
E6_PRIORS = (0, 1, 2, 5, 10, 30)
GROW_START, GROW_EVERY = 2, 10   # H11: die Gruppe startet mit 2 und wächst alle 10 Runden um einen Spieler

# --- vorgerechnete Messreihen ------------------------------------------------------------------------------------------------------------------
MEAS_RUNS = 1000                 # Läufe je Zelle der vorgerechneten Reihen
MEAS_ROUNDS = (1, 10, 60, 150, 400)
BASIN_STARTS = 5000              # Startverteilungen je Zelle des Einzugsbereichs
BASIN_NS = (2, 3, 4, 6, 8, 16, 32)
BASIN_RATIOS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)   # b/a

# --- Farben ---------------------------------------------------------------------------------------------------------------------------------
COLOR_MIN = "#C0762B"            # Kettenminimum
COLOR_LINE = "#4c78a8"
COLOR_GOOD = "#2F6B65"
COLOR_RED = "#d62728"
COLOR_GREY = "rgba(150,150,150,0.65)"
LEVEL_COLORS = ("#a50026", "#d73027", "#f46d43", "#fdae61", "#74add1", "#4575b4", "#313695")

# --- Regeln und Start ------------------------------------------------------------------------------------------------------------------------
RULES = ("Logit, Summe (V4-Fassung)", "Logit, Mittel", "Beste Antwort mit Zittern", "Fiktives Spiel")
RULE_SUM, RULE_MEAN, RULE_BEST, RULE_FICT = RULES
STARTS = ("gleichverteilt", "alle 7", "alle 4")
START_UNIFORM, START_TOP, START_MID = STARTS

# --- Presets --------------------------------------------------------------------------------------------------------------------------------
BASE = {"rule": RULE_SUM, "start": START_UNIFORM, "n": DEFAULT_N, "a": DEFAULT_A, "b": DEFAULT_B, "lam": DEFAULT_LAM, "phi": DEFAULT_PHI, "eps": DEFAULT_EPS,
        "prior": 0, "normalize": True, "rounds": DEFAULT_T, "runs": DEFAULT_RUNS, "seed": DEFAULT_SEED}
PRESETS = {
    "Standard (V4-Hook, n = 8)": dict(BASE),
    "Kooperativer Start, Summe": {**BASE, "start": START_TOP},
    "Kooperativer Start, Mittel": {**BASE, "rule": RULE_MEAN, "start": START_TOP, "lam": 50.0, "rounds": 400},
    "Nur zwei": {**BASE, "n": 2},
    "Ohne Rauschen": {**BASE, "rule": RULE_BEST, "eps": 0.0, "rounds": 400},
    "Billiger Einsatz": {**BASE, "b": 0.02, "n": 4, "rounds": 400},
    "Prämie": {**BASE, "a": 2.0, "rounds": 400},
    "Vorabsprache verblasst": {**BASE, "prior": 30, "rounds": 400},
    "Prämie plus Vorabsprache": {**BASE, "n": 16, "a": 1.0, "prior": 5, "rounds": 400},
}
PRESET_HELP = {
    "Standard (V4-Hook, n = 8)": "Das Rezept des früheren Skripts V4: Logit-Summe, gleichverteilter Start, n = 8, 60 Runden. Schon Runde 1 liegt bei 1,37 (der Zufall), Runde 60 bei 1,00: der Abfall ist fast ganz der Start, nicht das Lernen (Mittel über 1 000 Läufe).",
    "Kooperativer Start, Summe": "Alle 8 Partner starten auf 7. Die Summenfassung bricht in den ersten Runden ein (Runde 10: 1,44; Runde 60: 1,15; Mittel über 1 000 Läufe): ein Anlaufeffekt, denn die Schärfe beginnt bei einem Zehntel ihres Endwerts.",
    "Kooperativer Start, Mittel": "Die Mittelfassung (Λ = 50) hält länger, bricht aber ein: Runde 60: 6,00; Runde 150: 4,90; Runde 400: 2,16 (n = 8, alle starten auf 7, Mittel über 1 000 Läufe). 60 Runden zeigen das nicht.",
    "Nur zwei": "Bei n = 2 steigt das Minimum: Runde 1: 2,79 (Zufall: 2,86), Runde 60: 3,91 (Mittel über 1 000 Läufe). In diesem Modell lernen zwei Partner auf etwa Stufe 4, nicht auf 7.",
    "Ohne Rauschen": "Beste Antwort ohne Zittern, n = 8: jeder spielt das Minimum der anderen. Das Minimum bleibt in allen 1 000 gemessenen Läufen von Runde 1 bis Runde 400 gleich (Mittel 1,38): was in Runde 1 ausgelost wird, bleibt stehen.",
    "Billiger Einsatz": "Einsatzpreis b = 0,02 statt 0,1, n = 4: das Minimum steigt (Runde 60: 5,57; Runde 400: 6,89; Mittel über 1 000 Läufe). Bei n = 8 reicht derselbe Preis nicht (Runde 60: 1,71); dazu den Regler n auf 8 stellen.",
    "Prämie": "Prämie a = 2,0 bei Schärfe in Einheiten der Prämie (b/a = 0,05), n = 8: das Minimum steigt langsam (Runde 60: 3,13; Runde 400: 4,96; Mittel über 1 000 Läufe), erreicht die 7 aber nicht. Ohne Normierung der Schärfe wären es 1,37 in Runde 400.",
    "Vorabsprache verblasst": "n = 8, Vorabsprache s = 30 (fiktive Runden „alle auf 7“, ein Anfangszustand, kein Reden): Runde 10: 7,00; Runde 60: 6,70; Runde 400: 2,62 (Mittel über 1 000 Läufe). Die Absprache verblasst.",
    "Prämie plus Vorabsprache": "n = 16, Prämie a = 1,0 und Vorabsprache s = 5 zusammen: Runde 400: 6,95, in 94,6 % der Läufe steht die ganze Kette auf 7 (Mittel über 1 000 Läufe). Allein hält keines von beiden: die Prämie gibt 1,03, die Absprache 1,00.",
}
