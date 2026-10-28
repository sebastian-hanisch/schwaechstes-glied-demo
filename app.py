"""Das schwächste Glied: warum die Kette langsam wird - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Reihe Koordinationsspiele. Eine Umschlagkette aus n Partnern läuft mit der langsamsten Leistungsstufe; es gibt sieben gleich gute Gleichgewichte. Die Demo lässt vier
Lernregeln auf diesem Spiel laufen und fragt, was Start und was Lernen ist, warum das Minimum nach unten rutscht (Ratsche), wie klein der Einzugsbereich der obersten Stufe wird
und ob Einsatzpreis, Prämie, Teams, Vorabsprache oder wachsende Gruppen helfen.

Lauffähig mit: streamlit run app.py
"""

import math

import streamlit as st

import swg_constants as C
import swg_evaluation as V
import swg_exact as X
import swg_experiments as XP
import swg_replicator as RP
import swg_scenario as S
import swg_visualization as Z
from swg_layout import dunkles_design, kopf_css
from swg_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    on_rule_change,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Das schwächste Glied: warum die Kette langsam wird – Sebastian Hanisch", layout="wide")


# --- Zwischenspeicher (Schlüssel = alle Eingaben inklusive Zufalls-Seed) --------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _meas():
    return XP.load_measurements()


@st.cache_data(show_spinner=False)
def _analyse(s):
    return V.analyse(s)


@st.cache_data(show_spinner=False)
def _overlay(s):
    return XP.exact_overlay(s)


@st.cache_data(show_spinner=False)
def _e1_live(s):
    return XP.e1_live(s, C.EXP_RUNS)


@st.cache_data(show_spinner=False)
def _chain_rows(eps):
    return XP.e2_chain_rows(eps)


@st.cache_data(show_spinner=False)
def _drop_rows(eps):
    return XP.e2_first_drop_rows(eps, C.EXP_RUNS)


@st.cache_data(show_spinner=False)
def _basin_live(n, ratio, starts, seed):
    p0 = RP.random_starts(starts, seed)
    p, used = RP.run(p0, n, C.A_REF, C.A_REF * ratio, C.DEFAULT_C, C.E3_STEPS)
    lev = RP.end_level(p)
    share = [float((lev == k).mean()) for k in range(1, C.E_LEVELS + 1)]
    return {"n": n, "ratio": ratio, "starts": starts, "none": float((lev == 0).mean()), "share": share, "emin": float((RP.survival(p) ** n).sum(axis=-1).mean()),
            "max_steps": int(used.max()), "first": p0[0]}


@st.cache_data(show_spinner=False)
def _sharp(convention, start, n):
    return XP.e4_sharpness_cells(start, n, C.EXP_RUNS, convention, rounds=(1, 10, 60))


@st.cache_data(show_spinner=False)
def _forget(start, n):
    return XP.e4_forgetting_cells(start, n, C.EXP_RUNS, rounds=(1, 10, 60))


@st.cache_data(show_spinner=False)
def _price(n):
    return XP.e5_price_cells(n, C.EXP_RUNS, rounds=(1, 10, 60, 400))


@st.cache_data(show_spinner=False)
def _premium(n, normalize):
    return XP.e5_premium_cells(n, C.EXP_RUNS, normalize, rounds=(1, 10, 60, 400))


@st.cache_data(show_spinner=False)
def _teams():
    return XP.e6_team_cells(C.EXP_RUNS)


@st.cache_data(show_spinner=False)
def _prior(n):
    return XP.e6_prior_cells(n, C.EXP_RUNS)


@st.cache_data(show_spinner=False)
def _combo(n):
    return XP.e6_combo_cells(n, C.EXP_RUNS)


@st.cache_data(show_spinner=False)
def _grow(n, convention):
    return XP.e6_grow_cells(n, C.EXP_RUNS, convention)


def show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


def table(rows):
    st.dataframe(rows, hide_index=True, width="stretch")


def seconds_text(cells: int, runs: int, n: int, rounds: int) -> int:
    """Grobe Sekundenzahl für `cells` Zellen zu je runs · n · rounds Spielerrunden (nur für die Angabe am Knopf)."""
    return max(1, math.ceil(cells * runs * n * rounds / C.PLAYER_ROUNDS_PER_SECOND))


def cell_rows(cells: dict, label: str, fmt) -> list[dict]:
    """Tabellenzeilen aus Zellen {Schlüssel: Zelle}: Mittel je ausgewählter Runde und Anteil „oberste Stufe“ in der letzten."""
    rows = []
    for k, c in cells.items():
        row = {label: fmt(k)}
        for r, m in zip(c["rounds"], c["mean"]):
            row[f"Runde {r}"] = V.de(m, 2)
        row[f"Anteil Minimum {C.E_LEVELS} (Runde {c['rounds'][-1]})"] = V.pct(c["top"][-1], 1)
        rows.append(row)
    return rows


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in swg_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🔗 Das schwächste Glied: warum die Kette langsam wird")
st.markdown(
    """
**Fünftes Stück der Reihe Koordinationsspiele.** Das erste Stück (`koordinationsspiel-demo`) zeigt die Gleichgewichte eines Koordinationsspiels, das zweite (`gleichgewichtsauswahl-demo`), welches davon auf Dauer gewählt wird,
das dritte (`konventionswechsel-demo`), wie lange ein Wechsel dauert, das vierte (`globale-spiele-demo`), was Unsicherheit über den Zustand ändert. Hier ist das Spiel eine **Umschlagkette aus n Partnern**:
jeder wählt eine **Leistungsstufe 1 bis 7** (Schichtbesetzung, Torbelegung), und die Kette läuft mit der **langsamsten** Stufe. Jede Stufe, auf der alle Partner stehen, ist ein Gleichgewicht: **sieben gleich gut erreichbare Gleichgewichte.**
**Welches wählt eine Gruppe, die aus Erfahrung lernt, und was ändert sich mit Gruppengröße, Einsatzpreis und Vorabsprache?** Die Antwort in diesem Modell ist eine **Ratsche**: ein einzelner Fehler senkt das Minimum sofort,
es wieder zu heben müssen alle Partner zugleich höher gehen. Gezählt und gemessen, nicht behauptet.

Abgrenzung: `noregret-demo` lässt Lkw einander **meiden** (Überlastungsspiel, Hedge-Lernen), `replikator-demo` rechnet den Replikator für Strategien wiederholter Spiele, `nash-demo` Gleichgewichte kleiner Matrixspiele.
Hier ist das Ergebnis das **Minimum** der Gruppe, vier Lernregeln laufen auf demselben Spiel, und die Frage lautet: was davon ist **Start** und was **Lernen**?
**Lernregeln und Eigennutz sind Modellannahmen**, keine Messung an Menschen oder echten Umschlagstellen.

Weiter unten: sechs Tabs mit je einem **🔬 Experiment**, die Abschnitte **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Das Spiel", expanded=True):
    st.markdown(
        r"""
1. **Leistungsstufe und Auszahlung.** Jeder der $n$ Partner wählt eine Stufe $x \in \{1, \dots, 7\}$. Die Kette läuft mit dem Minimum. Ein Partner, dessen Mitspieler zusammen das Minimum $m$ haben, bekommt
   $a \cdot \min(x, m) - b \cdot x + c$: **Zeitgewinn $a$ je Kettenstufe, Eigenaufwand $b$ je eigener Stufe**, Grundniveau $c$. Die Werte $a = 0{,}2$, $b = 0{,}1$, $c = 0{,}6$ sind die des klassischen Laborspiels, hier nur Modellparameter.
2. **Sieben Gleichgewichte.** Steht jeder auf Stufe $k$, lohnt es nicht, höher zu gehen (kostet $b$, bringt nichts) und nicht, tiefer zu gehen (spart $b$ je Stufe, kostet aber $a$ je Stufe), solange $a > b$. Die Demo zählt die Gleichgewichte durch Aufzählung aller Profile (n bis 4) und findet genau diese sieben.
   Alle sieben sind **strikt**, und das Gleichgewicht mit allen auf Stufe 7 ist für alle das beste.
3. **Fehlertoleranz je Partner.** Eine Stufe zu halten lohnt sich nur, solange jeder der anderen sie mit höchstens $\delta^* = 1 - (b/a)^{1/(n-1)}$ verfehlt. Je größer die Gruppe, desto kleiner $\delta^*$: dieselbe Fehlerrate, die in einem Zweierteam harmlos ist, ist in einer großen Kette zu viel.
4. **Lernen statt Gleichgewichtswahl.** Die Partner kennen die Lage nicht, sie lernen: nach jeder Runde sehen sie nur das Minimum der anderen und bewerten alle sieben Stufen nach, was sie gebracht hätten. Wie, zeigt die Erklärtabelle.
        """
    )

with st.expander("Erklärtabelle: die Lernregeln und die zwei Fassungen"):
    st.markdown(
        """
| Regel | Was sie tut | Zufall in der Wahl | Wer setzt an |
|---|---|---|---|
| **Logit, Summe** (Fassung des früheren Skripts V4) | Jede Stufe sammelt die Auszahlung, die sie in den Vorrunden gebracht hätte (mit Vergessen φ); gewählt wird mit Wahrscheinlichkeit ∝ exp(λ · Summe). Die Schärfe wächst in den ersten Runden mit der Summe der Gewichte bis λ/(1−φ). | ja, immer | `noregret-demo` (Hedge) |
| **Logit, Mittel** | Wie oben, aber gewichtetes **Mittel** statt Summe: ∝ exp(Λ · Mittel). Die Schärfe ist von der ersten Runde an Λ. | ja, immer | `marl-demo` |
| **Beste Antwort mit Zittern** | Spielt die Stufe des Minimums der anderen aus der Vorrunde; mit Wahrscheinlichkeit ε eine gleichverteilte Stufe. | nur durch ε | `gleichgewichtsauswahl-demo` |
| **Fiktives Spiel** | Beste Antwort auf das Mittel **aller** Runden (φ = 1, Schärfe 1 000, praktisch ohne Zufall). | praktisch nein | `nash-demo` |
| **Replikator** (nur Experiment E3) | Anteile der Stufen in einer großen Gruppe wachsen, wenn ihre Auszahlung über dem Durchschnitt liegt; deterministisch, kein einzelner Partner. | nein | `replikator-demo` |

**Namensklärung.** Das frühere Skript V4 nannte die Summenfassung „Mittelwert“; der Code summiert. Hier heißt **„Summe“, was summiert wird**, und **„Mittel“, was gemittelt wird**. Beide laufen in der App, weil sie sich in Anlauf und Geschwindigkeit stark unterscheiden.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🧭 Lernregel und Start")
    rule = st.selectbox("Lernregel", C.RULES, key="rule_select", on_change=on_rule_change,
                        help="Logit-Summe und Logit-Mittel sind zwei Fassungen derselben Erfahrungsbewertung; beste Antwort mit Zittern und fiktives Spiel sind die beiden Grenzfälle ohne bzw. mit wenig Zufall. Beim Wechsel springt die Schärfe auf 5 (Summe) bzw. 50 (Mittel).")
    start = st.selectbox("Start (Runde 1)", C.STARTS, key="start_select",
                         help="Die Stufen in Runde 1: gleichverteilt gelost (Zufall) oder alle Partner auf 7 (kooperativer Start) bzw. 4. Das Lernen beginnt erst nach Runde 1.")
    is_logit = rule in (C.RULE_SUM, C.RULE_MEAN)
    is_best = rule == C.RULE_BEST
    st.header("🎚️ Lernen")
    lam = st.slider("Schärfe λ (Summe) bzw. Λ (Mittel)", *bounds("lam_slider"), key="lam_slider", step=0.1, disabled=not is_logit,
                    help="Wie scharf die Wahl der besten Stufe folgt. Summe: Endschärfe λ/(1−φ); Mittel: von Anfang an Λ. Nur für die beiden Logit-Regeln (fiktives Spiel: fest 1 000, beste Antwort: gibt es nicht).")
    phi = st.slider("Vergessen φ (Gewicht der Vorrunden)", *bounds("phi_slider"), key="phi_slider", step=0.05, disabled=not is_logit,
                    help="0: nur die letzte Runde zählt; 1: alle Runden zählen gleich. Nur für die beiden Logit-Regeln (fiktives Spiel: fest 1).")
    eps = st.slider("Zittern ε (nur beste Antwort)", *bounds("eps_slider"), key="eps_slider", step=0.005, disabled=not is_best,
                    help="Wahrscheinlichkeit, mit der ein Partner statt der besten Antwort eine gleichverteilte Stufe wählt. Nur für die beste Antwort; mit 0 gibt es keinen Zufall.")
    prior = st.slider("Vorabsprache s (fiktive Runden „alle auf 7“)", *bounds("prior_slider"), key="prior_slider", step=1, disabled=is_best,
                      help="Ein Anfangszustand der Lerner, kein Reden: jeder startet mit s fiktiven Runden Erfahrung, in denen alle auf 7 spielten. Nicht für die beste Antwort (sie hat kein Gedächtnis).")
    norm = st.toggle("Schärfe an die Prämie anpassen", key="norm_toggle", disabled=not is_logit,
                     help="An: die Schärfe gilt in Einheiten von a/0,2, so zählt nur das Verhältnis b/a (Prämie und Schärfe getrennt regelbar). Aus: die Schärfe gilt in Auszahlungseinheiten, eine höhere Prämie macht die Wahl zugleich schärfer.")
    st.header("⚖️ Spiel")
    n = st.slider("Zahl der Partner n", *bounds("n_slider"), key="n_slider", step=1, help="Größe der Kette. Die Fehlertoleranz je Partner sinkt mit n.")
    a = st.slider("Prämie a (Zeitgewinn je Kettenstufe)", *bounds("a_slider"), key="a_slider", step=0.1, help="Was eine Stufe des Kettenminimums jedem Partner bringt. Mindestens 0,2, damit a größer als b bleibt.")
    b = st.slider("Einsatzpreis b (Eigenaufwand je Stufe)", *bounds("b_slider"), key="b_slider", step=0.01, help="Was eine Stufe den Partner selbst kostet. Je kleiner b/a, desto größer die Fehlertoleranz.")
    st.header("🎲 Lauf")
    rounds = st.slider("Runden T", *bounds("t_slider"), key="t_slider", step=10, help="Bis 400 Runden: 60 Runden genügen nicht, die Mittelfassung fällt bei n = 8 erst zwischen Runde 150 und 400 unter 4.")
    runs = st.slider("Läufe", *bounds("runs_slider"), key="runs_slider", step=50, help=f"Unabhängige Wiederholungen. Läufe · n · Runden ist auf {V.thousands(C.RUN_BUDGET)} begrenzt; darüber rechnet die App mit weniger Läufen und sagt es.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"rule_select": rule, "start_select": start, "n_slider": int(n), "a_slider": float(a), "b_slider": float(b), "lam_slider": float(lam), "phi_slider": float(phi),
                   "eps_slider": float(eps), "prior_slider": int(prior), "norm_toggle": bool(norm), "t_slider": int(rounds), "runs_slider": int(runs), "seed_input": int(seed)})

settings = S.Settings(rule=rule, start=start, n=int(n), a=float(a), b=float(b), lam=float(lam), phi=float(phi), eps=float(eps), prior=float(prior), normalize=bool(norm),
                      rounds=int(rounds), runs=int(runs), seed=int(seed))
used_runs = S.budget_runs(settings)
with st.spinner(f"Rechne {V.thousands(used_runs)} Läufe mit {settings.n} Partnern über {settings.rounds} Runden (rund {max(1, math.ceil(S.estimated_seconds(settings)))} Sekunden) …"):
    view = _analyse(settings)
    overlay = _overlay(settings)
verdicts = []   # Verdikte für den PDF-Export

# --- Hauptansicht --------------------------------------------------------------------------------------------------------------------------
st.markdown("## 📊 Was wird aus dem Kettenminimum?")
if used_runs < settings.runs:
    st.info(f"Gerechnet mit {V.thousands(used_runs)} statt {V.thousands(settings.runs)} Läufen: Läufe · Partner · Runden ist auf {V.thousands(C.RUN_BUDGET)} begrenzt.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Minimum in Runde 1 (Mittel)", V.de(float(view.mean_min[0]), 2), help=f"Bei gleichverteilter Wahl ist es der reine Zufall: exakt {V.de(view.random_min, 2)} für n = {settings.n}.")
m2.metric(f"Minimum in Runde {settings.rounds} (Mittel)", V.de(float(view.mean_min[-1]), 2), help=f"Mittel über {V.thousands(view.runs_used)} Läufe, Standardfehler {V.de(float(view.se_min[-1]), 3)}.")
m3.metric(f"Läufe mit Minimum {settings.levels} in Runde {settings.rounds}", V.pct(view.share_top, 1), help="Anteil der Läufe, in denen die ganze Kette auf der obersten Stufe steht.")
m4.metric("Auszahlung je Partner", V.de(view.pay_end, 3), help=f"Mittel in der letzten Runde; alle auf Stufe 1: {V.de(view.pay_bottom, 2)}, alle auf Stufe {settings.levels}: {V.de(view.pay_top, 2)}.")

st.markdown("#### ❓ Kernfrage: Ist das Start oder Lernen?")
v_start = V.verdict_start_or_learning(view)
show(v_start)
verdicts.append(v_start)
v_pay = V.verdict_payoff(view)
show(v_pay)
verdicts.append(v_pay)
v_tol = V.verdict_tolerance(view)
show(v_tol)
verdicts.append(v_tol)

col_trace, col_mean = st.columns(2)
with col_trace:
    st.plotly_chart(Z.build_trace(view), width="stretch", key="trace_chart")
    st.caption(f"Ein einzelner Lauf (der erste): die Stufen aller {settings.n} Partner (dünn, leicht versetzt) und das Kettenminimum (dick).")
with col_mean:
    st.plotly_chart(Z.build_mean_curve(view, overlay), width="stretch", key="mean_chart")
    st.caption("Mittleres Kettenminimum über alle Läufe, dazu die mittlere Hälfte der Läufe." + (f" Grün: {overlay['label']}." if overlay is not None else " Eine exakte Rechnung gibt es nur für die beste Antwort bis n = 3 und für das Logit mit zwei Partnern in den ersten drei Runden."))
st.plotly_chart(Z.build_hist(view), width="stretch", key="hist_chart")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["1 Start oder Lernen?", "2 Die Ratsche", "3 Einzugsbereich", "4 Schärfe und Gedächtnis", "5 Einsatzpreis und Prämie", "6 Teams, Absprache, Wachsen"])
meas = _meas()
if meas is None:
    st.warning("Die vorgerechneten Messreihen (data/measurements.json) fehlen oder gehören zu einem anderen Rechencode: `python tools/make_measurements.py` ausführen. Die Experimente E1 und E3 zeigen dann keine Reihen.")

# --- Tab 1 ---------------------------------------------------------------------------------------------------------------------------------
with tab1:
    st.subheader("🔬 Experiment E1: Start oder Lernen?")
    st.caption("Runde 1 ist bei gleichverteiltem Start der reine Zufall; alles, was danach geschieht, ist Lernen. Die Tabelle zeigt das mittlere Kettenminimum je Gruppengröße und Runde. "
               + (f"Vorgerechnet mit {V.thousands(meas['runs'])} Läufen je Zelle (data/measurements.json)." if meas else ""))
    e1a, e1b = st.columns(2)
    e1_start = e1a.radio("Start", (C.START_UNIFORM, C.START_TOP), horizontal=True, key="e1_start")
    e1_rule = e1b.selectbox("Lernregel", list(XP.RULE_CONFIGS), key="e1_rule")
    if meas:
        cells1 = meas["e1_uniform" if e1_start == C.START_UNIFORM else "e1_top"][e1_rule]
        rl = meas["rounds"]
        table([{"n": n_, **{f"Runde {r}": V.de(c["mean"][i], 2) for i, r in enumerate(rl)}, f"Anteil Min {C.E_LEVELS} (Runde {rl[-1]})": V.pct(c["top"][-1], 1)} for n_, c in cells1.items()])
        st.plotly_chart(Z.build_start_or_learning({n_: cells1[n_] for n_ in cells1}, [X.uniform_expected_min(int(n_)) for n_ in cells1], [f"Runde {r}" for r in rl]), width="stretch", key="e1_chart")
        v1 = V.verdict_e1(cells1, e1_rule, e1_start)
        show(v1)
        verdicts.append(v1)
    st.markdown("**Mit den Reglern der Seitenleiste nachrechnen** (Lernregel, Schärfe, Preise, Start und Vorabsprache aus der Seitenleiste; n läuft durch die Gruppengrößen)")
    if st.button("▶️ Gruppengrößen durchrechnen", key="e1_button", help=f"{C.EXP_RUNS} Läufe je Gruppengröße, 60 Runden; rechnet rund {seconds_text(len(C.E1_NS), C.EXP_RUNS, sum(C.E1_NS) // len(C.E1_NS), 60)} Sekunden."):
        st.session_state["e1_signature"] = settings
    if st.session_state.get("e1_signature") == settings:
        with st.spinner("Rechne die Gruppengrößen …"):
            live1 = _e1_live(settings)
        table([{"n": n_, "Runde 1": V.de(c["mean"][0], 2), "Runde 10": V.de(c["mean"][1], 2), "Runde 60": V.de(c["mean"][2], 2), f"Anteil Min {C.E_LEVELS} (Runde 60)": V.pct(c["top"][2], 1)} for n_, c in live1.items()])
        v1l = V.verdict_e1(live1, V.rule_label(settings), settings.start)
        show(v1l)
        verdicts.append(v1l)

# --- Tab 2 ---------------------------------------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("🔬 Experiment E2: Die Ratsche")
    st.caption("Beste Antwort mit Zittern ε: jeder spielt die Stufe des Minimums der anderen, mit Wahrscheinlichkeit ε eine zufällige. **Hinab** reicht ein einziger Fehler, **hinauf** müssen alle Partner zugleich höher gehen. "
               "Die Kette über alle Profile (n ≤ 4) ist exakt gerechnet, die erste Abweichung aus „alle auf 7“ als Formel und als Simulation.")
    e2_eps = st.select_slider("Zittern ε", options=(0.01, 0.05, 0.1), value=0.01, key="e2_eps", format_func=lambda x: V.de(x, 2))
    with st.spinner("Löse die Kette (bis 2 401 Zustände) …"):
        rows2 = _chain_rows(float(e2_eps))
    st.plotly_chart(Z.build_ratchet(rows2), width="stretch", key="e2_chart")
    table([{"n": r["n"], "Zustände": V.thousands(r["states"]), "hinauf (alle 1 → Min ≥ 4)": V.big(r["up"]), "hinab (alle 7 → Min ≤ 3)": V.big(r["down"]), "stationäres E[Minimum]": V.de(r["stationary"], 4)} for r in rows2])
    with st.spinner(f"Simuliere die erste Abweichung ({C.EXP_RUNS} Läufe je Gruppengröße) …"):
        drop2 = _drop_rows(float(e2_eps))
    st.plotly_chart(Z.build_first_drop(drop2), width="stretch", key="e2_drop_chart")
    table([{"n": r["n"], "Formel": V.de(r["formula"], 2), "Simulation": V.de(r["sim"], 2), "Standardfehler": V.de(r["se"], 2), "nicht erreicht (400 Runden)": V.pct(r["censored"], 1)} for r in drop2])
    v2 = V.verdict_e2(rows2, drop2, float(e2_eps))
    show(v2)
    verdicts.append(v2)

# --- Tab 3 ---------------------------------------------------------------------------------------------------------------------------------
with tab3:
    st.subheader("🔬 Experiment E3: Einzugsbereich der obersten Stufe (Replikator)")
    st.caption("Replikator im Mittelwertfeld: eine große Gruppe, aus der n−1 Mitspieler gezogen werden, deterministisch. Welcher Anteil zufälliger Startverteilungen (Dirichlet(1, …, 1)) endet bei Stufe 7? "
               "Die Hirschjagd-Aussage „der Einzugsbereich des guten Gleichgewichts schrumpft“ für n Spieler. Zeitmaß: Einheiten der Prämie (der Pfad hängt nur an b/a).")
    if meas:
        grid3 = meas["basin"]
        st.plotly_chart(Z.build_basin_heatmap(grid3, C.BASIN_NS, C.BASIN_RATIOS), width="stretch", key="e3_heat")
        st.caption(f"Vorgerechnet: {V.thousands(meas['basin_starts'])} Zufallsstarts je Zelle, höchstens {V.thousands(C.E3_STEPS)} Euler-Schritte (η = {V.de(C.E3_ETA, 1)}), Anteil in Prozent.")
        ratio3 = st.select_slider("b/a der Tabelle", options=C.BASIN_RATIOS, value=0.1 if 0.1 in C.BASIN_RATIOS else C.BASIN_RATIOS[0], key="e3_ratio_table", format_func=lambda x: V.de(x, 2))
        table([{"n": n_, **{f"Stufe {k}": V.pct(grid3[f"{n_}|{ratio3}"]["share"][k - 1], 1) for k in range(1, C.E_LEVELS + 1)}, "E[Minimum] des Endzustands": V.de(grid3[f"{n_}|{ratio3}"]["emin"], 2)} for n_ in C.BASIN_NS])
        v3 = V.verdict_e3({n_: grid3[f"{n_}|{ratio3}"] for n_ in C.BASIN_NS}, float(ratio3))
        show(v3)
        verdicts.append(v3)
    st.markdown("**Selbst rechnen** (ein Wert von n und b/a, höchstens 500 Zufallsstarts)")
    e3a, e3b, e3c = st.columns(3)
    e3_n = e3a.slider("Partner n", C.N_MIN, C.N_MAX, 4, key="e3_n")
    e3_ratio = e3b.slider("b/a", 0.02, 0.95, 0.1, step=0.01, key="e3_ratio")
    e3_starts = e3c.slider("Zufallsstarts", 50, C.E3_LIVE_MAX_STARTS, 200, step=50, key="e3_starts")
    if st.button("▶️ Replikator rechnen", key="e3_button", help=f"Rechnet {e3_starts} Zufallsstarts, die Schritte laufen je Start bis zur Konvergenz; meist unter einer Sekunde."):
        st.session_state["e3_signature"] = (int(e3_n), float(e3_ratio), int(e3_starts), int(seed))
    if st.session_state.get("e3_signature") == (int(e3_n), float(e3_ratio), int(e3_starts), int(seed)):
        with st.spinner("Rechne den Replikator …"):
            live3 = _basin_live(int(e3_n), float(e3_ratio), int(e3_starts), int(seed))
        l3a, l3b = st.columns(2)
        with l3a:
            st.plotly_chart(Z.build_basin_levels(live3), width="stretch", key="e3_live_bar")
            st.caption(f"Endstufen von {live3['starts']} Zufallsstarts; nicht konvergiert: {V.pct(live3['none'], 1)}; E[Minimum] des Endzustands {V.de(live3['emin'], 2)}.")
        with l3b:
            st.plotly_chart(Z.build_replicator_path(live3["first"], int(e3_n), C.A_REF, C.A_REF * float(e3_ratio), C.DEFAULT_C), width="stretch", key="e3_live_path")
            st.caption("Der erste Zufallsstart: Anteile der sieben Stufen über die Schritte.")
        v3l = V.verdict_e3({int(e3_n): live3}, float(e3_ratio))
        show(v3l)
        verdicts.append(v3l)

# --- Tab 4 ---------------------------------------------------------------------------------------------------------------------------------
with tab4:
    st.subheader("🔬 Experiment E4: Schärfe und Gedächtnis")
    st.caption(f"Reihen mit {C.EXP_RUNS} Läufen je Zelle, Runde 1 / 10 / 60. Summe: Schärfe λ (Endschärfe λ/(1−φ) mit φ = 0,9). Mittel: Λ = 10 λ, also gleiche Endschärfe. Wo das Minimum am Startwert einfriert, steht es im Verdikt.")
    e4a, e4b, e4c = st.columns(3)
    e4_conv = e4a.radio("Fassung", ("sum", "mean"), format_func=lambda c_: "Summe" if c_ == "sum" else "Mittel", horizontal=True, key="e4_conv")
    e4_start = e4b.radio("Start", (C.START_UNIFORM, C.START_TOP), horizontal=True, key="e4_start")
    e4_n = e4c.selectbox("Partner n", (2, 4, 8), index=1, key="e4_n")
    sharp = _sharp(e4_conv, e4_start, int(e4_n))
    st.plotly_chart(Z.build_lines({f"Runde {r}": (list(sharp), [sharp[k]["mean"][i] for k in sharp]) for i, r in enumerate(sharp[next(iter(sharp))]["rounds"])}, "Schärfe λ", logx=True,
                                  xticks=list(sharp)), width="stretch", key="e4_sharp_chart")
    table(cell_rows(sharp, "λ", lambda k: V.de(k, 1)))
    v4 = V.verdict_e4_sharpness(sharp, int(e4_n), e4_start, e4_conv)
    show(v4)
    verdicts.append(v4)
    st.markdown("**Vergessen φ** (Mittelfassung, Schärfe 50)")
    forget = _forget(e4_start, int(e4_n))
    st.plotly_chart(Z.build_lines({f"Runde {r}": (list(forget), [forget[k]["mean"][i] for k in forget]) for i, r in enumerate(forget[next(iter(forget))]["rounds"])}, "Vergessen φ"), width="stretch", key="e4_phi_chart")
    table(cell_rows(forget, "φ", lambda k: V.de(k, 2)))
    v4f = V.verdict_e4_forgetting(forget, int(e4_n), e4_start)
    show(v4f)
    verdicts.append(v4f)

# --- Tab 5 ---------------------------------------------------------------------------------------------------------------------------------
with tab5:
    st.subheader("🔬 Experiment E5: Einsatzpreis und Prämie")
    st.caption(f"Können Anreize die Koordination retten? Summe, λ = 5, gleichverteilter Start, {C.EXP_RUNS} Läufe je Zelle, Runden 1 / 10 / 60 / 400. Prämie und Schärfe sind getrennt: "
               "**normiert** gilt die Schärfe in Einheiten der Prämie (reiner Effekt von b/a), **nicht normiert** macht eine höhere Prämie die Wahl zugleich schärfer.")
    e5_n = st.selectbox("Partner n", (2, 4, 8, 16), index=2, key="e5_n")
    est5 = seconds_text(len(C.E5_BS) + 2 * len(C.E5_AS), C.EXP_RUNS, int(e5_n), 400)
    if st.button("▶️ Reihen rechnen", key="e5_button", help=f"{len(C.E5_BS) + 2 * len(C.E5_AS)} Zellen bis Runde 400; rechnet rund {est5} Sekunden."):
        st.session_state["e5_signature"] = int(e5_n)
    if st.session_state.get("e5_signature") == int(e5_n):
        with st.spinner(f"Rechne die Reihen (rund {est5} Sekunden) …"):
            price = _price(int(e5_n))
            prem_norm = _premium(int(e5_n), True)
            prem_raw = _premium(int(e5_n), False)
        left5, right5 = st.columns(2)
        with left5:
            st.plotly_chart(Z.build_lines({f"Runde {r}": (list(price), [price[k]["mean"][i] for k in price]) for i, r in enumerate(price[next(iter(price))]["rounds"])}, "Einsatzpreis b"),
                            width="stretch", key="e5_price_chart")
        with right5:
            ks = list(prem_norm)
            st.plotly_chart(Z.build_lines({"normiert, Runde 60": (ks, [prem_norm[k]["mean"][2] for k in ks]), "normiert, Runde 400": (ks, [prem_norm[k]["mean"][3] for k in ks]),
                                           "nicht normiert, Runde 60": (ks, [prem_raw[k]["mean"][2] for k in ks]), "nicht normiert, Runde 400": (ks, [prem_raw[k]["mean"][3] for k in ks])}, "Prämie a"),
                            width="stretch", key="e5_prem_chart")
        table(cell_rows(price, "b", lambda k: V.de(k, 2)))
        v5p = V.verdict_e5_price(price, int(e5_n))
        show(v5p)
        verdicts.append(v5p)
        table(cell_rows(prem_norm, "a (normiert)", lambda k: V.de(k, 1)) + cell_rows(prem_raw, "a (nicht normiert)", lambda k: V.de(k, 1)))
        v5a = V.verdict_e5_premium(prem_norm, prem_raw, int(e5_n))
        show(v5a)
        verdicts.append(v5a)

# --- Tab 6 ---------------------------------------------------------------------------------------------------------------------------------
with tab6:
    st.subheader("🔬 Experiment E6: Teams, Vorabsprache, Prämie plus Absprache, wachsende Gruppen")
    st.caption(f"Vier Gegenproben mit je {C.EXP_RUNS} Läufen je Zelle (Summe, λ = 5, φ = 0,9, wenn nichts anderes steht). Die Vorabsprache ist ein Anfangszustand der Lerner (fiktive Runden „alle auf 7“), kein strategisches Reden.")
    st.markdown("**Teams im Pool** (16 Partner, Teams fest oder jede Runde neu gelost; Kennzahl: Mittel der Team-Minima)")
    if st.button("▶️ Teams rechnen", key="e6_team_button", help=f"{2 * len(C.POOL_TEAMS)} Zellen, 60 Runden; rechnet rund {seconds_text(2 * len(C.POOL_TEAMS), C.EXP_RUNS, C.POOL_SIZE, 60)} Sekunden."):
        st.session_state["e6_team_done"] = True
    if st.session_state.get("e6_team_done"):
        with st.spinner("Rechne die Teams …"):
            teams = _teams()
        st.plotly_chart(Z.build_bars([f"k = {k}" for k in C.POOL_TEAMS], {"fest": [teams[f"{k}|1"]["mean"][-1] for k in C.POOL_TEAMS], "jede Runde neu gelost": [teams[f"{k}|0"]["mean"][-1] for k in C.POOL_TEAMS]},
                                    ytitle="Team-Minimum in Runde 60"), width="stretch", key="e6_team_chart")
        table([{"Teamgröße k": k, "fest, Runde 60": V.de(teams[f"{k}|1"]["mean"][-1], 2), "gelost, Runde 60": V.de(teams[f"{k}|0"]["mean"][-1], 2),
                f"Teams mit Minimum {C.E_LEVELS} (fest)": V.pct(teams[f"{k}|1"]["top"][-1], 1)} for k in C.POOL_TEAMS])
        v6t = V.verdict_e6_teams(teams)
        show(v6t)
        verdicts.append(v6t)

    st.markdown("**Vorabsprache s** (erste Wahl gleichverteilt)")
    e6_n = st.selectbox("Partner n", (2, 4, 8, 16), index=2, key="e6_n")
    est6 = seconds_text(len(C.E6_PRIORS), C.EXP_RUNS, int(e6_n), 400)
    if st.button("▶️ Absprache rechnen", key="e6_prior_button", help=f"{len(C.E6_PRIORS)} Zellen bis Runde 400; rechnet rund {est6} Sekunden."):
        st.session_state["e6_prior_signature"] = int(e6_n)
    if st.session_state.get("e6_prior_signature") == int(e6_n):
        with st.spinner(f"Rechne die Absprache (rund {est6} Sekunden) …"):
            prior6 = _prior(int(e6_n))
        ks6 = list(prior6)
        st.plotly_chart(Z.build_lines({f"Runde {r}": ([int(k) for k in ks6], [prior6[k]["mean"][i] for k in ks6]) for i, r in enumerate(prior6[ks6[0]]["rounds"])}, "Vorabsprache s (fiktive Runden)"),
                        width="stretch", key="e6_prior_chart")
        table(cell_rows(prior6, "s", lambda k: str(k)))
        v6p = V.verdict_e6_prior(prior6, int(e6_n))
        show(v6p)
        verdicts.append(v6p)

    st.markdown("**Prämie plus Vorabsprache** (b = 0,1; Schärfe normiert; Gruppen mit 8 und 16 Partnern)")
    c6n = st.selectbox("Partner n", (8, 16), index=1, key="e6_combo_n")
    est6c = seconds_text(5, C.EXP_RUNS, int(c6n), 400)
    if st.button("▶️ Kombination rechnen", key="e6_combo_button", help=f"5 Zellen bis Runde 400; rechnet rund {est6c} Sekunden."):
        st.session_state["e6_combo_signature"] = int(c6n)
    if st.session_state.get("e6_combo_signature") == int(c6n):
        with st.spinner(f"Rechne die Kombination (rund {est6c} Sekunden) …"):
            combo6 = _combo(int(c6n))
        table(cell_rows(combo6, "a | s", lambda k: k.replace("|", " | ")))
        v6c = V.verdict_e6_combo(combo6, int(c6n))
        show(v6c)
        verdicts.append(v6c)

    st.markdown(f"**Wachsende Gruppen** (kooperativer Start; die Gruppe beginnt mit {C.GROW_START} Partnern und wächst alle 10 bzw. 25 Runden um einen, wer dazukommt übernimmt die Erfahrung eines zufälligen Mitglieds)")
    g1, g2 = st.columns(2)
    g_n = g1.selectbox("Partner n", (8, 16), index=0, key="e6_grow_n")
    g_conv = g2.radio("Fassung", ("sum", "mean"), format_func=lambda c_: "Summe (λ = 5)" if c_ == "sum" else "Mittel (Λ = 50)", horizontal=True, key="e6_grow_conv")
    est6g = seconds_text(3, C.EXP_RUNS, int(g_n), 400)
    if st.button("▶️ Wachsen rechnen", key="e6_grow_button", help=f"3 Zellen bis Runde 400; rechnet rund {est6g} Sekunden."):
        st.session_state["e6_grow_signature"] = (int(g_n), g_conv)
    if st.session_state.get("e6_grow_signature") == (int(g_n), g_conv):
        with st.spinner(f"Rechne das Wachsen (rund {est6g} Sekunden) …"):
            grow6 = _grow(int(g_n), g_conv)
        gr = grow6["fest"]["rounds"]
        st.plotly_chart(Z.build_lines({k: (gr, grow6[k]["mean"]) for k in grow6}, "Runde", logx=True, xticks=gr), width="stretch", key="e6_grow_chart")
        table(cell_rows(grow6, "Gruppe", lambda k: k))
        v6g = V.verdict_e6_grow(grow6, int(g_n), g_conv)
        show(v6g)
        verdicts.append(v6g)

# --- Verwandte Demos -----------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔗 Verwandte Demos")
st.markdown(
    """
- **`koordinationsspiel-demo`** (Stück 1 der Reihe): die Gleichgewichte und die Schwelle p* der Hirschjagd mit zwei Spielern; hier ist die Fehlertoleranz δ* das Gegenstück für n Partner.
- **`gleichgewichtsauswahl-demo`** (Stück 2): welche Konvention auf Dauer gewählt wird, mit Rauschen im 2×2-Fall.
- **`konventionswechsel-demo`** (Stück 3): wie lange ein Wechsel dauert (Rauschen und Wechselzeit im 2×2-Fall).
- **`globale-spiele-demo`** (Stück 4): Unsicherheit über den Zustand; hier kennen die Partner das Spiel und lernen nur.
- **`noregret-demo`**: Hedge-Lernen im Überlastungsspiel, in dem Lkw einander meiden; hier bestimmt das Minimum das Ergebnis.
- **`replikator-demo`**: der Replikator für Strategien wiederholter Spiele; hier E3 für das Minimum-Spiel.
- **`nash-demo`**: Gleichgewichte kleiner Matrixspiele; hier sieben strikte symmetrische Gleichgewichte mit n Spielern.
"""
)

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Lernregeln sind Modellannahmen** (vier Regeln, sieben Stufen, a = 0,2, b = 0,1, c = 0,6) | Gemessen sind diese Regeln, nicht das Verhalten von Menschen oder echten Umschlagstellen. Andere Regeln, Stufenzahlen oder Auszahlungen können andere Zahlen geben; gemessen ist, dass das Minimum bei kooperativem Start unter den drei Regeln mit Rauschen in 14 von 18 Zellen nach 400 Runden unter 4 liegt, nicht mehr. | `noregret-demo` |
| **Das Ergebnis hängt an Start und Fassung** | Schon die Fassung (Summe gegen Mittel) ändert die Geschwindigkeit um Größenordnungen; ein Befund gilt für die gemessene Zelle, nicht für „das Lernen“. | – |
| **Gleiche Partner** | Alle Partner lernen mit denselben Parametern; ungleiche Schärfe, Führungspersonen oder Gruppen mit Absprache als Strategie sind nicht gerechnet. | `koordinationsspiel-demo` |
| **Die Vorabsprache ist ein Anfangszustand der Lerner, kein Reden** | Sie bildet nicht ab, dass Menschen einander Zusagen machen oder brechen; Cheap Talk als Strategie ist nicht gerechnet. | – |
| **T und Läufe sind endlich** | Mittelwerte über Läufe mit Standardfehler; die Obergrenze von 400 Runden ist ein Budget, kein Grenzwert der Dynamik. Ein „bleibt“ heißt: kein Unterschied innerhalb von 4 Standardfehlern in diesen Läufen. | – |
| **Der Replikator ist ein Mittelwertfeld** | Er beschreibt eine große Gruppe, aus der Mitspieler unabhängig gezogen werden, nicht eine feste Kette; Zufallsstarts sind gleichverteilt über die Verteilungen (Dirichlet). | `replikator-demo` |
| **Wachsende Gruppen: eine einzige Übernahmeregel** | Wer dazukommt, übernimmt die Erfahrung eines zufälligen Mitglieds. Andere Regeln (Neuling lernt von Grund auf, Lehrling) sind nicht gerechnet; ob das Wachsen in Experimenten mit Menschen hilft, ist hier nicht Gegenstand. | – |
| **Nur eine Kette** | Netze aus mehreren sich überlappenden Ketten, Hubs und Zeitfenster sind nicht modelliert. | – |
    """
)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind $n$ Partner, Stufen $1, \dots, E$ ($E = 7$), Prämie $a$, Einsatzpreis $b$ und Grundniveau $c$; gesucht sind die Gleichgewichte und das Verhalten des Kettenminimums, wenn die Partner nach einer Lernregel wählen.
Die Herleitungen sind eigene Rechnung, durch Aufzählung, zweite Rechenwege und Simulation gestützt, nicht gegen Literatur geprüft.

| Symbol | Bedeutung |
|---|---|
| $x_i,\ m_i$ | Stufe von Partner $i$ und Minimum der Stufen aller anderen |
| $a,\ b,\ c$ | Prämie je Kettenstufe, Einsatzpreis je eigener Stufe, Grundniveau |
| $\delta^*$ | Fehlertoleranz je Partner |
| $N_x,\ W$ | Zähler (Auszahlungssumme der Stufe $x$) und Nenner (Summe der Gewichte) der Erfahrungsbewertung |
| $\varphi,\ \lambda,\ \Lambda$ | Vergessen, Schärfe der Summenfassung, Schärfe der Mittelfassung |
| $\varepsilon,\ q$ | Zittern der besten Antwort und $q = 1 - \varepsilon + \varepsilon/E$ |
| $S(k),\ u_x$ | $P(\text{Stufe} \ge k)$ und Auszahlung der Stufe $x$ im Replikator |

**Auszahlung** eines Partners:

$$\pi(x, m) = a \cdot \min(x, m) - b \cdot x + c$$

**Gleichgewichte.** Steht jeder auf $k$, bringt eine Abweichung nach oben $-b$ je Stufe und nach unten $-(a-b)$ je Stufe; für $a > b > 0$ ist jedes der $E$ Profile $(k, \dots, k)$ ein striktes Gleichgewicht. Der Grenznutzen von „Stufe $L$ halten“ gegenüber „eine Stufe tiefer“ ist, wenn jeder andere das Niveau mit Wahrscheinlichkeit $1-\delta$ erreicht,

$$a \cdot (1-\delta)^{n-1} - b$$

und verschwindet bei der **Fehlertoleranz**

$$\delta^* = 1 - \left(\frac{b}{a}\right)^{1/(n-1)}$$

**Erfahrungsbewertung** (gegenfaktisch, für alle Stufen $x$) nach beobachtetem Minimum $m_t$:

$$N_x \leftarrow \varphi\, N_x + \pi(x, m_t), \qquad W \leftarrow \varphi\, W + 1$$

**Wahlwahrscheinlichkeit.** Summenfassung (die des früheren Skripts V4) und Mittelfassung:

$$P(x) \propto \exp(\lambda\, N_x) \qquad\text{bzw.}\qquad P(x) \propto \exp\!\left(\Lambda\, \frac{N_x}{W}\right)$$

**Normierung** der Schärfe auf die Prämie (Bezugswert $0{,}2$), damit nur $b/a$ zählt:

$$\lambda_{\text{wirksam}} = \lambda \cdot \frac{0{,}2}{a}$$

**Erwartetes Minimum** unabhängiger Wahlen; bei Gleichverteilung ist Runde 1 reiner Zufall:

$$E[\min] = \sum_{k=1}^{E} \prod_{i=1}^{n} P(x_i \ge k), \qquad E[\min] = \sum_{k=1}^{E} \left(\frac{E-k+1}{E}\right)^{n}$$

**Ratsche.** Beste Antwort mit Zittern $\varepsilon$: nur das Profil „alle oben“ hat das oberste Minimum, die Wahrscheinlichkeit, dass alle wieder oben wählen, ist $q^n$. Die erste Runde mit Minimum unter $E$ liegt im Mittel bei

$$1 + \frac{1}{1 - q^{n}}$$

Erwartete Treffzeiten der Markov-Kette über alle $E^n$ Profile mit Übergangsmatrix $P$ lösen $(I - Q)\,t = \mathbf 1$ auf den Zuständen außerhalb der Zielmenge; sie wachsen für das Hinaufklettern wie $\varepsilon^{-n}$:

$$t = (I - Q)^{-1}\, \mathbf 1$$

**Replikator** im Mittelwertfeld ($n-1$ unabhängige Mitspieler aus der Verteilung $p$):

$$u_x = a \sum_{k \le x} S(k)^{\,n-1} - b\,x + c, \qquad \dot p_x = p_x\,(u_x - \bar u)$$

**Literatur.** Die Quellen stehen in der README mit Stand der Prüfung; Formeln und Zahlen dieser Demo sind eigene Rechnung.
        """
    )

st.markdown("---")
st.subheader("📄 Ergebnisse als PDF")
if st.button("PDF erstellen", key="pdf_button"):
    from swg_pdf_export import build_pdf
    st.session_state["pdf_bytes"] = build_pdf(view, verdicts)
if st.session_state.get("pdf_bytes"):
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="schwaechstes-glied.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
