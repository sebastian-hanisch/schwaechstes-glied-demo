"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Knopf, fährt jeden Regler an seine Grenzen und prüft, dass
kein Python-Fehler auftritt - insbesondere `StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key=) und `StreamlitAPIException` bei Reglern mit
berechneten Grenzen. Die ▶️-Knöpfe laufen einzeln (die Schleife lässt sie aus)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import swg_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer
AUTOPLAY_MARKER = "▶"  # ▶: Knöpfe mit Rechenlast, einzeln getestet
PRESET_LABELS = set(C.PRESETS)


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([w.value for w in at.warning] + [s.value for s in at.success] + [i.value for i in at.info] + [e.value for e in at.error])


def test_app_loads_without_exception():
    at = _fresh_app()
    assert "überwiegend der Start" in _texts(at)                    # Standard: Summe, n = 8, gleichverteilter Start
    assert not [w for w in at.warning if "vorgerechneten Messreihen" in w.value]       # die Messdatei passt zum Code


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label or label in PRESET_LABELS:
            continue                                                 # Presets laufen unten einzeln
        at = _fresh_app()
        at.button[index].click().run()
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_enabled_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            if slider.disabled:
                continue
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run()
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("rule", C.RULES)
@pytest.mark.parametrize("edge", ["min", "max"])
def test_corner_matrix_all_enabled_sidebar_sliders_at_their_minima_or_maxima(rule, edge):
    """Eckfall-Matrix (DEMO-PLAYBOOK §1d): je Lernregel alle zugehörigen Regler zugleich am Minimum bzw. Maximum (n = 2 bzw. 32, Runden 10 bzw. 400, Läufe 50 bzw. 1 000,
    Prämie 0,2 bzw. 2, Einsatzpreis 0,01 bzw. 0,19, Schärfe, Vergessen, Zittern, Vorabsprache). Ausgegraute Regler lassen sich nicht setzen und bleiben auf ihrem Wert."""
    at = _fresh_app()
    at.selectbox(key="rule_select").set_value(rule).run()
    for slider in at.sidebar.slider:
        if slider.disabled:
            continue
        slider.set_value(slider.min if edge == "min" else slider.max)
    at.run()
    assert not at.exception, (rule, edge, [str(e) for e in at.exception])


def test_corner_start_options_and_toggle_do_not_raise():
    for start in C.STARTS:
        at = _fresh_app()
        at.selectbox(key="start_select").set_value(start).run()
        assert not at.exception, start
    at = _fresh_app()
    at.toggle(key="norm_toggle").set_value(False).run()
    assert not at.exception and at.sidebar.toggle(key="norm_toggle").value is False


def test_largest_chain_with_most_rounds_and_most_runs_is_cut_to_the_budget_and_says_so():
    at = _fresh_app()
    at.slider(key="n_slider").set_value(C.N_MAX)
    at.slider(key="t_slider").set_value(C.T_MAX)
    at.slider(key="runs_slider").set_value(C.RUNS_MAX)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any("Gerechnet mit 781 statt 1.000 Läufen" in i.value for i in at.info)


def test_smallest_chain_with_one_hundred_runs_is_not_cut():
    at = _fresh_app()
    at.slider(key="n_slider").set_value(2).run()
    assert not at.exception and not [i for i in at.info if i.value.startswith("Gerechnet mit")]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_loads_and_shows_a_verdict(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any(m.value for m in at.markdown) and (_texts(at))
    assert at.session_state["rule_select"] == C.PRESETS[name]["rule"] and at.session_state["n_slider"] == C.PRESETS[name]["n"]


def test_switching_the_rule_sets_the_matching_sharpness():
    at = _fresh_app()
    assert at.session_state["lam_slider"] == 5.0
    at.selectbox(key="rule_select").set_value(C.RULE_MEAN).run()
    assert at.session_state["lam_slider"] == 50.0
    at.selectbox(key="rule_select").set_value(C.RULE_SUM).run()
    assert at.session_state["lam_slider"] == 5.0


def test_controls_that_do_not_apply_are_disabled():
    at = _fresh_app()
    assert at.sidebar.slider(key="eps_slider").disabled is True and at.sidebar.slider(key="lam_slider").disabled is False
    at.selectbox(key="rule_select").set_value(C.RULE_BEST).run()
    assert at.sidebar.slider(key="eps_slider").disabled is False
    assert all(at.sidebar.slider(key=k).disabled for k in ("lam_slider", "phi_slider", "prior_slider")) and at.sidebar.toggle(key="norm_toggle").disabled
    at.selectbox(key="rule_select").set_value(C.RULE_FICT).run()
    assert at.sidebar.slider(key="eps_slider").disabled and at.sidebar.slider(key="lam_slider").disabled and not at.sidebar.slider(key="prior_slider").disabled


def test_exact_reference_appears_for_best_reply_up_to_three_partners_and_for_two_logit_players():
    at = _fresh_app()
    assert any("Eine exakte Rechnung gibt es nur" in c.value for c in at.caption)
    at.selectbox(key="rule_select").set_value(C.RULE_BEST).run()
    at.slider(key="n_slider").set_value(3).run()
    assert any("Grün: exakt (Markov-Kette über alle Profile)" in c.value for c in at.caption)
    at.selectbox(key="rule_select").set_value(C.RULE_SUM).run()
    at.slider(key="n_slider").set_value(2).run()
    assert any("Grün: exakt (Pfadaufzählung, Runde 1 bis 3)" in c.value for c in at.caption)


def _live_e1_tables(at):
    """Die Live-Tabelle von E1 (Spalten n, Runde 1, 10, 60, Anteil) im Unterschied zur vorgerechneten (mit Runde 150 und 400)."""
    return [df for df in at.dataframe if f"Anteil Min {C.E_LEVELS} (Runde 60)" in df.value.columns.tolist()]


def test_the_play_buttons_work_individually():
    at = _fresh_app()
    at.button(key="e1_button").click().run()
    assert not at.exception and len(_live_e1_tables(at)) == 1

    at = _fresh_app()
    at.button(key="e3_button").click().run()
    assert not at.exception and at.session_state["e3_signature"][0] == 4 and "Einzugsbereich der obersten Stufe bei b/a = 0,10" in _texts(at)

    at = _fresh_app()
    at.button(key="e5_button").click().run()
    assert not at.exception and at.session_state["e5_signature"] == 8 and "Einsatzpreis, n = 8" in _texts(at) and "Prämie, n = 8" in _texts(at)

    for key, marker in (("e6_team_button", "Teams im Pool von 16"), ("e6_prior_button", "Vorabsprache, n = 8"), ("e6_combo_button", "Prämie und Vorabsprache, n = 16"),
                        ("e6_grow_button", "Wachsende Gruppen (Summe), n = 8")):
        at = _fresh_app()
        at.button(key=key).click().run()
        assert not at.exception, key
        assert marker in _texts(at), (key, marker)


def test_play_results_are_hidden_when_the_sidebar_settings_change():
    at = _fresh_app()
    at.button(key="e1_button").click().run()
    assert len(_live_e1_tables(at)) == 1
    at.slider(key="n_slider").set_value(5).run()
    assert not at.exception and _live_e1_tables(at) == []


def test_tab_two_and_three_and_four_show_their_results_without_a_button():
    at = _fresh_app()
    texts = _texts(at)
    assert "Ratsche bei Zittern ε = 0,010" in texts and "Einzugsbereich der obersten Stufe bei b/a = 0,10" in texts and "Schärfe-Reihe (Summe), n = 4" in texts and "Vergessen (Mittelfassung), n = 4" in texts
    at.select_slider(key="e2_eps").set_value(0.05).run()
    assert not at.exception and "Ratsche bei Zittern ε = 0,050" in _texts(at)


def test_pdf_button_creates_the_download():
    at = _fresh_app()
    at.button(key="pdf_button").click().run()
    assert not at.exception and at.session_state["pdf_bytes"][:4] == b"%PDF"


def test_permalink_sets_the_regulators_and_clamps_to_the_bounds():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["regel"] = C.RULE_MEAN
    at.query_params["n"] = "5"
    at.query_params["lam"] = "50"
    at.query_params["a"] = "9"                       # über der Grenze: wird auf 2,0 begrenzt
    at.query_params["norm"] = "nein"
    at.query_params["t"] = "400"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["rule_select"] == C.RULE_MEAN and at.session_state["n_slider"] == 5 and at.session_state["lam_slider"] == 50.0
    assert at.session_state["a_slider"] == 2.0 and at.session_state["norm_toggle"] is False and at.session_state["t_slider"] == 400


def test_permalink_ignores_values_outside_the_allowed_choices_and_garbage():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["regel"] = "Quatsch"
    at.query_params["start"] = "alle 9"
    at.query_params["b"] = "nan"
    at.query_params["n"] = "abc"
    at.query_params["norm"] = "vielleicht"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["rule_select"] == C.RULE_SUM and at.session_state["start_select"] == C.START_UNIFORM and at.session_state["b_slider"] == C.DEFAULT_B
    assert at.session_state["n_slider"] == C.DEFAULT_N and at.session_state["norm_toggle"] is True


def test_permalink_round_trip_after_a_preset_click():
    at = _fresh_app()
    at.sidebar.button(key="preset_Nur zwei").click().run()
    assert at.query_params["n"] == ["2"] or at.query_params["n"] == "2"
