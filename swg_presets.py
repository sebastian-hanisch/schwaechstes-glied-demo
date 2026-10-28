"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from collections.abc import Callable
from dataclasses import dataclass

import streamlit as st

import swg_constants as C


def parse_bool(text) -> bool:
    """Wahrheitswert aus der Adresszeile: 1/0, true/false, ja/nein (Groß- und Kleinschreibung egal); sonst ValueError."""
    t = str(text).strip().lower()
    if t in ("1", "true", "ja", "an"):
        return True
    if t in ("0", "false", "nein", "aus"):
        return False
    raise ValueError(text)


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None      # bei Auswahlreglern: die erlaubten Werte


B = C.BASE
SETTING_SPECS = {
    "rule_select": SettingSpec("regel", str, B["rule"], choices=C.RULES),
    "start_select": SettingSpec("start", str, B["start"], choices=C.STARTS),
    "n_slider": SettingSpec("n", int, B["n"], C.N_MIN, C.N_MAX),
    "a_slider": SettingSpec("a", float, B["a"], C.A_MIN, C.A_MAX),
    "b_slider": SettingSpec("b", float, B["b"], C.B_MIN, C.B_MAX),
    "lam_slider": SettingSpec("lam", float, B["lam"], C.LAM_MIN, C.LAM_MAX),
    "phi_slider": SettingSpec("phi", float, B["phi"], C.PHI_MIN, C.PHI_MAX),
    "eps_slider": SettingSpec("eps", float, B["eps"], C.EPS_MIN, C.EPS_MAX),
    "prior_slider": SettingSpec("s", int, B["prior"], C.PRIOR_MIN, C.PRIOR_MAX),
    "norm_toggle": SettingSpec("norm", parse_bool, B["normalize"], choices=(True, False)),
    "t_slider": SettingSpec("t", int, B["rounds"], C.T_MIN, C.T_MAX),
    "runs_slider": SettingSpec("lauf", int, B["runs"], C.RUNS_MIN, C.RUNS_MAX),
    "seed_input": SettingSpec("seed", int, B["seed"], 0, C.SEED_MAX),
}
PRESET_KEYS = {
    "rule": "rule_select", "start": "start_select", "n": "n_slider", "a": "a_slider", "b": "b_slider", "lam": "lam_slider", "phi": "phi_slider", "eps": "eps_slider",
    "prior": "prior_slider", "normalize": "norm_toggle", "rounds": "t_slider", "runs": "runs_slider", "seed": "seed_input",
}
RULE_LAM_DEFAULT = {C.RULE_SUM: 5.0, C.RULE_MEAN: 50.0}     # Schärfe, die zur Fassung passt (Summe λ, Mittel Λ)


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf erlaubte Werte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.choices is not None:
                if value not in spec.choices:
                    continue
            else:
                if value != value:      # NaN
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def on_rule_change():
    """Wechselt die Lernregel, springt die Schärfe auf den Wert, der zur Fassung passt (Summe 5, Mittel 50); danach bleibt der Regler frei."""
    default = RULE_LAM_DEFAULT.get(st.session_state.get("rule_select"))
    if default is not None:
        st.session_state["lam_slider"] = default


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
