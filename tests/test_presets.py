"""Presets: Vollständigkeit, gültige Werte, erstes Preset = Standardeinstellungen, Permalink-Grenzen, und je Preset die erwartete Verdikt-Zeile (DEMO-PLAYBOOK §11a, Punkt 7)."""

import pytest

import swg_constants as C
import swg_evaluation as V
import swg_presets as P
import swg_scenario as S


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name].strip(), name


def test_preset_values_are_inside_the_slider_bounds():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.choices is not None:
                assert preset[key] in spec.choices, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert S.Settings(**C.PRESETS[first]) == S.Settings()
    assert first.startswith("Standard")


def test_setting_specs_cover_every_preset_key_and_permalink_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_no_two_presets_are_identical():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_defaults_in_the_specs_are_the_base_settings():
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == C.BASE[key], key


def test_parse_bool_accepts_the_written_forms_and_rejects_the_rest():
    assert [P.parse_bool(x) for x in ("1", "True", "ja", "AN", " true ")] == [True] * 5
    assert [P.parse_bool(x) for x in ("0", "False", "nein", "aus")] == [False] * 4
    with pytest.raises(ValueError):
        P.parse_bool("vielleicht")


def test_lambda_default_follows_the_two_conventions():
    assert P.RULE_LAM_DEFAULT == {C.RULE_SUM: 5.0, C.RULE_MEAN: 50.0}


def test_the_rule_presets_use_the_matching_sharpness():
    for name, preset in C.PRESETS.items():
        if preset["rule"] == C.RULE_MEAN:
            assert preset["lam"] == 50.0, name
        if preset["rule"] == C.RULE_SUM:
            assert preset["lam"] == 5.0, name


# --- Verdikt je Preset (kein Preset ohne Effekt darf einen Effekt melden) ---------------------------------------------------------------------------
def _verdict(name):
    s = S.with_(S.settings_of_preset(name), runs=300)
    return V.verdict_start_or_learning(V.analyse(s))


EXPECTED = {
    "Standard (V4-Hook, n = 8)": ("warning", "überwiegend der Start"),
    "Kooperativer Start, Summe": ("warning", "Der ganze Abfall ist Lernen"),
    "Kooperativer Start, Mittel": ("warning", "Der ganze Abfall ist Lernen"),
    "Nur zwei": ("success", "Lernen hebt das Minimum"),
    "Ohne Rauschen": ("info", "Das Minimum bewegt sich nicht"),
    "Billiger Einsatz": ("success", "Lernen hebt das Minimum"),
    "Prämie": ("success", "Lernen hebt das Minimum"),
    "Vorabsprache verblasst": ("success", "Zwischendurch erreicht das Mittel"),
}


@pytest.mark.parametrize("name", [n for n in C.PRESETS if n in EXPECTED])
def test_each_preset_shows_the_expected_verdict(name):
    kind, text = _verdict(name)
    want_kind, want_text = EXPECTED[name]
    assert kind == want_kind and want_text in text, (name, text)
