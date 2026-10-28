"""Die vorgerechneten Messreihen (data/measurements.json) gehören zum aktuellen Rechencode und werden von der Rechnung reproduziert.

Eigene Datei, weil jede Änderung an swg_game, swg_learning, swg_scenario, swg_replicator, swg_exact, swg_experiments oder swg_rng den Erzeuger-Hash ändert: so bleibt die Meldung
„Messdatei neu erzeugen“ von den Tests der Rechnung getrennt. Reproduziert wird plattformsicher: Simulationszellen mit anderem Seed innerhalb von vier Standardfehlern,
der Einzugsbereich (deterministisch) auf ein Prozent genau."""

import json
import math

import pytest

import swg_constants as C
import swg_experiments as XP
import swg_exact as X
import swg_replicator as RP


@pytest.fixture(scope="module")
def stored():
    return json.loads(XP.MEAS_FILE.read_text(encoding="utf-8"))


def test_file_matches_the_generator_hash(stored):
    assert stored["generator_hash"] == XP.generator_hash(), "swg_game/learning/scenario/replicator/exact/experiments/rng wurde geändert: python tools/make_measurements.py ausführen"
    assert XP.load_measurements() is not None


def test_load_returns_none_for_a_stale_hash_and_for_a_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(XP, "generator_hash", lambda: "0000000000000000")
    assert XP.load_measurements() is None
    monkeypatch.undo()
    monkeypatch.setattr(XP, "MEAS_FILE", tmp_path / "fehlt.json")
    assert XP.load_measurements() is None


def test_axes_and_sizes_match_the_code(stored):
    assert stored["runs"] == C.MEAS_RUNS and stored["basin_starts"] == C.BASIN_STARTS and stored["rounds"] == list(C.MEAS_ROUNDS)
    for key in ("e1_uniform", "e1_top"):
        assert list(stored[key]) == list(XP.RULE_CONFIGS)
        for per_n in stored[key].values():
            assert list(per_n) == [str(n) for n in C.E1_NS]
            assert all(c["runs"] == C.MEAS_RUNS and c["rounds"] == list(C.MEAS_ROUNDS) for c in per_n.values())
    assert set(stored["basin"]) == {f"{n}|{r}" for n in C.BASIN_NS for r in C.BASIN_RATIOS}
    assert list(stored["price"]["8"]) == [str(b) for b in C.E5_BS] and list(stored["premium_norm"]["8"]) == [str(a) for a in C.E5_AS]
    assert list(stored["prior"]["8"]) == [str(s) for s in C.E6_PRIORS] and set(stored["teams"]) == {f"{k}|{f}" for k in C.POOL_TEAMS for f in (0, 1)}
    assert set(stored["grow"]) == {f"{n}|{c}" for n in (8, 16) for c in ("sum", "mean")}


def test_every_stored_mean_is_a_level_between_one_and_seven_and_every_share_a_share(stored):
    def walk(obj):
        if isinstance(obj, dict):
            if "mean" in obj and "se" in obj:
                assert all(1.0 <= m <= 7.0 for m in obj["mean"]) and all(s >= 0 for s in obj["se"]) and all(0.0 <= t <= 1.0 for t in obj["top"]), obj
            else:
                for v in obj.values():
                    walk(v)
    walk({k: v for k, v in stored.items() if k not in ("basin", "unchanged", "first_drop", "runs", "basin_starts", "rounds", "generator_hash")})


def test_basin_shares_are_distributions_without_unconverged_starts(stored):
    for key, cell in stored["basin"].items():
        assert cell["none"] == 0.0 and sum(cell["share"]) == pytest.approx(1.0, abs=1e-9), key
        assert 1.0 <= cell["emin"] <= 7.0 and cell["starts"] == C.BASIN_STARTS and cell["max_steps"] <= C.E3_STEPS


def test_the_basin_of_the_top_level_shrinks_with_n_in_every_column_of_the_grid(stored):
    for r in C.BASIN_RATIOS:
        tops = [stored["basin"][f"{n}|{r}"]["share"][-1] for n in C.BASIN_NS]
        assert all(x >= y - 0.005 for x, y in zip(tops, tops[1:])), (r, tops)


def test_a_basin_cell_is_reproduced_to_one_percent():
    for n, r in ((2, 0.1), (8, 0.5)):
        stored_cell = json.loads(XP.MEAS_FILE.read_text(encoding="utf-8"))["basin"][f"{n}|{r}"]
        again = RP.basin(n, r, C.BASIN_STARTS, XP.seed_of(f"basin|{n}|{r}"))
        assert max(abs(a - b) for a, b in zip(again["share"], stored_cell["share"])) < 0.01


@pytest.mark.parametrize("key,rule,n,idx", [("e1_uniform", "Logit, Summe (λ = 5)", "4", 1), ("e1_top", "Logit, Mittel (Λ = 50)", "8", 2), ("e1_top", "Beste Antwort (ε = 0,01)", "4", 2)])
def test_a_simulation_cell_is_reproduced_within_four_standard_errors(stored, key, rule, n, idx):
    start = C.START_UNIFORM if key == "e1_uniform" else C.START_TOP
    c = stored[key][rule][n]
    fresh = XP.cell(XP.base_settings(C.MEAS_RUNS, f"neu|{key}|{rule}|{n}", n=int(n), start=start, **XP.RULE_CONFIGS[rule]), (1, 10, 60, 150, 400)[: idx + 1])
    se = math.hypot(c["se"][idx], fresh["se"][idx])
    assert abs(c["mean"][idx] - fresh["mean"][idx]) < 4 * se, (rule, n, c["mean"][idx], fresh["mean"][idx])


def test_round_one_of_the_uniform_start_equals_the_random_minimum_in_every_stored_cell(stored):
    for rule, per_n in stored["e1_uniform"].items():
        for n, c in per_n.items():
            assert abs(c["mean"][0] - X.uniform_expected_min(int(n))) < 4 * max(c["se"][0], 1e-9), (rule, n)


def test_cooperative_start_cells_begin_at_seven(stored):
    for per_n in stored["e1_top"].values():
        assert all(c["mean"][0] == 7.0 and c["se"][0] == 0.0 for c in per_n.values())


def test_first_drop_rows_are_stored_with_five_times_the_runs_and_agree_with_the_formula(stored):
    for eps, rows in stored["first_drop"].items():
        for r in rows:
            assert r["formula"] == pytest.approx(X.first_drop_rounds(r["n"], float(eps)), rel=1e-12)
            assert r["censored"] <= 0.01
