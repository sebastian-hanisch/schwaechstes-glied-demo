"""Figuren: bauen ohne Fehler, feste Achsen (fixedrange), richtige Zahl der Spuren und Werte aus der Auswertung."""

import numpy as np
import pytest

import swg_evaluation as V
import swg_experiments as XP
import swg_replicator as RP
import swg_scenario as S
import swg_visualization as Z


@pytest.fixture(scope="module")
def view():
    return V.analyse(S.Settings(n=4, runs=40, rounds=20))


def fixed(fig):
    return fig.layout.xaxis.fixedrange is True and fig.layout.yaxis.fixedrange is True


def test_trace_has_one_line_per_partner_plus_the_chain_minimum(view):
    fig = Z.build_trace(view)
    assert len(fig.data) == 4 + 1 and fixed(fig) and fig.data[-1].name == "Kettenminimum"
    assert list(fig.data[-1].y) == list(view.trace.min(axis=1))


def test_mean_curve_has_band_mean_and_reference_and_optional_exact_overlay(view):
    fig = Z.build_mean_curve(view, None)
    assert [t.name for t in fig.data] == ["mittlere Hälfte der Läufe", "Mittel über die Läufe"] and fixed(fig) and len(fig.layout.shapes) == 1
    exact = {"label": "exakt", "x": [1, 2, 3], "y": [2.0, 2.5, 3.0]}
    assert len(Z.build_mean_curve(view, exact).data) == 3


def test_histogram_has_seven_bars_that_sum_to_one_hundred(view):
    fig = Z.build_hist(view)
    assert len(fig.data[0].x) == 7 and sum(fig.data[0].y) == pytest.approx(100.0) and fixed(fig)


def test_line_figure_uses_a_log_axis_on_request():
    fig = Z.build_lines({"a": ([1, 2, 4], [1.0, 2.0, 3.0])}, "x", logx=True, xticks=[1, 2, 4])
    assert fig.layout.xaxis.type == "log" and fixed(fig) and list(fig.layout.yaxis.range) == [0.8, 7.2]


def test_start_or_learning_figure_adds_the_random_curve():
    cells = {"2": {"mean": [2.8, 3.9]}, "4": {"mean": [1.9, 1.1]}}
    fig = Z.build_start_or_learning(cells, [2.857, 1.948], ["Runde 1", "Runde 60"])
    assert [t.name for t in fig.data] == ["Runde 1", "Runde 60", "Zufall (gleichverteilt)"]


def test_ratchet_and_first_drop_figures():
    rows = [{"n": 2, "up": 328.0, "down": 117.0}, {"n": 3, "up": 5e6, "down": 78.0}]
    fig = Z.build_ratchet(rows)
    assert len(fig.data) == 2 and fig.layout.yaxis.type == "log" and fixed(fig)
    drop = [{"n": 2, "formula": 59.6, "sim": 58.0, "se": 2.0, "censored": 0.0}, {"n": 8, "formula": 16.0, "sim": 17.0, "se": 0.5, "censored": 0.0}]
    assert len(Z.build_first_drop(drop).data) == 2


def test_basin_heatmap_and_bars_and_path():
    grid = {f"{n}|{r}": {"share": [0.0] * 6 + [0.5]} for n in (2, 4) for r in (0.1, 0.5)}
    fig = Z.build_basin_heatmap(grid, (2, 4), (0.1, 0.5))
    assert np.array(fig.data[0].z).shape == (2, 2) and fixed(fig)
    bars = Z.build_basin_levels({"share": [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05]})
    assert len(bars.data[0].x) == 7
    path = Z.build_replicator_path(RP.random_starts(1, 1)[0], 4, 0.2, 0.1, 0.6, steps=200)
    assert len(path.data) == 7 and fixed(path)


def test_grouped_bars_have_one_trace_per_series():
    fig = Z.build_bars(["k = 2", "k = 4"], {"fest": [4.0, 1.1], "gelost": [4.0, 1.0]})
    assert len(fig.data) == 2 and fig.layout.barmode == "group"


def test_every_figure_builds_for_the_largest_group_and_the_smallest():
    for n in (2, 32):
        v = V.analyse(S.Settings(n=n, runs=20, rounds=30))
        assert len(Z.build_trace(v).data) == n + 1 and Z.build_hist(v).data and Z.build_mean_curve(v, XP.exact_overlay(S.Settings(n=n, runs=20, rounds=30))).data
