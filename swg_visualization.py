"""Plotly-Figuren (alle Achsen fest, fixedrange). Nur Darstellung: jede Zahl kommt fertig aus Kern, Auswertung oder Messreihe."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import swg_constants as C
import swg_evaluation as V
import swg_replicator as RP


def _base(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(title=title, height=height, margin=dict(l=55, r=15, t=45 if title else 15, b=55), legend=dict(orientation="h", y=-0.28, x=0))
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _level_axis(fig: go.Figure, levels: int = C.E_LEVELS, title: str = "Leistungsstufe") -> None:
    fig.update_yaxes(title=title, range=[0.5, levels + 0.5], tickmode="array", tickvals=list(range(1, levels + 1)))


def _dec(x: float, nd: int = 2) -> str:
    return V.de(x, nd)


# --- Hauptansicht ---------------------------------------------------------------------------------------------------------------------------
def build_trace(view: V.MainView) -> go.Figure:
    """Die Stufen aller Partner im ersten Lauf (dünn) und das Kettenminimum (dick): die Kette läuft mit der langsamsten Stufe."""
    t = np.arange(1, view.trace.shape[0] + 1)
    fig = go.Figure()
    for i in range(view.trace.shape[1]):
        fig.add_trace(go.Scatter(x=t, y=view.trace[:, i] + (i - view.trace.shape[1] / 2) * 0.012, mode="lines", line=dict(width=1, shape="hv", color="rgba(76,120,168,0.55)"),
                                 name="Partner" if i == 0 else None, showlegend=i == 0, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=view.trace.min(axis=1), mode="lines", name="Kettenminimum", line=dict(width=3.5, shape="hv", color=C.COLOR_MIN)))
    fig.update_xaxes(title="Runde")
    _level_axis(fig, view.settings.levels)
    return _base(fig)


def build_mean_curve(view: V.MainView, exact: dict | None = None) -> go.Figure:
    """Mittleres Kettenminimum über alle Läufe mit dem Bereich zwischen oberem und unterem Quartil; Bezugslinien: Zufall bei gleichverteilter Wahl und (falls vorhanden) die exakte Rechnung.

    `exact`: {"label": Text, "x": Runden, "y": Werte} aus swg_exact (Kette der besten Antwort oder Pfadaufzählung des Logit)."""
    t = np.arange(1, len(view.mean_min) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([view.q75, view.q25[::-1]]), fill="toself", fillcolor="rgba(192,118,43,0.18)", line=dict(width=0),
                             name="mittlere Hälfte der Läufe", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=t, y=view.mean_min, mode="lines", name="Mittel über die Läufe", line=dict(width=3, color=C.COLOR_MIN)))
    fig.add_hline(y=view.random_min, line=dict(color=C.COLOR_GREY, dash="dash"), annotation_text=f"Zufall (gleichverteilt): {_dec(view.random_min)}", annotation_position="top right")
    if exact is not None:
        fig.add_trace(go.Scatter(x=exact["x"], y=exact["y"], mode="markers" if len(exact["x"]) <= 3 else "lines", name=exact["label"],
                                 marker=dict(size=10, symbol="diamond-open", color=C.COLOR_GOOD, line=dict(width=2)), line=dict(color=C.COLOR_GOOD, dash="dot", width=2)))
    fig.update_xaxes(title="Runde")
    fig.update_yaxes(title="Kettenminimum", range=[0.8, view.settings.levels + 0.2])
    return _base(fig)


def build_hist(view: V.MainView) -> go.Figure:
    """Anteil der Läufe je Kettenminimum in der letzten Runde."""
    levels = list(range(1, len(view.hist) + 1))
    fig = go.Figure(go.Bar(x=levels, y=100 * view.hist, marker_color=list(C.LEVEL_COLORS[: len(levels)]), text=[V.pct(h, 0) for h in view.hist], textposition="outside"))
    fig.update_xaxes(title=f"Kettenminimum in Runde {view.settings.rounds}", tickmode="array", tickvals=levels)
    fig.update_yaxes(title="Anteil der Läufe (%)", range=[0, 112])
    return _base(fig, height=300)


# --- Experimente ----------------------------------------------------------------------------------------------------------------------------
def build_lines(series: dict, xtitle: str, ytitle: str = "Kettenminimum", logx: bool = False, xticks: list | None = None, yrange=(0.8, C.E_LEVELS + 0.2), height: int = 360) -> go.Figure:
    """Mehrere Linien: `series` = {Name: (x-Liste, y-Liste)}."""
    fig = go.Figure()
    for i, (name, (xs, ys)) in enumerate(series.items()):
        color = C.LEVEL_COLORS[(i * 2) % len(C.LEVEL_COLORS)]
        fig.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="lines+markers", name=name, line=dict(width=2.5, color=color), marker=dict(size=7, color=color)))
    fig.update_xaxes(title=xtitle, type="log" if logx else "linear")
    if xticks is not None:
        fig.update_xaxes(tickmode="array", tickvals=xticks, ticktext=[str(x).replace(".", ",") for x in xticks])
    fig.update_yaxes(title=ytitle, range=list(yrange))
    return _base(fig, height=height)


def build_start_or_learning(cells: dict, uniform_curve: list[float], rounds_names: list[str]) -> go.Figure:
    """E1: Kettenminimum über der Gruppengröße in den ausgewählten Runden (Linien) und der Zufall bei gleichverteilter Wahl (gestrichelt)."""
    ns = sorted(cells, key=int)
    series = {name: ([int(n) for n in ns], [cells[n]["mean"][i] for n in ns]) for i, name in enumerate(rounds_names)}
    fig = build_lines(series, "Gruppengröße n", logx=True, xticks=[int(n) for n in ns])
    fig.add_trace(go.Scatter(x=[int(n) for n in ns], y=uniform_curve, mode="lines", name="Zufall (gleichverteilt)", line=dict(color=C.COLOR_GREY, dash="dash", width=2)))
    return fig


def build_ratchet(chain_rows: list[dict]) -> go.Figure:
    """E2: erwartete Runden hinauf (alle 1 → Minimum ≥ 4) und hinab (alle 7 → Minimum ≤ 3) je Gruppengröße, logarithmisch."""
    ns = [r["n"] for r in chain_rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[str(n) for n in ns], y=[r["up"] for r in chain_rows], name="hinauf (alle 1 → Minimum ≥ 4)", marker_color=C.COLOR_GOOD))
    fig.add_trace(go.Bar(x=[str(n) for n in ns], y=[r["down"] for r in chain_rows], name="hinab (alle 7 → Minimum ≤ 3)", marker_color=C.COLOR_RED))
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Gruppengröße n")
    fig.update_yaxes(title="erwartete Runden (logarithmisch)", type="log")
    return _base(fig)


def build_first_drop(rows: list[dict]) -> go.Figure:
    """E2: erste Runde mit Minimum unter 7: Formel gegen Simulation."""
    ns = [r["n"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[r["formula"] for r in rows], mode="lines+markers", name="Formel 1 + 1/(1 − q^n)", line=dict(color=C.COLOR_GOOD, width=2.5)))
    fig.add_trace(go.Scatter(x=ns, y=[r["sim"] for r in rows], mode="markers", name="Simulation (Mittel)", error_y=dict(type="data", array=[4 * r["se"] for r in rows], visible=True),
                             marker=dict(size=10, color=C.COLOR_MIN)))
    fig.update_xaxes(title="Gruppengröße n", type="log", tickmode="array", tickvals=ns)
    fig.update_yaxes(title="erste Runde mit Minimum < 7", type="log")
    return _base(fig)


def build_basin_heatmap(grid: dict, ns, ratios, key: str = "top") -> go.Figure:
    """E3: Einzugsbereich der obersten Stufe (Anteil der Zufallsstarts, die dort enden) über Gruppengröße und b/a."""
    z = []
    for r in ratios:
        row = []
        for n in ns:
            cell = grid[f"{n}|{r}"]
            row.append(100 * (cell["share"][-1] if key == "top" else cell["share"][0]))
        z.append(row)
    fig = go.Figure(go.Heatmap(z=z, x=[str(n) for n in ns], y=[V.de(r, 2) for r in ratios], colorscale="Viridis", zmin=0, zmax=100, colorbar=dict(title="%", thickness=12),
                               hovertemplate="n = %{x}, b/a = %{y}: %{z:.1f} %<extra></extra>", text=[[f"{v:.0f}" for v in row] for row in z], texttemplate="%{text}"))
    fig.update_xaxes(title="Gruppengröße n")
    fig.update_yaxes(title="b/a (Einsatzpreis zur Prämie)")
    return _base(fig, height=340)


def build_basin_levels(cell: dict) -> go.Figure:
    """E3: Anteil der Zufallsstarts, die bei Stufe 1 bis 7 enden (eine Zelle)."""
    levels = list(range(1, len(cell["share"]) + 1))
    fig = go.Figure(go.Bar(x=levels, y=[100 * v for v in cell["share"]], marker_color=list(C.LEVEL_COLORS[: len(levels)])))
    fig.update_xaxes(title="Endstufe", tickmode="array", tickvals=levels)
    fig.update_yaxes(title="Anteil der Starts (%)", range=[0, 100])
    return _base(fig, height=300)


def build_replicator_path(p0: np.ndarray, n: int, a: float, b: float, c: float, steps: int = 3000) -> go.Figure:
    """E3: die Entwicklung der Anteile aller sieben Stufen für einen Start (Euler-Schritte)."""
    p = p0.reshape(1, -1).copy()
    hist = [p[0].copy()]
    for _ in range(steps):
        p = RP.step(p, n, a, b, c)
        hist.append(p[0].copy())
    hist = np.array(hist)
    idx = np.unique(np.linspace(0, steps, 300).astype(int))
    fig = go.Figure()
    for lev in range(hist.shape[1]):
        fig.add_trace(go.Scatter(x=idx, y=hist[idx, lev], mode="lines", name=f"Stufe {lev + 1}", line=dict(width=2.2, color=C.LEVEL_COLORS[lev])))
    fig.update_xaxes(title="Schritt")
    fig.update_yaxes(title="Anteil", range=[0, 1.02])
    return _base(fig, height=320)


def build_bars(labels: list[str], series: dict, ytitle: str = "Kettenminimum", yrange=(0, C.E_LEVELS + 0.4)) -> go.Figure:
    """Gruppierte Säulen: `series` = {Name: Werte-Liste (je Label)}."""
    fig = go.Figure()
    for i, (name, ys) in enumerate(series.items()):
        fig.add_trace(go.Bar(x=labels, y=ys, name=name, marker_color=C.LEVEL_COLORS[(i * 2 + 1) % len(C.LEVEL_COLORS)], text=[V.de(y, 2) for y in ys], textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title=ytitle, range=list(yrange))
    return _base(fig)
