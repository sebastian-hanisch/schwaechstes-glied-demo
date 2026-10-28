"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Kennzahlen und die Verdikte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

from fpdf import FPDF

import swg_evaluation as V

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "≠": "!=", "√": "Wurzel", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"', "∞": "unendlich",
            "→": "->", "ε": "eps", "β": "beta", "α": "alpha", "θ": "theta", "κ": "kappa", "ρ": "rho", "λ": "lambda", "Λ": "Lambda", "φ": "phi", "σ": "sigma", "π": "pi", "δ": "delta",
            "Φ": "Phi", "Δ": "Delta", "×": "x", "↔": "<->", "≈": "~", "∝": "~", "‖": "||", "½": "1/2", "⁻": "^-", "¹": "1", "⁵": "5", "⁶": "6", "⁰": "0", "⁴": "4", "³": "3", "⁸": "8",
            "⁷": "7", "⁹": "9", "∈": "in", "…": "...", "‚": ",", "‘": "'", "’": "'", "”": '"', "µ": "mu"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1: Sonderzeichen ersetzen, Markdown-Sterne entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def build_pdf(view: V.MainView, verdicts: list[tuple[str, str]]) -> bytes:
    s = view.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Das schwächste Glied: warum die Kette langsam wird"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"{V.rule_label(s)}; Start {s.start}; n = {s.n} Partner; Praemie a = {V.de(s.a, 1)}, Einsatzpreis b = {V.de(s.b, 2)}, Grundniveau c = {V.de(s.c, 1)}; Vorabsprache s = {V.de(s.prior, 0)}; "
                                f"{s.rounds} Runden, {view.runs_used} Laeufe, Zufalls-Seed {s.seed}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Kennzahlen", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"Minimum in Runde 1 (Mittel): {V.de(float(view.mean_min[0]), 2)}; in Runde {s.rounds}: {V.de(float(view.mean_min[-1]), 2)}; Laeufe mit Minimum {s.levels}: {V.pct(view.share_top, 1)}; "
                                f"Auszahlung je Partner: {V.de(view.pay_end, 3)}; Fehlertoleranz je Partner: {V.pct(view.tolerance, 1)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Verdikte", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Lernregeln und Eigennutz sind Modellannahmen, keine Aussagen ueber Menschen oder echte Umschlagstellen. Alle Zahlen sind eigene Rechnung, "
                                "durch Aufzaehlung, zweite Rechenwege und Simulation gestuetzt, nicht gegen Literatur geprueft."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
