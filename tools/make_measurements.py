"""Rechnet alle vorgerechneten Messreihen (Regel × Gruppengröße, Einzugsbereiche, Schärfe, Einsatzpreis, Prämie, Teams, Vorabsprache, wachsende Gruppen, erste Abweichung)
neu und schreibt data/measurements.json samt Erzeuger-Hash.

Aufruf (im Demo-Ordner, mit dem Test-venv):  python tools/make_measurements.py
Ändert sich eine der Dateien swg_game.py, swg_learning.py, swg_scenario.py, swg_replicator.py, swg_experiments.py oder swg_rng.py, stimmt der Hash nicht mehr
(tests/test_measurements.py schlägt an, die App zeigt die vorgerechneten Reihen dann nicht); danach diesen Aufruf wiederholen. Dauer: einige Minuten; jede Zelle hat einen
festen Zufalls-Seed aus ihrer Bezeichnung (swg_experiments.seed_of), die Reihen sind also bei gleichem Code Zahl für Zahl dieselben."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import swg_experiments as XP  # noqa: E402


def main():
    start = time.time()

    def progress(msg):
        print(f"[{time.time() - start:6.1f} s] {msg}", flush=True)

    payload = XP.write_measurements(progress=progress)
    print(f"geschrieben: {XP.MEAS_FILE}  ({time.time() - start:.0f} s, Hash {payload['generator_hash']}, {payload['runs']} Läufe je Zelle, {payload['basin_starts']} Starts je Einzugsbereich)")


if __name__ == "__main__":
    main()
