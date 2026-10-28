# Das schwächste Glied: warum die Kette langsam wird – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-schwaechstes-glied-demo.streamlit.app/)** -->

Fünftes Stück der Reihe **Koordinationsspiele** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vehikel: **Umschlagkette aus n Partnern.** Jeder wählt eine **Leistungsstufe 1 bis 7** (Schichtbesetzung, Torbelegung), die Kette läuft mit der **langsamsten** Stufe. Jede Stufe, auf der alle Partner stehen, ist für a > b > 0 ein striktes Gleichgewicht (durch Aufzählung für n ≤ 4 geprüft): **sieben gleich gut erreichbare Gleichgewichte**, das mit allen auf 7 ist für alle das beste. Die Demo lässt **vier Lernregeln** (Logit-Erfahrungsbewertung in zwei Fassungen, beste Antwort mit Zittern, fiktives Spiel) auf diesem Spiel laufen und fragt: **was ist Start, was ist Lernen, warum rutscht das Minimum nach unten (eine Ratsche), wie klein wird der Einzugsbereich der obersten Stufe, und helfen Einsatzpreis, Prämie, Teams, Vorabsprache oder wachsende Gruppen?**

Abgrenzung: `koordinationsspiel-demo` (erstes Stück) zeigt die Gleichgewichte und die Schwelle p* der Hirschjagd für zwei Spieler; hier ist die Fehlertoleranz δ* das Gegenstück für n Partner. `gleichgewichtsauswahl-demo` (zweites Stück) zeigt, welche Konvention auf Dauer gewählt wird, `konventionswechsel-demo` (drittes Stück), wie lange ein Wechsel dauert, `globale-spiele-demo` (viertes Stück), was Unsicherheit über den Zustand ändert. `noregret-demo` lässt Lkw einander **meiden** (Überlastungsspiel, Hedge-Lernen), `replikator-demo` rechnet den Replikator für Strategien wiederholter Spiele, `nash-demo` Gleichgewichte kleiner Matrixspiele; hier ist das Ergebnis das **Minimum** der Gruppe. **Lernregeln und Eigennutz sind Modellannahmen**, keine Messung an Menschen oder echten Umschlagstellen.

## Kernfrage

*Welches der sieben gleich gut erreichbaren Gleichgewichte wählt eine Gruppe, die aus Erfahrung lernt, und was ändert sich mit Gruppengröße, Einsatzpreis und Vorabsprache?* Eigenes dieser Demo sind (1) die Trennung von **Start** (Runde 1 ist bei gleichverteilter Wahl der reine Zufall, exakt gerechnet) und **Lernen**, (2) die **Ratsche** mit exakter Markov-Kette, (3) die **Fehlertoleranz** δ* = 1 − (b/a)^(1/(n−1)) je Partner, (4) der **Einzugsbereich** der obersten Stufe im Replikator, (5) **Gegenproben** (Einsatzpreis, Prämie, Teams, Vorabsprache, wachsende Gruppen) bis 400 Runden.

## Modell

Spiel: n Partner wählen je eine Stufe x ∈ {1, …, 7}; ein Partner, dessen Mitspieler zusammen das Minimum m haben, bekommt **π(x, m) = a·min(x, m) − b·x + c** (Zeitersparnis je Sendung). Die Werte **a = 0,2, b = 0,1, c = 0,6** sind die des klassischen Laborspiels, hier nur Modellparameter; bei allen auf 7 sind es 1,30, bei allen auf 1 sind es 0,70. Lernregeln (jede sieht nach einer Runde nur das Minimum der anderen und bewertet **alle** sieben Stufen gegenfaktisch):

- **Logit-Erfahrungsbewertung mit Vergessen φ**, Zähler N_x ← φ·N_x + π(x, m), Nenner W ← φ·W + 1, in zwei Fassungen: **Summe** (Wahl ∝ exp(λ·N_x); die Schärfe wächst in den ersten Runden mit der Summe der Gewichte bis λ/(1−φ)) und **Mittel** (∝ exp(Λ·N_x/W), Schärfe von Anfang an Λ). Das frühere Skript V4 nannte die Summenfassung „Mittelwert“, der Code summiert; **in der App heißt „Summe“, was summiert wird.**
- **Beste Antwort mit Zittern ε:** spielt die Stufe des Minimums der anderen aus der Vorrunde, mit Wahrscheinlichkeit ε eine gleichverteilte Stufe.
- **Fiktives Spiel:** Mittel über alle Runden (φ = 1) mit Schärfe 1 000, praktisch die beste Antwort auf die gesamte Erfahrung.
- **Replikator** im Mittelwertfeld (nur Experiment E3): u_x = a·Σ_{k≤x} S(k)^(n−1) − b·x + c mit S(k) = P(Stufe ≥ k); Zeit in Einheiten der Prämie (der Pfad hängt nur an b/a).
- **Prämie und Schärfe getrennt:** „Schärfe an die Prämie anpassen“ (Standard an) misst die Schärfe in Einheiten von a/0,2; dann zählt nur b/a. Aus, macht eine höhere Prämie die Wahl zugleich schärfer.
- **Vorabsprache s** ist ein **Anfangszustand der Lerner** (s fiktive Runden, in denen alle auf 7 spielten), kein strategisches Reden. **Wachsende Gruppen:** Start mit 2 Partnern, alle 10 bzw. 25 Runden einer mehr; wer dazukommt übernimmt die Erfahrung eines zufälligen Mitglieds.

Alle Herleitungen sind eigene Rechnung, durch Aufzählung, zweite Rechenwege und Simulation gestützt, nicht gegen Literatur geprüft.

## Befunde und Korrekturen gegenüber dem Plan

Alle Zahlen sind in dieser Demo neu gemessen (`tests/test_claims.py`). Die Messreihen (1 000 Läufe je Zelle, Einzugsbereiche mit 5 000 Zufallsstarts) sind vorgerechnet (`data/measurements.json` mit Erzeuger-Hash, `tools/make_measurements.py`); die exakten Rechnungen (Aufzählung, Markov-Kette) laufen live. Simulationen haben feste Zufalls-Seeds (SplitMix64) und werden nur mit Marge von vier Standardfehlern verglichen. „Runde 10 / 60 / 400“ ist das mittlere Kettenminimum in dieser Runde.

| Nr. | Hypothese | Antwort | Beleg |
|---|---|---|---|
| H1 | Hook des früheren Skripts V4: bei n ≥ 4 fällt das Minimum binnen 10 Runden auf 1,0 bis 1,2, bei n = 2 steigt es auf etwa 4 | **Ja** | Summe (λ = 5, φ = 0,9), gleichverteilter Start, Runde 10 für n = 2 / 3 / 4 / 6 / 8 / 16 / 32: 3,59 / 1,64 / 1,18 / 1,02 / 1,00 / 1,00 / 1,00. n = 2: Runde 1 2,79, Runde 60 3,91. Ausnahme: n = 3 liegt mit 1,64 außerhalb von 1,0 bis 1,2 (`test_h1_*`) |
| H2 | Für große n ist der Abfall überwiegend Startverteilung, nicht Lernen | **Ja** | Runde 1 liegt in allen sieben Gruppengrößen innerhalb von vier Standardfehlern des exakten Zufallsminimums. Von 7 bis Runde 60 sind es für n = 3 / 4 / 6 / 8 / 16 / 32 schon in Runde 1 88 / 86 / 90 / 94 / 98 / 100 % der Strecke. Zufallsminimum n = 2 / 3 / 4 / 6 / 8 / 16 / 32: 2,857 / 2,286 / 1,948 / 1,571 / 1,372 / 1,090 / 1,007 (`test_h2_*`) |
| H3 | Mit Rauschen fällt das Minimum bei n ≥ 3 aus kooperativem Start unter jeder Regel, die Geschwindigkeit hängt an Schärfe und Fassung | **Ja, mit vier Ausnahmen** | Nach 400 Runden liegt das Minimum in 14 von 18 Zellen (drei Regeln × n = 3, 4, 6, 8, 16, 32) unter 4; die Ausnahmen: Summe n = 3 (4,06), Mittel n = 3 (6,95), n = 4 (6,71), n = 6 (4,85). Für n = 8: unter 4 ab Runde 10 (Summe), 60 (beste Antwort ε = 0,01), erst 400 (Mittel: Runde 60 6,00, Runde 150 4,90, Runde 400 2,16). **60 Runden genügen nicht.** Die Summe bricht anfangs ein, weil die Schärfe nach der ersten Beobachtung ein Zehntel der Endschärfe hat: p(7) = 0,41 (`test_h3_*`) |
| H4 | Rauschfreie Regeln (beste Antwort ohne Zittern, fiktives Spiel) lassen das Minimum auf dem Startwert | **Ja, mit einer Ausnahme** | Beste Antwort ε = 0: in 1 000 von 1 000 Läufen für jedes n = 2 bis 32 gleich in Runde 1 und Runde 400. Fiktives Spiel: ebenso für n ≥ 3, **nicht bei n = 2**: nur 39,1 % der Läufe unverändert, das Mittel steigt von 2,84 auf 3,92 (Gleichstände der Erfahrungsbewertung wirken als Zufall). Bei n = 2 vertauschen beide mit ε = 0 nur ihre Stufen; mit ε = 0,01 ist das stationäre E[min] exakt 2,857 (`test_h4_*`) |
| H5 | Ratsche: erste Abweichung aus „alle 7“ nach 1 + 1/(1 − q^n) Runden, Hinaufklettern wächst wie ε^−n | **Ja** | Exakte Kette (ε = 0,01): Runden von „alle 1“ bis Minimum ≥ 4 für n = 2 / 3 / 4: 328 / 5,3·10^6 / 9,4·10^8; von „alle 7“ bis Minimum ≤ 3: 117 / 78 / 59; stationäres E[min] für n ≥ 3 höchstens 1,0002 (nie über 1,03). Formel gegen Simulation (5 000 Läufe): ε = 0,01, n = 2 / 4 / 8 / 16 / 32: 59,6 / 30,5 / 16,0 / 8,8 / 5,2 gegen 58,9 / 31,6 / 16,3 / 8,9 / 5,1; in 10 von 10 Zellen (ε = 0,01 und 0,05) innerhalb von vier Standardfehlern. Die Treffzeiten für n = 3, 4 sind schlecht konditioniert und nur als Band geprüft (`test_h5_*`) |
| H6 | Der Einzugsbereich der Stufe 7 schrumpft mit n | **Ja** | Anteil der 5 000 Zufallsstarts, die der Replikator bei Stufe 7 enden lässt, n = 2 / 3 / 4 / 6 / 8 / 16 / 32: bei b/a = 0,1: 85,8 / 39,8 / 14,8 / 2,3 / 0,5 / 0,0 / 0,0 %; bei b/a = 0,5: 7,5 / 0,3 / 0 / 0 / 0 / 0 / 0 %. Gleichverteilung als Start: n = 2 endet bei Stufe 4, ab n = 3 bei Stufe 1 (`test_h6_*`) |
| H7 | Ein niedriger Einsatzpreis hilft kleinen Gruppen voll, großen nur teilweise | **Ja** | Runde 60 (a = 0,2): n = 2, b = 0,05 / 0,02: 6,98 / 7,00; n = 4, b = 0,01 / 0,02 / 0,05: 6,88 / 5,57 / 2,28 (b = 0,02 in Runde 400: 6,89); n = 8, b = 0,01 / 0,02: 3,16 / 1,71 (b = 0,01 in Runde 400: 4,93); n = 16, b = 0,01: 1,12. In Runde 400 steht für n = 8 und 16 bei keinem Einsatzpreis ein Lauf auf 7. Der Plan sagte „reicht aber nicht“: das gilt erst ab n = 8 (`test_h7_*`) |
| H8 | Die Prämie rettet kleine Gruppen, große nur zum Teil; ohne Trennung von Prämie und Schärfe ist sie viel schwächer | **Ja** | Schärfe auf die Prämie normiert, a = 2 (b/a = 0,05): n = 2 Runde 400: 7,00; n = 4: 6,88 (Runde 60) und 7,00 (Runde 400); n = 8: 3,13 und 4,96, nie die oberste Stufe; n = 16: 1,11 und 1,12. Nicht normiert (die Prämie macht die Wahl zugleich schärfer): n = 4, a = 2 in Runde 400: 2,70 gegen 7,00; n = 8: 1,37. **Korrektur:** „höhere Schärfe schadet“ gilt nicht allgemein: bei gleichverteiltem Start friert das Minimum bei hoher Schärfe am Startwert ein (n = 4, Summe, Runde 60 für λ = 2 / 5 / 20 / 50 / 200: 1,00 / 1,13 / 1,78 / 1,98 / 1,92, Start 1,92 bis 2,01), bei kooperativem Start hält hohe Schärfe die 7 (n = 4, λ = 5 / 20 / 50: 2,80 / 6,48 / 6,97). Die Reihe über die Schärfe ist **nicht monoton** (`test_h8_*`) |
| H9 | Eine Vorabsprache verblasst; ohne Erneuerung fällt auch s = 30 bei n ≥ 8 | **Ja** | s = 30, Runde 10 / 60 / 400: n = 8: 7,00 / 6,70 / 2,62; n = 16: 7,00 / 5,79 / 1,00; n = 4: 7,00 / – / 6,79 (81 % der Läufe auf 7); n = 2, s = 5, Runde 60: 6,99 (98,5 %). Ohne Absprache (s = 0) in Runde 400 für n = 8 und 16: 1,00 (`test_h9_*`) |
| H10 | Prämie und Vorabsprache zusammen halten n = 16 | **Ja (bis Runde 400)** | a = 1,0 und s = 5: n = 16 in Runde 10 / 60 / 400: 6,91 / 6,95 / 6,95, in 94,6 % der Läufe steht die Kette in Runde 400 auf 7; n = 8: 6,99 / 7,00 / 7,00 (99,9 %). Allein: Prämie a = 1,0 gibt für n = 16 1,03, Absprache s = 5 gibt 1,00. Über 400 Runden hinaus nicht gemessen (`test_h10_*`) |
| H11 | Wachsende Gruppen (Weber) erreichen hohe Leistung | **Modellbefund: nur verzögert** | Start mit 2 Partnern, alle 10 bzw. 25 Runden einer mehr, kooperativer Start. Summe, n = 8, Runde 10 / 60 / 150 / 400: volle Gruppe 1,42 / 1,14 / 1,04 / 1,00; alle 25 Runden 5,80 / 6,04 / 5,65 / 2,67. Mittel, n = 8: volle Gruppe 6,72 / 6,11 / 4,96 / 2,14, alle 25 Runden 6,98 / 6,97 / 6,56 / 3,54. n = 16: nach 400 Runden 1,00 bis 1,19 in allen sechs Zellen. Der Einbruch unter 4 kommt in 6 von 8 Zellen später (nicht bei n = 8, Mittel); in Runde 400 liegt das Wachsen in 6 von 8 Zellen über der vollen Gruppe, aber auf fallender Bahn (`test_h11_*`) |
| H12 | Kleinere Teams, jede Runde neu gelost, koordinieren besser | **Nein** | Pool von 16, Team-Minimum in Runde 60, Teamgröße 2 / 4 / 8 / 16, fest: 3,99 / 1,14 / 1,00 / 1,00; gelost: 4,00 / 1,00 / 1,00 / 1,00. Gelost ist in 0 von 4 Teamgrößen besser. Kleinere feste Teams helfen nur bis etwa Stufe 4 (`test_h12_*`) |

### Korrekturen gegenüber dem Plan

- **Hook:** nicht „das Minimum fällt auf 1“, sondern **Ratsche**: bei n ≥ 8 ist der Abfall fast ganz der Start (H2), bei kooperativem Start in der Summenfassung ein Anlaufeffekt, in der Mittelfassung fällt das Minimum bei n = 8 erst zwischen Runde 150 und 400 unter 4 (H3). Lernen allein bewegt das Minimum bei rauschfreien Regeln nicht (H4).
- **Zahlen mit mehr Läufen:** der Plan nannte für n = 2 (Summe, Runde 10 / 60) 3,75 / 4,04 aus vier Seeds zu je 250 Läufen; mit 1 000 Läufen und anderen Seeds sind es 3,59 / 3,91 (Standardfehler etwa 0,06). Einzugsbereiche: Plan 0,863 / 0,406 / 0,158 / 0,028 / 0,007 (n = 2 bis 8, b/a = 0,1), hier 0,858 / 0,398 / 0,148 / 0,023 / 0,005; die Unterschiede liegen im Stichprobenfehler der Zufallsstarts.
- **V4-Namensverwechslung:** die Summenfassung hieß im Skript „Mittelwert“; die App zeigt beide Fassungen und nennt sie nach dem, was gerechnet wird.
- **H4 gilt nicht für das fiktive Spiel bei n = 2** (siehe oben); der Plan sprach allgemein von rauschfreien Regeln.
- **H8 „höhere Schärfe schadet“** ist eingeschränkt (siehe oben).
- **Neu gemessen:** H10 (Plan: offen) und H11 (Plan: offen, nur Literatur); der Plan nannte für H10 nur den Wert bis Runde 60.
- **Port und Name:** Port 9207 (belegt: 9203 bis 9206, 9210, 9211), Präfix `swg`, Linie `koordination`.

## Methodik

- **Läufe:** alle Läufe eines Aufrufs laufen gleichzeitig (numpy über Läufe, Spieler und Stufen); die Gleichverteilten kommen blockweise aus SplitMix64 (`swg_rng.py`, wörtliche Kopie aus `ab-test-demo`, Kopie-Test). Zeitbudget: Läufe · Partner · Runden höchstens 10 Millionen; darüber rechnet die App mit weniger Läufen und sagt es (bei n = 32, 400 Runden 781 statt 1 000).
- **Exakte Referenzen:** Pfadaufzählung des Logit-Lernens (n = 2, drei Runden) und Markov-Kette der besten Antwort über alle Profile (n ≤ 3 in der Hauptansicht, n ≤ 4 im Experiment E2); beide erscheinen als grüne Bezugskurve.
- **Replikator:** Euler-Schritte mit η = 0,5, Starts je Zelle aus Dirichlet(1, …, 1) (normierte Exponentialverteilte aus SplitMix64); ein Start ist konvergiert, wenn ein Anteil über 1 − 10^−9 liegt; höchstens 20 000 Schritte (nicht konvergiert: 0 in allen Zellen).
- **„Fällt“, „bleibt“, „steigt“** heißen: Unterschied zweier Mittel unter −4, innerhalb ±4 oder über +4 Standardfehlern (quadratisch addiert).

## Ehrliche Grenzen

- **Modell, keine Messung:** vier Lernregeln, sieben Stufen, a = 0,2, b = 0,1, c = 0,6, gleiche Partner. Gemessen ist, dass das Minimum bei kooperativem Start unter den drei Regeln mit Rauschen in 14 von 18 Zellen nach 400 Runden unter 4 liegt (H3), nicht mehr; die Geschwindigkeit hängt an Start, Fassung und Schärfe um Größenordnungen.
- **Prämie unter 0,2 und Einsatzpreise über 0,19** gibt es nicht (a > b hält alle sieben Stufen strikt); die Reihen zur Prämie gehen von 0,2 bis 2,0.
- **H10 nur bis Runde 400, H11 nur mit einer Übernahmeregel** (der Neuling übernimmt die Erfahrung eines Mitglieds). Ob Wachsen in Laborgruppen hilft, ist hier weder bestätigt noch widerlegt; der Modellbefund (nur verzögert) sagt nichts über Menschen.
- **Vorabsprache ist ein Anfangszustand**, kein Reden; Cheap Talk als Strategie ist nicht gerechnet.
- **Mittelwerte über 1 000 Läufe** (Live-Experimente 300 Läufe); ein „bleibt“ heißt kein Unterschied innerhalb von vier Standardfehlern, nicht „gleich“.
- **Replikator ist ein Mittelwertfeld** (große Gruppe, unabhängige Mitspieler), kein Modell einer festen Kette; die Zufallsstarts sind gleichverteilt über die Verteilungen.
- **Treffzeiten der Kette** bei kleinem ε (n = 3, 4) sind schlecht konditioniert; sie stehen als Größenordnung (5,3·10^6, 9,4·10^8), nicht als Ziffern.
- **Beste Antwort bei Gleichstand** wählt die höhere Stufe; im Slider-Bereich (a ≥ 0,2 > b) tritt der Fall nicht auf.

## Quellen

- **Goeree und Holt (2005), Games and Economic Behavior 51, 349–364:** laut Abstract (Suchtreffer, nicht im Volltext gelesen) senken höhere Anstrengungskosten die Leistung im Minimum-Spiel, vorhergesagt durch ein stochastisches Potential. Die Demo bildet Lernmodelle nach und beansprucht keine Übereinstimmung; sie zeigt, dass im Lernmodell ein höherer Einsatzpreis das Minimum senkt (H7), ohne die Arbeit zu prüfen.
- **Weber (2006), American Economic Review 96, 114–126:** laut Abstract (Suchtreffer, nicht im Volltext gelesen) lassen sich große Gruppen durch schrittweises Wachsen aus kleinen Gruppen koordinieren, wenn die Neuen die Geschichte der Gruppe kennen. H11 ist eine eigene Gegenprobe im Lernmodell (Neuling übernimmt die Erfahrung eines Mitglieds) und beansprucht keinen Nachbau des Experiments.
- **Van Huyck, Battalio und Beil (1990), American Economic Review 80, 234–248:** Abstract in der Websuche nicht erhalten, **nicht geprüft und nicht verwendet**. Die Parameter a = 0,2, b = 0,1, c = 0,6 stammen aus dem früheren Skript V4 des Auftraggebers und sind hier an der Quelle nicht geprüft.

Es steht **kein Laborergebnis** in dieser README; alle Zahlen sind Messungen des Lernmodells.

## Verifikation

- **Handrechnung:** Auszahlungen (0,9 / 0,7 für die Einsätze 3 und 5; 1,30 bei allen auf 7), Gleichverteilung n = 2 (140/49 = 2,857), δ* für n = 3, b/a = 0,25 (0,5), Logit-Schritt (p = 0,186 / 0,307 / 0,506), stationäres E[min] einer Zwei-Stufen-Kette (1,25), Replikator-Schritt (`tests/conftest.py`).
- **Orakel** (`tests/test_oracle_schwaechstes_glied.py`): Aufzählung aller Profile n ≤ 4 gegen die Gleichgewichte (genau sieben, strikt); E[min] durch Aufzählung gegen die Formel; Pfadaufzählung des Logit in reinem Python (Rekursion, `math.exp`) gegen die vektorisierte, beide gegen die Simulation (bis 300 000 Läufe, Marge vier Standardfehler); Kette der besten Antwort in Python aufgebaut gegen `br_chain`, stationär per Potenzieren, Simulation gegen die Kette; Replikator gegen eine Python-Schleife und u_x gegen die Aufzählung der Mitspieler; δ* gegen den Grenznutzen durch Aufzählung; zweiter Weg mit scipy (nur Test).
- **Eigenschaftstests** (`tests/test_eigenschaften_schwaechstes_glied.py`): Grundniveau c ändert nichts, Prämie und Preis mit demselben Faktor ändern nichts bei normierter Schärfe, Umbenennen der Partner, stochastische Dominanz, gleicher Seed in einem zweiten Prozess mit anderer Hash-Randomisierung.
- **Claims** (`tests/test_claims.py`): jede Zahl dieser README, aus den gespeicherten Messreihen oder exakt gerechnet; Messreihen mit Erzeuger-Hash, Stichprobe der Zellen mit anderem Seed reproduziert (`tests/test_measurements.py`).
- **Zweig- und Randfalltests:** jede Regel weicht auf mindestens einer Zelle von den anderen ab, keine konstante Reihe, n = 2 und 32, Runde 1, ein Lauf, Überlauf der Exponentialfunktion, Budget; Verdikte als reine Funktionen an handgesetzten Werten (`tests/test_verdicts.py`).
- **Oberfläche:** AppTest (jeder Knopf, jeder Regler am Minimum und Maximum, Eckfall-Matrix je Lernregel, jedes Preset, Permalink, PDF), Layout und Footer.
- **Plattform:** Tests prüfen Simulationen nur mit festen SplitMix64-Seeds und vier Standardfehlern Marge, Gleitkommaaggregate mit `approx`, Lösungen kleiner gut konditionierter linearer Systeme (n = 2, 3) auf 10^-9, die schlecht konditionierten Treffzeiten (n = 3, 4, ε = 0,01) nur als Band von ein bis zehn Prozent. Die Linux-CI ist hier nicht lokal nachgestellt (kein Docker).

## Tests

317 Tests (unter Windows, Python 3.14 im Test-venv). Zusammensetzung: Spiel und exakte Einheiten, Lernregeln, Replikator, Orakel, Eigenschaften, Claims, Messreihen, Verdikte, Randfälle, Presets, Layout, AppTest-Rauchtests, Visualisierung, Footer, Laufzeit-Importe, SplitMix64-Kopie. Laufzeit der vollen Suite rund eine Minute seriell (45 Sekunden mit `-n 8`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Hauptansicht, sechs Experimente, Grenzen, Formeln, PDF |
| `swg_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets mit Hilfetexten |
| `swg_game.py` | Auszahlung, Minimum der anderen, Gleichgewichte, beste Antwort, δ* |
| `swg_learning.py` | Lernregeln (Logit Summe/Mittel, beste Antwort, fiktives Spiel), Teams, wachsende Gruppen |
| `swg_exact.py` | exakte Rechnungen: E[min], Pfadaufzählung, Markov-Kette, Treffzeiten, erste Abweichung |
| `swg_replicator.py` | Replikator, Zufallsstarts, Einzugsbereich |
| `swg_scenario.py` | Einstellungen (nur mit Schlüsselwörtern), Schärfe, Budget |
| `swg_experiments.py` | Zellen und Reihen der Experimente, vorgerechnete Messreihen mit Erzeuger-Hash |
| `swg_evaluation.py` | Kennzahlen und Verdikte als reine Funktionen |
| `swg_visualization.py` | Plotly-Abbildungen (Achsen fest) |
| `swg_presets.py` | Permalink, Presets, Zufalls-Seed-Button |
| `swg_layout.py` | fester Kopf |
| `swg_pdf_export.py` | PDF-Export |
| `swg_rng.py` | SplitMix64 (wörtliche Kopie aus `ab-test-demo/abt_rng.py`) |
| `data/measurements.json` | vorgerechnete Messreihen |
| `tools/make_measurements.py` | erzeugt `data/measurements.json` neu (rund drei Minuten) |
| `tests/` | alle Tests (siehe oben) |

## Bewusst nicht umgesetzt

- **Cheap Talk und Signale als Strategie** (Vorabsignale gehören in ein eigenes Stück der Reihe).
- **Netzstruktur** (wer mit wem in der Kette steht) und mehrere sich überlappende Ketten.
- **Echte Laborparameter und Laborergebnisse:** die Demo bildet das Lehrbuchmodell nach.
- **Ungleiche Partner** (verschiedene Schärfe, Führungspersonen) und andere Neulingsregeln beim Wachsen.
- **Gleichgewichte in gemischten Strategien:** gezählt werden die reinen symmetrischen; andere sind nicht gesucht.

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit Cloud: Python-Version in der App-Einstellung wählen (getestet mit 3.12 in der CI).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
