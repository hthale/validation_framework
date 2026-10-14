# Testpaket 09 – Regeln und latenter Raum gemeinsam prüfen

Voraussetzung: vorhandenes Projekt mit **installiertem Korrekturpaket 08** und
funktionsfähiger virtueller Python-Umgebung. Dieses Testpaket verwendet den
vorhandenen Framework-Code. Es hat **keinen Installer** und ersetzt keine
Projektdateien. Die DOP-Tabelle wird in Kopien bearbeitet; Modelle und Ergebnisse
entstehen in einem neuen Zeitordner. Keine fachlichen Entscheidungen und keine
Einträge im Delivery Memory.

## 1. Ablegen

ZIP entpacken. Den Ordner `testpaket_09_gemeinsame_pruefung` direkt in deinen
Projektordner `validierungs-framework` legen. Dort muss dann die Datei
`testpaket_09_gemeinsame_pruefung\run_joint_dop_check.py` liegen. Alle folgenden
Befehle in PowerShell im Projektordner ausführen.

## 2. Den gemeinsamen DOP-Versuch starten

```powershell
.\.venv\Scripts\python.exe .\testpaket_09_gemeinsame_pruefung\run_joint_dop_check.py `
  --project . `
  --input .\data\check_01\dop_training_dataset_with_ids.parquet `
  --output .\data\dop_joint_check_09 `
  --epochs 25 `
  --seeds 42 43 44
```

Es laufen drei Wiederholungen mit jeweils 25 Trainingsepochen. Die letzte Ausgabe
nennt den Ergebnisordner und den Bericht. Für einen reinen Techniktest:
`--epochs 3 --seeds 42`. Neue Aufrufe erzeugen neue Ordner. Nach einem Abbruch
ist der betreffende Lauf unvollständig und kein auswertbares Ergebnis.

## 3. Ergebnisse nachprüfen und öffnen

```powershell
$testRun = Get-ChildItem .\data\dop_joint_check_09 -Directory |
  Sort-Object Name -Descending |
  Select-Object -First 1

.\.venv\Scripts\python.exe .\testpaket_09_gemeinsame_pruefung\verify_joint_run.py `
  --project . --run "$($testRun.FullName)"

if ($LASTEXITCODE -eq 0) {
  Start-Process (Join-Path $testRun.FullName "bericht.html")
}
```

Zur gemeinsamen Auswertung **`summary.json` aus diesem Ergebnisordner** senden.
Für konkrete Einzelfälle zusätzlich `seed_42/changes.json`,
`seed_42/case_results.csv` und `seed_42/rules.json`. Frühere Ergebnisordner werden
nicht wiederverwendet. Der Verifier schreibt keine Ergebnisdateien und lädt
keine Pickle-Dateien. Er benötigt denselben Projekt-/Testcode wie beim Lauf und
die unveränderte ursprüngliche Eingabedatei.

## Was miteinander verglichen wird

| Teil | Herkunft | Auswahl/Prüfung |
|---|---|---|
| Vier Konsistenzprüfungen | Explizite Testannahmen im Testpaket | Keine statistische Auswahl |
| Profiling-Regeln | Vorhandenes Framework; drei Stichproben ausschließlich im Training | Mindestens zwei Vorschläge; danach mindestens fünf Validierungsfälle und 95 % Passrate |
| Abhängigkeitsregeln | Automatische Zielauswahl und bestehende Gruppen-CV/Baumableitung im Training | Bestehende Modellgates: Accuracy 95 %, Macro-F1/Precision 90 %; Blattgate: mindestens fünf Fälle, 95 % Passrate |
| LOF auf Eingangsmerkmalen | Skalierte Zahlen/Datumswerte, One-Hot-Kategorien und Zeichenhäufigkeiten | Feste Nachbarzahl 20, Neuheitsmodus, 5 % Prüfbudget |
| LOF im latenten Raum | Derselbe Informationsumfang, durch trainierten Autoencoder dargestellt | Gleiche Nachbarzahl, Fälle, Distanz und gleiches Budget |

Die Gruppierung verwendet wie bisher CRS, East- und North-Ursprung. Pro Seed
gehen **60 % der Gruppen ins Training, 20 % in die Validierung und 20 % in den
Test**. Die Zeilenanteile können abweichen. Rollen, Imputation, Skalierung,
Autoencoder, HDBSCAN und LOF werden nur auf Training angepasst. Die Validierung
wählt Regeln aus; Testfälle und Änderungsmarkierungen gehen in keine Lern- oder
Auswahlfunktion ein. Alle ausgewählten Regeln werden unverändert auf die
Testkopie vor und nach den Änderungen angewendet.

Der Profiling-Aufruf verwendet Trainingscluster und LOF-Strata des latenten
Raums zur Stichprobenbildung. Abhängigkeitsbäume lernen auf Originalattributen.
Die gleiche Regelbasis gilt für beide Kombinationen mit LOF. Der Vergleich der
LOF-Varianten untersucht deshalb den Beitrag ihrer Repräsentation, keinen
vollständigen Ausbau aller latenten Komponenten aus dem Framework.

## Acht festgelegte Änderungen

Jeder Typ erhält **drei unterschiedliche Testzeilen**, insgesamt 24 Änderungen.
Pro Zeile ändert sich genau eine Zelle. Die Auswahl erfolgt reproduzierbar ohne
Scores. Reichen die Voraussetzungen nicht aus, wird der Lauf vor dem Training
abgebrochen; Klassen werden nicht still weggelassen.

| Änderung | Betroffenes Feld | Bedeutung |
|---|---|---|
| Bandzahl + 1 | `tif_band_count` | Widerspricht bei vorher passendem Fall der Testannahme RGB→3/RGBI→4 |
| Pixelgröße X × 10 | `tif_pixel_size_x` | Widerspricht bei vorher passendem Fall der cm/m-Konsistenz mit Metadaten |
| Anderer EPSG-Code | `tif_crs` | Widerspricht dem unveränderten Metadaten-CRS |
| Rasterbreite + 1 | `tif_width` | Widerspricht den unveränderten Metadatenspalten |
| Andere Trainingskamera | `kamera_sensor` | Kontrollierter Austausch, kein bestätigtes reales Fehlerlabel |
| Fehlende Kamera | `kamera_sensor` | Kontrolliertes Entfernen eines vorhandenen Werts |
| Datum + 180 Tage | `aktualitaet` | Kontrollierte Verschiebung, kein bestätigtes reales Fehlerlabel |
| XML-Größe + 3 Trainingsstandardabweichungen | `xml_size_bytes` | Positive Verschiebung, auf ganze Bytes aufgerundet |

Die vier Konsistenztests sind **Versuchsannahmen, keine amtlich bestätigte
Produktspezifikation**. Bodenpixelgröße wird für diesen Test als cm, Rasterpixel
als m interpretiert. Ist der Referenzwert unbekannt/fehlend, ist die jeweilige
Prüfung nicht anwendbar; fehlt ein Rasterzielwert bei gültiger Referenz, entsteht
ein Hinweis. Unbekannte Spektralprodukte werden nicht auf RGBI umgedeutet. Es
werden keine tatsächlichen TIFF-/XML-Dateien verändert oder erneut eingelesen.

Die ersten vier Klassen sind bewusst so gebaut, dass ihr Konsistenztest anschlägt.
Das belegt dessen Funktion, keine unabhängige Trefferquote gegenüber realen
Fehlern. Die übrigen Änderungen können fachlich zulässig sein. Deshalb heißt die
Kennzahl **Trefferquote der Änderungen**, nicht allgemeine Fehlererkennungsrate.

## Was der Bericht zeigt

- Je Methode: markierte Änderungen, vorab markierte Änderungsfälle, erst nach
  Änderung markierte Fälle und insgesamt markierte Zeilen.
- Alle acht Typen getrennt, damit einfache Konstantenverletzungen den Befund zu
  komplexeren Abhängigkeiten nicht verdecken.
- Zusätzliche Treffer des latenten LOF über alle Regeln hinaus und über Regeln
  plus Eingangs-LOF hinaus – jeweils mit zusätzlichem Prüfaufwand.
- Alle Profiling-/ML-Kandidaten, ihre Validierungsbelegung, Passrate und Auswahl.
- Paarweise Überschneidungen und vom LOF-Vertrag verwendete/verworfene Felder.
- Änderungen, die wegen konstanter ausgeschlossener Felder oder Vorverarbeitung
  im LOF-Eingang unsichtbar bleiben. Sie werden im Nenner behalten.

Regeln markieren alle Verletzungen ihrer Bedingungen. LOF hat hingegen ein
Budget von `ceil(Testzeilen × 0.05)`. Die Vereinigung kann mehr als dieses Budget
beanspruchen: Mehr Treffer allein belegen daher keinen Vorteil bei gleichem
Arbeitsaufwand. „Erst nach Änderung“ bei einer Rangliste kann auch durch andere
Änderungen im gemeinsamen Budget beeinflusst werden. Einzelne neue
Regelverletzungen stehen zusätzlich in `new_rule_violations` der Falltabelle.

Keine automatische fachliche Freigabe: Auch 0 ausgewählte Abhängigkeiten ist
ein gültiger Befund. Der Bericht senkt keine Schwellen und setzt keinen Vorteil
des Autoencoders voraus. Unveränderte reale Daten sind nicht garantiert sauber;
ihre Auswahl ist kein nachgewiesener Fehlalarm. Die Seeds teilen sich Daten,
sind also keine unabhängigen Studien. Koordinatengruppen garantieren keine
räumliche/zeitliche Unabhängigkeit. Da der DOP-Bestand bereits untersucht wurde,
ist dies eine Entwicklungsprüfung, kein externer unberührter Abschlusstest.

Die neue 60/20/20-Aufteilung, Typen und Nenner unterscheiden sich von Paket 08.
Die Prozentwerte beider Pakete dürfen nicht unmittelbar als Fortschritt oder
Rückschritt verglichen werden. Es werden weder neue Clusterbeschreibungen als
Qualitätsregeln eingesetzt noch NAS-/TIFF-/XML-Einlesung oder ALKIS-Geometrien
getestet.

## Optional: positive und negative synthetische Kontrolle

Dieser zusätzliche Techniktest zeigt, dass der gemeinsame Ablauf auch
Abhängigkeitsregeln finden und mit Gegenbeispielen prüfen kann. Er ist für den
DOP-Versuch nicht erforderlich; die Daten sind bewusst einfach konstruiert.

```powershell
.\.venv\Scripts\python.exe .\testpaket_09_gemeinsame_pruefung\make_control_fixture.py

.\.venv\Scripts\python.exe .\testpaket_09_gemeinsame_pruefung\run_joint_dop_check.py `
  --project . --input .\data\dop_control_fixture_09.parquet `
  --output .\data\control_joint_check_09 --epochs 3 --seeds 42

$kontrolle = Get-ChildItem .\data\control_joint_check_09 -Directory |
  Sort-Object Name -Descending | Select-Object -First 1

.\.venv\Scripts\python.exe .\testpaket_09_gemeinsame_pruefung\verify_joint_run.py `
  --project . --run "$($kontrolle.FullName)" --control-fixture
```

Der Generator überschreibt eine vorhandene Datei nicht. Er erzeugt 400 Zeilen
mit 200 Gruppen. Die Referenzen für Bandzahl, Kamera und Pixelgröße sind
vorgegeben; `kontrolle` ist bei identischen Prädiktoren paarweise ausgeglichen.
Die zusätzliche Verifikation verlangt automatisch ausgewählte Regeln für die
drei Referenzen, korrekte Behandlung ihrer vier Kombinationen und Erkennung
der Gegenbeispiele. Für `kontrolle` darf keine Abhängigkeit ausgewählt werden.
Dieser Erfolg sagt nichts über den Nutzen des Autoencoders auf realen DOP aus.

## Dateien und lokale Paketprüfung

| Datei im Lauf | Inhalt |
|---|---|
| `bericht.html`, `summary.json` | Bericht und Kennzahlen über alle Seeds |
| `protocol.json`, `run_manifest.json` | Festlegungen, Softwareversionen, Code-/Artefaktprüfsummen |
| `input_with_ids.parquet` | Unveränderte Eingabekopie mit stabilen Kennungen |
| `seed_*/split.csv` | Gruppen und drei Teilmengen |
| `seed_*/test_before.parquet`, `test_after.parquet`, `changes.json` | Rohwert-Testkopien und sämtliche Änderungen |
| `seed_*/rules.json` | Erzeugte/ausgewählte Regeln, Ableitungs- und Validierungsbelege |
| `seed_*/case_results.csv`, `case_results.parquet` | Pro Fall alle Methoden und Kombinationen |
| `seed_*/rule_cases.parquet`, `scores.parquet` | Regelanwendbarkeit/-verletzung sowie LOF-Scores und Ränge vor/nach Änderung |
| `seed_*/representations.npz` | Eingangs- und latente Matrizen; Nachrechnen des LOF ohne erneutes Training |
| `seed_*/feature_contract.json`, `training_history.json` | Merkmalsvertrag und Training |
| `seed_*/autoencoder.pt`, `*.pkl` | Tatsächlich trainierte Modelle/Vorverarbeitung |

Gezielte neue Tests (die vorhandenen Projekttests werden nicht ersetzt):

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\testpaket_09_gemeinsame_pruefung\tests -v
```

Mitgelieferte `verification/`-Dateien dokumentieren Prüfungen in der lokalen
Entwicklungsumgebung des Pakets, nicht einen Lauf auf deinem Windows-Rechner.
Der Browser wurde nicht automatisiert bedient. Die Ergebnisdateien aus deinem
Lauf sind die Grundlage für unsere nächste Auswertung.
