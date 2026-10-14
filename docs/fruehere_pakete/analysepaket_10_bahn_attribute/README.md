# Bahnverkehr ausschließlich über Attribute untersuchen

Dieses Zusatzpaket bereitet die gelieferte `bahnverkehr.parquet` für den vorhandenen
generischen Framework-Lauf vor. Es ersetzt keine Dateien unter `src`, benötigt
keinen Installer und keine zusätzlichen Geometriebibliotheken. Voraussetzung sind
das bisher funktionierende Framework mit Paket 07/08 und dessen Python-Umgebung.

## Ablage und Start

Den ZIP-Inhalt so entpacken, dass folgender Pfad existiert:

`validierungs-framework/analysepaket_10_bahn_attribute/start_bahnverkehr.py`

Die Originaldatei hier ablegen:

`validierungs-framework/data/parquetdata/bahnverkehr.parquet`

PowerShell im Ordner `validierungs-framework` öffnen:

```powershell
.\.venv\Scripts\python.exe .\analysepaket_10_bahn_attribute\start_bahnverkehr.py `
  --project . `
  --input ".\data\parquetdata\bahnverkehr.parquet" `
  --output ".\data\bahnverkehr_attribute_10" `
  --rows 5000 `
  --epochs 5
```

5.000 Objekte und fünf Epochen sind der festgelegte erste Probelauf. Die Stichprobe
wird ausschließlich aus `gml_id` und Seed 42 ausgewählt. Sie ist unabhängig von
Zeilenreihenfolge, Auffälligkeiten und Attributwerten; seltene Kategorien können
fehlen. Genau das wird im Bericht dokumentiert. Es werden keine Fehler eingebaut.

Nach erfolgreichem Abschluss den neuesten vollständig abgeschlossenen Lauf öffnen:

```powershell
$bahnRun = Get-ChildItem .\data\bahnverkehr_attribute_10 -Directory |
  Where-Object {
    $status = Join-Path $_.FullName "run_status.json"
    (Test-Path $status) -and ((Get-Content $status -Raw | ConvertFrom-Json).status -eq "completed")
  } |
  Sort-Object Name -Descending |
  Select-Object -First 1
if ($null -eq $bahnRun) { throw "Kein abgeschlossener Bahnverkehr-Lauf gefunden." }
Start-Process (Join-Path $bahnRun.FullName "bericht.html")
```

Anschließend diese drei Dateien aus dem Ergebnisordner auswerten oder hochladen:

- `summary.json`: Anzahl Regeln, Cluster und technisch geprüfter Fälle;
- `preparation.json`: ausgewählte Attribute, fehlende Werte und Stichprobenabdeckung;
- `analyse/<Zeitstempel>/cluster_reports.json`: Wie gut Attributbäume die Cluster beschreiben.

## Was genau geschieht?

1. Das Programm liest ausschließlich die unten aufgeführten Attribute und `gml_id`.
   Geometrie- und Koordinatenspalten werden nicht eingelesen oder interpretiert.
   Die Dateiprüfsumme sichert die Originaldatei; sie ist kein Modellmerkmal.
2. Vorhandene Fachcodes werden als Texte mit dem Präfix `code:` gespeichert.
   Beispiel: `2322` wird zu `code:"2322"`. Führende Nullen bleiben erhalten.
   Der Präfix verhindert, dass die automatische Rollenheuristik Codes als metrische
   Größen behandelt. Es werden keine Wertearten vorgeschrieben oder ergänzt.
3. Codelisten werden vollständig als kategoriale Kombination gespeichert, z. B.
   `liste:["1100"]`. Reihenfolge und Wiederholungen bleiben erhalten. Fehlende Listen
   bleiben fehlend, leere Listen bleiben als `liste:[]` erkennbar. Verschachtelte oder
   fehlerhaft typisierte Listen lösen einen Fehler aus. Erkennt der bestehende
   Code wegen mehr als 20 Kombinationen eine falsche Rolle, wird vor Training abgebrochen.
4. Der Prüfzeitpunkt wird als Datum erhalten. Ein zusätzliches Merkmal erfasst seine
   Belegung, damit eine spätere Medianimputation diese Information nicht verdeckt.
   Die ausgewählten unveränderten Quellattribute werden separat gespeichert.
5. Der vorhandene `src/run_generic.py` wird als eigener Prozess aufgerufen. Er
   trainiert den Autoencoder, berechnet LOF/HDBSCAN/UMAP, erzeugt globale
   Abhängigkeitskandidaten und Profiling-Vorschläge und beschreibt die Cluster.
6. Die Ergebnisse werden automatisch auf eindeutige Identitäten, unerlaubte
   Merkmale, gültige Rollen, endliche Werte, Sortierung, Aufteilung und unveränderte
   Quell-/Projektdateien geprüft. Modelle werden wirklich trainiert; keine Attrappen.

## Bewusste Auswahl der Eingabeattribute

| Attribute | Rolle |
|---|---|
| `ergebnisderueberpruefung`, `istweiterenutzung`, `funktion`, `zustand` | Fachcodes als Kategorien |
| `anlass`, `advstandardmodell`, `sonstigesmodell`, `processstep_ax_datenerhebung`, `bahnkategorie` | Vollständige Codelisten als kategoriale Kombinationen |
| `datumderletztenueberpruefung` | Datumsmerkmal |
| `datumderletztenueberpruefung_fehlt` | Abgeleitete Belegungsinformation |
| `zweitname`, `unverschluesselt` | Vorhandene Texte; Rollenheuristik des Frameworks |
| `gml_id` | Ausschließlich Identität, nie Prädiktor oder Regelziel |

Nicht vorhandene optionale Attribute werden protokolliert. Vollständig leere und
konstante Spalten liefern keine variierenden Lernmerkmale. Die Liste ist eine
offengelegte produktspezifische Aufbereitung für diesen Versuch, keine Behauptung,
dass das Framework die Semantik unbekannter Produkte automatisch versteht.

`geometry`, `wkb_geometry`, Koordinaten, Geometriemerkmale, Bahnstreckennummern,
Fremdkennungen, externe Referenzen sowie technische Beginn-/Endzeitstempel gelangen
nicht ins Lernen. Neue, bisher unbekannte Spalten werden ebenfalls nicht still
übernommen. Es werden weder Fläche noch Umfang, Form, Topologie oder räumliche
Beziehungen untersucht. Anlass und Prüfzeitpunkt können administrative Abläufe
abbilden; ihre Beteiligung an Clustern muss entsprechend interpretiert werden.

## Aussagegrenzen dieses Schritts

- Autoencoder, LOF, Profiling und Clusterbeschreibungen beschreiben die gewählte
  Stichprobe. Fünf Epochen sind ein technischer/inhaltlicher Einstieg, keine optimierte Modellwahl.
- Das Modell behält die vorhandenen 64 latenten Dimensionen und Trainingsziele.
  Dies wird hier nicht als nachgewiesene Dimensionsreduktion bezeichnet.
- Kategorien gehen nach dem bestehenden Verfahren auch in den Zeichenencoder ein;
  daraus folgt kein semantisches Verständnis der ALKIS-Codes.
- Für globale Abhängigkeiten werden identische aufbereitete Attributkombinationen
  zusammengehalten und Gruppen auf Training, Validierung und reservierten Test
  aufgeteilt. Dies belegt keine räumliche/zeitliche Unabhängigkeit. Fachliche
  Vergleichseinheiten sind vor einer späteren Leistungsbewertung festzulegen.
- Der Testanteil wird nicht zur Auswahl globaler Abhängigkeitsregeln verwendet.
  Er ist **kein unberührter Testbestand für den zuvor trainierten Autoencoder**.
- Clusterbeschreibungen sind Beschreibungen der gefundenen Gruppen. Ihre
  Anpassungsgüte und Anzahl belegen weder korrekte Qualitätsregeln noch einen
  Vorteil gegenüber einer Darstellung ohne Autoencoder.
- Keine automatische Fachfreigabe, keine echte Fehlerquote, keine Fehlerinjektion,
  keine Änderung von Originaldaten oder Delivery Memory.
- Das Paket implementiert **noch keine neuen Abhängigkeitsregeln innerhalb von
  Clustern und keinen Vergleich mit/ohne latenten Raum**. Nach Sichtung des ersten
  Attributlaufs sind diese die nächsten Untersuchungen. Güteschwellen werden
  aufgrund dieses Probelaufs nicht einfach abgesenkt.

## Ergebnisstruktur

Unter `data/bahnverkehr_attribute_10/<Zeitstempel>/` liegen:

- `bericht.html`: Einstieg mit Herkunft der Ergebnisse und Links;
- `summary.json`, `preparation.json`, `verification.json`;
- `source_attributes.parquet`: ausgewählte Originalattribute der Stichprobe;
- `attributes.parquet`: aufbereitete Lern-/Prüftabelle mit gml_id;
- `framework.log`: vollständiges Protokoll des Framework-Prozesses;
- `analyse/<Zeitstempel>/`: alle regulären Framework-Artefakte;
- `run_manifest.json`: Prüfsummen und Softwareversionen;
- `run_status.json`: `running`, `completed` oder `failed`.

Jeder Aufruf erzeugt einen neuen Ergebnisordner. Er überschreibt keinen alten Lauf.
Die Originaldatei und `src` werden ausschließlich gelesen. Die hochgeladene
Original-Parquetdatei wird nicht im Downloadpaket dupliziert.

## Spätere Wiederholungsprüfung ohne Training

```powershell
.\.venv\Scripts\python.exe .\analysepaket_10_bahn_attribute\verify_bahn_run.py `
  --project . --run "$($bahnRun.FullName)"
```

Die Prüfung verlangt denselben Projekt-/Paketcode und dieselbe Originaldatei wie
beim Lauf. Nach späteren Updates können die Code-Prüfsummen absichtlich abweichen.

Tests für den Adapter:

```powershell
.\.venv\Scripts\python.exe -m unittest discover `
  -s .\analysepaket_10_bahn_attribute\tests -v
```

Für einen späteren vollständigen Bestandslauf sind `--rows 0 --epochs 25` möglich.
Der vorhandene Autoencoder wertet am Ende alle ausgewählten Zeilen zugleich aus;
ein Lauf mit 47.245 Zeilen kann viel Arbeitsspeicher benötigen. Erst den hier
beschriebenen 5.000er-Lauf abschließen und beurteilen.
