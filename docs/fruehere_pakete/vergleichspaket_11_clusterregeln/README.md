# Paket 11: Trägt der latente Raum zur Regelerzeugung bei?

Dieses Zusatzpaket vergleicht automatisch erzeugte Attributregeln in drei Varianten:

| Variante | Wie entstehen die Regeln? |
|---|---|
| Global | Ein Attributbaum pro automatisch erkanntem Regelziel |
| Attributcluster | HDBSCAN gruppiert die aufbereiteten Attribute; pro Gruppe werden Regeln gelernt |
| Latentcluster | Der vorhandene Autoencoder erzeugt eine Darstellung derselben Attribute; HDBSCAN und lokale Regelsuche folgen |

Die Auswahl erfolgt auf Validierungsdaten. Danach werden alle ausgewählten Regeln
auf zurückgehaltenen Testdaten geprüft. Der Test löscht keine ungünstige Regel und
wird nicht zum Nachtrainieren verwendet. Der Bericht zeigt Abdeckung UND
Übereinstimmung. Mehr Regeln oder mehr Cluster allein sind kein Erfolgsnachweis.

**Keine Geometrien. Kein Installer. Keine Ersetzung bestehender src-Dateien.**
Das Paket benötigt das bestehende Framework mit Paket 07/08, das daneben liegende
`analysepaket_10_bahn_attribute` und die bereits verwendete Python-Umgebung.

## 1. Entpacken

Die ZIP enthält einen Ordner `vergleichspaket_11_clusterregeln`. Diesen im
Projektordner ablegen. Anschließend müssen diese beiden Pfade vorhanden sein:

- `validierungs-framework/vergleichspaket_11_clusterregeln/run_rule_comparison.py`
- `validierungs-framework/analysepaket_10_bahn_attribute/attribute_prep.py`

Die unveränderte Originaldatei bleibt unter `data/parquetdata/bahnverkehr.parquet`.
Falls sie bei dir anders liegt, nur den Eingabepfad im Befehl anpassen.

## 2. Vergleich starten

PowerShell im Ordner `validierungs-framework` öffnen und ausführen:

```powershell
.\.venv\Scripts\python.exe .\vergleichspaket_11_clusterregeln\run_rule_comparison.py `
  --project . `
  --input ".\data\parquetdata\bahnverkehr.parquet" `
  --output ".\data\regelvergleich_11" `
  --rows 5000 `
  --epochs 5 `
  --seeds 42 43 44
```

Die Stichprobe ist dieselbe deterministische 5.000er-Auswahl wie in Paket 10. Die
Seeds erzeugen drei verschiedene Aufteilungen und Initialisierungen. Für jedes
automatisch erkannte Ziel wird ein eigener Autoencoder trainiert: Das Ziel darf
die Clusterbildung nicht schon verraten. Deshalb dauert dieser Vergleich länger
als ein einzelner Modelllauf. Fortschrittszeilen erscheinen je Seed und Ziel;
die einzelnen Epochen stehen unter `seed_*/target_*/training.log`.

## 3. Bericht öffnen

```powershell
$vergleich = Get-ChildItem .\data\regelvergleich_11 -Directory |
  Where-Object {
    $status = Join-Path $_.FullName "run_status.json"
    (Test-Path $status) -and ((Get-Content $status -Raw | ConvertFrom-Json).status -eq "completed")
  } |
  Sort-Object Name -Descending |
  Select-Object -First 1
if ($null -eq $vergleich) { throw "Kein abgeschlossener Vergleich gefunden." }
Start-Process (Join-Path $vergleich.FullName "bericht.html")
```

Für die anschließende gemeinsame Auswertung diese drei Dateien hochladen:

- `summary.json`: Vergleich der drei Methoden, je Seed;
- `target_metrics.json`: Abdeckung und Übereinstimmung je Ziel;
- `test_rule_evidence.json`: Testbelege für jede ausgewählte Regel.

Auch das Ergebnis „kein Vorteil nachweisbar“ ist ein gültiges Untersuchungsergebnis.
Vor der Auswertung keine Parameter ändern und nicht einen günstigen Seed auswählen.

## Was der Code macht

- `run_rule_comparison.py`: verwendet den Attributadapter aus Paket 10, erzeugt
  stabile Kennungen und Gruppenaufteilungen, trainiert die Modelle und friert die
  vollständige Auswahl vor der Testauswertung ein.
- `comparison_core.py`: bildet gemeinsame Eingabematrizen, schließt je Ziel dessen
  eigene Werte/direkte Ableitungen aus, führt Clusterzuordnung und Regelsuche aus
  und berechnet die Kennzahlen. Der Baumaufbau, Bedingungsexport und Regelprüfer
  stammen aus dem vorhandenen Framework.
- `report_comparison.py`: schreibt den offline nutzbaren HTML-Bericht.
- `verify_rule_comparison.py`: prüft Aufteilungen, gespeicherte Auswahl und
  konkrete Validierungs-/Testbelege erneut, ohne Modelle neu zu trainieren.
- `tests/test_comparison.py`: prüft bekannte Abhängigkeiten, die Ablehnung einer
  uninformativen Kontrollgröße, fehlende Zielwerte, unveränderliche Vorverarbeitung,
  Cluster-Reichweite, Rauschpunkte und Ausschluss von Zielinformationen.

## Bewusste Unterschiede zum ersten Bahnverkehr-Lauf

Das bisherige Programm und seine Modelle bleiben unverändert. Für diesen neuen
Versuch ist die Eingabe der drei Varianten vereinheitlicht: One-Hot-Kategorien und
numerisches Prüfdatum, ohne Zeichenencoder für Fachcodes. Der vorhandene Autoencoder
arbeitet als tabellarische Variante mit acht latenten Dimensionen. Fünf Epochen sind
ein festgelegter erster Vergleich, keine optimierte Konfiguration.

HDBSCAN verwendet für beide Darstellungen dieselben Parameter (50/10). Regeln
werden mit festen Bäumen (Tiefe 4, mindestens 5 Zeilen pro Blatt) gelernt. Es gibt
keine Parametersuche. Modellschwellen bleiben bei Accuracy 0,95, Macro-F1 0,90 und
Macro-Precision 0,90; einzelne Regeln benötigen mindestens 95 Prozent Bestätigung
sowie mindestens fünf Fälle aus fünf unterschiedlichen Gruppen. Ein konstanter
Zielwert innerhalb eines Clusters darf ein Kandidat sein, benötigt aber dieselben
Belege. `PROTOKOLL.md` beschreibt sämtliche Festlegungen.

Der Test vergleicht deshalb diese kontrollierte neue Konfiguration. Er belegt
nicht nachträglich die Güte der früheren 64-dimensionalen Darstellung aus Paket 10.
Lokale Regeln werden als **Regel plus obligatorische Clusterbedingung** gespeichert.
Die JSON-Umschläge sind absichtlich kein Importformat der bisherigen Prüfansicht:
Ein isolierter IF/THEN-Teil würde den lokalen Geltungsbereich verlieren.

## Aussagegrenzen

- Bestätigt wird Übereinstimmung mit vorhandenen Attributwerten, keine fachliche
  Richtigkeit und keine reale Fehlererkennungsrate. Es werden keine Fehler eingebaut.
- Fehlende optionale Angaben werden separat gezählt und nicht als Fachfehler benannt.
- Exakte Attributduplikate bleiben zusammen; das beweist keine räumliche oder
  lieferungsbezogene Unabhängigkeit. Der bereits betrachtete Bestand ist keine
  neue externe Testlieferung.
- HDBSCAN kann für neue Fälle keine Gruppe finden. Diese Fälle bleiben in der
  jeweiligen lokalen Variante ohne Regel. Das wird als fehlende Abdeckung sichtbar.
- Kleine Gruppen und seltene Ausprägungen können die Mindestbelege verfehlen.
  Drei Aufteilungen desselben Bestands sind keine unabhängigen Stichproben.
- Geometrie, Koordinaten, Kennungen, Scores und latente Koordinaten dienen nicht
  als Attributprädiktoren der Regelbäume. Die Gruppenbildung der latenten Variante
  verwendet die latente Darstellung der erlaubten Prädiktoren.
- Die Vorverarbeitung ist für Bahnverkehr explizit festgelegt. Sie behauptet kein
  automatisches semantisches Verständnis beliebiger Produkte.

## Dateien und Wiederholungsprüfung

Jeder Aufruf erzeugt einen neuen Ordner unter `data/regelvergleich_11`. Er enthält
den Bericht, Regeln, Kennzahlen, vollständige Testfälle, Protokoll und Prüfsummen.
Unter jedem Seed liegen die Aufteilung und je Ziel die trainierten Modelle,
Vorverarbeitung, Matrizen und Clusterzuordnungen. Original und Projektquellcode
werden am Ende nochmals auf unveränderte Prüfsummen geprüft. Kein Delivery Memory
wird geschrieben. Modellartefakte nur aus vertrauenswürdigen eigenen Läufen laden.

```powershell
.\.venv\Scripts\python.exe .\vergleichspaket_11_clusterregeln\verify_rule_comparison.py `
  --project . --run "$($vergleich.FullName)"
```

Diese Prüfung benötigt dieselben Quell-/Originaldateien wie beim Lauf.

Optionale isolierte Prüfungen, ohne einen Vergleichslauf zu starten:

```powershell
.\.venv\Scripts\python.exe -m unittest discover `
  -s .\vergleichspaket_11_clusterregeln\tests -v
```

## Technische Quellen

- HDBSCAN: Vorhersage der Zugehörigkeit zu vorhandenen Trainingsclustern,
  <https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html> und
  <https://hdbscan.readthedocs.io/en/latest/api.html#hdbscan.prediction.approximate_predict>.
- Scikit-learn: Trennung von fit/transform und Vermeidung von Datenleckage,
  <https://scikit-learn.org/stable/common_pitfalls.html#data-leakage>.

Die lokalen Testergebnisse und ihre Grenzen stehen in `PROBELAUF.md`.
