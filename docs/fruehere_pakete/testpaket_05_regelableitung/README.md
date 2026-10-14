# Testpaket 05: automatische Regelableitung kontrolliert prüfen

Dieses Zusatzpaket erzeugt einen künstlichen Referenzbestand und führt damit den
vorhandenen Profiling- und Entscheidungsbaum-Code Frameworks aus. Die
konkreten Regeln werden aus den Daten gelernt. Die vorab bekannten Zusammenhänge
werden nicht als fertige Regeln an die Lernkomponente übergeben.

## Start unter Windows

1. ZIP entpacken. Der Ordner `testpaket_05_regelableitung` muss direkt im
   vorhandenen Projektordner `validierungs-framework` liegen, neben `src`,
   `tests` und `data`.
2. Im Projektordner das Terminal mit aktivierter `.venv` öffnen.
3. Ausführen:

```powershell
python .\testpaket_05_regelableitung\run_synthetic_rule_check.py --project .
```

Es wird kein Installationsskript benötigt. Das Paket importiert den vorhandenen
Code aus `src`. Voraussetzung sind die bereits installierten Korrekturpakete
01–04 und deren Python-Abhängigkeiten einschließlich eines Parquet-Readers.
Es werden keine echten DOP-/ALKIS-Dateien gelesen und keine Projektdateien,
Konfigurationen, bisherigen Prüfläufe oder fachlichen Bewertungen überschrieben.

Die Ergebnisse liegen in einem neuen Unterordner:

```text
data/synthetic_rule_check_05/<Zeitstempel>/
```

Der genaue Pfad wird im Terminal ausgegeben. Den Bericht des neuesten Testlaufs
öffnest du anschließend so:

```powershell
$testRun = Get-ChildItem .\data\synthetic_rule_check_05 -Directory | Sort-Object Name -Descending | Select-Object -First 1
Start-Process (Join-Path $testRun.FullName "bericht.html")
```

Optional können die sechs gezielten technischen Tests ausgeführt werden:

```powershell
python .\testpaket_05_regelableitung\tests\test_synthetic_rule_check.py --project .
```

## Was erzeugt wird

600 synthetische Datensätze gehören zu 300 künstlichen Untersuchungseinheiten.
Jede Einheit hat zwei Beobachtungen; beide bleiben in derselben Teilmenge.
Der Standardlauf mit Seed 42 verwendet 360 Trainings-, 120 Validierungs- und
120 Testdatensätze. Die festen IDs hängen nur von Einheit und Beobachtung ab.

Die folgenden Zuordnungen sind **Annahmen des Testfalls, keine verbindlichen
DOP- oder ALKIS-Produktregeln**:

| Kanaltyp | Detailstufe | Bandanzahl | Pixelgröße in cm | Exportprofil |
|---|---|---:|---:|---|
| RGB | standard | 3 | 20 | P_A |
| RGB | fein | 3 | 10 | P_B |
| RGBI | standard | 4 | 20 | P_C |
| RGBI | fein | 4 | 10 | P_D |

Zusätzlich gilt im künstlichen Bestand `crs_code = 25832`.
Die vier Exportprofile sind frei erfunden. Zwei Zufallsmerkmale sind mögliche,
aber für die drei Zielgrößen irrelevante Prädiktoren.

Die negative Kontrollgröße `kontrollwert` nimmt bei identischen Prädiktoren pro
Einheit einmal K_A und einmal K_B an. Damit ist sie absichtlich nicht
vorhersagbar. Ein Lernverfahren darf Baumkandidaten dazu erzeugen; diese sollen
die Qualitätsauswahl nicht bestehen.

Die Ziel- und Merkmalsrollen sind in diesem kleinen Versuch vorab festgelegt.
Geprüft wird die automatische Ableitung der konkreten Regeln, nicht die
automatische Entscheidung, welche Spalten für jedes beliebige Produkt relevant
sind. Die Wahrheitsetiketten bleiben in einer separaten Datei.

## Ablauf und methodische Trennung

1. Daten und Referenzzuordnungen mit festem Seed erzeugen; Gruppen aufteilen.
2. Profiling-Regeln ausschließlich aus dem Trainingsanteil ableiten. Für diesen
   Funktionstest wird ein vollständiges Trainingsprofil verwendet, nicht das
   stratifizierte Bestandsprofiling des bisherigen DOP-Laufs.
3. Je Zielgröße den vorhandenen `train_dependency`-Code verwenden: Parameterwahl
   durch dreifache Gruppen-Cross-Validation im Training, anschließend separate
   Validierung. Alle Modelle erhalten dieselben vier vorher festgelegten
   Prädiktoren; andere Zielgrößen, IDs und Gruppenkennungen sind ausgeschlossen.
4. Bestehende Auswahlbedingungen beibehalten: Accuracy mindestens 0,95,
   Macro-Precision und Macro-F1 mindestens 0,90; je Regel mindestens fünf
   anwendbare Validierungsfälle und mindestens 0,95 Übereinstimmung.
5. Regeln und Trainingsberichte speichern und per Prüfsumme einfrieren.
6. Erst jetzt eine Kopie der reservierten Testdaten verändern. Keine erneute
   Anpassung, Parameterauswahl oder Schwellenänderung nach Testauswertung.
7. Eingefrorene Regeln auf unveränderte Testdaten und Fehlerkopie anwenden.

Die 120 Datensätze der Testkopie enthalten:

- Je acht Fehler in Bandanzahl, Pixelgröße und Exportprofil. Die falschen Werte
  bleiben jeweils innerhalb der bekannten Wertemenge; nur die Zuordnung stimmt
  nicht mehr. Eine einfache Wertemengenprüfung reicht hier nicht aus.
- Acht Verletzungen der CRS-Konstanz und acht fehlende Bandanzahlen.
- Acht gültige Produktwechsel mit gemeinsam angepasster Bandanzahl und passendem
  Exportprofil sowie acht gültige Änderungen irrelevanter Zufallsmerkmale.
- 64 unveränderte Datensätze.

Das ergibt 40 absichtliche Fehler und 80 gültige Fälle in der Testkopie.
Die Zuordnung zu Falltypen erfolgt vorab mit festem Seed, unabhängig von
Regelmeldungen. Jede Fehlerzeile enthält genau einen injizierten Fehler.

Unveränderte Testdaten und Testkopie bilden einen **gepaarten Vergleich** mit
denselben Kennungen, keine zwei unabhängigen Teststichproben. Gezählt werden
betroffene Datensätze; mehrere Regelmeldungen pro Datensatz ergeben nicht
mehrere erkannte Fehler.

Die Profiling-Vorschläge werden im Testpaket vollständig als gelernte
Bestandserwartungen untersucht. Die Modell- und Validierungsschwellen gelten
für die ML-Regeln. Das Feld `test_evaluated: false` im ursprünglichen
Trainingsbericht beschreibt die Auswahlphase; die anschließende Testauswertung
steht separat in `summary.json` und `rule_evaluations.json`.

## Mitgelieferter Referenzlauf

Mit dem vorhandenen Framework-Code wurden 12 Profiling-Vorschläge und acht
ML-Regeln mit bestandenen Auswahlbedingungen erzeugt. Alle drei vorgegebenen
Abhängigkeiten wurden wiedergefunden. Die negative Kontrollgröße wurde
zurückgehalten. Auf den 120 unveränderten Testdatensätzen gab es keine Meldung.

| Regelquelle | Erkannte Fehler in der Testkopie | Fehlalarme unter den 80 gültigen Fällen |
|---|---:|---:|
| Profiling | 16 von 40 | 0 |
| ML-Regeln | 32 von 40 | 0 |
| Beide gemeinsam | 40 von 40 | 0 |

Die beiden Quellen überschneiden sich bei fehlenden Bandanzahlen. Ihre
Trefferzahlen werden deshalb nicht addiert. Die sechs gezielten technischen
Tests haben bestanden; Softwareversionen stehen im Referenzlauf. Unterschiedlich
formulierte, gleich wirkende Baumpfade sind zwischen Bibliotheksversionen möglich.

## Ergebnisdateien

| Datei | Inhalt |
|---|---|
| `bericht.html` | Lesbarer Bericht mit Referenz, gelernten Regeln und Fehlererkennung |
| `summary.json` | Bestehen der Prüfkriterien, Regelzahlen, Treffer und Fehlalarme |
| `synthetic_reference.parquet` | Vollständiger künstlicher Referenzbestand |
| `train.parquet`, `validation.parquet` | Tatsächlich verwendete Lern- und Validierungsdaten |
| `test_clean.parquet`, `test_mixed.parquet` | Reservierte Testdaten und gezielt veränderte Kopie |
| `split_manifest.csv` | Feste Kennungen, Gruppen und Teilmengen |
| `reference_spec.json` | Vorab festgelegte Zusammenhänge, Merkmalsrollen und Versuchsparameter |
| `learned_rules.json` | Automatisch erzeugte Profiling- und ML-Regeln, einschließlich zurückgehaltener Kandidaten |
| `training_report.json` | Modellauswahl, Validierungskennzahlen und Auswahlbedingungen |
| `ground_truth.json` | Erwarteter Fehlerstatus je Testdatensatz; nie ML-Eingabe |
| `test_changes.json` | Kennung, Feld, alter/neuer Wert und Änderungstyp |
| `case_results.json`, `rule_evaluations.json` | Meldungen je Datensatz sowie Kennzahlen je Regel |
| `run_manifest.json` | Code- und Dateiprüfsummen, Parameter und Softwareversionen |
| `run_status.json` | Lauf abgeschlossen oder noch unvollständig |

Der Exitcode ist 0 bei bestandenem Funktionstest, 1 bei nicht bestandenen
Prüfkriterien und 2 bei einem abgefangenen Aufruf-/Eingabefehler. Auch bei einem
abgeschlossenen, nicht bestandenen Funktionstest wird ein Bericht geschrieben.

## Was dieser Versuch nicht belegt

Ein gutes Ergebnis ist bei diesen einfachen, vollständig bekannten
Zusammenhängen erwartbar. Es bestätigt die Funktion unter diesen Bedingungen,
nicht die Qualität jeder später automatisch erzeugten Regel. Reale Lieferungen
können seltene gültige Kombinationen, unbekannte Werte, fehlende Merkmale und
Lieferungsunterschiede aufweisen. Eine zufällige Kontrollgröße prüft keine
allgemeine Rate zufällig gefundener Regeln.

Autoencoder, LOF, HDBSCAN, UMAP, XML-/Raster-/Vektoreinlesung, Geometrieprüfungen
und fachliche Freigabe sind nicht Gegenstand dieses Testpakets. Die erzeugten
Regeln behalten den Status `proposed`.

## Reihenfolge für die anschließende Untersuchung

1. Diesen künstlichen Referenzversuch ausführen und Regeln sowie Soll-Ist-
   Vergleich ansehen. Den Lauf als kontrollierten Funktionstest dokumentieren.
2. Auf DOP einen getrennt dokumentierten Versuch mit festgelegten Lern- und
   Prüfbeständen durchführen. Regeln einfrieren; dann unveränderte Lieferungen
   und protokollierte Fehlerkopien prüfen. Unveränderte Daten sind ohne
   fachliche Prüfung nicht automatisch fehlerfrei.
3. Für ALKIS zunächst eine Objektart und passende Attribute auswählen. Einlesung,
   Untersuchungseinheiten, Merkmale und Produkterwartungen dafür definieren;
   anschließend denselben Untersuchungsablauf wiederverwenden. DOP und ALKIS
   werden nicht ungeprüft in einem gemeinsamen Modell vermischt.
4. Später getrennt untersuchen, welchen zusätzlichen Nutzen Anomalie-Ranglisten
   neben den Regeln bringen. Dafür sind eigene Referenzen und Kennzahlen nötig.

Die bereits untersuchten 500 DOP-Datensätze und der darauf trainierte Autoencoder
sind kein unberührter Testbestand des gesamten Frameworks. Eine spätere
Fehlerinjektion darin ist als explorative Untersuchung zu kennzeichnen.

Mitgelieferte Ergebnisse unter `beispiel_ergebnis` stammen vom dokumentierten
Referenzlauf dieses Pakets. Maßgeblich für deine Umgebung sind die Ergebnisse,
die der Befehl mit deinem tatsächlich vorhandenen Framework-Code erzeugt.
