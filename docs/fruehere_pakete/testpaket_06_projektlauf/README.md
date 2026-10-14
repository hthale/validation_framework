# Testpaket 06 – das Projekt über seine Programmeinstiege prüfen

Dieses Paket führt einen neuen technischen Gesamtdurchlauf **ab einer
Parquet-Merkmalstabelle** aus. Es benutzt deine vorhandenen Programme für
Identitäten, Autoencoder, LOF, HDBSCAN, UMAP, Regelerzeugung und Prüfansicht.
Die konkrete Regelliste und die Zielspalten werden vom Projektcode selbst
bestimmt. Es werden keine vorab gespeicherten Modelle oder Regeln eingesetzt.

## Start

Den Ordner `testpaket_06_projektlauf` aus der ZIP direkt in deinen vorhandenen
Ordner `validierungs-framework` legen, neben `src`, `tests` und `data`.
Es ist kein Installationsskript nötig; Testpaket 05 wird nicht vorausgesetzt.

Mit aktivierter Projekt-`.venv` im Projektordner ausführen:

```powershell
python .\testpaket_06_projektlauf\run_project_check.py --project .
```

Der genaue neue Ergebnisordner wird am Ende ausgegeben. Den Bericht des neuesten
Laufs öffnest du anschließend so:

```powershell
$projectRun = Get-ChildItem .\data\project_check_06 -Directory | Sort-Object Name -Descending | Select-Object -First 1
Start-Process (Join-Path $projectRun.FullName "bericht.html")
```

Der Bericht verlinkt die tatsächlich erzeugte Prüfansicht, den Regelkatalog und
die UMAP-Darstellung. Alle Daten und Ergebnisse werden bei jedem Start neu
erzeugt. Vorhandene Prüfläufe, fachliche Entscheidungen und Quelldateien werden
nicht überschrieben.

## Was der Test konkret startet

| Stufe | Vorhandener Programmeinstieg | Überprüftes Verhalten |
|---|---|---|
| Grundlagen | `src/check_foundation.py` | Parquet lesen, Kennungen erzeugen, Originalattribute erhalten |
| Repräsentation | `src/latentspace/main_v2.py` | Tatsächliches Autoencoder-Training, endliche latente Werte und Verluste |
| Scores und Darstellung | ebenfalls `main_v2.py` | LOF-Rangliste, HDBSCAN, UMAP, Modell- und Nachbarexporte |
| Regeln | `src/main_phase1_phase_2.py` | Automatische Zielauswahl, Profiling und Baumregeln, getrennte Phase-2-Validierung |
| Integration | `src/build_review.py` | Kennungen und Rangfolge erhalten, Regeln und Hinweise in Prüfansicht übertragen |
| Entscheidungsprotokoll | `src/review_decisions.py` | Trockenlauf, Import, Revision und Zurückweisung eines veralteten Imports |

Grundlagen, Ergebnisintegration und Entscheidungen werden als eigene
Python-Prozesse über ihre CLI aufgerufen. Die Latent- und Regelstufe werden
ebenfalls in eigenen Prozessen über ihre normalen `__main__`-Einstiege gestartet.
Ein kleiner Testaufruf setzt dort nur die erforderlichen Testeinstellungen:

- Ein-/Ausgabepfade und Splitmanifest liegen im neuen Testordner.
- Drei Epochen für einen kurzen technischen Trainingslauf; Architektur,
  Loss-Funktionen und Standardparameter bleiben aus der vorhandenen Konfiguration.
- Phase 0 eingeschaltet, LLM-Vorschläge und Delivery Memory ausgeschaltet.
- Zwei Rechenthreads; automatische Browser-/Explorer-Fenster bleiben geschlossen.

Die Anpassungen gelten ausschließlich im jeweiligen Testprozess. Es werden
keine Projektkonfigurationen auf der Festplatte verändert. Keine Modell-,
Score-, Lern- oder Prüfkomponente wird durch eine Attrappe ersetzt. Die
Güteschwellen werden nicht abgesenkt; die tatsächlich verwendeten Einstellungen
werden mitgespeichert.

## Künstliche Referenzdaten

320 Datensätze gehören zu 160 Einheiten mit je zwei Beobachtungen. Die
Gruppierung nutzt die regulären DOP-Gruppenspalten CRS, Ost- und Nordkoordinate.
Die Referenz enthält diese vorher festgelegten Testannahmen:

| Kanaltyp | Detailstufe | Bandanzahl | Bodenpixelgröße in cm | Exportprofil |
|---|---|---:|---:|---|
| RGB | standard | 3 | 20 | P_A |
| RGB | fein | 3 | 10 | P_B |
| RGBI | standard | 4 | 20 | P_C |
| RGBI | fein | 4 | 10 | P_D |

Die Exportprofile sind frei erfunden. Zusätzlich ist die synthetische
Standardabweichung jeweils das Doppelte der Bodenpixelgröße. Diese Festlegungen
sind **keine verbindlichen DOP-/ALKIS-Produktspezifikationen**.

Die Kontrollgröße `kontrollwert` hat bei ansonsten gleichen Modellprädiktoren
pro Einheit einmal K_A und einmal K_B. Ihre Regeln sollen die Auswahlbedingungen
nicht bestehen. IDs und Dateinamen werden von der regulären Abhängigkeitsanalyse
als Prädiktoren ausgeschlossen.

Bei genau einem Datensatz weicht der TIFF-Bezeichner absichtlich vom XML- und
Kachelbezeichner ab. Dieser Hinweis soll mit der richtigen Kennung bis zur
Prüfansicht gelangen. Es werden keine tatsächlichen XML- oder TIFF-Dateien
erzeugt; die Bezeichner sind Merkmale in der Parquet-Tabelle.

Der Soll-Ist-Vergleich prüft die drei Referenzabhängigkeiten über alle vier
Kombinationen: Richtige Kombinationen sollen passen; absichtlich vertauschte
Zielwerte sollen eine gelernte Regel verletzen. Eine bestimmte Formulierung oder
Reihenfolge der Baumpfade wird nicht verlangt. Die übrigen automatisch erzeugten
Kandidaten bleiben ebenfalls im normalen Regelkatalog sichtbar.

## Absichtlich ungültige Übergaben

Zusätzlich muss dein Projekt vier Fälle kontrolliert zurückweisen:

1. Doppelte Eingabeidentität in der Grundlagenprüfung.
2. Eine Phase-0-Ergebnisdatei, der ein Datensatz fehlt.
3. Eine geänderte Eingabe zusammen mit einem alten Phase-1/2-Lauf.
4. Den erneuten Import derselben Entscheidung mit veralteter Revision.

Nicht jeder Programmabbruch zählt als Erfolg: Der Test prüft den Rückgabestatus
und die passende Fehlermeldung. Ein Importfehler oder eine fehlende Abhängigkeit
wird daher nicht als korrekt erkannter Datenfehler gewertet.

Der Entscheidungsimport läuft auf einer **separaten Kopie der künstlichen
Prüfansicht**. Reviewer und Begründung kennzeichnen ihn als automatisierten
Techniktest. Er stellt keine fachliche Entscheidung über reale Daten dar. Die
eigentliche synthetische Prüfansicht bleibt mit 320 offenen Fällen erhalten.

## Ergebnisse und Fehlerdiagnose

Jeder Start erzeugt einen eigenen Zeitstempelordner unter:

```text
data/project_check_06/
```

| Datei oder Unterordner | Inhalt |
|---|---|
| `bericht.html` | Lesbarer Überblick und Links zur erzeugten Prüfansicht |
| `summary.json` | Sämtliche Prüfkriterien, Regelzahlen und Programmrückgaben |
| `run_status.json` | Laufstatus; im Fehlerfall zuletzt abgeschlossene Schritte |
| `run_manifest.json` | Prüfsummen des Quellcodes und aller Ergebnisse, Versionen |
| `effective_latent_config.json`, `effective_rule_config.json` | Tatsächlich verwendete Testeinstellungen |
| `synthetic_input.parquet`, `reference_spec.json` | Künstliche Eingabe und vorher festgelegte Erwartungen |
| `foundation`, `latent`, `phase12`, `review` | Normale Ausgaben der jeweiligen Projektprogramme |
| `review_import_test` | Gesonderte synthetische Prüfansicht für den Importtest |
| `negative_cases` | Absichtlich ungültige Eingaben für die Ablehnungstests |
| `logs` | Vollständige Programmausgaben jeder Stufe |
| `failure_traceback.txt` | Bei einem Abbruch: genaue Fehlerstelle |

Die erste UMAP-Ausführung kann wegen Initialisierung länger dauern. Während
einer Stufe zeigt das Terminal ihren Namen; Details stehen bereits in der
zugehörigen Logdatei. Benötigt werden die Pakete deiner vorhandenen `.venv`,
unter anderem PyTorch, hdbscan, umap-learn und ein Parquet-Reader. Fehlende
Voraussetzungen werden protokolliert; der Test installiert nichts nach.

Ein bestandener Lauf endet mit `project_check_passed: true` und Exitcode 0.
Ein Fehler führt zu Exitcode 1 und einem Diagnoseordner. Die erwarteten
Ablehnungsfälle haben in ihren einzelnen Logdateien einen Fehlerstatus und sind
ausdrücklich als solche dokumentiert.

Optional können die gezielten Tests des Testläufers ausgeführt werden:

```powershell
python .\testpaket_06_projektlauf\tests\test_project_check.py --project .
```

## Aussagekraft

Dieser Lauf prüft die Zusammenarbeit der vorhandenen Programme ab Parquet.
Paket 05 hatte zuvor die Regelerzeugung und Fehlererkennung unter kontrollierter
Trennung geprüft. Paket 06 ergänzt den tatsächlichen Projektablauf einschließlich
der automatisch ausgewählten Zielspalten und der Modellstufe.

Die drei Trainingsepochen dienen der technischen Ausführung. Es gibt bewusst
keine Bestehensvorgabe für die konkrete Clusterzahl, LOF-Reihenfolge oder Höhe
der Verluste. Die Ausgabe sind Auffälligkeitshinweise, keine Fehleretiketten.

Der Autoencoder und das Bestandsprofiling sehen im normalen Projektablauf den
gesamten künstlichen Bestand. Die getrennte Phase-2-Aufteilung gilt für
überwachte Attributabhängigkeiten. Der anschließende semantische Referenztest
ist kein unabhängiger Leistungstest des gesamten Frameworks. Die aktuelle
Prüfansicht verwendet ausgewählte ML-Regeln nur im Validierungsanteil.

Nicht geprüft werden die Gewinnung der Merkmale aus echten XML-/TIFF-/NAS-
Dateien, reale ALKIS-Geometrien, fachliche Gültigkeit der Produktannahmen,
Fehlererkennungsleistung auf neuen Lieferungen oder die Bedienung der
Browseroberfläche. Die HTML-Daten und die gespeicherten Revisionen werden
geprüft; ein automatisierter Klicktest ist nicht enthalten.

Nach erfolgreichem Lauf folgt ein eigener DOP-Versuch mit dokumentierter
Datenaufteilung und eingefrorenen Regeln, danach eine ausgewählte ALKIS-
Objektart mit passender Einlesung und Merkmalsbildung. Die bereits untersuchten
500 DOP-Datensätze bleiben explorative Daten; ihre bisherige Verwendung durch
den Autoencoder lässt sich nicht durch eine neue Aufteilung rückgängig machen.
