# Validierungsframework für Geodatenlieferungen

Begleitender Quellcode und Versuchsergebnisse zur Masterarbeit von Hendrik Thale:

**„Konzeption eines Frameworks zur Validierung von Geodatenlieferungen auf Basis heuristischer und maschineller Lernverfahren“**

## 1. Zweck und Untersuchungsumfang

Das Projekt untersucht, wie bekannte Prüfbedingungen und datengetriebene Analysen in einem Validierungsframework verbunden werden können. Der Prototyp verarbeitet vorbereitete Merkmalstabellen und erzeugt Profiling-Regeln, Kandidaten für Attributabhängigkeiten, Clusterbeschreibungen und Auffälligkeitshinweise.

Verwendet werden Merkmale digitaler Orthophotos (DOP), ausgewählte ALKIS-Sachattribute sowie künstliche Daten mit festgelegten Referenzbeziehungen. In den dafür vorgesehenen Versuchen werden gezielte Änderungen an getrennten Testdaten untersucht.

Der generische Programmeinstieg beginnt bei einer flachen Parquet-Merkmalstabelle. Eine allgemeine automatische Erschließung unbekannter Rohdatenprodukte ist nicht umgesetzt. Bildinhalte und ALKIS-Geometrien sind nicht Gegenstand der untersuchten Analysen. Die Delivery Memory ist im generischen Lauf deaktiviert; ihre Nutzung über aufeinanderfolgende Lieferungen wurde nicht praktisch nachgewiesen.

## 2. Projektstruktur

Die Anleitung bezieht sich auf die geordnete Projektfassung. Befehle werden im Hauptordner des entpackten Projekts ausgeführt.

| Pfad | Inhalt |
| --- | --- |
| `src/` | Framework-Code, generischer Programmeinstieg, Repräsentationslernen, Regelableitung und Ergebnisintegration |
| `adapters/alkis_bahnverkehr/` | Aufbereitung der ausgewählten ALKIS-Bahnverkehrsattribute |
| `experimente/` | Funktionsprüfungen und Methodenvergleiche einschließlich ihrer Anleitungen und Versuchsprotokolle |
| `tests/` | Tests der Framework-Funktionen |
| `werkzeuge/` | Hilfsprogramme, unter anderem zur Prüfung der geordneten Projektstruktur |
| `docs/` | Ergänzende Projekt- und Abgabedokumentation |
| `data/` | Eingabetabellen und gespeicherte Versuchsläufe |
| `requirements.txt` | Vorhandene Projekt-Abhängigkeitsliste |
| `requirements-lock.txt` | Für die Abgabe exportierter Paketstand der verwendeten Python-Umgebung |
| `python-version.txt` | Python-Version der für die Abgabe dokumentierten Umgebung |

Die lokalen `README.md`- und `PROTOKOLL.md`-Dateien bleiben bei den jeweiligen Experimenten. Vorlagen wie `src/review_template.html` gehören zum ausführbaren Projekt und müssen ebenfalls enthalten sein.

## 3. Einrichtung unter Windows / PowerShell

Die dokumentierte Python-Version steht in `python-version.txt`. Die Befehle setzen voraus, dass `python` diese Version aufruft. Zunächst prüfen:

```powershell
Get-Content .\python-version.txt
python --version
```

Danach eine eigene virtuelle Umgebung im entpackten Projekt anlegen und die Abhängigkeiten installieren:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r .\requirements-lock.txt
& .\.venv\Scripts\python.exe -m pip check
```

Eine Aktivierung der Umgebung ist für die folgenden Befehle nicht erforderlich, da der Python-Interpreter direkt aufgerufen wird. Die ursprüngliche `.venv` ist nicht Bestandteil der Abgabe.

Der Paketstand in `requirements-lock.txt` dokumentiert die zum Zeitpunkt des Exports verwendete Umgebung. Angaben zur Umgebung älterer Versuche stehen zusätzlich in den jeweiligen Laufprotokollen beziehungsweise Manifesten.

## 4. Erster Funktionslauf mit künstlichen Daten

Voraussetzung ist die beigefügte Datei `data/generic_example_07.parquet` mit 320 künstlichen Datensätzen.

```powershell
& .\.venv\Scripts\python.exe .\src\run_generic.py `
    --input ".\data\generic_example_07.parquet" `
    --output ".\data\abgabe_funktionslauf" `
    --epochs 3
```

Der Lauf trainiert den Autoencoder, berechnet LOF und Cluster, leitet Regelkandidaten ab und erstellt einen Bericht. Drei Epochen dienen hier einem kurzen technischen Funktionslauf; damit wird keine Güte für reale Daten nachgewiesen.

Die Ergebnisse liegen in einem neu angelegten Unterordner mit Zeitstempel. Der Bericht lässt sich mit folgendem Befehl öffnen:

```powershell
$lauf = Get-ChildItem .\data\abgabe_funktionslauf -Directory |
    Sort-Object Name -Descending |
    Select-Object -First 1

Start-Process (Join-Path $lauf.FullName "bericht.html")
```

Zu den Ausgaben des generischen Laufs gehören:

| Datei | Inhalt |
| --- | --- |
| `bericht.html` | Übersicht über den Lauf und seine Ergebnisse |
| `input_with_ids.parquet` | Eingabekopie mit Kennungen zur Ergebniszuordnung |
| `feature_contract.json` | Technische Merkmalsrollen und verwendete Lernmerkmale |
| `rule_catalog.json` | Profiling-Regeln, Abhängigkeitskandidaten und Clusterbeschreibungen |
| `dependency_reports.json` | Auswertung der untersuchten Zielmerkmale |
| `latent_outlier_results_lof.parquet` | Latente Vektoren, LOF-Ergebnisse und Clusterzuordnungen |
| `umap.png`, `umap.html` | Darstellung der latenten Vektoren in zwei Dimensionen |
| `training_history.json` | Trainingsverlauf |
| `model_config.json` | Modell- und Aufbereitungseinstellungen |

## 5. Versuche und gespeicherte Ergebnisse

Die folgenden Pfade ordnen die vorhandenen Versuchsprogramme und die zugehörigen Ergebnisordner ein. Innerhalb der Ergebnisordner kennzeichnen Zeitstempel einzelne Läufe. Für die Ergebnisse der Masterarbeit sind die dort dokumentierten Konfigurationen und Laufstände maßgeblich.

| Untersuchung | Programmeinstieg oder Programmordner | Ergebnisordner unter `data/` |
| --- | --- | --- |
| Synthetische Regel-Funktionsprüfung mit 600 Zeilen | `experimente/regelableitung/` | `synthetic_rule_check_05/` |
| Technischer Integrationstest | `experimente/integrationstest/` | `project_check_06/` |
| Generischer Lauf mit 320 produktneutralen Zeilen | `src/run_generic.py` | `generic_example_runs_07/` |
| Generische DOP-Bestandsanalyse | `src/run_generic.py` | `dop_generic_07/` |
| DOP-Vergleich der Repräsentationen für LOF | `src/run_latent_comparison.py` | `dop_latent_comparison_08/` |
| Gemeinsame DOP-Regel- und Auffälligkeitsprüfung | `experimente/dop_pruefung/` | `dop_joint_check_09/` |
| ALKIS-Attributaufbereitung und Bestandslauf | `adapters/alkis_bahnverkehr/` | `bahnverkehr_attribute_10/` |
| Globaler und clusterbezogener ALKIS-Regelvergleich | `experimente/clusterregeln/` | `regelvergleich_11/` |
| ALKIS-Vergleich mit der Mehrheitsvorhersage | `experimente/mehrheitsvergleich/` | `mehrheitsvergleich_12/` |
| Kontrollierter synthetischer Methodenvergleich | `experimente/synthetischer_vergleich/` | `kontrollvergleich_13/` |

Die Aufrufe und Voraussetzungen der einzelnen Experimente stehen in den zugehörigen Anleitungen und Versuchsprotokollen. Ein neuer Lauf ist getrennt von den gespeicherten Ergebnissen anzulegen. Die gespeicherten Ausgaben werden für die Abgabe nicht nachträglich überschrieben.

### Eingaben und vollständige Laufordner

Der generische Beispielbestand liegt unter `data/generic_example_07.parquet`. Die 600-zeilige Referenztabelle liegt im zugehörigen Lauf unter `data/synthetic_rule_check_05/<Lauf>/synthetic_reference.parquet`.

Die vorbereitete DOP-Tabelle mit Kennungen liegt unter `data/check_01/dop_training_dataset_with_ids.parquet`. Für ALKIS sind die vorbereiteten Attributtabellen und ihre Aufbereitungsnachweise im verwendeten Lauf unter `data/bahnverkehr_attribute_10/` maßgeblich. Die ursprüngliche Bahnverkehrsdatei wird im dokumentierten Projekt unter `data/parquetdata/bahnverkehr.parquet` angesprochen.

Gespeicherte Versuchsläufe umfassen neben Berichten auch die dazugehörigen Eingaben, Aufteilungen, Konfigurationen, Metriken, Änderungsprotokolle und gegebenenfalls Modelle. Die Bestandteile eines ausgewählten Laufs bleiben zusammen. Dies gilt beim kontrollierten synthetischen Vergleich für sämtliche verwendeten Seeds und Varianten.

Falls reale Quelldaten nicht mitgeliefert werden, muss die ergänzende Datenübersicht unter `docs/` angeben, welche Dateien fehlen und welche Aufrufe dadurch nicht ausführbar sind. Die künstlichen Daten ermöglichen einen Funktionslauf ohne diese realen Quelldaten.

## 6. Fachliche Prüfansicht

Die gespeicherte DOP-Prüfansicht kann ohne erneutes Training geöffnet werden:

```powershell
Start-Process ".\data\output_review_dop_check_04\20260922_160812_685173\review.html"
```

Die Ansicht zeigt Prüfhinweise und Originalmerkmale und ermöglicht die Erfassung fachlicher Entscheidungen mit Begründung. Die Bewertung verändert die Originaldaten nicht. Änderungen müssen über die vorgesehene Exportfunktion heruntergeladen und für eine Übernahme in den gespeicherten Stand mit dem zugehörigen Importprogramm verarbeitet werden.

Für die reine Anzeige genügt die HTML-Datei. Für den Entscheidungsimport und eine erneute Erstellung der Ansicht müssen auch die zugehörigen Dateien, insbesondere `review_bundle.json` und `review_history.json`, erhalten bleiben. Der vollständige Review-Lauf gehört deshalb zur dokumentierten Ergebnisablage.

## 7. Technische Prüfungen

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
& .\.venv\Scripts\python.exe .\werkzeuge\pruefe_struktur.py --tests
```

Diese Prüfungen kontrollieren Funktionen, Importpfade und ausgewählte Fehlerfälle. Sie ersetzen weder die fachliche Bewertung realer Daten noch die erneute Ausführung der Methodenvergleiche. Die abschließende Prüfung des Abgabearchivs sollte nach dem Entpacken in einen anderen Ordner erfolgen.

Einige historische Manifeste enthalten absolute Pfade des ursprünglichen Rechners. Solche Pfadangaben werden als Teil des damaligen Protokolls erhalten. Eine historische Prüfdatei kann deshalb auf einem anderen Rechner nicht ohne Weiteres ausführbar sein. Das Lesen der gespeicherten Ergebnisse, ein neuer Programmlauf und die erneute Verifikation eines historischen Laufs sind getrennte Schritte.

## 8. Einordnung der Ergebnisse

- Im generischen Bestandslauf verwenden Repräsentationslernen, LOF, Profiling und Clusterbeschreibungen den eingegebenen Gesamtbestand. Das Lernen von Attributabhängigkeiten besitzt eine gesonderte Trainings- und Validierungsaufteilung mit einem dabei unbenutzten Testanteil.
- Die kontrollierten Methodenvergleiche haben eigene Aufteilungen und Prüfprotokolle. Ihre Ergebnisse dürfen nicht mit denen eines beschreibenden Bestandslaufs gleichgesetzt werden.
- HDBSCAN und LOF arbeiten im generischen Lauf auf den vollständigen latenten Vektoren. UMAP dient ihrer Darstellung in zwei Dimensionen.
- Ein hoher LOF-Wert ist ein Prüfhinweis und kein bestätigter Fehler. Ein ausgewählter Regelkandidat ist noch keine fachlich freigegebene Prüfregel.
- Reale unveränderte Daten werden ohne fachliche Prüfung nicht pauschal als fehlerfrei behandelt.
- Ein allgemeiner Vorteil der Autoencoder-Darstellung wird nicht vorausgesetzt. Die Aussagekraft der Vergleiche wird in der Masterarbeit anhand der jeweiligen Prüfaufgabe bewertet.

## 9. Dokumentation und Weitergabe

`docs/` enthält ergänzende Unterlagen zur Projektfassung, die Zuordnung der beigefügten Daten und Ergebnisse sowie gegebenenfalls die für die Abgabe vorgesehenen Abbildungen. Die Experimentanleitungen bleiben zusätzlich in den jeweiligen Programmordnern.

Vor einer öffentlichen Weitergabe sind die Nutzungs- und Weitergabebedingungen der enthaltenen Daten zu berücksichtigen. Zugangsdaten und lokale Umgebungsdateien gehören nicht in das Archiv. Eine pauschale Nutzungserlaubnis für Daten oder Fremdsoftware wird durch diese README nicht erteilt.
