# Analysepaket 12 – Mehrheitsvergleich und kontrollierte Änderungen

Den Ordner `analysepaket_12_mehrheitsvergleich` aus der ZIP direkt in den
vorhandenen Projektordner entpacken. Er liegt neben `src`, `data`,
`analysepaket_10_bahn_attribute` und `vergleichspaket_11_clusterregeln`.
Kein Installer, keine ersetzten Projektdateien, kein neues Autoencoder-Training.
Die bestehende virtuelle Umgebung aus Paket 11 wird weiterverwendet.

Im Projektordner in PowerShell starten:

```powershell
.\.venv\Scripts\python.exe .\analysepaket_12_mehrheitsvergleich\run_baseline_check.py --project .
```

Das Skript sucht den neuesten vollständig abgeschlossenen Lauf unter
`data\regelvergleich_11`. Der verwendete Pfad wird zu Beginn angezeigt.
Für einen bestimmten Lauf:

```powershell
.\.venv\Scripts\python.exe .\analysepaket_12_mehrheitsvergleich\run_baseline_check.py `
  --project . `
  --run ".\data\regelvergleich_11\DEIN_LAUFORDNER"
```

Der Lauf prüft zunächst alle Dateien und die gespeicherte Auswertung von Paket 11.
Fehlende oder geänderte Voraussetzungen werden vor dem neuen Ausgabeordner
abgewiesen. Die ursprüngliche Parquet-Datei muss am aufgezeichneten Pfad liegen,
und der Projektcode muss noch zum Vergleichslauf passen. Fehler nicht durch
Bearbeiten von Prüfsummen oder Manifesten umgehen.

Danach entsteht ein neuer Lauf unter `data\mehrheitsvergleich_12`. Bericht öffnen:

```powershell
$vergleich12 = Get-ChildItem .\data\mehrheitsvergleich_12 -Directory |
  Where-Object {
    (Test-Path (Join-Path $_.FullName "run_status.json")) -and
    ((Get-Content (Join-Path $_.FullName "run_status.json") -Raw | ConvertFrom-Json).status -eq "completed")
  } |
  Sort-Object Name -Descending |
  Select-Object -First 1
if ($null -eq $vergleich12) { throw "Kein abgeschlossener Paket-12-Lauf gefunden." }
Start-Process (Join-Path $vergleich12.FullName "bericht.html")
```

Für die gemeinsame Ergebnisbesprechung diese drei Dateien aus dem neuen Lauf
bereitstellen: `summary.json`, `majority_comparison.json`, `mutation_metrics.json`.

## Was gemacht wird

1. Bereits ausgewählte Regeln und alle Kandidaten nach Regelart darstellen.
   Eine konstante Erwartung im Cluster behält die Clusterbedingung.
2. Häufigsten Zielwert ausschließlich aus Training bestimmen und auf genau
   denselben Testfällen mit den vorhandenen Regelmethoden vergleichen.
3. Originale unverändert als Kontrolle behalten. Auf getrennten Kopien Zielwerte
   in beide Richtungen ändern und auch fehlende Werte prüfen. Vorherige und neue
   Abweichungen sowie Enthaltungen getrennt berichten.

Die vollständigen Änderungen stehen als `changed_copies/.../original.parquet`,
`changed.parquet`, `pairs.parquet` bereit. `paired_changes.parquet` enthält die
zugehörigen Ergebnisse aller vier Methoden. Keine Kopie verändert die Originaldatei.

Der Datumsindikator und sein Datum werden konsistent gemeinsam verändert. Sonst
wird jeweils nur das Zielattribut geändert. Router-Prädiktoren müssen unverändert
sein, weshalb der gespeicherte Cluster gültig bleibt. Der Framework-Regelprüfer
wird erneut angewendet. Die Analyse misst keine Änderungen an Prädiktoren und
keine LOF-Erkennungsleistung. Details: `PROTOKOLL.md`.

## Nachprüfung / Tests

Die Nachprüfung läuft automatisch, bevor ein Lauf als abgeschlossen markiert wird.
Sie kann später erneut gestartet werden:

```powershell
.\.venv\Scripts\python.exe .\analysepaket_12_mehrheitsvergleich\verify_baseline_check.py `
  --project . --run "$($vergleich12.FullName)"
.\.venv\Scripts\python.exe -m unittest discover -s .\analysepaket_12_mehrheitsvergleich\tests -v
```

Die Tests verwenden kleine künstliche Tabellen. Es gibt keine Netzwerkzugriffe
und keine Änderungen an `src`, vorhandenen Modellen oder fachlichen Entscheidungen.

## Ergebnis richtig lesen

Mehr Übereinstimmungen mit vorhandenen Werten bedeuten nicht automatisch höhere
fachliche Korrektheit. Originaldaten besitzen hier keine geprüften Fehlerlabels.
Eine hohe Quote nach künstlichen Änderungen kann schon durch die einfachste
Mehrheitsvorhersage entstehen. Deshalb enthält der Bericht unveränderte
Nichtmehrheitswerte und die Gegenrichtung zur Mehrheit. Es werden keine Schwellen
gesenkt und keine positiven Aussagen über den latenten Raum erzwungen.
