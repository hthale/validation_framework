# Testpaket 13 – Kontrollierte Regeln und separater LOF-Vergleich

Den enthaltenen Ordner `testpaket_13_kontrollierter_vergleich` direkt in den
vorhandenen Ordner `validierungs-framework` entpacken, neben `src`, `data` und
den Paketen 10, 11 und 12. Alle drei vorherigen Paketordner werden benötigt.
Das Paket ergänzt einen Versuch; es installiert oder überschreibt keine Dateien.
Es verwendet die bestehende virtuelle Umgebung des erfolgreich ausgeführten
Vergleichs 11. Keine zusätzlichen Bibliotheken gegenüber diesem Lauf erforderlich.

Ein vollständig gespeicherter lokaler Probelauf ist unter
`beispielauswertung\bericht.html` enthalten. Er kann nach dem Entpacken sofort
geöffnet werden. Er stammt aus der Entwicklungsumgebung, nicht aus deinem
Windows-Projekt. Die Nachprüfung mit aufgezeichneten Quellpfaden ist für deine
eigenen neu erzeugten Läufe vorgesehen; der Beispielordner dokumentiert den
bereits durchgeführten technischen Probelauf samt Modellartefakten.

## Start in PowerShell

Im Projektordner ausführen:

```powershell
.\.venv\Scripts\python.exe .\testpaket_13_kontrollierter_vergleich\run_controlled_comparison.py --project .
```

Es werden ausschließlich neue künstliche Tabellen erzeugt. Keine ALKIS-/DOP-Datei
angeben. Standard: 1.200 Einheiten mit zwei Beobachtungen, drei Seeds, zwei Szenarien,
25 Epochen je Autoencoder, Dimension 3. Es werden zwölf Autoencoder trainiert;
dieser Lauf dauert deshalb länger als Paket 12. Der tatsächliche Trainingsfortschritt
steht zusätzlich in `seed_*/<Szenario>/*_representation/training.log`.

Ergebnisse unter `data\kontrollvergleich_13\<Zeitstempel>`. Bericht öffnen:

```powershell
$vergleich13 = Get-ChildItem .\data\kontrollvergleich_13 -Directory |
  Where-Object {
    (Test-Path (Join-Path $_.FullName "run_status.json")) -and
    ((Get-Content (Join-Path $_.FullName "run_status.json") -Raw | ConvertFrom-Json).status -eq "completed")
  } |
  Sort-Object Name -Descending |
  Select-Object -First 1
if ($null -eq $vergleich13) { throw "Kein abgeschlossener Paket-13-Lauf gefunden." }
Start-Process (Join-Path $vergleich13.FullName "bericht.html")
```

Für die Ergebnisbesprechung bitte diese drei Dateien aus dem neuen Lauf verwenden:

- `summary.json`
- `rule_metrics.json`
- `lof_metrics.json`

`paired_comparisons.json` enthält zusätzlich die direkten Methodenunterschiede.

## Was der Versuch zeigt

Es gibt drei gültige Gruppen mit unterschiedlichen erwarteten Prüfwerten.
Die seltene Gruppe C bleibt gültig. Ein unabhängiges Kontrollziel soll keine
ausgewählte Regel erzeugen. Das Framework lernt die Regelbedingungen mit seinem
vorhandenen Baum-/Regelcode; die erwarteten Zusammenhänge werden nicht als fertige
Regeln eingespeist. Wahrheit und Fehlerlabels dienen nur der Auswertung.

Verglichen werden globale Baumregeln, Regeln in Attributclustern, Regeln in
Latentclustern und eine einfache Mehrheitsvorhersage. Zusätzliche irrelevante
Attribute bilden ein zweites, vorab festgelegtes Szenario. Die bisherigen
Güteschwellen gelten weiter. Keine Methode muss gewinnen, damit der Techniktest
besteht. Auch null ausgewählte lokale Regeln bleiben ein sichtbares Ergebnis.

LOF ist eine eigene Aufgabe. Es vergleicht dieselben fachlichen Attribute auf drei
Darstellungen: vorverarbeitete Attribute, PCA und Autoencoder. Alle drei erhalten
dasselbe Prüfbudget. Die seltenen gültigen Fälle werden separat ausgewiesen.
Eine fixe Schwelle aus der normalen Validierung wird ergänzend berichtet.

Der Test beinhaltet echte neue Trainingsläufe und eingefrorene Vorhersagen auf
getrennten Einheiten. Er ist keine Wiederverwendung der alten Rankings und keine
Attrappe der Framework-Modelle. Dennoch betrifft seine Aussage nur die künstlichen
Datenmechanismen. Ein Erfolg wäre kein Nachweis realer DOP-/ALKIS-Güte.

## Details und Nachprüfung

`PROTOKOLL.md` beschreibt vorab den Generator, alle Parameter, Merkmalsverträge,
Fehlerarten, Kontrollfälle und Grenzen. Trainingsartefakte, Regeln, Ablehnungsgründe,
Skalierungen, LOF, Falltabellen und Prüfsummen werden gespeichert. Kennungen und
Referenzlabels gelangen nicht in die Modelle. Geometrien werden nicht verwendet.

Nachprüfung eines fertigen Laufs (trainiert keinen Autoencoder neu):

```powershell
.\.venv\Scripts\python.exe .\testpaket_13_kontrollierter_vergleich\verify_controlled_comparison.py `
  --project . --run "$($vergleich13.FullName)"
```

Gezielte Tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\testpaket_13_kontrollierter_vergleich\tests -v
```

CLI-Optionen `--epochs`, `--seeds`, `--units` und `--output` sind für ausdrücklich
dokumentierte weitere Versuche verfügbar. Für die erste Auswertung die Vorgaben
beibehalten. Parameter nicht anhand des jeweils besten Testergebnisses auswählen.
