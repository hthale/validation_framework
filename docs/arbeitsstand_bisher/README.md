# Arbeitsstand für die Masterarbeit · 23.09.2026

Diese Unterlagen halten den belegten Projektstand fest und bereiten das Schreiben von Kapitel 5 vor. Sie installieren keine neue Framework-Version.

1. Den Ordner `arbeitsstand_masterarbeit` aus dem ZIP direkt in vorhandenen Projektordner entpacken.
2. `START_HIER.html` im Browser öffnen.
3. Die Bestandsaufnahme mit dem untenstehenden Befehl ausführen. Sie liest Dateinamen und Python-Prüfsummen, startet keine Projektprogramme und schreibt ausschließlich einen neuen Inventarordner.

```powershell
.\.venv\Scripts\python.exe .\arbeitsstand_masterarbeit\projektinventar.py --project .
```

Am Ende werden die Pfade zu `inventar.md` und `inventar.json` ausgegeben. Für den Abgleich des tatsächlichen Windows-Projekts genügt zunächst `inventar.json`. Es enthält keine Quellcodeinhalte oder Datensatzwerte.

| Datei | Zweck |
|---|---|
| `PROJEKTSTAND.md` | Fehlerexperimente, Ergebnisse, Beitrag des latenten Raums und offene Grenzen |
| `SCHREIBPLAN_KAPITEL_5.md` | Gliederung, Zuordnung zu tatsächlichen Modulen und Einstieg für morgen |
| `BEREINIGUNG.md` | Was aktiv bleibt, was zur Evaluation gehört und was später archiviert werden kann |
| `ergebnisbelege.json` | Aus den hochgeladenen Ergebnissen abgeleitete Zahlen mit Herkunft und Prüfsummen |
| `referenz_code.json` | Prüfsummen der hier verfügbaren Arbeitskopie; keine Behauptung über des aktuellen lokalen Stand |
| `projektinventar.py` | Bestandsaufnahme ohne Installation, Training, Verschieben oder Löschen |

Die Ergebnisbelege stammen aus hochgeladenen Berichten. Die Codezuordnung wurde an der hier verfügbaren Arbeitskopie mit dem Kernstand 07/08 und den Zusatzprogrammen bis 13 geprüft.  Windows-Projekt wurde in diesem Schritt nicht direkt eingesehen oder verändert. Abweichende Prüfsummen sind ein Abgleichhinweis und nicht automatisch ein Fehler.
