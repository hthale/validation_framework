# Technischer Nachweis und Ergebnis des lokalen Probelaufs

Datum: 23.09.2026. Entwicklungsumgebung Linux, Python 3.12.14. Alle sechs
Python-Dateien wurden zusätzlich auf Syntaxkompatibilität mit Python 3.11 geprüft.
Ein nativer Windows-Lauf wurde nicht durchgeführt.

- 14 gezielte Tests unter Pandas 2.2.3 bestanden.
- Dieselben 14 Tests unter Pandas 3.0.1 bestanden.
- Vollständiger Standardlauf: Seeds 42/43/44, zwei Szenarien, 1.200 Einheiten
  mit zwei Beobachtungen, zwölf tatsächlich trainierte Autoencoder zu je 25 Epochen.
- 48 Regelauswertungen und 54 LOF-Auswertungen gespeichert.
- 295 technische Nachprüfungen bestanden, einschließlich erneuter Anwendung
  der ausgewählten Regeln und LOF aus gespeicherten Trainings-/Testmatrizen.
- Der unter Pandas 2 gespeicherte vollständige Lauf wurde unter Pandas 3 mit
  denselben 295 Prüfungen erfolgreich erneut ausgewertet; kein neues AE-Training.
- Nullkontrolle in allen 18 Kombinationen aus Seed/Szenario/Regelmethode abgewiesen.
- Ungültige Einheitenzahl vor Anlegen einer Ausgabe abgewiesen; nachträglich
  veränderte Referenzlabel-Datei durch die Artefaktprüfung abgewiesen.
- 43 lokale Berichtlinks auf vorhandene Dateien geprüft. Keine visuelle
  Browserprüfung oder automatisierte Browserbedienung.

## Beobachtetes Ergebnis, ohne Parameteranpassung

Bei vier informativen Merkmalen finden globale Regeln und beide Clusterverfahren
in allen Seeds sämtliche 48 synthetischen Prüfcodefehler ohne Fehlmeldung auf
gültigen Testfällen. Die einfache Mehrheitsvorhersage liefert dagegen je Seed
130 Fehlmeldungen auf gültigen Fällen und verfehlt zwei Fehler der seltenen Gruppe.

Mit zwölf zusätzlichen unabhängigen Merkmalen erkennen Attributclusterregeln
45/43/38 Fehler (Seeds 42/43/44); Latentclusterregeln erkennen 48/48/47. Beide
liefern in diesen Läufen keine Fehlmeldung auf gültigen Fällen. Attributcluster
decken keine der 22 seltenen gültigen Testzeilen im gemischten Bestand ab,
Latentcluster 20/20/15. Globale Baumregeln erkennen weiterhin alle 48 Fehler,
decken alle seltenen gültigen Fälle ab und erzeugen keine Fehlmeldung.

Beim separaten LOF-Vergleich ergibt sich kein allgemeiner Gewinn des Autoencoders:
Unter zusätzlichen Störmerkmalen findet LOF auf den vorverarbeiteten Attributen
beim Budget von 48 Fällen je nach Seed 18/17/21 Prüfcodefehler; LOF im latenten Raum
5/4/6. Bei übernommenen falschen Messwerten lauten die Trefferzahlen 27/27/25
für Attribute und 13/14/13 für den latenten Raum. PCA findet bei dieser zweiten
Fehlerart 42/41/43 und verdeutlicht, dass Dimensionsreduktion und Autoencoder-Nutzen
getrennt beurteilt werden müssen.

Diese Ergebnisse gelten nur für den festgelegten künstlichen Mechanismus und
diese Konfiguration. Sie belegen weder eine generelle Überlegenheit noch eine
generelle Nutzlosigkeit latenter Räume und keine reale DOP-/ALKIS-Erkennungsleistung.
Negative Ergebnisse wurden beibehalten; kein Seed und kein Parameter wurde nach
Testgüte ausgewählt. Vollständiger Beispielbericht und Artefakte: `beispielauswertung`.
