# Technische Prüfung vor Bereitstellung

Geprüft am 23.09.2026 unter Linux / Python 3.12.14. Zusätzlich wurde der
Python-3.11-Sprachumfang aller fünf Python-Dateien per Parser geprüft.
Es wurde kein nativer Windows-Lauf durchgeführt.

- 12 gezielte Tests mit Pandas 2.2.3: bestanden.
- Dieselben 12 Tests mit Pandas 3.0.1: bestanden.
- Vollständige Zusatzanalyse eines vorhandenen Bahnverkehr-Vergleichslaufs mit
  5.000 Zeilen, drei Seeds und sechs Zielen je Seed: unter beiden Versionen bestanden.
- Je Lauf 119 Artefakt-/Fallprüfungen bestanden, einschließlich Nachrechnen mit
  dem Framework-Regelprüfer. 51 Prüfungen bestätigen unveränderte Router-Eingaben.
- 2.082 kontrollierte Änderungen, jeweils mit unveränderter Vergleichskopie,
  ergeben 8.328 Methodenauswertungen. Die Summen umfassen wiederholte Aufteilungen.
- Mehrheitsvergleich, Änderungskennzahlen, Regeltypzahlen und Zusammenfassung
  sind zwischen Pandas 2 und 3 inhaltlich identisch.
- Ein unter Pandas 2 gespeicherter Lauf wurde unter Pandas 3 erfolgreich erneut
  geprüft. Fehlwerte bleiben im Parquet-Rundlauf korrekt erhalten.
- Unvollständiger Lauf, falscher Experimenttyp und veränderte Eingabeartefakte
  werden vor Anlegen einer Ausgabe abgewiesen. Veränderte neue Fallartefakte
  werden bei der Nachprüfung ebenfalls abgewiesen.
- Acht lokale Berichtlinks auf vorhandene Ergebnisdateien geprüft. Keine
  automatisierte Browserbedienung oder visuelle Browserprüfung durchgeführt.

Die synthetischen Tests enthalten ausdrücklich eine echte bedingte Abhängigkeit,
die Nichtmehrheitswerte korrekt vorhersagt und die Mehrheitsmethode übertrifft.
Weitere Tests behandeln die Gegenrichtung zur Mehrheit, bereits vorher
abweichende Fälle, fehlende Werte, Enthaltungen außerhalb von Clustern,
unveränderte Identitäten, Datumsableitungen und Reihenfolgeunabhängigkeit.

Der vollständige technische Prüflauf verwendet einen lokal reproduzierten
Paket-11-Lauf. Er ist nicht identisch mit Hendriks hochgeladenen Ergebnisdateien;
insbesondere dürfen seine Regelzahlen nicht als neue Auswertung des Windows-Laufs
ausgegeben werden. Dessen unveränderte 1.237 hochgeladene Kandidaten wurden
zusätzlich auf die neue Typzuordnung geprüft: drei ausgewählte globale
Attributabhängigkeitsinstanzen und 161 ausgewählte konstante Clustererwartungen.
Eine vollständige Zusatzanalyse dieses Windows-Laufs erfolgt erst mit dem Skript
auf seinen vorhandenen vollständigen Artefakten.

Dieser Nachweis betrifft technische Funktion und Konsistenz. Er belegt weder
reale Fehlererkennungsleistung noch einen allgemeinen Vorteil latenter Räume.
