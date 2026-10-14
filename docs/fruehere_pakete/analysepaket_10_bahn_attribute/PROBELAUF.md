# Bereits ausgeführter Probelauf

Stand: 23.09.2026. Verwendet wurde die hochgeladene Originaldatei
`bahnverkehr.parquet` mit 47.245 Objekten. Keine Geometrie wurde als Merkmal
verwendet, keine Polygone wurden ausgewertet und keine Fehler wurden eingefügt.

| Prüfung / Ergebnis | Beobachtung |
|---|---:|
| Reproduzierbare Stichprobe, Seed 42 | 5.000 Objekte |
| Training tatsächlich durchgeführt | 5 Epochen |
| Verschiedene vollständige Attributkombinationen | 1.295 |
| Aufbereitete tabellarische Dimensionen | 49 |
| Zusätzlich genutzte Zeichenkettenfelder | 7 |
| Latente Dimensionen, bestehende Konfiguration | 64 |
| Globale Abhängigkeitskandidaten | 46 |
| Davon mit bestandenen Güteschwellen | 1 |
| Profiling-Vorschläge | 11 |
| HDBSCAN-Cluster | 223 |
| Als Rauschen zugeordnet | 361 Objekte |
| Blattregeln zur Clusterbeschreibung | 58 |
| Übereinstimmung des ausgewählten Beschreibungsbaums mit den Clusterlabels | rund 32,7 % |
| Im Beschreibungsbaum betrachtete Objekte | 4.639 |
| Abschließende technische Prüfungen | 15 bestanden |

Die Übereinstimmung beschreibt nur die Anpassung an vorhandene Clusterlabels,
keine Erkennungsleistung. Das bisherige Beschreibungsziel von 90 % wurde nicht
erreicht. Die vielen Cluster sind deshalb kein Beleg für einen erfolgreichen
latenten Raum. Unveränderte Standardparameter wurden beibehalten; sie wurden
nicht nach dem Ergebnis verändert, um ein besseres Resultat zu erzeugen.

Der einzige ausgewählte globale Abhängigkeitskandidat betrifft die Belegung
des Prüfzeitpunkts: Bei vorhandenem Prüfungsergebnis war in den 1.205 anwendbaren
Validierungsfällen auch der Prüfzeitpunkt vorhanden. Er stammt aus der globalen
Regelsuche auf Attributen. Auch dieses Ergebnis weist keinen Nutzen des latenten
Raums nach und ist keine bestätigte fachliche Pflichtregel.

Einige seltene Codeausprägungen fehlen in der Stichprobe. Beispielsweise wurde
`zustand=4000` nicht ausgewählt. Dadurch bleibt `zustand` in dieser Stichprobe
unter den belegten Werten konstant und wird vom bestehenden Encoder ausgelassen.
Alle Abdeckungsinformationen stehen in `verification/preparation.json`.

## Technische Validierung

- Zehn gezielte Adaptertests bestehen unter Pandas 2.2.3 und 3.0.1.
- Ein echter 5.000er-Lauf mit fünf Epochen wurde unter beiden Pandas-Versionen
  ausgeführt; Modell-/Regelparameter und Stichprobenauswahl blieben gleich.
- Der finale Paketstand wurde vollständig unter Pandas 3.0.1 ausgeführt.
- Dessen Artefakte und alle gespeicherten Ausgabeprüfsummen wurden anschließend
  mit dem finalen Verifier sowohl unter Pandas 2.2.3 als auch 3.0.1 erfolgreich geprüft.
- Original-Parquetdatei und vorhandener Projektcode sind unverändert.
- Die Code-Syntax wurde für Python 3.11 geprüft; die tatsächlichen Läufe verwendeten
  Python 3.12 unter Linux. Eine native Windows-Ausführung wurde hier nicht getestet.
- Die HTML-Verknüpfungen wurden statisch geprüft. Keine automatisierte Browserbedienung.

Die Dateien unter `verification` dokumentieren den Probelauf. Sie enthalten
ausgewählte Nachweise, nicht sämtliche trainierten Modelle oder alle Quelldaten.
Die Originaldatei wird im Paket nicht dupliziert. Ein eigener Lauf erzeugt die
vollständigen Ergebnisartefakte in Projekt.

## Wofür dieser Schritt dient

Der erste Lauf schafft eine nachvollziehbare Grundlage für die nächste Entscheidung:
Welche Attribute prägen die Gruppen, sind diese Gruppen sinnvoll beschreibbar,
und lohnt sich eine Erweiterung um gruppenspezifische Abhängigkeitsregeln?
Ein Vergleich mit denselben Attributen ohne Autoencoder sowie eine getrennte
Leistungsprüfung sind weiterhin ausstehend. Die Ergebnisse rechtfertigen keine
Aussage, dass der latente Raum die Validierung bereits verbessert.
