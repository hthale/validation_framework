# Lokaler Probelauf mit den gelieferten Bahnverkehr-Attributen

Am 23.09.2026 wurde der vollständige Vergleich mit der unveränderten Datei
`bahnverkehr.parquet` ausgeführt. Quellbestand: 47.245 Objekte. Stichprobe: dieselben
5.000 Kennungen wie in Paket 10, Stichprobenseed 42. Es wurden keine Geometrien
gelesen oder abgeleitete Geometriemerkmale verwendet.

Original-Prüfsumme (SHA-256):
`7658f61d9b9b0d4778923eb95aae6ac0098f391380e9250511dbd152ef2adfb9`

Der abgeschlossene Lauf heißt `20260923_125453_590179`. Drei Aufteilungen × sechs
automatisch im Training gewählte Ziele ergaben **18 tatsächlich trainierte
Autoencoder**, jeweils fünf Epochen mit acht latenten Dimensionen. HDBSCAN und
Regelsuche wurden für beide Darstellungen wirklich ausgeführt. Es kamen weder
vorgegebene fachliche Regeln noch Attrappen zum Einsatz.

## Vergleich auf zurückgehaltenen Fällen

„Abdeckung“ zählt Ziel-Datensatz-Paare. „Übereinstimmung“ ist der Anteil bestätigender
Werte an den abgedeckten Paaren; fehlende Ziele bleiben im Nenner. Das ist keine
gemessene Fehlererkennungsrate. Die Seeds sind wiederholte Aufteilungen desselben
Bestands und werden nicht als unabhängige Studien zusammengezählt.

| Seed | Variante | Ausgewählte Regeln | Davon im Test gestützt | Abdeckung | Übereinstimmung |
|---|---|---:|---:|---:|---:|
| 42 | Global | 1 | 1 | 16,63 % | 100,00 % |
| 42 | Attributcluster | 20 | 14 | 13,09 % | 99,20 % |
| 42 | Latentcluster | 28 | 25 | 20,08 % | 99,65 % |
| 43 | Global | 1 | 1 | 16,63 % | 100,00 % |
| 43 | Attributcluster | 30 | 21 | 21,87 % | 99,92 % |
| 43 | Latentcluster | 35 | 21 | 22,03 % | 99,53 % |
| 44 | Global | 1 | 1 | 16,65 % | 100,00 % |
| 44 | Attributcluster | 21 | 15 | 15,31 % | 99,56 % |
| 44 | Latentcluster | 23 | 16 | 15,73 % | 99,46 % |

Die globale Auswahl betrifft die Belegung des Prüfdatums. Lokale Kandidaten wurden
außerdem für `ergebnisderueberpruefung` und `processstep_ax_datenerhebung` ausgewählt.
Für `anlass`, `bahnkategorie` und `funktion` bestand in keiner der drei Varianten
und Aufteilungen eine Regel die festgelegten Auswahlbedingungen. Diese erfolglosen
Ziele bleiben in der Auswertung enthalten.

Ein konkretes Beispiel aus Seed 42: Für `ergebnisderueberpruefung` deckten die
ausgewählten Attributcluster-Regeln 232 Testfälle ab; die Latentcluster-Regeln 336.
Bei beiden stimmten in diesen Fällen die vorhandenen Werte mit der Vorhersage
überein. Für dieses Ziel hatte die globale Regelsuche keine ausgewählte Regel.
Das zeigt einen konkreten Beitrag der lokalen Suche und in dieser Aufteilung eine
größere Abdeckung durch die latente Gruppierung. Es ist keine fachliche Freigabe.

Über alle Ziele liefert die latente Variante hier bei jedem Seed mehr Abdeckung
als die Attributcluster-Variante. Bei Seed 43 und 44 ist ihre Übereinstimmungsrate
etwas niedriger. Zusätzliche korrekt abgedeckte Paare dürfen deshalb nicht ohne
die zusätzlichen Abweichungen und verlorenen Fälle beurteilt werden. Viele
Kandidaten sind lokale konstante Aussagen; mehr solche Aussagen allein wären
kein Beleg für komplexere fachliche Abhängigkeiten.

Der Befund gilt für die festgelegte tabellarische Variante mit acht Dimensionen,
fünf Epochen und den vorgegebenen Cluster-/Baumparametern. Die Anzahl lokaler
Modelle hängt von der gefundenen Clusterstruktur ab. Ein Vergleich mit anderer
Dimensionsreduktion oder eine Optimierung ist hier nicht enthalten. Er bewertet
nicht rückwirkend die frühere 64-dimensionale Darstellung. Eine neue externe
Lieferung und fachliche Prüfung sind für weitergehende Aussagen notwendig.

## Technische Absicherung

- **13 gezielte Funktionstests bestanden:** unter Pandas 3.0.1 und Pandas 2.2.3.
  Darunter bekannte Abhängigkeit/Gegenbeispiel, uninformative Kontrollgröße,
  lokale Reichweite, Rauschen, Zielkopien und direkte Ableitungen, Mindestanzahl
  an Gruppen, fehlende Werte, eingefrorene Vorverarbeitung und gespeicherte
  Prüfsummen.
- **800 Prüfbedingungen des vollständigen Laufs bestanden:** wiederholte
  Integritäts-/Evidenzprüfungen über Ziele, Methoden und ausgewählte Regeln;
  dies sind keine 800 unabhängigen Leistungsversuche.
- **54 Zielmutationsprüfungen bestanden:** drei Methoden × sechs Ziele × drei
  Seeds. Änderungen des Zielwerts verändern die Prädiktormatrix und die
  Vorhersagen nicht. Die bekannte Datumsfamilie wird ebenfalls ausgeschlossen.
- Die separat startbare Wiederholungsprüfung rechnet sämtliche gespeicherten
  Regelbelege unter Pandas 2 erneut nach und prüft alle Export-Prüfsummen.
- HTML-Verweise auf neun Ergebnisdateien sind gültig. Python-3.11-Syntax wurde
  für alle fünf Python-Dateien geprüft. Kein tatsächlicher Windows-/Python-3.11-
  Lauf und keine automatisierte Browserbedienung wurden durchgeführt.
- Original-Parquetdatei und bestehender Projektcode blieben unverändert.

Bei den vorbereitenden Technikläufen wurden ein lokaler Importpfad des bestehenden
Autoencoders, die Gleichheitsprüfung leerer Pandas-/Arrow-Textwerte und ein
Namenskonflikt der Eingabe-/Quellcode-Prüfsummen korrigiert. Danach wurde der Lauf
mit unveränderten wissenschaftlichen Parametern vollständig wiederholt.
Güteschwellen, Merkmale und Auswahl wurden nach Sichtung der Testwerte nicht
angepasst. Die abschließenden Kennzahlen stimmen mit der vorherigen Berechnung
überein; die abschließende separate Wiederholungsprüfung ist ebenfalls enthalten.

## Umgebung und beigefügte Belege

Linux, Python 3.12.14, NumPy 2.3.5, Pandas 3.0.1, Scikit-learn 1.8.0,
PyTorch 2.8.0+cpu, PyArrow 23.0.1, HDBSCAN 0.8.44.

`belege_lokaler_probelauf/bericht.html` zeigt den lokalen Bericht. Die dortigen
JSON-Dateien enthalten Regeln, Kennzahlen, Prüfprotokoll und Versionen als Auszug
des vollständigen Laufs. Große Modell-/Matrixdateien und Originaldaten sind nicht
mitverpackt. Dieser Belegauszug ist kein vollständiger, erneut ausführbarer
Ergebnisordner für `verify_rule_comparison.py`. Dafür den eigenen vollständigen
Lauf verwenden. Versions- und Plattformunterschiede können Ergebnisse verändern.
