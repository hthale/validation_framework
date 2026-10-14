# Kontrollierter Attributvergleich 13

Dieses Protokoll wird vor dem ersten Modelltraining als Teil des Laufs gespeichert.
Es legt einen neuen synthetischen Versuch fest. Frühere reale DOP-/ALKIS-Ergebnisse
bleiben erhalten. Es gibt keine nachträgliche Optimierung anhand des Tests und
keine Bedingung, dass der Autoencoder gewinnen muss.

## Künstlicher Datenmechanismus

Pro Seed 1.200 Einheiten, zwei Beobachtungen je Einheit. Drei gültige Gruppen:
A (70 %), B (25 %), C (5 %). Der Prüfcode lautet jeweils A, B oder C. Vier rein
tabellarische Messwerte liegen unabhängig gleichverteilt im Intervall Zentrum
+/-0,6. Zentren: -4 für A, 0 für B und 4 für C. Dies sind keine Koordinaten,
Polygone oder Geometriemerkmale und keine fachliche Produktspezifikation.

In Szenario `basis` werden diese vier Werte verwendet. `stoermerkmale` ergänzt
zwölf unabhängige gleichverteilte Werte zwischen -1 und 1. Sie besitzen keine
Beziehung zum Prüfcode. Für denselben Seed bleiben die Signalwerte und Einheiten
zwischen beiden Szenarien identisch. Die Zusatzmerkmale werden für beide
Darstellungen gleichermaßen zugelassen; ihre bekannte Irrelevanz wird dem
Framework nicht mitgeteilt.

Die beiden Beobachtungen einer Einheit besitzen identische fachliche Attribute.
Das separate Kontrollziel ist einmal K0 und einmal K1. Es ist somit für identische
Prädiktoren exakt ausgeglichen und soll keine ausgewählte Abhängigkeit erzeugen.
Kennungen, Gruppennamen, Partition, Normalgruppentyp, seltene-Gruppe-Markierung,
Fehlerlabels und Beobachtungsnummer sind niemals Modellmerkmale. Referenzmetadaten
werden getrennt von der Attributtabelle gespeichert.

Die vorhandene Zielerkennung wird anhand der Trainingsdaten ausgeführt. Die
Evaluation betrachtet die vorab benannten Ziele `pruefcode` und `kontrolle`.
Der Versuch beansprucht keine neue Prüfung universeller Merkmals-/Zielauswahl.

## Aufteilung und Parameter

60/20/20 % der Einheiten pro Normalgruppe gehen in Training, Validierung und Test.
Beide Beobachtungen bleiben zusammen. Quoten werden deterministisch gerundet;
Aufteilung innerhalb der Gruppen ist zufällig. Seeds 42, 43, 44 erzeugen neue
Ziehungen desselben Generators. Es sind keine drei realen Produkte/Lieferungen.

Standard: 25 Epochen, latente Dimension 3, vorhandener tabellarischer Autoencoder
mit unveränderten übrigen Trainingsgewichten/Augmentationen. Dimension 3 ist
vorab gewählt, um auch die vierdimensionalen Basismerkmale tatsächlich zu
komprimieren; sie ist keine auf dem Test gefundene optimale Einstellung.
CLI-Abweichungen werden dokumentiert; Ergebnisse verschiedener Konfigurationen
sind getrennte Versuche, nicht austauschbare Wiederholungen.

Regelparameter unverändert aus Paket 11: Tiefe 4, mindestens 5 Blattfälle,
mindestens 50 beschriftete Trainingsfälle aus 3 Gruppen pro Modell; mindestens
5 Validierungsfälle aus 5 Gruppen und 95 % Übereinstimmung pro Regel; Modell-Gate
Accuracy >=0,95, Makro-Präzision >=0,90, Makro-F1 >=0,90. HDBSCAN: min_cluster_size
50, min_samples 10, eom/Euclidean. Neue Fälle verwenden eingefrorene Router.
Rauschen -1 wird von lokalen Regeln nicht abgedeckt.

Die gesamte Auswahl über alle Seeds/Szenarien wird vor der ersten Testauswertung
festgeschrieben. Alle Kandidaten, Ablehnungsgründe, ausgewählten Regeln, Modelle,
Aufteilungen und Prüfsummen werden gespeichert. Der reservierte Test beeinflusst
weder Merkmalsstatistiken noch Training, Auswahl, Schwellen oder Parameter.

## Regelaufgabe

Globale Bäume, Bäume in Attributclustern und Bäume in Latentclustern verwenden
dieselben Messwerte (ggf. plus Zusatzmerkmale). Weder Prüfcode noch Kontrollziel
sind hier Prädiktoren oder Clustermerkmale. Ein gemeinsamer eingefrorener Router
pro Szenario/Seed ist für beide Zielprüfungen zulässig, da beide Ziele aus seinem
identischen Merkmalsvertrag ausgeschlossen sind.

Eine zusätzliche ungefilterte Baseline sagt stets den häufigsten nichtfehlenden
Trainings-Prüfcode voraus; lexikalischer JSON-Gleichstandentscheid. Sie ist kein
fachlich freigegebenes Regelmodell. Globale/clusterbezogene konstante Erwartungen
und zusätzliche Attributbedingungen bleiben als verschiedene Regeltypen sichtbar.

Ein vollständig gültiger Testbestand wird ausgewertet. Zusätzlich werden in
20 % der Testeinheiten (je Normalgruppe über feste Hash-Reihenfolge ausgewählt)
jeweils die ersten Beobachtungen verändert: A->B, B->C, C->A. Die zweite Beobachtung
bleibt gültig. Dadurch entstehen rund 10 % fehlerhafte Testzeilen. Es werden auch
seltene gültige Werte in den häufigsten Wert geändert. Alle Methoden erhalten
identische Kennungen, Fälle und Fehler. Keine Auswahl anhand von Score/Abdeckung.

Die Zieländerung darf weder Eingabematrix noch Geltungsbereich der Regel verändern.
Regeln werden mit dem bestehenden Framework-Regelprüfer ausgeführt. Enthaltung
zählt im Nenner aller Fehler als nicht erkannt. Berichtet: Abdeckung, synthetischer
Recall, Precision, Fehlmeldungen auf gültigen Fällen, seltene gültige Fälle und
deren Abdeckung; zusätzlich gruppengemittelter Fehlalarmanteil. Nur für diesen
Generator sind Originale als gültig und erzeugte Widersprüche als Fehler bekannt.

## Getrennte LOF-Aufgabe

LOF sieht die fachlichen Messwerte und den beobachteten Prüfcode. Das unabhängige
Kontrollziel ist ausgeschlossen. Hier soll das Modell die gemeinsame Verteilung
der vollständigen Attribute prüfen. Daher wird ein eigener Autoencoder gelernt;
die Darstellung der Regelaufgabe wäre für reine Zielwertänderungen blind.

Drei vorab festgelegte Darstellungen derselben Eingangsinformation:

1. Trainingsmedian/StandardScaler für Zahlen, One-Hot-Kodierung für Prüfcode.
2. PCA auf dieser Matrix mit Dimension 3, Fit ausschließlich im Training.
3. Der vorhandene Autoencoder mit Dimension 3, Fit ausschließlich im Training.

Keine nachträgliche Skalierung von PCA/Latentkoordinaten, keine Gewichtssuche.
LOF mit 20 Nachbarn, Euclidean, novelty=True; Fit nur auf normalen Trainingsdaten.
Bewertung mit -score_samples ausschließlich auf unbekannten Validierungs- und
Testeinheiten. Das ist ein induktiver Test mit fester Referenz, kein erneutes
transduktives Fitten auf einem Gemisch normaler und fehlerhafter Prüfdaten.

Getrennte Fehlerbestände:

- Prüfcodeänderungen wie oben;
- `messwert_b` wird bei denselben betroffenen Einheiten durch einen tatsächlich
  im Training vorkommenden Wert der nächsten Normalgruppe ersetzt. Andere Werte
  und Prüfcode bleiben erhalten. Univariate Werte sind bekannt, ihre Kombination
  verletzt die vorab festgelegten Signalintervalle.

Primärer Vergleich: gleiches Prüfbudget ceil(10 % der Testzeilen), absteigender
Score, Gleichstand nach zielunabhängigem Hash der Kennung. Das Budget ist kein
automatisches Fehlerlabel. Precision/Recall@Budget, Fehlmeldungen auf gültigen
Fällen und seltene gültige Fälle werden gemeinsam berichtet. AP und ROC-AUC
bewerten die Rangfolge zusätzlich; für reine Normalbestände ohne Fehler ist AP
nicht definiert. Für den vollständig normalen Bestand führt ein festes Budget
zwangsläufig zur Auswahl normaler Fälle.

Sekundär: feste Schwelle am oberen 95-%-Quantil normaler Validierungsscores
(Quantilmethode higher; Testscore strikt größer). Keine Testfehlerlabels zur
Schwellenwahl. Der erreichte Fehlalarmanteil auf dem Test wird gemessen und muss
nicht 5 % betragen. Die Schwelle ist methodenspezifisch, die Validierungsregel gleich.

API-Referenz (am 23.09.2026 geprüft):
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
Die Dokumentation verlangt bei novelty=True neue, nicht zum Fit gehörende
Beobachtungen für score_samples; die Scores können sich vom transduktiven LOF
unterscheiden. Die tatsächlich verwendeten Softwareversionen stehen im Manifest.

## Interpretation

Regelableitung und LOF beantworten unterschiedliche Fragen. Beide Ergebnisse
werden getrennt gegen ihre Vergleichsmethoden beurteilt. Güteunterschiede werden
pro Seed/Szenario ausgewiesen, nicht durch Auswahl des besten Laufs verborgen.
Ein Kompressionsgewinn, den PCA ebenfalls erzielt, ist kein eindeutiger Beleg
für einen speziell durch den Autoencoder bewirkten Vorteil.

Die künstlichen Labels belegen keine reale DOP-/ALKIS-Erkennungsleistung.
Ein positives Ergebnis gilt für die festgelegten Mechanismen; ein negatives
Ergebnis bleibt ein zulässiges Ergebnis. Der Versuch ersetzt weder fachliche
Regelfreigabe noch unabhängige reale Testlieferungen. Kein Geometry-/Dateiimport,
keine Änderung des Projekts oder vergangener Ergebnisse, keine Delivery-Memory-
Einträge, keine Browserautomatisierung.
