# Festgelegter Vergleich 11: Regeln mit und ohne latente Gruppen

Dieses Protokoll wird mit den konkreten Parametern vor dem Training gespeichert.
Es ist ein erster interner Vergleich, keine Optimierung und kein Nachweis der
Gültigkeit einer fachlichen ALKIS-Regel. Die Datei wurde bereits explorativ in
Paket 10 betrachtet. Ein späterer unabhängiger Lieferungsbestand bleibt nötig.

## Drei Varianten

1. **Global:** ein Attributbaum je automatisch erkanntem Ziel.
2. **Attributcluster:** HDBSCAN auf der gemeinsamen aufbereiteten Attributmatrix;
   innerhalb jeder Trainingsgruppe ein Attributbaum oder eine konstante Aussage.
3. **Latentcluster:** der bestehende Autoencoder lernt dieselbe Attributmatrix;
   HDBSCAN gruppiert seine Darstellung; dieselbe lokale Regelsuche folgt.

Bei allen Varianten sind Zielauswahl, Ausgangsattribute, Stichprobe, Aufteilung,
Baumparameter und Güteschwellen gleich. Clusterregeln haben ausdrücklich eine
zusätzliche Bedingung: die Zuordnung durch das gespeicherte Clustermodell.
Sie dürfen nicht als globale IF/THEN-Regeln importiert werden. HDBSCAN-Rauschen
(-1) erhält keine lokale Regel und keinen versteckten globalen Ersatz.

## Eingabe und Aufteilung

Der unveränderte Attribute-Adapter aus Paket 10 liest nur seine freigegebenen
Spalten. Geometrie, Koordinaten und daraus abgeleitete Merkmale werden weder
eingelesen noch verwendet. gml_id dient ausschließlich der Identität.
Standard: dieselben 5.000 nach Kennung/Seed 42 ausgewählten Objekte wie Paket 10.
Identische vollständige Attributkombinationen bleiben gemeinsam in einer
Teilmenge. Das sichert keine räumliche, zeitliche oder lieferungsbezogene
Unabhängigkeit. 60/20/20 Prozent beziehen sich auf diese Gruppen, nicht Zeilen.
Standardmäßig werden drei Aufteilungen mit Seeds 42, 43, 44 untersucht. Es sind
wiederholte Aufteilungen desselben Bestands, keine drei unabhängigen Datensätze.

## Schutz vor Zirkelschlüssen

Für jedes automatisch im Training gewählte Ziel wird ein eigener Encoder und
je ein eigenes Clustermodell trainiert. Die Zielspalte wird zuvor ausgeschlossen.
Die Familie Prüfdatum/Prüfdatum-fehlt wird gemeinsam ausgeschlossen, wenn eines
dieser Merkmale das Ziel ist. Im Training exakt gleiche Zielkopien werden ebenfalls
ausgeschlossen. Andere korrelierte Attribute bleiben zulässige Prädiktoren.
Der Test verändert weder Rollen, Prädiktoren, Encoder, Cluster, Baum noch Auswahl.
Die Zielspalte wird bei Prüfdatensätzen zusätzlich probeweise geändert: Darstellung,
Gruppenzuordnung und Regelvorhersage müssen unverändert bleiben.

## Repräsentation und feste Parameter

Codes und Codelisten werden als Kategorien behandelt, nicht als Zahlen oder
Zeichenfolgen mit behaupteter Semantik. Kategoriale fehlende Werte bekommen eine
eigene Kategorie; unbekannte Kategorien werden beim eingefrorenen One-Hot-Encoder
ignoriert und gezählt. Prüfdatum wird als UTC-Tage seit 1970 dargestellt. Numerische
Werte werden im Training mit Median imputiert und standardisiert. Die Attributbäume
arbeiten auf denselben Quellmerkmalen mit ihrer vorhandenen Median-/One-Hot-
Vorverarbeitung. Konstante Merkmale ohne Fehlstellen und ganz leere Spalten werden
entfernt; eine ansonsten konstante Kategorie mit Fehlstellen bleibt informativ.
Freitext mit mehr als 20 Trainingsausprägungen wird für alle drei Varianten
ausgeschlossen. Neue Spalten werden nicht ungeprüft übernommen.

Das ist eine offen ausgewiesene **tabellarische Variante** des vorhandenen
Autoencoders. Kein Zeichenencoder: alle Methoden erhalten dieselben Informationen.
Standard: 8 latente Dimensionen, 5 Epochen, vorhandene Rekonstruktions- und
Kontrastivgewichte (1 und 0,1), übrige Architektur aus LatentV2Config. Kein UMAP-
Clustering. HDBSCAN für beide Darstellungen: euklidisch, EOM,
min_cluster_size=50, min_samples=10, prediction_data=True. Prüfzeilen werden
mittels approximate_predict den eingefrorenen Trainingsclustern zugeordnet;
es wird kein Clustering des Gesamtbestands vorgenommen.

Regelbäume: feste Tiefe 4, min_samples_leaf=5. Keine Parametersuche/CV; dies
vermeidet die erneute Anpassung einer gesamten Clusterpipeline in inneren Folds.
Der vorhandene Baumaufbau, Bedingungsexport und Regelprüfer werden verwendet.
Auch ein konstanter Zielwert kann ein Kandidat sein (global oder in einer Gruppe),
muss aber dieselben Prüfbedingungen erfüllen. Mindestens 50 Trainingszeilen mit
Zielwert und 3 unterschiedliche Trainingsgruppen pro Modell.

## Auswahl und einmalige interne Testauswertung

Modellschwellen auf vorhandenen Validierungszielwerten wie bisher: Accuracy ≥0,95,
Macro-F1 ≥0,90, Macro-Precision ≥0,90. Jede ausgewählte Regel muss mindestens
5 anwendbare Validierungszeilen aus mindestens 5 Gruppen und mindestens 95 Prozent
Übereinstimmung erreichen. Fehlende Zielwerte werden getrennt ausgewiesen und
zählen für diese konservative Auswahl nicht als Bestätigung. Das macht ein
optional fehlendes Fachattribut nicht zu einem fachlichen Fehler.

Erst wenn die Auswahl für **alle Ziele und alle Seeds** mit Prüfsumme gespeichert
ist, wird der Test transformiert, zugeordnet und ausgewertet. Keine Nachauswahl,
kein Wiederanlernen. Eine ausgewählte Regel gilt im Bericht nur dann als im Test
erneut gestützt, wenn sie auch dort ≥5 Fälle, ≥5 Gruppen und ≥95 Prozent
Übereinstimmung erreicht. Zu wenig Belege heißt unzureichend prüfbar. Widersprüche
werden dokumentiert, nicht durch Löschen ungünstiger Regeln beseitigt.

Berichtet werden für dieselben Ziel-Datensatz-Paare: Abdeckung, Übereinstimmung,
Abweichungen bei vorhandenem Ziel, fehlende Ziele, Konflikte und Enthaltungen;
zusätzlich gruppengemittelte Werte, damit wiederholte Kombinationen sichtbar sind.
Paarweise werden zusätzliche korrekte und zusätzliche abweichende Vorhersagen
gegenüber der jeweiligen anderen Methode ausgewiesen. Regelanzahl, Clusterzahl
oder bessere Trainingsverluste allein begründen keinen latenten Mehrwert.
Es gibt keine echten Fehlerlabels und in diesem Vergleich keine Fehlerinjektion.
Ein positives Ergebnis bezieht sich auf diese Konfiguration und diesen Bestand;
ein negatives Ergebnis wird ebenfalls vollständig ausgegeben.
