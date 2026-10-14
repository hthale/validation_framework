# Festgehaltener Projektstand · 23.09.2026

## 1. Sind schon Fehler in den Daten?

**Ja, in gesonderten Testdaten wurden gezielt Änderungen und synthetische Fehler erzeugt. Die realen Originaldateien wurden dabei nicht überschrieben.** Für DOP und ALKIS liegt weiterhin keine systematisch fachlich geprüfte reale Testmenge mit vollständigen Fehlerlabels vor. Unverändert bedeutet deshalb nicht nachweislich fehlerfrei.

| Untersuchung | Was wurde verändert? | Was dürfen wir daraus ableiten? |
|---|---|---|
| Ursprüngliche DOP- und Bahnverkehr-Bestandsanalysen, Pakete 07/10; Regelvergleich 11 | Keine gezielten Änderungen der Originalwerte | Kandidaten, Verteilungen, Cluster, Abdeckung und Übereinstimmung mit beobachteten Werten |
| DOP-Vergleiche 08/09 | Attribute in separaten Testkopien; Paket 09 verwendet acht Änderungsarten mit je drei Fällen pro Seed | Empfindlichkeit gegenüber Änderungen; teilweise Verletzung ausdrücklich festgelegter Testannahmen, keine allgemeine reale Fehlerquote |
| Bahnverkehr-Zusatzanalyse 12 | 2.082 Original-/Änderungspaare über die wiederholten Aufteilungen: häufigster Wert zu anderem Wert, Gegenrichtung und fehlender Wert | Reaktion der vorhandenen Regeln auf Veränderungen. Ein anderer ALKIS-Code kann fachlich zulässig sein |
| Kontrollierter Vergleich 13 | Künstliche Daten mit vorab festgelegter Gültigkeit; Prüfcode oder ein Messwert wird in einer separaten Testvariante widersprüchlich geändert | Fehlererkennung und Fehlmeldungen relativ zu diesem bekannten künstlichen Mechanismus |
| Frühere Funktionstests 05/06/07 | Künstliche bekannte Zusammenhänge und Gegenbeispiele | Technische Funktion der Regelableitung und Integration |

Paket 13 verwendet pro Seed 2.400 Zeilen aus 1.200 Einheiten. Training, Validierung und Test sind nach Einheiten getrennt. Der Test enthält 480 Zeilen. In jeder fehlerhaltigen Variante sind 48 Zeilen verändert und 432 weiterhin gültig; zusätzlich gibt es eine vollständig gültige Kontrollvariante. Die Fehler gelangen nicht als Trainingslabels in die Modelle. Die Trainingsaugmentation des Autoencoders ist davon getrennt: Sie erzeugt verrauschte Eingabesichten als Lernaufgabe und liefert keine fachlichen Fehlerlabels.

Die beiden Fehlertypen sind getrennte Testvarianten desselben Bestands. Die zwei Szenarien mit und ohne Zusatzmerkmale teilen sich ebenfalls ihre zugrunde liegenden Fälle je Seed. Diese Auswertungen dürfen nicht zu unabhängigen Datensätzen oder Studien aufaddiert werden.

## 2. Was funktioniert derzeit?

Der gemeinsame Einstieg `src/run_generic.py` analysiert eine flache Parquet-Merkmalstabelle. Er erzeugt bzw. erhält Kennungen, bestimmt technische Merkmalsrollen und bereitet Merkmale auf. Ein Autoencoder lernt eine Darstellung; LOF priorisiert Auffälligkeiten und HDBSCAN bildet Gruppen. UMAP dient der Darstellung.

Daneben entstehen mehrere unterschiedliche Ergebnisse:

| Ergebnisart | Entstehung | Bedeutung |
|---|---|---|
| Profiling-Kandidaten | Wiederholte Stichproben, Verteilungen und Einzelspaltenmuster | Beobachtete Eigenschaften wie Wertebereiche, Belegung oder Kategorien |
| Globale Abhängigkeitskandidaten | Entscheidungsbäume auf Originalattributen; getrennte Trainings-/Validierungsteilmenge | Bedingte Erwartungen an Zielattribute |
| Clusterbeschreibungen | Bäume erklären bereits gebildete Cluster durch Attribute | Beschreiben die Gruppen; sind keine automatisch freigegebenen Qualitätsregeln |
| Clusterbezogene Abhängigkeitskandidaten | Zusätzliche Vergleichsprogramme 11/13 lernen Regeln innerhalb eingefrorener Gruppen | Erwartungen mit obligatorischem Cluster-Geltungsbereich |
| Auffälligkeitsrang | LOF auf einer festgelegten Darstellung | Reihenfolge für eine Prüfung, keine automatische fachliche Fehlerentscheidung |

**Der latente Raum trägt zur Clusterbildung, zu deren Beschreibung, zum LOF und indirekt zur Stichprobenbildung des Profilings bei.** Die globale Attributregelerzeugung ist ein eigener Lernpfad. Die lokalen Abhängigkeitsvergleiche sind bisher zusätzliche Programme; sie laufen nicht automatisch als Teil von `run_generic.py`.

Der generische Einstieg lernt Autoencoder und Cluster auf dem gesamten eingegebenen Bestand. Auch Profiling und Clusterbeschreibungen beziehen sich auf diesen Bestand. Nur sein Abhängigkeitslernen besitzt eine getrennte Aufteilung und einen dabei unbenutzten Testanteil. Die kontrollierten Vergleiche 09/11/13 verwenden dagegen auf Training angepasste Modelle und getrennte Auswertung. Die Aussage „das gesamte Framework wurde unabhängig getestet“ wäre für den generischen Bericht allein falsch.

## 3. Ergebnisse, die wir für die Arbeit verwenden können

### DOP: Anwendungsbeispiel und explorativer Änderungsversuch

Der von dir protokollierte generische Lauf mit 500 Zeilen erzeugte 50 Abhängigkeitskandidaten, davon keinen mit bestandenen Güteschwellen. Hinzu kamen 83 Profiling-Kandidaten und zunächst zehn Clusterbeschreibungen bei 18 Clustern. Diese Zahlen sind Laufzahlen und keine Anzahl bestätigter Fachregeln.

Die nachträgliche Clusterbeschreibung aus Paket 08 dokumentiert 17 Beschreibungen und etwa 90,3 % Übereinstimmung mit vorhandenen Clusterlabels auf 433 berücksichtigten Fällen. Das vollständige Beschreibungsziel wurde trotzdem nicht erreicht. Das ist Anpassungsgüte am selben Bestand, keine unabhängige Fehlererkennung.

Im hochgeladenen LOF-Vergleich 08 lag die mittlere Trefferquote künstlicher Änderungen beim Eingangsraum bei etwa 15,8 % und beim latenten Raum bei 8,6 %. Der gemeinsame Versuch 09 ist ein eigenes Protokoll: Dort lagen die Mittelwerte bei 72,2 % für die vereinigten Regeln, 80,6 % für Regeln plus Eingangs-LOF und 75,0 % für Regeln plus latenten LOF. Die Vereinigungen können einen unterschiedlichen Prüfaufwand verursachen. Protokolle 08/09 sind wegen verschiedener Aufteilungen und Änderungen nicht unmittelbar als Entwicklungsfortschritt vergleichbar.

### ALKIS-Bahnverkehr: Attribute, Regeln und ihre Reichweite

Aus 47.245 Quellobjekten wurden 5.000 ausgewählt. Geometrien, Koordinaten und daraus abgeleitete Merkmale bleiben außerhalb dieser Untersuchung. Der Adapter bereitet ausgewählte Fachcodes, Codelisten, Texte und das Prüfdatum auf.

Die Pakete 11/12 dokumentieren insgesamt 1.237 Kandidaten und 164 ausgewählte Regelinstanzen über drei wiederholte Aufteilungen: drei globale Abhängigkeiten, 75 konstante Erwartungen in Attributclustern und 86 konstante Erwartungen in Latentclustern. Das sind keine 164 unabhängigen fachlichen Zusammenhänge.

Auf den jeweils abgedeckten Fällen weichen die Vorhersagen keiner der drei Regelmethoden von der globalen Trainingsmehrheit ab. Der bisherige Unterschied liegt damit vor allem im Geltungsbereich und in der Abdeckung. Eine konstante Erwartung innerhalb eines Clusters behält ihre Clusterbedingung; sie darf nicht als unbedingte Regel auf alle Objekte angewendet werden. Fachliche Fehlererkennungsleistung ist hier nicht belegt.

### Künstlicher Vergleich 13: bekannter Fehlermechanismus

Im Szenario mit zwölf zusätzlichen, uninformativen Merkmalen wurden je Seed 48 Fehler durch widersprüchliche Prüfcodes erzeugt:

| Regelmethode | Seed 42: erkannt/48 | Seed 43: erkannt/48 | Seed 44: erkannt/48 | Fehlmeldungen auf je 432 gültigen Fällen |
|---|---:|---:|---:|---|
| Globale Regeln | 48 | 48 | 48 | 0 / 0 / 0 |
| Regeln in Attributclustern | 45 | 43 | 38 | 0 / 0 / 0 |
| Regeln in Latentclustern | 48 | 48 | 47 | 0 / 0 / 0 |

Die Abdeckung beträgt bei Attributclustern etwa 92,9 / 89,6 / 78,3 %, bei Latentclustern 99,6 / 99,6 / 97,9 % und bei globalen Regeln jeweils 100 %. Nicht abgedeckte Fehler zählen als nicht erkannt. Bei den seltenen gültigen Fällen decken Attributcluster in diesem Szenario 0 von 22 je Seed ab; Latentcluster 20 / 20 / 15 von 22, globale Regeln jeweils alle 22. Das zeigt eine verbleibende Einschränkung lokaler Verfahren.

**Belegter begrenzter Nutzen:** Latentcluster sind in diesem künstlichen Szenario für lokale Regeln robuster gegenüber den Zusatzmerkmalen als Attributcluster. Gegenüber dem globalen Regelbaum entsteht kein Vorteil. Ohne Zusatzmerkmale finden alle drei Regelmethoden sämtliche Fehler. Die uninformative Kontrollgröße wird überall abgewiesen.

LOF ergibt ein anderes Bild. Bei gleichem Budget von 48 ausgewählten Fällen pro Seed und Fehlertyp ergeben sich im Szenario mit Zusatzmerkmalen folgende Summen über die drei Seeds:

| LOF-Darstellung | Prüfcodefehler erkannt / 144 | Messwertfehler erkannt / 144 |
|---|---:|---:|
| Aufbereitete Attribute | 56 | 79 |
| PCA | 45 | 126 |
| Autoencoder | 15 | 41 |

Die Auswahl umfasst je Spalte und Methode ebenfalls 144 Fälle. Die übrigen ausgewählten Fälle sind unter dem künstlichen Gültigkeitsmodell gültig. Hier ist kein Vorteil des Autoencoders nachgewiesen. Regelaufgabe und LOF verwenden bewusst getrennte Darstellungen: Das Regelziel ist aus den Regelprädiktoren und deren Clustering ausgeschlossen; LOF betrachtet auch den beobachteten Prüfcode.

## 4. Was die Masterarbeit beanspruchen kann

Verwendbar sind die Konzeption und prototypische Umsetzung einer automatisierten Analyse von Merkmalstabellen, die nachvollziehbare Ableitung statistischer Regelkandidaten, die Priorisierung von Auffälligkeiten sowie der untersuchte Beitrag unterschiedlicher Repräsentationen. Positive und negative Befunde gehören gleichermaßen in die Evaluation.

Noch nicht belegt sind ein allgemeiner Vorteil des latenten Raums, eine vollständige automatische Validierung beliebiger Rohprodukte, fachlich garantierte Regeln und eine reale Fehlererkennungsrate für DOP oder ALKIS. Der vorhandene Bahnverkehr-Adapter zeigt eine bewusst festgelegte Produktaufbereitung; er ist keine automatische Erkennung der Bedeutung beliebiger Attribute.

Weitere Grenzen für die Umsetzung: Die lokale Fallentscheidung ist vorhanden, aber kein vollständig integrierter Freigabeprozess für Regelversionen. Die generische Berichtsausgabe und die ältere Prüfansicht sind unterschiedliche Einstiege. Die aktiven Versuche schreiben keine Delivery Memory; damit wurde keine Verbesserung über mehrere historische Lieferungen nachgewiesen. Die vorhandenen Zeichenencoder sind einfache Zeichenmittelungen und keine Sprachmodelle; Zeichenreihenfolgen werden durch diese Mittelung nicht unterschieden. NAS-/TIFF-/XML-Import und Geometrievalidierung sind durch die hier ausgewerteten Tests ab Parquet nicht nachgewiesen.

## 5. Festlegung für den nächsten Arbeitsschritt

Zunächst den dokumentierten Stand sichern und Kapitel 5 auf dieser Grundlage beginnen. Globale Regeln bleiben der primäre Abhängigkeitspfad; latente Cluster werden als untersuchte Ergänzung dargestellt. Der direkte Attributvergleich und PCA bleiben als Vergleichsmethoden sichtbar. Keine nachträgliche Schwellenabsenkung oder Auswahl ausschließlich günstiger Seeds.

Eine spätere fachliche Fehlerstudie würde einen separat dokumentierten Prüfbestand, eindeutige Sollvorgaben, ein Änderungsprotokoll und eine fachlich geprüfte Stichprobe unveränderter Fälle benötigen. Das ist eine mögliche Erweiterung, keine Voraussetzung dafür, morgen die bereits umgesetzten Komponenten zu beschreiben.

## Herkunft und Reichweite dieser Notiz

Zahlen aus lokalen Läufen wurden den hochgeladenen Dateien `summary.json`, `summary(1).json` bis `summary(6).json`, `rule_metrics.json` und `lof_metrics.json` sowie den Konsolenprotokollen entnommen. `ergebnisbelege.json` hält die Dateiprüfsummen und ausgewählte Kennzahlen fest. Die drei zuletzt hochgeladenen Ergebnisdateien wurden auf konsistente Zähler geprüft; damit wurde der gesamter Windows-Lauf nicht erneut ausgeführt. Ergebnisse der eigenen Linux-Probeläufe werden nicht an ihre Stelle gesetzt.

Die Codebeschreibung beruht auf der hier verfügbaren Arbeitskopie des Kerns 07/08 und den Zusatzprogrammen bis 13. Änderungen, die du nur lokal vorgenommen hast, werden erst mit der neuen Bestandsaufnahme sichtbar. Der Schreibplan bezieht sich auf die hochgeladene PDF-Fassung `6039258_Masterarbeit_Hendrik_Thale(5).pdf`.
