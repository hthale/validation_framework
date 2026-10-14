# Zusatzanalyse 12: Regelarten, Trainingsmehrheit und Zieländerungen

## Zweck und Ausgangslage

Explorative Folgeanalyse eines abgeschlossenen Vergleichslaufs aus Paket 11.
Die Idee wurde nach Sichtung seiner Ergebnisse festgelegt. Es handelt sich
deshalb nicht um einen neuen, unabhängigen oder vorab unberührten Testbestand.
Keine fachliche Wahrheit über die ursprünglichen ALKIS-Werte wird vorausgesetzt.
Nur bereits aufbereitete Attribute werden gelesen; Geometrien bleiben ungenutzt.

## Unveränderte Voraussetzungen

Eingabe, gespeicherte Aufteilungen, Merkmalsrollen, Autoencoder, HDBSCAN-Router,
Baumregeln, Schwellen und Auswahl aus Paket 11 bleiben unverändert. Die Prüfsummen
des alten Laufs und des zugehörigen Quellcodes werden vor und nach der Analyse
geprüft. Der alte Lauf muss vollständig abgeschlossen sein. Das Zusatzpaket
erzeugt einen neuen Ausgabeordner und keinen Eintrag im fachlichen Review.

## Regelarten

Die ursprünglichen Umschläge und ihre Auswahl bleiben erhalten. Für die Anzeige
wird jeder Kandidat einer von vier Klassen zugeordnet:

- globale Attributabhängigkeit: mindestens eine innere Baumbedingung;
- globale konstante Erwartung: keine innere Baumbedingung;
- Attributabhängigkeit innerhalb eines Clusters: Scope plus innere Bedingungen;
- konstante Erwartung innerhalb eines Clusters: Scope, keine inneren Bedingungen.

Die Clusterzuordnung ist weiterhin eine operative Bedingung. Ein konstanter
Zielwert innerhalb eines Clusters ist nicht gleichbedeutend mit einer global
unbedingten oder sinnlosen Regel. Instanzen über Seeds und verschiedene Cluster
werden ausdrücklich nicht als unabhängige unterschiedliche Fachregeln gezählt.

## Vergleich ohne Cluster

Pro ursprünglichem Seed und Ziel wird der häufigste nichtfehlende Zielwert nur
aus der Trainingspartition bestimmt. Bei Gleichstand entscheidet die lexikalische
Reihenfolge der kanonischen JSON-Texte. Häufigkeiten und Trainingskennungen werden
gespeichert und vor der Testauswertung per Prüfsumme festgehalten.

Diese einfache Methode sagt diesen Wert auf allen Fällen voraus. Sie unterliegt
keinem Modell-Gate und wird nicht als fachlich freigegebene Regel ausgegeben.
Sie ist ein bewusst uneingeschränkter Vergleich, während die alten Regelmethoden
auf einen Teilbestand beschränkt sind. Ihre Testabdeckung beträgt daher 100 %.

Berichtet werden Abdeckung, Übereinstimmung und Abweichung zu beobachteten Werten,
fehlende Zielwerte und Enthaltungen. Zusätzlich werden genau die von der jeweiligen
Regelmethode abgedeckten Fälle mit der globalen Mehrheit verglichen. Andere
Vorhersagen und zusätzliche Übereinstimmungen sind getrennt von selektiver
Abdeckung sichtbar. Makro-Präzision/-Recall/-F1 beziehen sich auf vorhandene
Zielwerte innerhalb der Abdeckung; sie ersetzen keine Abdeckungsangabe.

Der komplette ursprüngliche Testanteil bleibt unverändert als Kontrolle erhalten.
Er wird zusätzlich in Originalmehrheit, andere vorhandene Originalwerte und
fehlende Originalwerte aufgeteilt. Ein anderer Originalwert ist nicht zwingend
selten, und eine Abweichung davon ist kein bestätigter Fehlalarm.

## Kontrollierte Änderungen

Pro Seed/Ziel/Richtung höchstens 50 Referenzgruppen (CLI-Option --max-groups,
tatsächlicher Wert im Protokoll). Eine feste Hash-Reihenfolge bestimmt zuerst
einen Vertreter jeder Gruppe und danach die ausgewählten Gruppen. Die Auswahl
hängt nicht von Scores, Vorhersagen, Abdeckung oder Regelgüte ab. Sie ist wegen
der vorgegebenen Richtungen nach dem ursprünglichen Zielwert geschichtet.

Alle vier Methoden sehen dieselben Paare aus Originalkopie und veränderter Kopie:

1. Häufigster Trainingswert -> anderer im Training beobachteter Wert.
2. Anderer vorhandener Originalwert -> häufigster Trainingswert.
3. Vorhandener Originalwert -> fehlender Wert.

Eine Alternative wird per Hash aus den lexikalisch geordneten anderen
Trainingsklassen gewählt. Es werden keine neuen fachlichen Codes erfunden.
Die künstliche Veränderung kann eine tatsächlich plausible Kombination ergeben;
ohne Fachprüfung ist sie kein Fehlerlabel. Ein einzelner ausgewählter Originalfall
kann in verschiedenen Richtungen vorkommen, alle Herkunftskennungen bleiben erhalten.

Die abgeleitete Datums-Anwesenheitskennzeichnung wird mit ihrer bekannten,
ausgeschlossenen Datumsfamilie konsistent geändert: "fehlt" setzt das Datum leer,
"vorhanden" setzt den Median der vorhandenen Trainingsdaten. Ein eigenständig
fehlender Anwesenheitsindikator wird übersprungen, da der Adapter ihn ableitet.
Der eingesetzte Median erzeugt keine fachlich verifizierte Datumsangabe.

Vor der Regelprüfung müssen sämtliche Router-Prädiktoren und alle nicht zur
Änderungsfamilie gehörenden Originalattribute exakt übereinstimmen. Das Ziel und
seine Änderungsfamilie dürfen nicht unter den Router-/Regelprädiktoren stehen.
Damit haben Original und Kopie dieselbe eingefrorene Transformation und Route.
Gespeicherte Routen werden wiederverwendet; keine Modelle oder Pickles werden
geladen, kein fit/Neu-Clustering wird ausgeführt. Der reguläre Framework-Regelprüfer
wird auf den geänderten Attributen im unveränderten Scope erneut ausgeführt.

Erfasst werden neue Abweichungen, entfernte Abweichungen, Abweichungen vor und
nach der Änderung sowie Enthaltungen. Eine bereits abweichende Originalkopie
wird nicht als neue Erkennung gezählt. Nenner sind alle ausgewählten Paare bzw.
explizit die zuvor übereinstimmenden abgedeckten Paare. Enthaltung ist kein Erfolg.
Fehlende Zielwerte erhalten eine eigene Kennzahl; sie sind nicht still normale Codes.

## Reichweite

Der Versuch zeigt Reaktionen ausgewählter Regeln und ihrer Geltungsbereiche auf
kontrollierte Zieländerungen. Er prüft keine Korruption der Router-Prädiktoren,
kein LOF, keine neu eingelesenen Produkte und keine automatische fachliche Freigabe.
Insbesondere kann er einen Nutzen der gelernten Repräsentation nicht allein aus
einer hohen Änderungsquote ableiten. Der Vergleich von Attribut- und Latentclustern
zeigt Unterschiede der vorhandenen Geltungsbereiche auf demselben Bestand.

Referenzgruppen sind die unveränderten Gruppen des ursprünglichen Laufs. Sie
entsprechen identischen Attributkombinationen, nicht unabhängigen räumlichen
Einheiten oder Lieferungen. Mehrere Seeds sind wiederholte Aufteilungen desselben
Bestands und keine unabhängigen Experimente. Negative Resultate bleiben erhalten.
Es wird keine Regel nachträglich anhand dieser Testauswertung ausgewählt.
