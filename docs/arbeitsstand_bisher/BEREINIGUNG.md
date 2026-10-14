# Projektbereinigung ohne Verlust der Nachweise

## Was mit diesem Schritt erledigt ist

Der Projektstand, die Ergebnisarten, die Programmeinstiege und ihre Verwendung in der Masterarbeit sind konsolidiert. Es wurde kein vorhandener Projektcode ersetzt und kein Ergebnislauf gelöscht. Die tatsächliche Windows-Verzeichnisstruktur ist hier nicht direkt verfügbar. `projektinventar.py` macht diesen Abgleich als nächsten Schritt möglich.

Eine vollständige Umstrukturierung vor dem Schreiben ist nicht erforderlich. Der konkrete erste Schritt ist ein übersichtlicher, dokumentierter Arbeitsstand mit einer klaren Trennung zwischen Kern, Adaptern, Evaluation und Entwicklungsgeschichte.

## Zuordnung der vorhandenen Bestandteile

| Bestandteil | Rolle für die Arbeit | Vorgehen |
|---|---|---|
| `src/` und `tests/` | Kernimplementierung und technische Nachweise | Aktiv behalten; tatsächliche Einstiegspfade dokumentieren |
| `src/run_generic.py` | Gemeinsamer Analyse-Einstieg ab Parquet | Als Hauptablauf beschreiben |
| `analysepaket_10_bahn_attribute/` | Adapter für ALKIS-Bahnverkehr | Aktiv behalten; wird auch von den späteren Vergleichen gebraucht |
| `testpaket_09_gemeinsame_pruefung/` | DOP-Versuch mit Regeln und zwei LOF-Darstellungen | Als Evaluationsprogramm behalten |
| `vergleichspaket_11_clusterregeln/` | Globale und lokale Regelvergleiche | Als Evaluationsprogramm und Erweiterung behalten |
| `analysepaket_12_mehrheitsvergleich/` | Mehrheitsvergleich und kontrollierte Zieländerungen | Als Evaluationsprogramm behalten |
| `testpaket_13_kontrollierter_vergleich/` | Künstliche Fehlerstudie mit Vergleichsmethoden | Als Evaluationsprogramm behalten |
| Testpakete 05/06 und Beispielprüfung 07 | Funktions-/Integrationsnachweise | Für Tests/Anhang aufbewahren; keine drei zusätzlichen realen Leistungsnachweise daraus machen |
| Installationsordner und ZIPs der Korrekturpakete 01–04/07/08 | Entwicklungsgeschichte und Wiederherstellbarkeit | Nach dem Abgleich außerhalb der täglichen Arbeitsansicht archivieren; Paket 07 enthält zusätzlich Beispiel-/Prüfprogramme |
| Alte Modelle und Ergebnisordner | Reproduzierbarkeit und Vergleich der Entwicklungsstände | Vollständige Läufe behalten; über einen Ergebnisindex auf wichtige Läufe verweisen |
| `backups/` | Wiederherstellung | Mindestens bis zum gesicherten Abschlussstand behalten |
| `arbeitsstand_masterarbeit/` | Dokumentation des Schreibstands | Als Einstieg und Notizgrundlage verwenden |

## Warum die Pakete 10–13 noch zusammenbleiben

Die Programme enthalten derzeit konkrete Pfadabhängigkeiten: 13 lädt Funktionen aus 12, 12 lädt 11, und 11 verwendet den Adapter aus 10 sowie Module aus `src`. Das bloße Verschieben eines einzelnen Ordners nach `archive` oder `experiments` kann diese Aufrufe unterbrechen. Einige Verifikationen vergleichen außerdem Quellpfade und Prüfsummen früherer Läufe.

Eine spätere saubere Zielstruktur kann `src/`, `tests/`, `adapters/`, `experiments/`, `docs/`, `data/` und `archive/` enthalten. Sie ist **ein Refactoring-Vorschlag, noch nicht die vorhandene Struktur**. Dafür müssten Imports und Pfadauflösung angepasst, bestehende Läufe weiter auffindbar gehalten und die betroffenen Abläufe gezielt geprüft werden. Alte Manifeste werden dabei nicht nachträglich umgeschrieben, um eine neue Struktur als alten Lauf auszugeben.

## Konkrete Reihenfolge

1. Die mitgelieferte Bestandsaufnahme ausführen. Sie erfasst Code-Dateien, Prüfsummen, Programmeinstiege, bekannte Paketabhängigkeiten und eine Übersicht erreichbarer Ergebnisordner. Sie lädt keine Modelle und liest keine Parquet-Zeilen.
2. Die inventarisierte Codefassung und die zugehörigen Manifeste als Arbeitsstand sichern. Falls bereits Git verwendet wird, dort einen bezeichneten Stand sichern; sonst eine getrennte Kopie von Code, Tests, Konfiguration und Dokumentation anlegen. Vorhandene Originaldaten und Ergebnisverzeichnisse behalten.
3. In der täglichen Arbeitsansicht auf den gemeinsamen Einstieg und die vier ausgewählten Versuchsprogramme verweisen. Kapitel 5 beschreibt Komponenten und Schnittstellen, nicht die Korrekturpaketfolge.
4. Historische Installationsdateien erst nach dem lokalen Abgleich archivieren. Ein gleicher Dateihash zeigt nur gleiche Inhalte, nicht dass einer der Pfade entbehrlich ist.
5. Falls noch Zeit für Refactoring bleibt: zunächst gemeinsame Versuchsfunktionen bündeln und Pfadauflösung vereinheitlichen. Das ist ein eigener Arbeitsschritt mit gezielter Prüfung; für den Schreibbeginn ist er nicht erforderlich.

## Was wir jetzt nicht als fertige Funktion darstellen

- Keine universelle semantische Erkennung beliebiger Rohprodukte: Der Kern beginnt ab einer geeigneten Tabelle, Bahnverkehr benötigt den Adapter.
- Keine durchgehend integrierte lokale Regelpipeline im Hauptprogramm: Die Vergleiche 11/13 sind ergänzende Programme.
- Kein vollständiger Regel-Freigabeprozess mit Rückführung in spätere Lieferungen: Die vorhandene lokale Fallprüfung deckt einen Teil dieses Konzepts ab.
- Kein produktiv nachgewiesener Delivery-Memory-Zyklus: Die dokumentierten aktiven Versuche schreiben ihn nicht.

## Nachweise für Kapitel 6 und den Anhang

Für jeden berichteten Versuch gemeinsam aufbewahren: Eingangskopie bzw. Eingabeprüfsumme, Programm- und Softwarestand, Konfiguration, Teilmengen, Änderungsprotokoll, Modellartefakte, Regeln mit Ablehnungsgründen, Fallauswertung, Kennzahlen und Bericht. Nicht nur die HTML-Seite behalten.

Die Inventur erkennt vorhandene Läufe anhand bekannter Status-/Manifestdateien. Sie bestätigt nicht deren wissenschaftliche Güte und ersetzt keine zugehörige Laufverifikation. Fehlende Dateien und abweichende Prüfsummen bleiben als Hinweise sichtbar.
