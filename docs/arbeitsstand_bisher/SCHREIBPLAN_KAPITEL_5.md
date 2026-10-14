# Schreibgrundlage für Kapitel 5

Stand: 23.09.2026. Arbeitsplan für den Einstieg am 24.09.2026; kein Ersatz für den endgültigen wissenschaftlichen Text.

## Anschluss an die vorhandene Arbeit

 Hochgeladene PDF enthält bereits Kapitel 4 zur Konzeption. Kapitel 5 heißt dort „Prototypische Untersuchung datengetriebener Komponenten“ und besitzt die Punkte 5.1 Datengrundlage, 5.2 Aufbau des Prototyps und 5.3 (Durchführung). Diese Grundstruktur ist weiterhin passend. Als präzisierter Titel bietet sich **„Prototypische Umsetzung und Versuchsaufbau“** an. Kapitel 6 übernimmt die Ergebnisbewertung.

Die Paketnummern gehören in eine technische Zuordnung oder den Anhang. Der Haupttext sollte erklären, wie die Komponenten zusammenwirken und welche Entscheidungen getroffen wurden. Die Reihenfolge der bisherigen Korrekturpakete ist keine geeignete Kapitelgliederung.

## Empfohlene Gliederung und Inhalt

| Abschnitt | Was hinein gehört | Beleg bzw. Abgrenzung |
|---|---|---|
| 5.1 Datengrundlage und Untersuchungsumfang | DOP-Merkmalstabelle mit 500 Fällen; ALKIS-Bahnverkehr mit 47.245 Quellobjekten und 5.000 ausgewählten Objekten; künstliche Daten | Herkunft und Zeitraum der Lieferungen anhand Originalunterlagen ergänzen; keine ungeprüfte Fehlerfreiheit behaupten |
| 5.2.1 Softwareaufbau und Programmeinstiege | Python, modulare Verarbeitung, Parquet als gemeinsame Eingabeschnittstelle, generischer Einstieg und zusätzliche Versuchseinstiege | Reale Softwareversionen dem Manifest des jeweils beschriebenen Laufs entnehmen |
| 5.2.2 Identität und Vorverarbeitung | `record_key`, gespeicherte ID-Kopie, Rollen, fehlende Werte, numerische Skalierung, Kategorien und Textpfad, Gruppen | Keine Kennungen oder Ergebnisspalten als Regelprädiktoren; ALKIS nur ausgewählte Attribute |
| 5.2.3 Repräsentationslernen und Auffälligkeiten | Autoencoder, Rekonstruktions- und weitere Trainingsziele, LOF, HDBSCAN und UMAP | Konkrete Parameter je Versuch; UMAP dient der Darstellung, nicht der LOF-Eingabe |
| 5.2.4 Automatische Regelkandidaten | Profiling, globale Bäume, Export und Prüfung von Bedingungen, Clusterbeschreibung; Erweiterung um lokale Regeln | Globale Abhängigkeiten, Clusterbeschreibung und lokale Abhängigkeiten ausdrücklich unterscheiden |
| 5.2.5 Ergebnisexport und Nachvollziehbarkeit | Regeln, Fallkennungen, Scores, Berichte, Modellartefakte, Aufteilungen, Prüfsummen; ergänzende lokale Prüfansicht | Kandidat, bestandene Güteschwelle und fachliche Freigabe sind unterschiedliche Zustände |
| 5.3.1 Technische Funktionsprüfung | Tests zu Kennungen, Zuordnung, Regelanwendung, Ausgabe und bekannten synthetischen Beziehungen | Erfolgreiche Tests belegen ihre geprüften Eigenschaften, keine allgemeine fachliche Güte |
| 5.3.2 Versuche mit DOP | Bestandsanalyse und anschließend kontrollierte Änderungen von Testkopien | Aufteilung, Änderungsarten und Prüfaufwand erklären; Zahlenbewertung nach Kapitel 6 |
| 5.3.3 Versuche mit ALKIS-Attributen | Adapter, Stichprobe, globale/attributlokale/latentlokale Regeln, Mehrheitsvergleich | Gleiche Prädiktoren, zielbezogener Ausschluss und Enthaltung außerhalb der Cluster |
| 5.3.4 Kontrollierter künstlicher Vergleich | Bekannte Gültigkeit, seltene gültige Gruppe, Zusatzmerkmale, Fehler, Vergleichsmethoden, fixe Auswahlregeln | Fehlertypen und Referenzmechanismus vor der Testauswertung festgelegt |
| 5.3.5 Bewertungsgrößen und Aussagegrenzen | Regelabdeckung, Erkennung einschließlich Enthaltungen, Fehlmeldungen, LOF-Budget, Anpassungsgüte | Reale Änderungen und synthetisch bekannte Fehler getrennt auswerten |

Für Kapitel 6 fehlt in der bisherigen Gliederung ein deutlich benannter Punkt zur **automatischen Regelgewinnung**. Vorschlag: 6.1 Technische Funktionsfähigkeit, 6.2 Regelgewinnung und Geltungsbereiche, 6.3 Auffälligkeitserkennung und Repräsentationsvergleich, 6.4 Grenzen und Übertragbarkeit. Darin erhält der latente Raum eine eigene begründete Rolle, ohne alle Ergebnisse ihm zuzuschreiben.

## Zuordnung zur vorhandenen Implementierung

Alle Pfade beziehen sich auf den Projektordner. Es sind vorhandene Module der geprüften Arbeitskopie, keine vorgeschlagene neue Ordnerstruktur.

| Baustein | Tatsächliche Dateien/Funktionen | Beitrag zum Text |
|---|---|---|
| Generische Steuerung | `src/run_generic.py`: `analyze` | Hauptablauf und gespeicherte Ergebnisse |
| Identität/Rollen | `src/record_identity.py`; `src/generic_schema.py`: `attach_identity`, `infer_columns`, `normalize_columns`, `choose_groups`, `tabular_preparation` | Technischer Eingabevertrag, Einlesekennung und Feature-Auswahl |
| Bahnverkehr-Adapter | `analysepaket_10_bahn_attribute/attribute_prep.py`, `start_bahnverkehr.py` | Fachcodes/Codelisten und ausdrücklicher Geometrieausschluss |
| Encoder und Training | `src/latentspace/main_v2.py`: `train_model`; `multimodal_autoencoder.py`, `tabular_encoder.py`, `string_encoder.py` | Tatsächliche Modellstruktur und Lernziele |
| Auffälligkeiten/Cluster | `src/latentspace/anomaly_scores.py`; `main_v2.py`: `compute_hdbscan_raw`, `project_umap` | LOF als primärer Rang, Cluster und Visualisierung |
| Profiling | `src/generic_rules.py`: `learn_profiles`; `src/profiling.py`, `src/sampling.py` | Wiederholte Vorschläge, latente Strata nur für Stichprobenauswahl |
| Globale Abhängigkeiten | `src/generic_rules.py`: `learn_dependencies`; `src/phase2_ml.py`: `train_dependency`; `src/phase2_split.py`, `phase2_settings.py` | Zielauswahl, gruppierte Teilmengen, Parameterwahl, Güteschwellen |
| Ausführbare Bedingungen | `src/tree_rule_export.py`, `src/rule_conditions.py`, `src/rule_engine.py` | Imputation und Datentypen im Export, Anwendbarkeit und Regelverletzung |
| Clusterbeschreibung | `src/generic_clusters.py`: `describe_clusters` | Beschreibender Baum und Übereinstimmung mit festen Clusterlabels |
| Lokale Regelversuche | `vergleichspaket_11_clusterregeln/comparison_core.py`: `learn_method`, `evaluate_selected` | Gemeinsamer Vergleichskern mit obligatorischem Geltungsbereich |
| Künstliche Fehlerstudie | `testpaket_13_kontrollierter_vergleich/synthetic_data.py`, `evaluation.py`, `run_controlled_comparison.py` | Getrennte Regel-/LOF-Aufgaben, bekannte Fehler und eingefrorene Auswahl |
| Berichte/Entscheidungen | `src/generic_report.py`; ergänzend `build_review.py`, `result_integration.py`, `review_decisions.py` | Generischer Bericht versus gesonderter lokaler Entscheidungsablauf |
| Provenienz | `src/experiment_io.py`, `review_common.py` sowie Manifeste der Läufe | Nachvollziehbare Konfiguration, Versionsangaben und Prüfsummen |

## Konfigurationen nicht vermischen

| Modus | Lernumfang | Latente Dimension / Epochen | Bedeutung der Auswertung |
|---|---|---|---|
| Generischer DOP-Lauf 07 | Autoencoder/Cluster auf dem Bestand; Abhängigkeiten separat aufgeteilt | Standarddimension 64 / 25 Epochen laut protokolliertem Aufruf | Explorative Bestandsbeschreibung und Kandidaten |
| Bahnverkehr-Bestandslauf 10 | Generischer Lauf auf 5.000 Attributobjekten | Standarddimension 64 / 5 Epochen | Explorative Bestandsbeschreibung |
| Bahnverkehr-Regelvergleich 11 | Training/Validierung/Test; eigene Darstellung je Ziel ohne Zielinformation | 8 / 5 | Vergleich globaler und lokaler Regeln |
| Zusatzanalyse 12 | Gespeicherte Modelle, Regeln und Routen aus 11 | Kein neues Training | Mehrheitsvergleich und Reaktion auf Zieländerungen |
| Künstlicher Vergleich 13 | Training/Validierung/Test nach erzeugten Einheiten; getrennte Regel-/LOF-Modelle | 3 / 25 | Bekannter künstlicher Fehlermechanismus; mit/ohne Zusatzmerkmale |

Der generische Kern verwendet für globale Bäume eine Gruppen-CV zur Auswahl aus Baumtiefen 4/8 und Blattgrößen 5/10/20. Die Vergleiche 11/13 verwenden dagegen vorab feste Tiefe 4 und Blattgröße 5 sowie HDBSCAN-Parameter 50/10. Diese Unterschiede als Versuchsentscheidung dokumentieren. Sie verhindern, dass Befunde aus 13 nachträglich als Validierung jedes früheren Modells gelten.

64 latente Dimensionen bedeuten nicht zwangsläufig eine Verdichtung gegenüber jeder Eingangsmatrix. Die Dimension 3 im künstlichen Vergleich wurde gezielt für eine Kompression auch der vier Basismerkmale gewählt. Im Text die konkrete Dimensionszahl und Eingabematrix nennen, statt jeden latenten Raum pauschal als Reduktion zu bezeichnen.

## Einstiegstext für 5.2

Die folgenden Absätze beschreiben die eigene Implementierung und können nach dem Abgleich des lokalen Projektstands als Rohfassung verwendet werden. Sie sind keine Literaturparaphrase und behaupten keine bisher unbelegte Güte:

> Der Prototyp untersucht die datengetriebene Validierung anhand tabellarisch aufbereiteter Merkmale. Als gemeinsame Eingabeschnittstelle dient eine Parquet-Tabelle, deren Zeilen über eindeutige Kennungen mit den erzeugten Ergebnissen verbunden bleiben. Für die Untersuchung werden DOP-Merkmale und ausgewählte Attribute der ALKIS-Objektart Bahnverkehr verwendet. Geometrien und geometrisch abgeleitete Merkmale sind nicht Bestandteil der ALKIS-Untersuchung. Der generische Programmeinstieg koordiniert die technische Merkmalsauswahl, das Repräsentationslernen, die Ermittlung von Auffälligkeiten und die Erzeugung von Regelkandidaten.
>
> Die Analyse umfasst mehrere sich ergänzende Verarbeitungspfade. Ein Autoencoder erzeugt eine latente Darstellung, auf der im generischen Lauf die Gruppierung durch HDBSCAN und die Priorisierung durch LOF erfolgen. Die resultierenden Cluster werden anhand der Ausgangsattribute beschrieben. Globale Abhängigkeitskandidaten werden durch einen gesonderten Entscheidungsbaumpfad aus den Attributen gelernt. Profiling-Verfahren ergänzen diese Kandidaten um beobachtete Eigenschaften einzelner Merkmale. Die Ergebnisse werden mit ihrer Herkunft und ihrem jeweiligen Aussagebereich gespeichert.
>
> Für die Untersuchung des Repräsentationsbeitrags werden ergänzende Vergleichsprogramme eingesetzt. Diese unterscheiden globale Regeln, Regeln innerhalb von Attributclustern und Regeln innerhalb von Latentclustern. Die Gültigkeit einer lokalen Regel umfasst dabei stets die zugehörige Clusterbedingung. Anders als bei der beschreibenden Bestandsanalyse werden die Modellanpassung, die Auswahl anhand einer Validierungsteilmenge und die abschließende Auswertung auf getrennten Gruppen vorgenommen. Die Bewertung der damit erzielten Ergebnisse erfolgt in Kapitel 6.

## Morgen in dieser Reihenfolge schreiben

1. **5.1 fertigstellen:** Datensätze, Stichprobe, Attributumfang, Herkunft, fehlende reale Fehlerlabels und Abgrenzung zu Geometrien. Ungeklärte Herkunftsangaben als offene Stellen markieren.
2. **5.2.1 und 5.2.2 schreiben:** Der Programmablauf und der Umgang mit Kennungen/Vorverarbeitung sind schon konkret belegbar.
3. **5.2.3 und 5.2.4 ergänzen:** Modellstruktur, getrennte Regelpfade und den Status der lokalen Vergleichserweiterung beschreiben.
4. **5.3 mit einer Versuchstabelle beginnen:** Datensatz, Fragestellung, Aufteilung, Methoden, Änderungen, Kennzahlen und Aussagegrenze je Versuch.

Algorithmische Definitionen und theoretische Begründungen mit den Primärquellen in Kapitel 2/4 verbinden. Eigene Konfigurationsentscheidungen als solche kennzeichnen; keine Quelle als Beleg für die eigene Implementierung oder den gemessenen Vorteil ausgeben. Ergebniszahlen und deren Bewertung vorwiegend in Kapitel 6 behandeln.

Die PDF-Fassung enthält außerdem im Inhaltsverzeichnis bei 4.6 eine nicht definierte Textmarke. Beim nächsten Word-Durchgang den Querverweis reparieren und das Inhaltsverzeichnis aktualisieren.
