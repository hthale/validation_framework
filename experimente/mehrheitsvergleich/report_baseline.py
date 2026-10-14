"""Lokaler Bericht mit getrennten Nennern für Originale und Änderungskopien."""
from html import escape

from comparison_core import LABELS, METHODS, summarize_cases
from baseline_core import BASELINE
from experiment_io import html_report, details_json

NAMES = {**LABELS, BASELINE: "Häufigster Trainingswert"}
KINDS_DE = {"global_attribute_dependency": "Globale Attributabhängigkeit",
            "global_constant_expectation": "Globale konstante Erwartung",
            "cluster_attribute_dependency": "Zusätzliche Attributabhängigkeit im Cluster",
            "cluster_constant_expectation": "Konstante Erwartung im Cluster"}
SCENARIOS_DE = {"majority_to_nonmajority": "Häufigster → anderer Trainingswert",
                "nonmajority_to_majority": "Anderer Originalwert → häufigster Trainingswert",
                "present_to_missing": "Vorhandener Wert → fehlender Wert"}


def table(headers, rows):
    def cell(value):
        return escape("–" if value is None else str(value))
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+cell(h)+'</th>' for h in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+cell(v)+'</td>' for v in row)+'</tr>' for row in rows)+\
        '</tbody></table></div>'


def percent(value):
    return None if value is None else f"{100*value:.2f} %".replace(".", ",")


def render_report(out, summary, counts, comparisons, metrics, availability, verification):
    import pandas as pd
    from review_common import read_json
    group_limit = read_json(out/"protocol.json")["max_reference_groups_per_scenario"]
    cases = pd.read_parquet(out/"original_cases.parquet")
    parts = [f'<p>Zusatzanalyse von <code>{escape(summary["source_run"])}</code>.</p>',
        '<article><strong>Was dieser Bericht beantwortet</strong><p>Welche Regelarten wurden ausgewählt? '
        'Sagen sie auf denselben Fällen etwas anderes voraus als der häufigste Trainingswert? '
        'Wie reagieren sie auf gezielte Änderungen gegenüber unveränderten Kontrollfällen?</p>'
        '<p>Die Modelle und die Regelwahl bleiben eingefroren. Es findet kein neues Training statt. '
        'Nur Attribute werden verwendet, keine Geometrien.</p></article>',
        '<p>Dies ist eine nachträgliche, explorative Auswertung desselben bereits betrachteten internen Testanteils. '
        'Originalwerte sind nicht fachlich als fehlerfrei bestätigt; geänderte Werte sind keine bestätigten Sachfehler. '
        'Abweichungen werden deshalb weder als reale Fehlererkennungsrate noch als Fehlalarmrate ausgegeben.</p>',
        '<h2>1. Gefundene Regelarten</h2><p>Eine Clusterzuordnung ist selbst eine Bedingung. '
        '„Konstant im Cluster“ bedeutet: Innerhalb dieses Clusters verlangt die Regel überall denselben Zielwert, '
        'ohne zusätzliche Attributbedingung. Gezählt werden Regelinstanzen je Seed; verschiedene Cluster werden nicht zusammengelegt.</p>']
    rows = [[r["seed"], NAMES[r["method"]], KINDS_DE[r["rule_kind"]], r["candidates"], r["selected"]]
            for r in counts if r["candidates"]]
    parts.append(table(["Seed", "Methode", "Regelart", "Kandidaten", "Ausgewählt"], rows))
    parts.append('<h2>2. Unveränderte Originalfälle</h2><p>Die zusätzliche Vergleichsmethode sagt für jedes Ziel '
        'immer den häufigsten nichtleeren Trainingswert voraus. Sie ist bewusst ungefiltert und keine freigegebene Regel. '
        'Sie deckt alle Fälle ab; die drei bisherigen Regelmethoden können sich enthalten. '
        'Die Zeilen zählen Ziel-Datensatz-Paare, nicht unterschiedliche Personen oder unabhängige Lieferungen.</p>')
    rows = []
    for (seed, method), sub in cases.groupby(["seed", "method"], sort=True):
        m = summarize_cases(sub)
        same = [r for r in comparisons if r["seed"] == seed and r["method"] == method and r["stratum"] == "all"]
        rows.append([seed, NAMES[method], m["covered"], m["target_record_pairs"], percent(m["coverage"]),
            percent(m["agreement_all_covered"]), m["disagree_observed"], m["missing_target_covered"],
            sum(r["different_predictions_from_majority"] for r in same), sum(r["net_agreement_gain_on_same_covered"] for r in same)])
    parts.append(table(["Seed", "Methode", "Abgedeckt", "Alle Paare", "Abdeckung", "Übereinstimmung im abgedeckten Bestand",
        "Abweichung bei vorhandenem Ziel", "Ziel fehlt bei Abdeckung", "Andere Vorhersage als Mehrheit", "Zusätzliche Übereinstimmungen auf denselben Fällen"], rows))
    parts.append('<p>Die letzten beiden Spalten vergleichen exakt dieselben abgedeckten Fälle. '
        'Null andere Vorhersagen bedeutet: Der Unterschied zur globalen Mehrheit liegt hier in der Auswahl der Fälle. '
        'Diese Auswahl kann nützlich sein, belegt aber allein keine bessere Vorhersage von Minderheitswerten. '
        'Gruppengewichtete Kennzahlen und Makro-F1 je Ziel stehen in den Detaildaten.</p>')
    for seed in summary["seeds"]:
        rows = [[r["target"], NAMES[r["method"]], r["target_record_pairs"], r["covered"], r["agree"], r["disagree_observed"],
                 r["different_predictions_from_majority"]] for r in comparisons if r["seed"] == seed and r["stratum"] == "original_nonmajority"]
        parts.append(f'<details><summary>Seed {seed}: Originalwerte außerhalb der Trainingsmehrheit</summary>'+
            '<p>Diese Kontrollfälle bleiben unverändert. Eine Abweichung ist ohne fachliche Prüfung kein bestätigter Fehlalarm. '
            '„Außerhalb der Mehrheit“ bedeutet nicht zwingend „selten“.</p>'+table(["Ziel", "Methode", "Originalfälle", "Abgedeckt", "Übereinstimmung", "Abweichung", "Andere Vorhersage"], rows)+'</details>')
    parts.append('<h2>3. Kontrollierte Änderungen mit unveränderten Vergleichskopien</h2>'
        f'<p>Je Seed, Ziel und Richtung werden höchstens {group_limit} Referenzgruppen verwendet. '
        'Pro Gruppe wird ein Fall über eine feste Hash-Reihenfolge gewählt, unabhängig von Scores und Regelabdeckung. '
        'Alle vier Methoden erhalten dieselben Fallpaare. Alternative Werte stammen aus dem Training.</p>'
        '<p>„Neue Abweichung“ zählt nur, wenn die unveränderte Kopie zuvor übereinstimmte. '
        '„Abweichung entfernt“ zeigt die Gegenrichtung. Bereits vorher bestehende Abweichungen gelten nicht als neue Treffer. '
        'Enthaltung wird separat gezählt und ist kein erfolgreicher Test.</p>')
    for scenario, label in SCENARIOS_DE.items():
        rows = []
        for seed in summary["seeds"]:
            for method in (*METHODS, BASELINE):
                selected = [r for r in metrics if r["seed"] == seed and r["method"] == method and r["scenario"] == scenario]
                if not selected:
                    continue
                names = ("paired_cases", "abstained_pairs", "original_disagreement", "changed_disagreement",
                         "new_disagreement", "disagreement_removed", "both_disagree")
                rows.append([seed, NAMES[method], *(sum(r[c] for r in selected) for c in names)])
        parts.append('<h3>'+escape(label)+'</h3>'+table(["Seed", "Methode", "Paare", "Enthalten", "Vorher abweichend", "Nachher abweichend", "Neue Abweichung", "Abweichung entfernt", "Beide abweichend"], rows))
    parts.append('<p>Die Referenzgruppen bleiben die Gruppen der Originaldaten; geänderte Kopien werden nicht neu gruppiert. '
        'Die Datums-Anwesenheitskennzeichnung wird gemeinsam mit ihrem ausgeschlossenen Datumsfeld geändert. '
        'Eine unabhängig fehlende Anwesenheitskennzeichnung wird nicht erzeugt.</p>'
        '<p><strong>Reichweite:</strong> Nur Zielwerte bzw. ihre ausgeschlossene Datumsfamilie werden geändert. '
        'Identische Router-Prädiktoren rechtfertigen die Wiederverwendung der gespeicherten Clusterzuordnung. '
        'Der Versuch prüft keine Änderungen an Router-Prädiktoren, kein LOF und keine neue Clusterbildung.</p>')
    parts.append('<h2>4. Nachvollziehbarkeit</h2>'+details_json(availability, "Verfügbare und ausgewählte Fälle je Richtung"))
    parts.append(details_json(verification, f'{verification["check_count"]} Nachprüfungen bestanden'))
    parts.append('<p>Wiederholte Seeds verwenden denselben Datenbestand. Summen über Seeds sind keine unabhängigen Beobachtungen. '
        'Der Bericht übernimmt die ursprüngliche Auswahl; die Testgüte wählt keine Regel nachträglich aus.</p>')
    parts.append('<ul>'+''.join(f'<li><a href="{name}">{label}</a></li>' for name, label in [
        ("summary.json", "Zusammenfassung"), ("majority_comparison.json", "Mehrheitsvergleich und Originalkontrollen"),
        ("mutation_metrics.json", "Änderungsergebnisse je Seed und Ziel"), ("majority_models.json", "Häufigkeiten aus Training"),
        ("rule_classification.json", "Alle Regeln mit Typ und Scope"), ("local_majority_diagnostics.json", "Lokale Mehrheitsvergleiche aus Paket 11"),
        ("paired_changes.parquet", "Alle ausgewerteten Fallpaare"), ("PROTOKOLL.md", "Versuchsprotokoll")])+'</ul>')
    html_report(out/"bericht.html", "Regelarten, Trainingsmehrheit und kontrollierte Änderungen", ''.join(parts))
