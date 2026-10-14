"""Vergleichsbericht ohne automatische Erfolgsaussage über den latenten Raum."""
from html import escape
from pathlib import Path

from experiment_io import html_report, details_json
from review_common import read_json

NAMES = {"global": "Globale Baumregeln", "attribute_clusters": "Regeln in Attributclustern",
         "latent_clusters": "Regeln in Latentclustern", "training_majority": "Trainingsmehrheit",
         "attributes": "LOF auf Attributen", "pca": "LOF nach PCA", "latent": "LOF im latenten Raum"}
SCENARIOS = {"basis": "Vier informative Messwerte", "stoermerkmale": "Dieselben Messwerte + zwölf unabhängige Zusatzmerkmale"}


def pct(value):
    return "–" if value is None else f"{value*100:.1f} %".replace(".", ",")


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+\
        '</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def render_report(out):
    out = Path(out)
    summary = read_json(out/"summary.json")
    rules = read_json(out/"rule_metrics.json")
    lof = read_json(out/"lof_metrics.json")
    nulls = read_json(out/"null_control.json")
    protocol = read_json(out/"protocol.json")
    comparisons = read_json(out/"paired_comparisons.json")
    verification = read_json(out/"verification.json")
    parts = ['<article><strong>Was wird hier geprüft?</strong><p>Drei künstliche, gültige Attributgruppen '
        'besitzen unterschiedliche Prüfwerte A, B und C. Gruppe C umfasst nur 5 % und bleibt ausdrücklich gültig. '
        'Falsche Prüfwerte und unpassende Messwertkombinationen werden anhand einer bekannten Generatorregel erzeugt. '
        'Damit gibt es für diesen Kunstfall überprüfbare Fehlerlabels.</p>'
        '<p>Die Modelle erhalten weder Gruppentyp, Referenzlabels noch Fehlerkennzeichnungen. '
        'Alle Vergleichsmethoden sehen innerhalb derselben Aufgabe dieselben Merkmale, Einheiten und Änderungen. '
        'Eine Überlegenheit des latenten Raums ist kein Bestehenskriterium.</p></article>',
        f'<p>{summary["units_per_seed"]} künstliche Einheiten mit jeweils zwei Beobachtungen pro Seed; '
        f'{summary["epochs"]} Epochen, {summary["latent_dimensions"]} latente Dimensionen, '
        f'{summary["autoencoders_trained"]} tatsächlich trainierte Autoencoder. Keine Geometrien.</p>',
        '<p>Training, Validierung und Test enthalten getrennte Einheiten. Die beiden Beobachtungen einer Einheit '
        'bleiben zusammen. Die seltene Gruppe ist in allen drei Teilmengen vertreten. '
        'Jeder Seed erzeugt eine neue Zufallsziehung desselben Generators; die beiden Szenarien teilen ihre Signalwerte.</p>',
        '<h2>1. Automatisch abgeleitete Regeln</h2><p>Globale Bäume und Bäume innerhalb der beiden Clusterarten '
        'lernen den Prüfwert aus den Messwerten. Der Prüfwert und das unabhängige Kontrollattribut sind keine Prädiktoren. '
        'Die Zielerkennung läuft auf dem Training; für diese Evaluation sind Prüfwert und Nullkontrolle vorab festgelegt. '
        'Alle bisherigen Güteschwellen aus Paket 11 gelten weiter.</p>',
        '<p>Der Hauptvergleich nutzt einen gemischten Prüfbestand: In etwa 20 % der Testeinheiten wird '
        'eine von zwei Beobachtungen verändert. Rund 10 % der Zeilen sind damit synthetische Fehler. '
        'Der normale Partner bleibt als Kontrolle erhalten. Enthaltungen zählen bei der Fehlererkennung als nicht erkannt.</p>']
    for scenario, title in SCENARIOS.items():
        parts.append('<h3>'+escape(title)+'</h3>')
        rows = []
        for r in rules:
            if r["scenario"] == scenario and r["variant"] == "code_swap":
                rows.append([r["seed"], NAMES[r["method"]], pct(r["coverage"]),
                    f'{r["true_positives"]}/{r["errors"]}', pct(r["recall_all_errors"]),
                    f'{r["false_positives"]}/{r["clean"]}',
                    f'{r["rare_clean_false_positives"]}/{r["rare_clean"]}',
                    f'{r["rare_clean_covered"]}/{r["rare_clean"]}'])
        parts.append(table(["Seed", "Methode", "Abdeckung", "Fehler erkannt", "Recall aller Fehler",
            "Fehlmeldungen auf gültigen Fällen", "Davon seltene gültige Fälle", "Seltene gültige Fälle abgedeckt"], rows))
        parts.append('<details><summary>Unveränderter, vollständig gültiger Testbestand</summary>'+table(
            ["Seed", "Methode", "Abdeckung", "Fehlmeldungen", "Seltene gültige Fälle abgedeckt"],
            [[r["seed"], NAMES[r["method"]], pct(r["coverage"]), f'{r["false_positives"]}/{r["clean"]}',
              f'{r["rare_clean_covered"]}/{r["rare_clean"]}'] for r in rules if r["scenario"] == scenario and r["variant"] == "clean"])+ '</details>')
    parts.append('<p><strong>Nullkontrolle:</strong> K0 und K1 kommen bei identischen Prädiktoren innerhalb jeder '
        'Einheit gleich oft vor. Eine ausgewählte Abhängigkeit wäre hier verdächtig. Ergebnis: '+
        ('in allen Vergleichen abgewiesen.' if summary['null_control_rejected_everywhere'] else 'mindestens eine Regel ausgewählt; Detailprüfung nötig.')+'</p>')
    parts.append(details_json(nulls, 'Nullkontrolle je Seed, Szenario und Methode'))
    parts.append('<h2>2. Separater LOF-Vergleich</h2><p>Hier gehören Messwerte und beobachteter Prüfcode '
        'zur Eingabe. Das Kontrollattribut ist ausgeschlossen. Dafür wird ein eigener Autoencoder ausschließlich '
        'auf gültigen Trainingsdaten gelernt. LOF arbeitet auf standardisierten/One-Hot-kodierten Attributen, '
        'einer PCA mit gleicher Dimension oder der gelernten Darstellung. Die Eingabeinformation ist identisch.</p>'
        '<p>Der Hauptvergleich verwendet für jede Methode dasselbe feste Prüfbudget von 10 % der Testzeilen. '
        'Precision bezeichnet den Fehleranteil unter den ausgewählten Fällen, Recall den Anteil der gefundenen '
        'synthetischen Fehler. Beides hängt vom festgelegten künstlichen Prüfbestand ab. '
        'Average Precision bewertet ergänzend die gesamte Rangfolge; deren Zufallsreferenz entspricht ungefähr dem Fehleranteil.</p>')
    for scenario, title in SCENARIOS.items():
        parts.append('<h3>'+escape(title)+'</h3>')
        for variant, description in (("code_swap", "Falscher Prüfcode bei unveränderten Messwerten"),
                                     ("signal_swap", "Messwert aus einer anderen gültigen Gruppe übernommen")):
            rows = []
            for r in lof:
                if r["scenario"] == scenario and r["variant"] == variant:
                    b = r["budget"]
                    rows.append([r["seed"], NAMES[r["representation"]], b["selected"],
                        f'{b["true_positives"]}/{r["errors"]}', pct(b["precision"]), pct(b["recall"]),
                        f'{b["rare_clean_selected"]}/{b["rare_clean"]}', pct(r["average_precision"])])
            parts.append('<p>'+description+'</p>'+table(["Seed", "Darstellung", "Budget", "Fehler erkannt", "Precision", "Recall", "Seltene gültige Fälle ausgewählt", "Average Precision"], rows))
    parts.append('<details><summary>Zusätzlich: feste Schwelle aus normalen Validierungsdaten</summary>'
        '<p>Die Schwelle ist das obere 95-%-Quantil der normalen Validierungsscores. '
        'Es gilt strikt Score &gt; Schwelle. Der tatsächliche Test-Fehlalarmanteil kann davon abweichen. '
        'Hier werden weder Testlabels zur Schwellenwahl genutzt noch Budgets nachträglich angepasst.</p>'+table(
        ["Szenario", "Seed", "Änderung", "Darstellung", "Recall", "Fehlalarmanteil gültiger Fälle"],
        [[r["scenario"], r["seed"], r["variant"], NAMES[r["representation"]], pct(r["threshold"]["recall"]),
          pct(r["threshold"]["false_positive_rate_clean"])] for r in lof])+ '</details>')
    parts.append('<h2>3. Einordnung und Nachvollziehbarkeit</h2>'
        '<p>Ein Vorteil bei Regeln und ein Vorteil bei LOF sind unterschiedliche Ergebnisse. '
        'Eine hohe Zahl ausgewählter Regeln oder schöner Cluster ist kein Gütenachweis. '
        'Zusätzliche Treffer müssen zusammen mit Fehlmeldungen, Abdeckung und seltenen gültigen Fällen gelesen werden. '
        'Wenn PCA ähnlich gut abschneidet, ist ein Gewinn nicht allein dem Autoencoder zuzuschreiben.</p>'
        '<p>Die Referenzdefinition ist eine frei festgelegte Testannahme. Sie ist keine DOP-/ALKIS-Spezifikation. '
        'Die Ergebnisse belegen technische Möglichkeiten und Grenzen in diesen zwei künstlichen Szenarien. '
        'Sie ersetzen keine fachlich gelabelten realen Lieferungen. Keine Parameter wurden anhand der Testergebnisse optimiert.</p>')
    parts.append(details_json(comparisons, "Gepaarte Unterschiede, einschließlich negativer Ergebnisse"))
    parts.append(details_json(verification, f'{verification["check_count"]} technische Nachprüfungen'))
    parts.append('<ul>'+''.join(f'<li><a href="{name}">{label}</a></li>' for name,label in [
        ("summary.json", "Zusammenfassung"), ("rule_metrics.json", "Regelergebnisse"), ("lof_metrics.json", "LOF-Ergebnisse"),
        ("paired_comparisons.json", "Methodenunterschiede"), ("null_control.json", "Nullkontrolle"),
        ("protocol.json", "Parameter und Ablauf"), ("PROTOKOLL.md", "Versuchsprotokoll")])+ '</ul>')
    for seed in summary["seeds"]:
        for scenario in SCENARIOS:
            prefix = f"seed_{seed}/{scenario}"
            parts.append(f'<details><summary>Regeln und Fälle: Seed {seed}, {scenario}</summary><ul>'+''.join(
                f'<li><a href="{prefix}/{name}">{label}</a></li>' for name,label in [
                    ("selected_rules.json", "Ausgewählte Regeln"), ("rule_classification.json", "Alle Regeltypen"),
                    ("model_reports.json", "Güteschwellen und Ablehnungen"), ("test_labels.parquet", "Synthetische Referenzlabels"),
                    ("rule_cases.parquet", "Regelprüfungen je Fall"), ("lof_cases.parquet", "LOF-Scores und Ranglisten")])+ '</ul></details>')
    parts.append('<p>LOF-API: <a href="'+escape(protocol["documentation"])+ '">offizielle scikit-learn-Dokumentation</a>. '
        'novelty=True wird ausschließlich mit neuen Validierungs-/Testeinheiten ausgewertet.</p>')
    html_report(out/"bericht.html", "Kontrollierter Vergleich: Regeln und LOF", ''.join(parts))
