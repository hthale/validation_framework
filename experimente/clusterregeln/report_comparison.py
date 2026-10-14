"""Offline-HTML; Vorhersagen, Abdeckung und Auswahl sichtbar getrennt."""
from html import escape

from comparison_core import LABELS
from experiment_io import html_report, details_json


def percent(value):
    return "–" if value is None else f"{value*100:.1f} %"


def render_report(out, summary, candidates, audits):
    content = ('<p class="notice"><strong>Was wird verglichen?</strong> Globale Attributregeln, '
        'Regeln in Attributclustern und Regeln in Latentclustern. Alle Varianten verwenden je Ziel '
        'dieselben Ausgangsmerkmale und zurückgehaltenen Fälle. Geometrien sind ausgeschlossen.</p>'
        '<p>Das Zielattribut und seine bekannten direkten Ableitungen fehlen beim Lernen der Darstellung '
        'und der Gruppen. Die Gruppen werden pro Ziel gebildet. Vorverarbeitung, Autoencoder und HDBSCAN '
        'lernen nur am Training. Die Auswahl endet vor der Testauswertung.</p>'
        '<p><strong>Dieser Versuch misst Übereinstimmung mit vorhandenen Attributwerten, keine echte '
        'Fehlererkennungsrate.</strong> Ein fehlendes optionales Attribut ist kein bestätigter Fehler. '
        'Die Datei wurde zuvor explorativ betrachtet; der Vergleich ersetzt keine neue unabhängige Lieferung.</p>')
    content += '<h2>Ergebnis je Aufteilung</h2><p>Abdeckung bezieht sich auf Ziel-Datensatz-Paare. '
    content += 'Bei sechs Zielen zählt ein Objekt sechs Mal. „Im Test gestützt“ verlangt mindestens fünf '
    content += 'anwendbare Zeilen aus fünf Gruppen und mindestens 95 % Übereinstimmung. Keine Nachauswahl im Test.</p>'
    content += '<div class="scroll"><table><thead><tr><th>Seed</th><th>Variante</th><th>Ausgewählte Regeln</th>'
    content += '<th>Im Test gestützt</th><th>Zu wenig Testbelege</th><th>Abdeckung</th><th>Übereinstimmung¹</th>'
    content += '<th>Abweichende Werte</th><th>Fehlendes Ziel</th></tr></thead><tbody>'
    for m in summary["method_metrics"]:
        a = m["test_rule_assessments"]
        content += '<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in [m["seed"], LABELS[m["method"]],
            m["selected_rule_count"], a["supported"], a["insufficient_support"], percent(m["coverage"]),
            percent(m["agreement_all_covered"]), m["disagree_observed"], m["missing_target_covered"]]) + '</tr>'
    content += '</tbody></table></div><p>¹ Anteil bestätigender Werte an allen abgedeckten Fällen, einschließlich '
    content += 'fehlender Ziele im Nenner. Werte nur für vorhandene Ziele und gruppengewichtete Ergebnisse '
    content += 'stehen in den Details. Viele Regeln allein belegen keinen Mehrwert.</p>'
    content += '<h2>Was kommt gegenüber der Vergleichsvariante hinzu?</h2><div class="scroll"><table>'
    content += '<tr><th>Seed</th><th>Vergleich</th><th>Zusätzlich korrekt abgedeckt</th><th>Zusätzlich abweichend</th>'
    content += '<th>Zusätzlich Ziel fehlt</th><th>Nur Referenz korrekt abgedeckt</th></tr>'
    for r in summary["paired_comparisons"]:
        content += '<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in [r["seed"],
            LABELS[r["method"]] + ' gegenüber ' + LABELS[r["reference"]], r["additional_agree"],
            r["additional_disagree_observed"], r["additional_missing_target"], r["reference_only_agree"]]) + '</tr>'
    content += '</table></div><p>Die Seeds sind wiederholte Aufteilungen desselben Bestands. Die Zeilen werden '
    content += 'nicht als unabhängige Versuche zusammengerechnet. Ein Vorteil muss Abdeckung und Übereinstimmung '
    content += 'gemeinsam berücksichtigen; diese Tabelle erklärt keinen automatischen Sieger.</p>'
    content += '<h2>Ausgewählte Regeln und ihre Testbelege</h2><p>Lokale Regeln gelten nur nach Zuordnung durch '
    content += 'den zugehörigen eingefrorenen Router. Ihre IF/THEN-Teile dürfen nicht global verwendet werden.</p>'
    audit_map = {a["rule_id"]: a for a in audits}
    selected = [r for r in candidates if r["selected_on_validation"]]
    if not selected:
        content += '<p>Keine Regel bestand die vorab festgelegten Validierungsbedingungen.</p>'
    for r in selected:
        content += '<article><h3>' + escape(r["rule_id"]) + '</h3><p>' + escape(r["expression"]) + '</p>'
        content += details_json({"selection": r, "test": audit_map[r["rule_id"]]}, "Bedingungen und Belege") + '</article>'
    content += '<h2>Nachvollziehen</h2><ul>'
    links = (("summary.json", "Vergleich und Kennzahlen"), ("target_metrics.json", "Kennzahlen je Ziel"),
             ("all_candidates.json", "Alle Kandidaten mit Ablehnungsgründen"),
             ("all_model_reports.json", "Modellgüte und Mehrheitsbaseline"),
             ("test_rule_evidence.json", "Testbelege je ausgewählter Regel"),
             ("protocol.json", "Parameter und Datenherkunft"), ("PROTOKOLL.md", "Versuchsprotokoll"),
             ("selection_lock.json", "Vor Test eingefrorene Auswahl"),
             ("verification.json", "Technische Prüfungen"))
    for path, name in links:
        content += f'<li><a href="{path}">{escape(name)}</a></li>'
    content += '</ul>' + details_json(summary, "Sämtliche Kennzahlen und Aufteilungen")
    content += '<p>Je Seed und Ziel enthalten die Unterordner Vorverarbeitung, Autoencoder, HDBSCAN-Router, '
    content += 'Trainingsverlauf, konkrete Matrizen und Zuordnungen. Der tabellarische Encoder verwendet acht '
    content += 'latente Dimensionen und keine Zeichenkodierung administrativer Codes. Die Baumtiefe und '
    content += 'Clusterparameter sind vorgegeben. Dieser Versuch bewertet diese Konfiguration.</p>'
    html_report(out/"bericht.html", "Regeln: global, in Attributclustern und in Latentclustern", content)
