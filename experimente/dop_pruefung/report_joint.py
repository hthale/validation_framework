"""Offline-Bericht ohne externe Ressourcen."""
from html import escape
import json

from joint_protocol import METHOD_LABELS, OPERATORS, KNOWN


def table(headers, rows):
    def cell(value): return escape(str(value))
    return ('<div class="scroll"><table><thead><tr>' + ''.join('<th>'+cell(h)+'</th>' for h in headers)
            + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+cell(v)+'</td>' for v in row)+'</tr>' for row in rows)
            + '</tbody></table></div>')


def render(out, summary, protocol):
    from experiment_io import html_report, details_json
    from review_common import json_value
    text = ('<p class="notice">Dieser Versuch prüft, welche vorgegebenen und automatisch gelernten Regeln '
            'sowie welche LOF-Variante dieselben künstlichen Änderungen markieren. '
            'Die Quelldaten bleiben erhalten. Es werden keine fachlichen Entscheidungen eingetragen.</p>')
    text += ('<h2>Was dieses Ergebnis bedeutet</h2><p>Die vier Konsistenzprüfungen sind bewusst vorgegeben. '
             'Profiling- und Abhängigkeitsregeln erzeugt dein Framework aus den Trainingsdaten. '
             'Nur Kandidaten mit bestandenen Validierungsschwellen werden auf die Testkopie angewendet. '
             'Der latente Raum liefert zusätzlich die LOF-Repräsentation und die Trainingsstrata für das Profiling.</p>'
             '<p>„Markiert“ bedeutet ein Regelhinweis oder ein Platz im LOF-Prüfbudget. '
             '„Erst nach Änderung“ zählt geänderte Zeilen, die vorher bei dieser Methode nicht markiert waren. '
             'Eine neue einzelne Regelverletzung kann auch bei einer bereits markierten Zeile entstehen; '
             'diese Details stehen in case_results und rule_cases.</p>')
    mean = summary["mean_recall_of_changes"]
    delta = (mean["latent"] - mean["baseline"])*100
    text += f'<p>Mittlere Trefferquote der Änderungen im LOF-Budget: Eingangsmerkmale {100*mean["baseline"]:.1f} %, '
    text += f'latenter Raum {100*mean["latent"]:.1f} %; Differenz {delta:+.1f} Prozentpunkte. '
    text += 'Das ist eine Beschreibung dieser Wiederholungen, keine Signifikanzprüfung oder allgemeine Nutzenbehauptung.</p>'
    for rep in summary["repetitions"]:
        seed = rep["seed"]
        prefix = f"seed_{seed}/"
        text += f'<h2>Seed {seed}</h2>'
        text += table(["Teilmenge", "Zeilen", "Gruppen"], [(p, v["rows"], v["groups"]) for p, v in rep["split"].items()])
        text += (f'<p>{rep["changed_rows"]} geänderte Zeilen, davon {rep["input_invisible_changes"]} '
                 'durch die LOF-Eingangsaufbereitung unsichtbar. Diese bleiben im Nenner. '
                 f'Prüfbudget je LOF: {rep["lof"]["baseline"]["review_budget_k"]} Zeilen. '
                 f'Maximal mögliche LOF-Trefferquote bei diesem Budget: '
                 f'{100*rep["lof"]["baseline"]["maximum_possible_recall_at_budget"]:.1f} %.</p>')
        rows = []
        for method, label in METHOD_LABELS.items():
            m = rep["methods"][method]
            rows.append((label, f'{m["changed_selected_after"]}/{m["changed_rows"]}',
                         m["changed_selected_before"], m["newly_selected_changed"],
                         m["selected_after"], m["unchanged_selected_after"]))
        text += table(["Methode", "Änderungen markiert", "Änderungsfälle vorab markiert¹", "Erst nach Änderung",
                       "Alle markierten Zeilen", "Unveränderte markiert"], rows)
        text += ('<p>¹ Die Vorher-Spalte zählt alle später geänderten Kennungen, die vorab markiert waren; '
                 'sie müssen nachher nicht mehr markiert sein. Regeln haben kein festes Zeilenbudget. '
                 'Die Vereinigung mit LOF vergrößert den Prüfaufwand. Unveränderte markierte Zeilen sind keine bestätigten Fehlalarme.</p>')
        text += '<h3>Zusätzlicher Beitrag des latenten LOF</h3>'
        rows = []
        for key, label in (("latent_beyond_rules", "Zusätzlich zu allen Regeln"),
                           ("latent_beyond_rules_baseline", "Zusätzlich zu Regeln und Eingangs-LOF")):
            m = rep["incremental"][key]
            rows.append((label, m["additional_changed_rows"], m["newly_selected_additional_changed"],
                         m["additional_review_rows"], m["additional_unchanged_rows"]))
        text += table(["Vergleich", "Weitere Änderungen", "Davon erst nach Änderung ausgewählt", "Weitere Prüfzeilen", "Davon unverändert"], rows)
        text += '<h3>Nach Änderungsart</h3><p>Zellen: markierte Änderungen / erst nach Änderung markiert.</p>'
        rows = []
        for op in OPERATORS:
            m = rep["per_operator"][op["name"]]
            rows.append([op["label"], m["known"]["changed_rows"]] +
                        [f'{m[n]["changed_selected_after"]} / {m[n]["newly_selected_changed"]}'
                         for n in ("known", "profiling", "dependency", "baseline", "latent")])
        text += table(["Änderung", "Fälle", "Vorgegeben", "Profiling", "Abhängigkeit", "Eingangs-LOF", "Latenter LOF"], rows)
        text += '<h3>Automatisch erzeugte Regeln</h3>'
        cnt = rep["rule_counts"]
        text += f'<p>Profiling: {cnt["profiling_candidates"]} Kandidaten, {cnt["profiling_selected"]} ausgewählt. '
        text += f'Abhängigkeiten: {cnt["dependency_candidates"]} Kandidaten, {cnt["dependency_selected"]} ausgewählt. '
        text += 'Auch null ausgewählte Abhängigkeiten ist ein gültiger Befund; Schwellen wurden nicht gelockert.</p>'
        rules = json.loads((out / prefix / "rules.json").read_text(encoding="utf-8"))
        for kind, label in (("profiling", "Profiling"), ("dependency", "Abhängigkeiten")):
            text += f'<details><summary>{label}: alle Kandidaten mit Auswahlstatus</summary>'
            for rule in rules[kind+"_candidates"]:
                meta = rule["metadata"]
                selected = meta.get("eligible_for_joint_check", meta.get("eligible_for_review", False))
                metric = meta["validation"]
                text += '<article><strong>'+escape(rule["rule_id"])+': '+('ausgewählt' if selected else 'nicht ausgewählt')+'</strong>'
                text += '<p>'+escape(rule["expression"])+'</p>'
                rate = metric["pass_rate"]
                text += f'<p>Validierung: {metric["coverage_count"]} anwendbare Fälle; Passrate '+('nicht bestimmbar' if rate is None else f'{100*rate:.1f} %')+'.</p>'
                text += details_json(json_value(rule), "Vollständige Regel")+'</article>'
            text += '</details>'
        text += details_json(rep["overlap_after"], "Paarweise Überschneidungen")
        text += details_json({"used": rep["used_lof_columns"], "unused": rep["unused_lof_columns"]}, "Berücksichtigte und unberücksichtigte LOF-Felder")
        text += '<p>'+' · '.join(f'<a href="{prefix}{name}">{name}</a>' for name in
                ("changes.json", "case_results.csv", "rules.json", "metrics.json", "split.csv"))+'</p>'
    text += '<h2>Vorgegebene Testannahmen</h2>'+table(["Prüfung", "Annahme"], [(r["rule_id"], r["expression"]) for r in KNOWN])
    text += '<h2>Grenzen und Nachvollziehbarkeit</h2><ul>'+''.join('<li>'+escape(v)+'</li>' for v in protocol["limitations"] )+'</ul>'
    text += details_json({k:v for k,v in protocol.items() if k != "code_at_start"}, "Vorab festgelegtes Versuchsprotokoll")
    text += '<p>'+' · '.join(f'<a href="{name}">{name}</a>' for name in ("summary.json", "protocol.json", "run_manifest.json"))+'</p>'
    html_report(out / "bericht.html", "Gemeinsamer DOP-Prüfversuch 09", text)
