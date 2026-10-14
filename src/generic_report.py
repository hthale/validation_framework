"""Offline-Bericht: Regelkandidaten und Auffälligkeiten als gleichwertige Ergebnisse."""
from __future__ import annotations

from html import escape
import json
from decimal import Decimal
from datetime import time

import numpy as np

from review_common import atomic_text, json_value
from rule_engine import evaluate_rule_on_dataframe
from generic_clusters import complexity_html


def original_value(value):
    """Zusätzliche Parquet-Skalare verlustfrei für die Anzeige beschreiben."""
    if isinstance(value, Decimal):
        return {"encoding": "decimal", "value": str(value)}
    if isinstance(value, time):
        return {"encoding": "time", "value": value.isoformat()}
    if isinstance(value, dict):
        return {str(k): original_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [original_value(v) for v in value]
    return json_value(value)


def build_cases(raw, normalized, latent, profile_rules, dependencies):
    raw_by_key = raw.set_index("record_key", drop=False)
    partitions = dependencies["split"].set_index("record_key")["partition"].to_dict() if dependencies["split"] is not None else {}
    cases = {}
    for row in latent.to_dict("records"):
        key = row["record_key"]
        cases[key] = {"record_key": key, "rank": int(row["outlier_rank"]), "lof": float(row["lof_raw"]),
                      "cluster": int(row["cluster_label"]), "partition": partitions.get(key, "not_split"),
                      "decision": "unreviewed", "findings": [], "raw_values": original_value(raw_by_key.loc[key].to_dict())}
    assessments, issues = [], []
    def apply(frame, rules, family):
        for rule in rules:
            try:
                applies, valid, invalid = evaluate_rule_on_dataframe(frame, rule)
            except (ValueError, TypeError, KeyError) as exc:
                issues.append({"rule_id": rule["rule_id"], "error": str(exc)})
                continue
            assessments.append({"rule_id": rule["rule_id"], "family": family,
                                "population_count": len(frame), "coverage_count": int(applies.sum()),
                                "invalid_count": int(invalid.sum()), "pass_rate": float(valid.sum()/applies.sum()) if applies.any() else None})
            for idx in frame.index[invalid]:
                row = frame.loc[idx]
                cases[row["record_key"]]["findings"].append({"rule_id": rule["rule_id"], "family": family,
                    "expression": rule["expression"], "evaluated_values": json_value({c: row[c] for c in rule["columns"] if c in row})})
    apply(normalized, profile_rules, "profiling")
    if dependencies["split"] is not None:
        keys = set(dependencies["split"].loc[dependencies["split"]["partition"].eq("validation"), "record_key"])
        frame = dependencies["evaluation_frame"]
        apply(frame.loc[frame["record_key"].isin(keys)], dependencies["rules"], "dependency_validation_only")
    return sorted(cases.values(), key=lambda x: x["rank"]), assessments, issues


def render_report(path, summary, contract, candidates, profiles, clusters, cases, reports):
    def e(value): return escape(str(value), quote=True)
    def dump(value): return e(json.dumps(json_value(value), ensure_ascii=False, indent=2, allow_nan=False))
    def card(rule, family):
        meta = rule.get("metadata", {})
        validation = meta.get("validation", {})
        if family == "Abhängigkeit":
            label = "Güteschwellen bestanden; fachlich offen" if meta.get("eligible_for_review") else "Dokumentierter Kandidat; Schwellen nicht bestanden"
            metrics = f"Validierung: {validation.get('coverage_count', 0)} anwendbare Fälle; Passrate {validation.get('pass_rate')}"
        elif family == "Clusterbeschreibung":
            label, metrics = "Beschreibung vorhandener Cluster", "Keine Validierungsregel; Anpassung und Beschreibung im selben Bestand."
        else:
            label, metrics = "Profiling-Vorschlag", "Im vorhandenen Bestand beobachtet; keine bestätigte Produktanforderung."
        expression = rule.get("display_expression", rule["expression"])
        return f'<article class="filterable"><p class="tag">{e(family)} · {e(label)}</p><h3>{e(expression)}</h3><p>{e(metrics)}</p><details><summary>Herkunft, Bedingungen und Kennzahlen</summary><pre>{dump(rule)}</pre></details></article>'
    cards = ''.join(card(r, "Abhängigkeit") for r in sorted(candidates, key=lambda r: not r.get("metadata", {}).get("eligible_for_review", False)))
    cards += ''.join(card(r, "Profiling") for r in profiles)
    cluster_cards = ''.join(card(r, "Clusterbeschreibung") for r in clusters)
    rows = []
    for case in cases:
        details = ''.join(f'<li>{e(f["expression"])} <small>({e(f["family"])})</small></li>' for f in case["findings"])
        rows.append(f'<tr class="filterable"><td>{case["rank"]}</td><td>{case["lof"]:.4f}</td><td>{case["cluster"]}</td><td>{e(case["partition"])}</td><td>{len(case["findings"])}</td><td><details><summary>{e(case["record_key"])}</summary><ul>{details}</ul><pre>{dump(case["raw_values"])}</pre></details></td></tr>')
    schema_rows = ''.join(f'<tr class="filterable"><td>{e(name)}</td><td>{e(info["role"])}</td><td>{e(info["reason"])}</td></tr>' for name, info in contract['columns'].items())
    counts = [("Datensätze", summary["row_count"]), ("Abhängigkeitskandidaten", len(candidates)),
              ("Davon mit bestandenen Schwellen", summary["eligible_ml_rule_count"]),
              ("Profiling-Vorschläge", len(profiles)), ("Clusterbeschreibungen", len(clusters))]
    counters = ''.join(f'<div class="counter"><strong>{v}</strong><span>{e(k)}</span></div>' for k,v in counts)
    payload = json.dumps(json_value({"summary": summary, "case_count": len(cases), "rule_ids": [r["rule_id"] for r in profiles+candidates+clusters]}), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")
    html = '''<!doctype html><html lang="de"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Automatische Analyse · Regeln und Auffälligkeiten</title><style>
:root{font-family:system-ui,sans-serif;color:#17323b;background:#f3f6f7}body{margin:0}header{background:#153e44;color:white;padding:30px max(5vw,20px)}main{max-width:1360px;margin:auto;padding:24px}h1{font-size:27px;margin:0 0 10px}h2{font-size:23px}h3{font-size:16px;overflow-wrap:anywhere}.counters{display:flex;gap:12px;flex-wrap:wrap}.counter,article{background:white;border:1px solid #d6e0e3;border-radius:9px;padding:18px}.counter{flex:1;min-width:145px}.counter strong{display:block;font-size:26px}.counter span,small,.tag{font-size:13px;color:#496773}article{margin:12px 0}input{font:inherit;padding:12px;border:1px solid #819b9f;border-radius:6px;width:min(90%,600px)}nav{display:flex;gap:20px;flex-wrap:wrap;margin:24px 0}a{color:#106376}table{border-collapse:collapse;width:100%;background:white}td,th{text-align:left;border-bottom:1px solid #dae3e5;padding:11px;vertical-align:top}td{overflow-wrap:anywhere}pre{white-space:pre-wrap;word-break:break-word;max-height:440px;overflow:auto;font-size:12px}summary{cursor:pointer}.notice{background:#e4eff0;padding:16px;border-left:4px solid #287878;line-height:1.6}.scroll{overflow:auto}footer{padding:24px;font-size:13px}button{font:inherit}section{margin-top:35px}li{margin:8px 0}
</style><header><h1>Automatische Analyse</h1><p>Regelkandidaten, Clusterbeschreibungen und Auffälligkeiten aus einer Merkmalstabelle.</p></header><main>'''
    html += '<div class="counters">' + counters + '</div>' + f'<p class="notice">{e(summary["scope_note"])}<br>Gruppierung: {e(summary["grouping"]["mode"])}. {e(summary["grouping"]["note"])}</p>'
    html += '<nav><a href="#regeln">Regelkandidaten</a><a href="#cluster">Clusterbeschreibungen</a><a href="#faelle">Auffälligkeiten</a><a href="#merkmale">Merkmalsauswahl</a><a href="#dateien">Ergebnisdateien</a></nav><label for="search">Einträge durchsuchen</label><br><input id="search" placeholder="Merkmal, Regeltext oder Kennung"><p id="search-status" aria-live="polite"></p>'
    html += '<section id="regeln"><h2>Automatisch erzeugte Regelkandidaten</h2><p>Alle Kandidaten werden dokumentiert. Bestandene Güteschwellen sind keine fachliche Freigabe.</p>' + (cards or '<p>Keine Kandidaten unter den verwendeten Voraussetzungen gefunden.</p>') + '</section>'
    html += '<section id="cluster"><h2>Clusterbeschreibungen aus dem latenten Raum</h2>' + complexity_html(reports) + (cluster_cards or '<p>Keine beschreibbaren Cluster mit ausreichender Belegung gefunden.</p>') + '<details><summary>Modellberichte</summary><pre>' + dump(reports) + '</pre></details></section>'
    html += '<section id="faelle"><h2>Auffälligkeiten und Regelabweichungen</h2><p>Sortierung nach LOF. Ein hoher Wert ist keine Fehlerwahrscheinlichkeit. Alle Fälle sind fachlich offen. Abhängigkeitskandidaten werden nur im Validierungsanteil angewendet; Profiling beschreibt den gesamten Bestand.</p><div class="scroll"><table><thead><tr><th>Rang</th><th>LOF</th><th>Cluster</th><th>Teilmenge</th><th>Regelhinweise</th><th>Kennung und Originalwerte</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div></section>'
    html += '<section id="merkmale"><h2>Automatische Merkmalsauswahl</h2><p>Die folgende Auswahl gehört zur Bestandsanalyse. Der gesonderte Vertrag für das Abhängigkeitslernen wird ausschließlich auf dessen Trainingsanteil bestimmt und separat gespeichert.</p><table><thead><tr><th>Spalte</th><th>Rolle</th><th>Begründung</th></tr></thead><tbody>' + schema_rows + '</tbody></table></section>'
    html += '<section id="dateien"><h2>Ergebnisdateien</h2><ul>' + ''.join(f'<li><a href="{e(name)}">{e(name)}</a></li>' for name in ("summary.json", "input_with_ids.parquet", "feature_contract.json", "dependency_feature_contract.json", "rule_catalog.json", "dependency_reports.json", "dependency_split_summary.json", "cluster_reports.json", "latent_outlier_results_lof.parquet", "top_outliers.csv", "normalization_issues.json", "umap.png", "run_manifest.json")) + '</ul></section>'
    html += f'<script type="application/json" id="analysis-data">{payload}</script>'
    html += '''</main><footer>Prototypischer Analysebericht. Keine automatische Fachfreigabe und kein bestätigter Fehlerstatus. Entscheidungen aus früheren Prüfansichten werden nicht übernommen.</footer><script>
const input=document.getElementById('search');input.addEventListener('input',()=>{const q=input.value.toLocaleLowerCase();let n=0;document.querySelectorAll('.filterable').forEach(e=>{const match=e.textContent.toLocaleLowerCase().includes(q);e.hidden=!match;if(match)n++;});document.getElementById('search-status').textContent=q?`${n} passende Einträge`:'';});
</script></html>'''
    atomic_text(path, html)
