"""Lesbare Exporte und lokale Prüfansicht aus unveränderlichen Hinweisen."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from review_common import atomic_text, json_value, write_json
from review_decisions import replay_history

CASE_COLUMNS = ["record_key", "display_name", "outlier_rank", "lof_raw", "lof_z", "lof_review_selected",
                "cluster_label", "partition", "domain_deviation_count", "profiling_deviation_count",
                "dependency_deviation_count", "dependency_evaluation_status", "naming_observation_count",
                "decision", "reason", "reviewer", "reviewed_at", "revision"]


def csv_text(rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", delimiter=";")
    writer.writeheader()
    for row in rows:
        clean = {}
        for key in columns:
            value = row.get(key)
            # CSV ist eine Anzeige. Potenzielle Tabellenformeln bleiben in JSON unverändert erhalten.
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                value = "'" + value
            if isinstance(value, (list, dict)):
                value = json.dumps(json_value(value), ensure_ascii=False)
            clean[key] = value
        writer.writerow(clean)
    return "\ufeff" + stream.getvalue()


def build_html(bundle, history, state, review_dir=""):
    template = Path(__file__).with_name("review_template.html").read_text(encoding="utf-8")
    value = {"bundle": bundle, "revision": history["revision"], "state": state, "review_dir": review_dir}
    payload = json.dumps(json_value(value), ensure_ascii=False, allow_nan=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return template.replace("__REVIEW_DATA__", payload)


def write_views(directory, bundle, history):
    directory = Path(directory)
    state = replay_history(bundle, history)
    cases = [{**r, **state.get(r["record_key"], {"decision": "unreviewed", "reason": "", "reviewer": "", "reviewed_at": "", "revision": 0})} for r in bundle["cases"]]
    counts = {decision: sum(r["decision"] == decision for r in cases)
              for decision in ("unreviewed", "confirmed_error", "plausible", "needs_clarification")}
    summary = {**bundle["summary"], "bundle_id": bundle["bundle_id"], "review_revision": history["revision"],
               "review_counts": counts, "rule_approval_performed": False,
               "note": "Plausibel bezieht sich auf die geprüften Hinweise; keine vollständige Fehlerfreiheit des Datensatzes nachgewiesen."}
    write_json(directory / "review_summary.json", summary)
    write_json(directory / "review_state.json", {"bundle_id": bundle["bundle_id"], "revision": history["revision"], "decisions": list(state.values())})
    atomic_text(directory / "review_cases.csv", csv_text(cases, CASE_COLUMNS))
    findings = [{"record_key": r["record_key"], **f} for r in cases for f in r["findings"]]
    atomic_text(directory / "review_findings.csv", csv_text(findings, ["record_key", "finding_id", "category", "rule_id", "expression", "rule_status", "evidence_type", "values"]))
    atomic_text(directory / "review.html", build_html(bundle, history, state, str(directory.resolve())))
