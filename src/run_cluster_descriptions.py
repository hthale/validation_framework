"""Cluster eines abgeschlossenen generischen Laufs neu beschreiben; kein AE-Training."""
from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import pandas as pd

from experiment_io import new_output, finish_output, read_verified_run, html_report, details_json
from generic_clusters import describe_clusters, cluster_frame, complexity_html
from generic_schema import normalize_columns
from phase2_settings import Phase2Settings
from review_common import file_sha256, write_json


def analyze(args):
    run = args.run.resolve(strict=True)
    names = ("input_with_ids.parquet", "feature_contract.json", "latent_outlier_results_lof.parquet",
             "model_config.json", "cluster_reports.json")
    hashes = read_verified_run(run, names)
    contract = json.loads((run / "feature_contract.json").read_text(encoding="utf-8"))
    raw = pd.read_parquet(run / "input_with_ids.parquet")
    latent = pd.read_parquet(run / "latent_outlier_results_lof.parquet")
    normalized, issues = normalize_columns(raw, contract)
    # Prüft eindeutige, vollständige Kennungen, bevor Ausgabeverzeichnisse entstehen.
    work, _ = cluster_frame(normalized, contract, latent)
    pd.testing.assert_frame_equal(raw.sort_values("record_key").reset_index(drop=True),
        latent[raw.columns].sort_values("record_key").reset_index(drop=True), check_dtype=False)
    config = json.loads((run / "model_config.json").read_text(encoding="utf-8"))
    seed = int(config["configuration"]["random_state"])
    old = json.loads((run / "cluster_reports.json").read_text(encoding="utf-8"))
    out = new_output(args.output)
    try:
        rules, reports = describe_clusters(normalized, contract, latent, Phase2Settings(random_state=seed))
        work.to_parquet(out / "cluster_evaluation_frame.parquet", index=False)
        write_json(out / "cluster_descriptions.json", rules)
        write_json(out / "cluster_reports.json", reports)
        write_json(out / "previous_cluster_reports.json", old)
        write_json(out / "normalization_issues.json", issues)
        report = reports[0]
        summary = {"source_run": str(run), "row_count": len(raw), "autoencoder_training_performed": False,
                   "cluster_labels_changed": False, "lof_ranking_changed": False,
                   "dependency_rules_changed": False, "source_files_unchanged": True,
                   "cluster_description_count": len(rules), "selected_variant": report["selected_variant"],
                   "in_sample_fidelity": report.get("in_sample_fidelity"),
                   "fidelity_target_met": report["fidelity_target_met"], "approved_rule_count": 0,
                   "independent_validation": False}
        write_json(out / "summary.json", summary)
        content = ('<p class="notice">Neue Beschreibungen der bereits vorhandenen Cluster. '
                   'Autoencoder, Clusterlabels, LOF-Rangfolge und Attributabhängigkeiten bleiben unverändert. '
                   'Anpassung und Bewertung erfolgen auf denselben Daten; keine unabhängige Güteprüfung.</p>'
                   + complexity_html(reports) + '<h2>Ausgewählte Beschreibungen</h2>')
        for rule in rules:
            content += ('<article><h3>' + escape(rule.get("display_expression", rule["expression"])) + '</h3>'
                        + details_json(rule, "Bedingungen, Herkunft und Kennzahlen") + '</article>')
        content += '<p>Datumsschwellen in der Anzeige sind angenähert. Die exportierten numerischen Bedingungen, einschließlich Imputation und float32-Konvertierung, sind verbindlich.</p>'
        content += details_json(old, "Vorheriger Clusterbericht") + details_json(reports, "Neuer Clusterbericht")
        content += '<h2>Dateien</h2><ul>' + ''.join(f'<li><a href="{name}">{name}</a></li>' for name in (
            "summary.json", "cluster_reports.json", "cluster_descriptions.json", "cluster_evaluation_frame.parquet",
            "previous_cluster_reports.json", "run_manifest.json")) + '</ul>'
        html_report(out / "bericht.html", "Clusterbeschreibungen · Vergleich", content)
        if any(file_sha256(run / n) != h for n, h in hashes.items()):
            raise ValueError("Quelllauf während der Auswertung verändert.")
        finish_output(out, {"source_run": str(run), "source_run_sha256": hashes, "protocol": "descriptive_surrogate_08"})
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Ergebnisordner: {out}\nBericht: {out / 'bericht.html'}")
        return out
    except Exception as exc:
        write_json(out / "run_status.json", {"status": "failed", "error": str(exc)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/cluster_descriptions_08")
    args = p.parse_args()
    try: analyze(args)
    except (ValueError, OSError, KeyError, AssertionError) as exc:
        p.exit(2, f"Clusterbeschreibung abgebrochen: {exc}\n")


if __name__ == "__main__":
    main()
