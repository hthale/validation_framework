"""Bahnverkehr ausschließlich über Attribute mit dem vorhandenen Framework analysieren."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import subprocess
import sys


def bootstrap(project):
    project = Path(project).resolve(strict=True)
    required = ["run_generic.py", "generic_schema.py", "generic_rules.py", "generic_clusters.py",
                "review_common.py", "experiment_io.py", "latentspace/main_v2.py"]
    missing = [p for p in required if not (project / "src" / p).is_file()]
    if missing:
        raise ValueError(f"Vorhandenes Framework mit Paket 07/08 erforderlich. Fehlend: {missing}")
    sys.path.insert(0, str(project / "src"))
    return project


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path("."))
    p.add_argument("--input", type=Path, default=Path("data/parquetdata/bahnverkehr.parquet"))
    p.add_argument("--output", type=Path, default=Path("data/bahnverkehr_attribute_10"))
    p.add_argument("--rows", type=int, default=5000, help="Reproduzierbare Stichprobe; 0 = gesamter Bestand.")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=2)
    return p


def hashes(root):
    from review_common import file_sha256
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(root.rglob("*.py"))}


def render_report(out, preparation, analysis, checks, relative_analysis):
    from experiment_io import html_report
    from review_common import read_json
    e = lambda v: escape(str(v))
    full = {x["column"]: x for x in preparation["full_population_statistics"]}
    rows = ""
    for item in preparation["sample_statistics"]:
        c = item["column"]
        rows += "<tr>" + "".join(f"<td>{e(v)}</td>" for v in (
            c, preparation["roles"][c]["role"], full[c]["nonmissing"], item["nonmissing"],
            item["distinct"], item["missing"], ", ".join(preparation["unseen_in_sample"].get(c, [])))) + "</tr>"
    catalog = read_json(out / relative_analysis / "rule_catalog.json")
    links = [(relative_analysis + "/bericht.html", "Regeln, Clusterbeschreibungen und Auffälligkeiten öffnen"),
             (relative_analysis + "/umap.png", "UMAP-Darstellung öffnen"),
             (relative_analysis + "/rule_catalog.json", "Alle Regelkandidaten"),
             (relative_analysis + "/feature_contract.json", "Tatsächlich verwendete Modellmerkmale"),
             ("preparation.json", "Aufbereitung und ausgeschlossene Spalten"),
             ("summary.json", "Zusammenfassung"), ("verification.json", "Technische Prüfungen")]
    examples = "".join("<article><p>" + e(r.get("display_expression", r["expression"])) + "</p></article>"
                       for r in catalog["cluster_descriptions"][:5])
    html = (
        "<p>Reine Attributanalyse von Bahnverkehr. Geometrien, Koordinaten und abgeleitete "
        "Geometriemerkmale sind ausgeschlossen.</p>"
        f"<p><strong>{preparation['selected_rows']:,} von {preparation['source_rows']:,} Objekten</strong>; "
        f"{analysis['epochs']} Trainingsepochen; {analysis['latent_dimensions']} latente Dimensionen.</p>"
        "<div class='notice'>Dies ist eine explorative Bestandsanalyse. Autoencoder, LOF, Profiling und "
        "Clusterbeschreibungen verwenden die gewählte Stichprobe. Nur die globalen Abhängigkeiten besitzen "
        "eine getrennte Trainings-/Validierungsaufteilung mit reserviertem Testanteil. Die Duplikatgruppierung "
        "bestätigt keine räumliche, zeitliche oder lieferungsbezogene Unabhängigkeit.</div>"
        "<h2>Ergebnisse öffnen</h2><ul>" + "".join(f'<li><a href="{e(h)}">{e(t)}</a></li>' for h,t in links) + "</ul>"
        "<h2>Was wurde gefunden?</h2><table><tr><th>Ergebnis</th><th>Anzahl</th><th>Herkunft</th></tr>"
        f"<tr><td>Globale Abhängigkeitskandidaten</td><td>{analysis['ml_candidate_count']}</td>"
        "<td>Entscheidungsbäume auf Originalattributen nach Aufbereitung</td></tr>"
        f"<tr><td>Davon mit bestandenen Schwellen</td><td>{analysis['eligible_ml_rule_count']}</td><td>Separate Validierung; fachlich offen</td></tr>"
        f"<tr><td>Profiling-Vorschläge</td><td>{analysis['profiling_rule_count']}</td><td>Stratifizierte Stichproben mit LOF und Clustern</td></tr>"
        f"<tr><td>Cluster / Beschreibungen</td><td>{analysis['cluster_count']} / {analysis['cluster_description_count']}</td>"
        "<td>HDBSCAN im latenten Raum; nachträgliche Beschreibung mit Attributbäumen</td></tr></table>"
        "<p>Die Clusterbeschreibungen hängen vom latenten Raum ab. Sie sind Beschreibungen, keine "
        "bestätigten Qualitätsregeln. Eine große Anzahl ist kein Gütenachweis. Gruppenspezifische "
        "Attributabhängigkeiten und ein Vergleich mit einer Variante ohne latenten Raum gehören zum "
        "nachfolgenden Versuch und werden hier noch nicht berechnet.</p>"
        "<h2>Beispiele für Clusterbeschreibungen</h2>" + (examples or "<p>Keine beschreibbaren Cluster gefunden.</p>") +
        "<h2>Merkmale und Belegung</h2><p>code:&quot;2322&quot; bezeichnet den vorhandenen Code 2322. "
        "liste:[&quot;1100&quot;] bezeichnet die vollständige vorhandene Codeliste. Die Vorsilben verhindern "
        "eine numerische Interpretation; sie führen keine fachlichen Regeln oder Labels ein. "
        "Leere Listen und fehlende Listen bleiben unterscheidbar. Die unveränderten ausgewählten "
        "Quellattribute stehen in source_attributes.parquet.</p>"
        "<div class='scroll'><table><tr><th>Merkmal</th><th>Rolle</th><th>Belegt im Gesamtbestand</th>"
        "<th>Belegt in Stichprobe</th><th>Ausprägungen in Stichprobe</th><th>Fehlend in Stichprobe</th>"
        "<th>In Stichprobe fehlende Codeausprägungen</th></tr>" + rows + "</table></div>"
        f"<p>{preparation['distinct_attribute_combinations']} verschiedene vollständige Attributkombinationen "
        "in der Stichprobe. Seltene Codes können in einer zufälligen Stichprobe fehlen.</p>"
        "<p>Fehlende kategoriale Werte erhalten im vorhandenen Encoder eine eigene Kategorie. "
        "Prüfzeitpunkte werden in UTC-Tage umgerechnet und bei fehlenden Werten imputiert; ihre "
        "Belegung wird zusätzlich als eigenes Merkmal erhalten. Konstante und vollständig leere "
        "Merkmale werden vom vorhandenen Lernverfahren ausgelassen. Administratives Datum und Anlass "
        "können Bearbeitungsabläufe statt fachlicher Produktunterschiede abbilden.</p>"
        f"<p>{len(checks)} technische Prüfungen bestanden. Keine Fehlerlabels, Fehlerinjektionen oder "
        "automatischen Freigaben. Originaldatei und Projektquellcode sind unverändert.</p>"
    )
    html_report(out / "bericht.html", "Bahnverkehr · Attributanalyse", html)


def analyze(args):
    if not __debug__:
        raise ValueError("Bitte ohne Python-Option -O starten, damit alle Prüfungen aktiv sind.")
    project = bootstrap(args.project)
    from attribute_prep import (read_attributes, prepare_attributes, select_sample, column_statistics,
                                check_contract, SCALAR_CODES, LIST_CODES)
    from generic_schema import infer_columns
    from review_common import file_sha256, write_json, read_json
    from experiment_io import finish_output
    from verify_bahn_run import verify_analysis

    if args.epochs < 1 or args.threads < 1:
        raise ValueError("Positive Epochenzahl und Threadzahl erforderlich.")
    source = (project / args.input).resolve(strict=True)
    source_hash = file_sha256(source)
    project_hash = hashes(project / "src")
    package_hash = hashes(Path(__file__).resolve().parent)
    print("[01_aufbereitung] Lese ausschließlich ausgewählte Attribute ...", flush=True)
    source_frame, preparation = read_attributes(source)
    prepared_full = prepare_attributes(source_frame)
    sample = select_sample(prepared_full, args.rows, args.seed)
    if len(sample) < 12:
        raise ValueError("Mindestens 12 Datensätze erforderlich.")
    if len(sample.drop(columns="gml_id").drop_duplicates()) < 6:
        raise ValueError("Mindestens sechs unterschiedliche Attributkombinationen für den gemeinsamen Lauf erforderlich.")
    contract = infer_columns(sample, excluded=["gml_id"])
    check_contract(sample, contract)
    original_sample = source_frame.set_index("gml_id", drop=False).loc[sample.gml_id].reset_index(drop=True)
    unseen = {c: sorted(set(prepared_full[c].dropna())-set(sample[c].dropna()))
              for c in (*SCALAR_CODES, *LIST_CODES) if c in sample}
    preparation.update(source_path=str(source), input_sha256=source_hash, source_rows=len(source_frame),
        selected_rows=len(sample), requested_rows=args.rows, seed=args.seed,
        sample_policy="SHA256(seed,gml_id), kleinste Hashes; unabhängig von Werten, Scores und Quellreihenfolge",
        identity_columns=["gml_id"], geometry_used=False,
        full_population_statistics=column_statistics(prepared_full), sample_statistics=column_statistics(sample),
        unseen_in_sample=unseen, roles=contract["columns"],
        distinct_attribute_combinations=len(sample.drop(columns="gml_id").drop_duplicates()),
        scope="Explorative Bestandsanalyse; keine Fehlererkennungsleistung und kein Nutzenvergleich.",
        field_selection="Dokumentierte produktspezifische Auswahl; keine Pflichtbelegungen oder Wertearten vorgegeben.",
        date_missing_flag="Belegung des Prüfzeitpunkts; keine zusätzliche Kopie eines ML-Zielwertes.",
        project_source_at_start=project_hash, package_source_at_start=package_hash)
    out = (project / args.output).resolve() / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "run_status.json", {"status":"running", "stage":"preparation"})
    try:
        original_sample.to_parquet(out / "source_attributes.parquet", index=False)
        sample.to_parquet(out / "attributes.parquet", index=False)
        write_json(out / "preparation.json", preparation)
        print(f"{len(sample)} von {len(source_frame)} Objekten; keine Geometriemerkmale.", flush=True)
        command = [sys.executable, "-u", str(project / "src/run_generic.py"),
                   "--input", str(out / "attributes.parquet"), "--output", str(out / "analyse"),
                   "--epochs", str(args.epochs), "--seed", str(args.seed), "--threads", str(args.threads),
                   "--id-columns", "gml_id"]
        write_json(out / "invocation.json", {"command": command, "working_directory": str(project)})
        print("[02_framework] Starte den vorhandenen generischen Projektcode ...", flush=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        with (out / "framework.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return_code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if return_code:
            raise RuntimeError(f"Framework beendet mit Status {return_code}. Siehe {out / 'framework.log'}")
        candidates = sorted((out / "analyse").iterdir())
        if len(candidates) != 1:
            raise ValueError("Genau ein neuer Analyselauf erwartet.")
        run = candidates[0]
        relative = run.relative_to(out).as_posix()
        checks = verify_analysis(out, run, preparation, expected_epochs=args.epochs)
        if file_sha256(source) != source_hash or hashes(project / "src") != project_hash:
            raise ValueError("Originaleingabe oder Projektquellcode wurde während des Laufs verändert.")
        if hashes(Path(__file__).resolve().parent) != package_hash:
            raise ValueError("Paketquellcode wurde während des Laufs verändert.")
        checks.extend(["Originaldatei unverändert", "Projekt- und Paketquellcode unverändert"])
        summary = read_json(run / "summary.json")
        result = {**{k: summary[k] for k in ("row_count", "epochs", "ml_candidate_count", "eligible_ml_rule_count",
                    "profiling_rule_count", "cluster_count", "cluster_description_count", "noise_count")},
                  "source_rows": len(source_frame), "geometry_used": False,
                  "analysis_directory": relative, "technical_check_passed": True,
                  "checks_passed": len(checks), "independent_evaluation_performed": False,
                  "local_dependency_learning_performed": False,
                  "latent_benefit_demonstrated": None, "scope": preparation["scope"]}
        write_json(out / "summary.json", result)
        write_json(out / "verification.json", {"passed": True, "checks": checks})
        render_report(out, preparation, summary, checks, relative)
        finish_output(out, {"kind":"bahn_attributes_10", "input_sha256":source_hash,
                            "geometry_used":False, "package_source_sha256":package_hash})
        print("[03_abgeschlossen]", flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print(f"Ergebnisordner: {out}\nBericht: {out / 'bericht.html'}", flush=True)
        return out
    except BaseException as exc:
        write_json(out / "run_status.json", {"status":"failed", "error":f"{type(exc).__name__}: {exc}"})
        raise


def main():
    p = parser()
    try:
        analyze(p.parse_args())
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        p.exit(2, f"Attributanalyse abgebrochen: {exc}\n")


if __name__ == "__main__":
    main()
