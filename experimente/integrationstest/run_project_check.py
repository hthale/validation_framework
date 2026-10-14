"""Integrationstest ab Parquet über die normalen Programmeinstiege.

Im vorhandenen Projekt: python testpaket_06_projektlauf/run_project_check.py --project .
Kein Installer, kein Überschreiben von src/config.py, keine echten Lieferungen.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
from html import escape
from importlib import import_module
from importlib.metadata import version
from itertools import product
import json
import os
from pathlib import Path
import platform
import re
import runpy
import shutil
import subprocess
import sys
import time
import traceback


SCENARIO = "synthetic_project_integration_v1"
TARGETS = ("tif_band_count", "bodenpixelgroesse", "exportprofil")
CONTROL = "kontrollwert"
REQUIRED = ("check_foundation.py", "latentspace/main_v2.py", "latentspace/config_latent.py",
            "main_phase1_phase_2.py", "build_review.py", "review_decisions.py", "review_common.py",
            "phase2_settings.py", "review_template.html")
DEP_MODULES = ("numpy", "pandas", "sklearn", "torch", "hdbscan", "umap", "matplotlib", "joblib")
DIST_NAMES = ("numpy", "pandas", "scikit-learn", "torch", "hdbscan", "umap-learn", "matplotlib", "joblib")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_snapshot(source):
    return {p.relative_to(source).as_posix(): sha(p) for p in sorted(source.rglob("*"))
            if p.is_file() and p.suffix in {".py", ".html"}}


def prepare_imports(project):
    source = Path(project).resolve() / "src"
    missing = [name for name in REQUIRED if not (source / name).is_file()]
    if missing:
        raise ValueError("Projekt mit Korrekturpaketen 01–04 angeben. Fehlend: " + ", ".join(missing))
    # main_v2.py wird normalerweise direkt aus src/latentspace ausgeführt.
    for folder in (source / "latentspace", source):
        sys.path.insert(0, str(folder))
    return source


def expected_values(channel, detail):
    profiles = {("RGB", "standard"): "P_A", ("RGB", "fein"): "P_B",
                ("RGBI", "standard"): "P_C", ("RGBI", "fein"): "P_D"}
    pixel = 20 if detail == "standard" else 10
    return {"tif_band_count": 3 if channel == "RGB" else 4,
            "bodenpixelgroesse": pixel, "standardabweichung": pixel * 2,
            "exportprofil": profiles[(channel, detail)]}


def create_input():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    combinations = list(product(("RGB", "RGBI"), ("standard", "fein")))
    assignments = np.repeat(np.arange(4), 40)
    rng.shuffle(assignments)
    rows = []
    for group, index in enumerate(assignments):
        channel, detail = combinations[int(index)]
        common = {"delivery_id": SCENARIO, "source_folder": "synthetic_project_check",
                  "spektralkanaele": channel, "detailstufe": detail,
                  **expected_values(channel, detail), "koordinatenreferenzsystem_lage": 25832,
                  "koordinatenreferenzsystem_hoehe": 7837,
                  "koordinatenursprung_east": 380000 + (group % 20) * 2000,
                  "koordinatenursprung_north": 5800000 + (group // 20) * 2000,
                  "dateiformat": "Cloud Optimized GeoTIFF",
                  "zufall_zahl": float(rng.random())}
        first_label = int(rng.integers(0, 2))
        for observation in range(2):
            name = f"synthetic_{group:04d}_{observation}"
            rows.append({**common, "xml_name": name + ".xml", "tif_name": name + ".tif",
                         "kachelname": name, CONTROL: ("K_A", "K_B")[(first_label + observation) % 2]})
    frame = pd.DataFrame(rows)
    # Ein vorab bekannter Benennungshinweis soll bis zur Prüfansicht gelangen.
    frame.loc[0, "tif_name"] = "synthetic_0000_0_abweichung.tif"
    return frame


def run_worker(project, stage, settings_file):
    source = prepare_imports(project)
    options = read_json(settings_file)
    if stage == "preflight":
        for name in DEP_MODULES:
            import_module(name)
        import pandas as pd
        pd.io.parquet.get_engine("auto")
        # Diese Imports prüfen auch die internen Modell-/Regelabhängigkeiten.
        for name in ("latentspace.main_v2", "main_phase1_phase_2", "build_review"):
            import_module(name)
        import importlib.util
        optional_plotly = importlib.util.find_spec("plotly") is not None
        versions = {name: version(name) for name in DIST_NAMES}
        versions["python"] = platform.python_version()
        if optional_plotly:
            versions["plotly"] = version("plotly")
        write_json(options["preflight_result"], {"versions": versions, "optional_plotly": optional_plotly})
        print("Voraussetzungen und interne Imports geprüft.")
    elif stage == "latent":
        from dataclasses import asdict
        import torch
        import webbrowser
        import latentspace.config_latent as config_latent

        config_type = config_latent.LatentV2Config
        overrides = {"parquet_path": options["input_with_ids"], "output_root_dir": options["latent_base"],
                     "epochs": options["epochs"], "enable_delivery_memory": False,
                     "delivery_memory_dir": str(Path(options["run_dir"]) / "unused_delivery_memory")}
        cfg = config_type(**overrides)
        write_json(Path(options["run_dir"]) / "effective_latent_config.json", asdict(cfg))
        config_latent.LatentV2Config = partial(config_type, **overrides)
        torch.set_num_threads(2)
        # Nur automatische Desktop-Fenster deaktivieren; keine Modellfunktion ersetzen.
        os.startfile = lambda *args, **kwargs: None
        webbrowser.open = lambda *args, **kwargs: False
        sys.argv = [str(source / "latentspace/main_v2.py")]
        runpy.run_path(sys.argv[0], run_name="__main__")
    elif stage == "phase12":
        import config as cfg
        from dataclasses import asdict

        cfg.USE_PHASE0 = True
        cfg.USE_LLM_SUGGESTIONS = False
        write_json(Path(options["run_dir"]) / "effective_rule_config.json", {
            "USE_PHASE0": cfg.USE_PHASE0, "USE_LLM_SUGGESTIONS": cfg.USE_LLM_SUGGESTIONS,
            "PHASE2_SETTINGS": asdict(cfg.PHASE2_SETTINGS),
            "thresholds": {"accuracy": cfg.PHASE2_ACCURACY_THRESHOLD, "f1_macro": cfg.PHASE2_F1_THRESHOLD,
                           "precision_macro": cfg.PHASE2_PRECISION_THRESHOLD},
            "target_selection": "automatic_by_project_code", "SAMPLE_RUNS": cfg.SAMPLE_RUNS,
            "SAMPLE_FRACTION": cfg.SAMPLE_FRACTION,
        })
        sys.argv = [str(source / "main_phase1_phase_2.py"), "--input", options["input_with_ids"],
                    "--phase0", options["phase0_file"], "--output", options["phase12_base"],
                    "--split-file", options["split_file"]]
        runpy.run_path(sys.argv[0], run_name="__main__")
    else:
        raise ValueError(f"Unbekannter Arbeitsschritt: {stage}")


def execute(run_dir, label, command, project, expected_failure=None):
    """Echte Unterprozesse, mit Protokoll und überprüftem Rückgabestatus."""
    logs = run_dir / "logs"
    logs.mkdir(exist_ok=True)
    log = logs / (label + ".log")
    env = os.environ.copy()
    search_path = [str(project / "src"), str(project / "src/latentspace")]
    if env.get("PYTHONPATH"):
        search_path.append(env["PYTHONPATH"])
    env.update(PYTHONPATH=os.pathsep.join(search_path), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1",
               MPLBACKEND="Agg", NUMBA_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    print(f"[{label}] läuft ...", flush=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run([str(arg) for arg in command], cwd=project, env=env,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    text = log.read_text(encoding="utf-8", errors="replace")
    ok = (result.returncode == 0 if expected_failure is None
          else result.returncode != 0 and expected_failure in text)
    if not ok:
        print("\n".join(text.splitlines()[-35:]), flush=True)
        raise RuntimeError(f"{label} hat das erwartete Ergebnis nicht geliefert. Protokoll: {log}")
    elapsed = round(time.monotonic() - started, 2)
    print(f"[{label}] bestanden ({elapsed:.1f} s)", flush=True)
    return {"stage": label, "command": [str(arg) for arg in command], "exit_code": result.returncode,
            "expected_rejection": expected_failure, "passed": True, "seconds": elapsed,
            "log": str(log.relative_to(run_dir))}


def unique_run(base, required):
    base = Path(base)
    runs = [p for p in base.iterdir() if p.is_dir() and (p / required).is_file()] if base.exists() else []
    if len(runs) != 1:
        raise ValueError(f"Genau ein vollständiger neuer Lauf erwartet in {base}; gefunden: {len(runs)}")
    return runs[0]


def extract_payload(path):
    match = re.search(r'<script type="application/json" id="review-data">(.*?)</script>',
                      Path(path).read_text(encoding="utf-8"), re.S)
    if not match:
        raise ValueError("Prüfansicht enthält keinen lesbaren Datenblock.")
    return json.loads(match.group(1))


def check_generated_relations(frame, rules):
    import pandas as pd
    from rule_engine import evaluate_rule_on_dataframe

    # Vorab bekannte vier Kombinationen; Regelwortlaut und Baumstruktur sind frei.
    reference = frame.groupby(["spektralkanaele", "detailstufe"], sort=True).head(1).copy()
    result = []
    for target in TARGETS:
        selected = [r for r in rules if r.get("metadata", {}).get("target_col") == target]
        covered = pd.Series(False, index=reference.index)
        invalid_clean = pd.Series(False, index=reference.index)
        invalid_changed = pd.Series(False, index=reference.index)
        changed = reference.copy(deep=True)
        values = sorted(reference[target].unique().tolist())
        changed[target] = changed[target].map({v: values[(i + 1) % len(values)] for i, v in enumerate(values)})
        for rule in selected:
            applies, _, invalid = evaluate_rule_on_dataframe(reference, rule)
            covered |= applies
            invalid_clean |= invalid
            invalid_changed |= evaluate_rule_on_dataframe(changed, rule)[2]
        passed = len(reference) == 4 and bool(covered.all()) and not bool(invalid_clean.any()) and bool(invalid_changed.all())
        result.append({"target": target, "selected_rule_count": len(selected), "reference_combinations": len(reference),
                       "correct_combinations_pass": not bool(invalid_clean.any()),
                       "wrong_target_combinations_flagged": int(invalid_changed.sum()), "recovered": passed})
    return result


def write_report(folder, summary):
    passed = summary["project_check_passed"]
    state = "Projektdurchlauf bestanden" if passed else "Projektdurchlauf: Abweichungen prüfen"
    table = lambda heads, rows: ("<table><thead><tr>" + "".join(f"<th>{escape(str(x))}</th>" for x in heads)
                                + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{escape(str(x))}</td>" for x in row) + "</tr>" for row in rows) + "</tbody></table>")
    link = lambda path, label: f'<a href="{escape(path, quote=True)}">{escape(label)}</a>'
    document = ["<!doctype html><html lang='de'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
                "<title>Framework – Test des Projektdurchlaufs</title><style>body{margin:0;background:#f2f6f7;color:#193c42;font:16px/1.55 system-ui,sans-serif}header{background:#123c42;color:white;padding:30px max(24px,calc((100vw - 1080px)/2))}main{max-width:1080px;margin:24px auto;padding:0 24px 40px}section{background:white;border:1px solid #d7e1e3;border-radius:10px;padding:24px;margin:20px 0}h1{line-height:1.2}h2{font-size:1.3rem}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;padding:10px;vertical-align:top;border-bottom:1px solid #d7e1e3;overflow-wrap:anywhere}th{background:#eef4f5}.note{padding:16px;background:#fff5d9;border-left:4px solid #b98616}a{color:#006b72}code{overflow-wrap:anywhere}</style></head><body>",
                f"<header><p>VALIDIERUNGSFRAMEWORK · TESTPAKET 06</p><h1>{state}</h1><p>Normale Programmeinstiege, neue synthetische Daten, tatsächliches Modelltraining.</p></header><main>",
                "<section><h2>Was geprüft wurde</h2><p>Der Lauf erzeugt eine künstliche Parquet-Merkmalstabelle, führt die Grundlagenprüfung aus, trainiert den Autoencoder und berechnet LOF, HDBSCAN und UMAP. Anschließend laufen die reguläre Regelerzeugung mit automatischer Zielauswahl und die Ergebnisintegration.</p>",
                f"<p><strong>{summary['rows']} Datensätze</strong>, <strong>{summary['epochs']} Trainingsepochen</strong>, <strong>{summary['latent_dimensions']} latente Dimensionen</strong>.</p>",
                "<p class='note'>Dies ist ein technischer Integrationstest ab Parquet. Er misst keine reale Fehlererkennungsleistung und prüft keine TIFF-/XML-/NAS-Einlesung oder ALKIS-Geometrien. Der Autoencoder verwendet den gesamten künstlichen Bestand. Nur die Attributabhängigkeiten besitzen eine getrennte Trainings-/Validierungsaufteilung. Eine bestimmte Clusterzahl oder LOF-Rangfolge ist kein Bestehenskriterium.</p></section>",
                "<section><h2>Programmeinstiege</h2>"]
    document.append(table(["Schritt", "Erwarteter Rückgabestatus", "Ergebnis"],
                          [[r["stage"], "Ablehnung" if r["expected_rejection"] else "Erfolgreich", "Bestanden"] for r in summary["stages"]]))
    document.append("<p>Latent- und Regelkonfiguration werden nur im jeweiligen Testprozess angepasst. Zielspalten und Baumregeln wählt der vorhandene Projektcode selbst. Modelle, Scores und Regeln werden nicht durch Attrappen ersetzt.</p></section>")
    document.append("<section><h2>Automatisch gefundene Regeln</h2>")
    document.append(f"<p>{summary['profiling_rule_count']} wiederholt vorgeschlagene Profiling-Regeln, {summary['ml_candidate_count']} ML-Kandidaten, davon {summary['eligible_ml_rule_count']} mit bestandenen Güteschwellen; zusätzlich {summary['cluster_description_count']} beschreibende Clusterregeln.</p>")
    document.append(table(["Vorab festgelegte Referenz", "Gefundene Regeln", "Korrekte Kombinationen akzeptiert", "Falsche Kombinationen erkannt"],
                          [[r["target"], r["selected_rule_count"], "Ja" if r["correct_combinations_pass"] else "Nein", f"{r['wrong_target_combinations_flagged']}/4"] for r in summary["reference_relations"]]))
    document.append("<p>Im Testbestand gilt RGB → 3 Bänder, RGBI → 4 Bänder und standard → 20 cm, fein → 10 cm. Vier frei erfundene Exportprofile hängen von beiden Merkmalen ab. Das sind synthetische Testannahmen, keine Produktspezifikation. Die Kontrollgröße K_A/K_B ist bei gleichen Prädiktoren paarweise ausgeglichen und soll keine ausgewählte Regel erzeugen.</p></section>")
    document.append("<section><h2>Nachvollziehbare Prüfungen</h2>")
    document.append(table(["Prüfung", "Ergebnis"], [[r["description"], "Bestanden" if r["passed"] else "Nicht bestanden"] for r in summary["checks"]]))
    document.append("<p>Die Prüfung des Entscheidungsimports verwendet ausschließlich eine gesonderte Kopie der synthetischen Prüfansicht. Der Eintrag ist als automatischer Techniktest bezeichnet und stellt keine fachliche Bewertung dar. Die Browserbedienung wird nicht automatisiert geprüft.</p></section>")
    document.append("<section><h2>Ergebnisse öffnen</h2><ul>")
    for name, rel in summary["links"].items():
        document.append("<li>" + link(rel, name) + "</li>")
    document.append("</ul><p>Alle Programmprotokolle liegen unter <code>logs</code>. Die bewusst abgewiesenen Eingaben erzeugen dort erwartete Fehlermeldungen. <code>summary.json</code> enthält die Prüfergebnisse, <code>run_manifest.json</code> Code-Prüfsummen und Softwareversionen.</p></section></main></body></html>")
    (folder / "bericht.html").write_text("\n".join(document), encoding="utf-8")


def run(project, output=None, epochs=3):
    project = Path(project).resolve()
    source = prepare_imports(project)
    sources_before = source_snapshot(source)
    base = Path(output).resolve() if output else project / "data/project_check_06"
    folder = base / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    options = {"run_dir": str(folder), "epochs": epochs,
               "preflight_result": str(folder / "dependencies.json"),
               "input_with_ids": str(folder / "foundation/synthetic_input_with_ids.parquet"),
               "latent_base": str(folder / "latent"), "phase12_base": str(folder / "phase12"),
               "split_file": str(folder / "split/manifest.csv")}
    settings_file = folder / "test_options.json"
    write_json(settings_file, options)
    write_json(folder / "run_status.json", {"status": "running", "scenario": SCENARIO})
    stages, checks = [], []
    def require(name, condition):
        checks.append({"description": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)
    def worker(stage):
        return [sys.executable, Path(__file__).resolve(), "--project", project, "--worker", stage, "--settings", settings_file]
    try:
        stages.append(execute(folder, "01_voraussetzungen", worker("preflight"), project))
        import numpy as np
        import pandas as pd
        from review_common import verify_bundle
        from latentspace.anomaly_scores import SCORE_COLUMNS
        from record_identity import IDENTITY_COLUMNS

        raw = create_input()
        input_file = folder / "synthetic_input.parquet"
        raw.to_parquet(input_file, index=False)
        input_hash = sha(input_file)
        write_json(folder / "reference_spec.json", {"scenario": SCENARIO, "rows": len(raw), "groups": 160,
            "seed": 42, "target_selection": "automatic", "expected_dependency_targets": list(TARGETS),
            "reference_mapping": [{"spektralkanaele": c, "detailstufe": d, **expected_values(c, d)}
                                  for c, d in product(("RGB", "RGBI"), ("standard", "fein"))],
            "naming_observation_xml_name": "synthetic_0000_0.xml", "negative_control": CONTROL,
            "scope": "Technischer Integrationstest ab Parquet; künstliche Annahmen; keine reale Erkennungsleistung."})
        stages.append(execute(folder, "02_grundlagen", [sys.executable, source / "check_foundation.py", "--input", input_file, "--output", folder / "foundation"], project))
        identified = pd.read_parquet(options["input_with_ids"])
        identified_hash = sha(options["input_with_ids"])
        require("Alle 320 Eingaben erhalten eindeutige Kennungen", len(identified) == 320 and identified.record_key.is_unique)
        pd.testing.assert_frame_equal(raw, identified[raw.columns])
        require("Grundlagenprüfung verändert keine Originalattribute", True)
        require("Grundlagenprüfung enthält keine technisch ungültige Regel", read_json(folder / "foundation/summary.json")["invalid_rule_count"] == 0)

        stages.append(execute(folder, "03_latent_lof_cluster", worker("latent"), project))
        latent_dir = unique_run(options["latent_base"], "latent_outlier_results_lof.parquet")
        phase0_file = latent_dir / "latent_outlier_results_lof.parquet"
        phase0 = pd.read_parquet(phase0_file)
        latent_cols = [c for c in phase0 if re.fullmatch(r"latent_\d+", c)]
        latent_info = read_json(latent_dir / "run_info_lof.json")
        history = read_json(latent_dir / "training_history_v2.json")
        require("Autoencoder wurde tatsächlich für die angeforderten Epochen trainiert", latent_info["training_performed"] is True and len(history) == epochs)
        require("Latente Werte, Verluste und LOF sind endlich", len(latent_cols) == latent_info["latent_dim"] and np.isfinite(phase0[latent_cols + ["lof_raw", "reconstruction_error", "umap_x", "umap_y"]].to_numpy(dtype=float)).all() and all(np.isfinite(list(r.values())).all() for r in history))
        require("Phase 0 enthält genau die Eingabekennungen", len(phase0) == len(identified) and set(phase0.record_key) == set(identified.record_key))
        canonical = phase0.sort_values(["lof_raw", "record_key"], ascending=[False, True], kind="stable")
        require("Exportierte LOF-Ränge reichen eindeutig von 1 bis 320", phase0.outlier_rank.tolist() == list(range(1, 321)))
        require("Export ist nach LOF absteigend und bei Gleichstand nach Kennung sortiert", phase0.record_key.tolist() == canonical.record_key.tolist())
        pd.testing.assert_frame_equal(identified.set_index("record_key").sort_index(),
                                      phase0.set_index("record_key")[identified.columns.drop("record_key")].sort_index())
        require("Originalattribute bleiben in Phase 0 den richtigen Kennungen zugeordnet", True)
        require("Modell, Vorverarbeitung, Tokenizer und UMAP wurden gespeichert", all((latent_dir / name).is_file() and (latent_dir / name).stat().st_size > 0 for name in ("multimodal_autoencoder_v2.pt", "tabular_preprocessor_v2.pkl", "char_tokenizer_v2.pkl", "umap_plot_lof.png")))
        options["phase0_file"] = str(phase0_file)
        write_json(settings_file, options)

        stages.append(execute(folder, "04_regelerzeugung", worker("phase12"), project))
        phase12 = unique_run(options["phase12_base"], "run_info.json")
        selected = read_json(phase12 / "phase2_ml_rules.json")
        candidates = read_json(phase12 / "phase2_all_dependency_candidates.json")
        reports = read_json(phase12 / "phase2_ml_training_report.json")
        dependencies = [r for r in reports if r.get("source") == "ml_phase2"]
        relations = check_generated_relations(identified, selected)
        require("Automatische Zielauswahl und Regeln finden alle drei Referenzabhängigkeiten", all(r["recovered"] for r in relations))
        control = [r for r in dependencies if r["target_col"] == CONTROL]
        require("Uninformative Kontrollgröße erzeugt keine ausgewählten Regeln", len(control) == 1 and control[0]["rule_count"] == 0 and control[0]["model_gate_passed"] is False)
        forbidden = set(SCORE_COLUMNS) | set(IDENTITY_COLUMNS)
        require("IDs, Scores und latente Koordinaten werden keine Regelprädiktoren", all(not (forbidden & set(r["feature_cols_used"])) and not any(c.startswith("latent_") for c in r["feature_cols_used"]) for r in dependencies))
        split = pd.read_csv(phase12 / "phase2_split_manifest.csv")
        require("Eine Gruppe gehört immer zu genau einer Teilmenge", split.groupby("group_id").partition.nunique().max() == 1 and set(split.record_key) == set(identified.record_key))
        require("Regelauswahl verwendet den reservierten Phase-2-Testanteil nicht", bool(dependencies) and all(r["test_evaluated"] is False for r in dependencies))

        review_base = folder / "review"
        review_command = [sys.executable, source / "build_review.py", "--input", options["input_with_ids"],
                          "--phase0", phase0_file, "--phase12-dir", phase12, "--output", review_base]
        stages.append(execute(folder, "05_ergebnisintegration", review_command, project))
        review = unique_run(review_base, "review_bundle.json")
        bundle = verify_bundle(read_json(review / "review_bundle.json"))
        require("Prüfansicht erhält sämtliche Kennungen und die LOF-Reihenfolge", [r["record_key"] for r in bundle["cases"]] == canonical.record_key.tolist())
        require("Alle Fälle bleiben vor fachlicher Prüfung offen", read_json(review / "review_summary.json")["review_counts"]["unreviewed"] == 320)
        require("Prüfansicht enthält alle erzeugten ML-Kandidaten", len([r for r in bundle["rule_catalog"] if r["category"] == "dependency_candidates"]) == len(candidates))
        expected_name_key = identified.loc[identified.xml_name.eq("synthetic_0000_0.xml"), "record_key"].item()
        naming_keys = [r["record_key"] for r in bundle["cases"] if r["naming_observation_count"]]
        require("Bekannter Benennungshinweis gelangt zum richtigen Datensatz", naming_keys == [expected_name_key])
        require("Integration meldet keine technisch ungültigen Regeln", bundle["summary"]["rule_evaluation_issue_count"] == 0)
        payload = extract_payload(review / "review.html")
        require("HTML enthält genau die integrierten Daten und Revision 0", payload["bundle"] == bundle and payload["revision"] == 0)

        negative = folder / "negative_cases"
        negative.mkdir()
        duplicate = raw.copy(deep=True)
        duplicate.loc[1, "xml_name"] = duplicate.loc[0, "xml_name"]
        duplicate_file = negative / "duplicate_input.parquet"
        duplicate.to_parquet(duplicate_file, index=False)
        stages.append(execute(folder, "06_doppelte_kennung", [sys.executable, source / "check_foundation.py", "--input", duplicate_file, "--output", negative / "duplicate_output"], project, "doppelten record_key"))
        require("Bei doppelter Kennung entsteht keine Grundlagen-Ausgabe", not (negative / "duplicate_output").exists())
        missing_phase0 = negative / "missing_phase0_row.parquet"
        phase0.iloc[:-1].to_parquet(missing_phase0, index=False)
        negative_options = {**options, "run_dir": str(negative), "phase0_file": str(missing_phase0),
                            "phase12_base": str(negative / "wrong_phase0_output"),
                            "split_file": str(negative / "negative_split.csv")}
        negative_settings = negative / "test_options.json"
        write_json(negative_settings, negative_options)
        stages.append(execute(folder, "07_falsche_phase0_zuordnung", [sys.executable, Path(__file__).resolve(),
                              "--project", project, "--worker", "phase12", "--settings", negative_settings],
                              project, "Phase-0-Ergebnis passt nicht zum Eingabebestand"))
        require("Bei unvollständiger Phase 0 entsteht kein Regellauf", not (negative / "wrong_phase0_output").exists())
        changed = identified.copy(deep=True)
        changed.loc[0, "bodenpixelgroesse"] = 999
        changed_file = negative / "changed_input.parquet"
        changed.to_parquet(changed_file, index=False)
        stages.append(execute(folder, "08_veraltete_ergebnisse", [sys.executable, source / "build_review.py", "--input", changed_file, "--phase0", phase0_file, "--phase12-dir", phase12, "--output", negative / "stale_output"], project, "stimmt nicht mit dem Phase-1/2-Lauf überein"))
        require("Geänderte Eingabe wird nicht mit alten Prüfergebnissen vermischt", not (negative / "stale_output").exists())

        roundtrip = folder / "review_import_test"
        shutil.copytree(review, roundtrip)
        request_file = folder / "synthetic_technical_decision.json"
        write_json(request_file, {"schema_version": 1, "bundle_id": bundle["bundle_id"], "base_revision": 0,
            "reviewer": "Automatisierter Integrationstest", "decisions": [{"record_key": expected_name_key,
            "decision": "needs_clarification", "reason": "AUTOMATISCHER TECHNIKTEST auf künstlichen Daten: Import und Revision prüfen; keine fachliche Bewertung."}]})
        apply = [sys.executable, source / "review_decisions.py", "apply", "--review-dir", roundtrip, "--decisions", request_file]
        history_before = sha(roundtrip / "review_history.json")
        stages.append(execute(folder, "09_import_trockenlauf", apply + ["--dry-run"], project))
        require("Trockenlauf ändert kein Entscheidungsprotokoll", sha(roundtrip / "review_history.json") == history_before)
        stages.append(execute(folder, "10_import_speichern", apply, project))
        updated = read_json(roundtrip / "review_summary.json")
        require("Techniktest-Import erzeugt genau eine dokumentierte Entscheidung", updated["review_revision"] == 1 and updated["review_counts"]["needs_clarification"] == 1 and updated["review_counts"]["unreviewed"] == 319)
        require("Ansicht übernimmt die neue Revision", extract_payload(roundtrip / "review.html")["revision"] == 1)
        history_after = sha(roundtrip / "review_history.json")
        stages.append(execute(folder, "11_veralteten_import_abweisen", apply, project, "Veraltete Bearbeitungsrevision"))
        require("Wiederholter Import ändert das Protokoll nicht", sha(roundtrip / "review_history.json") == history_after)
        require("Die eigentliche Prüfansicht bleibt unbewertet", sha(review / "review_history.json") == history_before)
        require("Eingabe, ID-Kopie und Quellcode sind unverändert", sha(input_file) == input_hash and sha(options["input_with_ids"]) == identified_hash and source_snapshot(source) == sources_before)
        require("Delivery Memory wurde nicht beschrieben", not (folder / "unused_delivery_memory").exists())

        summary = {"scenario": SCENARIO, "project_check_passed": all(r["passed"] for r in checks),
                   "rows": len(identified), "epochs": epochs, "latent_dimensions": len(latent_cols),
                   "profiling_rule_count": len(read_json(phase12 / "stable_profile_rules.json")),
                   "ml_candidate_count": len(candidates), "eligible_ml_rule_count": len(selected),
                   "cluster_description_count": len(read_json(phase12 / "ssl_cluster_rules.json")),
                   "cluster_count": int(phase0.loc[phase0.cluster_label.ne(-1), "cluster_label"].nunique()),
                   "noise_rows": int(phase0.cluster_label.eq(-1).sum()),
                   "reference_relations": relations, "negative_control_report": control[0],
                   "checks": checks, "stages": stages,
                   "scope": "Technischer Gesamtdurchlauf ab Parquet; keine Aussage über reale Fehlererkennungsleistung oder Rohdatei-Adapter.",
                   "links": {"Unbewertete synthetische Prüfansicht": (review / "review.html").relative_to(folder).as_posix(),
                             "Automatisch erzeugte Abhängigkeitsregeln": (phase12 / "phase2_all_dependency_candidates.json").relative_to(folder).as_posix(),
                             "UMAP-Darstellung": (latent_dir / "umap_plot_lof.png").relative_to(folder).as_posix(),
                             "Techniktest des Entscheidungsimports": (roundtrip / "review.html").relative_to(folder).as_posix()}}
        write_json(folder / "summary.json", summary)
        write_report(folder, summary)
        write_json(folder / "run_manifest.json", {"scenario": SCENARIO, "project": str(project),
            "created_at": datetime.now(timezone.utc).isoformat(), "versions": read_json(folder / "dependencies.json")["versions"],
            "source_sha256": sources_before, "test_script_sha256": sha(__file__),
            "source_unchanged": True, "input_sha256": input_hash,
            "artifact_sha256": {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*"))
                                if p.is_file() and p.name not in {"run_status.json", "run_manifest.json"}}})
        write_json(folder / "run_status.json", {"status": "completed", "project_check_passed": True})
        print(json.dumps({"project_check_passed": True, "rows": len(identified), "autoencoder_training_performed": True,
                          "epochs": epochs, "reference_relations_recovered": sum(r["recovered"] for r in relations),
                          "eligible_ml_rule_count": len(selected), "checks_passed": len(checks),
                          "expected_rejections_verified": sum(r["expected_rejection"] is not None for r in stages)}, indent=2), flush=True)
        print("Ergebnisordner:", folder, flush=True)
        print("Bericht im Browser öffnen:", folder / "bericht.html", flush=True)
        return summary, folder
    except Exception as exc:
        write_json(folder / "run_status.json", {"status": "failed", "project_check_passed": False,
            "error": str(exc), "checks_completed": checks, "stages_completed": stages})
        (folder / "failure_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print("Test abgebrochen. Diagnoseordner:", folder, flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optionaler Ergebnis-Basisordner")
    parser.add_argument("--epochs", type=int, default=3, help="Kurzer technischer Trainingslauf, Standard 3")
    parser.add_argument("--worker", choices=("preflight", "latent", "phase12"), help=argparse.SUPPRESS)
    parser.add_argument("--settings", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Mindestens eine Trainingsepoche erforderlich.")
    if args.worker:
        if args.settings is None:
            parser.error("Interner Arbeitsschritt benötigt --settings.")
        run_worker(args.project, args.worker, args.settings)
        return 0
    try:
        run(args.project, args.output, args.epochs)
    except Exception as exc:
        print(f"Projekttest nicht bestanden: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
