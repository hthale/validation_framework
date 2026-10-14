"""Artefakte prüfen, ohne Modelle erneut zu trainieren."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify_analysis(out, run, preparation, expected_epochs):
    if not __debug__:
        raise ValueError("Prüfung ohne Python-Option -O starten.")
    import numpy as np
    import pandas as pd
    from attribute_prep import prepare_attributes, check_contract, ATTRIBUTES, DATE_MISSING, FORBIDDEN
    from experiment_io import read_verified_run
    from review_common import read_json
    from record_identity import build_record_key
    read_verified_run(run, ["input_with_ids.parquet", "feature_contract.json", "summary.json", "rule_catalog.json",
                            "latent_outlier_results_lof.parquet", "training_history.json", "dependency_split.csv",
                            "dependency_feature_contract.json", "autoencoder.pt", "preprocessor.pkl", "tokenizer.pkl"])
    checks = ["Framework-Lauf abgeschlossen; relevante Artefaktprüfsummen stimmen"]
    original = pd.read_parquet(out / "source_attributes.parquet")
    attributes = pd.read_parquet(out / "attributes.parquet")
    regenerated = prepare_attributes(original)
    # Pandas 2/3: fehlende Skalare können als None, NaN oder pd.NA zurückkommen.
    pd.testing.assert_frame_equal(attributes.isna(), regenerated.isna(), check_dtype=False)
    pd.testing.assert_frame_equal(attributes.astype(object).where(attributes.notna(), None),
                                  regenerated.astype(object).where(regenerated.notna(), None),
                                  check_dtype=False, check_exact=True)
    checks.append("Attributtabelle aus unveränderten ausgewählten Quellattributen reproduziert")
    assert set(attributes) <= {"gml_id", *ATTRIBUTES, DATE_MISSING}
    assert not set(attributes) & FORBIDDEN
    assert not preparation["geometry_columns_read"] and not preparation["geometry_features_created"]
    checks.append("Keine Geometrie, Koordinaten, Fremdkennungen oder nicht freigegebenen Spalten in der Eingabe")
    contract = read_json(run / "feature_contract.json")
    check_contract(attributes, contract)
    checks.append("Fachcodes kategorial; Prüfzeitpunkt als Datum; gml_id technisch")
    saved = pd.read_parquet(run / "input_with_ids.parquet")
    saved_attributes = saved[list(attributes)].astype(object).where(saved[list(attributes)].notna(), None)
    expected_attributes = attributes.astype(object).where(attributes.notna(), None)
    pd.testing.assert_frame_equal(saved_attributes, expected_attributes, check_dtype=False, check_exact=True)
    checks.append("Framework erhält sämtliche aufbereiteten Attributwerte und ihre Reihenfolge")
    pd.testing.assert_series_equal(saved.record_key, build_record_key(attributes, identity_columns=["gml_id"]),
                                   check_dtype=False)
    checks.append("Stabile und eindeutige Identitäten ausschließlich aus gml_id")
    latent = pd.read_parquet(run / "latent_outlier_results_lof.parquet")
    assert len(latent) == len(attributes) == preparation["selected_rows"]
    assert latent.record_key.is_unique and set(latent.record_key) == set(saved.record_key)
    matched = latent.set_index("record_key").loc[saved.record_key, list(attributes)].reset_index(drop=True)
    pd.testing.assert_frame_equal(matched.astype(object).where(matched.notna(), None),
                                  expected_attributes, check_dtype=False, check_exact=True)
    checks.append("Alle ausgewählten Objekte genau einmal im Ergebnis")
    ordered = latent.sort_values(["lof_raw", "record_key"], ascending=[False, True], kind="stable")
    assert ordered.record_key.tolist() == latent.record_key.tolist()
    assert latent.outlier_rank.tolist() == list(range(1, len(latent)+1))
    assert np.isfinite(latent.lof_raw).all()
    checks.append("LOF-Ränge endlich, vollständig und bei Gleichstand eindeutig")
    zcols = [c for c in latent if c.startswith("latent_")]
    summary = read_json(run / "summary.json")
    assert len(zcols) == summary["latent_dimensions"] and np.isfinite(latent[zcols].to_numpy()).all()
    history = read_json(run / "training_history.json")
    assert len(history) == expected_epochs and np.isfinite([list(x.values()) for x in history]).all()
    assert summary["autoencoder_training_performed"] is True
    checks.append("Autoencoder wirklich trainiert; latente Werte und Verluste endlich")
    lf = contract["latent_features"]
    sources = set(lf["numeric_columns"] + lf["categorical_columns"] + lf["string_columns"])
    sources -= {x["feature"] for x in lf["derived_features"]}
    sources |= {x["source"] for x in lf["derived_features"]}
    assert sources <= set(ATTRIBUTES) | {DATE_MISSING}
    checks.append("Tatsächliche Modellmerkmale stammen ausschließlich aus freigegebenen Attributen")
    split = pd.read_csv(run / "dependency_split.csv")
    assert set(split.record_key) == set(saved.record_key)
    assert split.groupby("group_id").partition.nunique().max() == 1
    duplicate_groups = attributes.drop(columns="gml_id").fillna("__CHECK_MISSING__").apply(
        lambda row: json.dumps(row.tolist(), ensure_ascii=False), axis=1)
    part_by_key = split.set_index("record_key").partition
    assert pd.DataFrame({"same":duplicate_groups, "part":saved.record_key.map(part_by_key)}).groupby("same").part.nunique().max() == 1
    checks.append("Identische Attributkombinationen bleiben in derselben Abhängigkeitsteilmenge")
    dependency_contract = read_json(run / "dependency_feature_contract.json")
    assert dependency_contract["scope"] == "training_partition_only"
    checks.append("Merkmalsrollen der Abhängigkeitssuche nur aus deren Training bestimmt")
    catalog = read_json(run / "rule_catalog.json")
    for candidate in catalog["dependency_candidates"]:
        assert candidate["metadata"]["test_evaluated"] is False
        assert candidate["metadata"]["evaluation_scope"] == "validation_partition"
        assert set(candidate["columns"]) <= set(ATTRIBUTES) | {DATE_MISSING}
    assert not catalog["approved_rules"] and not catalog["domain_rules"]
    assert summary["confirmed_error_count"] == 0 and summary["delivery_memory_written"] is False
    checks.append("Keine vorgegebenen Produktregeln, Testauswertung, Fachfreigaben oder Fehlerlabels")
    return checks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, default=Path("."))
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args()
    from start_bahnverkehr import bootstrap, hashes
    project = bootstrap(args.project)
    from review_common import read_json, file_sha256
    from experiment_io import read_verified_run
    out = (project / args.run).resolve(strict=True)
    try:
        manifest = read_json(out / "run_manifest.json")
        read_verified_run(out, list(manifest["output_sha256"]))
        preparation = read_json(out / "preparation.json")
        summary = read_json(out / "summary.json")
        assert hashes(project / "src") == preparation["project_source_at_start"], "Projektcode inzwischen geändert"
        assert hashes(Path(__file__).resolve().parent) == preparation["package_source_at_start"], "Paketcode inzwischen geändert"
        assert file_sha256(Path(preparation["source_path"])) == preparation["input_sha256"], "Originaldatei geändert"
        run = (out / summary["analysis_directory"]).resolve(strict=True)
        if not run.is_relative_to(out):
            raise ValueError("Analyselauf liegt außerhalb des Ergebnisordners.")
        checks = verify_analysis(out, run, preparation, summary["epochs"])
        print(json.dumps({"passed":True, "checks":checks, "all_output_hashes_verified":True}, ensure_ascii=False, indent=2))
    except (AssertionError, ValueError, OSError) as exc:
        p.exit(2, f"Prüfung fehlgeschlagen: {exc}\n")


if __name__ == "__main__":
    main()
