"""Rangpositionen protokollierter Fehlerinjektionen in einem LOF-Ergebnis.

Ein Treffer bedeutet hier, dass ein protokollierter Fall unter den ersten K
Rangplätzen liegt. Der Join allein ist keine erfolgreiche Fehlererkennung.
Die unveränderte DOP-Beispieldatei besitzt kein solches Injektionsprotokoll.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from latentspace.anomaly_scores import rank_by_lof


def evaluate_injections(injections: pd.DataFrame, results: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if not {"record_key_after", "error_type"}.issubset(injections.columns):
        raise ValueError("Injektionsprotokoll benötigt record_key_after und error_type.")
    keys = injections["record_key_after"].astype("string")
    if keys.isna().any() or keys.str.strip().eq("").any():
        raise ValueError("Injektionsprotokoll enthält fehlende Kennungen.")
    ranked = rank_by_lof(results)
    columns = ["record_key", "outlier_rank", "lof_raw", "lof_z"]
    reserved = set(columns) - {"record_key"}
    if reserved.intersection(injections.columns):
        raise ValueError("Injektionsprotokoll enthält bereits Ergebnis-Spalten.")
    merged = injections.assign(record_key_after=keys).merge(
        ranked[columns], left_on="record_key_after", right_on="record_key",
        how="left", sort=False, validate="many_to_one", suffixes=("_inj", "_res"),
    )
    missing = int(merged["outlier_rank"].isna().sum())
    if missing:
        raise ValueError(f"{missing} Injektionseinträge konnten nicht zugeordnet werden. IDs und passenden Ergebnislauf prüfen.")
    total = len(merged)
    summary = {
        "primary_score_name": "lof_raw", "evaluation_unit": "injection_log_entry",
        "num_injection_entries": total,
        "num_distinct_injected_records": int(keys.nunique()),
        "matched_entries": total, "population_count": len(ranked),
        "note": "Mehrere Injektionen pro Datensatz werden als mehrere Protokolleinträge gezählt. Keine Aussage zu natürlichen Fehlern oder Fehlalarmen.",
    }
    for k in (10, 50, 100, 500):
        hits = int(merged["outlier_rank"].le(k).sum())
        summary[f"top{k}_hits"] = hits
        summary[f"top{k}_entry_recall"] = hits / total if total else None
        summary[f"top{k}_effective_budget"] = min(k, len(ranked))
    return merged, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--injected-log", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        merged, summary = evaluate_injections(
            pd.read_csv(args.injected_log, dtype={"record_key_after": "string"}),
            pd.read_parquet(args.results),
        )
        args.output.mkdir(parents=True, exist_ok=False)
        merged.to_csv(args.output / "evaluation_injected_errors_detailed.csv", index=False, encoding="utf-8-sig")
        by_type = merged.groupby("error_type", dropna=False).agg(
            count=("outlier_rank", "size"), best_rank=("outlier_rank", "min"),
            mean_rank=("outlier_rank", "mean"), mean_lof_raw=("lof_raw", "mean"),
        ).reset_index()
        by_type.to_csv(args.output / "evaluation_injected_errors_by_type.csv", index=False, encoding="utf-8-sig")
        (args.output / "evaluation_injected_errors_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Auswertung abgebrochen: {exc}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
