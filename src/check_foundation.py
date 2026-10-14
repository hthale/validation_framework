"""Erster DOP-Prüflauf ohne Autoencoder, LLM oder Änderungen an der Eingabedatei.

Aufruf aus dem Projektordner:
python src/check_foundation.py --input data/parquetdata/dop_training_dataset.parquet --output data/check_01
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from classification import classify_and_prioritize_rules
from dop_observations import dop_name_observations
from project_profile import EXPECTED_SPECTRAL_CHANNELS
from record_identity import with_record_key
from rule_engine import build_domain_specific_rules, validate_rules_on_full_dataframe


def json_compatible(value):
    """Fehlende Werte auch in Verstoßbeispielen als JSON-null speichern."""
    if isinstance(value, dict):
        return {str(k): json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(v) for v in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output_file = args.output / f"{args.input.stem}_with_ids.parquet"
    if output_file.resolve() == args.input.resolve():
        raise ValueError("Die Eingabedatei darf nicht überschrieben werden.")
    df = with_record_key(pd.read_parquet(args.input))
    rules = build_domain_specific_rules(df, expected_spectral_channels=EXPECTED_SPECTRAL_CHANNELS)
    results, examples = validate_rules_on_full_dataframe(df, rules)
    observations = dop_name_observations(df)
    summary = {
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "row_count": len(df), "unique_record_keys": int(df["record_key"].nunique()),
        "expected_spectral_channels": EXPECTED_SPECTRAL_CHANNELS,
        "naming_observation_count": len(observations),
        "evaluated_rule_count": sum(r["evaluation_status"] == "evaluated" for r in results),
        "invalid_rule_count": sum(r["evaluation_status"] == "invalid_rule" for r in results),
        "note": "Benennungsabweichungen sind Beobachtungen; Domänenregeln enthalten Projektannahmen.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    pd.DataFrame(results).to_csv(args.output / "rule_results.csv", index=False, encoding="utf-8-sig")
    observations.to_csv(args.output / "naming_observations.csv", index=False, encoding="utf-8-sig")
    for name, data in {
        "summary.json": summary, "rule_results.json": results,
        "violation_examples.json": examples,
        "classified_rules.json": classify_and_prioritize_rules(results),
    }.items():
        (args.output / name).write_text(json.dumps(json_compatible(data), ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Prüfergebnisse: {args.output.resolve()}")
    print(f"Eingabekopie mit IDs: {output_file.resolve()}")


if __name__ == "__main__":
    main()
