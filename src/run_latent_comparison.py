"""Paarweiser, gruppengetrennter Vergleich: LOF ohne Autoencoder und im latenten Raum."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from html import escape
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from experiment_io import new_output, finish_output, html_report, details_json
from generic_schema import attach_identity, choose_groups, infer_columns, normalize_columns, tabular_preparation
from latent_comparison import (PROTOCOL, grouped_partition, transform_frozen, source_columns,
                               baseline_matrix, inject_changes, novel_lof, evaluate_pair)
from review_common import file_sha256, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/latent_comparison_08")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--n-neighbors", type=int, default=20)
    p.add_argument("--review-fraction", type=float, default=.05)
    p.add_argument("--change-fraction", type=float, default=.15)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-string-fields", type=int, default=12)
    p.add_argument("--id-columns", nargs="+", default=[])
    p.add_argument("--group-columns", nargs="+", default=[])
    p.add_argument("--exclude-columns", nargs="+", default=[])
    return p


def latent_encode(model, matrix, strings, torch):
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(matrix), 256):
            out = model(torch.tensor(matrix[start:start+256], dtype=torch.float32),
                        [torch.tensor(s[start:start+256], dtype=torch.long) for s in strings])
            chunks.append(out["z_shared"].detach().cpu().numpy())
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise ValueError("Nicht endlicher latenter Raum im Vergleich.")
    return result


def render_comparison(out, summary, protocol):
    def pct(v): return f"{100*v:.1f} %"
    content = ('<p class="notice">Gleiche Ausgangsmerkmale, getrennte Gruppen und identische künstliche Änderungen. '
               'Gemessen wird die Priorisierung veränderter Prüfzeilen. Eine Änderung ist nicht automatisch ein fachlicher Fehler. '
               'Unveränderte Zeilen im Prüfbudget sind deshalb keine bestätigten Fehlalarme.</p>')
    delta = summary["paired_mean_recall_difference"]
    direction = "höher" if delta > 0 else "niedriger" if delta < 0 else "gleich"
    content += (f'<h2>Ergebnis dieses Versuchs</h2><p>Die mittlere Trefferquote auf künstlich geänderten Zeilen '
                f'liegt mit Autoencoder {abs(delta)*100:.1f} Prozentpunkte {direction} '
                f'als bei der festgelegten Baseline ohne Autoencoder. '
                'Das Ergebnis gilt für diese Daten, Änderungsarten und Einstellungen. '
                'Die Wiederholungen sind keine unabhängigen Studien; ein positiver Nutzen wird nicht vorausgesetzt.</p>')
    rows = []
    for repeat in summary["repetitions"]:
        for name, metrics in repeat["methods"].items():
            values = (repeat["seed"], "Ohne Autoencoder" if name == "baseline" else "Latenter Raum",
                      metrics["test_rows"], metrics["review_budget_k"],
                      f"{metrics['selected_changed']}/{metrics['changed_rows']}",
                      pct(metrics["recall_of_changes_at_k"]), pct(metrics["share_of_changes_in_review"]),
                      metrics["selected_unchanged"], metrics["newly_selected_changed"],
                      metrics["input_invisible_changes"])
            rows.append('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in values) + '</tr>')
    content += ('<div class="scroll"><table><thead><tr><th>Seed</th><th>Methode</th><th>Prüfzeilen</th>'
                '<th>Budget</th><th>Änderungen im Budget</th><th>Trefferquote der Änderungen</th>'
                '<th>Änderungsanteil im Budget</th><th>Unveränderte im Budget</th>'
                '<th>Erst nach Änderung ausgewählt</th><th>Durch Vorverarbeitung unsichtbar</th>'
                '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')
    content += ('<p>„Erst nach Änderung ausgewählt“ vergleicht dieselben Kennungen vor und nach der Änderung. '
                'Bereits vorher auffällige Datensätze werden zusätzlich ausgewiesen. '
                'Durch Imputation unsichtbare Änderungen bleiben im Nenner. '
                'Das Prüfbudget begrenzt die maximal mögliche Trefferquote; diese Grenze und die Zufallserwartung stehen in den Details.</p>')
    content += '<h2>Zusammenfassung über Seeds</h2>' + details_json(summary["aggregate"], "Mittelwerte und Spannweiten")
    content += '<h2>Änderungsarten, Merkmale und Gruppen</h2>'
    for repeat in summary["repetitions"]:
        content += details_json(repeat, f"Seed {repeat['seed']}: vollständige Auswertung")
    content += '<h2>Versuchsprotokoll</h2>' + details_json(protocol, "Vor dem Training gespeicherte Festlegungen")
    content += '<ul>' + ''.join('<li>' + escape(v) + '</li>' for v in PROTOCOL["limitations"]) + '</ul>'
    content += '<h2>Dateien</h2><ul>' + ''.join(f'<li><a href="{name}">{name}</a></li>' for name in (
        "summary.json", "protocol.json", "input_with_ids.parquet", "run_manifest.json")) + '</ul>'
    for repeat in summary["repetitions"]:
        sub = f"seed_{repeat['seed']}"
        content += f'<p>Seed {repeat["seed"]}: ' + ' · '.join(
            f'<a href="{sub}/{name}">{name}</a>' for name in (
                "metrics.json", "changes.json", "scores.parquet", "split.csv", "feature_contract.json", "training_history.json")) + '</p>'
    html_report(out / "bericht.html", "LOF mit und ohne Autoencoder", content)


def analyze(args):
    if min(args.epochs, args.n_neighbors, args.threads, args.max_string_fields) < 1:
        raise ValueError("Epochen, Nachbarn, Threads und maximale Textfeldzahl müssen positiv sein.")
    if (not args.seeds or len(args.seeds) != len(set(args.seeds))
            or any(not 0 <= s < 2**32 for s in args.seeds)):
        raise ValueError("Eindeutige Seeds zwischen 0 und 2**32-1 angeben.")
    if not 0 < args.review_fraction <= 1 or not 0 < args.change_fraction < 1:
        raise ValueError("Prüfbudget muss in (0,1] und Änderungsanteil in (0,1) liegen.")
    source = args.input.resolve(strict=True)
    if source.suffix.lower() != ".parquet":
        raise ValueError("Eine flache Parquet-Eingabetabelle verwenden.")
    source_hash = file_sha256(source)
    raw, identity = attach_identity(pd.read_parquet(source), args.id_columns)
    missing = set(args.id_columns+args.group_columns+args.exclude_columns)-set(raw)
    if missing:
        raise ValueError(f"Angegebene Spalten fehlen: {sorted(missing)}")
    grouped, grouping = choose_groups(raw, args.group_columns)
    # Alle Aufteilungen validieren, bevor ein Modell trainiert wird.
    partitions = {s: grouped_partition(grouped, s) for s in args.seeds}
    excluded = sorted(set(args.id_columns+args.exclude_columns+grouping["columns"]))
    latent_dir = str(Path(__file__).resolve().parent / "latentspace")
    if latent_dir not in sys.path: sys.path.insert(0, latent_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")
    import torch
    import joblib
    from latentspace.config_latent import LatentV2Config
    from latentspace.main_v2 import train_model, encode_string_columns
    torch.set_num_threads(args.threads)
    out = new_output(args.output)
    protocol = {**PROTOCOL, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "input_sha256": source_hash, "grouping": grouping, "identity": identity,
                "parameter_selection_using_test_labels": False, "model_retrained_per_seed": True}
    write_json(out / "protocol.json", protocol)
    raw.to_parquet(out / "input_with_ids.parquet", index=False)
    repetitions = []
    try:
        for seed, (ordered, split) in partitions.items():
            print(f"[Seed {seed}] Aufteilung und Vorverarbeitung", flush=True)
            sub = out / f"seed_{seed}"
            sub.mkdir()
            split.to_csv(sub / "split.csv", index=False)
            train_keys = set(split.loc[split.partition.eq("train"), "record_key"])
            raw_train = ordered.loc[ordered.record_key.isin(train_keys)].reset_index(drop=True)
            raw_test = ordered.loc[~ordered.record_key.isin(train_keys)].reset_index(drop=True)
            contract = infer_columns(raw_train, excluded=excluded, scope="training_partition_only")
            train, train_issues = normalize_columns(raw_train, contract)
            test, test_issues = normalize_columns(raw_test, contract)
            frame, prepared, metadata = tabular_preparation(train, contract, args.max_string_fields)
            metadata["fit_scope"] = "training_partition_only"
            contract.update(latent_features=metadata, grouping=grouping, explicit_exclusions=excluded)
            write_json(sub / "feature_contract.json", contract)
            write_json(sub / "normalization_issues.json", {"train": train_issues, "test": test_issues})
            changed, changes = inject_changes(test, train, contract, metadata, args.change_fraction, seed)
            write_json(sub / "changes.json", changes)
            test.to_parquet(sub / "test_before.parquet", index=False)
            changed.to_parquet(sub / "test_after.parquet", index=False)
            raw_train.to_parquet(sub / "train_original.parquet", index=False)
            before_frame, before_tab = transform_frozen(test, prepared, metadata)
            after_frame, after_tab = transform_frozen(changed, prepared, metadata)
            _, train_again = transform_frozen(train, prepared, metadata)
            if not np.array_equal(train_again, prepared["X_tab"]):
                raise ValueError("Training und eingefrorene Transformation verwenden unterschiedliche Aufbereitung.")
            cfg = LatentV2Config(parquet_path=str(source), output_root_dir=str(sub), epochs=args.epochs,
                random_state=seed, lof_n_neighbors=args.n_neighbors, string_columns=tuple(metadata["string_columns"]),
                known_numeric_columns=(), known_date_columns=(), ignore_columns=(), enable_delivery_memory=False)
            write_json(sub / "model_config.json", asdict(cfg))
            print(f"[Seed {seed}] Autoencoder: {len(train)} Trainingszeilen, {len(test)} Prüfzeilen", flush=True)
            result = train_model(frame, cfg, prepared_tabular=prepared)
            history = result["training_history"]
            if len(history) != args.epochs or not all(np.isfinite(list(h.values())).all() for h in history):
                raise ValueError("Trainingsverlauf ist unvollständig oder nicht endlich.")
            if list(result["used_string_cols"]) != metadata["string_columns"]:
                raise ValueError("Autoencoder verwendet andere Textfelder als der gespeicherte Vertrag.")
            tokenizer = result["tokenizer"]
            train_str, _ = encode_string_columns(frame, cfg, tokenizer)
            before_str, _ = encode_string_columns(before_frame, cfg, tokenizer)
            after_str, _ = encode_string_columns(after_frame, cfg, tokenizer)
            baselines = [baseline_matrix(tab, strings, tokenizer.vocab_size) for tab, strings in (
                (prepared["X_tab"], train_str), (before_tab, before_str), (after_tab, after_str))]
            # Die gleiche Batchgröße bei allen drei Inferenzaufrufen vermeidet
            # unnötige numerische Unterschiede zwischen Referenz und Prüfkopien.
            latents = [latent_encode(result["model"], tab, strings, torch) for tab, strings in (
                (prepared["X_tab"], train_str), (before_tab, before_str), (after_tab, after_str))]
            torch.save(result["model"].state_dict(), sub / "autoencoder.pt")
            joblib.dump(prepared["preprocessor"], sub / "preprocessor.pkl")
            joblib.dump(tokenizer, sub / "tokenizer.pkl")
            write_json(sub / "training_history.json", history)
            # Matrizen erlauben unabhängiges Nachrechnen von LOF ohne Modelltraining.
            np.savez_compressed(sub / "representations.npz", baseline_train=baselines[0], baseline_before=baselines[1],
                                baseline_after=baselines[2], latent_train=latents[0], latent_before=latents[1], latent_after=latents[2])
            keys = test.record_key.tolist()
            changed_keys = {c["record_key"] for c in changes}
            changed_mask = np.array([k in changed_keys for k in keys])
            input_visible = np.any(baselines[1] != baselines[2], axis=1)
            methods, score_tables = {}, []
            for name, matrices in (("baseline", baselines), ("latent", latents)):
                scores_before, scores_after, lof_model, effective = novel_lof(*matrices, args.n_neighbors)
                if not np.allclose(scores_before[~changed_mask], scores_after[~changed_mask], rtol=0, atol=1e-9):
                    raise ValueError("Unveränderte Prüffälle erhielten andere Scores im eingefrorenen Modell.")
                cases, metrics = evaluate_pair(keys, scores_before, scores_after, changes, args.review_fraction)
                visible_map = dict(zip(keys, input_visible.tolist()))
                representation_map = dict(zip(keys, np.any(matrices[1] != matrices[2], axis=1).tolist()))
                cases["input_visible_change"] = cases.record_key.map(visible_map)
                cases["representation_changed"] = cases.record_key.map(representation_map)
                cases["method"] = name
                metrics.update(n_neighbors_effective=effective, representation_dimensions=matrices[0].shape[1],
                               input_invisible_changes=int((changed_mask & ~input_visible).sum()),
                               representation_unchanged_despite_change=int((changed_mask & ~np.any(matrices[1] != matrices[2], axis=1)).sum()))
                methods[name] = metrics
                score_tables.append(cases)
                joblib.dump(lof_model, sub / f"lof_{name}.pkl")
            pd.concat(score_tables, ignore_index=True).to_parquet(sub / "scores.parquet", index=False)
            used = source_columns(metadata)
            record = {"seed": seed, "train_rows": len(train), "test_rows": len(test),
                      "train_groups": int(split.loc[split.partition.eq("train"), "group_id"].nunique()),
                      "test_groups": int(split.loc[split.partition.eq("test"), "group_id"].nunique()),
                      "group_overlap": 0, "epochs": args.epochs, "autoencoder_training_performed": True,
                      "fit_scope": "training_partition_only", "source_columns_used": used,
                      "input_columns_not_used": sorted(set(raw)-set(used)),
                      "methods": methods,
                      "recall_difference_latent_minus_baseline": methods["latent"]["recall_of_changes_at_k"]-methods["baseline"]["recall_of_changes_at_k"]}
            write_json(sub / "metrics.json", record)
            repetitions.append(record)
            print(f"[Seed {seed}] Änderungen im Budget: ohne AE {methods['baseline']['selected_changed']}, "
                  f"mit AE {methods['latent']['selected_changed']} von {len(changes)}", flush=True)
        aggregate = {}
        for method in ("baseline", "latent"):
            aggregate[method] = {}
            for metric in ("recall_of_changes_at_k", "share_of_changes_in_review", "average_precision_for_changes", "roc_auc_for_changes"):
                values = [r["methods"][method][metric] for r in repetitions]
                aggregate[method][metric] = {"mean": float(np.mean(values)), "min": min(values), "max": max(values)}
        summary = {"protocol": PROTOCOL["name"], "row_count": len(raw), "seeds": args.seeds, "epochs": args.epochs,
                   "repetitions": repetitions, "aggregate": aggregate,
                   "paired_mean_recall_difference": float(np.mean([r["recall_difference_latent_minus_baseline"] for r in repetitions])),
                   "confirmed_error_metrics_available": False, "test_labels_used_for_fitting": False,
                   "source_input_unchanged": True, "delivery_memory_written": False,
                   "scope": PROTOCOL["scope"]}
        if file_sha256(source) != source_hash:
            raise ValueError("Quelldatei während des Versuchs verändert.")
        write_json(out / "summary.json", summary)
        render_comparison(out, summary, protocol)
        finish_output(out, {"source_input": str(source), "input_sha256": source_hash, "protocol": PROTOCOL["name"]})
        print(json.dumps({k: summary[k] for k in ("row_count", "seeds", "epochs", "paired_mean_recall_difference", "scope")}, ensure_ascii=False, indent=2))
        print(f"Ergebnisordner: {out}\nBericht: {out / 'bericht.html'}")
        return out
    except Exception as exc:
        write_json(out / "run_status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        print(f"Vergleich unvollständig: {out}", file=sys.stderr)
        raise


def main():
    p = parser()
    args = p.parse_args()
    try: analyze(args)
    except (ValueError, OSError, ImportError) as exc:
        p.exit(2, f"Vergleich abgebrochen: {exc}\n")


if __name__ == "__main__":
    main()
