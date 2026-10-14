"""Beschreibende Bäume für feste Cluster: Datumsfelder und Komplexitätsvergleich.

Alle Kennzahlen sind Anpassungsgüte im selben Bestand, keine unabhängige
Vorhersagegüte und keine automatische fachliche Regelfreigabe.
"""
from __future__ import annotations

from dataclasses import replace
from html import escape
import numpy as np
import pandas as pd

from phase2_ml import build_ssl_cluster_rules
from record_identity import validate_record_keys
from rule_conditions import condition_text


DEPTHS = (4, 6, 8)
FIDELITY_TARGET = .90
SELECTION = (
    "Kleinster Baum mit mindestens 90 % Übereinstimmung und einem vorhergesagten "
    "Blatt für jedes eingeschlossene Cluster; sonst höchste Übereinstimmung, "
    "bei Gleichstand höherer Macro-Recall und weniger Blätter. Nur In-Sample-Beschreibung."
)


def cluster_frame(normalized, contract, latent, include_dates=True):
    for frame in (normalized, latent):
        validate_record_keys(frame["record_key"])
    if set(normalized.record_key) != set(latent.record_key):
        raise ValueError("Clusterkennungen stimmen nicht vollständig mit der Eingabe überein.")
    cols = [c for c, d in contract["columns"].items()
            if d["role"] in {"numeric", "categorical"} and c in normalized]
    work = normalized[["record_key", *cols]].copy()
    derived = []
    if include_dates:
        for col, info in contract["columns"].items():
            if info["role"] != "date" or col not in normalized:
                continue
            name = f"__generic_cluster_date_{len(derived)}"
            if name in normalized:
                raise ValueError(f"Reservierter abgeleiteter Spaltenname: {name}")
            parsed = pd.to_datetime(normalized[col], errors="coerce", utc=True)
            work[name] = parsed.map(lambda v: v.timestamp()/86400 if pd.notna(v) else np.nan)
            derived.append({"source": col, "feature": name,
                            "transformation": "UTC days since Unix epoch"})
    joined = work.merge(latent[["record_key", "cluster_label"]], on="record_key",
                        validate="one_to_one", sort=False)
    labels = pd.to_numeric(joined.cluster_label, errors="coerce")
    if labels.isna().any() or not np.isfinite(labels).all() or not (labels == np.floor(labels)).all() or (labels < -1).any():
        raise ValueError("Clusterlabels müssen ganze Zahlen ab -1 sein.")
    joined["cluster_label"] = labels.astype(int)
    return joined, derived


def _display_expression(rule, derived):
    mapping = {d["feature"]: d for d in derived}
    parts = []
    for condition in rule["metadata"]["conditions"]:
        if condition["feature"] not in mapping:
            parts.append(condition_text(condition))
            continue
        source = mapping[condition["feature"]]["source"]
        threshold = pd.Timestamp("1970-01-01", tz="UTC") + pd.to_timedelta(condition["value"], unit="D")
        parts.append(f"{source} {condition['operator']} {threshold.isoformat()} (Datum, angenähert)")
    return "IF " + " AND ".join(parts) + f" THEN cluster_label = {rule['metadata']['predicted_value']}"


def describe_clusters(normalized, contract, latent, settings):
    variants = []
    artifacts = {}
    for dates, depth in [(False, 4), *((True, d) for d in DEPTHS)]:
        work, derived = cluster_frame(normalized, contract, latent, dates)
        rules, reports = build_ssl_cluster_rules(work, settings=replace(settings, cluster_max_depth=depth))
        report = reports[0]
        variant_id = f"{'mit_datum' if dates else 'ohne_datum'}_tiefe_{depth}"
        accuracy = report.get("in_sample_fidelity", {}).get("accuracy")
        lengths = [len(r["metadata"]["conditions"]) for r in rules]
        variant = {
            "variant": variant_id, "dates_included": dates, "max_depth": depth,
            "leaf_rule_count": len(rules), "mean_conditions": float(np.mean(lengths)) if lengths else None,
            "max_conditions": max(lengths, default=0),
            "in_sample_fidelity": report.get("in_sample_fidelity", {}),
            "predicted_cluster_count": len(report.get("predicted_clusters", [])),
            "included_cluster_count": len(report.get("included_clusters", [])),
            "clusters_without_predicted_leaf": report.get("clusters_without_predicted_leaf", []),
            "fidelity_target_met": bool(accuracy is not None and accuracy >= FIDELITY_TARGET
                                        and not report.get("clusters_without_predicted_leaf")),
            "success": report["success"], "reason": report["reason"],
        }
        variants.append(variant)
        artifacts[variant_id] = (rules, report, derived)
    usable = [v for v in variants if v["dates_included"] and v["success"]]
    if not usable:
        chosen = variants[1]
    else:
        satisfying = [v for v in usable if v["fidelity_target_met"]]
        if satisfying:
            chosen = min(satisfying, key=lambda v: (v["leaf_rule_count"], v["max_depth"]))
        else:
            chosen = max(usable, key=lambda v: (v["in_sample_fidelity"]["accuracy"],
                                                v["in_sample_fidelity"]["recall_macro"],
                                                -v["leaf_rule_count"], -v["max_depth"]))
    rules, report, derived = artifacts[chosen["variant"]]
    report.update(description_schema_version=2, date_features=derived, complexity_comparison=variants,
                  selected_variant=chosen["variant"], selection_policy=SELECTION,
                  fidelity_target=FIDELITY_TARGET, fidelity_target_met=chosen["fidelity_target_met"],
                  quality_rules_approved=False, test_evaluated=False,
                  date_display_note="Datumsanzeige ist angenähert; verbindlich sind UTC-Tage, Imputation und float32-Bedingungen in metadata.")
    for rule in rules:
        rule["display_expression"] = _display_expression(rule, derived)
        rule["metadata"].update(derived_features=derived, selected_variant=chosen["variant"],
                                 approved=False, date_display_note=report["date_display_note"])
    return rules, [report]


def complexity_html(reports):
    """Kompakte Tabelle für generischen Bericht und gesonderte Neuauswertung."""
    html = ""
    for report in reports:
        comparisons = report.get("complexity_comparison", [])
        if not comparisons:
            continue
        rows = []
        for v in comparisons:
            metrics = v["in_sample_fidelity"]
            def pct(value): return "—" if value is None else f"{100*value:.1f} %"
            rows.append("<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in (
                v["variant"] + (" · ausgewählt" if v["variant"] == report["selected_variant"] else ""),
                v["leaf_rule_count"], v["max_conditions"], pct(metrics.get("accuracy")),
                pct(metrics.get("recall_macro")), f"{v['predicted_cluster_count']}/{v['included_cluster_count']}"
            )) + "</tr>")
        html += ("<p>Die Tabelle zeigt, wie genau ein lesbarer Baum die vorhandenen Cluster nachbildet. "
                 "Eine hohe Übereinstimmung bestätigt weder die Clusterqualität noch die Fehlererkennung.</p>"
                 "<div class='scroll'><table><thead><tr><th>Variante</th><th>Blattregeln</th>"
                 "<th>Max. Bedingungen</th><th>Übereinstimmung</th><th>Macro-Recall</th>"
                 "<th>Als Blatt vorhergesagte Cluster</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
                 f"<p>{escape(SELECTION)} Ziel erreicht: {'Ja' if report['fidelity_target_met'] else 'Nein'}.</p>"
                 f"<p>{report['model_rows']} beschriebene Zeilen; "
                 f"{report['excluded_noise_or_missing_rows']} Rauschpunkte; "
                 f"{report['excluded_small_cluster_rows']} Zeilen in zu kleinen Clustern.</p>")
    return html
