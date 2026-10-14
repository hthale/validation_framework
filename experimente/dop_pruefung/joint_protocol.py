"""Festes DOP-Änderungsprotokoll; keine offizielle Produktspezifikation."""
from __future__ import annotations

from itertools import combinations
import math
import numpy as np
import pandas as pd

GROUP_COLUMNS = ["koordinatenreferenzsystem_lage", "koordinatenursprung_east", "koordinatenursprung_north"]
METHOD_LABELS = {
    "known": "Vorgegebene Konsistenzprüfungen",
    "profiling": "Gelernte Profiling-Regeln",
    "dependency": "Gelernte Abhängigkeitsregeln",
    "baseline": "LOF auf Eingangsmerkmalen",
    "latent": "LOF im latenten Raum",
    "rules": "Alle Regeln gemeinsam",
    "rules_baseline": "Alle Regeln + Eingangs-LOF",
    "rules_latent": "Alle Regeln + latenter LOF",
    "all": "Alle Regeln + beide LOF",
}
KNOWN = [
    {"rule_id": "known__spectral_bands", "columns": ["spektralkanaele", "tif_band_count"],
     "expression": "RGB -> 3 Bänder; RGBI -> 4 Bänder (explizite Testannahme)",
     "parameters": {"RGB": 3, "RGBI": 4}},
    {"rule_id": "known__pixel_consistency", "columns": ["bodenpixelgroesse", "tif_pixel_size_x", "tif_pixel_size_y"],
     "expression": "Bodenpixelgröße in cm = 100 * Betrag der Rasterpixelgröße in m, für beide Achsen",
     "parameters": {"absolute_tolerance_cm": 1e-6, "relative_tolerance": 0.0}},
    {"rule_id": "known__crs_consistency", "columns": ["koordinatenreferenzsystem_lage", "tif_crs"],
     "expression": "Numerischer EPSG-Code der Metadaten stimmt mit Raster-EPSG-Code überein", "parameters": {}},
    {"rule_id": "known__dimension_consistency", "columns": ["anzahl_spalten", "anzahl_zeilen", "tif_width", "tif_height"],
     "expression": "Positive ganzzahlige Metadatenmaße stimmen mit Rasterbreite und Rasterhöhe überein", "parameters": {}},
]
for _r in KNOWN:
    _r.update(source="predeclared_test_assumption", approved=False,
              note="Manuell vorgegebene Testannahme. Weder automatisch gelernt noch fachlich freigegeben.")

OPERATORS = [
    {"name": "band_count", "label": "Bandzahl + 1", "column": "tif_band_count", "reference": KNOWN[0]["rule_id"]},
    {"name": "pixel_size", "label": "Pixelgröße X mal 10", "column": "tif_pixel_size_x", "reference": KNOWN[1]["rule_id"]},
    {"name": "crs", "label": "Anderer EPSG-Code", "column": "tif_crs", "reference": KNOWN[2]["rule_id"]},
    {"name": "raster_width", "label": "Rasterbreite + 1", "column": "tif_width", "reference": KNOWN[3]["rule_id"]},
    {"name": "camera_swap", "label": "Andere beobachtete Kamera", "column": "kamera_sensor", "reference": None},
    {"name": "camera_missing", "label": "Kamera fehlt", "column": "kamera_sensor", "reference": None},
    {"name": "date_shift", "label": "Aktualität + 180 Tage", "column": "aktualitaet", "reference": None},
    {"name": "xml_size", "label": "XML-Größe + 3 Trainings-Standardabweichungen (auf Bytes aufgerundet)", "column": "xml_size_bytes", "reference": None},
]
PROTOCOL = {
    "name": "joint_dop_check_09", "version": 1,
    "scope": "Explorativer Änderungsversuch ab Parquet; keine reale Fehlererkennungsleistung",
    "split": "60/20/20 % der Gruppen: Training/Validierung/Test; Zeilenanteile können abweichen",
    "fit_scope": "Schema, Vorverarbeitung, Autoencoder, HDBSCAN, LOF und Regelableitung ausschließlich im Training",
    "rule_selection": "Gruppen-CV im Training; Güteschwellen nur auf Validierung; Regelauswahl ohne Zugriff auf Testdaten oder Änderungen, unverändert auf beide Testkopien angewandt",
    "profiling": "Bestehendes dreifaches Profiling mit latenten Trainingsclustern/LOF-Strata; mindestens 2 Vorschläge; Validierung >= 5 Fälle und >= 95 % Passrate",
    "dependency": "Bestehende automatische Zielauswahl und train_dependency; Modellgates Accuracy >= .95, Macro-F1 >= .90, Macro-Precision >= .90; Blattgate >= 5 Fälle und >= .95 Passrate",
    "known_checks": KNOWN,
    "operators": OPERATORS,
    "assignment": "Pro Typ exakt cases_per_type verschiedene Testzeilen; je Zeile genau eine Zelle; Auswahl ohne Scores. Fehlende Voraussetzungen führen zum Abbruch, nicht zum stillen Weglassen einer Klasse.",
    "lof": "Beide Varianten: novelty=True, Fit Training, identische Fälle und ceil(Testzeilen * review_fraction); Gleichstand nach Kennung",
    "baseline": "Standardisierte Zahlen/UTC-Tage, One-Hot-Kategorien und Zeichenhäufigkeiten; dieselben ausgewählten Felder, Trunkierung und Zeicheninformation wie MeanCharEncoder",
    "distance": "Euklidisch; keine nachträgliche Skalierung des latenten Raums oder Optimierung anhand der Änderungen",
    "constant_fields": "Auch im Training konstante Qualitätsfelder werden geändert; verworfene Felder bleiben im Nenner und werden als unsichtbar ausgewiesen",
    "union": "Vereinigung aller Regelhinweise und jeweiliger LOF-Auswahl; kein gemeinsames fixes Budget. Zusätzliche Treffer UND zusätzlicher Prüfaufwand werden berichtet.",
    "causal_scope": "Vorher/Nachher zeigt zusätzliche Hinweise. Neue Auswahl bei LOF hängt auch von anderen geänderten Zeilen im Rangbudget ab.",
    "limitations": [
        "Die vier Konsistenztests sind explizite Versuchsannahmen, keine verifizierte amtliche DOP-Spezifikation. RGBI wird nicht aus Dateinamen gefordert.",
        "Die ersten vier Änderungen verletzen ihren jeweiligen Konsistenztest. Ihre Erkennung durch genau diesen Test ist ein Funktionstest, keine unabhängige Leistungsvalidierung.",
        "Kamerawechsel, fehlende Kamera, Datum und Dateigröße sind kontrollierte Änderungen ohne bestätigtes fachliches Fehlerlabel.",
        "Unveränderte Originalzeilen sind nicht als fehlerfrei bestätigt; Hinweise darauf sind keine nachgewiesenen Fehlalarme.",
        "Mehrere Seeds verwenden überlappende Daten. Sie liefern beschreibende Wiederholungen, keine unabhängigen Studien oder Signifikanzprüfung.",
        "Koordinatengruppen garantieren keine räumliche, zeitliche oder lieferungsweite Unabhängigkeit.",
        "Die DOP-Daten wurden bereits untersucht. Dieser neue Versuch ist eine Entwicklungsprüfung; kein unberührter externer Abschlusstest.",
        "Profiling benutzt latente Trainingsstrata; der direkte Vergleich der beiden LOF-Varianten isoliert nur ihre Repräsentation. Die Regelbasis ist für beide Kombinationen identisch.",
        "Clusterbeschreibungen werden nicht als Qualitätsregeln gezählt. TIFF-/XML-/NAS-Einlesung, Geometrien und ALKIS sind nicht Teil dieses Tests.",
    ],
}


def require_columns(frame):
    required = {"record_key", *GROUP_COLUMNS}
    required.update(c for r in KNOWN for c in r["columns"])
    required.update(o["column"] for o in OPERATORS)
    missing = required - set(frame)
    if missing:
        raise ValueError(f"DOP-Testspalten fehlen: {sorted(missing)}")
    if frame.record_key.isna().any() or not frame.record_key.is_unique:
        raise ValueError("Eindeutige nichtleere Kennungen erforderlich.")


def known_results(frame):
    """Eine Zeile je Regel/Fall, mit explizitem Status für nicht anwendbare Fälle."""
    require_columns(frame)
    n = lambda col: pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    def positive_integer(s):
        return s.notna() & s.gt(0) & s.eq(s.round())
    target_bands = frame.spektralkanaele.astype("string").str.strip().str.upper().map(KNOWN[0]["parameters"])
    band = n("tif_band_count")
    band_applies = target_bands.notna()
    band_valid = band.eq(target_bands).fillna(False)
    cm, px, py = n("bodenpixelgroesse"), n("tif_pixel_size_x"), n("tif_pixel_size_y")
    pixel_applies = cm.gt(0) & cm.notna()
    pixel_valid = (px.abs().gt(0) & py.abs().gt(0)
                   & np.isclose(px.abs()*100, cm, atol=1e-6, rtol=0)
                   & np.isclose(py.abs()*100, cm, atol=1e-6, rtol=0))
    expected_crs = n("koordinatenreferenzsystem_lage")
    actual_crs = pd.to_numeric(frame.tif_crs.astype("string").str.strip().str.upper()
                               .str.extract(r"^(?:EPSG\s*:\s*)?(\d+)$", expand=False), errors="coerce")
    crs_applies = positive_integer(expected_crs)
    crs_valid = actual_crs.eq(expected_crs).fillna(False)
    cols, rows, width, height = (n(c) for c in ("anzahl_spalten", "anzahl_zeilen", "tif_width", "tif_height"))
    dim_applies = positive_integer(cols) & positive_integer(rows)
    dim_valid = positive_integer(width) & positive_integer(height) & cols.eq(width) & rows.eq(height)
    frames = []
    for rule, applies, valid in zip(KNOWN, (band_applies, pixel_applies, crs_applies, dim_applies),
                                    (band_valid, pixel_valid, crs_valid, dim_valid)):
        frames.append(pd.DataFrame({"record_key": frame.record_key.astype(str), "rule_id": rule["rule_id"],
            "applicable": applies.to_numpy(bool), "violation": (applies & ~valid).to_numpy(bool)}))
    return pd.concat(frames, ignore_index=True)


def inject_cases(test, train, cases_per_type, seed):
    """Single-cell-Änderungen anhand fester Voraussetzungen, niemals anhand eines Scores."""
    from review_common import json_value
    require_columns(test)
    if cases_per_type < 1 or len(test) <= cases_per_type * len(OPERATORS):
        raise ValueError("Für alle acht Typen und unveränderte Kontrollzeilen sind zu wenige Testzeilen vorhanden.")
    before = test.sort_values("record_key").reset_index(drop=True)
    after = before.copy(deep=True)
    reference = known_results(before).set_index(["record_key", "rule_id"])
    cameras = sorted(train.kamera_sensor.dropna().astype(str).unique())
    xml = pd.to_numeric(train.xml_size_bytes, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    step = math.ceil(3 * float(xml.std(ddof=0))) if len(xml) > 1 else 0
    if step < 1 or not math.isfinite(step):
        raise ValueError("XML-Größe variiert im Training nicht; vorgegebener Änderungstyp ist nicht ausführbar.")
    rng = np.random.default_rng(seed)
    used, log = set(), []
    for operator in OPERATORS:
        eligible = []
        for idx in range(len(before)):
            if idx in used:
                continue
            key, old = str(before.at[idx, "record_key"]), before.at[idx, operator["column"]]
            if pd.isna(old):
                continue
            ref = operator["reference"]
            if ref and (not reference.at[(key, ref), "applicable"] or reference.at[(key, ref), "violation"]):
                continue
            if operator["name"] == "camera_swap" and not any(v != str(old) for v in cameras):
                continue
            if operator["name"] == "date_shift" and pd.isna(pd.to_datetime(old, errors="coerce", utc=True)):
                continue
            if operator["name"] == "xml_size" and not np.isfinite(float(old)):
                continue
            eligible.append(idx)
        if len(eligible) < cases_per_type:
            raise ValueError(f"Für {operator['name']} nur {len(eligible)} geeignete Zeilen, benötigt: {cases_per_type}.")
        for idx in rng.choice(eligible, size=cases_per_type, replace=False):
            col, name = operator["column"], operator["name"]
            old = before.at[idx, col]
            if name in {"band_count", "raster_width"}: new = int(old) + 1
            elif name == "pixel_size": new = float(old) * 10
            elif name == "crs": new = "EPSG:4326" if "4326" not in str(old) else "EPSG:25832"
            elif name == "camera_swap": new = str(rng.choice([v for v in cameras if v != str(old)]))
            elif name == "camera_missing": new = None
            elif name == "date_shift":
                date = pd.to_datetime(old, utc=True) + pd.Timedelta(days=180)
                new = date.date().isoformat() if len(str(old)) == 10 else date.isoformat()
            else: new = int(old) + step
            # object erlaubt fehlende Werte ohne Änderung anderer Zellen durch dtype-Casts.
            if name in {"camera_missing", "crs", "date_shift", "camera_swap"}:
                after[col] = after[col].astype(object)
            after.at[idx, col] = new
            used.add(int(idx))
            log.append({"record_key": str(before.at[idx, "record_key"]), "operator": name, "column": col,
                        "before": json_value(old), "after": json_value(new),
                        "test_reference_rule": operator["reference"],
                        "is_confirmed_real_error": False})
    post = known_results(after).set_index(["record_key", "rule_id"])
    for change in log:
        if change["test_reference_rule"] and not post.at[(change["record_key"], change["test_reference_rule"]), "violation"]:
            raise ValueError("Geplante Konsistenzverletzung wurde nicht erzeugt.")
    return before, after, log


def evaluate_rules(frame, rules):
    from rule_engine import evaluate_rule_on_dataframe
    frames = []
    for rule in rules:
        applicable, _, invalid = evaluate_rule_on_dataframe(frame, rule)
        frames.append(pd.DataFrame({"record_key": frame.record_key.astype(str), "rule_id": rule["rule_id"],
                                   "applicable": applicable.to_numpy(bool), "violation": invalid.to_numpy(bool)}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({
        "record_key": pd.Series(dtype=str), "rule_id": pd.Series(dtype=str),
        "applicable": pd.Series(dtype=bool), "violation": pd.Series(dtype=bool)})


def count_signals(cases, before, after, mask=None):
    selected = np.ones(len(cases), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    changed = cases.changed.to_numpy(bool) & selected
    unchanged = ~cases.changed.to_numpy(bool) & selected
    before, after = np.asarray(before, bool), np.asarray(after, bool)
    total = int(changed.sum())
    return {"changed_rows": total, "selected_after": int((after & selected).sum()),
            "changed_selected_after": int((after & changed).sum()),
            "changed_selected_before": int((before & changed).sum()),
            "newly_selected_changed": int((after & ~before & changed).sum()),
            "no_longer_selected_changed": int((~after & before & changed).sum()),
            "unchanged_selected_after": int((after & unchanged).sum()),
            "recall_of_changes": float((after & changed).sum()/total) if total else None}


def joint_metrics(cases):
    """Keine Gleichsetzung von Budgetauswahl, Regelverletzung und fachlichem Fehler."""
    result = {"methods": {}, "per_operator": {}, "overlap_after": {}, "incremental": {}}
    for name in METHOD_LABELS:
        result["methods"][name] = count_signals(cases, cases[name+"_before"], cases[name+"_after"])
    for op in OPERATORS:
        result["per_operator"][op["name"]] = {name: count_signals(cases, cases[name+"_before"],
            cases[name+"_after"], cases.operator.eq(op["name"])) for name in METHOD_LABELS}
    for left, right in combinations(("known", "profiling", "dependency", "baseline", "latent"), 2):
        overlap = cases[left+"_after"] & cases[right+"_after"]
        result["overlap_after"][left+"__"+right] = {"rows": int(overlap.sum()), "changed_rows": int((overlap & cases.changed).sum())}
    for method, reference in (("baseline", "rules"), ("latent", "rules"), ("latent", "rules_baseline")):
        additional = cases[method+"_after"] & ~cases[reference+"_after"]
        new = additional & ~cases[method+"_before"] & cases.changed
        result["incremental"][method+"_beyond_"+reference] = {
            "additional_review_rows": int(additional.sum()),
            "additional_changed_rows": int((additional & cases.changed).sum()),
            "additional_unchanged_rows": int((additional & ~cases.changed).sum()),
            "newly_selected_additional_changed": int(new.sum())}
    return result
