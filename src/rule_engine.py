from __future__ import annotations

from collections import defaultdict
import re
import pandas as pd
from rule_conditions import evaluate_conditions, normalize_condition



def format_numeric_value(value: float):
    """
    Formatiert numerische Werte lesbar.
    Ganzzahlen werden ohne .0 ausgegeben.
    """
    if value is None:
        return None
    if float(value).is_integer():
        return int(value)
    return value


def make_rule(
    rule_id: str,
    rule_name: str,
    rule_type: str,
    columns: list[str],
    expression: str,
    source: str,
    reason: str,
    origin_runs: list[int] | None = None,
    metadata: dict | None = None,
    evidence_type: str | None = None,
    evidence_detail: str | None = None,
) -> dict:
    """
    Erstellt ein einheitliches Regelobjekt.
    evidence_type / evidence_detail dokumentieren die Herkunft
    einer Regel methodisch sauberer für die Thesis.
    """
    #Grundstruktur der Regeln
    return {
        "rule_id": rule_id,
        "rule_name": rule_name,
        "rule_type": rule_type,
        "columns": columns,
        "expression": expression,
        "source": source,
        "status": "proposed",
        "reason": reason,
        "origin_runs": origin_runs or [],
        "metadata": metadata or {},
        "evidence_type": evidence_type or "",
        "evidence_detail": evidence_detail or "",
    }

#null_rate: Anteil fehlender Werte
#Beispiel: 0.0 bedeutet: keine Nullwerte
#const_stddev_range_threshold: Schwellwert für den Hinweis stable_numeric
#max_numeric_distinct_for_in_rule: Wenn numerische Spalte nur wenige verschiedene Werte hat, lieber IN (...) statt range
#enable_string_rules: Ob Regex-Regeln überhaupt erzeugt werden sollen
#min_regex_support_ratio: Mindestanteil, wie stark ein Regex-Muster im Sample unterstützt sein muss
#enable_string_length_rules: Ob String-Längenregeln erzeugt werden sollen
def build_single_column_rule_proposals(
    col: str,
    dtype_group: str,
    profile: dict,
    null_rate: float,
    const_stddev_range_threshold: float,
    max_numeric_distinct_for_in_rule: int,
    enable_string_rules: bool,
    min_regex_support_ratio: float,
    enable_string_length_rules: bool,
) -> tuple[list[dict], list[dict]]:
    """
    Erzeugt Regelvorschläge aus Profiling-Ergebnissen.
    Zusätzlich werden reine Hinweise gespeichert, wenn eine
    Beobachtung noch nicht als valide Regel materialisiert wird.
    """
    proposed_rules = []
    profiling_hints = []


    if null_rate == 0:
        proposed_rules.append(
            make_rule(
                rule_id=f"{col}__not_null",
                rule_name=f"{col} must not be null",
                rule_type="not_null",
                columns=[col],
                expression=f"{col} IS NOT NULL",
                source="profiling",
                reason="Spalte ist im Sample immer befüllt",
                evidence_type="profiling_runs",
                evidence_detail="Regel in allen betrachteten Sample-Läufen beobachtet",
            )
        )
    # alle Werte gleich → constant, wenige diskrete Werte → in_set, sonst range
    if dtype_group == "numeric":
        min_val = profile.get("min")
        max_val = profile.get("max")
        distinct_count = profile.get("distinct_count")
        distinct_values_if_small = profile.get("distinct_values_if_small")
        std_range_ratio = profile.get("stability_ratio_stddev_div_range")

        if min_val is not None and max_val is not None:
            min_fmt = format_numeric_value(min_val)
            max_fmt = format_numeric_value(max_val)

            if min_val == max_val:
                proposed_rules.append(
                    make_rule(
                        rule_id=f"{col}__constant",
                        rule_name=f"{col} constant value",
                        rule_type="constant",
                        columns=[col],
                        expression=f"{col} = {min_fmt}",
                        source="profiling",
                        reason="Alle Werte im Sample identisch",
                        metadata={"value": min_fmt},
                        evidence_type="profiling_runs",
                        evidence_detail="Beobachtete Konstanz im Sample",
                    )
                )
            elif distinct_values_if_small is not None and distinct_count <= max_numeric_distinct_for_in_rule:
                formatted_values = [format_numeric_value(v) for v in distinct_values_if_small]
                proposed_rules.append(
                    make_rule(
                        rule_id=f"{col}__in_set_numeric",
                        rule_name=f"{col} limited value set",
                        rule_type="in_set_numeric",
                        columns=[col],
                        expression=f"{col} IN ({', '.join(str(v) for v in formatted_values)})",
                        source="profiling",
                        reason="Wenige diskrete numerische Werte im Sample",
                        metadata={"allowed_values": formatted_values},
                        evidence_type="profiling_runs",
                        evidence_detail="Kleine diskrete Wertemenge im Sample",
                    )
                )
            else:
                proposed_rules.append(
                    make_rule(
                        rule_id=f"{col}__range",
                        rule_name=f"{col} observed range",
                        rule_type="range",
                        columns=[col],
                        expression=f"{col} >= {min_fmt} AND {col} <= {max_fmt}",
                        source="profiling",
                        reason="Beobachteter Wertebereich im Sample",
                        metadata={"min": min_fmt, "max": max_fmt},
                        evidence_type="profiling_runs",
                        evidence_detail="Range aus Profiling-Min/Max",
                    )
                )

        if std_range_ratio is not None and std_range_ratio < const_stddev_range_threshold:
            profiling_hints.append({
                "column": col,
                "hint_type": "stable_numeric",
                "message": f"stddev/range = {std_range_ratio:.4f} < {const_stddev_range_threshold}",
            })
    # Für kategoriale Spalten: Wenn alle Werte im Sample beobachtet wurden, IN (...); wenn nur ein Wert, dann constant_category
    elif dtype_group == "categorical":
        values = profile.get("all_values_if_small")
        if values:
            quoted = ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)
            proposed_rules.append(
                make_rule(
                    rule_id=f"{col}__in_set",
                    rule_name=f"{col} categorical allowed set",
                    rule_type="in_set",
                    columns=[col],
                    expression=f"{col} IN ({quoted})",
                    source="profiling",
                    reason="Wertebereich der Kategorie im Sample vollständig beobachtet",
                    metadata={"allowed_values": values},
                    evidence_type="profiling_runs",
                    evidence_detail="Kleine kategoriale Wertemenge im Sample",
                )
            )

        top_values = profile.get("top_values", {})
        if len(top_values) == 1:
            only_val = next(iter(top_values.keys()))
            escaped_val = str(only_val).replace("'", "''")
            proposed_rules.append(
                make_rule(
                    rule_id=f"{col}__constant_category",
                    rule_name=f"{col} constant category",
                    rule_type="constant_category",
                    columns=[col],
                    expression=f"{col} = '{escaped_val}'",
                    source="profiling",
                    reason="Nur ein kategorialer Wert im Sample",
                    metadata={"value": only_val},
                    evidence_type="profiling_runs",
                    evidence_detail="Konstante kategoriale Ausprägung im Sample",
                )
            )
    # Für Strings: Regex-Muster, wenn stark unterstützt; optional auch Längenregel
    elif dtype_group == "string":
        pattern_info = profile.get("pattern_info", {})
        pattern = pattern_info.get("pattern")
        support_ratio = pattern_info.get("support_ratio", 0.0)
        length_min = profile.get("length_min")
        length_max = profile.get("length_max")

        if pattern:
            profiling_hints.append({
                "column": col,
                "hint_type": "regex_pattern",
                "message": pattern,
            })

            # Optional: echte Regex-Regel erzeugen, wenn das Muster stabil genug erscheint
            if enable_string_rules and support_ratio >= min_regex_support_ratio:
                proposed_rules.append(
                    make_rule(
                        rule_id=f"{col}__regex",
                        rule_name=f"{col} regex pattern",
                        rule_type="regex_match",
                        columns=[col],
                        expression=f"REGEX({col}, '{pattern}')",
                        source="profiling",
                        reason="Dominantes String-Muster im Sample",
                        metadata={"pattern": pattern},
                        evidence_type="profiling_runs",
                        evidence_detail=f"Regex mit support_ratio={support_ratio:.4f}",
                    )
                )

        # Optional: Längenregel materialisieren
        if enable_string_length_rules and length_min is not None and length_max is not None:
            proposed_rules.append(
                make_rule(
                    rule_id=f"{col}__length_range",
                    rule_name=f"{col} string length range",
                    rule_type="length_range",
                    columns=[col],
                    expression=f"LENGTH({col}) BETWEEN {length_min} AND {length_max}",
                    source="profiling",
                    reason="Beobachteter String-Längenbereich im Sample",
                    metadata={"min_length": length_min, "max_length": length_max},
                    evidence_type="profiling_runs",
                    evidence_detail="String-Längenbereich aus Profiling",
                )
            )
    # aktuell nur Profiling-Hinweis, keine echte Regel, da Datumserkennung und -vergleich komplexer ist und im Sample ggf. nicht stabil genug erscheint
    elif dtype_group == "date":
        min_val = profile.get("min")
        max_val = profile.get("max")
        if min_val is not None and max_val is not None:
            profiling_hints.append({
                "column": col,
                "hint_type": "date_range",
                "message": f"{min_val} .. {max_val}",
            })

    return proposed_rules, profiling_hints


def attach_single_column_rule_proposals(
    profile_result: dict,
    const_stddev_range_threshold: float,
    max_numeric_distinct_for_in_rule: int,
    enable_string_rules: bool,
    min_regex_support_ratio: float,
    enable_string_length_rules: bool,
) -> dict:
    """
    Hängt die aus dem Profiling abgeleiteten Regelvorschläge
    an das Profiling-Ergebnis an.
    """
    proposed_rules = []
    profiling_hints = []

    for col, col_info in profile_result["column_profiles"].items():
        dtype_group = col_info["dtype_group"]
        null_rate = col_info["null_rate"]
        profile = col_info["profile"]

        rules, hints = build_single_column_rule_proposals(
            col=col,
            dtype_group=dtype_group,
            profile=profile,
            null_rate=null_rate,
            const_stddev_range_threshold=const_stddev_range_threshold,
            max_numeric_distinct_for_in_rule=max_numeric_distinct_for_in_rule,
            enable_string_rules=enable_string_rules,
            min_regex_support_ratio=min_regex_support_ratio,
            enable_string_length_rules=enable_string_length_rules,
        )
        proposed_rules.extend(rules)
        profiling_hints.extend(hints)

    profile_result["proposed_rules"] = proposed_rules
    profile_result["profiling_hints"] = profiling_hints
    return profile_result


def merge_stable_proposed_rules(all_rule_sets: list[list[dict]], min_occurrences: int) -> list[dict]:
    """
    Führt Regeln aus mehreren Läufen zusammen.
    Beibehalten werden Regeln, die in mindestens min_occurrences Läufen vorkommen.
    """
    grouped = defaultdict(list)

    for run_idx, rules in enumerate(all_rule_sets, start=1):
        seen_in_run = set()
        for rule in rules:
            key = (
                tuple(rule["columns"]),
                rule["rule_type"],
                rule["expression"],
            )
            if key in seen_in_run:
                continue
            seen_in_run.add(key)
            grouped[key].append({"run": run_idx, "rule": rule})

    stable_rules = []
    for _, occurrences in grouped.items():
        if len(occurrences) >= min_occurrences:
            base_rule = occurrences[0]["rule"].copy()
            base_rule["origin_runs"] = [entry["run"] for entry in occurrences]
            base_rule["metadata"] = {
                **base_rule.get("metadata", {}),
                "sample_occurrence_count": len(occurrences),
            }
            stable_rules.append(base_rule)

    stable_rules.sort(key=lambda r: (r["rule_type"], ",".join(r["columns"]), r["expression"]))
    return stable_rules


def build_domain_specific_rules(
    df: pd.DataFrame, *, expected_spectral_channels: str | None = None
) -> list[dict]:
    """
    Baut manuell definierte bzw. fachlich angenommene Domänenregeln.
    Wichtig für die Thesis:
    Diese Regeln sind methodisch NICHT das Ergebnis von Profiling,
    sondern explizit modellierte Fachannahmen.
    """
    rules = []

    if {"bodenpixelgroesse", "standardabweichung"}.issubset(df.columns):
        rules.append(
            make_rule(
                rule_id="domain__pixel_size_to_stddev",
                rule_name="bodenpixelgroesse determines standardabweichung",
                rule_type="cross_column_map",
                columns=["bodenpixelgroesse", "standardabweichung"],
                expression="IF bodenpixelgroesse=10 THEN standardabweichung=20; IF bodenpixelgroesse=20 THEN standardabweichung=40",
                source="domain",
                reason="Fachregel für DOP-Metadaten",
                metadata={"mapping": {"10": 20, "20": 40}},
                evidence_type="project_assumption",
                evidence_detail="Projektintern modellierte Fachregel; extern zu belegen",
            )
        )

    if "dateiformat" in df.columns:
        rules.append(
            make_rule(
                rule_id="domain__dateiformat_cogtiff",
                rule_name="dateiformat must be Cloud Optimized GeoTIFF",
                rule_type="constant_category",
                columns=["dateiformat"],
                expression="dateiformat = 'Cloud Optimized GeoTIFF'",
                source="domain",
                reason="Erwartetes DOP-Ausgabeformat",
                metadata={"value": "Cloud Optimized GeoTIFF"},
                evidence_type="project_assumption",
                evidence_detail="Projektintern modellierte Fachregel; externe Spezifikation empfohlen",
            )
        )

    if "koordinatenreferenzsystem_lage" in df.columns:
        rules.append(
            make_rule(
                rule_id="domain__crs_lage_25832",
                rule_name="CRS Lage must be 25832",
                rule_type="constant",
                columns=["koordinatenreferenzsystem_lage"],
                expression="koordinatenreferenzsystem_lage = 25832",
                source="domain",
                reason="Erwartetes CRS Lage",
                metadata={"value": 25832},
                evidence_type="project_assumption",
                evidence_detail="Projektintern modellierte Fachregel; externe Spezifikation empfohlen",
            )
        )

    if "koordinatenreferenzsystem_hoehe" in df.columns:
        rules.append(
            make_rule(
                rule_id="domain__crs_hoehe_7837",
                rule_name="CRS Höhe must be 7837",
                rule_type="constant",
                columns=["koordinatenreferenzsystem_hoehe"],
                expression="koordinatenreferenzsystem_hoehe = 7837",
                source="domain",
                reason="Erwartetes CRS Höhe",
                metadata={"value": 7837},
                evidence_type="project_assumption",
                evidence_detail="Projektintern modellierte Fachregel; externe Spezifikation empfohlen",
            )
        )

    if expected_spectral_channels not in {None, "RGB", "RGBI"}:
        raise ValueError("expected_spectral_channels muss None, 'RGB' oder 'RGBI' sein.")
    if "spektralkanaele" in df.columns and expected_spectral_channels is not None:
        rules.append(
            make_rule(
                rule_id=f"domain__spektralkanaele_{expected_spectral_channels.lower()}",
                rule_name=f"spektralkanaele must be {expected_spectral_channels}",
                rule_type="constant_category",
                columns=["spektralkanaele"],
                expression=f"spektralkanaele = '{expected_spectral_channels}'",
                source="domain",
                reason="Erwartete Kanalbelegung",
                metadata={"value": expected_spectral_channels},
                evidence_type="configured_project_requirement",
                evidence_detail="Explizite Projektvorgabe; fachliche Freigabe separat dokumentieren",
            )
        )

    if {"xml", "tif", "kachelname"}.issubset(df.columns):
        rules.append(
            make_rule(
                rule_id="domain__xml_tif_kachelname_consistency",
                rule_name="xml tif and kachelname must match",
                rule_type="cross_column_consistency",
                columns=["xml", "tif", "kachelname"],
                expression="basename(xml)=basename(tif)=kachelname",
                source="domain",
                reason="Datei- und Kachelbezeichner müssen konsistent sein",
                evidence_type="project_assumption",
                evidence_detail="Projektintern modellierte Fachregel; externe Spezifikation empfohlen",
            )
        )

    if {"kachelname", "aktualitaet"}.issubset(df.columns):
        rules.append(
            make_rule(
                rule_id="domain__kachelname_date_matches_aktualitaet",
                rule_name="date in kachelname must match aktualitaet",
                rule_type="cross_column_consistency",
                columns=["kachelname", "aktualitaet"],
                expression="date_from(kachelname)=aktualitaet",
                source="domain",
                reason="Datum im Kachelnamen muss zur Aktualität passen",
                evidence_type="project_assumption",
                evidence_detail="Projektintern modellierte Fachregel; externe Spezifikation empfohlen",
            )
        )

    if {"xml", "aktualitaet"}.issubset(df.columns):
        rules.append(
            make_rule(
                rule_id="domain__xml_date_matches_aktualitaet",
                rule_name="date in xml path must match aktualitaet",
                rule_type="cross_column_consistency",
                columns=["xml", "aktualitaet"],
                expression="date_from(xml)=aktualitaet",
                source="domain",
                reason="Datum im XML-Pfad muss zur Aktualität passen",
                evidence_type="project_assumption",
                evidence_detail="Projektintern modellierte Fachregel; externe Spezifikation empfohlen",
            )
        )

    if {"tif", "aktualitaet"}.issubset(df.columns):
        rules.append(
            make_rule(
                rule_id="domain__tif_date_matches_aktualitaet",
                rule_name="date in tif path must match aktualitaet",
                rule_type="cross_column_consistency",
                columns=["tif", "aktualitaet"],
                expression="date_from(tif)=aktualitaet",
                source="domain",
                reason="Datum im TIF-Pfad muss zur Aktualität passen",
                evidence_type="project_assumption",
                evidence_detail="Projektintern modellierte Fachregel; externe Spezifikation empfohlen",
            )
        )

    if {"xml", "koordinatenursprung_east", "koordinatenursprung_north"}.issubset(df.columns):
        rules.append(
            make_rule(
                rule_id="domain__xml_coords_match_origin",
                rule_name="east north in xml path must match origin coordinates",
                rule_type="cross_column_consistency",
                columns=["xml", "koordinatenursprung_east", "koordinatenursprung_north"],
                expression="coords_from(xml)=origin_coords",
                source="domain",
                reason="Koordinaten im XML-Pfad müssen zu den Ursprungskoordinaten passen",
                evidence_type="project_assumption",
                evidence_detail="Projektintern modellierte Fachregel; externe Spezifikation empfohlen",
            )
        )

    if {"tif", "koordinatenursprung_east", "koordinatenursprung_north"}.issubset(df.columns):
        rules.append(
            make_rule(
                rule_id="domain__tif_coords_match_origin",
                rule_name="east north in tif path must match origin coordinates",
                rule_type="cross_column_consistency",
                columns=["tif", "koordinatenursprung_east", "koordinatenursprung_north"],
                expression="coords_from(tif)=origin_coords",
                source="domain",
                reason="Koordinaten im TIF-Pfad müssen zu den Ursprungskoordinaten passen",
                evidence_type="project_assumption",
                evidence_detail="Projektintern modellierte Fachregel; externe Spezifikation empfohlen",
            )
        )

    return rules


def _extract_date_from_text(value: object) -> str | None:
    """
    Extrahiert ein Datum im Format YYYY-MM-DD aus einem String.
    """
    if pd.isna(value):
        return None
    text = str(value)
    match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    return match.group(1) if match else None


def _extract_basename_without_extension(value: object) -> str | None:
    """
    Extrahiert den Dateinamen ohne Pfad und ohne Endung.
    """
    if pd.isna(value):
        return None
    text = str(value).strip()
    if "/" in text:
        text = text.split("/")[-1]
    if "." in text:
        text = ".".join(text.split(".")[:-1])
    return text or None


def _extract_coords_from_path(value: object) -> tuple[str | None, str | None]:
    """
    Extrahiert East/North aus einer 9-stelligen Kachelkennung
    am Anfang des Pfades, z. B. 326125800 -> 612000 / 5800000.
    """
    if pd.isna(value):
        return None, None

    text = str(value)
    match = re.search(r"^(\d{9})", text)
    if not match:
        return None, None

    num = match.group(1)
    east = num[2:5] + "000"
    north = num[5:9] + "000"

    return east, north


def _evaluate_rule_mask(df: pd.DataFrame, rule: dict) -> pd.Series:
    """
    Prüft die eigentliche Regelbedingung.
    Rückgabe:
    - True  = Regel erfüllt
    - False = Regel verletzt
    """
    rule_type = rule["rule_type"]
    columns = rule["columns"]
    metadata = rule.get("metadata", {})

    if rule_type == "not_null":
        col = columns[0]
        return df[col].notna()

    if rule_type in {"constant", "constant_category"}:
        col = columns[0]
        value = metadata.get("value")
        return df[col] == value

    if rule_type in {"in_set", "in_set_numeric"}:
        col = columns[0]
        allowed_values = metadata.get("allowed_values", [])
        return df[col].isin(allowed_values)

    if rule_type == "range":
        col = columns[0]
        min_val = metadata.get("min")
        max_val = metadata.get("max")
        return df[col].between(min_val, max_val, inclusive="both")

    if rule_type == "regex_match":
        col = columns[0]
        pattern = metadata.get("pattern")
        return df[col].astype("string").str.match(pattern, na=False)

    if rule_type == "length_range":
        col = columns[0]
        min_length = metadata.get("min_length")
        max_length = metadata.get("max_length")
        lengths = df[col].astype("string").str.len()
        return lengths.between(min_length, max_length, inclusive="both")

    if rule_type == "cross_column_map":
        if rule["rule_id"] == "domain__pixel_size_to_stddev":
            pixel = df["bodenpixelgroesse"]
            std = df["standardabweichung"]
            return (
                ((pixel == 10) & (std == 20)) |
                ((pixel == 20) & (std == 40))
            )
    
    if rule_type == "cross_column_consistency":
        if rule["rule_id"] == "domain__xml_tif_kachelname_consistency":
            xml_base = df["xml"].apply(_extract_basename_without_extension).astype("string")
            tif_base = df["tif"].apply(_extract_basename_without_extension).astype("string")
            kachel = df["kachelname"].astype("string")
            return (xml_base == tif_base) & (xml_base == kachel)

        if rule["rule_id"] == "domain__kachelname_date_matches_aktualitaet":
            extracted = df["kachelname"].apply(_extract_date_from_text)
            actual = pd.to_datetime(df["aktualitaet"], errors="coerce").dt.strftime("%Y-%m-%d")
            return extracted == actual

        if rule["rule_id"] == "domain__xml_date_matches_aktualitaet":
            extracted = df["xml"].apply(_extract_date_from_text)
            actual = pd.to_datetime(df["aktualitaet"], errors="coerce").dt.strftime("%Y-%m-%d")
            return extracted == actual

        if rule["rule_id"] == "domain__tif_date_matches_aktualitaet":
            extracted = df["tif"].apply(_extract_date_from_text)
            actual = pd.to_datetime(df["aktualitaet"], errors="coerce").dt.strftime("%Y-%m-%d")
            return extracted == actual

        if rule["rule_id"] == "domain__xml_coords_match_origin":
            extracted = df["xml"].apply(_extract_coords_from_path)
            east_extracted = extracted.apply(lambda x: x[0])
            north_extracted = extracted.apply(lambda x: x[1])
            east_actual = df["koordinatenursprung_east"].astype("Int64").astype("string")
            north_actual = df["koordinatenursprung_north"].astype("Int64").astype("string")
            return (east_extracted == east_actual) & (north_extracted == north_actual)

        if rule["rule_id"] == "domain__tif_coords_match_origin":
            extracted = df["tif"].apply(_extract_coords_from_path)
            east_extracted = extracted.apply(lambda x: x[0])
            north_extracted = extracted.apply(lambda x: x[1])
            east_actual = df["koordinatenursprung_east"].astype("Int64").astype("string")
            north_actual = df["koordinatenursprung_north"].astype("Int64").astype("string")
            return (east_extracted == east_actual) & (north_extracted == north_actual)


    if rule_type in {"ml_dependency", "ssl_cluster_rule"}:
        metadata = rule.get("metadata", {})
        target_col = metadata.get("target_col")
        predicted_value = metadata.get("predicted_value")
        # Die Wenn-Bedingung wird ausschließlich in der Anwendbarkeit geprüft.
        # Außerhalb ihres Geltungsbereichs zählt eine Regel nicht als bestanden.
        return df[target_col] == predicted_value

    raise ValueError(f"Nicht unterstützte Regel: {rule.get('rule_id')} ({rule_type})")


def _compute_applicable_mask(df: pd.DataFrame, rule: dict) -> pd.Series:
    """
    Bestimmt, auf welche Zeilen eine Regel überhaupt anwendbar ist.
    """
    rule_type = rule["rule_type"]
    columns = rule["columns"]

    if rule_type == "not_null":
        # Fehlende Werte müssen gerade von dieser Regel geprüft werden.
        return pd.Series(True, index=df.index, dtype=bool)

    if rule_type in {"ml_dependency", "ssl_cluster_rule"}:
        return evaluate_conditions(df, rule["metadata"]["conditions"])

    if rule_type == "cross_column_map" and rule["rule_id"] == "domain__pixel_size_to_stddev":
        # Für andere Pixelgrößen macht diese konkrete Wenn-dann-Regel keine Aussage.
        return df["bodenpixelgroesse"].isin([10, 20])

    if rule_type in {
        "constant",
        "constant_category",
        "in_set",
        "in_set_numeric",
        "range",
        "regex_match",
        "length_range",
    }:
        col = columns[0]

        if col not in df.columns:
            return pd.Series(False, index=df.index)

        return df[col].notna()

    # Für Cross-Column- und sonstige Mehrspalten-Regeln
    missing_cols = [c for c in columns if c not in df.columns]
    if missing_cols:
        return pd.Series(False, index=df.index)

    return df[columns].notna().all(axis=1)

def evaluate_rule_on_dataframe(df: pd.DataFrame, rule: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Gibt drei Masken zurück:
    - applicable_mask: Regel ist fachlich/technisch anwendbar
    - valid_mask: Regel ist erfüllt
    - invalid_mask: Regel ist verletzt

    Wichtig:
    invalid_mask wird nur dort gesetzt, wo applicable_mask True ist.
    """
    required = list(rule["columns"])
    if rule["rule_type"] in {"ml_dependency", "ssl_cluster_rule"}:
        metadata = rule.get("metadata", {})
        if not metadata.get("target_col") or "conditions" not in metadata or "predicted_value" not in metadata:
            raise ValueError("Unvollständige Metadaten einer Baumregel.")
        required += [metadata["target_col"]]
        required += [normalize_condition(c)["feature"] for c in metadata["conditions"]]
    if not required:
        raise ValueError("Regel ohne referenzierte Spalten.")
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Für die Regel fehlen Spalten: {missing}")

    applicable_mask = _compute_applicable_mask(df, rule).fillna(False).astype(bool)
    raw_valid_mask = _evaluate_rule_mask(df, rule).fillna(False)

    valid_mask = applicable_mask & raw_valid_mask
    invalid_mask = applicable_mask & (~raw_valid_mask)

    return applicable_mask, valid_mask, invalid_mask


def validate_rules_on_full_dataframe(
    df: pd.DataFrame,
    rules: list[dict],
    # maximal 20 Beispiel-Fehler pro Regel gespeichert
    max_example_violations: int = 20,
) -> tuple[list[dict], dict]:
    """
    Validiert Regeln auf dem Vollbestand.

    Wichtig:
    coverage_count ist jetzt nicht pauschal len(df),
    sondern die Zahl der Zeilen, auf die die Regel überhaupt anwendbar ist.
    """
    validation_results = []
    violation_examples = {}

    total_rows = len(df)

    for rule in rules:
        error = ""
        try:
            applicable_mask, valid_mask, invalid_mask = evaluate_rule_on_dataframe(df, rule)
            applicable_count = int(applicable_mask.sum())
            valid_count = int(valid_mask.sum())
            invalid_count = int(invalid_mask.sum())
            coverage_rate = applicable_count / total_rows if total_rows else 0.0
            # Ohne Anwendungsfälle ist die Qualität unbekannt, nicht fehlerfrei.
            violation_rate = invalid_count / applicable_count if applicable_count else None
            pass_rate = valid_count / applicable_count if applicable_count else None
            evaluation_status = "evaluated" if applicable_count else "not_applicable"
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
            evaluation_status = "invalid_rule"
            applicable_count = valid_count = invalid_count = None
            coverage_rate = violation_rate = pass_rate = None
            invalid_mask = pd.Series(False, index=df.index, dtype=bool)

        result = {
            "rule_id": rule["rule_id"],
            "rule_name": rule["rule_name"],
            "rule_type": rule["rule_type"],
            "columns": ",".join(rule["columns"]),
            "expression": rule["expression"],
            "source": rule["source"],
            "status": rule["status"],
            "evaluation_status": evaluation_status,
            "evaluation_error": error,
            "population_count": total_rows,
            "coverage_count": applicable_count,
            "coverage_rate": coverage_rate,
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "violation_rate": violation_rate,
            "pass_rate": pass_rate,
            "origin_runs": ",".join(str(x) for x in rule.get("origin_runs", [])),
            "reason": rule.get("reason", ""),
            "evidence_type": rule.get("evidence_type", ""),
            "evidence_detail": rule.get("evidence_detail", ""),
        }
        validation_results.append(result)

        # Beispiele müssen bis zum Ursprungsdatensatz rückverfolgbar bleiben.
        condition_cols = []
        if not error:
            condition_cols = [normalize_condition(c)["feature"] for c in rule.get("metadata", {}).get("conditions", [])]
        example_cols = [c for c in dict.fromkeys(
            ["record_key", "delivery_id"] + rule["columns"] + condition_cols
        ) if c in df.columns]
        example_df = df.loc[invalid_mask, example_cols].head(max_example_violations).copy()
        violation_examples[rule["rule_id"]] = example_df.to_dict(orient="records")

    return validation_results, violation_examples


def split_rules_by_source(rules: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Trennt Regeln nach ihrer Herkunft:
    - profiling
    - domain
    - ml_phase2
    """
    profiling_rules = [r for r in rules if r.get("source") == "profiling"]
    domain_rules = [r for r in rules if r.get("source") == "domain"]
    ml_rules = [r for r in rules if r.get("source") == "ml_phase2"]
    return profiling_rules, domain_rules, ml_rules
