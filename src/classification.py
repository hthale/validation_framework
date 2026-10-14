import pandas as pd


def classify_rule(row: pd.Series) -> str:
    """
    Klassifiziert Regeln für eine operative Review-Priorisierung.
    WICHTIG:
    Das ist bewusst KEIN statistisch kalibrierter Confidence-Score,
    sondern eine heuristische Arbeitsklassifikation.

    Logik:
    - fachlich starke Domain-/Cross-Column-Regeln ohne Verstöße -> strong
    - einfache Basisregeln ohne Verstöße -> basic
    - Range-/Regex-/Length-Regeln ohne Verstöße -> good
    - kleine Verstöße -> review
    - mittlere Verstöße -> weak
    - hohe Verstöße -> reject
    """
    if row.get("evaluation_status") == "invalid_rule":
        return "invalid_rule"
    if row.get("coverage_count", 0) == 0 or pd.isna(row["violation_rate"]):
        return "not_evaluated"
    rule_type = row["rule_type"]
    violation_rate = float(row["violation_rate"])
    source = row["source"]

    if violation_rate == 0:
        if source == "domain" or rule_type in {"cross_column_map", "cross_column_consistency"}:
            return "candidate_strong"

        if rule_type in {
            "not_null",
            "constant",
            "constant_category",
            "in_set",
            "in_set_numeric",
        }:
            return "candidate_basic"

        if rule_type in {"range", "regex_match", "length_range"}:
            return "candidate_good"

        return "candidate_good"

    if violation_rate <= 0.0001:
        return "candidate_review"

    if violation_rate <= 0.001:
        return "candidate_review"

    if violation_rate <= 0.01:
        return "candidate_weak"

    return "candidate_reject"


def compute_priority_score(row: pd.Series) -> float:
    """
    Berechnet einen operativen Prioritätsscore.

    Höher = zuerst prüfen.
    Der Score ist eine Heuristik und dient nur der Sortierung.
    """
    if row["rule_class"] in {"invalid_rule", "not_evaluated"}:
        return 0.0
    violation_rate = float(row["violation_rate"])
    invalid_count = int(row["invalid_count"])
    source = row["source"]
    rule_type = row["rule_type"]
    rule_class = row["rule_class"]

    score = 100.0

    # Verstöße bestrafen
    score -= violation_rate * 10000

    # Absolute Fehler leicht bestrafen
    score -= min(invalid_count, 1000) * 0.01

    # Domain-Regeln priorisieren
    if source == "domain":
        score += 10

    # Cross-Column-Regeln priorisieren
    if rule_type in {"cross_column_map", "cross_column_consistency"}:
        score += 10

    # Basisregeln etwas niedriger priorisieren
    if rule_class == "candidate_basic":
        score -= 5

    return round(score, 4)


def recommend_action(row: pd.Series) -> str:
    """
    Leitet aus der Heuristik eine Review-Empfehlung ab.
    """
    classification = row["rule_class"]

    if classification == "invalid_rule":
        return "repair_rule"
    if classification == "not_evaluated":
        return "collect_evidence"

    if classification == "candidate_strong":
        return "review_now"

    if classification == "candidate_good":
        return "review_now"

    if classification == "candidate_basic":
        return "batch_review"

    if classification == "candidate_review":
        return "review_later"

    if classification == "candidate_weak":
        return "manual_check"

    return "reject_or_rework"


def classify_and_prioritize_rules(validation_results: list[dict]) -> list[dict]:
    """
    Klassifiziert und sortiert Regeln.
    """
    df = pd.DataFrame(validation_results).copy()

    if df.empty:
        return []

    df["rule_class"] = df.apply(classify_rule, axis=1)
    df["priority_score"] = df.apply(compute_priority_score, axis=1)
    df["recommended_action"] = df.apply(recommend_action, axis=1)

    df = df.sort_values(
        by=["priority_score", "violation_rate", "invalid_count", "rule_id"],
        ascending=[False, True, True, True]
    ).reset_index(drop=True)

    # pandas macht aus None sonst NaN; als JSON bleibt die fehlende Evidenz null.
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


def summarize_rule_classes(classified_rules: list[dict]) -> dict:
    """
    Erstellt eine kompakte Zusammenfassung der Klassenverteilung.
    """
    df = pd.DataFrame(classified_rules)

    if df.empty:
        return {
            "total_rules": 0,
            "by_class": {},
            "by_action": {},
            "by_source": {},
        }

    return {
        "total_rules": int(len(df)),
        "by_class": df["rule_class"].value_counts().to_dict(),
        "by_action": df["recommended_action"].value_counts().to_dict(),
        "by_source": df["source"].value_counts().to_dict(),
    }
