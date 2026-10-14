from __future__ import annotations

import json
from typing import Any

import pandas as pd
from openai import OpenAI


def build_schema_summary_for_llm(
    profile_summary: dict,
    excluded_columns: list[str] | None = None,
    max_columns: int | None = None,
) -> list[dict[str, Any]]:
    """
    Baut aus deinem Phase-1-Profiling eine kompakte Schema-Zusammenfassung
    für das LLM.

    WICHTIG:
    bewusst keine kompletten Datenwerte oder Rohzeilen weiter,
    sondern nur:
    - Spaltenname
    - Typ
    - Nullrate
    - kleine Profilzusammenfassung

    reduziert Rauschen und hält Schritt 1 wirklich auf Schema-Ebene.
    """
    excluded = set(excluded_columns or [])
    columns_meta = []

    for col, info in profile_summary.get("columns", {}).items():
        if col in excluded:
            continue

        entry = {
            "name": col,
            "dtype_group": info.get("dtype_group"),
            "null_rate_min": info.get("null_rate_min"),
            "null_rate_max": info.get("null_rate_max"),
        }
        columns_meta.append(entry)

    # Optional nur die ersten N Spalten verwenden
    if max_columns is not None:
        columns_meta = columns_meta[:max_columns]

    return columns_meta


def _build_prompt(columns_meta: list[dict[str, Any]]) -> str:
    """
    Baut den Prompt für Schritt 1:
    Das LLM soll nur plausible Abhängigkeiten vorschlagen,
    noch keine finalen Regeln erzeugen.
    """
    schema_json = json.dumps(columns_meta, ensure_ascii=False, indent=2)

    return f"""
Du unterstützt bei Schritt 1 der ML-basierten Regelableitung.

Aufgabe:
Schlage plausible Abhängigkeiten zwischen Spalten vor.
WICHTIG:
- Verwende ausschließlich die übergebenen echten Spaltennamen.
- Erfinde keine zusätzlichen Spalten.
- Liefere nur Vorschläge für Abhängigkeiten, keine finalen Datenregeln.
- Bevorzuge kompakte, fachlich plausible Zusammenhänge.
- Vermeide triviale Selbstbezüge.
- Vermeide rein technische oder interne Hilfsspalten.
- Gib eine kurze Begründung je Vorschlag.
- Kennzeichne die geschätzte Plausibilität grob als high / medium / low.

Erwartete Form pro Vorschlag:
- target_column
- feature_columns
- rationale
- confidence

Hier ist das Schema:
{schema_json}
""".strip()


def suggest_dependencies_with_openai(
    profile_summary: dict,
    api_key: str,
    model: str = "gpt-4.1-mini",
    excluded_columns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Ruft ein LLM auf und lässt strukturierte Abhängigkeitsvorschläge erzeugen.

    HINWEIS:
    Diese Funktion nutzt Structured Outputs über ein JSON-Schema,
    damit die Antwort maschinenlesbar bleibt.
    """
    client = OpenAI(api_key=api_key)

    columns_meta = build_schema_summary_for_llm(
        profile_summary=profile_summary,
        excluded_columns=excluded_columns,
    )

    prompt = _build_prompt(columns_meta)

    # JSON-Schema für strukturierte Ausgabe
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Du bist ein Assistent für Schema-Grounding und "
                            "Feature-Dependency-Vorschläge in Geodaten-Metadaten."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": prompt,
                    }
                ],
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "dependency_suggestions",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "suggestions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "target_column": {"type": "string"},
                                    "feature_columns": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "rationale": {"type": "string"},
                                    "confidence": {
                                        "type": "string",
                                        "enum": ["high", "medium", "low"],
                                    },
                                },
                                "required": [
                                    "target_column",
                                    "feature_columns",
                                    "rationale",
                                    "confidence",
                                ],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["suggestions"],
                    "additionalProperties": False,
                },
            }
        },
    )

    # In den Responses-Ausgaben steckt der finale Text als JSON-String
    raw_text = response.output_text
    parsed = json.loads(raw_text)

    # Zusätzliche harte Schema-Prüfung:
    # Nur Vorschläge behalten, die echte Spalten referenzieren.
    valid_column_names = {c["name"] for c in columns_meta}

    cleaned_suggestions = []
    for s in parsed.get("suggestions", []):
        target = s["target_column"]
        features = s["feature_columns"]

        if target not in valid_column_names:
            continue
        if any(f not in valid_column_names for f in features):
            continue
        if target in features:
            continue

        cleaned_suggestions.append(s)

    return {
        "model": model,
        "input_column_count": len(columns_meta),
        "suggestions": cleaned_suggestions,
    }


def filter_dependency_suggestions(
    suggestions: list[dict[str, Any]],
    allowed_confidence: tuple[str, ...] = ("high", "medium"),
    max_feature_count: int = 4,
) -> list[dict[str, Any]]:
    """
    Filtert die LLM-Vorschläge nach einfachen operativen Kriterien.

    Ziel:
    - nur ausreichend plausible Vorschläge behalten
    - keine zu großen Feature-Sets zulassen
    """
    filtered = []

    for s in suggestions:
        if s.get("confidence") not in allowed_confidence:
            continue
        if len(s.get("feature_columns", [])) > max_feature_count:
            continue
        filtered.append(s)

    return filtered