import re


def classify_token(token: str) -> str:
    """
    Klassifiziert einzelne Tokens eines Dateinamens.
    """
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", token):
        return r"\d{4}-\d{2}-\d{2}"

    if re.fullmatch(r"\d+", token):
        return r"\d+"

    if re.fullmatch(r"[a-zA-Z]+", token):
        return re.escape(token)

    return re.escape(token)


def infer_regex_from_names(names: list[str]) -> str:
    """
    Leitet aus mehreren Datensatznamen einen Regex ab.
    Annahme: Namen sind mit '_' segmentiert.
    """
    if not names:
        return r"^.*$"

    split_names = [name.split("_") for name in names]
    token_lengths = {len(tokens) for tokens in split_names}

    # Falls die Namen unterschiedlich viele Segmente haben,
    # wird zur Sicherheit eine Alternation der konkreten Namen erzeugt.
    if len(token_lengths) != 1:
        escaped_names = [re.escape(name) for name in sorted(set(names))]
        return r"^(" + "|".join(escaped_names) + r")$"

    token_count = token_lengths.pop()
    regex_parts = []

    for i in range(token_count):
        tokens_at_pos = [tokens[i] for tokens in split_names]
        unique_tokens = set(tokens_at_pos)

        # Alle Tokens an dieser Position gleich -> fester String
        if len(unique_tokens) == 1:
            regex_parts.append(re.escape(tokens_at_pos[0]))
            continue

        classified_tokens = {classify_token(token) for token in tokens_at_pos}

        # Alle Tokens unterscheiden sich zwar, haben aber dieselbe Klasse
        if len(classified_tokens) == 1:
            regex_parts.append(classified_tokens.pop())
        else:
            # Unterschiedliche feste Werte -> Alternation
            alternatives = sorted(re.escape(token) for token in unique_tokens)
            regex_parts.append("(" + "|".join(alternatives) + ")")

    return "^" + "_".join(regex_parts) + "$"


def generate_regex_rule_candidates(
    products: list[dict],
    min_examples: int = 2,
    max_examples: int = 5
) -> list[dict]:
    """
    Erzeugt pro Produkt Regex-Regelvorschläge auf Basis der dataset_name-Werte.
    """
    regex_rules = []

    for product in products:
        product_name = product["product_name"]

        dataset_names = sorted(
            set(
                dataset["dataset_name"]
                for dataset in product["datasets"]
                if dataset["dataset_type"] != "unknown"
                and dataset["dataset_name"] != "unassigned_files"
            )
        )

        if len(dataset_names) < min_examples:
            continue

        regex_pattern = infer_regex_from_names(dataset_names)

        regex_rules.append({
            "rule_id": f"{product_name}_dataset_name_regex",
            "product_name": product_name,
            "field": "dataset_name",
            "rule_type": "regex",
            "value": regex_pattern,
            "example_names": dataset_names[:max_examples],
            "example_count": len(dataset_names),
            "confidence": 1.0,
            "approved": False
        })

    return regex_rules