from pathlib import Path
from collections import defaultdict
import json
import re

# Externe Komponente: erzeugt Regex-Regeln aus bereits erkannten Datensätzen
from regex_generator import generate_regex_rule_candidates


def scan_products(base_folder: Path) -> list[dict]:
    """
    Durchläuft die oberste Datenstruktur:
    Jeder Unterordner in /data wird als eigenes 'Produkt' interpretiert.
    """
    products = []

    # Jeder direkte Unterordner entspricht einem Datenprodukt (z. B. dop, fg222, ...)
    for product_folder in sorted(p for p in base_folder.iterdir() if p.is_dir()):
        datasets = scan_product_directory(product_folder, base_folder)

        products.append({
            "product_name": product_folder.name,
            "product_path": str(product_folder.relative_to(base_folder)),
            "dataset_count": len(datasets),
            "datasets": datasets
        })

    return products


def scan_product_directory(product_folder: Path, base_folder: Path) -> list[dict]:
    """
    Erkennt innerhalb eines Produktordners einzelne Datensätze.
    Ein Datensatz = Gruppe von Dateien mit gleichem Basisnamen (stem).
    Beispiel:
        dop_... .tif + dop_... .xml -> ein Datensatz
    """
    # Alle Dateien rekursiv einsammeln
    all_files = [p for p in product_folder.rglob("*") if p.is_file()]

    # Gruppierung nach Basisnamen (ohne Endung)
    files_by_stem = defaultdict(list)
    for file_path in all_files:
        files_by_stem[file_path.stem].append(file_path)

    datasets = []
    assigned_files = set()

    # Jede Gruppe von Dateien wird als potenzieller Datensatz betrachtet
    for stem, file_list in sorted(files_by_stem.items()):
        suffixes = {f.suffix.lower() for f in file_list}
        relative_files = sorted(str(f.relative_to(base_folder)) for f in file_list)

        # Heuristik zur Bestimmung des Datensatztyps anhand der vorhandenen Dateien
        if ".tif" in suffixes and ".xml" in suffixes:
            dataset_type = "raster_with_xml"
        elif ".shp" in suffixes and ".dbf" in suffixes and ".shx" in suffixes:
            dataset_type = "shapefile_dataset"
        elif ".geojson" in suffixes:
            dataset_type = "geojson_dataset"
        elif ".json" in suffixes:
            dataset_type = "json_dataset"
        elif ".tif" in suffixes:
            dataset_type = "raster_dataset"
        else:
            # Unbekannte Kombinationen werden ignoriert
            continue

        dataset = {
            "dataset_name": stem,  # zentral für Regex-Lernen
            "dataset_type": dataset_type,
            "product_name": product_folder.name,
            "files": relative_files,
            "profile": build_basic_profile(base_folder, file_list, dataset_type),
            "rule_candidates": []  # aktuell noch ungenutzt (Erweiterungspunkt)
        }

        datasets.append(dataset)

        # Merken, welche Dateien bereits einem Datensatz zugeordnet wurden
        assigned_files.update(str(f.resolve()) for f in file_list)

    # Restliche Dateien, die keinem Datensatz zugeordnet werden konnten
    unassigned_files = sorted(
        str(f.relative_to(base_folder))
        for f in all_files
        if str(f.resolve()) not in assigned_files
    )

    if unassigned_files:
        datasets.append({
            "dataset_name": "unassigned_files",
            "dataset_type": "unknown",
            "product_name": product_folder.name,
            "files": unassigned_files,
            "profile": {
                "file_count": len(unassigned_files)
            },
            "rule_candidates": []
        })

    return datasets


def build_basic_profile(base_folder: Path, file_list: list[Path], dataset_type: str) -> dict:
    """
    Erstellt einfache, generische Metadaten (Features) pro Datensatz.
    Diese sind die Grundlage für spätere Regelableitung.
    """
    extensions = sorted({f.suffix.lower() for f in file_list})

    # Datei-Größen können später für Ausreißer-Erkennung genutzt werden
    file_sizes = {
        str(f.relative_to(base_folder)): f.stat().st_size
        for f in file_list
    }

    profile = {
        "file_count": len(file_list),
        "extensions": extensions,
        "file_sizes_bytes": file_sizes,
        "dataset_type": dataset_type
    }

    # Datensatztyp-spezifische Checks
    if dataset_type == "raster_with_xml":
        profile["required_extensions_present"] = all(
            ext in extensions for ext in [".tif", ".xml"]
        )

    elif dataset_type == "shapefile_dataset":
        required = [".shp", ".dbf", ".shx"]
        optional = [".prj", ".cpg", ".qix"]

        profile["required_extensions_present"] = all(ext in extensions for ext in required)
        profile["optional_extensions_present"] = [ext for ext in optional if ext in extensions]

    elif dataset_type in {"geojson_dataset", "json_dataset", "raster_dataset"}:
        profile["required_extensions_present"] = True

    # Erste einfache Mustererkennung (noch heuristisch, nicht regelbasiert)
    profile["filename_pattern_guess"] = guess_filename_pattern(file_list)

    return profile


def guess_filename_pattern(file_list: list[Path]) -> str:
    """
    Sehr einfache Generalisierung eines Dateinamens:
    - Zahlen -> <NUM>
    - Datum -> <DATE>
    Nur als erste Näherung (nicht für echte Validierung geeignet).
    """
    names = [f.stem for f in file_list]

    if not names:
        return "unknown"

    name = names[0]

    name = re.sub(r"\d{4}-\d{2}-\d{2}", "<DATE>", name)
    name = re.sub(r"\d+", "<NUM>", name)

    return name


def generate_basic_rule_candidates(products: list[dict]) -> list[dict]:
    """
    Erstellt einfache, deterministische Regeln auf Datensatz-Ebene.
    (noch NICHT aggregiert auf Produktebene)
    """
    rule_candidates = []

    for product in products:
        for dataset in product["datasets"]:
            if dataset["dataset_type"] == "unknown":
                continue

            profile = dataset["profile"]

            # Feste Regel: Anzahl Dateien im Datensatz
            rule_candidates.append({
                "rule_id": f"{product['product_name']}_{dataset['dataset_name']}_file_count",
                "product_name": product["product_name"],
                "dataset_name": dataset["dataset_name"],
                "field": "file_count",
                "rule_type": "equals",
                "value": profile["file_count"],
                "confidence": 1.0,
                "approved": False
            })

            # Feste Regel: benötigte Dateiendungen vorhanden
            rule_candidates.append({
                "rule_id": f"{product['product_name']}_{dataset['dataset_name']}_extensions",
                "product_name": product["product_name"],
                "dataset_name": dataset["dataset_name"],
                "field": "extensions",
                "rule_type": "contains_all",
                "value": profile["extensions"],
                "confidence": 1.0,
                "approved": False
            })

    return rule_candidates


def print_summary(products: list[dict]) -> None:
    """
    Debug-/Analyse-Ausgabe: zeigt erkannte Struktur.
    """
    print("\nErkannte Produkte und Datensätze:\n")

    for product in products:
        print(f"Produkt: {product['product_name']}")
        print(f"Pfad:    {product['product_path']}")
        print(f"Anzahl Datensätze: {product['dataset_count']}")

        for dataset in product["datasets"]:
            print(f"  - Name: {dataset['dataset_name']}")
            print(f"    Typ:  {dataset['dataset_type']}")
            print("    Dateien:")
            for file in dataset["files"]:
                print(f"      - {file}")
            print(f"    Profil: {dataset['profile']}")
        print()


def print_regex_summary(regex_rules: list[dict]) -> None:
    """
    Ausgabe der automatisch gelernten Regex-Regeln pro Produkt.
    """
    print("\nRegex-Regelvorschläge pro Produkt:\n")

    for rule in regex_rules:
        print(f"Produkt: {rule['product_name']}")
        print(f"Regex:   {rule['value']}")
        print(f"Anzahl Beispiele: {rule['example_count']}")
        print("Beispielnamen:")
        for name in rule["example_names"]:
            print(f"  - {name}")
        print()


def main() -> None:
    """
    Zentrale Pipeline:
    1. Daten scannen
    2. Profile erzeugen
    3. einfache Regeln generieren
    4. Regex-Regeln lernen (produktweit)
    5. Ergebnisse speichern
    """
    base_folder = Path(
        r"C:\Users\Hendrik\Desktop\Studium\Projekte\Masterarbeit\validierungs-framework\data"
    )

    output_folder = base_folder.parent / "outputs"
    output_folder.mkdir(exist_ok=True)

    # 1. Struktur erkennen
    products = scan_products(base_folder)

    # 2. einfache (harte) Regeln pro Datensatz
    basic_rule_candidates = generate_basic_rule_candidates(products)

    # 3. automatische Regex-Regeln pro Produkt (aus vielen Beispielen gelernt)
    regex_rule_candidates = generate_regex_rule_candidates(
        products,
        min_examples=2,
        max_examples=5
    )

    # Debug-Ausgaben
    print_summary(products)
    print_regex_summary(regex_rule_candidates)

    # Speicherung der Ergebnisse (wichtig für spätere Pipeline-Schritte)
    output_profiles = output_folder / "detected_products_with_profiles.json"
    with open(output_profiles, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2, ensure_ascii=False)

    output_rules = output_folder / "rule_candidates.json"
    with open(output_rules, "w", encoding="utf-8") as f:
        json.dump(basic_rule_candidates, f, indent=2, ensure_ascii=False)

    output_regex_rules = output_folder / "regex_rule_candidates.json"
    with open(output_regex_rules, "w", encoding="utf-8") as f:
        json.dump(regex_rule_candidates, f, indent=2, ensure_ascii=False)

    print(f"Profile gespeichert unter: {output_profiles}")
    print(f"Basis-Regelvorschläge gespeichert unter: {output_rules}")
    print(f"Regex-Regelvorschläge gespeichert unter: {output_regex_rules}")


if __name__ == "__main__":
    main()