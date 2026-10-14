from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import numpy as np

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


DOP_DIR = Path(
    r"C:\Users\Hendrik\Desktop\Studium\Projekte\Masterarbeit\geo-quality-framework\data\training\DOP"
)

OUTPUT_PARQUET = Path(
    r"C:\Users\Hendrik\Desktop\Studium\Projekte\Masterarbeit\validierungs-framework\data\parquetdata\dop_training_dataset.parquet"
)


def clean_tag(tag: str) -> str:
    return tag.split("}")[-1].strip()


def parse_xml_flat(xml_path: Path) -> dict:
    result = {}

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for elem in root.iter():
            if elem is root:
                continue

            tag = clean_tag(elem.tag)
            text = elem.text.strip() if elem.text else None

            if text not in (None, ""):
                result[tag] = text

        result["xml_parse_error"] = None

    except Exception as e:
        result["xml_parse_error"] = str(e)

    return result


def parse_tif_metadata(tif_path: Path) -> dict:
    result = {
        "tif_exists": tif_path.exists(),
        "tif_name": tif_path.name if tif_path.exists() else None,
        "tif_path": str(tif_path) if tif_path.exists() else None,
        "tif_size_bytes": tif_path.stat().st_size if tif_path.exists() else None,
    }

    if not tif_path.exists():
        return result

    if not RASTERIO_AVAILABLE:
        result["tif_read_error"] = "rasterio_not_installed"
        return result

    try:
        with rasterio.open(tif_path) as src:
            result.update({
                "tif_width": src.width,
                "tif_height": src.height,
                "tif_band_count": src.count,
                "tif_dtype": str(src.dtypes[0]) if src.dtypes else None,
                "tif_crs": str(src.crs),
                "tif_pixel_size_x": float(src.transform.a),
                "tif_pixel_size_y": float(abs(src.transform.e)),
                "tif_bounds_left": float(src.bounds.left),
                "tif_bounds_bottom": float(src.bounds.bottom),
                "tif_bounds_right": float(src.bounds.right),
                "tif_bounds_top": float(src.bounds.top),
                "tif_read_error": None,
            })

    except Exception as e:
        result["tif_read_error"] = str(e)

    return result


def safe_number(value):
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    try:
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        return float(text.replace(",", "."))
    except ValueError:
        return value


def main():
    rows = []

    for xml_path in sorted(DOP_DIR.glob("*.xml")):
        xml_data = parse_xml_flat(xml_path)

        kachelname = xml_data.get("kachelname", xml_path.stem)

        tif_path = DOP_DIR / f"{kachelname}.tif"

        if not tif_path.exists():
            tif_path = DOP_DIR / f"{xml_path.stem}.tif"

        row = {
            "source_folder": str(DOP_DIR),
            "xml_name": xml_path.name,
            "xml_path": str(xml_path),
            "xml_size_bytes": xml_path.stat().st_size,
            "kachelname_from_filename": xml_path.stem,
        }

        row.update(xml_data)
        row.update(parse_tif_metadata(tif_path))

        rows.append(row)

    df = pd.DataFrame(rows)

    for col in df.columns:
        df[col] = df[col].map(safe_number)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print("DOP-Dataset erstellt.")
    print(f"Zeilen:  {len(df)}")
    print(f"Spalten: {len(df.columns)}")
    print(f"Parquet: {OUTPUT_PARQUET}")

    print("\nSpalten:")
    for col in df.columns:
        print(f"  - {col}")


if __name__ == "__main__":
    main()