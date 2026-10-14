"""Künstlicher DOP-Kontrollbestand mit bekannten Abhängigkeiten; keine echten Produktanforderungen."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def make_fixture():
    rows = []
    for group in range(200):
        mode, detail = group % 2, (group // 2) % 2
        camera, spectral, bands = ("Kamera_A", "RGB", 3) if mode == 0 else ("Kamera_B", "RGBI", 4)
        cm = 10 if detail else 20
        for repeat in range(2):
            rows.append({"record_key": f"fixture09:{group:04d}:{repeat}", "koordinatenreferenzsystem_lage": 25832,
                "koordinatenursprung_east": 300000 + group*2000, "koordinatenursprung_north": 5800000,
                "spektralkanaele": spectral, "tif_band_count": bands, "kamera_sensor": camera,
                "detailstufe": "fein" if detail else "standard", "bodenpixelgroesse": cm,
                "tif_pixel_size_x": cm/100, "tif_pixel_size_y": -cm/100, "tif_crs": "EPSG:25832",
                "anzahl_spalten": 10000, "anzahl_zeilen": 10000, "tif_width": 10000, "tif_height": 10000,
                "aktualitaet": (pd.Timestamp("2024-01-01")+pd.Timedelta(days=group)).date().isoformat(),
                "xml_size_bytes": 1100 + int(np.random.default_rng(group).integers(0, 400)),
                "kontrolle": "K_A" if repeat == 0 else "K_B"})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/dop_control_fixture_09.parquet"))
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Existierende Dateien nicht ersetzen; Erstellung über exklusiven Dateihandle.
    with args.output.open("xb") as stream:
        make_fixture().to_parquet(stream, index=False)
    print(f"400 künstliche Zeilen, 200 Gruppen: {args.output.resolve()}")
    print("Referenzen: RGB -> 3 Bänder/Kamera_A, RGBI -> 4 Bänder/Kamera_B; fein -> 10 cm, standard -> 20 cm.")
    print("kontrolle ist bei identischen Prädiktoren paarweise ausgeglichen und soll keine ausgewählte Abhängigkeit erzeugen.")


if __name__ == "__main__": main()
