"""Unbekanntes Kunstprodukt mit drei vorab festgelegten Zusammenhängen.

Die Annahmen dienen nur der technischen Funktionsprüfung, keiner Produktspezifikation.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def example():
    rows = []
    rng = np.random.default_rng(71)
    for unit in range(160):
        mode = unit % 2
        variant = (unit // 2) % 2
        reading = float(rng.normal())
        for repeat in range(2):
            rows.append({
                "lokal_id": f"probe-{unit:04d}-{repeat}", "entity_code": f"E-{unit:04d}",
                "modus": ["M", "N"][mode], "variante": ["grob", "fein"][variant],
                "anzahl": [2, 5][mode], "abstand": [12, 4][variant],
                "ablage": f"P-{mode}-{variant}", "kontrolle": ["K_A", "K_B"][repeat],
                "messwert": reading, "konstante": "METADATEN",
            })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("data/generic_example_07.parquet"))
    args = p.parse_args()
    if args.output.exists():
        p.error(f"Datei existiert bereits: {args.output}. Andere --output-Datei wählen.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    example().to_parquet(args.output, index=False)
    print(f"320 künstliche Zeilen: {args.output.resolve()}")
    print("Referenzen: M -> anzahl 2, N -> 5; grob -> abstand 12, fein -> 4; ablage hängt von beiden ab.")
    print("kontrolle ist bei identischen Prädiktoren paarweise ausgeglichen und soll keine ausgewählte Abhängigkeit erzeugen.")


if __name__ == "__main__":
    main()
