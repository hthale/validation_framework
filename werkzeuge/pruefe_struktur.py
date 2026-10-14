"""Prueft kopierten Code und Imports, optional die betroffenen vorhandenen Tests."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


BOOTSTRAPS = [
    ("experimente/dop_pruefung", "run_joint_dop_check", ["."]),
    ("adapters/alkis_bahnverkehr", "start_bahnverkehr", ["."]),
    ("experimente/clusterregeln", "run_rule_comparison", ["src", "adapters/alkis_bahnverkehr"]),
    ("experimente/mehrheitsvergleich", "run_baseline_check", ["src", "adapters/alkis_bahnverkehr", "experimente/clusterregeln"]),
    ("experimente/synthetischer_vergleich", "run_controlled_comparison", ["src", "adapters/alkis_bahnverkehr", "experimente/clusterregeln", "experimente/mehrheitsvergleich"]),
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--tests", action="store_true", help="Vorhandene Tests zu DOP, ALKIS und den drei Vergleichen ausfuehren.")
    args = parser.parse_args()
    project = args.project.resolve(strict=True)
    manifest = json.loads((project / "docs/ordnungsprotokoll.json").read_text(encoding="utf-8"))
    status = json.loads((project / "ordnungsstatus.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise ValueError("Die Kopie ist nicht vollstaendig abgeschlossen.")
    checked = 0
    # data wurde beim Kopieren vollstaendig geprueft; hier wird kein Versuch wiederholt.
    for item in manifest["files"]:
        if item["role"] == "data":
            continue
        path = (project / item["target"]).resolve()
        if not path.is_relative_to(project) or not path.is_file() or sha(path) != item["target_sha256"]:
            raise ValueError("Datei fehlt oder hat sich seit dem Ordnen geaendert: " + item["target"])
        if path.suffix == ".py":
            ast.parse(path.read_bytes(), filename=str(path), feature_version=(3, 11))
        checked += 1
    for rel, expected in manifest["generated_sha256"].items():
        path = (project / rel).resolve()
        if not path.is_relative_to(project) or not path.is_file() or sha(path) != expected:
            raise ValueError("Erzeugte Datei fehlt oder hat sich geaendert: " + rel)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Bewusst getrennte Prozesse, damit alte Modulimports keine falschen Pfade verdecken.
    for folder, name, expected in BOOTSTRAPS:
        print(f"Importpfade: {folder}", flush=True)
        code = ("import importlib,json,pathlib,sys; "
                "project=pathlib.Path(sys.argv[1]); folder=project/sys.argv[2]; "
                "sys.path.insert(0,str(folder)); m=importlib.import_module(sys.argv[3]); "
                "assert pathlib.Path(m.__file__).resolve().parent==folder.resolve(); "
                "result=m.bootstrap(project); result=result if isinstance(result,tuple) else (result,); "
                "expected=[(project/p).resolve() for p in json.loads(sys.argv[4])]; "
                "assert [pathlib.Path(p).resolve() for p in result]==expected")
        subprocess.run([sys.executable, "-c", code, str(project), folder, name, json.dumps(expected)],
                       cwd=project, env=env, check=True)
    if args.tests:
        # Keine Umgebungsvariable darf versehentlich noch auf das alte Projekt zeigen.
        env["FRAMEWORK_PROJECT"] = str(project)
        env["FRAMEWORK_TEST_PROJECT"] = str(project)
        for folder, _, _ in BOOTSTRAPS:
            print(f"Vorhandene Tests: {folder}", flush=True)
            subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", f"{folder}/tests", "-v"],
                           cwd=project, env=env, check=True)
    print(json.dumps({"structure_check_passed": True, "files_checked": checked,
                      "bootstrap_chains_checked": len(BOOTSTRAPS), "existing_test_suites_run": 5 if args.tests else 0,
                      "autoencoder_retrained": False, "historical_runs_revalidated": False}, indent=2))


if __name__ == "__main__":
    main()
