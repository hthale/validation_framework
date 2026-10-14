"""Lokale, getrennte Versuchsartefakte; keine Änderung eines früheren Laufs."""
from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform

from review_common import atomic_text, file_sha256, write_json


def new_output(root):
    out = Path(root).resolve() / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "run_status.json", {"status": "running"})
    return out


def finish_output(out, provenance):
    src = Path(__file__).resolve().parent
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "torch", "pyarrow", "hdbscan", "umap-learn"):
        try: versions[name] = version(name)
        except PackageNotFoundError: versions[name] = None
    write_json(out / "run_manifest.json", {
        **provenance, "python": platform.python_version(), "versions": versions,
        "source_sha256": {p.relative_to(src).as_posix(): file_sha256(p) for p in sorted(src.rglob("*.py"))},
        "output_sha256": {p.relative_to(out).as_posix(): file_sha256(p)
                          for p in sorted(out.rglob("*")) if p.is_file() and p.name != "run_status.json"},
    })
    write_json(out / "run_status.json", {"status": "completed"})


def read_verified_run(run, names):
    run = Path(run).resolve(strict=True)
    state = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
    if state.get("status") != "completed":
        raise ValueError("Nur einen vollständig abgeschlossenen generischen Lauf verwenden.")
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    hashes = {}
    for name in names:
        p = run / name
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(run):
            raise ValueError(f"Ergebnisdatei fehlt oder verweist außerhalb des Laufs: {name}")
        actual = file_sha256(p)
        if manifest.get("output_sha256", {}).get(name) != actual:
            raise ValueError(f"Ergebnisdatei passt nicht zum gespeicherten Manifest: {name}")
        hashes[name] = actual
    return hashes


def html_report(path, title, content):
    atomic_text(path, '<!doctype html><html lang="de"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>' + escape(title) + '</title>'
        '<style>body{font-family:system-ui,sans-serif;background:#f3f6f7;color:#17323b;margin:0}'
        'header{background:#153e44;color:white;padding:28px 5vw}main{max-width:1220px;margin:auto;padding:26px}'
        'article,.notice{background:white;padding:18px;border:1px solid #d6e0e3;border-radius:8px;margin:16px 0}'
        'table{border-collapse:collapse;background:white;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #d6e0e3}'
        '.scroll{overflow:auto}p{line-height:1.6}pre{white-space:pre-wrap;word-break:break-word;max-height:450px;overflow:auto}'
        'h2{margin-top:32px}h3{font-size:16px;overflow-wrap:anywhere}summary{cursor:pointer}a{color:#106376}</style>'
        '<header><h1>' + escape(title) + '</h1></header><main>' + content + '</main></html>')


def details_json(value, label="Details"):
    return ('<details><summary>' + escape(label) + '</summary><pre>' +
            escape(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)) + '</pre></details>')
