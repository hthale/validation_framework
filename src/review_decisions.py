"""Fachliche Entscheidungen prüfen, mit Revision protokollieren und Ansichten erneuern."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

from review_common import digest, read_json, utc_now, verify_bundle, write_json

DECISIONS = {"unreviewed", "confirmed_error", "plausible", "needs_clarification"}


def initial_history(bundle):
    return {"schema_version": 1, "bundle_id": bundle["bundle_id"], "revision": 0, "events": []}


def _validate_changes(changes, keys):
    if not isinstance(changes, list) or not changes:
        raise ValueError("Mindestens eine fachliche Entscheidung erforderlich.")
    seen = set()
    for item in changes:
        if not isinstance(item, dict) or set(item) != {"record_key", "decision", "reason"}:
            raise ValueError("Eine Entscheidung enthält genau record_key, decision und reason.")
        key = item["record_key"]
        if not isinstance(key, str) or key not in keys or key in seen:
            raise ValueError("Unbekannter oder doppelter record_key in Entscheidungen.")
        seen.add(key)
        if not isinstance(item["decision"], str) or item["decision"] not in DECISIONS:
            raise ValueError("Unbekannte fachliche Entscheidung.")
        if not isinstance(item["reason"], str) or not item["reason"].strip():
            raise ValueError("Jede Entscheidung, auch das Zurücksetzen, benötigt eine Begründung.")
        if len(item["reason"]) > 10000:
            raise ValueError("Begründung überschreitet 10.000 Zeichen.")


def replay_history(bundle, history):
    verify_bundle(bundle)
    if history.get("schema_version") != 1 or history.get("bundle_id") != bundle["bundle_id"]:
        raise ValueError("Entscheidungsprotokoll gehört nicht zu diesem Prüflauf.")
    events = history.get("events")
    if not isinstance(events, list) or type(history.get("revision")) is not int or history["revision"] != len(events):
        raise ValueError("Ungültige Revision des Entscheidungsprotokolls.")
    keys = {r["record_key"] for r in bundle["cases"]}
    state, previous = {}, ""
    for revision, event in enumerate(events, 1):
        if event.get("revision") != revision or event.get("previous_event_hash") != previous:
            raise ValueError("Unterbrochene Revisionsfolge im Entscheidungsprotokoll.")
        if event.get("event_hash") != digest({k: v for k, v in event.items() if k != "event_hash"}):
            raise ValueError("Ein gespeicherter Entscheidungseintrag wurde verändert.")
        if not isinstance(event.get("reviewer"), str) or not event["reviewer"].strip():
            raise ValueError("Prüfername fehlt im Protokoll.")
        _validate_changes(event.get("decisions"), keys)
        for item in event["decisions"]:
            state[item["record_key"]] = {**item, "reviewer": event["reviewer"], "reviewed_at": event["recorded_at"], "revision": revision}
        previous = event["event_hash"]
    return state


def prepare_event(bundle, history, request):
    state = replay_history(bundle, history)
    if not isinstance(request, dict) or set(request) != {"schema_version", "bundle_id", "base_revision", "reviewer", "decisions"}:
        raise ValueError("Entscheidungsdatei hat nicht das erwartete Schema.")
    if request["schema_version"] != 1 or request["bundle_id"] != bundle["bundle_id"]:
        raise ValueError("Entscheidungsdatei gehört nicht zu diesem Prüflauf.")
    if type(request["base_revision"]) is not int or request["base_revision"] != history["revision"]:
        raise ValueError("Veraltete Bearbeitungsrevision. review.html neu öffnen und Änderungen anhand des aktuellen Stands übernehmen.")
    reviewer = request["reviewer"]
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("Prüfername ist erforderlich.")
    _validate_changes(request["decisions"], {r["record_key"] for r in bundle["cases"]})
    changes = []
    for item in request["decisions"]:
        previous = state.get(item["record_key"], {"decision": "unreviewed", "reason": ""})
        if item["decision"] != previous["decision"] or item["reason"].strip() != previous["reason"]:
            changes.append({**item, "reason": item["reason"].strip()})
    if not changes:
        raise ValueError("Keine neuen Entscheidungen oder Begründungen vorhanden.")
    event = {"revision": history["revision"] + 1, "recorded_at": utc_now(), "reviewer": reviewer.strip(),
             "previous_event_hash": history["events"][-1]["event_hash"] if history["events"] else "", "decisions": changes}
    event["event_hash"] = digest(event)
    return event


@contextmanager
def review_lock(directory):
    path = Path(directory) / ".review.lock"
    try:
        stream = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise ValueError("Ein anderer Schreibvorgang ist aktiv. Nach einem Abbruch die Sperrdatei erst nach Prüfung entfernen.") from exc
    try:
        with stream:
            stream.write(utc_now())
        yield
    finally:
        path.unlink()


def apply_decisions(directory, request, dry_run=False):
    directory = Path(directory)
    with review_lock(directory):
        bundle = verify_bundle(read_json(directory / "review_bundle.json"))
        history = read_json(directory / "review_history.json")
        event = prepare_event(bundle, history, request)
        if dry_run:
            return {"dry_run": True, "decision_count": len(event["decisions"]), "next_revision": event["revision"]}
        updated = {**history, "revision": event["revision"], "events": history["events"] + [event]}
        # Das Protokoll ist die maßgebliche Datei. Exportansichten lassen sich daraus wiederherstellen.
        write_json(directory / "review_history.json", updated)
        from review_output import write_views
        try:
            write_views(directory, bundle, updated)
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError("Entscheidungen sind protokolliert; Ansichten konnten nicht erneuert werden. Mit review_decisions.py render wiederherstellen.") from exc
        return {"dry_run": False, "decision_count": len(event["decisions"]), "revision": event["revision"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    apply = sub.add_parser("apply", help="Heruntergeladene fachliche Entscheidungen übernehmen.")
    apply.add_argument("--review-dir", type=Path, required=True)
    apply.add_argument("--decisions", type=Path, required=True)
    apply.add_argument("--dry-run", action="store_true")
    render = sub.add_parser("render", help="Ansichten aus dem vorhandenen Protokoll wiederherstellen.")
    render.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "apply":
            result = apply_decisions(args.review_dir, read_json(args.decisions), args.dry_run)
            print(result)
        else:
            from review_output import write_views
            with review_lock(args.review_dir):
                write_views(args.review_dir, read_json(args.review_dir / "review_bundle.json"), read_json(args.review_dir / "review_history.json"))
            print("Ansichten wiederhergestellt.")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Nicht übernommen: {exc}\n")


if __name__ == "__main__":
    main()
