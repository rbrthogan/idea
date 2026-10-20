#!/usr/bin/env python3
"""Read-only Firestore fingerprints and active-lease checks for a deployment.

Capture uses a server read timestamp and includes subcollections beneath missing
parent documents. Output contains counts and keyed hashes, never document values,
raw user IDs, credentials or run tokens. This evidence is not a backup.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import sys


ACTIVE_STATUSES = {"starting", "queued", "in_progress", "resuming", "continuing", "stopping"}
STATUSES = ACTIVE_STATUSES | {"paused", "complete", "error", "force_stopped", "stopped"}
KINDS = {"users", "settings", "evolutions", "chunks", "chunk_versions", "evolution_tombstones",
         "checkpoints", "templates", "billing_runs", "billing_events", "runs", "system"}


class AcceptanceError(Exception):
    pass


def canonical(value):
    """Retain Firestore value types without printing an unsupported value."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return {"$float": repr(value) if math.isfinite(value) else str(value)}
    if isinstance(value, datetime):
        formatter = getattr(value, "rfc3339", None)
        return {"$timestamp": formatter() if callable(formatter) else value.isoformat()}
    if isinstance(value, bytes):
        return {"$bytes": base64.b64encode(value).decode("ascii")}
    if isinstance(value, dict):
        return {"$map": [[key, canonical(value[key])] for key in sorted(value)]}
    if isinstance(value, (list, tuple)):
        return {"$array": [canonical(item) for item in value]}
    if hasattr(value, "latitude") and hasattr(value, "longitude"):
        return {"$geo": [value.latitude, value.longitude]}
    if hasattr(value, "_document_path"):
        return {"$reference": value._document_path}
    raise AcceptanceError("Unsupported Firestore value type; capture is incomplete.")


def fingerprint(key: bytes, purpose: str, value) -> str:
    encoded = json.dumps(canonical(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(key, purpose.encode() + b"\0" + encoded, hashlib.sha256).hexdigest()


def private_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never truncate an earlier baseline or follow a symlink.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)


def load_key(path: Path) -> bytes:
    if path.is_symlink():
        raise AcceptanceError("Fingerprint key must not be a symlink.")
    if not path.exists():
        private_write(path, secrets.token_bytes(32))
    if path.stat().st_mode & 0o077:
        raise AcceptanceError("Fingerprint key must have private permissions (0600).")
    key = path.read_bytes()
    if len(key) != 32:
        raise AcceptanceError("Fingerprint key must contain exactly 32 bytes.")
    return key


def _number(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def capture(client, *, project: str, database: str, key: bytes, max_documents: int = 100_000) -> dict:
    # Missing documents still return the server's read timestamp. Subsequent
    # enumerations and reads share that timestamp, avoiding a torn manifest.
    anchor = client.collection("system").document("run_lock").get(timeout=20)
    read_time = anchor.read_time
    if not isinstance(read_time, datetime):
        raise AcceptanceError("Firestore did not supply a consistent read timestamp.")
    now_ms = int(read_time.timestamp() * 1000)
    documents = {}
    counts = Counter()
    leases = []
    missing_parents = 0
    visited = 0

    def scan(collection):
        nonlocal missing_parents, visited
        # list_documents(show_missing=True in the SDK) includes nonexistent
        # ancestor documents that nevertheless contain stored child documents.
        for reference in collection.list_documents(timeout=20, read_time=read_time):
            visited += 1
            if visited > max_documents:
                raise AcceptanceError("Document limit reached; capture is incomplete. Increase --max-documents deliberately.")
            snapshot = reference.get(timeout=20, read_time=read_time)
            if snapshot.exists:
                data = snapshot.to_dict()
                parts = reference.path.split("/")
                kind = parts[-2] if parts[-2] in KINDS else "other"
                path_hash = fingerprint(key, "path", reference.path)
                documents[path_hash] = {"kind": kind, "content": fingerprint(key, "content", data)}
                counts[kind] += 1
                is_run = len(parts) == 4 and parts[0] == "users" and parts[2:] == ["runs", "active"]
                is_global = reference.path == "system/run_lock"
                if is_run or is_global:
                    status = data.get("status")
                    lease_ms = _number(data.get("lease_expires_at_ms"))
                    owned = bool(data.get("active") or data.get("is_running") or status in ACTIVE_STATUSES)
                    if is_global:
                        owned = bool(data.get("owner_id"))
                    leases.append({
                        "document": path_hash, "scope": "global" if is_global else "user",
                        "status": status if status in STATUSES else "unknown",
                        "appears_active": owned,
                        "live_lease": lease_ms is not None and lease_ms > now_ms,
                        "lease_expires_at_ms": lease_ms,
                        "heartbeat_at_ms": _number(data.get("heartbeat_at_ms")),
                        "stop_requested": bool(data.get("stop_requested")),
                        "owner_kind": "worker" if str(data.get("owner_id", "")).startswith("worker-") else "web_or_unknown",
                    })
            else:
                missing_parents += 1
            for child in reference.collections(timeout=20, read_time=read_time):
                scan(child)

    for collection in client.collections(timeout=20, read_time=read_time):
        scan(collection)
    return {
        "format_version": 1, "not_a_backup": True, "project": project, "database": database,
        "read_time": canonical(read_time)["$timestamp"], "completed_at": datetime.now(timezone.utc).isoformat(),
        "key_id": fingerprint(key, "key-id", "deployment-acceptance"),
        "document_count": len(documents), "missing_parent_count": missing_parents,
        "counts": dict(sorted(counts.items())), "documents": documents, "leases": leases,
        "live_user_work_count": sum(item["scope"] == "user" and item["appears_active"] and item["live_lease"] for item in leases),
        "expired_user_work_count": sum(item["scope"] == "user" and item["appears_active"] and not item["live_lease"] for item in leases),
        "limitations": ["Auto-rating and template generation have no shared durable active-work marker.",
                        "This capture does not contain Firebase Auth users or Secret Manager secrets."],
    }


def compare(before: dict, after: dict) -> dict:
    for field in ("format_version", "key_id", "project", "database"):
        if before.get(field) != after.get(field):
            raise AcceptanceError(f"Manifests have different {field}; refusing an invalid comparison.")
    old, new = before["documents"], after["documents"]
    added, removed = sorted(new.keys() - old.keys()), sorted(old.keys() - new.keys())
    changed = sorted(key for key in old.keys() & new.keys() if old[key] != new[key])
    def records(keys, source):
        return [{"document": key, "kind": source[key]["kind"]} for key in keys]
    return {"unchanged": not (added or removed or changed), "before_count": len(old), "after_count": len(new),
            "added": records(added, new), "removed": records(removed, old), "changed": records(changed, new)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("capture", help="Read production Firestore; never write cloud data")
    snapshot.add_argument("--project", required=True)
    snapshot.add_argument("--database", default="(default)")
    snapshot.add_argument("--key-file", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    snapshot.add_argument("--max-documents", type=int, default=100_000)
    diff = commands.add_parser("compare", help="Compare local private fingerprint files")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            result = compare(json.loads(args.before.read_text()), json.loads(args.after.read_text()))
            print(json.dumps(result, indent=2))
            return 0 if result["unchanged"] else 2
        if os.getenv("FIRESTORE_EMULATOR_HOST"):
            raise AcceptanceError("FIRESTORE_EMULATOR_HOST is set; refusing to label emulator data as production.")
        if not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", args.project):
            raise AcceptanceError("An explicit valid Google Cloud project ID is required.")
        if not args.database.strip() or args.max_documents < 1:
            raise AcceptanceError("Choose a nonempty database ID and positive document limit.")
        if args.output.exists() or args.output.is_symlink():
            raise AcceptanceError("Output already exists; preserve the earlier baseline and use a new filename.")
        key = load_key(args.key_file)
        from google.cloud import firestore
        client = firestore.Client(project=args.project, database=args.database)
        try:
            result = capture(client, project=args.project, database=args.database, key=key, max_documents=args.max_documents)
        finally:
            client.close()
        private_write(args.output, json.dumps(result, indent=2, sort_keys=True).encode())
        print(json.dumps({name: result[name] for name in (
            "project", "database", "read_time", "document_count", "missing_parent_count", "counts",
            "live_user_work_count", "expired_user_work_count", "leases", "limitations")}, indent=2))
        return 0
    except AcceptanceError as error:
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:
        # Cloud errors may include resource identifiers. Do not print payloads,
        # credentials, raw document paths, or arbitrary SDK exception messages.
        print(f"Acceptance capture failed ({type(error).__name__}); no complete evidence was written.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
