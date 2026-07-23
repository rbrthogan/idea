# Database module for Firestore operations
# Provides CRUD operations for user-scoped data with encryption for sensitive fields

import os
import json
import base64
import hashlib
import logging
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Any
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from google.cloud import firestore

logger = logging.getLogger(__name__)

# Firestore client (lazy initialization)
_db = None


def get_db() -> firestore.Client:
    """Get or create Firestore client."""
    global _db
    if _db is None:
        _db = firestore.Client()
    return _db


# --- Run state / leases (for resilience & multi-tenant coordination) ---

RUNS_COLLECTION = "runs"
ACTIVE_RUN_DOC_ID = "active"
GLOBAL_RUN_LOCK_DOC_ID = "run_lock"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _run_doc_ref(db: firestore.Client, user_id: str):
    return db.collection("users").document(user_id).collection(RUNS_COLLECTION).document(ACTIVE_RUN_DOC_ID)


def _global_lock_ref(db: firestore.Client):
    return db.collection("system").document(GLOBAL_RUN_LOCK_DOC_ID)


async def get_active_run(user_id: str) -> Optional[Dict[str, Any]]:
    """Get the active run state for a user (if any)."""
    db = get_db()
    doc = _run_doc_ref(db, user_id).get()
    if doc.exists:
        return doc.to_dict()
    return None


async def update_active_run(user_id: str, updates: Dict[str, Any], lease_seconds: Optional[int] = None,
                           owner_id: Optional[str] = None) -> None:
    """Update active run state (merge). Optionally refresh the lease."""
    db = get_db()
    now_ms = _now_ms()
    update_data = dict(updates)
    update_data["updated_at_ms"] = now_ms
    if owner_id:
        update_data["owner_id"] = owner_id
    if lease_seconds is not None:
        update_data["heartbeat_at_ms"] = now_ms
        update_data["lease_expires_at_ms"] = now_ms + int(lease_seconds * 1000)
    _run_doc_ref(db, user_id).set(update_data, merge=True)


async def claim_active_run(user_id: str, run_data: Dict[str, Any], lease_seconds: int,
                           owner_id: str) -> Dict[str, Any]:
    """
    Claim the user's active run slot. Returns {"ok": bool, "data": existing_or_new}.
    """
    db = get_db()
    now_ms = _now_ms()
    lease_expires_at_ms = now_ms + int(lease_seconds * 1000)
    doc_ref = _run_doc_ref(db, user_id)
    active_statuses = {"starting", "in_progress", "resuming", "continuing", "stopping"}

    @firestore.transactional
    def _claim(transaction):
        doc = doc_ref.get(transaction=transaction)
        if doc.exists:
            existing = doc.to_dict()
            existing_status = existing.get("status")
            existing_lease = existing.get("lease_expires_at_ms", 0)
            if existing_status in active_statuses and existing_lease > now_ms:
                return {"ok": False, "data": existing}

        data = dict(run_data)
        data.update({
            "owner_id": owner_id,
            "status": data.get("status") or "in_progress",
            "is_running": True,
            "active": True,
            # Always reset stale stop flags when claiming a fresh run slot.
            "stop_requested": False,
            "heartbeat_at_ms": now_ms,
            "lease_expires_at_ms": lease_expires_at_ms,
            "updated_at_ms": now_ms,
            "created_at_ms": data.get("created_at_ms") or now_ms,
        })
        transaction.set(doc_ref, data, merge=True)
        return {"ok": True, "data": data}

    return _claim(db.transaction())


async def claim_global_run_lock(owner_id: str, user_id: str, evolution_id: str,
                                lease_seconds: int) -> Dict[str, Any]:
    """
    Claim the global run lock (single slot). Returns {"ok": bool, "data": existing_or_new}.
    """
    db = get_db()
    now_ms = _now_ms()
    lease_expires_at_ms = now_ms + int(lease_seconds * 1000)
    doc_ref = _global_lock_ref(db)

    @firestore.transactional
    def _claim(transaction):
        doc = doc_ref.get(transaction=transaction)
        if doc.exists:
            existing = doc.to_dict()
            existing_lease = existing.get("lease_expires_at_ms", 0)
            if existing_lease > now_ms:
                return {"ok": False, "data": existing}

        data = {
            "owner_id": owner_id,
            "owner_user_id": user_id,
            "evolution_id": evolution_id,
            "heartbeat_at_ms": now_ms,
            "lease_expires_at_ms": lease_expires_at_ms,
            "updated_at_ms": now_ms,
            "created_at_ms": now_ms,
        }
        transaction.set(doc_ref, data, merge=True)
        return {"ok": True, "data": data}

    return _claim(db.transaction())


async def refresh_global_run_lock(owner_id: str, lease_seconds: int) -> None:
    """Refresh the global run lock lease if owned by this instance."""
    db = get_db()
    now_ms = _now_ms()
    doc_ref = _global_lock_ref(db)
    doc = doc_ref.get()
    if not doc.exists:
        return
    data = doc.to_dict()
    if data.get("owner_id") != owner_id:
        return
    doc_ref.set({
        "heartbeat_at_ms": now_ms,
        "lease_expires_at_ms": now_ms + int(lease_seconds * 1000),
        "updated_at_ms": now_ms,
    }, merge=True)


async def release_global_run_lock(owner_id: str) -> None:
    """Release the global run lock if owned by this instance."""
    db = get_db()
    doc_ref = _global_lock_ref(db)
    doc = doc_ref.get()
    if not doc.exists:
        return
    data = doc.to_dict()
    if data.get("owner_id") != owner_id:
        return
    now_ms = _now_ms()
    doc_ref.set({
        "lease_expires_at_ms": now_ms,
        "updated_at_ms": now_ms,
    }, merge=True)


# --- Encryption utilities for API keys ---


class EncryptionConfigurationError(RuntimeError):
    """Raised when the API-key encryption secret is absent or unsafe."""


class APIKeyDecryptionError(RuntimeError):
    """Raised when a stored API key cannot be decrypted with this deployment key."""

def _get_encryption_key() -> bytes:
    """
    Derive encryption key from environment secret.
    In production, use Secret Manager for the master key.
    """
    master_key = os.getenv("ENCRYPTION_KEY")
    if not master_key:
        raise EncryptionConfigurationError(
            "ENCRYPTION_KEY is required before API keys can be stored or read"
        )
    if len(master_key.encode("utf-8")) < 32:
        raise EncryptionConfigurationError(
            "ENCRYPTION_KEY must contain at least 32 bytes"
        )
    salt = os.getenv("ENCRYPTION_SALT", "idea-evolution-salt").encode()

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    return base64.urlsafe_b64encode(kdf.derive(master_key.encode()))


def encrypt_api_key(api_key: str) -> str:
    """Encrypt an API key for storage."""
    f = Fernet(_get_encryption_key())
    return f.encrypt(api_key.encode()).decode()


def decrypt_api_key(encrypted_key: str) -> str:
    """Decrypt an API key from storage."""
    f = Fernet(_get_encryption_key())
    return f.decrypt(encrypted_key.encode()).decode()


# --- User Settings ---

async def get_user_settings(user_id: str) -> Dict[str, Any]:
    """Get user settings including encrypted API key."""
    db = get_db()
    doc = db.collection("users").document(user_id).collection("settings").document("main").get()

    if doc.exists:
        data = doc.to_dict()
        # Don't decrypt API key here - let caller decide
        return data
    return {}


async def save_user_api_key(user_id: str, api_key: str) -> None:
    """Save encrypted API key for user."""
    db = get_db()
    encrypted = encrypt_api_key(api_key)

    # Ensure parent user document exists (required for admin listing)
    user_ref = db.collection("users").document(user_id)
    user_doc = user_ref.get()
    if not user_doc.exists:
        user_ref.set({
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat()
        })
    else:
        user_ref.update({
            "last_activity": datetime.utcnow().isoformat()
        })

    db.collection("users").document(user_id).collection("settings").document("main").set({
        "api_key_encrypted": encrypted,
        "api_key_set": True,
        "updated_at": datetime.utcnow().isoformat()
    }, merge=True)


async def get_user_api_key(user_id: str) -> Optional[str]:
    """Get decrypted API key for user."""
    settings = await get_user_settings(user_id)
    encrypted = settings.get("api_key_encrypted")
    if encrypted:
        try:
            return decrypt_api_key(encrypted)
        except EncryptionConfigurationError:
            raise
        except (InvalidToken, UnicodeDecodeError, ValueError, TypeError) as exc:
            raise APIKeyDecryptionError(
                f"Stored API key for user {user_id!r} cannot be decrypted"
            ) from exc
    return None


async def delete_user_api_key(user_id: str) -> None:
    """Delete user's API key."""
    db = get_db()
    db.collection("users").document(user_id).collection("settings").document("main").update({
        "api_key_encrypted": firestore.DELETE_FIELD,
        "api_key_set": False,
        "updated_at": datetime.utcnow().isoformat()
    })


# --- Evolutions ---

EVOLUTION_CHUNKS_COLLECTION = "chunks"
EVOLUTION_CHUNK_VERSIONS_COLLECTION = "chunk_versions"
EVOLUTION_TOMBSTONES_COLLECTION = "evolution_tombstones"
EVOLUTION_FIELD_CHUNK_SIZE = 240_000
EVOLUTION_CHUNK_GC_GRACE_SECONDS = 600
EVOLUTION_CHUNK_SCHEMA_VERSION = 2

_EXPECTED_REVISION_UNSET = object()


class EvolutionSaveConflict(RuntimeError):
    """Raised when a stale evolution save loses an optimistic publish race."""


class EvolutionStorageIntegrityError(RuntimeError):
    """Raised when a published evolution cannot be reconstructed exactly."""

# Fields that may contain nested arrays and need JSON serialization.
# Firestore does not support arrays within arrays, so legacy records stored these
# as JSON strings. Large fields now live in chunk documents to avoid the 1 MiB
# document limit while preserving the same public API from get_evolution().
EVOLUTION_JSON_FIELDS = [
    'history',                    # [[ideas], [ideas], ...]
    'population',                 # [ideas]
    'diversity_history',          # nested diversity samples
    'island_diversity_history',   # nested island diversity samples
    'migration_history',          # migration event details
    'contexts',                   # may contain nested structures
    'specific_prompts',           # may contain nested structures
    'breeding_prompts',           # may contain nested structures
    'tournament_history',         # pairwise comparison details
    'comparison_cache',           # comparison cache map
    'token_counts',               # dict with nested diagnostics
    'config',                     # dict with config values
    'fitness_elo_stats',
    'fitness_diversity_stats',
    'fitness_metric_stats',
]

EVOLUTION_NESTED_FIELDS = EVOLUTION_JSON_FIELDS

EVOLUTION_EXTERNALIZED_FIELDS = {
    'history',
    'population',
    'diversity_history',
    'island_diversity_history',
    'migration_history',
    'contexts',
    'specific_prompts',
    'breeding_prompts',
    'tournament_history',
    'comparison_cache',
    'token_counts',
}


def _chunk_text(value: str, chunk_size: Optional[int] = None) -> List[str]:
    if chunk_size is None:
        chunk_size = EVOLUTION_FIELD_CHUNK_SIZE
    if chunk_size <= 0:
        chunk_size = EVOLUTION_FIELD_CHUNK_SIZE
    return [value[i:i + chunk_size] for i in range(0, len(value), chunk_size)] or [""]


def _summarize_history(history: Any) -> Dict[str, int]:
    if not isinstance(history, list):
        return {"generation_count": 0, "total_ideas": 0, "last_generation_size": 0}
    return {
        "generation_count": len(history),
        "total_ideas": sum(len(gen) for gen in history if isinstance(gen, list)),
        "last_generation_size": len(history[-1]) if history and isinstance(history[-1], list) else 0,
    }


def _prepare_evolution_save_data(
    data: Dict[str, Any],
    user_id: str,
    updated_at: str,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    save_data = dict(data)
    save_data["updated_at"] = updated_at
    save_data["user_id"] = user_id

    if "history" in save_data:
        save_data["_history_summary"] = _summarize_history(save_data.get("history"))
    if "population" in save_data and isinstance(save_data.get("population"), list):
        save_data["_population_summary"] = {"population_size": len(save_data["population"])}

    chunk_payloads: Dict[str, Dict[str, Any]] = {}
    externalized_fields: Dict[str, Dict[str, Any]] = {}
    chunk_version = str(uuid.uuid4())
    created_at_ms = _now_ms()
    save_data["_save_revision"] = chunk_version

    for field in EVOLUTION_JSON_FIELDS:
        if field not in save_data or save_data[field] is None:
            continue

        serialized = json.dumps(save_data[field], default=str)
        if field in EVOLUTION_EXTERNALIZED_FIELDS:
            chunks = _chunk_text(serialized)
            payload_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            externalized_fields[field] = {
                "schema_version": EVOLUTION_CHUNK_SCHEMA_VERSION,
                "encoding": "json",
                "chunk_count": len(chunks),
                "chunk_version": chunk_version,
                "payload_sha256": payload_sha256,
            }
            for index, chunk in enumerate(chunks):
                chunk_id = f"{field}__{chunk_version}__{index:04d}"
                chunk_payloads[chunk_id] = {
                    "field": field,
                    "index": index,
                    "chunk_count": len(chunks),
                    "chunk_version": chunk_version,
                    "schema_version": EVOLUTION_CHUNK_SCHEMA_VERSION,
                    "value": chunk,
                    "updated_at": updated_at,
                    "created_at_ms": created_at_ms,
                }
            save_data.pop(field, None)
        else:
            save_data[field] = serialized

    if externalized_fields:
        save_data["_externalized_fields"] = externalized_fields
    else:
        save_data.pop("_externalized_fields", None)

    return save_data, chunk_payloads


def _deserialize_evolution_json_fields(
    data: Dict[str, Any],
    *,
    strict: bool = False,
) -> None:
    for field in EVOLUTION_JSON_FIELDS:
        if field in data and isinstance(data[field], str):
            try:
                data[field] = json.loads(data[field])
            except json.JSONDecodeError:
                if strict:
                    raise EvolutionStorageIntegrityError(
                        f"stored evolution field {field!r} is not valid JSON"
                    ) from None


def _chunk_doc_id(field: str, chunk_version: str, index: int) -> str:
    return f"{field}__{chunk_version}__{index:04d}"


def _chunk_version_ref(doc_ref: Any, chunk_version: str) -> Any:
    return doc_ref.collection(EVOLUTION_CHUNK_VERSIONS_COLLECTION).document(
        chunk_version
    )


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _load_externalized_evolution_fields(doc_ref: Any, data: Dict[str, Any]) -> None:
    externalized = data.get("_externalized_fields")
    if externalized is None:
        return
    if not isinstance(externalized, dict):
        raise EvolutionStorageIntegrityError(
            "evolution externalized-field manifest must be an object"
        )

    chunk_collection = doc_ref.collection(EVOLUTION_CHUNKS_COLLECTION)
    manifest_revision = data.get("_save_revision")
    for field, meta in externalized.items():
        if field not in EVOLUTION_EXTERNALIZED_FIELDS:
            raise EvolutionStorageIntegrityError(
                f"unsupported externalized evolution field {field!r}"
            )
        if not isinstance(meta, dict):
            raise EvolutionStorageIntegrityError(
                f"chunk manifest for {field!r} must be an object"
            )
        try:
            chunk_count = int(meta.get("chunk_count", 0))
        except (TypeError, ValueError):
            raise EvolutionStorageIntegrityError(
                f"chunk count for {field!r} is invalid"
            ) from None
        chunk_version = str(meta.get("chunk_version") or "")
        if chunk_count <= 0 or not chunk_version:
            raise EvolutionStorageIntegrityError(
                f"chunk manifest for {field!r} is incomplete"
            )
        try:
            schema_version = int(meta.get("schema_version", 1))
        except (TypeError, ValueError):
            raise EvolutionStorageIntegrityError(
                f"chunk schema version for {field!r} is invalid"
            ) from None
        expected_digest = meta.get("payload_sha256")
        if schema_version >= EVOLUTION_CHUNK_SCHEMA_VERSION and not _valid_sha256(
            expected_digest
        ):
            raise EvolutionStorageIntegrityError(
                f"chunk digest for {field!r} is missing or invalid"
            )
        if (
            schema_version >= EVOLUTION_CHUNK_SCHEMA_VERSION
            and chunk_version != manifest_revision
        ):
            raise EvolutionStorageIntegrityError(
                f"chunk version for {field!r} does not match the published revision"
            )
        if field in data:
            raise EvolutionStorageIntegrityError(
                f"evolution field {field!r} is both inline and externalized"
            )

        if schema_version >= EVOLUTION_CHUNK_SCHEMA_VERSION:
            version_doc = _chunk_version_ref(doc_ref, chunk_version).get()
            version_data = version_doc.to_dict() if version_doc.exists else {}
            if (
                not version_doc.exists
                or version_data.get("state") != "current"
                or str(version_data.get("chunk_version") or "") != chunk_version
            ):
                raise EvolutionStorageIntegrityError(
                    f"published chunk version for {field!r} is missing or inactive"
                )

        chunks: List[str] = []
        for index in range(chunk_count):
            chunk_doc = chunk_collection.document(_chunk_doc_id(field, chunk_version, index)).get()
            if not chunk_doc.exists:
                raise EvolutionStorageIntegrityError(
                    f"missing chunk {index + 1}/{chunk_count} for {field!r}"
                )
            chunk_data = chunk_doc.to_dict() or {}
            if chunk_data.get("field") != field:
                raise EvolutionStorageIntegrityError(
                    f"chunk field mismatch for {field!r} at index {index}"
                )
            try:
                stored_index = int(chunk_data.get("index"))
            except (TypeError, ValueError):
                stored_index = -1
            if stored_index != index:
                raise EvolutionStorageIntegrityError(
                    f"chunk index mismatch for {field!r} at index {index}"
                )
            if str(chunk_data.get("chunk_version") or "") != chunk_version:
                raise EvolutionStorageIntegrityError(
                    f"chunk version mismatch for {field!r} at index {index}"
                )
            if schema_version >= EVOLUTION_CHUNK_SCHEMA_VERSION:
                try:
                    stored_schema_version = int(chunk_data.get("schema_version"))
                except (TypeError, ValueError):
                    stored_schema_version = -1
                if stored_schema_version != schema_version:
                    raise EvolutionStorageIntegrityError(
                        f"chunk schema mismatch for {field!r} at index {index}"
                    )
                try:
                    stored_count = int(chunk_data.get("chunk_count"))
                except (TypeError, ValueError):
                    stored_count = -1
                if stored_count != chunk_count:
                    raise EvolutionStorageIntegrityError(
                        f"chunk count mismatch for {field!r} at index {index}"
                    )
            chunk_value = chunk_data.get("value")
            if not isinstance(chunk_value, str):
                raise EvolutionStorageIntegrityError(
                    f"chunk payload for {field!r} at index {index} is not text"
                )
            chunks.append(chunk_value)

        raw_value = "".join(chunks)
        if expected_digest is not None:
            actual_digest = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()
            if actual_digest != expected_digest:
                raise EvolutionStorageIntegrityError(
                    f"chunk digest mismatch for {field!r}"
                )
        if meta.get("encoding") == "json":
            try:
                data[field] = json.loads(raw_value)
            except json.JSONDecodeError:
                raise EvolutionStorageIntegrityError(
                    f"chunk payload for {field!r} is not valid JSON"
                ) from None
        elif meta.get("encoding") in {None, "text"} and schema_version < 2:
            # Compatibility with the earliest chunk-manifest prototype.
            data[field] = raw_value
        else:
            raise EvolutionStorageIntegrityError(
                f"chunk encoding for {field!r} is unsupported"
            )


def _delete_evolution_chunks(doc_ref: Any) -> None:
    for collection_name in (
        EVOLUTION_CHUNKS_COLLECTION,
        EVOLUTION_CHUNK_VERSIONS_COLLECTION,
    ):
        collection = doc_ref.collection(collection_name)
        for child_doc in collection.stream():
            collection.document(child_doc.id).delete()


def _delete_staged_chunk_version(
    doc_ref: Any,
    chunk_version: str,
    chunk_ids: Optional[List[str]] = None,
) -> None:
    chunk_collection = doc_ref.collection(EVOLUTION_CHUNKS_COLLECTION)
    if chunk_ids is None:
        chunk_ids = [
            chunk_doc.id
            for chunk_doc in chunk_collection.stream()
            if (chunk_doc.to_dict() or {}).get("chunk_version") == chunk_version
        ]
    for chunk_id in chunk_ids:
        chunk_collection.document(chunk_id).delete()
    _chunk_version_ref(doc_ref, chunk_version).delete()


def _evolution_tombstone_ref(
    db: firestore.Client,
    user_id: str,
    evolution_id: str,
) -> Any:
    return (
        db.collection("users")
        .document(user_id)
        .collection(EVOLUTION_TOMBSTONES_COLLECTION)
        .document(evolution_id)
    )


def _manifest_chunk_versions(data: Dict[str, Any]) -> set[str]:
    return {
        str(meta.get("chunk_version"))
        for meta in (data.get("_externalized_fields") or {}).values()
        if isinstance(meta, dict) and meta.get("chunk_version")
    }


def _publish_evolution_manifest(
    db: firestore.Client,
    doc_ref: Any,
    save_data: Dict[str, Any],
    expected_revision: Optional[str],
    tombstone_ref: Optional[Any] = None,
) -> bool:
    """Atomically publish a manifest only if its base revision is still current."""

    new_versions = _manifest_chunk_versions(save_data)
    new_revision = str(save_data.get("_save_revision") or "")
    published_at_ms = _now_ms()

    def _read_publish_inputs(transaction: Optional[Any] = None):
        current_doc = doc_ref.get(transaction=transaction)
        tombstone_doc = (
            tombstone_ref.get(transaction=transaction)
            if tombstone_ref is not None
            else None
        )
        staged_versions = {
            version: _chunk_version_ref(doc_ref, version).get(transaction=transaction)
            for version in new_versions
        }
        return current_doc, tombstone_doc, staged_versions

    def _can_publish(current_doc, tombstone_doc, staged_versions) -> bool:
        if tombstone_doc is not None and tombstone_doc.exists:
            return False
        current_data = current_doc.to_dict() if current_doc.exists else {}
        if current_data.get("_deleted"):
            return False
        if current_data.get("_save_revision") != expected_revision:
            return False
        for version, version_doc in staged_versions.items():
            version_data = version_doc.to_dict() if version_doc.exists else {}
            if (
                not version_doc.exists
                or version_data.get("state") != "staged"
                or version_data.get("chunk_version") != version
            ):
                return False
        return True

    def _version_updates(current_data: Dict[str, Any]):
        current_versions = _manifest_chunk_versions(current_data)
        current_update = {
            "chunk_version": new_revision,
            "state": "current",
            "published_at": firestore.SERVER_TIMESTAMP,
            "published_at_ms": published_at_ms,
            "updated_at_ms": published_at_ms,
        }
        retired_update = {
            "state": "retired",
            "retired_at": firestore.SERVER_TIMESTAMP,
            "retired_at_ms": published_at_ms,
            "updated_at_ms": published_at_ms,
        }
        return current_versions, current_update, retired_update

    # Minimal in-memory test doubles do not implement transactions. Production
    # Firestore clients always do; the fallback keeps pure serialization tests
    # lightweight without weakening the deployed path.
    if not callable(getattr(db, "transaction", None)):
        current_doc, tombstone_doc, staged_versions = _read_publish_inputs()
        if not _can_publish(current_doc, tombstone_doc, staged_versions):
            return False
        current_data = current_doc.to_dict() if current_doc.exists else {}
        current_versions, current_update, retired_update = _version_updates(
            current_data
        )
        doc_ref.set(save_data)
        for version in new_versions:
            _chunk_version_ref(doc_ref, version).set(current_update, merge=True)
        for version in current_versions - new_versions:
            _chunk_version_ref(doc_ref, version).set(retired_update, merge=True)
        return True

    @firestore.transactional
    def _publish(transaction):
        current_doc, tombstone_doc, staged_versions = _read_publish_inputs(transaction)
        if not _can_publish(current_doc, tombstone_doc, staged_versions):
            return False
        current_data = current_doc.to_dict() if current_doc.exists else {}
        current_versions, current_update, retired_update = _version_updates(
            current_data
        )
        transaction.set(doc_ref, save_data)
        for version in new_versions:
            transaction.set(
                _chunk_version_ref(doc_ref, version),
                current_update,
                merge=True,
            )
        for version in current_versions - new_versions:
            transaction.set(
                _chunk_version_ref(doc_ref, version),
                retired_update,
                merge=True,
            )
        return True

    return bool(_publish(db.transaction()))


def _delete_stale_evolution_chunks(doc_ref: Any) -> None:
    """Delete retired/abandoned versions only after their lifecycle grace period."""
    current_doc = doc_ref.get()
    current_data = current_doc.to_dict() if current_doc.exists else {}
    current_versions = _manifest_chunk_versions(current_data)
    cutoff_ms = _now_ms() - int(EVOLUTION_CHUNK_GC_GRACE_SECONDS * 1000)
    version_collection = doc_ref.collection(EVOLUTION_CHUNK_VERSIONS_COLLECTION)
    eligible_versions: set[str] = set()
    tracked_versions: set[str] = set()
    for version_doc in version_collection.stream():
        version_data = version_doc.to_dict() or {}
        chunk_version = str(
            version_data.get("chunk_version") or version_doc.id or ""
        )
        if not chunk_version or chunk_version in current_versions:
            continue
        tracked_versions.add(chunk_version)
        state = version_data.get("state")
        if state == "retired":
            age_value = version_data.get("retired_at")
            fallback_ms = version_data.get("retired_at_ms")
        elif state == "staged":
            age_value = None
            fallback_ms = version_data.get("created_at_ms")
        elif state == "deleting":
            eligible_versions.add(chunk_version)
            continue
        else:
            continue
        if isinstance(age_value, datetime):
            lifecycle_ms = int(age_value.timestamp() * 1000)
        else:
            try:
                lifecycle_ms = int(fallback_ms)
            except (TypeError, ValueError):
                continue
        if lifecycle_ms <= cutoff_ms:
            eligible_versions.add(chunk_version)

    chunk_collection = doc_ref.collection(EVOLUTION_CHUNKS_COLLECTION)
    # Older writers did not create lifecycle documents.  Treat an unreferenced
    # version as an abandoned stage, but only after the same grace period.
    untracked_created_at: Dict[str, int] = {}
    for chunk_doc in chunk_collection.stream():
        chunk_data = chunk_doc.to_dict() or {}
        chunk_version = str(chunk_data.get("chunk_version") or "")
        if (
            not chunk_version
            or chunk_version in current_versions
            or chunk_version in tracked_versions
        ):
            continue
        try:
            created_at_ms = int(chunk_data.get("created_at_ms"))
        except (TypeError, ValueError):
            continue
        previous = untracked_created_at.get(chunk_version)
        untracked_created_at[chunk_version] = (
            created_at_ms if previous is None else max(previous, created_at_ms)
        )
    eligible_versions.update(
        version
        for version, created_at_ms in untracked_created_at.items()
        if created_at_ms <= cutoff_ms
    )

    if not eligible_versions:
        return

    for chunk_doc in chunk_collection.stream():
        chunk_data = chunk_doc.to_dict() or {}
        if chunk_data.get("chunk_version") in eligible_versions:
            chunk_collection.document(chunk_doc.id).delete()
    for chunk_version in eligible_versions:
        version_collection.document(chunk_version).delete()


async def save_evolution(
    user_id: str,
    evolution_id: str,
    data: Dict[str, Any],
    *,
    expected_revision: Any = _EXPECTED_REVISION_UNSET,
) -> str:
    """Save an evolution for a user.

    Note: Firestore doesn't support nested arrays, so we serialize fields like
    'history' (which is [[ideas], [ideas]]) to JSON. Large fields are stored in
    chunk documents so completed runs do not exceed Firestore's document limit.
    """
    db = get_db()
    updated_at = datetime.utcnow().isoformat()
    if expected_revision is _EXPECTED_REVISION_UNSET:
        expected_revision = data.get("_save_revision")
    if expected_revision is not None:
        if not isinstance(expected_revision, str) or not expected_revision:
            raise ValueError("expected evolution revision must be a non-empty string")
    save_data, chunk_payloads = _prepare_evolution_save_data(data, user_id, updated_at)
    chunk_version = str(save_data["_save_revision"])

    doc_ref = db.collection("users").document(user_id).collection("evolutions").document(evolution_id)
    tombstone_ref = _evolution_tombstone_ref(db, user_id, evolution_id)
    chunk_collection = doc_ref.collection(EVOLUTION_CHUNKS_COLLECTION)
    staged_chunk_ids: List[str] = []
    if chunk_payloads:
        _chunk_version_ref(doc_ref, chunk_version).set(
            {
                "chunk_version": chunk_version,
                "state": "staged",
                "created_at_ms": _now_ms(),
                "updated_at_ms": _now_ms(),
            }
        )
    try:
        for chunk_id, chunk_data in chunk_payloads.items():
            chunk_collection.document(chunk_id).set(chunk_data)
            staged_chunk_ids.append(chunk_id)
    except Exception:
        _delete_staged_chunk_version(doc_ref, chunk_version, staged_chunk_ids)
        raise

    try:
        published = _publish_evolution_manifest(
            db,
            doc_ref,
            save_data,
            expected_revision,
            tombstone_ref=tombstone_ref,
        )
    except Exception:
        # A transport error can be ambiguous: Firestore may have committed the
        # manifest before the client observed the failure.  Never delete this
        # version here, because it may already be current.  A genuinely
        # abandoned stage is collected after the normal grace period.
        raise
    if not published:
        _delete_staged_chunk_version(
            doc_ref,
            chunk_version,
            list(chunk_payloads),
        )
        raise EvolutionSaveConflict(
            f"Evolution {evolution_id!r} changed while this save was being published"
        )
    try:
        _delete_stale_evolution_chunks(doc_ref)
    except Exception:
        logger.exception(
            "Evolution %s was published but stale chunk cleanup failed",
            evolution_id,
        )
    return chunk_version


async def get_evolution(user_id: str, evolution_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific evolution for a user."""
    db = get_db()
    doc_ref = db.collection("users").document(user_id).collection("evolutions").document(evolution_id)

    def _visible_data(snapshot: Any) -> Optional[Dict[str, Any]]:
        if not snapshot.exists:
            return None
        snapshot_data = snapshot.to_dict() or {}
        if snapshot_data.get("_deleted"):
            return None
        return snapshot_data

    doc = doc_ref.get()
    data = _visible_data(doc)
    if data is None:
        return None
    initial_revision = data.get("_save_revision")
    try:
        _load_externalized_evolution_fields(doc_ref, data)
    except EvolutionStorageIntegrityError:
        # A reader may have fetched the old manifest immediately before a
        # concurrent publish.  Retry exactly once with the new manifest; an
        # unchanged revision is genuine corruption and must remain visible.
        latest = _visible_data(doc_ref.get())
        if latest is None:
            return None
        if latest.get("_save_revision") == initial_revision:
            raise
        data = latest
        _load_externalized_evolution_fields(doc_ref, data)
    _deserialize_evolution_json_fields(data, strict=True)
    return data


async def list_evolutions(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """List all evolutions for a user."""
    db = get_db()
    docs = (db.collection("users").document(user_id).collection("evolutions")
            .order_by("updated_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream())

    evolutions = []
    for doc in docs:
        data = doc.to_dict() or {}
        if data.get("_deleted"):
            continue
        data["id"] = doc.id
        # Keep list queries cheap: deserialize fields present on the summary doc,
        # but do not fetch externalized history chunks for every row.
        _deserialize_evolution_json_fields(data)
        evolutions.append(data)
    return evolutions


def _publish_evolution_tombstone(
    db: firestore.Client,
    doc_ref: Any,
    tombstone_ref: Any,
    *,
    user_id: str,
    evolution_id: str,
) -> bool:
    """Atomically hide an evolution and reserve its id against stale saves."""

    deleted_at_ms = _now_ms()

    def _tombstone_data(current_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "evolution_id": evolution_id,
            "user_id": user_id,
            "deleted_revision": current_data.get("_save_revision"),
            "deleted_at": firestore.SERVER_TIMESTAMP,
            "deleted_at_ms": deleted_at_ms,
        }

    if not callable(getattr(db, "transaction", None)):
        current_doc = doc_ref.get()
        if not current_doc.exists:
            return False
        current_data = current_doc.to_dict() or {}
        tombstone_ref.set(_tombstone_data(current_data))
        doc_ref.delete()
        return True

    @firestore.transactional
    def _delete(transaction):
        current_doc = doc_ref.get(transaction=transaction)
        if not current_doc.exists:
            return False
        current_data = current_doc.to_dict() or {}
        transaction.set(tombstone_ref, _tombstone_data(current_data))
        transaction.delete(doc_ref)
        return True

    return bool(_delete(db.transaction()))


async def delete_evolution(user_id: str, evolution_id: str) -> bool:
    """Delete an evolution without allowing an in-flight stale save to revive it."""
    db = get_db()
    doc_ref = db.collection("users").document(user_id).collection("evolutions").document(evolution_id)
    tombstone_ref = _evolution_tombstone_ref(db, user_id, evolution_id)
    deleted = _publish_evolution_tombstone(
        db,
        doc_ref,
        tombstone_ref,
        user_id=user_id,
        evolution_id=evolution_id,
    )
    if deleted:
        try:
            _delete_evolution_chunks(doc_ref)
        except Exception:
            # The parent is already atomically hidden and its tombstone blocks
            # resurrection.  Cleanup is idempotent and can safely be retried.
            logger.exception(
                "Evolution %s was deleted but chunk cleanup failed",
                evolution_id,
            )
        return True
    # A previous attempt may have committed the tombstone before cleanup.
    if tombstone_ref.get().exists:
        _delete_evolution_chunks(doc_ref)
        return False
    return False


async def rename_evolution(user_id: str, evolution_id: str, new_name: str) -> bool:
    """Rename an evolution."""
    db = get_db()
    doc_ref = db.collection("users").document(user_id).collection("evolutions").document(evolution_id)
    doc = doc_ref.get()
    if doc.exists:
        doc_ref.update({
            "name": new_name,
            "updated_at": datetime.utcnow().isoformat()
        })
        return True
    return False


# --- Checkpoints ---

async def save_checkpoint(user_id: str, checkpoint_id: str, data: Dict[str, Any]) -> None:
    """Save a checkpoint for a user."""
    db = get_db()
    data["updated_at"] = datetime.utcnow().isoformat()
    data["user_id"] = user_id

    db.collection("users").document(user_id).collection("checkpoints").document(checkpoint_id).set(data)


async def get_checkpoint(user_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific checkpoint for a user."""
    db = get_db()
    doc = db.collection("users").document(user_id).collection("checkpoints").document(checkpoint_id).get()

    if doc.exists:
        return doc.to_dict()
    return None


async def list_checkpoints(user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """List all checkpoints for a user."""
    db = get_db()
    docs = (db.collection("users").document(user_id).collection("checkpoints")
            .order_by("updated_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream())

    checkpoints = []
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        checkpoints.append(data)
    return checkpoints


async def delete_checkpoint(user_id: str, checkpoint_id: str) -> bool:
    """Delete a checkpoint."""
    db = get_db()
    doc_ref = db.collection("users").document(user_id).collection("checkpoints").document(checkpoint_id)
    doc = doc_ref.get()
    if doc.exists:
        doc_ref.delete()
        return True
    return False


# --- User Templates ---

async def save_user_template(user_id: str, template_id: str, data: Dict[str, Any]) -> None:
    """Save a custom template for a user."""
    db = get_db()
    data["updated_at"] = datetime.utcnow().isoformat()
    data["user_id"] = user_id

    db.collection("users").document(user_id).collection("templates").document(template_id).set(data)


async def get_user_template(user_id: str, template_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific template for a user."""
    db = get_db()
    doc = db.collection("users").document(user_id).collection("templates").document(template_id).get()

    if doc.exists:
        return doc.to_dict()
    return None


async def list_user_templates(user_id: str) -> List[Dict[str, Any]]:
    """List all custom templates for a user."""
    db = get_db()
    docs = db.collection("users").document(user_id).collection("templates").stream()

    templates = []
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        templates.append(data)
    return templates


async def delete_user_template(user_id: str, template_id: str) -> bool:
    """Delete a user template."""
    db = get_db()
    doc_ref = db.collection("users").document(user_id).collection("templates").document(template_id)
    doc = doc_ref.get()
    if doc.exists:
        doc_ref.delete()
        return True
    return False

# --- Admin functions ---

async def get_all_users_summary() -> List[Dict[str, Any]]:
    """Get summary of all users for admin view."""
    db = get_db()
    users = []

    # Get all user documents
    user_docs = db.collection("users").stream()

    for user_doc in user_docs:
        user_id = user_doc.id

        # Count evolutions
        evolutions = list(db.collection("users").document(user_id).collection("evolutions").limit(100).stream())
        # Count checkpoints
        checkpoints = list(db.collection("users").document(user_id).collection("checkpoints").limit(100).stream())

        # Combined count for admin display
        total_evolution_count = len(evolutions) + len(checkpoints)

        # Count templates
        templates = list(db.collection("users").document(user_id).collection("templates").limit(100).stream())
        template_count = len(templates)

        # Get settings for last activity
        settings = db.collection("users").document(user_id).collection("settings").document("main").get()
        last_activity = None
        if settings.exists:
            last_activity = settings.to_dict().get("updated_at")

        users.append({
            "user_id": user_id,
            "evolution_count": total_evolution_count,
            "template_count": template_count,
            "last_activity": last_activity,
            "has_api_key": settings.exists and settings.to_dict().get("api_key_set", False)
        })

    return users
