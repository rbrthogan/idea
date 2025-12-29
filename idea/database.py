# Database module for Firestore operations
# Provides CRUD operations for user-scoped data with encryption for sensitive fields

import os
import json
import base64
from datetime import datetime
from typing import Optional, Dict, List, Any
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from google.cloud import firestore

# Firestore client (lazy initialization)
_db = None


def get_db() -> firestore.Client:
    """Get or create Firestore client."""
    global _db
    if _db is None:
        _db = firestore.Client()
    return _db


# --- Encryption utilities for API keys ---

def _get_encryption_key() -> bytes:
    """
    Derive encryption key from environment secret.
    In production, use Secret Manager for the master key.
    """
    master_key = os.getenv("ENCRYPTION_KEY", "default-dev-key-change-in-production")
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
        except Exception as e:
            print(f"Failed to decrypt API key for user {user_id}: {e}")
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

# Fields that may contain nested arrays and need JSON serialization
# Firestore doesn't support arrays within arrays, so we serialize these to JSON strings
EVOLUTION_NESTED_FIELDS = [
    'history',           # [[ideas], [ideas], ...]
    'population',        # [ideas]
    'diversity_history', # [[diversity scores], ...]
    'contexts',          # may contain nested structures
    'specific_prompts',  # may contain nested structures
    'breeding_prompts',  # may contain nested structures
    'token_counts',      # dict with nested data
    'config',            # dict with config values
]


async def save_evolution(user_id: str, evolution_id: str, data: Dict[str, Any]) -> None:
    """Save an evolution for a user.

    Note: Firestore doesn't support nested arrays, so we serialize fields like
    'history' (which is [[ideas], [ideas]]) to JSON strings.
    """
    db = get_db()

    # Make a copy to avoid modifying the original
    save_data = dict(data)
    save_data["updated_at"] = datetime.utcnow().isoformat()
    save_data["user_id"] = user_id

    # Serialize nested array fields to JSON strings
    for field in EVOLUTION_NESTED_FIELDS:
        if field in save_data and save_data[field] is not None:
            save_data[field] = json.dumps(save_data[field], default=str)

    db.collection("users").document(user_id).collection("evolutions").document(evolution_id).set(save_data)


async def get_evolution(user_id: str, evolution_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific evolution for a user."""
    db = get_db()
    doc = db.collection("users").document(user_id).collection("evolutions").document(evolution_id).get()

    if doc.exists:
        data = doc.to_dict()
        # Deserialize JSON string fields back to nested arrays
        for field in EVOLUTION_NESTED_FIELDS:
            if field in data and isinstance(data[field], str):
                try:
                    data[field] = json.loads(data[field])
                except json.JSONDecodeError:
                    pass  # Keep as string if not valid JSON
        return data
    return None


async def list_evolutions(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """List all evolutions for a user."""
    db = get_db()
    docs = (db.collection("users").document(user_id).collection("evolutions")
            .order_by("updated_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream())

    evolutions = []
    for doc in docs:
        data = doc.to_dict()
        data["id"] = doc.id
        # Deserialize JSON string fields back to native types
        for field in EVOLUTION_NESTED_FIELDS:
            if field in data and isinstance(data[field], str):
                try:
                    data[field] = json.loads(data[field])
                except json.JSONDecodeError:
                    pass  # Keep as string if not valid JSON
        evolutions.append(data)
    return evolutions


async def delete_evolution(user_id: str, evolution_id: str) -> bool:
    """Delete an evolution."""
    db = get_db()
    doc_ref = db.collection("users").document(user_id).collection("evolutions").document(evolution_id)
    doc = doc_ref.get()
    if doc.exists:
        doc_ref.delete()
        return True
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


# --- System Templates (read-only, shared across users) ---

SYSTEM_TEMPLATE_IDS = ["airesearch", "game_design", "drabble"]


async def get_system_templates() -> List[Dict[str, Any]]:
    """Get all system templates (read from YAML files)."""
    from idea.prompts.loader import list_available_templates

    all_templates = list_available_templates()
    system_templates = []

    for template_id in SYSTEM_TEMPLATE_IDS:
        if template_id in all_templates:
            template_info = all_templates[template_id]
            if "error" not in template_info:
                system_templates.append({
                    "id": template_id,
                    "name": template_info.get("name", template_id),
                    "description": template_info.get("description", ""),
                    "is_system": True
                })

    return system_templates


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
